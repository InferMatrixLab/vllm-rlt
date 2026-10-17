"""One CUDA owner process per P/D worker. Control messages never carry tensors."""

import hashlib
import json
import os
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field, replace

import torch

from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, OuroConfig, OuroForCausalLM
from vllm_rlt.profiling import ProfileConfig
from vllm_rlt.request import Stage

from .protocol import PROFILE_COMMANDS, CommandKind, EventKind, ProfileAction
from .transport import NixlConnector, kv_segments, partition_segments


@dataclass
class Work:
    tid: str
    request: object
    slot: int
    peer: str
    trace_id: str | None = None
    target_tables: tuple = ()
    target_slot: int = 0
    cached_tokens: int = 0
    target_cached_tokens: int = 0
    chunks: deque = field(default_factory=deque)
    received: set = field(default_factory=set)
    next_sequence: int = 0
    expected: int | None = None
    committed: bool = False
    cancelled: bool = False
    active: bool = False
    compute_done: bool = False
    compute_released: bool = False
    last_event: object = None
    acknowledged: bool = False
    started: float = field(default_factory=time.monotonic)


class PDWorker:
    def __init__(self, role, device, name, channel, model_source, options, pd_config):
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.manual_seed(options["seed"])
        self.role, self.channel, self.config = role, channel, pd_config
        dtype = getattr(torch, options["dtype"])
        self.model = (
            OuroForCausalLM(model_source).to(device=f"cuda:{device}", dtype=dtype)
            if isinstance(model_source, OuroConfig)
            else AutoModelForCausalLM.from_pretrained(
                model_source,
                revision=options["revision"],
                device=f"cuda:{device}",
                dtype=dtype,
            )
        )
        self.engine = LLMEngine(self.model, **options["engine"])
        self.cache = self.engine.cache_manager
        self.active_limit = self.engine.scheduler.config.max_num_seqs
        self.limit = self.active_limit + (
            pd_config.max_receiving_requests
            if role == "decode"
            else pd_config.max_draining_requests
        )
        self.hidden = torch.empty(
            (self.limit, self.model.config.hidden_size), device=f"cuda:{device}", dtype=dtype
        )
        self.free_slots = list(reversed(range(self.limit)))
        self.connector = NixlConnector(name, self.cache, self.hidden, pd_config)
        self.work = {}
        self.running = True
        self.prefill_tokens = 0
        self.transfer_cursor = 0
        digest = hashlib.sha256()
        for parameter in self.model.state_dict().values():
            digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        self.fingerprint = hashlib.sha256(
            json.dumps(
                dict(
                    model=asdict(self.model.config),
                    weights=digest.hexdigest(),
                    dtype=options["dtype"],
                    layout=self.cache.layout,
                    attention=self.cache.attention_info,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.send(
            EventKind.READY,
            info=self.connector.info,
            fingerprint=self.fingerprint,
            model=asdict(self.model.config),
            num_blocks=self.cache.num_blocks,
            block_size=self.cache.block_size,
            depths=self.cache.storage_depths,
            max_seqs=self.limit,
            active_limit=self.active_limit,
            watermark_blocks=self.cache.watermark_blocks,
        )

    def send(self, kind: EventKind, **fields):
        self.channel.send(dict(kind=kind.value, **fields))

    def remove(self, w):
        self.engine.model_runner.release(w.tid)
        if w.tid in self.engine.scheduler.requests:
            self.engine.scheduler.abort(w.tid)
        if self.cache.has_allocation(w.tid):
            self.cache.free(w.tid)
            self.cache.unpin_transfer(w.tid, w.tid)
        self.free_slots.append(w.slot)
        del self.work[w.tid]
        self.send(EventKind.RELEASED, tid=w.tid, role=self.role)

    def accept(self, command):
        tid = command["tid"]
        if tid in self.work:
            raise RuntimeError("duplicate transfer generation")
        if not self.free_slots:
            self.send(EventKind.REJECTED, tid=tid, role=self.role)
            return
        params = command["params"]
        # P owns only prompt KV and never samples. D owns all sampling/RNG.
        actual_params = replace(params, max_tokens=1) if self.role == "prefill" else params
        self.engine.add_request(
            tid, command["tokens"], actual_params, trace_id=command.get("trace_id")
        )
        request = self.engine.scheduler.requests[tid]
        self.engine.scheduler.queues[Stage.WAITING].remove(tid)
        capacity = len(command["tokens"]) + actual_params.max_tokens - 1
        prefix = self.cache.lookup_prefix(command["tokens"])
        hit = len(prefix) * self.cache.block_size
        initial = len(command["tokens"]) if self.cache.incremental_allocation else capacity
        if not self.cache.allocate(tid, capacity, initial_tokens=initial, prefix=prefix):
            self.engine.scheduler.abort(tid)
            self.send(EventKind.REJECTED, tid=tid, role=self.role)
            return
        self.cache.pin_transfer(tid, tid)
        w = Work(
            tid, request, self.free_slots.pop(), command["peer"], trace_id=command.get("trace_id")
        )
        w.cached_tokens = hit
        self.work[tid] = w
        if self.role == "decode":
            request.stage = Stage.RECEIVING
            self.send(
                EventKind.RESERVED,
                tid=tid,
                tables=self.cache.plane_block_tables(tid),
                slot=w.slot,
                cached_tokens=hit,
            )
        else:
            w.target_tables, w.target_slot = command["tables"], command["slot"]
            w.target_cached_tokens = command.get("cached_tokens", 0)
            request.num_prefilled_tokens = hit
            if hit > w.target_cached_tokens:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.cache.device))
                self.queue_chunk(w, w.target_cached_tokens, hit, event, False)
            self.engine.scheduler.enqueue(request, Stage.PREFILL)
            self.send(EventKind.PREFILL_STARTED, tid=tid)

    def command(self, command):
        kind = command["kind"]
        action = PROFILE_COMMANDS.get(kind)
        if action is not None:
            try:
                if action is ProfileAction.START:
                    result = self.engine.start_profile(
                        ProfileConfig(**command["config"]),
                        scheduled=command["scheduled"],
                        session_id=command["session_id"],
                        capture_time=command["capture_time"],
                        rank=command["rank"],
                        role=self.role,
                        managed_manifest=False,
                    )
                else:
                    current = self.engine.profile_status().get("session_id")
                    if current is not None and current != command["session_id"]:
                        raise ValueError("profile session ID mismatch")
                    if action is ProfileAction.STOP:
                        result = self.engine.stop_profile()
                    else:
                        result = self.engine.profile_status()
                self.send(
                    EventKind.PROFILE_REPLY,
                    control_id=command["control_id"],
                    session_id=command["session_id"],
                    result=result,
                )
            except Exception as exc:
                self.send(
                    EventKind.PROFILE_REPLY,
                    control_id=command["control_id"],
                    session_id=command["session_id"],
                    error=str(exc),
                )
            return
        if kind == CommandKind.CONNECT:
            for peer in command["peers"]:
                self.connector.connect(peer)
            self.send(EventKind.CONNECTED)
        elif kind in (CommandKind.RESERVE, CommandKind.PREFILL):
            self.accept(command)
        elif kind == CommandKind.CANCEL:
            w = self.work.get(command["tid"])
            if w is None:
                self.send(EventKind.RELEASED, tid=command["tid"], role=self.role)
                return
            w.cancelled = True
            if self.role == "prefill":
                w.chunks.clear()
                queue = self.engine.scheduler.queues[Stage.PREFILL]
                if w.tid in queue:
                    queue.remove(w.tid)
            elif w.active:
                self.remove(w)  # All imports ended before decode activation.
        elif kind == CommandKind.CANCEL_SAFE:
            w = self.work.get(command["tid"])
            if w is not None:
                if not w.cancelled:
                    raise RuntimeError("unexpected cancellation acknowledgement")
                self.remove(w)
        elif kind == CommandKind.COMMIT:
            w = self.work.get(command["tid"])
            if w is not None and not w.cancelled:
                if w.expected is not None and w.expected != command["count"]:
                    raise RuntimeError("conflicting transfer commit")
                if type(command["count"]) is not int or command["count"] <= 0:
                    raise RuntimeError("invalid transfer commit count")
                w.expected = command["count"]
        elif kind == CommandKind.ACK:
            w = self.work.get(command["tid"])
            if w is not None:
                w.acknowledged = True
        elif kind == CommandKind.STOP:
            if self.work or self.connector.pending:
                raise RuntimeError("stop requested before PD requests drained")
            self.running = False
        else:
            raise ValueError(f"unknown PD command {kind}")

    def queue_chunk(self, w, start, end, event, final):
        peer = self.connector.peers[w.peer]
        tables = self.cache.plane_block_tables(w.tid)
        start = max(start, w.target_cached_tokens)
        source = list(kv_segments(self.connector.info, tables, start, end)) if start < end else []
        target = list(kv_segments(peer, w.target_tables, start, end)) if start < end else []
        if final:
            size = self.connector.info["hidden_bytes"]
            source.append(
                (
                    self.connector.info["hidden_ptr"] + w.slot * size,
                    size,
                    self.connector.info["device"],
                )
            )
            target.append((peer["hidden_ptr"] + w.target_slot * size, size, peer["device"]))
        for local, remote, size in partition_segments(
            source, target, self.config.transfer_chunk_bytes, self.config.max_transfer_descriptors
        ):
            w.chunks.append((event, w.next_sequence, local, remote, size))
            w.next_sequence += 1

    def prefill_step(self):
        # GPU submission can block the owner thread on CUDA launch backpressure.
        # Let the event loop advance ready handoffs before submitting more compute;
        # it must keep polling transfers and control messages if credits are full.
        if any(w.chunks and w.chunks[0][0].query() for w in self.work.values()):
            return
        queue = self.engine.scheduler.queues[Stage.PREFILL]
        # Limit unsubmitted data as well as posted network work. Each request
        # can have at most one unfinished chunk; ready requests still progress.
        eligible = []
        for tid in list(queue):
            w = self.work[tid]
            if not w.chunks and (w.last_event is None or w.last_event.query()):
                eligible.append(tid)
        if not eligible or self.connector.inflight_bytes >= self.config.max_inflight_bytes:
            return
        original = self.engine.scheduler.queues[Stage.PREFILL]
        selected = set(eligible)
        self.engine.scheduler.queues[Stage.PREFILL] = deque(eligible)
        batch = self.engine.scheduler._take(Stage.PREFILL)
        unused = self.engine.scheduler.queues[Stage.PREFILL]
        self.engine.scheduler.queues[Stage.PREFILL] = deque(
            tid for tid in original if tid not in selected
        )
        self.engine.scheduler.queues[Stage.PREFILL].extend(unused)
        ticket = self.engine.model_runner.submit(batch)
        stream = self.engine.model_runner.core_stream or torch.cuda.current_stream(
            self.cache.device
        )
        with torch.cuda.stream(stream):
            for item in batch.items:
                w = self.work[item.request.request_id]
                request = w.request
                request.num_prefilled_tokens += item.token_count
                final = request.num_prefilled_tokens == len(request.prompt_token_ids)
                if final:
                    self.hidden[w.slot].copy_(request.hidden_state)
                event = torch.cuda.Event()
                event.record(stream)
                w.last_event = event
                self.queue_chunk(w, item.token_start, request.num_prefilled_tokens, event, final)
                self.prefill_tokens += item.token_count
                self.cache.publish_prefix(
                    w.tid, request.prompt_token_ids, request.num_prefilled_tokens, event
                )
                if final:
                    w.compute_done = True
                else:
                    self.engine.scheduler.enqueue(request, Stage.PREFILL)
        # submit keeps its events; the prefill ticket has no readback slot.
        del ticket
        return True

    def progress(self):
        self.cache.poll_prefixes()
        _, notifications = self.connector.poll()
        for peer, tid, sequence in notifications:
            w = self.work.get(tid)
            if w is None or w.cancelled:
                continue  # Old generations cannot activate a reused request ID.
            if self.role != "decode" or peer != w.peer or type(sequence) is not int or sequence < 0:
                raise RuntimeError("invalid transfer completion")
            if w.active and sequence not in w.received:
                raise RuntimeError("late KV write after decode activation")
            w.received.add(sequence)
        work = list(self.work.values())
        if work:
            start = self.transfer_cursor % len(work)
            work = work[start:] + work[:start]
            self.transfer_cursor += 1
        if self.role == "decode" and self.engine.scheduler.config.policy == "priority":
            work.sort(key=lambda w: (w.request.sampling_params.priority, w.started))
        for w in work:
            if self.role == "prefill":
                if w.cancelled:
                    if not self.connector.busy(w.tid):
                        self.remove(w)
                    continue
                if w.compute_done and not w.compute_released and w.last_event.query():
                    self.engine.model_runner.release(w.tid)
                    w.request.hidden_state = None
                    w.compute_released = True
                    self.send(EventKind.PREFILL_COMPLETE, tid=w.tid)
                # Round robin: at most one bounded transfer per request per tick.
                if w.chunks:
                    event, seq, local, remote, size = w.chunks[0]
                    if event.query() and self.connector.submit(
                        w.tid, seq, w.peer, local, remote, size
                    ):
                        w.chunks.popleft()
                if w.compute_done and not w.chunks and not w.committed:
                    w.committed = True
                    self.send(EventKind.COMMIT, tid=w.tid, count=w.next_sequence)
                if w.acknowledged and not self.connector.busy(w.tid):
                    self.remove(w)
            elif not w.cancelled and not w.active and w.expected is not None:
                if any(seq >= w.expected for seq in w.received):
                    raise RuntimeError("unexpected transfer sequence")
                if (
                    len(w.received) == w.expected
                    and sum(
                        x.active and x.request.stage != Stage.WAITING for x in self.work.values()
                    )
                    < self.active_limit
                ):
                    self.cache.mark_imported_prefix(
                        w.tid, len(w.request.prompt_token_ids), start=w.cached_tokens
                    )
                    self.cache.publish_prefix(
                        w.tid, w.request.prompt_token_ids, len(w.request.prompt_token_ids)
                    )
                    self.cache.unpin_transfer(w.tid, w.tid)
                    w.request.num_prefilled_tokens = len(w.request.prompt_token_ids)
                    w.request.loops_done = self.model.config.total_ut_steps
                    w.request.hidden_state = self.hidden[w.slot]
                    self.engine.scheduler.enqueue(w.request, Stage.CODA)
                    w.active = True
                    self.send(EventKind.ACTIVATED, tid=w.tid, seconds=time.monotonic() - w.started)

    def run(self):
        while self.running:
            for _ in range(self.config.max_control_messages):
                if not self.channel.poll():
                    break
                self.command(self.channel.recv())
            self.progress()
            if self.role == "prefill":
                submitted = self.prefill_step()
                if submitted:
                    self.engine.profiling.step()
            elif any(w.active for w in self.work.values()):
                for output in self.engine.step():
                    self.send(EventKind.OUTPUT, tid=output.request_id, output=output)
                    if output.finished:
                        w = self.work.pop(output.request_id)
                        self.free_slots.append(w.slot)
                        self.send(EventKind.RELEASED, tid=w.tid, role=self.role)
            if not self.work and not self.channel.poll():
                time.sleep(0.001)
        self.engine.model_runner.synchronize()
        self.engine.close()
        self.connector.close()
        self.send(
            EventKind.STOPPED,
            prefill_tokens=self.prefill_tokens,
            bytes_sent=self.connector.bytes_sent,
            transfers=self.connector.transfers,
            transfer_seconds=self.connector.transfer_seconds,
            used_blocks=self.cache.num_used_blocks,
        )


def worker_main(role, device, name, channel, model_source, options, pd_config):
    # Keep owner reachable on fatal errors. The coordinator terminates senders
    # before receivers; no failed receiver reuses a potentially writable page.
    owner = None
    try:
        owner = PDWorker(role, device, name, channel, model_source, options, pd_config)
        owner.run()
    except BaseException:
        try:
            channel.send(
                dict(kind=EventKind.FATAL.value, error=traceback.format_exc(), pid=os.getpid())
            )
            while True:
                time.sleep(1)
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        channel.close()
