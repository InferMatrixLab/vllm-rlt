"""PD transfer adapter. Request computation remains in StageEngine."""

import hashlib
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace

import torch

from vllm_rlt.core.execution import WorkKey
from vllm_rlt.request import Request, RequestStatus

from .transport import NixlConnector, kv_segments, partition_segments


@dataclass
class RequestTransfer:
    tid: str
    request: Request
    slot: int
    peer: str
    outgoing: bool = False
    trace_id: str | None = None
    target_tables: tuple = ()
    target_slot: int = 0
    cached_tokens: int = 0
    target_cached_tokens: int = 0
    chunks: deque = field(default_factory=deque)
    transfer_boundary_count: int = 0
    transfer_boundary_first: tuple[int, int] | None = None
    transfer_boundary_last: tuple[int, int] | None = None
    transfer_overlaps: int = 0
    received: set = field(default_factory=set)
    next_sequence: int = 0
    expected: int | None = None
    committed: bool = False
    cancelled: bool = False
    ready: bool = False
    active: bool = False
    cancel_confirmed: bool = False
    compute_done: bool = False
    last_event: object = None
    acknowledged: bool = False
    started: float = field(default_factory=time.monotonic)


class PDStageConnector:
    def __init__(self, engine, hidden, transport, config, *, notify):
        self.engine = engine
        self.cache = engine.cache_manager
        self.hidden = hidden
        self.transport = transport
        self.config = config
        self.notify = notify
        self.transfers = {}
        self.free_slots = list(reversed(range(len(hidden))))
        self.transfer_cursor = 0
        self.prefill_tokens = 0

    @classmethod
    def create(cls, engine, *, config, notify):
        """Allocate transport-owned staging in the stage's owner process."""
        stage = engine.stage_config
        limit = engine.scheduler.config.max_num_seqs
        if stage.produces_tokens:
            limit += config.max_receiving_requests
        if stage.accepts_prompt:
            limit += config.max_draining_requests
        parameter = next(engine.model.parameters())
        hidden = torch.empty(
            (limit, engine.model.config.hidden_size), device=parameter.device, dtype=parameter.dtype
        )
        transport = NixlConnector(stage.stage_id, engine.cache_manager, hidden, config)
        return cls(engine, hidden, transport, config, notify=notify)

    def startup_info(self):
        """Advertise registered memory and verify peers use compatible model weights."""
        digest = hashlib.sha256()
        for parameter in self.engine.model.state_dict().values():
            digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        parameter = next(self.engine.model.parameters())
        fingerprint = hashlib.sha256(
            json.dumps(
                dict(
                    model=asdict(self.engine.model.config),
                    weights=digest.hexdigest(),
                    dtype=str(parameter.dtype).removeprefix("torch."),
                    layout=self.cache.layout,
                    attention=self.cache.attention_info,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return dict(
            info=self.transport.info, fingerprint=fingerprint, transfer_slots=len(self.hidden)
        )

    @property
    def has_pending(self):
        return bool(self.transfers or self.transport.pending)

    def command(self, command):
        """Handle transfer commands; leave generic stage commands to the worker."""
        kind = command["kind"]
        if kind == "connect":
            for peer in command["peers"]:
                self.transport.connect(peer)
            self.notify("connected")
        elif kind in ("reserve", "prefill"):
            self.reserve(
                command["tid"],
                command["tokens"],
                command["params"],
                receiving=kind == "reserve",
                peer=command["peer"],
                trace_id=command.get("trace_id"),
                target_tables=command.get("tables", ()),
                target_slot=command.get("slot", 0),
                target_cached_tokens=command.get("cached_tokens", 0),
            )
        elif kind == "cancel":
            self.cancel(command["tid"])
        elif kind == "cancel_safe":
            self.cancel_safe(command["tid"])
        elif kind == "commit":
            self.commit(command["tid"], command["count"])
        elif kind == "ack":
            self.acknowledge(command["tid"])
        elif kind == "abort" and command["request_id"] in self.transfers:
            raise ValueError("transferred requests must use the transfer cancellation protocol")
        else:
            return False
        return True

    def close(self):
        if self.has_pending:
            raise RuntimeError("cannot close a connector with pending transfers")
        self.transport.close()
        return dict(
            prefill_tokens=self.prefill_tokens,
            bytes_sent=self.transport.bytes_sent,
            transfers=self.transport.transfers,
            transfer_seconds=self.transport.transfer_seconds,
        )

    def _remove(self, transfer):
        if transfer.tid not in self.engine.scheduler.requests or self.engine.release_transfer(
            transfer.request
        ):
            self._forget(transfer)

    def _forget(self, transfer):
        self.free_slots.append(transfer.slot)
        del self.transfers[transfer.tid]
        self.notify("released", tid=transfer.tid, role=self.engine.stage_config.role)

    def output_finished(self, request_id):
        """The engine has retired a completed decode request; release its staging slot."""
        transfer = self.transfers.get(request_id)
        if transfer is not None:
            if not transfer.active:
                transfer.active = True
                self.notify(
                    "activated", tid=transfer.tid, seconds=time.monotonic() - transfer.started
                )
            if request_id not in self.engine.scheduler.requests:
                self._forget(transfer)

    def exports(self, request):
        transfer = self.transfers.get(request.request_id)
        return transfer is not None and transfer.request is request and transfer.outgoing

    def can_schedule(self, request):
        if not self.exports(request):
            return True
        transfer = self.transfers.get(request.request_id)
        return bool(
            transfer is not None
            and transfer.request is request
            and not transfer.cancelled
            and not transfer.chunks
            and (transfer.last_event is None or transfer.last_event.query())
            and self.transport.inflight_bytes < self.config.max_inflight_bytes
        )

    def prefill_submitted(self, request, *, work, start, progress, event, stream):
        transfer = self.transfers[request.request_id]
        if transfer.request is not request:
            raise RuntimeError("stale transfer generation")
        if work == WorkKey.PROMPT_PRELUDE or (
            work == WorkKey.PROMPT_RECURRENT
            and self.engine.scheduler.config.wavefront_chunked_prefill
        ):
            transfer.transfer_overlaps += self.transport.busy(transfer.tid)
        if not progress.chunk_complete:
            transfer.last_event = event
            return
        stream = stream or torch.cuda.current_stream(self.cache.device)
        with torch.cuda.stream(stream):
            if event is not None:
                stream.wait_event(event)
            if progress.prompt_complete:
                self.hidden[transfer.slot].copy_(request.hidden_state)
            ready = torch.cuda.Event()
            ready.record(stream)
        transfer.last_event = ready
        self.prefill_tokens += request.num_prefilled_tokens - start
        self.queue_chunk(
            transfer, start, request.num_prefilled_tokens, ready, progress.prompt_complete
        )
        transfer.compute_done = progress.prompt_complete

    def queue_chunk(self, transfer, start, end, event, final):
        peer = self.transport.peers[transfer.peer]
        tables = self.cache.plane_block_tables(transfer.tid)
        start = max(start, transfer.target_cached_tokens)
        if start < end:
            boundary = (start, end)
            if transfer.transfer_boundary_first is None:
                transfer.transfer_boundary_first = boundary
            transfer.transfer_boundary_last = boundary
            transfer.transfer_boundary_count += 1
        source = list(kv_segments(self.transport.info, tables, start, end)) if start < end else []
        target = list(kv_segments(peer, transfer.target_tables, start, end)) if start < end else []
        if final:
            size = self.transport.info["hidden_bytes"]
            source.append(
                (
                    self.transport.info["hidden_ptr"] + transfer.slot * size,
                    size,
                    self.transport.info["device"],
                )
            )
            target.append((peer["hidden_ptr"] + transfer.target_slot * size, size, peer["device"]))
        for local, remote, size in partition_segments(
            source, target, self.config.transfer_chunk_bytes, self.config.max_transfer_descriptors
        ):
            transfer.chunks.append((event, transfer.next_sequence, local, remote, size))
            transfer.next_sequence += 1

    def reserve(
        self,
        tid,
        tokens,
        params,
        *,
        receiving,
        peer,
        trace_id=None,
        target_tables=(),
        target_slot=0,
        target_cached_tokens=0,
    ):
        """Reserve one local Request and a connector-owned staging slot per generation."""
        if tid in self.transfers:
            raise RuntimeError("duplicate transfer generation")
        if not self.free_slots:
            self.notify("rejected", tid=tid, role=self.engine.stage_config.role)
            return
        # P owns only prompt KV and never samples. D owns all sampling/RNG.
        actual_params = replace(params, max_tokens=1) if not receiving else params
        request = self.engine.reserve_transfer(
            tid,
            tokens,
            actual_params,
            receiving=receiving,
            trace_id=trace_id,
        )
        if request is None:
            self.notify("rejected", tid=tid, role=self.engine.stage_config.role)
            return
        hit = request.num_prefilled_tokens
        transfer = RequestTransfer(
            tid,
            request,
            self.free_slots.pop(),
            peer,
            outgoing=not receiving,
            trace_id=trace_id,
        )
        transfer.cached_tokens = hit
        self.transfers[tid] = transfer
        if receiving:
            self.notify(
                "reserved",
                tid=tid,
                tables=self.cache.plane_block_tables(tid),
                slot=transfer.slot,
                cached_tokens=hit,
            )
        else:
            transfer.target_tables, transfer.target_slot = target_tables, target_slot
            transfer.target_cached_tokens = target_cached_tokens
            if hit > transfer.target_cached_tokens:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.cache.device))
                self.queue_chunk(transfer, transfer.target_cached_tokens, hit, event, False)
            self.notify("prefill_started", tid=tid)

    def cancel(self, transfer_id):
        transfer = self.transfers.get(transfer_id)
        if transfer is None:
            self.notify("released", tid=transfer_id, role=self.engine.stage_config.role)
            return
        transfer.cancelled = True
        if transfer.outgoing:
            transfer.chunks.clear()
            self.engine.suspend_transfer(transfer.request)
        elif transfer.ready:
            self._remove(transfer)  # Complete receipt proves all remote writes have ended.

    def cancel_safe(self, transfer_id):
        """Coordinator confirms remote writers stopped; resources may now be retired."""
        transfer = self.transfers.get(transfer_id)
        if transfer is not None:
            if not transfer.cancelled:
                raise RuntimeError("unexpected cancellation acknowledgement")
            if transfer.outgoing:
                raise RuntimeError("source cancellation must drain local and remote accesses")
            transfer.cancel_confirmed = True
            self._remove(transfer)

    def commit(self, transfer_id, count):
        transfer = self.transfers.get(transfer_id)
        if transfer is None or transfer.cancelled:
            return
        if transfer.outgoing:
            raise RuntimeError("commit requires an incoming transfer")
        if transfer.expected is not None and transfer.expected != count:
            raise RuntimeError("conflicting transfer commit")
        if type(count) is not int or count <= 0:
            raise RuntimeError("invalid transfer commit count")
        transfer.expected = count

    def acknowledge(self, transfer_id):
        transfer = self.transfers.get(transfer_id)
        if transfer is not None:
            if not transfer.outgoing:
                raise RuntimeError("acknowledgement requires an outgoing transfer")
            transfer.acknowledged = True

    def progress(self):
        self.cache.poll_prefixes()
        _, notifications = self.transport.poll()
        for peer, tid, sequence in notifications:
            transfer = self.transfers.get(tid)
            if transfer is None or transfer.cancelled:
                continue  # Old generations cannot activate a reused request ID.
            if (
                transfer.outgoing
                or peer != transfer.peer
                or type(sequence) is not int
                or sequence < 0
            ):
                raise RuntimeError("invalid transfer completion")
            if transfer.ready and sequence not in transfer.received:
                raise RuntimeError("late KV write after decode activation")
            transfer.received.add(sequence)
        transfers = list(self.transfers.values())
        if transfers:
            start = self.transfer_cursor % len(transfers)
            transfers = transfers[start:] + transfers[:start]
            self.transfer_cursor += 1
        for transfer in transfers:
            if transfer.outgoing:
                if transfer.cancelled:
                    if not self.transport.busy(transfer.tid) and (
                        transfer.last_event is None or transfer.last_event.query()
                    ):
                        # The hidden staging copy may outlive the model event.
                        # Do not reuse its slot until both local and remote users finish.
                        self._remove(transfer)
                    continue
                if (
                    transfer.compute_done
                    and self.engine.scheduler.has_admission_slot(transfer.request)
                    and self.engine.release_transfer_compute(
                        transfer.request, ready_event=transfer.last_event
                    )
                ):
                    self.notify(
                        "prefill_complete",
                        tid=transfer.tid,
                        boundaries=(
                            transfer.transfer_boundary_first,
                            transfer.transfer_boundary_last,
                            transfer.transfer_boundary_count,
                        ),
                        overlaps=transfer.transfer_overlaps,
                    )
                # Round robin: at most one bounded transfer per request per tick.
                if transfer.chunks:
                    event, seq, local, remote, size = transfer.chunks[0]
                    if event.query() and self.transport.submit(
                        transfer.tid, seq, transfer.peer, local, remote, size
                    ):
                        transfer.chunks.popleft()
                if transfer.compute_done and not transfer.chunks and not transfer.committed:
                    transfer.committed = True
                    self.notify("commit", tid=transfer.tid, count=transfer.next_sequence)
                if (
                    transfer.acknowledged
                    and transfer.committed
                    and not self.engine.scheduler.has_admission_slot(transfer.request)
                    and not self.transport.busy(transfer.tid)
                ):
                    self._remove(transfer)
            elif transfer.cancelled:
                if transfer.ready or transfer.cancel_confirmed:
                    self._remove(transfer)
            elif transfer.ready:
                if transfer.request.request_id not in self.engine.scheduler.requests:
                    self._forget(transfer)
                elif not transfer.active and transfer.request.status != RequestStatus.RECEIVING:
                    transfer.active = True
                    self.notify(
                        "activated", tid=transfer.tid, seconds=time.monotonic() - transfer.started
                    )
            elif transfer.expected is not None:
                if any(seq >= transfer.expected for seq in transfer.received):
                    raise RuntimeError("unexpected transfer sequence")
                if len(transfer.received) == transfer.expected:
                    if not self.engine.accept_import(
                        transfer.request,
                        self.hidden[transfer.slot],
                        cached_tokens=transfer.cached_tokens,
                    ):
                        raise RuntimeError("stale import activation")
                    transfer.ready = True
                    self.notify("received", tid=transfer.tid)
