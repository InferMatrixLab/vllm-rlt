"""CPU coordinator for independent prefill/decode GPU pools.

D reservations precede P computation. Control channels carry only metadata and
outputs; NIXL moves KV directly between registered worker allocations.
"""

import multiprocessing as mp
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.models import OuroConfig, resolve_model_config
from vllm_rlt.profiling import ProfileConfig
from vllm_rlt.profiling_artifacts import _write_json_atomic, profile_timestamp
from vllm_rlt.request import FinishReason, Request, RequestOutput, Stage
from vllm_rlt.sampling_params import SamplingParams

from .config import PDConfig
from .worker import worker_main


@dataclass
class Peer:
    name: str
    role: str
    process: object
    channel: object
    info: dict = field(default_factory=dict)
    blocks: int = 0
    slots: int = 0
    compute_slots: int = 0
    ready: bool = False
    connected: bool = False
    stopped: bool = False


@dataclass
class Pending:
    request: Request
    tid: str
    trace_id: str | None
    created: float = field(default_factory=time.monotonic)
    p: str | None = None
    d: str | None = None
    phase: str = "waiting"
    p_blocks: int = 0
    d_blocks: int = 0
    cancelled: bool = False
    p_released: bool = False
    p_compute_released: bool = False
    d_released: bool = False
    finished: bool = False
    timings: dict = field(default_factory=dict)


class PDEngine:
    def __init__(
        self,
        model,
        *,
        pd_config=None,
        prefill_cache_config=None,
        decode_cache_config=None,
        prefill_scheduler_config=None,
        decode_scheduler_config=None,
        execution_config=None,
        exit_config=None,
        attention_backend="flash_attn",
        dtype="bfloat16",
        revision=None,
        seed=0,
        speculative_config=None,
    ):
        if speculative_config is not None:
            raise ValueError("speculative decoding is not yet supported by PD")
        if isinstance(model, (str, Path)):
            model, revision, _ = resolve_model_config(model, revision=revision)
        self.config = pd_config or PDConfig()
        self.exit_config = exit_config or ExitConfig("ouro_delayed")
        execution = execution_config or ExecutionConfig(async_scheduling=True)
        p_scheduler = prefill_scheduler_config or SchedulerConfig(
            max_num_seqs=8, max_num_batched_tokens=2048, prefill_chunk_size=2048
        )
        d_scheduler = decode_scheduler_config or SchedulerConfig(
            max_num_seqs=128, max_num_batched_tokens=128
        )
        p_cache = prefill_cache_config or CacheConfig()
        d_cache = decode_cache_config or CacheConfig()
        if (p_cache.layout, p_cache.block_size) != (d_cache.layout, d_cache.block_size):
            raise ValueError("P and D must use the same KV layout and block size")
        if attention_backend not in ("triton", "flash_attn", "flash_attn_4"):
            raise ValueError("PD supports CUDA Triton or paged FA4 attention")
        self.scheduling_policy = p_scheduler.policy
        self.peers = {}
        self.requests = {}
        self.transfers = {}
        self.scheduler = SimpleNamespace(requests=self.requests)
        self.cache_manager = SimpleNamespace(num_used_blocks=0)
        self.metrics = deque(maxlen=1024)
        self.worker_metrics = {}
        self.outputs = deque()
        self.closed = False
        self.failure = None
        self._profile_session = None
        self._profile_replies = {}
        self._profile_statuses = {}
        self._profile_recording = False
        self._profile_errors = []
        context = mp.get_context("spawn")
        epoch = uuid.uuid4().hex
        try:
            for role, devices, cache, scheduler in (
                ("prefill", self.config.prefill_devices, p_cache, p_scheduler),
                ("decode", self.config.decode_devices, d_cache, d_scheduler),
            ):
                for device in devices:
                    name = f"{epoch}-{role}-{device}"
                    parent, child = context.Pipe()
                    options = dict(
                        dtype=dtype,
                        revision=revision,
                        seed=seed,
                        engine=dict(
                            cache_config=cache,
                            scheduler_config=scheduler,
                            exit_config=self.exit_config,
                            execution_config=(
                                replace(execution, cuda_graphs=False)
                                if role == "prefill"
                                else execution
                            ),
                            attention_backend=attention_backend,
                        ),
                    )
                    process = context.Process(
                        target=worker_main,
                        args=(role, device, name, child, model, options, self.config),
                        name=f"vllm-rlt-{role}-{device}",
                    )
                    process.start()
                    child.close()
                    self.peers[name] = Peer(name, role, process, parent)
            deadline = time.monotonic() + self.config.startup_timeout
            self._wait(lambda: all(p.ready for p in self.peers.values()), deadline)
            fingerprints = {p.info["fingerprint"] for p in self.peers.values()}
            if len(fingerprints) != 1:
                raise ValueError("P/D model weights or KV configuration do not match")
            self.model = SimpleNamespace(
                config=OuroConfig(**next(iter(self.peers.values())).info["model"])
            )
            for peer in self.peers.values():
                self._send(
                    peer.name,
                    "connect",
                    peers=[p.info["info"] for p in self.peers.values() if p.role != peer.role],
                )
            self._wait(lambda: all(p.connected for p in self.peers.values()), deadline)
        except BaseException:
            self._terminate()
            raise

    def _send(self, destination, kind, **fields):
        self.peers[destination].channel.send(dict(kind=kind, **fields))

    def _wait(self, condition, deadline):
        while not condition():
            self._poll()
            if time.monotonic() > deadline:
                raise TimeoutError("PD worker startup or shutdown deadline exceeded")
            time.sleep(0.001)

    def _poll(self):
        for peer in self.peers.values():
            for _ in range(self.config.max_control_messages):
                if not peer.channel.poll():
                    break
                try:
                    message = peer.channel.recv()
                except EOFError:
                    if peer.stopped:
                        break
                    raise RuntimeError(f"PD worker {peer.name} disconnected") from None
                self._message(peer, message)
            if not peer.process.is_alive() and not peer.stopped:
                raise RuntimeError(f"PD worker {peer.name} exited {peer.process.exitcode}")

    def _message(self, peer, m):
        kind = m["kind"]
        if kind == "profile_reply":
            pending = self._profile_replies.get(m.get("control_id"))
            if pending is not None and m.get("session_id") == self._profile_session:
                pending[peer.name] = m
            return
        if kind == "fatal":
            raise RuntimeError(f"PD {peer.role} failed:\n{m['error']}")
        if kind == "ready":
            peer.info, peer.ready = m, True
            return
        if kind == "connected":
            peer.connected = True
            return
        if kind == "stopped":
            peer.stopped = True
            self.worker_metrics[peer.name] = m
            return
        w = self.transfers.get(m.get("tid"))
        if w is None:
            return  # Delayed messages from retired generations.
        if peer.name not in (w.p, w.d):
            raise RuntimeError("PD response from wrong worker")
        now = time.monotonic() - w.created
        if kind == "reserved":
            w.timings["d_reserved"] = now
            if w.cancelled:
                self._send(w.d, "cancel_safe", tid=w.tid)
                return
            w.phase = "prefill"
            self._send(
                w.p,
                "prefill",
                tid=w.tid,
                tokens=w.request.prompt_token_ids,
                params=w.request.sampling_params,
                trace_id=w.trace_id,
                peer=self.peers[w.d].info["info"]["agent"],
                tables=m["tables"],
                slot=m["slot"],
                cached_tokens=m.get("cached_tokens", 0),
            )
        elif kind == "prefill_started":
            w.timings["prefill_started"] = now
        elif kind == "prefill_complete":
            if not w.p_compute_released:
                peer.compute_slots -= 1
                w.p_compute_released = True
            w.timings["prefill_compute_completed"] = now
        elif kind == "commit":
            w.timings["prefill_submitted"] = now
            if not w.cancelled:
                self._send(w.d, "commit", tid=w.tid, count=m["count"])
        elif kind == "activated":
            w.phase = "decode"
            w.timings["d_activated"] = now
            self._send(w.p, "ack", tid=w.tid)
        elif kind == "output":
            if not w.cancelled:
                output = replace(m["output"], request_id=w.request.request_id)
                w.request.generated_token_ids = list(output.token_ids)
                w.request.exit_depths = list(output.exit_depths)
                w.timings.setdefault("first_token", now)
                self.outputs.append(output)
                if output.finished:
                    w.finished = True
                    w.timings["finished"] = now
                    self.requests.pop(w.request.request_id, None)
        elif kind == "released":
            if peer.role == "prefill" and not w.p_released:
                w.p_released = True
                if not w.p_compute_released:
                    peer.compute_slots -= 1
                    w.p_compute_released = True
                peer.slots -= 1
                peer.blocks -= w.p_blocks
                if w.cancelled and not w.d_released:
                    self._send(w.d, "cancel_safe", tid=w.tid)
            elif peer.role == "decode" and not w.d_released:
                w.d_released = True
                peer.slots -= 1
                peer.blocks -= w.d_blocks
        elif kind == "rejected":
            # Coordinator holds exact credits; rejection indicates a state mismatch.
            raise RuntimeError("PD reserved capacity rejected by worker")
        else:
            raise RuntimeError(f"unknown PD response {kind}")
        if w.p_released and w.d_released:
            self.metrics.append(
                dict(
                    request_id=w.request.request_id,
                    transfer_id=w.tid,
                    cancelled=w.cancelled,
                    timings=w.timings,
                )
            )
            del self.transfers[w.tid]
        self.cache_manager.num_used_blocks = sum(p.blocks for p in self.peers.values())

    def _blocks(self, peer, tokens):
        return (
            (tokens + peer.info["block_size"] - 1) // peer.info["block_size"] * peer.info["depths"]
        )

    def add_request(self, request_id, prompt_token_ids, sampling_params=None, *, trace_id=None):
        if self.closed or self.failure:
            raise RuntimeError("PD engine is unavailable")
        if not isinstance(request_id, str) or not request_id or request_id in self.requests:
            raise ValueError("request ID must be nonempty and unique among active requests")
        if len(self.transfers) >= self.config.max_pending_requests:
            raise ValueError("PD pending-request limit reached")
        params = sampling_params or SamplingParams()
        if params.loopcd is not None and params.loopcd.strength != 0:
            raise ValueError("LoopCD P/D reference transfer is not implemented")
        cfg = self.model.config
        if not prompt_token_ids or any(
            type(t) is not int or not 0 <= t < cfg.vocab_size for t in prompt_token_ids
        ):
            raise ValueError("prompt must contain valid model token IDs")
        maximum = params.max_loops or cfg.total_ut_steps
        if maximum > cfg.total_ut_steps or params.min_loops > maximum:
            raise ValueError("requested loop bounds exceed model depth")
        if len(prompt_token_ids) + params.max_tokens - 1 > cfg.max_position_embeddings:
            raise ValueError("prompt plus decode exceeds context capacity")
        if trace_id is not None and (not isinstance(trace_id, str) or not trace_id):
            raise ValueError("trace_id must be a nonempty string")
        if self.exit_config.mode == "trace":
            trace_id = trace_id if trace_id is not None else request_id
            trace = self.exit_config.depths_by_request.get(trace_id, [])
            if (
                len(trace) < params.max_tokens
                or trace[0] != cfg.total_ut_steps
                or any(
                    type(d) is not int or not params.min_loops <= d <= maximum
                    for d in trace[1 : params.max_tokens]
                )
            ):
                raise ValueError("unknown or invalid exit trace")
        elif trace_id is not None:
            raise ValueError("trace_id requires trace exit mode")
        for role in ("prefill", "decode"):
            tokens = len(prompt_token_ids) + (params.max_tokens - 1 if role == "decode" else 0)
            if not any(
                self._blocks(p, tokens) <= p.info["num_blocks"]
                for p in self.peers.values()
                if p.role == role
            ):
                raise ValueError(f"request exceeds {role} KV pool capacity")
        request = Request(request_id, list(prompt_token_ids), params)
        w = Pending(request, uuid.uuid4().hex, trace_id)
        self.requests[request_id] = request
        self.transfers[w.tid] = w

    def _admit(self):
        # A bounded FIFO scan allows fitting requests to bypass temporary pressure.
        # Age protection stops new admissions after a waiting request's deadline
        # would otherwise be hidden by continuous small arrivals.
        waiting = list(self.transfers.values())[: self.config.max_pending_requests]
        if self.scheduling_policy == "priority":
            waiting.sort(key=lambda w: (w.request.sampling_params.priority, w.created))
        for w in waiting:
            if w.cancelled or w.phase != "waiting":
                continue
            candidates = {}
            for role in ("prefill", "decode"):
                length = len(w.request.prompt_token_ids) + (
                    w.request.sampling_params.max_tokens - 1 if role == "decode" else 0
                )
                eligible = [
                    p
                    for p in self.peers.values()
                    if p.role == role
                    and p.slots < p.info["max_seqs"]
                    and (role != "prefill" or p.compute_slots < p.info["active_limit"])
                    and p.blocks + self._blocks(p, length)
                    <= p.info["num_blocks"] - (p.info["watermark_blocks"] if p.slots else 0)
                ]
                if not eligible:
                    break
                candidates[role] = min(
                    eligible,
                    key=lambda p: (p.blocks / p.info["num_blocks"], p.slots / p.info["max_seqs"]),
                )
            if len(candidates) != 2:
                if time.monotonic() - w.created > self.config.request_timeout / 2:
                    break
                continue
            p, d = candidates["prefill"], candidates["decode"]
            w.p, w.d = p.name, d.name
            w.p_blocks = self._blocks(p, len(w.request.prompt_token_ids))
            w.d_blocks = self._blocks(
                d, len(w.request.prompt_token_ids) + w.request.sampling_params.max_tokens - 1
            )
            p.blocks += w.p_blocks
            d.blocks += w.d_blocks
            p.slots += 1
            p.compute_slots += 1
            d.slots += 1
            w.phase = "reserving"
            w.timings["admitted"] = time.monotonic() - w.created
            self._send(
                d.name,
                "reserve",
                tid=w.tid,
                tokens=w.request.prompt_token_ids,
                params=w.request.sampling_params,
                trace_id=w.trace_id,
                peer=p.info["info"]["agent"],
            )

    def abort_request(self, request_id):
        request = self.requests.pop(request_id)
        w = next(w for w in self.transfers.values() if w.request is request)
        w.cancelled = True
        self.outputs = deque(o for o in self.outputs if o.request_id != request_id)
        if self.closed or w.phase == "waiting":
            del self.transfers[w.tid]
        else:
            self._send(w.d, "cancel", tid=w.tid)
            if w.p_released:
                self._send(w.d, "cancel_safe", tid=w.tid)
            else:
                self._send(w.p, "cancel", tid=w.tid)
        request.stage = Stage.FINISHED
        request.finish_reason = FinishReason.ABORT
        return RequestOutput.from_request(request)

    def has_unfinished_requests(self):
        return bool(self.transfers)

    def step(self):
        if self.closed:
            raise RuntimeError("PD engine is closed")
        try:
            self._poll()
            if any(
                time.monotonic() - w.created > self.config.request_timeout
                for w in self.transfers.values()
            ):
                raise TimeoutError("PD request/transfer deadline exceeded")
            self._admit()
            outputs = list(self.outputs)
            self.outputs.clear()
            if not outputs:
                time.sleep(0.0005)
            return outputs
        except BaseException as error:
            self.failure = str(error)
            self._terminate()
            raise

    def _profile_manifest(self):
        result = dict(
            session_id=self._profile_session,
            recording=self._profile_recording,
            errors=getattr(self, "_profile_errors", []),
            ranks={
                str(rank): dict(
                    worker=name, **self._profile_statuses.get(name, {"state": "missing"})
                )
                for rank, name in enumerate(self.peers)
            },
        )
        result["artifacts_complete"] = all(
            rank.get("artifacts_complete", False) for rank in result["ranks"].values()
        )
        result["success"] = (
            bool(result["ranks"])
            and not result["errors"]
            and all(rank.get("success", False) for rank in result["ranks"].values())
        )
        if self._profile_session:
            result["capture_time"] = self._profile_timestamp
            _write_json_atomic(
                self._profile_root / f"manifest-{self._profile_timestamp}.json", result
            )
        return result

    def _profile_control(self, action, *, timeout=None, **fields):
        if self.closed:
            raise RuntimeError("PD engine is closed")
        control_id = uuid.uuid4().hex
        replies = {}
        self._profile_replies[control_id] = replies
        deadline = time.monotonic() + (self.config.startup_timeout if timeout is None else timeout)
        try:
            for rank, name in enumerate(self.peers):
                self._send(
                    name,
                    "profile_" + action,
                    control_id=control_id,
                    session_id=self._profile_session,
                    rank=rank,
                    **fields,
                )
            while len(replies) < len(self.peers):
                self._poll()
                if time.monotonic() >= deadline:
                    raise TimeoutError("PD profile control timed out")
                time.sleep(0.001)
        finally:
            self._profile_replies.pop(control_id, None)
            for name in self.peers:
                reply = replies.get(name)
                self._profile_statuses[name] = (
                    reply.get("result", {"state": "failed", "error": reply.get("error")})
                    if reply
                    else {"state": "missing", "error": "no profile acknowledgment"}
                )
            if len(replies) == len(self.peers) and not any("error" in r for r in replies.values()):
                self._profile_recording = any(
                    r.get("recording", False) for r in self._profile_statuses.values()
                )
            self._profile_manifest()
        errors = {name: reply["error"] for name, reply in replies.items() if "error" in reply}
        if errors:
            raise RuntimeError(f"PD profiling failed: {errors}")
        return self._profile_manifest()

    def start_profile(self, config, *, scheduled=False):
        if isinstance(config, dict):
            config = ProfileConfig(**config)
        if self._profile_session:
            self.profile_status()
        if self._profile_recording:
            raise RuntimeError("a profiling session is already active")
        if self._profile_session:
            if any(
                s.get("recording") or not s.get("artifacts_complete", True)
                for s in self._profile_statuses.values()
            ):
                raise RuntimeError("previous profile session has not finished")
        self._profile_session = uuid.uuid4().hex
        self._profile_timestamp = profile_timestamp(config.output_dir)
        self._profile_root = Path(config.output_dir).expanduser().resolve()
        self._profile_root.mkdir(parents=True, exist_ok=True)
        self._profile_statuses = {}
        self._profile_errors = []
        self._profile_recording = True
        try:
            return self._profile_control(
                "start",
                config=asdict(config),
                scheduled=scheduled,
                capture_time=self._profile_timestamp,
            )
        except BaseException as exc:
            self._profile_errors.append(str(exc))
            self._profile_recording = False
            try:
                self._profile_control("stop", timeout=self.config.shutdown_timeout)
            except Exception:
                pass  # Preserve the original failure; per-rank failures remain in manifest.
            raise

    def stop_profile(self):
        if not self._profile_session:
            return {"recording": False, "ranks": {}}
        self._profile_control("stop")
        self._profile_recording = False
        return self._profile_manifest()

    def profile_status(self):
        if not self._profile_session:
            return {"recording": False, "ranks": {}}
        if self.closed:
            return self._profile_manifest()
        return self._profile_control("status")

    def wait_for_profile_artifacts(self, timeout=None):
        if self._profile_session:
            self.profile_status()
        if self._profile_recording:
            raise RuntimeError("stop profiling before waiting for artifacts")
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._profile_session:
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            result = self._profile_control("status", timeout=remaining)
            if all(s.get("artifacts_complete", False) for s in self._profile_statuses.values()):
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("PD profile artifacts are still processing")
            time.sleep(0.02)
        return {"recording": False, "ranks": {}}

    def _terminate(self):
        if self._profile_session:
            self._profile_recording = False
            for name in self.peers:
                self._profile_statuses[name] = dict(
                    state="incomplete", error=self.failure or "terminated"
                )
            try:
                self._profile_manifest()
            except OSError:
                pass
        # Fail closed: stop all remote writers before destroying receive pools.
        for role in ("prefill", "decode"):
            selected = [p for p in self.peers.values() if p.role == role]
            for p in selected:
                if p.process.is_alive():
                    p.process.terminate()
            for p in selected:
                p.process.join(5)
                if p.process.is_alive():
                    p.process.kill()
                    p.process.join(5)
                p.channel.close()
        self.cache_manager.num_used_blocks = 0
        self.closed = True

    def close(self):
        if self.closed:
            return
        try:
            for rid in list(self.requests):
                self.abort_request(rid)
            deadline = time.monotonic() + self.config.shutdown_timeout
            self._wait(lambda: not self.transfers, deadline)
            if self._profile_session:
                self.stop_profile()
                self.wait_for_profile_artifacts(max(0, deadline - time.monotonic()))
            for peer in self.peers.values():
                self._send(peer.name, "stop")
            self._wait(lambda: all(p.stopped for p in self.peers.values()), deadline)
            for peer in self.peers.values():
                peer.process.join(max(0, deadline - time.monotonic()))
                if peer.process.is_alive():
                    raise TimeoutError("PD worker did not exit")
                peer.channel.close()
            self.closed = True
        except BaseException:
            self._terminate()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
