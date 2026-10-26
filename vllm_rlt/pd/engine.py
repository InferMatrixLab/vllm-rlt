"""CPU coordinator for independent prefill/decode GPU pools.

D reservations precede P computation. Control channels carry only metadata and
outputs and CPU-staged full-weight chunks; NIXL moves KV between GPU pools.
"""

import time
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path

import torch

from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.engine.llm_engine import StageClient
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.models import prepare_model
from vllm_rlt.profiling import ProfileConfig
from vllm_rlt.profiling_artifacts import _write_json_atomic, profile_timestamp
from vllm_rlt.request import FinishReason, Request, RequestOutput, RequestPhase
from vllm_rlt.sampling_params import SamplingParams

from .config import PDConfig
from .connector import PDStageConnector


@dataclass
class Peer:
    name: str
    role: str
    client: StageClient
    reserved_blocks: int = 0
    slots: int = 0


@dataclass
class Pending:
    request: Request
    tid: str
    trace_id: str | None
    created: float = field(default_factory=time.monotonic)
    p: str | None = None
    d: str | None = None
    phase: str = "waiting"
    p_reserved_blocks: int = 0
    d_reserved_blocks: int = 0
    cancelled: bool = False
    p_released: bool = False
    d_released: bool = False
    finished: bool = False
    timings: dict = field(default_factory=dict)
    latest_output: RequestOutput | None = None


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
        model_config=None,
        seed=0,
        speculative_config=None,
        terminate_workers_on_failure: bool = True,
    ):
        self.speculative_config = speculative_config
        model, revision, self.model_config = prepare_model(
            model, revision=revision, model_config=model_config
        )
        if self.model_config.model_type == "huginn_raven":
            raise ValueError("PD does not support Huginn")
        self.config = pd_config or PDConfig()
        self.exit_config = exit_config or ExitConfig(
            "ouro" if speculative_config else "ouro_delayed"
        )
        execution = execution_config or ExecutionConfig(async_scheduling=True)
        self._async_speculative = bool(speculative_config and execution.async_scheduling)
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
        self.metrics = deque(maxlen=1024)
        self.worker_metrics = {}
        self.outputs = deque()
        self.closed = False
        self.failure = None
        self.terminate_workers_on_failure = terminate_workers_on_failure
        self._weight_version = 0
        self._update_version: int | None = None
        self._update_failed = False
        self._paused = False
        self._profile_session = None
        self._profile_statuses = {}
        self._profile_recording = False
        self._profile_errors = []
        epoch = uuid.uuid4().hex
        try:
            for role, devices, cache, scheduler in (
                ("prefill", self.config.prefill_devices, p_cache, p_scheduler),
                ("decode", self.config.decode_devices, d_cache, d_scheduler),
            ):
                for device in devices:
                    name = f"{epoch}-{role}-{device}"
                    client = StageClient(
                        model,
                        model_config=self.model_config,
                        device=device,
                        stage_config=StageConfig(stage_id=name, role=role),
                        connector_factory=partial(PDStageConnector.create, config=self.config),
                        max_control_messages=self.config.max_control_messages,
                        startup_timeout=self.config.startup_timeout,
                        shutdown_timeout=self.config.shutdown_timeout,
                        wait_for_ready=False,
                        dtype=dtype,
                        revision=revision,
                        seed=seed,
                        cache_config=cache,
                        scheduler_config=scheduler,
                        exit_config=(
                            ExitConfig("ouro_delayed")
                            if role == "prefill"
                            and speculative_config is not None
                            and execution.async_scheduling
                            else self.exit_config
                        ),
                        speculative_config=speculative_config if role == "decode" else None,
                        execution_config=execution,
                        attention_backend=attention_backend,
                    )
                    self.peers[name] = Peer(name, role, client)
            deadline = time.monotonic() + self.config.startup_timeout
            self._wait(lambda: all(p.client.ready for p in self.peers.values()), deadline)
            fingerprints = {p.client.info["fingerprint"] for p in self.peers.values()}
            if len(fingerprints) != 1:
                raise ValueError("P/D model weights or KV configuration do not match")
            for peer in self.peers.values():
                self._send(
                    peer.name,
                    "connect",
                    peers=[
                        p.client.info["info"] for p in self.peers.values() if p.role != peer.role
                    ],
                )
            self._wait(lambda: all(p.client.connected for p in self.peers.values()), deadline)
        except BaseException:
            self._terminate()
            raise

    def _send(self, destination, kind, **fields):
        self.peers[destination].client.send(kind, **fields)

    def _wait(self, condition, deadline):
        while not condition():
            self._poll()
            if time.monotonic() > deadline:
                raise TimeoutError("PD worker startup or shutdown deadline exceeded")
            time.sleep(0.001)

    def _poll(self):
        for peer in self.peers.values():
            for message in peer.client.poll():
                self._message(peer, message)

    def _message(self, peer, m):
        kind = m["kind"]
        if kind in ("ready", "connected", "reply"):
            return  # StageClient owns common worker lifecycle state.
        if kind == "stopped":
            self.worker_metrics[peer.name] = m
            return
        w = self.transfers.get(m["request_id"] if kind == "output" else m.get("tid"))
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
                peer=self.peers[w.d].client.info["info"]["agent"],
                tables=m["tables"],
                slot=m["slot"],
                cached_tokens=m.get("cached_tokens", 0),
            )
        elif kind == "prefill_started":
            w.timings["prefill_started"] = now
        elif kind == "prefill_complete":
            w.timings["prefill_compute_completed"] = now
            w.timings["prefill_transfer_boundaries"] = m["boundaries"]
            w.timings["prefill_transfer_overlaps"] = m["overlaps"]
        elif kind == "commit":
            w.timings["prefill_submitted"] = now
            if not w.cancelled:
                self._send(w.d, "commit", tid=w.tid, count=m["count"])
        elif kind == "received":
            w.timings["d_received"] = now
            self._send(w.p, "ack", tid=w.tid)
        elif kind == "activated":
            w.phase = "decode"
            w.timings["d_activated"] = now
        elif kind == "output":
            if not w.cancelled:
                output = replace(m["output"], request_id=w.request.request_id)
                if output.weight_version != w.request.weight_version:
                    raise RuntimeError("PD workers returned a different policy version")
                w.latest_output = output
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
                peer.slots -= 1
                peer.reserved_blocks -= w.p_reserved_blocks
                if w.cancelled and not w.d_released:
                    self._send(w.d, "cancel_safe", tid=w.tid)
            elif peer.role == "decode" and not w.d_released:
                w.d_released = True
                peer.slots -= 1
                peer.reserved_blocks -= w.d_reserved_blocks
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

    def _blocks(self, peer, tokens):
        return (
            (tokens + peer.client.info["block_size"] - 1)
            // peer.client.info["block_size"]
            * peer.client.info["depths"]
        )

    def add_request(self, request_id, prompt_token_ids, sampling_params=None, *, trace_id=None):
        if self.closed or self.failure:
            raise RuntimeError("PD engine is unavailable")
        if self._paused or self._update_version is not None or self._update_failed:
            raise RuntimeError("Generation admission is paused")
        if not isinstance(request_id, str) or not request_id or request_id in self.requests:
            raise ValueError("request ID must be nonempty and unique among active requests")
        if len(self.transfers) >= self.config.max_pending_requests:
            raise ValueError("PD pending-request limit reached")
        params = sampling_params or SamplingParams()
        cfg = self.model_config
        if self.speculative_config is not None:
            if (
                params.exit_threshold != 1.0
                or (params.max_loops or cfg.total_ut_steps) != cfg.total_ut_steps
            ):
                raise ValueError(
                    "speculative requests require fixed target depth and exit_threshold=1"
                )
            if self._async_speculative and params.logprobs is not None:
                raise ValueError("async speculative decoding does not yet support logprobs")
            if self._async_speculative and params.temperature != 0:
                raise ValueError("async speculative decoding currently requires greedy sampling")
        if not prompt_token_ids or any(
            type(t) is not int or not 0 <= t < cfg.vocab_size for t in prompt_token_ids
        ):
            raise ValueError("prompt must contain valid model token IDs")
        if any(t >= cfg.vocab_size for t in params.stop_token_ids):
            raise ValueError("stop token IDs must be within the model vocabulary")
        maximum = params.max_loops or cfg.total_ut_steps
        if maximum > cfg.total_ut_steps or params.min_loops > maximum:
            raise ValueError("requested loop bounds exceed model depth")
        params = replace(params, max_loops=maximum, top_k=min(params.top_k, cfg.vocab_size))
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
                self._blocks(p, tokens) <= p.client.info["num_blocks"]
                for p in self.peers.values()
                if p.role == role
            ):
                raise ValueError(f"request exceeds {role} KV pool capacity")
        request = Request(
            request_id, list(prompt_token_ids), params, weight_version=self._weight_version
        )
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
                    and p.slots < p.client.info["transfer_slots"]
                    and p.reserved_blocks + self._blocks(p, length)
                    <= p.client.info["num_blocks"]
                    - (p.client.info["watermark_blocks"] if p.slots else 0)
                ]
                if not eligible:
                    break
                candidates[role] = min(
                    eligible,
                    key=lambda p: (
                        p.reserved_blocks / p.client.info["num_blocks"],
                        p.slots / p.client.info["transfer_slots"],
                    ),
                )
            if len(candidates) != 2:
                if time.monotonic() - w.created > self.config.request_timeout / 2:
                    break
                continue
            p, d = candidates["prefill"], candidates["decode"]
            w.p, w.d = p.name, d.name
            w.p_reserved_blocks = self._blocks(p, len(w.request.prompt_token_ids))
            w.d_reserved_blocks = self._blocks(
                d, len(w.request.prompt_token_ids) + w.request.sampling_params.max_tokens - 1
            )
            p.reserved_blocks += w.p_reserved_blocks
            d.reserved_blocks += w.d_reserved_blocks
            p.slots += 1
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
                peer=p.client.info["info"]["agent"],
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
        request.phase = RequestPhase.FINISHED
        request.finish_reason = FinishReason.ABORT
        if w.latest_output is not None:
            return replace(w.latest_output, finished=True, finish_reason=FinishReason.ABORT.value)
        return RequestOutput.from_request(request)

    def get_weight_version(self) -> int:
        return self._weight_version

    def pause_generation(self):
        self._paused = True

    def resume_generation(self):
        if self._update_version is not None or self._update_failed:
            raise RuntimeError("Complete the weight update before resuming generation")
        self._paused = False

    def _weight_control(self, action, **arguments):
        controls = []
        try:
            for peer in self.peers.values():
                control_id = peer.client.submit_control(action, **arguments)
                controls.append((peer, control_id))
            self._wait(
                lambda: all(cid in peer.client.replies for peer, cid in controls),
                time.monotonic() + self.config.request_timeout,
            )
            replies = [peer.client.take_control_reply(cid) for peer, cid in controls]
            errors = [reply["error"] for reply in replies if "error" in reply]
            if errors:
                raise RuntimeError("PD weight control failed: " + "; ".join(errors))
            return [reply.get("result") for reply in replies]
        except Exception:
            # Peers can have copied or committed before another peer fails.
            self._update_failed = True
            raise
        finally:
            for peer, cid in controls:
                peer.client.discard_control(cid)

    def reset_prefix_cache(self):
        if self.has_unfinished_requests():
            raise RuntimeError("Finish active PD transfers before resetting cached KV")
        self._weight_control("reset_prefix_cache")

    def start_weight_update(self, version: int | None = None):
        if self.closed or self.failure or self.has_unfinished_requests():
            raise RuntimeError("Weight updates require an available, idle PD engine")
        latest = max(self._weight_version, self._update_version or 0)
        version = latest + 1 if version is None else version
        if type(version) is not int or version <= latest:
            raise ValueError("PD weight version must exceed all previous update attempts")
        self._update_version, self._update_failed = version, False
        self._weight_control("start_weight_update", version=version)

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        if self._update_version is None or self._update_failed:
            raise RuntimeError("Start a full PD weight update first")
        # CPU staging avoids CUDA tensor ownership across spawned worker processes.
        weights = [(name, tensor.detach().cpu()) for name, tensor in weights]
        self._weight_control("update_weights", weights=weights)

    def finish_weight_update(self):
        if self._update_version is None or self._update_failed:
            raise RuntimeError("No successful PD weight update to finish")
        versions = self._weight_control("finish_weight_update")
        if any(version != self._update_version for version in versions):
            self._update_failed = True
            raise RuntimeError("PD workers committed different policy versions")
        self._weight_version, self._update_version = self._update_version, None

    def request_ids(self):
        return tuple(self.requests)

    def has_request(self, request_id):
        return request_id in self.requests

    def stats(self):
        # Coordinator credits are reservations, not measured worker KV usage.
        return dict(
            requests=len(self.requests),
            reserved_blocks=sum(peer.reserved_blocks for peer in self.peers.values()),
        )

    def has_unfinished_requests(self):
        return bool(self.transfers)

    def step(self):
        if self.closed:
            raise RuntimeError("PD engine is closed")
        if self._update_version is not None or self._update_failed:
            raise RuntimeError("Generation is unavailable during a PD weight update")
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
            errors=self._profile_errors,
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
        method = {"start": "start_profile", "stop": "stop_profile", "status": "profile_status"}[
            action
        ]
        calls, replies = {}, {}
        deadline = time.monotonic() + (self.config.startup_timeout if timeout is None else timeout)
        try:
            for rank, (name, peer) in enumerate(self.peers.items()):
                kwargs = dict(fields)
                if action == "start":
                    kwargs.update(rank=rank, managed_manifest=False)
                calls[name] = peer.client.submit_control(
                    method, session_id=self._profile_session, **kwargs
                )
            while len(replies) < len(calls):
                self._poll()
                for name, control_id in calls.items():
                    reply = self.peers[name].client.take_control_reply(control_id)
                    if reply is not None:
                        if reply.get("session_id") != self._profile_session:
                            raise RuntimeError("PD profile reply session mismatch")
                        replies[name] = reply
                if len(replies) == len(calls):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("PD profile control timed out")
                time.sleep(0.001)
        finally:
            for name, control_id in calls.items():
                self.peers[name].client.discard_control(control_id)
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
        if not self.terminate_workers_on_failure:
            self.closed = True
            return
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
                p.client.terminate()
            for p in selected:
                p.client.join(5, force=True)
        for peer in self.peers.values():
            peer.reserved_blocks = peer.slots = 0
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
                peer.client.request_stop()
            self._wait(lambda: all(p.client.stopped for p in self.peers.values()), deadline)
            for peer in self.peers.values():
                peer.client.join(max(0, deadline - time.monotonic()))
            self.closed = True
        except BaseException:
            self._terminate()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
