import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from vllm_rlt.attention import BackendCapabilities, create_backend
from vllm_rlt.config import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SchedulerConfig,
    resolve_cuda_graph_rows,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import plan_cache
from vllm_rlt.core.scheduler import Scheduler
from vllm_rlt.core.scheduling_policy import requires_staged_prefill
from vllm_rlt.engine.connector import StageConnector
from vllm_rlt.engine.preemption import PreemptionManager
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.profiling import Profiler
from vllm_rlt.request import FinishReason, Request, RequestOutput, RequestPhase
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.executor import Executor
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.output import ExitSignal, ModelRunnerOutput
from vllm_rlt.worker.sampler import RngRegistry
from vllm_rlt.worker.speculative import SpeculativeRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm_rlt.worker.async_speculative import AsyncSpeculativeRunner


@dataclass(frozen=True)
class SignalHandle:
    ticket: object
    row: int


def _huginn_packed_prefill_disabled() -> bool:
    """Huginn packs prefill like Ouro; VLLM_RLT_HUGINN_PACKED_PREFILL=0 keeps per-row prefill.

    Measured on GH200 (Huginn-0125, 32 depths): 1.2-2.1x throughput and -39..-64%
    TTFT at 512-2048-token prompts; GSM8K-87 (70 paired questions) 21 vs 20 correct
    with 47/70 identical token sequences, i.e. no regression.
    """
    import os

    return os.environ.get("VLLM_RLT_HUGINN_PACKED_PREFILL", "1") == "0"


def packed_prefill_eligible(prefill_uva, *, device_type, layout, boundary_kv, staged_prefill):
    """Whether an unset ``prefill_uva`` may resolve to packed prefill.

    Explicit settings win. Packed prefill needs CUDA packed attention and
    last_exited KV; staged prefill (wavefront chunking, mixed mode) keeps the
    per-row path, which its schedulers were written against. Boundary-KV models
    (Huginn) pack unless VLLM_RLT_HUGINN_PACKED_PREFILL=0.
    """
    if prefill_uva is not None:
        return bool(prefill_uva)
    if boundary_kv and _huginn_packed_prefill_disabled():
        return False
    return device_type == "cuda" and layout == "last_exited" and not staged_prefill


class StageEngine:
    """Single-device loop-level engine with synchronous or pipelined scheduling."""

    def __init__(
        self,
        model,
        *,
        cache_config=None,
        scheduler_config=None,
        attention_backend="torch",
        exit_config=None,
        execution_config=None,
        speculative_config=None,
        stage_config=None,
        connector: StageConnector | None = None,
    ):
        self.model = model
        self.stage_config = stage_config or StageConfig()
        self.profiling = Profiler(next(model.parameters()).device)
        cache_config = cache_config or CacheConfig()
        scheduler_config = scheduler_config or SchedulerConfig()
        parameter = next(model.parameters())
        config = model.config
        if getattr(model, "requires_boundary_kv", False):
            if cache_config.layout != "last_exited":
                raise ValueError("Huginn requires last_exited KV")
            if cache_config.enable_prefix_caching:
                raise ValueError("Huginn prefix caching is not yet supported")
        self.exit_config = exit_config or ExitConfig()
        self.execution_config = resolve_cuda_graph_rows(
            execution_config or ExecutionConfig(), scheduler_config, speculative_config
        )
        staged_prefill = requires_staged_prefill(
            scheduler_config, produces_tokens=self.stage_config.produces_tokens
        )
        if getattr(model, "requires_boundary_kv", False):
            if staged_prefill:
                raise ValueError(
                    "Huginn does not support staged prefill required by this scheduler"
                )
            if self.execution_config.async_scheduling:
                raise ValueError("Huginn requires synchronous scheduling")
            if self.execution_config.prefill_uva is True and _huginn_packed_prefill_disabled():
                raise ValueError(
                    "Huginn packed prefill is disabled by VLLM_RLT_HUGINN_PACKED_PREFILL=0"
                )
            if speculative_config is not None:
                raise ValueError("Huginn speculative decoding is not yet supported")
        self.speculative_config = speculative_config
        self.pipeline_speculative = bool(
            speculative_config and speculative_config.mode == "pipeline"
        )
        self.async_speculative = bool(speculative_config and self.execution_config.async_scheduling)
        if speculative_config is not None:
            if cache_config.layout != "last_exited":
                raise ValueError("speculative decoding requires last_exited KV")
            if speculative_config.target_loops != config.total_ut_steps:
                raise ValueError("speculative target_loops must equal the model full depth")
            if self.exit_config.mode != "ouro":
                raise ValueError("speculative decoding requires fixed-depth ouro exit mode")
            if self.async_speculative and parameter.device.type != "cuda":
                raise ValueError("async speculative decoding requires CUDA")
            if scheduler_config.mode != "refill" or (
                self.async_speculative
                and not self.pipeline_speculative
                and scheduler_config.enable_preemption
            ):
                raise ValueError(
                    "speculative decoding requires refill scheduling without preemption"
                )
            # Retained q distributions and sampling scratch coexist with target
            # logits. Reserve beyond ordinary prefill/core profiling, even when
            # requests later choose sampling rather than greedy.
            scratch = scheduler_config.max_num_batched_tokens * (
                config.vocab_size * 24 + config.hidden_size * parameter.element_size() * 2
            )
            if self.async_speculative:
                # Two in-flight rounds, plus device and pinned page-table buffers.
                width = (config.max_position_embeddings + cache_config.block_size - 1) // (
                    cache_config.block_size
                )
                scratch = 2 * scratch + 8 * scheduler_config.max_num_seqs * (
                    config.total_ut_steps * width
                )
            cache_config = replace(
                cache_config, memory_reserve_bytes=cache_config.memory_reserve_bytes + scratch
            )
        if self.execution_config.cuda_graphs and parameter.device.type != "cuda":
            raise ValueError("CUDA graphs require CUDA with Triton or FlashAttention")
        if self.execution_config.cuda_graphs and speculative_config is not None:
            rows = min(
                scheduler_config.max_num_seqs * (speculative_config.num_speculative_tokens + 1),
                scheduler_config.max_num_batched_tokens,
            )
            if rows > self.execution_config.cuda_graph_max_batch_size:
                logger.warning(
                    "CUDA Graph row limit %d cannot cover %d verification rows "
                    "(max_num_seqs=%d, K=%d); larger batches run eager. "
                    "Raise cuda_graph_max_batch_size, or leave it unset to cover them, "
                    "and budget additional graph memory.",
                    self.execution_config.cuda_graph_max_batch_size,
                    rows,
                    scheduler_config.max_num_seqs,
                    speculative_config.num_speculative_tokens,
                )
        if self.execution_config.async_scheduling:
            if not self.async_speculative and self.exit_config.mode not in (
                "ouro_delayed",
                "random_lookahead",
                "trace",
            ):
                raise ValueError(
                    "async scheduling requires ouro_delayed, random_lookahead or trace exit mode"
                )
        if self.execution_config.prefill_uva and (
            parameter.device.type != "cuda" or cache_config.layout != "last_exited"
        ):
            raise ValueError("prefill_uva requires CUDA packed prefill with last_exited KV")
        if staged_prefill and cache_config.layout != "last_exited":
            raise ValueError("staged prefill requires last_exited KV planes")
        auto_packed = packed_prefill_eligible(
            self.execution_config.prefill_uva,
            device_type=parameter.device.type,
            layout=cache_config.layout,
            boundary_kv=getattr(model, "requires_boundary_kv", False),
            staged_prefill=staged_prefill,
        )

        def backend(packed_prefill):
            return create_backend(
                attention_backend,
                parameter.device,
                parameter.dtype,
                config.head_dim,
                cache_config.block_size,
                required_capabilities=BackendCapabilities(
                    packed_prefill=packed_prefill,
                    cuda_graphs=self.execution_config.cuda_graphs,
                    async_scheduling=(
                        parameter.device.type == "cuda" and self.execution_config.async_scheduling
                    ),
                    device_context_lengths=self.async_speculative,
                ),
            )

        attention = backend(bool(self.execution_config.prefill_uva))
        if auto_packed:
            # Keep the backend the user would get anyway; only Triton needs the
            # explicit opt-in to pack prefill, FlashAttention-4 packs natively.
            if attention.info["selected_backend"] == "triton":
                attention = backend(True)
            auto_packed = bool(attention.capabilities.packed_prefill)
        if self.execution_config.prefill_uva is None:
            self.execution_config = replace(self.execution_config, prefill_uva=auto_packed)
        num_blocks, self.memory_plan = plan_cache(
            model,
            cache_config,
            scheduler_config,
            self.execution_config,
            attention.info["selected_backend"],
            speculative=speculative_config,
            staged_prefill=staged_prefill,
        )
        self.cache_manager = KVCacheManager(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_loops=config.total_ut_steps,
            num_blocks=num_blocks,
            layout=cache_config.layout,
            block_size=cache_config.block_size,
            device=parameter.device,
            dtype=parameter.dtype,
            backend=attention_backend,
            attention=attention,
            enable_prefix_caching=cache_config.enable_prefix_caching,
            incremental_allocation=cache_config.incremental_allocation,
            watermark_ratio=cache_config.watermark_ratio,
            recurrent_layers=getattr(model, "recurrent_kv_layers", None),
        )
        self.scheduler = Scheduler(
            scheduler_config,
            self.cache_manager,
            speculative_config,
            produces_tokens=self.stage_config.produces_tokens,
            total_ut_steps=config.total_ut_steps,
            eos_token_id=config.eos_token_id,
            exit_mode=self.exit_config.mode,
        )
        self.connector = connector
        # One RNG registry per engine, shared by both runners: the CODA and
        # speculative paths must advance the same per-request slot, and only the
        # engine knows when a request terminates.
        self.rng = RngRegistry()
        self.model_runner = ModelRunner(
            model,
            self.cache_manager,
            exit_config=self.exit_config,
            execution_config=self.execution_config,
            scheduler_config=scheduler_config,
            staged_prefill=self.scheduler.staged_prefill,
            rng=self.rng,
        )
        self.speculative_runner = (
            SpeculativeRunner(
                model,
                self.cache_manager,
                speculative_config,
                self.execution_config,
                rng=self.rng,
                runtime=self.model_runner.runtime,
                compute_stream=self.model_runner.core_stream,
            )
            if speculative_config is not None and not self.async_speculative
            else None
        )
        if self.async_speculative:
            from vllm_rlt.worker.async_speculative import AsyncSpeculativeRunner

            self.speculative_runner = AsyncSpeculativeRunner(
                model,
                self.cache_manager,
                speculative_config,
                scheduler_config,
                execution=self.execution_config,
                compute_stream=self.model_runner.core_stream,
                copy_stream=self.model_runner.copy_stream,
                rng=self.rng,
                runtime=self.model_runner.runtime,
            )
        self.executor = Executor(self.model_runner, speculation=self.speculative_runner)
        self._pending_speculative = []
        self._retiring_speculative = {}
        self._exit_traces = {
            key: tuple(values) for key, values in (self.exit_config.depths_by_request or {}).items()
        }
        # Unconsumed exit scores: request ID -> SignalHandle.
        # Delayed policies consume the preceding loop's score after submitting the next.
        self._pending_exit_signals = {}
        self._pending_coda = []
        self._inflight = []
        self._overlap_boundary = False
        self.last_schedule = None
        self.preemption = PreemptionManager(self)
        if scheduler_config.enable_preemption:
            self.scheduler.suspend_callback = self.preemption.suspend
            self.scheduler.resume_callback = self.preemption.resume

    @property
    def connector(self):
        return self._connector

    @connector.setter
    def connector(self, connector):
        self._connector = connector
        self._connector_blocked = set()
        self.scheduler.execution_ready = (
            self._execution_ready
            if connector is not None or not self.stage_config.produces_tokens
            else None
        )
        self.scheduler.exports_request = self._exports_request if connector is not None else None

    def _exports_request(self, request):
        return self._connector.exports(request)

    def _execution_ready(self, request):
        if not self.stage_config.produces_tokens:
            if self.connector is None:
                raise RuntimeError("a prefill stage requires an outgoing connector")
            if not self.connector.exports(request):
                raise ValueError("prefill-only stage requires an outgoing route per request")
        ready = self.connector.can_schedule(request)
        if not ready:
            self._connector_blocked.add(request.request_id)
        return ready

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        *,
        trace_id: str | None = None,
    ):
        if not self.stage_config.accepts_prompt:
            raise ValueError("decode stage requests must enter through reserve_transfer")
        self._add_request(request_id, prompt_token_ids, sampling_params, trace_id=trace_id)

    def _add_request(self, request_id, prompt_token_ids, sampling_params, *, trace_id=None):
        """Validate and register a stage-local request from either ingress path."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        params = sampling_params or SamplingParams()
        if (
            self.async_speculative
            or (self.speculative_config and self.speculative_config.mode == "pipeline")
        ) and params.temperature != 0:
            mode = "async" if self.async_speculative else "pipeline"
            raise ValueError(f"{mode} speculative decoding currently requires greedy sampling")
        if request_id in self._retiring_speculative:
            self._retire_speculative(wait=True, request_id=request_id)
        config = self.model.config
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if any(type(t) is not int or not 0 <= t < config.vocab_size for t in prompt_token_ids):
            raise ValueError("prompt token IDs must be integers within the model vocabulary")
        max_loops = params.max_loops or config.total_ut_steps
        if max_loops > config.total_ut_steps or params.min_loops > max_loops:
            raise ValueError("requested loop bounds exceed the model's supported depth")
        if self.speculative_config is not None and (
            max_loops != self.speculative_config.target_loops or params.exit_threshold != 1.0
        ):
            raise ValueError("speculative requests require fixed target depth and exit_threshold=1")
        if trace_id is not None:
            if not isinstance(trace_id, str) or not trace_id:
                raise ValueError("trace_id must be a nonempty string")
            if self.exit_config.mode != "trace":
                raise ValueError("trace_id requires trace exit mode")
        trace = ()
        if self.exit_config.mode == "trace":
            key = request_id if trace_id is None else trace_id
            if key not in self._exit_traces:
                raise ValueError(f"unknown exit trace: {key!r}; specify a configured trace_id")
            trace = self._exit_traces[key]
            if (
                len(trace) < params.max_tokens
                or trace[0] != config.total_ut_steps
                or any(
                    type(d) is not int or not params.min_loops <= d <= max_loops
                    for d in trace[1 : params.max_tokens]
                )
            ):
                raise ValueError(
                    "exit trace must cover every output and obey prefill/decode bounds"
                )
        capacity = len(prompt_token_ids) + params.max_tokens - 1
        if capacity > config.max_position_embeddings:
            raise ValueError("prompt plus decode positions exceed the model context length")
        cache = self.cache_manager
        required = cache.required_blocks(capacity)
        if required > cache.num_blocks:
            raise ValueError(
                f"request requires {required} KV blocks but cache has {cache.num_blocks}; "
                "increase num_blocks or reduce prompt/max_tokens"
            )
        self.scheduler.add_request(
            Request(
                request_id,
                list(prompt_token_ids),
                params,
                exit_trace=trace,
                # Prefill always runs the model's full depth, whatever decode
                # loop bound the request asks for.
                prefill_depth_ranges=Request.prefill_boundaries(
                    len(prompt_token_ids), config.total_ut_steps
                ),
            )
        )

    def request_ids(self):
        return tuple(self.scheduler.requests)

    def has_request(self, request_id):
        request = self.scheduler.requests.get(request_id)
        return request is not None and request.phase != RequestPhase.FINISHED

    def stats(self):
        return dict(
            requests=len(self.scheduler.requests), used_blocks=self.cache_manager.num_used_blocks
        )

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests or bool(self._pending_speculative)

    def close(self):
        """Drain owned GPU work before serving teardown drops tensor owners."""
        self.profiling.close()
        self.model_runner.synchronize()
        if self.async_speculative:
            cast("AsyncSpeculativeRunner", self.speculative_runner).synchronize()
        for rid in list(self.scheduler.requests):
            self.abort_request(rid)
        if self.async_speculative:
            self._discard_speculative()

    def _discard_speculative(self):
        """Release speculative owners after the caller has drained both streams."""
        self._retire_speculative(wait=True)
        for ticket in self._pending_speculative:
            ticket.retire(discard=True)
        self._pending_speculative.clear()
        cast("AsyncSpeculativeRunner", self.speculative_runner).discard_buffers()

    def abort_request(self, request_id: str) -> RequestOutput:
        request = self.scheduler.requests[request_id]
        if request.phase == RequestPhase.FINISHED:
            return RequestOutput.from_request(request)
        if self.async_speculative:
            request = self.scheduler.requests[request_id]
            self._finish(request, FinishReason.ABORT)
            return RequestOutput.from_request(request)
        self.preemption.discard_snapshot(request_id)
        if self.speculative_runner is not None:
            self.speculative_runner.release(request)
        self.model_runner.release(request_id)
        self._pending_exit_signals.pop(request_id, None)
        return RequestOutput.from_request(self.scheduler.abort(request_id))

    def start_profile(self, config, *, scheduled=False, **identity):
        return self.profiling.start(config, scheduled=scheduled, **identity)

    def stop_profile(self):
        return self.profiling.stop()

    def profile_status(self):
        return self.profiling.status()

    def wait_for_profile_artifacts(self, timeout=None):
        return self.profiling.wait(timeout)

    def step(self) -> list[RequestOutput]:
        if not self.profiling.recording:
            return self._step()
        try:
            outputs = self._step()
            self.profiling.step()
            return outputs
        except BaseException:
            self.profiling.stop()
            raise

    def _step(self) -> list[RequestOutput]:
        if self.execution_config.async_scheduling:
            try:
                if self.async_speculative:
                    return self._step_async_speculative()
                return self._step_async()
            except Exception:
                if self.connector is not None:
                    # The deployment owner must terminate remote writers before
                    # pages can be reclaimed after a fatal transport/compute error.
                    raise
                # A submission can fail before recording an event. Drain owned
                # streams before invalidating requests or recycling any memory.
                self.model_runner.synchronize()
                if self.async_speculative:
                    cast("AsyncSpeculativeRunner", self.speculative_runner).synchronize()
                for rid in list(self.scheduler.requests):
                    self.abort_request(rid)
                if self.async_speculative:
                    self._discard_speculative()
                self._pending_exit_signals.clear()
                self._pending_coda.clear()
                self._inflight.clear()
                raise
        batch = self._schedule()
        self.last_schedule = batch
        if batch is None:
            # PD imports wait for external KV completion; yield to the IPC loop.
            if not self._waiting_for_connector():
                raise RuntimeError("scheduler made no progress")
            return []
        try:
            ticket = self.executor.submit(batch)
            # Only host-dependent decisions collect now. Prompt/PRELUDE submissions
            # advance logically under their event without a readback wait.
            result = ticket.collect() if ticket.has_result else None
            return self._update(batch, result)
        except Exception:
            if self.connector is not None:
                raise  # Transfer leases remain quarantined until peer shutdown.
            # A failed execution may have partially written KV; invalidate the affected requests.
            for scheduled_request in batch.requests:
                if scheduled_request.request.request_id in self.scheduler.requests:
                    self.abort_request(scheduled_request.request.request_id)
            raise

    def _schedule(self, *, prefer_recurrent=False):
        self._connector_blocked.clear()
        return self.scheduler.schedule(prefer_recurrent=prefer_recurrent)

    def _waiting_for_connector(self):
        # Capture this scheduling attempt's readiness observation. Re-querying
        # an event here can misclassify progress that occurred after selection.
        transfer_waiters = {
            r.request_id
            for r in self.scheduler.requests.values()
            if r.phase in (RequestPhase.RECEIVING, RequestPhase.TRANSFERRING)
            or (
                r.request_id in self._connector_blocked
                and r.request_id in self.scheduler.queues[r.phase]
            )
        }
        return all(
            r.request_id in transfer_waiters
            # An admitted export may retain a compute slot while backpressured,
            # even before it reaches TRANSFERRING. Queued arrivals must yield too.
            # A WAITING request absent from its queue remains an invariant error.
            or (transfer_waiters and r.request_id in self.scheduler.queues[RequestPhase.WAITING])
            for r in self.scheduler.requests.values()
        )

    def _notify_connector_prefill(self, scheduled_request, phase, progress):
        request = scheduled_request.request
        if self.connector is not None and self.connector.exports(request):
            self.connector.prefill_submitted(
                request,
                phase=phase,
                start=(
                    scheduled_request.token_start
                    if phase == RequestPhase.PREFILL
                    else progress.transfer_start
                    if progress.transfer_start is not None
                    else scheduled_request.chunk_start
                ),
                progress=progress,
                event=self.model_runner.events.get(request.request_id),
                stream=self.model_runner.core_stream,
            )

    def reserve_transfer(
        self, request_id, prompt_token_ids, sampling_params, *, receiving, trace_id=None
    ):
        """Create one stage-local Request and reserve/pin its transfer allocation.

        The wire ID includes the transfer generation. Subsequent chunk messages
        reuse this Request; a repeated reservation is rejected by add_request.
        None means admission failed without retaining a request or KV allocation.
        """
        if receiving:
            self._add_request(request_id, prompt_token_ids, sampling_params, trace_id=trace_id)
        else:
            self.add_request(request_id, prompt_token_ids, sampling_params, trace_id=trace_id)
        request = self.scheduler.requests[request_id]
        plan = self.scheduler.reserve_transfer(request)
        if plan is None:
            self.scheduler.abort(request_id)
            return None
        count = len(prompt_token_ids)
        hit = plan.cached_tokens
        request.num_prefilled_tokens = request.num_preluded_tokens = hit
        request.prefill_window_origin = hit
        request.prefill_reclaimable_tokens = hit
        request.prefill_depth_ranges[:] = Request.prefill_boundaries(
            count, self.model.config.total_ut_steps, start=hit
        )
        # Reservation owns transport resources, not a compute slot. Outgoing
        # requests compete with local arrivals in the scheduler's WAITING queue.
        if receiving:
            self.scheduler.queues[RequestPhase.WAITING].remove(request_id)
            request.phase = RequestPhase.RECEIVING
        return request

    def suspend_transfer(self, request):
        """Cancel future computation, retaining in-flight state and transfer leases."""
        return self.scheduler.suspend_for_transfer(request)

    def release_transfer_compute(self, request, *, ready_event):
        """Release compute state after the connector's final staging read completes.

        KV and transfer staging remain leased until delivery or safe cancellation.
        Query is nonblocking; a stale Request can never release its replacement.
        """
        if self.scheduler.requests.get(request.request_id) is not request:
            return False
        if request.phase != RequestPhase.TRANSFERRING:
            raise ValueError("request is not awaiting transfer completion")
        if not ready_event.query():
            return False
        self.model_runner.release(request.request_id)
        request.hidden_state = None
        request.compute_released = True
        return True

    def release_transfer(self, request):
        """Called only after remote accesses are known to have ended."""
        if self.scheduler.requests.get(request.request_id) is not request:
            return False
        rid = request.request_id
        if self.async_speculative:
            if request.phase != RequestPhase.FINISHED:
                self._finish(request, FinishReason.ABORT)
            self._retire_speculative()
            if self.scheduler.requests.get(rid) is request:
                return False
        else:
            self.model_runner.release(rid)
            self.scheduler.abort(rid)
        if self.cache_manager.has_allocation(rid):
            self.cache_manager.unpin_transfer(rid, rid)
        return True

    def accept_import(self, request, hidden, *, cached_tokens=0):
        """Queue a valid imported Request for ordinary scheduler admission.

        The transfer generation is checked by the connector before this call;
        object identity additionally rejects local cancellation/ID reuse.
        """
        if self.scheduler.requests.get(request.request_id) is not request:
            return False
        if (
            request.phase != RequestPhase.RECEIVING
            or request.request_id in self.scheduler.queues[RequestPhase.WAITING]
        ):
            raise ValueError("request is not awaiting an import")
        count = len(request.prompt_token_ids)
        self.cache_manager.mark_imported_prefix(request.request_id, count, start=cached_tokens)
        self.cache_manager.publish_prefix(request.request_id, request.prompt_token_ids, count)
        request.num_prefilled_tokens = request.num_preluded_tokens = count
        request.prefill_window_origin = count
        request.prefill_reclaimable_tokens = count
        request.loops_done = self.model.config.total_ut_steps
        request.hidden_state = hidden
        # Keep RECEIVING until the scheduler grants a compute slot. The lease
        # protects the imported allocation while it waits alongside local work.
        self.scheduler.queues[RequestPhase.WAITING].append(request.request_id)
        return True

    def _rewind_split_prefill(self, request):
        """Rewind an in-flight chunk; the caller rebuilds its queue membership."""
        if request.phase not in (
            RequestPhase.PREFILLING_PRELUDE,
            RequestPhase.PREFILLING_RECURRENT,
        ):
            return
        frontier = request.num_prefilled_tokens
        request.num_preluded_tokens = frontier
        request.prefill_window_origin = frontier
        request.prefill_reclaimable_tokens = frontier
        request.prefill_depth_ranges[:] = Request.prefill_boundaries(
            len(request.prompt_token_ids),
            self.model.config.total_ut_steps,
            start=frontier,
        )
        request.phase = RequestPhase.PREFILLING_PRELUDE

    def _apply_update(self, update) -> list[RequestOutput]:
        for scheduled_request, phase, progress in update.prefill:
            self._notify_connector_prefill(scheduled_request, phase, progress)
        if update.exited:
            self.model_runner.finalize_many(update.exited)
        if self.speculative_runner is not None:
            self.speculative_runner.stats.committed_tokens += update.committed_tokens
            self.speculative_runner.stats.accepted_tokens += update.accepted_tokens
        for request in update.finished:
            self._finish(request, request.finish_reason)
        return [RequestOutput.from_request(request) for request in update.outputs]

    def _update(
        self, batch, result: ModelRunnerOutput | None, *, delivered: bool = False
    ) -> list[RequestOutput]:
        return self._apply_update(
            self.scheduler.update_from_output(
                batch,
                result,
                request_events=self.model_runner.events,
                delivered=delivered,
            )
        )

    def _finish(self, request, reason: FinishReason):
        if self.async_speculative:
            # Logical finish is immediate; allocation ownership survives until
            # the final submitted round has stopped accessing the request.
            request.phase = RequestPhase.FINISHED
            request.finish_reason = reason
            for queue in self.scheduler.queues.values():
                while request.request_id in queue:
                    queue.remove(request.request_id)
            self._retiring_speculative[request.request_id] = request
            return
        if self.speculative_runner is not None:
            self.speculative_runner.release(request)
        self.model_runner.release(request.request_id)
        self.cache_manager.poll_prefixes()
        self._pending_exit_signals.pop(request.request_id, None)
        self.scheduler.finish(request, reason)

    def _retire_speculative(self, *, wait=False, request_id=None):
        runner = cast("AsyncSpeculativeRunner", self.speculative_runner)
        for rid, request in list(self._retiring_speculative.items()):
            if request_id is not None and rid != request_id:
                continue
            if not wait and not runner.request_ready(request):
                continue
            runner.release(request)
            self.model_runner.release(rid)
            self.cache_manager.poll_prefixes()
            self.scheduler.finish(request, request.finish_reason)
            del self._retiring_speculative[rid]

    def _collect_speculative(self):
        assert self.speculative_config is not None
        ticket = self._pending_speculative[0]
        runner = cast("AsyncSpeculativeRunner", self.speculative_runner)
        if not ticket.ready():
            runner.readback_waits += 1
        result = ticket.collect()
        for scheduled_request, state, round_id in zip(
            ticket.batch.requests, ticket.states, ticket.rounds
        ):
            request = scheduled_request.request
            if (
                self.scheduler.requests.get(request.request_id) is not request
                or request.phase == RequestPhase.FINISHED
            ):
                runner.discarded_rounds += 1
                continue
            if state.delivered != round_id:
                raise RuntimeError("out-of-order speculative round delivery")
            state.delivered += 1
        outputs = self._update(ticket.batch, result)
        ticket.retire()
        self._pending_speculative.pop(0)
        self._retire_speculative()
        return outputs

    def _step_async_speculative(self):
        self._retire_speculative()
        outputs = []
        # Waiting admissions need a quiescent point before the scheduler can
        # select a victim. Victim selection remains entirely in Scheduler.
        if (
            self.pipeline_speculative
            and self.scheduler.config.enable_preemption
            and self.scheduler.queues[RequestPhase.WAITING]
        ):
            while self._pending_speculative:
                outputs.extend(self._collect_speculative())
        # Submit ahead before collecting, even when a small round already
        # completed. Two leased buffers provide a hard global queue bound.
        if len(self._pending_speculative) == 2:
            outputs.extend(self._collect_speculative())
        batch = self._schedule()
        self.last_schedule = batch
        if batch is not None:
            if batch.phase == RequestPhase.SPECULATIVE:
                ticket = self.executor.submit(batch)
                self._pending_speculative.append(ticket)
                self._apply_update(self.scheduler.update_after_submission(batch))
            else:
                # Bootstrap uses the same submit contract; only actual host
                # results (the first sampled token) require collection here.
                ticket = self.executor.submit(batch)
                result = ticket.collect() if ticket.has_result else None
                outputs.extend(self._update(batch, result))
        elif self._pending_speculative:
            outputs.extend(self._collect_speculative())
        elif self._retiring_speculative:
            self._retire_speculative(wait=True)
        elif self.scheduler.has_unfinished_requests and not self._waiting_for_connector():
            raise RuntimeError("speculative scheduler made no progress")
        return outputs

    def _deliver_coda(self, ticket):
        return self._update(ticket.batch, ticket.collect(), delivered=True)

    def _collect_coda(self, wait=False):
        outputs = []
        pending = []
        for ticket in self._pending_coda:
            if ticket.ready() or wait:
                outputs.extend(self._deliver_coda(ticket))
                wait = False
            else:
                pending.append(ticket)
        self._pending_coda = pending
        return outputs

    def _step_async(self):
        # Hold readback buffers until their DMA completes, even for discarded scores.
        self._inflight = [t for t in self._inflight if not t.ready()]
        outputs = self._collect_coda()
        batch = self._schedule(
            # If the preceding core is still running, boundary work can overlap
            # the independent next core. Once it has completed, refill first:
            # splitting on a briefly pending coda readback fragments the batch.
            prefer_recurrent=self._overlap_boundary
            and any(
                t.batch.phase == RequestPhase.RECURRENT and not t.ready() for t in self._inflight
            )
        )
        self._overlap_boundary = False
        self.last_schedule = batch
        if batch is None:
            # PD imports wait for external KV completion; yield to the IPC loop.
            if self._pending_coda:
                outputs.extend(self._collect_coda(wait=True))
            elif not self._waiting_for_connector():
                raise RuntimeError("scheduler made no progress")
            return outputs
        if batch.phase == RequestPhase.CODA:
            # Bound speculation to one output per request. Its next prelude and
            # core may run before delivery (including a possible EOS), but never
            # sample another token until that output's stop decision is known.
            while any(i.request.num_output_placeholders for i in batch.requests):
                outputs.extend(self._collect_coda(wait=True))
            batch = replace(
                batch,
                requests=[
                    i
                    for i in batch.requests
                    if self.scheduler.requests.get(i.request.request_id) is i.request
                ],
            )
            self.last_schedule = batch
            if not batch.requests:
                return outputs
        ticket = self.executor.submit(batch)
        self._inflight.append(ticket)
        if batch.phase == RequestPhase.CODA:
            self._pending_coda.append(ticket)
            runner = self.model_runner
            self._overlap_boundary = (
                runner.boundary_stream is not None
                and runner.boundary_stream is not runner.core_stream
            )
        signals = {}
        for row, scheduled_request in enumerate(batch.requests):
            phase = batch.phase_for(scheduled_request)
            request = scheduled_request.request
            if phase == RequestPhase.PRELUDE:
                self._pending_exit_signals.pop(request.request_id, None)
            elif phase == RequestPhase.RECURRENT:
                # Submission is already enqueued. Only now may the preceding score
                # block CPU delivery; max-depth/trace decisions require no readback.
                previous = self._pending_exit_signals.pop(request.request_id, None)
                if previous is None or not self.scheduler.needs_exit_signal(request):
                    continue
                old_ticket, old_row = previous.ticket, previous.row
                if old_ticket.batch.requests[old_row].request is not request:
                    continue
                signals[row] = ExitSignal(
                    request=request,
                    position=old_ticket.positions[old_row],
                    depth=old_ticket.depths[old_row] + 1,
                    score=old_ticket.collect().gate_scores[old_row],
                )
        update = self.scheduler.update_after_submission(
            batch,
            completion=ticket.event,
            request_events=self.model_runner.events,
            signals=signals,
        )
        for row in update.next_prelude:
            batch.requests[row].request.input_token_tensor = ticket.device_values[row]
        for row in update.retain_signal:
            self._pending_exit_signals[batch.requests[row].request.request_id] = SignalHandle(
                ticket, row
            )
        outputs.extend(self._apply_update(update))
        return outputs
