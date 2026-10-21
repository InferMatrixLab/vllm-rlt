from dataclasses import replace
from typing import TYPE_CHECKING, cast

from vllm_rlt.attention import BackendCapabilities, create_backend
from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import plan_cache
from vllm_rlt.core.scheduler import PrefillProgress, Scheduler
from vllm_rlt.engine.connector import StageConnector
from vllm_rlt.engine.preemption import PreemptionManager
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.profiling import Profiler
from vllm_rlt.request import FinishReason, Request, RequestOutput, RequestPhase
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.executor import Executor
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.sampler import RngRegistry
from vllm_rlt.worker.speculative import SpeculativeRunner

if TYPE_CHECKING:
    from vllm_rlt.worker.async_speculative import AsyncSpeculativeRunner


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
        self.execution_config = execution_config or ExecutionConfig()
        if getattr(model, "requires_boundary_kv", False):
            if not self.execution_config.atomic_prefill:
                raise ValueError("Huginn requires atomic prefill")
            if self.execution_config.async_scheduling or self.execution_config.prefill_uva:
                raise ValueError("Huginn requires synchronous scheduling without prefill UVA")
            if speculative_config is not None:
                raise ValueError("Huginn speculative decoding is not yet supported")
        self.speculative_config = speculative_config
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
            if scheduler_config.enable_preemption or scheduler_config.mode != "refill":
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
        if not self.execution_config.atomic_prefill:
            if cache_config.layout != "last_exited":
                raise ValueError("split prefill requires last_exited KV planes")
        attention = create_backend(
            attention_backend,
            parameter.device,
            parameter.dtype,
            config.head_dim,
            cache_config.block_size,
            required_capabilities=BackendCapabilities(
                packed_prefill=self.execution_config.prefill_uva,
                cuda_graphs=self.execution_config.cuda_graphs,
                async_scheduling=(
                    parameter.device.type == "cuda" and self.execution_config.async_scheduling
                ),
                device_context_lengths=self.async_speculative,
            ),
        )
        num_blocks, self.memory_plan = plan_cache(
            model,
            cache_config,
            scheduler_config,
            self.execution_config,
            attention.info["selected_backend"],
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
            atomic_prefill=self.execution_config.atomic_prefill,
            produces_tokens=self.stage_config.produces_tokens,
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
        # Unconsumed exit scores: request ID -> (submission, row, token position, depth).
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
        if self.async_speculative and params.temperature != 0:
            raise ValueError("async speculative decoding currently requires greedy sampling")
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

    def _update_speculative(self, batch, results):
        outputs = []
        for scheduled_request, result in zip(batch.requests, results):
            request = scheduled_request.request
            params = request.sampling_params
            eos = self.model.config.eos_token_id
            eos_ids = eos if isinstance(eos, (tuple, list)) else [eos]
            emitted = 0
            reason = None
            for token in result.token_ids:
                request.generated_token_ids.append(token)
                request.exit_depths.append(self.speculative_config.target_loops)
                emitted += 1
                if token in eos_ids and not params.ignore_eos:
                    reason = FinishReason.STOP
                    break
                if len(request.generated_token_ids) >= params.max_tokens:
                    reason = FinishReason.LENGTH
                    break
            self.speculative_runner.stats.committed_tokens += emitted
            self.speculative_runner.stats.accepted_tokens += min(result.accepted_count, emitted)
            if reason is not None:
                self._finish(request, reason)
            else:
                # The last emitted token is correction/bonus, not yet forwarded.
                self.cache_manager.truncate_suffix(
                    request.request_id, scheduled_request.token_start + emitted
                )
                request.loops_done = 0
                self.scheduler.enqueue(request, RequestPhase.SPECULATIVE)
            outputs.append(RequestOutput.from_request(request))
        return outputs

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

    def _prompt_destination(self, request):
        exporting = self.connector is not None and self.connector.exports(request)
        return RequestPhase.TRANSFERRING if exporting else RequestPhase.CODA

    def _apply_prefill_progress(self, batch, scheduled_request):
        """One prompt frontier update for local and remotely consumed outputs."""
        request = scheduled_request.request
        phase = batch.phase_for(scheduled_request)
        event = self.model_runner.events.get(request.request_id)
        if phase == RequestPhase.PREFILL:
            request.num_prefilled_tokens += scheduled_request.token_count
            final = request.num_prefilled_tokens == len(request.prompt_token_ids)
            self.cache_manager.publish_prefix(
                request.request_id, request.prompt_token_ids, request.num_prefilled_tokens, event
            )
            if final:
                request.loops_done = self.model.config.total_ut_steps
            self.scheduler.enqueue(
                request, self._prompt_destination(request) if final else RequestPhase.PREFILL
            )
            progress = PrefillProgress(True, final)
            start = scheduled_request.token_start
        else:
            progress = self._apply_split_prefill_progress(
                request, phase, scheduled_request, event=event
            )
            start = scheduled_request.chunk_start
        if self.connector is not None and self.connector.exports(request):
            self.connector.prefill_submitted(
                request,
                phase=phase,
                start=start,
                progress=progress,
                event=event,
                stream=self.model_runner.core_stream,
            )
        return progress

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
        request.prefill_chunk_start = hit
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
        request.loops_done = self.model.config.total_ut_steps
        request.hidden_state = hidden
        # Keep RECEIVING until the scheduler grants a compute slot. The lease
        # protects the imported allocation while it waits alongside local work.
        self.scheduler.queues[RequestPhase.WAITING].append(request.request_id)
        return True

    def _apply_split_prefill_progress(
        self, request, phase, scheduled_request, *, event=None
    ) -> PrefillProgress:
        """Advance host-side logical progress for a submitted prefill request.

        The device work may still be pending. The event orders prefix publication
        and consumers with that work; this transition does not wait for it.
        """
        total_depths = self.model.config.total_ut_steps
        start = scheduled_request.token_start
        end = start + scheduled_request.token_count

        if phase == RequestPhase.PREFILLING_PRELUDE:
            request.prefill_chunk_start = start
            request.num_preluded_tokens = end
            self.scheduler.enqueue(request, RequestPhase.PREFILLING_RECURRENT)
            return PrefillProgress(False, False)

        # Assertion: stage == RequestPhase.PREFILLING_RECURRENT
        depth = scheduled_request.depth
        boundaries = request.prefill_depth_ranges
        boundaries[depth][0] = end
        if depth + 1 < total_depths:
            boundaries[depth + 1][1] = end
        else:
            request.num_prefilled_tokens = end

        chunk_complete = end == request.num_preluded_tokens
        if depth + 1 < total_depths or not chunk_complete:
            self.scheduler.enqueue(request, RequestPhase.PREFILLING_RECURRENT)
            return PrefillProgress(False, False)

        self.cache_manager.publish_prefix(
            request.request_id,
            request.prompt_token_ids,
            request.num_prefilled_tokens,
            event,
        )
        request.prefill_chunk_start = end
        prompt_complete = end == len(request.prompt_token_ids)
        if prompt_complete:
            request.loops_done = total_depths
            next_stage = self._prompt_destination(request)
        else:
            next_stage = RequestPhase.PREFILLING_PRELUDE
        self.scheduler.enqueue(request, next_stage)
        return PrefillProgress(True, prompt_complete)

    def _rewind_split_prefill(self, request):
        """Rewind an in-flight chunk; the caller rebuilds its queue membership."""
        if self.execution_config.atomic_prefill or request.phase not in (
            RequestPhase.PREFILLING_PRELUDE,
            RequestPhase.PREFILLING_RECURRENT,
        ):
            return
        frontier = request.num_prefilled_tokens
        request.num_preluded_tokens = frontier
        request.prefill_chunk_start = frontier
        request.prefill_depth_ranges[:] = Request.prefill_boundaries(
            len(request.prompt_token_ids),
            self.model.config.total_ut_steps,
            start=frontier,
        )
        request.phase = RequestPhase.PREFILLING_PRELUDE

    def _update(self, batch, result) -> list[RequestOutput]:
        if batch.is_speculative:
            return self._update_speculative(batch, result)
        outputs = []
        for index, scheduled_request in enumerate(batch.requests):
            request = scheduled_request.request
            # Ignore results for a request that is no longer registered: it may
            # have been cancelled, or its ID may now belong to a new Request.
            # Compare object identity so an old result cannot update the new request.
            if self.scheduler.requests.get(request.request_id) is not request:
                continue
            phase = batch.phase_for(scheduled_request)
            params = request.sampling_params
            if phase in (
                RequestPhase.PREFILL,
                RequestPhase.PREFILLING_PRELUDE,
                RequestPhase.PREFILLING_RECURRENT,
            ):
                self._apply_prefill_progress(batch, scheduled_request)
            elif phase == RequestPhase.PRELUDE:
                request.loops_done = 0
                request.remaining_probability = 1.0
                request.pending_exit_depth = None
                self._pending_exit_signals.pop(request.request_id, None)
                self.scheduler.enqueue(request, RequestPhase.RECURRENT)
            elif phase == RequestPhase.RECURRENT:
                request.loops_done += 1
                if self.exit_config.mode == "trace":
                    should_exit = self._trace_exit(request)
                elif self.exit_config.mode == "ouro":
                    request.remaining_probability *= 1.0 - result[index]
                    should_exit = self._should_exit(request)
                else:
                    should_exit = self._delayed_exit(request, result[index])
                if should_exit:
                    self.model_runner.finalize(request)
                    self.scheduler.enqueue(request, RequestPhase.CODA)
                else:
                    self.scheduler.enqueue(request, RequestPhase.RECURRENT)
            elif phase == RequestPhase.CODA:
                token_id = result[index]
                request.generated_token_ids.append(token_id)
                request.exit_depths.append(request.loops_done)
                eos = self.model.config.eos_token_id
                eos_ids = eos if isinstance(eos, (tuple, list)) else [eos]
                if token_id in eos_ids and not params.ignore_eos:
                    self._finish(request, FinishReason.STOP)
                elif len(request.generated_token_ids) >= params.max_tokens:
                    self._finish(request, FinishReason.LENGTH)
                else:
                    self.scheduler.enqueue(
                        request,
                        RequestPhase.SPECULATIVE
                        if self.speculative_config
                        else RequestPhase.PRELUDE,
                    )
                outputs.append(RequestOutput.from_request(request))
        return outputs

    def _should_exit(self, request: Request) -> bool:
        """Evaluate the policy after recording the completed loop's actual hazard."""
        params = request.sampling_params
        max_loops = params.max_loops or self.model.config.total_ut_steps
        reached_threshold = (
            params.exit_threshold < 1.0
            and request.loops_done >= params.min_loops
            and 1.0 - request.remaining_probability >= params.exit_threshold
        )
        return request.loops_done >= max_loops or reached_threshold

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
        results, frontiers = ticket.collect()
        outputs = []
        for scheduled_request, state, round_id, result, frontier in zip(
            ticket.batch.requests,
            ticket.states,
            ticket.rounds,
            results,
            frontiers,
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
            request.num_output_placeholders -= scheduled_request.token_count
            request.num_speculative_rounds -= 1
            if request.num_output_placeholders < 0 or request.num_speculative_rounds < 0:
                raise RuntimeError("invalid speculative reservation count")
            expected = len(request.prompt_token_ids) + len(request.generated_token_ids) - 1
            if frontier != expected + len(result.token_ids):
                raise RuntimeError("speculative GPU frontier disagrees with committed prefix")
            params = request.sampling_params
            eos = self.model.config.eos_token_id
            eos_ids = eos if isinstance(eos, (tuple, list)) else [eos]
            emitted = 0
            for token in result.token_ids:
                request.generated_token_ids.append(token)
                request.exit_depths.append(self.speculative_config.target_loops)
                emitted += 1
                if token in eos_ids and not params.ignore_eos:
                    self._finish(request, FinishReason.STOP)
                    break
                if len(request.generated_token_ids) >= params.max_tokens:
                    self._finish(request, FinishReason.LENGTH)
                    break
            runner.stats.committed_tokens += emitted
            runner.stats.accepted_tokens += min(result.accepted_count, emitted)
            # In particular, do NOT truncate KV here: the GPU may already be
            # writing the following round at a later actual position.
            if request.phase != RequestPhase.FINISHED:
                queue = self.scheduler.queues[RequestPhase.SPECULATIVE]
                if request.request_id not in queue:
                    self.scheduler.enqueue(request, RequestPhase.SPECULATIVE)
            outputs.append(RequestOutput.from_request(request))
        ticket.retire()
        self._pending_speculative.pop(0)
        self._retire_speculative()
        return outputs

    def _step_async_speculative(self):
        self._retire_speculative()
        outputs = []
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
                for scheduled_request in batch.requests:
                    request = scheduled_request.request
                    request.num_output_placeholders += scheduled_request.token_count
                    request.num_speculative_rounds += 1
                    self.scheduler.enqueue(request, RequestPhase.SPECULATIVE)
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

    def _trace_exit(self, request):
        target = request.exit_trace[request.num_scheduled_outputs]
        return request.loops_done >= target

    def _delayed_exit(self, request, score):
        params = request.sampling_params
        maximum = params.max_loops or self.model.config.total_ut_steps
        if request.loops_done >= maximum or request.pending_exit_depth == request.loops_done:
            return True
        target = request.loops_done + 1
        if self._delayed_signal(request, score, request.loops_done):
            request.pending_exit_depth = target
        return False

    def _delayed_signal(self, request, score, signal_depth):
        """Consume each signal once; Ouro's minimum applies to the trigger round."""
        params = request.sampling_params
        if self.exit_config.mode == "ouro_delayed":
            request.remaining_probability *= 1.0 - score
            score = 1.0 - request.remaining_probability
            eligible_depth = signal_depth
        else:
            eligible_depth = signal_depth + 1
        return (
            params.exit_threshold < 1
            and eligible_depth >= params.min_loops
            and score >= params.exit_threshold
        )

    def _deliver_coda(self, ticket):
        outputs = []
        result = ticket.collect()
        for index, scheduled_request in enumerate(ticket.batch.requests):
            request = scheduled_request.request
            if self.scheduler.requests.get(request.request_id) is not request:
                continue
            if ticket.output_indices[index] != len(request.generated_token_ids):
                raise RuntimeError("out-of-order coda delivery")
            if request.num_output_placeholders != 1:
                raise RuntimeError("invalid pending output count")
            request.num_output_placeholders -= 1
            token_id = result[index]
            request.generated_token_ids.append(token_id)
            # Current loops_done may already describe the NEXT token.
            request.exit_depths.append(ticket.depths[index])
            params = request.sampling_params
            eos = self.model.config.eos_token_id
            eos_ids = eos if isinstance(eos, (tuple, list)) else [eos]
            if token_id in eos_ids and not params.ignore_eos:
                self._finish(request, FinishReason.STOP)
            elif len(request.generated_token_ids) >= params.max_tokens:
                self._finish(request, FinishReason.LENGTH)
            outputs.append(RequestOutput.from_request(request))
        return outputs

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
            for index, scheduled_request in enumerate(batch.requests):
                request = scheduled_request.request
                request.num_output_placeholders += 1
                if request.num_scheduled_outputs < request.sampling_params.max_tokens:
                    request.input_token_tensor = ticket.device_values[index]
                    self.scheduler.enqueue(request, RequestPhase.PRELUDE)
        elif not any(batch.phase_for(i) == RequestPhase.RECURRENT for i in batch.requests):
            self._update(batch, None)
        else:
            # Submit r FIRST. While the GPU runs r, consume r-1's signal to
            # determine whether this token may enter r+1. No speculative extra loop.
            exited = []
            for index, scheduled_request in enumerate(batch.requests):
                if batch.phase_for(scheduled_request) != RequestPhase.RECURRENT:
                    self._update(replace(batch, requests=[scheduled_request]), None)
                    continue
                request = scheduled_request.request
                previous = self._pending_exit_signals.pop(request.request_id, None)
                request.loops_done += 1
                params = request.sampling_params
                maximum = params.max_loops or self.model.config.total_ut_steps
                should_exit = (
                    self._trace_exit(request)
                    if self.exit_config.mode == "trace"
                    else request.loops_done >= maximum
                )
                if previous is not None and not should_exit:
                    old_ticket, old_index, position, signal_depth = previous
                    if position != request.position or signal_depth != request.loops_done - 1:
                        raise RuntimeError("stale lookahead signal")
                    score = old_ticket.collect()[old_index]
                    should_exit = self._delayed_signal(request, score, signal_depth)
                if should_exit:
                    request.pending_exit_depth = request.loops_done
                    exited.append(request)
                    self.scheduler.enqueue(request, RequestPhase.CODA)
                else:
                    if (
                        self.exit_config.mode in ("ouro_delayed", "random_lookahead")
                        and params.exit_threshold < 1
                    ):
                        self._pending_exit_signals[request.request_id] = (
                            ticket,
                            index,
                            request.position,
                            request.loops_done,
                        )
                    self.scheduler.enqueue(request, RequestPhase.RECURRENT)
            self.model_runner.finalize_many(exited)
        return outputs
