from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch

from vllm_rlt.attention import BackendCapabilities, create_backend
from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import plan_cache
from vllm_rlt.core.scheduler import Scheduler
from vllm_rlt.engine.preemption import PreemptionManager
from vllm_rlt.profiling import Profiler
from vllm_rlt.request import FinishReason, Request, RequestOutput, Stage
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.loopcd import reserved_bytes, validate
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.speculative import SpeculativeRunner

if TYPE_CHECKING:
    from vllm_rlt.worker.async_speculative import AsyncSpeculativeRunner


class LLMEngine:
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
    ):
        self.model = model
        self.profiling = Profiler(next(model.parameters()).device)
        cache_config = cache_config or CacheConfig()
        scheduler_config = scheduler_config or SchedulerConfig()
        parameter = next(model.parameters())
        config = model.config
        self.exit_config = exit_config or ExitConfig()
        self.execution_config = execution_config or ExecutionConfig()
        self.prefill_depth = self.execution_config.prefill_depth or config.total_ut_steps
        if self.prefill_depth > config.total_ut_steps:
            raise ValueError("prefill_depth exceeds the model's supported depth")
        if self.execution_config.prefill_depth is not None and (
            cache_config.layout != "last_exited"
            or cache_config.enable_prefix_caching
            or self.execution_config.async_scheduling
            or speculative_config is not None
        ):
            raise ValueError(
                "prefill_depth override requires synchronous last_exited without prefix/speculation"
            )
        self.speculative_config = speculative_config
        self.async_speculative = bool(speculative_config and self.execution_config.async_scheduling)
        if speculative_config is not None:
            if cache_config.layout != "last_exited":
                raise ValueError("speculative decoding requires last_exited KV")
            if speculative_config.target_loops != config.total_ut_steps:
                raise ValueError("speculative target_loops must equal the model full depth")
            if self.exit_config.mode != "ouro":
                raise ValueError("speculative decoding requires fixed-depth ouro exit mode")
            if self.async_speculative and self.execution_config.cuda_graphs:
                raise ValueError(
                    "speculative decoding requires eager execution without CUDA graphs"
                )
            if self.async_speculative and (
                parameter.device.type != "cuda" or attention_backend != "triton"
            ):
                raise ValueError("async speculative decoding requires CUDA and Triton attention")
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
                # Two in-flight rounds, plus device and pinned page-table banks.
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
            ),
        )
        num_blocks, self.memory_plan = plan_cache(
            model,
            cache_config,
            scheduler_config,
            self.execution_config,
            attention.info["selected_backend"],
        )
        self.memory_plan["loopcd_reserve_bytes"] = reserved_bytes(
            config, scheduler_config, self.execution_config, parameter.element_size()
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
        )
        self.scheduler = Scheduler(scheduler_config, self.cache_manager, speculative_config)
        self.speculative_runner = (
            SpeculativeRunner(model, self.cache_manager, speculative_config, self.execution_config)
            if speculative_config is not None
            else None
        )
        self.model_runner = ModelRunner(
            model,
            self.cache_manager,
            exit_config=self.exit_config,
            execution_config=self.execution_config,
            scheduler_config=scheduler_config,
        )
        if self.async_speculative:
            from vllm_rlt.worker.async_speculative import AsyncSpeculativeRunner

            self.speculative_runner = AsyncSpeculativeRunner(
                model,
                self.cache_manager,
                speculative_config,
                scheduler_config,
                multi_stream=self.execution_config.multi_stream,
            )
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
            self.scheduler.preempt_callback = self.preemption.preempt
            self.scheduler.resume_callback = self.preemption.resume

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        *,
        trace_id: str | None = None,
    ):
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
        validate(
            self.model,
            params,
            self.execution_config,
            self.cache_manager,
            self.scheduler.config,
            self.exit_config,
            self.speculative_config,
        )
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
                or trace[0] != self.prefill_depth
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
            Request(request_id, list(prompt_token_ids), params, exit_trace=trace)
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
            self._retire_speculative(wait=True)
            for ticket in self._pending_speculative:
                ticket.collect()
                ticket.retire()
            self._pending_speculative.clear()

    def abort_request(self, request_id: str) -> RequestOutput:
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
                # A submission can fail before recording an event. Drain owned
                # streams before invalidating requests or recycling any memory.
                self.model_runner.synchronize()
                if self.async_speculative:
                    cast("AsyncSpeculativeRunner", self.speculative_runner).synchronize()
                for rid in list(self.scheduler.requests):
                    self.abort_request(rid)
                if self.async_speculative:
                    self._retire_speculative(wait=True)
                    for ticket in self._pending_speculative:
                        ticket.collect()
                        ticket.retire()
                    self._pending_speculative.clear()
                    # Also release a bank whose submission failed before it
                    # could return a ticket. Both streams were drained above.
                    for bank in cast("AsyncSpeculativeRunner", self.speculative_runner).banks:
                        bank.keepalive.clear()
                        bank.leased = False
                self._pending_exit_signals.clear()
                self._pending_coda.clear()
                self._inflight.clear()
                raise
        batch = self.scheduler.schedule()
        self.last_schedule = batch
        if batch is None:
            # PD imports wait for external KV completion; yield to the IPC loop.
            if any(r.stage != Stage.RECEIVING for r in self.scheduler.requests.values()):
                raise RuntimeError("scheduler made no progress")
            return []
        try:
            if batch.stage == Stage.SPECULATIVE:
                result = self.speculative_runner.execute(batch)
                return self._update_speculative(batch, result)
            result = self.model_runner.execute(batch)
            return self._update(batch, result)
        except Exception:
            # A failed execution may have partially written KV; invalidate the affected requests.
            for item in batch.items:
                if item.request.request_id in self.scheduler.requests:
                    self.abort_request(item.request.request_id)
            raise

    def _update_speculative(self, batch, results):
        outputs = []
        for item, result in zip(batch.items, results):
            request = item.request
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
                self.cache_manager.truncate_suffix(request.request_id, item.token_start + emitted)
                request.loops_done = 0
                self.scheduler.enqueue(request, Stage.SPECULATIVE)
            outputs.append(RequestOutput.from_request(request))
        return outputs

    def _update(self, batch, result) -> list[RequestOutput]:
        outputs = []
        for index, item in enumerate(batch.items):
            request = item.request
            # Ignore results for a request that is no longer registered: it may
            # have been cancelled, or its ID may now belong to a new Request.
            # Compare object identity so an old result cannot update the new request.
            if self.scheduler.requests.get(request.request_id) is not request:
                continue
            params = request.sampling_params
            if batch.stage == Stage.PREFILL:
                request.num_prefilled_tokens += item.token_count
                event = self.model_runner.events.get(request.request_id)
                self.cache_manager.publish_prefix(
                    request.request_id,
                    request.prompt_token_ids,
                    request.num_prefilled_tokens,
                    event,
                )
                if request.num_prefilled_tokens == len(request.prompt_token_ids):
                    request.loops_done = self.prefill_depth
                    self.scheduler.enqueue(request, Stage.CODA)
                else:
                    self.scheduler.enqueue(request, Stage.PREFILL)
            elif batch.stage == Stage.PRELUDE:
                request.loops_done = 0
                request.remaining_probability = 1.0
                request.pending_exit_depth = None
                self._pending_exit_signals.pop(request.request_id, None)
                self.scheduler.enqueue(request, Stage.RECURRENT)
            elif batch.stage == Stage.RECURRENT:
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
                    self.scheduler.enqueue(request, Stage.CODA)
                else:
                    self.scheduler.enqueue(request, Stage.RECURRENT)
            elif batch.stage == Stage.CODA:
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
                        request, Stage.SPECULATIVE if self.speculative_config else Stage.PRELUDE
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
            request.stage = Stage.FINISHED
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
        ticket = self._pending_speculative.pop(0)
        runner = cast("AsyncSpeculativeRunner", self.speculative_runner)
        if not ticket.ready():
            runner.readback_waits += 1
        results, frontiers = ticket.collect()
        outputs = []
        for item, state, round_id, result, frontier in zip(
            ticket.batch.items,
            ticket.states,
            ticket.rounds,
            results,
            frontiers,
        ):
            request = item.request
            if (
                self.scheduler.requests.get(request.request_id) is not request
                or request.stage == Stage.FINISHED
            ):
                runner.discarded_rounds += 1
                continue
            if state.delivered != round_id:
                raise RuntimeError("out-of-order speculative round delivery")
            state.delivered += 1
            request.num_output_placeholders -= item.token_count
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
            if request.stage != Stage.FINISHED:
                queue = self.scheduler.queues[Stage.SPECULATIVE]
                if request.request_id not in queue:
                    self.scheduler.enqueue(request, Stage.SPECULATIVE)
            outputs.append(RequestOutput.from_request(request))
        ticket.retire()
        self._retire_speculative()
        return outputs

    def _step_async_speculative(self):
        runner = cast("AsyncSpeculativeRunner", self.speculative_runner)
        self._retire_speculative()
        outputs = []
        # Submit ahead before collecting, even when a small round already
        # completed. Two leased banks provide a hard global queue bound.
        if len(self._pending_speculative) == 2:
            outputs.extend(self._collect_speculative())
        batch = self.scheduler.schedule()
        self.last_schedule = batch
        if batch is not None:
            if batch.stage == Stage.SPECULATIVE:
                ticket = runner.submit_round(batch)
                self._pending_speculative.append(ticket)
                for item in batch.items:
                    request = item.request
                    request.num_output_placeholders += item.token_count
                    request.num_speculative_rounds += 1
                    self.scheduler.enqueue(request, Stage.SPECULATIVE)
            else:
                # Prefill/bootstrap retain native semantics. All model compute
                # shares the speculative stream; the first coda is a CPU boundary.
                with torch.cuda.stream(runner.compute_stream):
                    result = self.model_runner.execute(batch)
                    event = torch.cuda.Event()
                    event.record(runner.compute_stream)
                    for item in batch.items:
                        self.model_runner.events[item.request.request_id] = event
                    outputs.extend(self._update(batch, result))
        elif self._pending_speculative:
            outputs.extend(self._collect_speculative())
        elif self._retiring_speculative:
            self._retire_speculative(wait=True)
        elif self.scheduler.has_unfinished_requests:
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
        for index, item in enumerate(ticket.batch.items):
            request = item.request
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
        batch = self.scheduler.schedule(
            # If the preceding core is still running, boundary work can overlap
            # the independent next core. Once it has completed, refill first:
            # splitting on a briefly pending coda readback fragments the batch.
            prefer_recurrent=self._overlap_boundary
            and any(t.batch.stage == Stage.RECURRENT and not t.ready() for t in self._inflight)
        )
        self._overlap_boundary = False
        self.last_schedule = batch
        if batch is None:
            # PD imports wait for external KV completion; yield to the IPC loop.
            if self._pending_coda:
                outputs.extend(self._collect_coda(wait=True))
            elif any(r.stage != Stage.RECEIVING for r in self.scheduler.requests.values()):
                raise RuntimeError("scheduler made no progress")
            return outputs
        if batch.stage == Stage.CODA:
            # Bound speculation to one output per request. Its next prelude and
            # core may run before delivery (including a possible EOS), but never
            # sample another token until that output's stop decision is known.
            while any(i.request.num_output_placeholders for i in batch.items):
                outputs.extend(self._collect_coda(wait=True))
            batch = replace(
                batch,
                items=[
                    i
                    for i in batch.items
                    if self.scheduler.requests.get(i.request.request_id) is i.request
                ],
            )
            self.last_schedule = batch
            if not batch.items:
                return outputs
        ticket = self.model_runner.submit(batch)
        self._inflight.append(ticket)
        if batch.stage == Stage.CODA:
            self._pending_coda.append(ticket)
            runner = self.model_runner
            self._overlap_boundary = (
                runner.boundary_stream is not None
                and runner.boundary_stream is not runner.core_stream
            )
            for index, item in enumerate(batch.items):
                request = item.request
                request.num_output_placeholders += 1
                if request.num_scheduled_outputs < request.sampling_params.max_tokens:
                    request.input_token_tensor = ticket.device_values[index]
                    self.scheduler.enqueue(request, Stage.PRELUDE)
        elif batch.stage != Stage.RECURRENT:
            self._update(batch, None)
        else:
            # Submit r FIRST. While the GPU runs r, consume r-1's signal to
            # determine whether this token may enter r+1. No speculative extra loop.
            exited = []
            for index, item in enumerate(batch.items):
                request = item.request
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
                    self.scheduler.enqueue(request, Stage.CODA)
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
                    self.scheduler.enqueue(request, Stage.RECURRENT)
            self.model_runner.finalize_many(exited)
        return outputs
