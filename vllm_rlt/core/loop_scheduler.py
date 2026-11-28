"""Schedule one loop operation using vLLM's request and KV interfaces."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.outputs import RequestError
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import CachedRequestData, IterationDetails, NewRequestData
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import RequestStatus

from vllm_rlt.config import (
    LoopRuntimeConfig,
    supports_cross_depth_decode,
    supports_prefill_pipeline,
)
from vllm_rlt.core.loop import LoopLayout, LoopOp, LoopSpan
from vllm_rlt.core.output import LoopModelRunnerOutput, LoopSchedulerOutput
from vllm_rlt.core.request import LoopRequest

logger = init_logger("vllm.rlt.loop_scheduler")


@dataclass(slots=True)
class _SpanProgress:
    start: int
    count: int
    depth_limit: int
    prefill: bool
    produces_token: bool
    adaptive: bool
    next_op: LoopOp | None
    next_depth: int = 0
    completed_depth: int = -1
    draft_coda: bool = False
    draft_token: int | None = None
    input_token: int | None = None
    tentative: bool = False
    awaiting_coda_at: int | None = None
    coda_tail: bool = False
    chunk_id: int = 0
    ready_since: int = 0
    skips: int = 0


@dataclass(slots=True)
class _RequestState:
    request: LoopRequest
    spans: dict[int, _SpanProgress] = field(default_factory=dict)
    known_to_worker: bool = False
    in_flight: int = 0
    newly_allocated: tuple[list[int], ...] | None = None
    next_chunk_id: int = 0
    entry_ready_since: int = 0
    entry_skips: int = 0


@dataclass(slots=True)
class _ReadyWork:
    state: _RequestState
    progress: _SpanProgress | None
    op: LoopOp
    depth: int
    rows: int
    prefill: bool
    rank: tuple[int, int]
    since: int


@dataclass(slots=True)
class _Batch:
    op: LoopOp
    work: list[_ReadyWork]
    full: bool
    entry_candidates: list[_ReadyWork] | None = None


@dataclass(slots=True)
class _Submission:
    state: _RequestState
    work: _SpanProgress
    span: LoopSpan
    advanced: bool


class LoopScheduler(SchedulerInterface):
    connector = None
    ec_connector = None

    def __init__(
        self,
        vllm_config,
        kv_cache_config,
        structured_output_manager,
        block_size,
        hash_block_size,
        mm_registry=MULTIMODAL_REGISTRY,
        include_finished_set=False,
        log_stats=False,
    ):
        config = vllm_config.scheduler_config
        self.layout = LoopLayout.from_model_config(vllm_config.model_config)
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_active_seqs = getattr(config, "max_num_active_seqs", None) or self.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_scheduled_tokens = min(
            getattr(config, "max_num_scheduled_tokens", None) or self.max_num_batched_tokens,
            self.max_num_batched_tokens,
        )
        options = LoopRuntimeConfig.from_additional_config(vllm_config.additional_config)
        self.prefill_chunk_size = min(
            self.max_num_scheduled_tokens,
            options.prefill_chunk_size or self.max_num_scheduled_tokens,
            getattr(config, "long_prefill_token_threshold", 0) or self.max_num_scheduled_tokens,
        )
        self.prefill_window = options.prefill_window or (
            2 * self.max_num_scheduled_tokens * self.layout.num_loops
        )
        if self.prefill_window < self.prefill_chunk_size * self.layout.num_loops:
            raise ValueError("RLT prefill window must fit a full chunk through all CORE rounds")
        self._prefill_window_blocked = False
        self._priority_policy = getattr(config, "policy", "fcfs") == "priority"
        self.self_speculation = options.self_speculation
        self.trace_steps = options.trace_steps
        self.decode_pipeline = options.decode_pipeline
        self.async_prefill_lookahead = options.async_prefill_lookahead
        ordinary_tokens = not (self.self_speculation or self.decode_pipeline)
        self.split_prefill_coda = (
            options.split_prefill_coda and ordinary_tokens and self.layout.has_coda
        )
        self._mixed_prefill_core = supports_prefill_pipeline(vllm_config, options)
        self._cross_depth_decode = supports_cross_depth_decode(vllm_config, options)
        self._unified_coda = options.unified_coda and self._mixed_prefill_core
        if self.decode_pipeline and (
            not self.self_speculation
            or not config.async_scheduling
            or getattr(vllm_config.model_config.hf_text_config, "loop_share_cache", False)
        ):
            raise ValueError("rlt_decode_pipeline requires async self-speculation and per-pass KV")
        if self.self_speculation and not self.layout.has_coda:
            raise ValueError("rlt self-speculation requires Inkling's Coda")
        if options.silu_and_mul and not self.layout.early_exit:
            raise ValueError("rlt_silu_and_mul requires an Ouro model")
        self.log_stats = log_stats
        self.kv = KVCacheManager(
            kv_cache_config,
            self.max_model_len,
            block_size,
            hash_block_size or block_size,
            enable_caching=vllm_config.cache_config.enable_prefix_caching,
            log_stats=log_stats,
        )
        self._states: dict[str, _RequestState] = {}
        self._schedule_step = 0
        self._in_flight: dict[int, tuple[_Submission, ...]] = {}
        self._finished: set[str] = set()
        self._discarded_states: dict[str, tuple[int, ...]] = {}
        self._finished_by_client: dict[int, set[str]] = defaultdict(set)
        self._rejected_by_client: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        self._include_finished_set = include_finished_set
        self._pause_state = PauseState.UNPAUSED

    def add_request(self, request: LoopRequest) -> None:
        try:
            self._validate_request(request)
        except VLLMValidationError as error:
            self._rejected_by_client[request.client_index].append(
                EngineCoreOutput(
                    request.request_id,
                    [],
                    finish_reason=FinishReason.ERROR,
                    error=RequestError(code="validation_error", message=str(error)),
                )
            )
            return
        self._states[request.request_id] = _RequestState(request)

    def _validate_request(self, request: LoopRequest) -> None:
        if request.request_id in self._states:
            raise VLLMValidationError(f"duplicate request ID {request.request_id}")
        if (
            request.sampling_params is None
            or request.mm_features
            or request.prompt_embeds is not None
        ):
            raise VLLMValidationError("rlt accepts text generation requests with prompt token IDs")
        if request.num_tokens < 1:
            raise VLLMValidationError("rlt requires a nonempty prompt")
        sampling = request.sampling_params
        if sampling.prompt_logprobs is not None or sampling.prompt_logprob_token_ids is not None:
            raise VLLMValidationError("prompt logprobs are not supported by the rlt scheduler")
        params = request.loop_params
        limits = (params.target_loops, params.min_loops, params.max_loops)
        if any(limit is not None and limit > self.layout.num_loops for limit in limits):
            raise VLLMValidationError(f"loop limits must be at most {self.layout.num_loops}")
        if (
            params.target_loops not in (None, self.layout.num_loops)
            and not self.layout.variable_depth
        ):
            raise VLLMValidationError("the checkpoint requires its configured fixed loop depth")
        if not self.layout.early_exit and any(
            value is not None
            for value in (params.exit_threshold, params.min_loops, params.max_loops)
        ):
            raise VLLMValidationError("the checkpoint has no adaptive exit gate")
        if request.use_structured_output:
            raise VLLMValidationError("structured output is not yet supported by the rlt scheduler")
        if self.self_speculation and request.sampling_params.temperature != 0:
            raise VLLMValidationError("rlt one-token self-speculation currently requires greedy")

    def schedule(self, throttle_prefills: bool = False) -> LoopSchedulerOutput:
        self._schedule_step += 1
        capacity = self._prefill_capacity()
        ready = self._ready_work(throttle_prefills, capacity)
        selected, selected_op = [], None
        failed = set()
        for batch in self._batches(ready, capacity):
            selected = self._reserve_batch(batch, capacity, failed)
            if selected:
                selected_op = batch.op
                self._remember_skips(ready, selected, failed)
                break

        output = (
            LoopSchedulerOutput.for_step(selected_op, (span for _, span in selected))
            if selected_op is not None
            else LoopSchedulerOutput.make_empty()
        )
        output.finished_req_ids, self._finished = self._finished, set()
        output.discarded_states, self._discarded_states = self._discarded_states, {}
        cached = output.scheduled_cached_reqs
        context_requests = 0
        context_tokens = 0
        submissions = []
        emitted_requests = set()
        for state, span in selected:
            request = state.request
            work = state.spans[span.token_start]
            if span.request_id not in emitted_requests:
                if not state.known_to_worker:
                    output.scheduled_new_reqs.append(
                        NewRequestData.from_request(
                            request,
                            self.kv.get_block_ids(span.request_id),
                            list(request.all_token_ids),
                        )
                    )
                    state.known_to_worker = True
                else:
                    self._append_cached_request(
                        cached,
                        state,
                        span.token_start if work.awaiting_coda_at is not None else None,
                    )
                state.newly_allocated = None
                emitted_requests.add(span.request_id)
            if span.prefill:
                context_requests += 1
                context_tokens += span.token_count
            state.in_flight += 1
            assert selected_op is not None
            work.ready_since = 0
            work.skips = 0
            advanced = self._on_submit(state, work, selected_op, span)
            submissions.append(_Submission(state, work, span, advanced))
        if submissions:
            self._in_flight[id(output)] = tuple(submissions)
        output.iteration_details = IterationDetails(
            num_prefill_requests=context_requests,
            num_prefill_tokens=context_tokens,
            num_decode_requests=len(selected) - context_requests,
            num_decode_tokens=output.total_num_scheduled_tokens - context_tokens,
        )
        return output

    def _ready_reserved_spans(self):
        if self._pause_state == PauseState.PAUSED_ALL:
            return []
        return [
            (state, work)
            for state in self._states.values()
            if state.request.status == RequestStatus.RUNNING
            and state.spans
            and not (self.decode_pipeline and state.in_flight)
            for work in self._ready_spans(state)
            if work is not None and work.next_op is not None
        ]

    def _prefill_capacity(self):
        outstanding = sum(
            work.count * (work.depth_limit - work.completed_depth - 1)
            for state in self._states.values()
            for work in state.spans.values()
            if work.prefill
        )
        loops = self.layout.num_loops
        available = (self.prefill_window - outstanding) // loops
        refill = min(self.max_num_scheduled_tokens, self.prefill_window // loops)
        if available <= 0:
            self._prefill_window_blocked = True
        elif available >= refill:
            self._prefill_window_blocked = False
        return max(available, 0)

    def _rank(self, state, work, op, prefill, since, skips):
        if skips >= 2:
            return 0, since
        if not prefill:
            tier = {LoopOp.CODA: 1, LoopOp.CORE: 3, LoopOp.PRELUDE: 4}[op]
        elif work is not None and work.produces_token and (
            op is LoopOp.CODA
            or not self.layout.has_coda and work.next_depth + 1 == work.depth_limit
        ):
            tier = 5
        else:
            tier = 6
        return tier, state.request.priority if self._priority_policy else 0

    def _ready_work(self, throttle_prefills, prefill_capacity):
        ready, entries = [], []
        for state in self._schedule_candidates(throttle_prefills):
            if self.decode_pipeline and state.in_flight:
                continue
            for work in self._ready_spans(state):
                if work is None:
                    entry = self._entry_work(state, prefill_capacity)
                    if entry is not None:
                        entries.append(entry)
                elif work.next_op is not None:
                    if not work.ready_since:
                        work.ready_since = self._schedule_step
                    ready.append(_ReadyWork(
                        state, work, work.next_op, work.next_depth, work.count, work.prefill,
                        self._rank(state, work, work.next_op, work.prefill,
                                   work.ready_since, work.skips), work.ready_since,
                    ))
        if self._prefill_window_blocked and ready:
            entries = [entry for entry in entries if not entry.prefill]
        ready.extend(entries)
        # FCFS is request arrival order, including across stages and CORE depths.
        ready.sort(key=lambda item: (
            item.rank, item.state.request.arrival_time,
            item.progress.chunk_id if item.progress is not None else item.state.next_chunk_id,
        ))
        return ready

    def _entry_work(self, state, prefill_capacity):
        request = state.request
        last = next(reversed(state.spans.values()), None)
        start = request.num_computed_tokens if last is None else last.start + last.count
        prefill = start < request.num_prompt_tokens or not state.known_to_worker
        available = request.num_tokens - start
        pipeline_entry = (
            not prefill and self.decode_pipeline and len(state.spans) == 1
            and start >= request.num_tokens
            and self._lookahead_token(state) is not None
        )
        if not prefill and (self._pending_coda(state) is not None or pipeline_entry):
            available = 1
        rows = min(available, self.max_num_scheduled_tokens)
        if prefill:
            rows = min(rows, self.prefill_chunk_size, prefill_capacity)
        if rows <= 0:
            return None
        if not state.entry_ready_since:
            state.entry_ready_since = self._schedule_step
        op = self.layout.first_op
        rank = self._rank(state, None, op, prefill, state.entry_ready_since, state.entry_skips)
        if pipeline_entry and rank[0] > 0:
            # An explicit decode pipeline prepares the draft successor before
            # validating it with the current token's deeper CORE.
            rank = 2, rank[1]
        return _ReadyWork(
            state, None, op, 0, rows, prefill, rank, state.entry_ready_since,
        )

    def _take_work(self, candidates, prefill_capacity, initial=()):
        chosen = list(initial)
        budget = self.max_num_scheduled_tokens - sum(item.rows for item in chosen)
        keys = {(item.state.request.request_id, item.depth) for item in chosen}
        for item in candidates:
            if not budget or len(chosen) == self.max_num_seqs:
                break
            key = item.state.request.request_id, item.depth
            if key in keys:
                continue
            rows = item.rows
            if item.progress is None:
                rows = min(rows, budget, prefill_capacity) if item.prefill else min(rows, budget)
            if not 0 < rows <= budget:
                continue
            keys.add(key)
            chosen.append(item if rows == item.rows else replace(item, rows=rows))
            budget -= rows
            if item.progress is None and item.prefill:
                prefill_capacity -= rows
        return chosen, budget == 0 or len(chosen) == self.max_num_seqs

    def _batches(self, ready, prefill_capacity):
        queues = defaultdict(list)
        for item in ready:
            queues[item.op].append(item)

        batches = []
        for op, queue in queues.items():
            if op is LoopOp.CORE:
                work, full = self._pack_core(queue, prefill_capacity)
            else:
                candidates = queue
                if op is LoopOp.CODA and not self._unified_coda:
                    candidates = [item for item in queue if item.depth == queue[0].depth]
                work, full = self._take_work(candidates, prefill_capacity)
            if work:
                entries = queue if op is LoopOp.PRELUDE else None
                batches.append(_Batch(op, work, full, entries))

        if not batches:
            return
        batches.sort(key=lambda batch: (batch.work[0].rank, batch.work[0].since))
        stage = self._select_stage(batches)
        yield stage
        yield from (batch for batch in batches if batch is not stage)

    def _pack_core(self, queue, prefill_capacity):
        depth = queue[0].depth
        same_depth = [item for item in queue if item.depth == depth]
        direct, full = self._take_work(same_depth, prefill_capacity)
        if full or len(same_depth) == len(queue):
            return direct, full

        if self.decode_pipeline:
            # The tentative successor may share the validation batch of its own request.
            owners = {item.state.request.request_id for item in direct}
            successors = [
                item for item in queue if item.depth != depth and item.progress is not None
                and item.progress.tentative and item.state.request.request_id in owners
            ]
            return self._take_work(successors, prefill_capacity, direct)
        if not self._mixed_prefill_core:
            return direct, full

        compatible = [
            item for item in queue if item.depth == depth
            or (item.progress is not None and not item.progress.draft_coda)
        ]
        work, packed_full = self._take_work(compatible, prefill_capacity)
        if any(item.prefill for item in work) or (
            self._cross_depth_decode and all(self._ordinary_decode(item) for item in work)
        ):
            return work, packed_full

        # A prefill row can make cross-depth execution compatible even when
        # the first pass filled the remaining capacity with pure decode.
        with_prefill, full = self._take_work(
            (item for item in compatible if item.depth != depth and item.prefill),
            prefill_capacity,
            direct,
        )
        if not full and len(with_prefill) > len(direct):
            return self._take_work(compatible, prefill_capacity, with_prefill)
        return with_prefill, full

    @staticmethod
    def _ordinary_decode(item):
        work = item.progress
        return (
            not item.prefill and item.rows == 1 and work is not None
            and not (work.adaptive or work.tentative or work.draft_coda)
        )

    def _select_stage(self, batches):
        first = batches[0]
        if first.work[0].rank[0] == 0 or not self._mixed_prefill_core:
            return first

        by_op = {batch.op: batch for batch in batches}
        core = by_op.get(LoopOp.CORE)
        if core is None:
            return first

        if first.op is LoopOp.CODA and self._unified_coda:
            if self._priority_policy and (
                core.work[0].state.request.priority > first.work[0].state.request.priority
            ):
                return first
            if (core.full and not core.work[0].prefill) or (
                LoopOp.PRELUDE not in by_op and self._can_merge_coda(first, core)
            ):
                return core
        elif first.op is LoopOp.CORE and not core.full:
            prelude = by_op.get(LoopOp.PRELUDE)
            if prelude is not None and prelude.work[0].rank[0] == 4 and (
                not self._priority_policy
                or prelude.work[0].state.request.priority <= core.work[0].state.request.priority
            ):
                return prelude
        return first

    def _can_merge_coda(self, coda, core):
        if not all(self._ordinary_decode(item) for item in coda.work):
            return False
        incoming = [
            item for item in core.work if self._ordinary_decode(item)
            and item.depth + 1 == item.progress.depth_limit
        ]
        if not incoming:
            return False
        combined = (*coda.work, *incoming)
        capacity = min(self.max_num_seqs, self.max_num_scheduled_tokens)
        return len(combined) <= capacity and len({
            item.state.request.request_id for item in combined
        }) == len(combined)

    def _reserve_batch(self, batch, prefill_capacity, failed):
        selected = []
        budget = self.max_num_scheduled_tokens
        for item in batch.entry_candidates or batch.work:
            if not budget or len(selected) == self.max_num_seqs:
                break
            state, work = item.state, item.progress
            if work is not None and state.spans.get(work.start) is not work:
                continue
            if work is None:
                rows = min(item.rows, budget, prefill_capacity) if item.prefill else item.rows
                if not rows:
                    continue
                work = self._reserve_span(
                    state, rows, selected, allow_preemption=not (item.prefill and state.spans),
                )
                if work is None:
                    failed.add(state.request.request_id)
                    continue
                if work.prefill:
                    prefill_capacity -= work.count
            selected.append((state, self._make_span(state.request, work, batch.op)))
            budget -= work.count
        if batch.op is LoopOp.PRELUDE and self._mixed_prefill_core:
            owners = [state for state, span in selected if span.prefill]
            while owners and budget and prefill_capacity:
                remaining = []
                for state in owners:
                    if not budget or not prefill_capacity:
                        break
                    if not self._prefill_window_open(state):
                        continue
                    limit = min(budget, prefill_capacity)
                    work = self._reserve_span(state, limit, selected, allow_preemption=False)
                    if work is None:
                        continue
                    selected.append((state, self._make_span(state.request, work, batch.op)))
                    budget -= work.count
                    prefill_capacity -= work.count
                    remaining.append(state)
                owners = remaining
        if batch.op is LoopOp.CODA and self.split_prefill_coda and (
            budget and len(selected) < self.max_num_seqs
        ):
            picked = {span.request_id for _, span in selected}
            for item in batch.work:
                picked.add(item.state.request.request_id)
            for item in self._ready_reserved_spans():
                state, work = item
                if (
                    work.next_op is LoopOp.CODA and work.count > budget
                    and state.request.request_id not in picked
                    and self._can_split_prefill_coda(state, work)
                ):
                    selected.append((state, self._split_prefill_coda(state, work, budget)))
                    break
        return selected

    def _remember_skips(self, ready, selected, failed):
        positions = {(span.request_id, span.token_start) for _, span in selected}
        for item in ready:
            state, work = item.state, item.progress
            request_id = state.request.request_id
            if work is None:
                if state.entry_ready_since and request_id not in failed:
                    state.entry_skips = min(state.entry_skips + 1, 2)
            elif (request_id, work.start) not in positions:
                work.skips = min(work.skips + 1, 2)

    def _schedule_candidates(self, throttle_prefills):
        if self._pause_state == PauseState.PAUSED_ALL:
            return
        running, waiting = [], []
        for state in self._states.values():
            if state.request.status == RequestStatus.RUNNING:
                running.append(state)
            elif state.request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                waiting.append(state)

        # A running request can preempt a later candidate while extending its KV.
        for state in running:
            if state.request.status == RequestStatus.RUNNING:
                yield state
        if self._pause_state == PauseState.PAUSED_NEW:
            return
        num_running = sum(state.request.status == RequestStatus.RUNNING for state in running)
        if throttle_prefills and num_running:
            return
        if self._priority_policy:
            waiting.sort(key=lambda state: (
                state.entry_skips < 2,
                state.entry_ready_since if state.entry_skips >= 2 else state.request.priority,
                state.request.arrival_time,
            ))
        yield from waiting[:max(0, self.max_num_active_seqs - num_running)]

    def _prefill_window_open(self, state):
        if len(state.spans) >= self.layout.num_loops:
            return False
        last = next(reversed(state.spans.values()))
        return last.start + last.count < state.request.num_prompt_tokens

    def _ready_spans(self, state):
        if not state.spans:
            yield None
            return
        active = tuple(state.spans.values())
        if self._mixed_prefill_core and self._prefill_window_open(state):
            yield None
        elif self.decode_pipeline and len(active) == 1 and self._lookahead_token(state) is not None:
            yield None
        elif self._pending_coda(state) is not None:
            yield None
        previous = None
        for work in active:
            if work.awaiting_coda_at is None and (
                previous is None
                or (work.next_op is LoopOp.CORE and previous.completed_depth >= work.next_depth)
            ):
                yield work
            previous = work

    def _make_span(self, request, work, op):
        last_core = op is LoopOp.CORE and work.next_depth + 1 == work.depth_limit
        return LoopSpan(
            request.request_id,
            work.start,
            work.count,
            work.next_depth,
            prefill=work.prefill,
            last_core=last_core,
            sample=work.produces_token
            and not work.draft_coda
            and (op is LoopOp.CODA or (last_core and not self.layout.has_coda)),
            exit_threshold=(
                request.loop_params.exit_threshold if op is LoopOp.CORE and work.adaptive else None
            ),
            min_loops=request.loop_params.min_loops or 1,
            propose_draft=(
                self.self_speculation
                and not work.prefill
                and op is LoopOp.CORE
                and work.next_depth == 0
                and not work.tentative
                and work.depth_limit > 1
            ),
            draft_coda=work.draft_coda,
            input_token=work.input_token,
            chunk_id=work.chunk_id,
            explicit_position=work.prefill and work.start != request.num_computed_tokens,
        )

    @staticmethod
    def _can_split_prefill_coda(state, work) -> bool:
        return (
            work.prefill
            and not work.coda_tail
            and state.in_flight == 0
            and len(state.spans) == 1
            and work.start == state.request.num_computed_tokens
            and work.start + work.count <= state.request.num_prompt_tokens
            and work.completed_depth + 1 == work.depth_limit
        )

    def _split_prefill_coda(self, state, work, count: int) -> LoopSpan:
        tail = replace(
            work,
            start=work.start + count,
            count=work.count - count,
            awaiting_coda_at=work.start,
            coda_tail=True,
        )
        state.spans[tail.start] = tail
        work.count = count
        work.produces_token = False
        return replace(
            self._make_span(state.request, work, LoopOp.CODA),
            coda_remainder=tail.count,
        )

    def _lookahead_token(self, state):
        request = state.request
        first = next(iter(state.spans.values()))
        if (
            first.prefill
            or first.tentative
            or request.num_output_tokens + 1 >= request.max_tokens
            or first.start + first.count >= self.max_model_len
        ):
            return None
        return first.draft_token

    def _pending_coda(self, state) -> _SpanProgress | None:
        return None

    def _reserve_span(self, state, budget, selected, *, allow_preemption=True):
        request = state.request
        start = request.num_computed_tokens
        hit_blocks, hit_count = self.kv.empty_kv_cache_blocks, 0
        if state.spans:
            last = next(reversed(state.spans.values()))
            start = last.start + last.count
        elif not state.known_to_worker and start == 0:
            hit_blocks, hit_count, _ = self.kv.get_computed_blocks(request)
        start += hit_count
        input_token = (
            self._lookahead_token(state)
            if self.decode_pipeline and start >= request.num_tokens and len(state.spans) == 1
            else None
        )
        pending = self._pending_coda(state)
        produces_token = input_token is not None or (pending is not None and not pending.prefill)
        count = 1 if produces_token else min(budget, request.num_tokens - start)
        if start < request.num_prompt_tokens:
            count = min(count, self.prefill_chunk_size)
        if count < 1:
            return None
        reserve = start + count - request.num_computed_tokens - hit_count
        blocks = self.kv.allocate_slots(
            request,
            reserve,
            num_new_computed_tokens=hit_count,
            new_computed_blocks=hit_blocks,
            delay_cache_blocks=True,
        )
        if (
            blocks is None
            and allow_preemption
            and pending is None
            and request.status == RequestStatus.RUNNING
            and self._preempt_for(state, selected)
        ):
            blocks = self.kv.allocate_slots(
                request,
                reserve,
                num_new_computed_tokens=hit_count,
                new_computed_blocks=hit_blocks,
                delay_cache_blocks=True,
            )
        if blocks is None:
            return None

        request.num_computed_tokens += hit_count
        request.status = RequestStatus.RUNNING
        new_blocks = blocks.get_block_ids()
        if state.newly_allocated is not None:
            new_blocks = tuple(
                old + new for old, new in zip(state.newly_allocated, new_blocks, strict=True)
            )
        state.newly_allocated = new_blocks
        prefill = start < request.num_prompt_tokens or not state.known_to_worker
        params = request.loop_params
        depth_limit = self.layout.num_loops
        adaptive = False
        if not prefill:
            depth_limit = params.target_loops or params.max_loops or depth_limit
            adaptive = params.exit_threshold is not None and params.exit_threshold < 1.0
        state.spans[start] = _SpanProgress(
            start,
            count,
            depth_limit,
            prefill,
            produces_token or start + count == request.num_tokens,
            adaptive,
            self.layout.first_op,
            input_token=input_token,
            tentative=input_token is not None,
            awaiting_coda_at=pending.start if pending is not None else None,
            chunk_id=state.next_chunk_id,
        )
        state.next_chunk_id += 1
        state.entry_ready_since = 0
        state.entry_skips = 0
        return state.spans[start]

    def _preempt_for(self, waiting, selected) -> bool:
        selected_ids = {span.request_id for _, span in selected}
        for victim in reversed(self._states.values()):
            request = victim.request
            if (
                victim is waiting
                or request.status != RequestStatus.RUNNING
                or victim.in_flight
                or request.request_id in selected_ids
                or not self._can_replay(request)
            ):
                continue
            self._preempt(victim)
            return True
        return False

    @staticmethod
    def _can_replay(request: LoopRequest) -> bool:
        return not request.num_output_tokens or request.loop_params == type(request.loop_params)()

    def _preempt(self, state):
        request = state.request
        self.kv.free(request)
        if state.known_to_worker:
            self._finished.add(request.request_id)
        state.known_to_worker = False
        state.spans.clear()
        state.next_chunk_id = 0
        request.num_computed_tokens = 0
        request.status = RequestStatus.PREEMPTED
        request.num_preemptions += 1

    @staticmethod
    def _append_cached_request(
        cached: CachedRequestData, state: _RequestState, position: int | None
    ) -> None:
        request = state.request
        cached.req_ids.append(request.request_id)
        cached.new_block_ids.append(state.newly_allocated)
        cached.num_computed_tokens.append(
            request.num_computed_tokens if position is None else position
        )
        cached.num_output_tokens.append(request.num_output_tokens)

    def _on_submit(self, state, work, op, span) -> bool:
        work.next_op = None
        return False

    def _advance(self, work: _SpanProgress, op: LoopOp, span: LoopSpan, exited: bool) -> None:
        if op is LoopOp.PRELUDE:
            work.next_op = LoopOp.CORE
        elif op is LoopOp.CORE:
            if span.propose_draft:
                work.draft_coda = True
                work.next_op = LoopOp.CODA
            elif span.last_core or exited:
                work.next_op = LoopOp.CODA if self.layout.has_coda else None
            else:
                work.next_depth += 1
                work.next_op = LoopOp.CORE
        elif span.draft_coda:
            work.draft_coda = False
            work.next_depth = 1
            work.next_op = LoopOp.CORE

    def update_from_output(self, scheduler_output, model_runner_output):
        submissions = self._in_flight.pop(id(scheduler_output), ())
        grouped, self._rejected_by_client = self._rejected_by_client, defaultdict(list)
        token_rows = dict(zip(model_runner_output.req_ids, model_runner_output.sampled_token_ids))
        row_indices = (
            {request_id: i for i, request_id in enumerate(model_runner_output.req_ids)}
            if model_runner_output.logprobs is not None
            else {}
        )
        exit_depths = (
            model_runner_output.exit_depths
            if isinstance(model_runner_output, LoopModelRunnerOutput)
            else ()
        )
        draft_tokens = getattr(model_runner_output, "draft_tokens", None) or {}
        if exit_depths and len(exit_depths) != len(submissions):
            raise RuntimeError("rlt exit results do not match the submitted spans")
        op = scheduler_output.op
        for index, submission in enumerate(submissions):
            state, work, span = submission.state, submission.work, submission.span
            request = state.request
            if self._states.get(span.request_id) is not state:
                continue
            state.in_flight -= 1
            if request.is_finished():
                self._release_finished(state)
                continue

            assert op is not None
            if op is LoopOp.CORE:
                work.completed_depth = span.depth
            exited = bool(exit_depths) and exit_depths[index] is not None
            if span.draft_coda:
                work.draft_token = draft_tokens.get(span.request_id)
                if work.draft_token is None:
                    raise RuntimeError(f"rlt draft Coda dropped request {span.request_id}")
            terminal = (op is LoopOp.CODA and not span.draft_coda) or (
                op is LoopOp.CORE and (span.last_core or exited) and not self.layout.has_coda
            )
            if not submission.advanced:
                self._advance(work, op, span, exited)
            if not terminal:
                continue
            if request.num_computed_tokens != span.token_start:
                raise RuntimeError("rlt completed a token span ahead of its committed prefix")
            request.num_computed_tokens = span.token_end
            if work.prefill:
                self.kv.cache_blocks(request, min(span.token_end, request.num_prompt_tokens))
            del state.spans[span.token_start]
            if work.produces_token:
                tokens = token_rows.get(span.request_id, [])
                if not tokens:
                    raise RuntimeError(f"rlt final op did not sample request {span.request_id}")
                if work.draft_token is not None:
                    accepted = work.draft_token == tokens[0]
                    if self.trace_steps:
                        logger.info(
                            "rlt draft verdict request=%s accepted=%s", span.request_id, accepted
                        )
                    if accepted:
                        tokens[0] = work.draft_token
                if self.trace_steps and not self.layout.has_coda:
                    logger.info(
                        "rlt commit request=%s position=%d depth=%d",
                        span.request_id,
                        span.token_end,
                        work.completed_depth + 1,
                    )
                logprobs = None
                assert request.sampling_params is not None
                if request.sampling_params.num_logprobs is not None:
                    if isinstance(model_runner_output, LoopModelRunnerOutput):
                        logprobs = model_runner_output.logprobs_by_request.get(span.request_id)
                    elif model_runner_output.logprobs is not None:
                        logprobs = model_runner_output.logprobs.slice_request(
                            row_indices[span.request_id], len(tokens)
                        )
                if self.decode_pipeline and (lookahead := state.spans.get(span.token_end)):
                    retained = lookahead.input_token == tokens[0]
                    if self.trace_steps:
                        logger.info(
                            "rlt pipeline request=%s position=%d retained=%s",
                            span.request_id,
                            span.token_end,
                            retained,
                        )
                    if retained:
                        lookahead.tentative = False
                    else:
                        del state.spans[span.token_end]
                        self._discarded_states[span.request_id] = (span.token_end,)
                self._commit_tokens(request, tokens, grouped, logprobs=logprobs)
            lookahead = state.spans.get(span.token_end)
            if (
                lookahead is not None
                and lookahead.awaiting_coda_at == span.token_start
                and not request.is_finished()
            ):
                lookahead.awaiting_coda_at = None
            self._release_finished(state)

        clients = grouped.keys() | self._finished_by_client.keys()
        results = {
            client: EngineCoreOutputs(
                outputs=grouped.get(client, []),
                finished_requests=self._finished_by_client.get(client),
            )
            for client in clients
        }
        self._finished_by_client.clear()
        return results

    def _commit_tokens(self, request, tokens, grouped, *, logprobs):
        accepted = []
        self._resolve_output_placeholder(request)
        for token in tokens:
            request.append_output_token_ids(token)
            accepted.append(token)
            if check_stop(request, self.max_model_len):
                break
        grouped[request.client_index].append(
            EngineCoreOutput(
                request.request_id,
                accepted,
                new_logprobs=(
                    logprobs.slice_request(0, len(accepted)) if logprobs is not None else None
                ),
                finish_reason=request.get_finished_reason(),
                stop_reason=request.stop_reason,
            )
        )

    def _resolve_output_placeholder(self, request):
        pass

    def _release_finished(self, state):
        request = state.request
        if not request.is_finished() or state.in_flight:
            return
        self.kv.free(request)
        self._states.pop(request.request_id)
        if state.known_to_worker:
            self._finished.add(request.request_id)
        if self._include_finished_set:
            self._finished_by_client[request.client_index].add(request.request_id)

    def finish_requests(self, request_ids: str | Iterable[str] | None, finished_status):
        if request_ids is None:
            request_ids = tuple(self._states)
        elif isinstance(request_ids, str):
            request_ids = (request_ids,)
        finished = []
        for request_id in request_ids:
            state = self._states.get(request_id)
            if state is not None and not state.request.is_finished():
                state.request.status = finished_status
                finished.append(state.request)
                self._release_finished(state)
        return finished

    def has_request(self, request_id):
        return request_id in self._states

    def get_num_unfinished_requests(self):
        return sum(not state.request.is_finished() for state in self._states.values())

    def has_finished_requests(self):
        return bool(self._finished or self._rejected_by_client)

    @property
    def pause_state(self):
        return self._pause_state

    def set_pause_state(self, pause_state):
        self._pause_state = pause_state

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if reset_running_requests:
            running = [
                state
                for state in self._states.values()
                if state.request.status == RequestStatus.RUNNING
            ]
            if self._in_flight or any(not self._can_replay(state.request) for state in running):
                return False
            for state in running:
                self._preempt(state)
        return self.kv.reset_prefix_cache()

    def reset_encoder_cache(self):
        pass

    def get_request_counts(self):
        running = sum(
            state.request.status == RequestStatus.RUNNING for state in self._states.values()
        )
        return running, self.get_num_unfinished_requests() - running

    def get_kv_cache_usage(self):
        return self.kv.usage

    def make_stats(self):
        if not self.log_stats:
            return None
        running, waiting = self.get_request_counts()
        return SchedulerStats(
            num_running_reqs=running,
            num_waiting_reqs=waiting,
            kv_cache_usage=self.kv.usage,
        )

    def get_grammar_bitmask(self, scheduler_output):
        return None

    def update_draft_token_ids(self, draft_token_ids):
        pass

    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
        pass

    def shutdown(self):
        for state in self._states.values():
            self.kv.free(state.request)
        self._states.clear()
        self._in_flight.clear()
