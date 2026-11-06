"""CPU scheduling at request-phase/loop boundaries, independent of model execution."""

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum, auto

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.core.scheduling_policy import create_scheduling_policy, requires_staged_prefill
from vllm_rlt.request import FinishReason, Request, RequestPhase
from vllm_rlt.worker.output import ExitSignal, ModelRunnerOutput

# Items of these phases cover a contiguous span of prompt tokens, so their KV
# frontier is the span end rather than the request's decode position.
_SPAN_PHASES = (
    RequestPhase.PREFILL,
    RequestPhase.PREFILLING_PRELUDE,
    RequestPhase.PREFILLING_RECURRENT,
    RequestPhase.SPECULATIVE,
)
# Stages that must reserve KV capacity for that frontier before selection.
_CAPACITY_PHASES = _SPAN_PHASES + (RequestPhase.PRELUDE, RequestPhase.RECURRENT)


@dataclass(frozen=True)
class ScheduledRequest:
    request: Request
    # PREFILL and SPECULATIVE use a contiguous span; ordinary decode uses one row.
    token_start: int = 0
    token_count: int = 1
    # Recurrence position/depth are snapshotted for both prompt and generation.
    # The engine must not infer them from mutable request state after submission.
    depth: int | None = None
    chunk_start: int | None = None
    completes_prompt: bool = False
    phase: RequestPhase | None = None
    # Row offset in the request's prefill hidden-state buffer. None lets the runner
    # derive token_start - chunk_start.
    prefill_hidden_row_offset: int | None = None


@dataclass(frozen=True)
class SchedulerOutput:
    # None means a mixed batch; each scheduled request carries its own phase.
    phase: RequestPhase | None
    requests: list[ScheduledRequest]
    # Actual number of recurrent stages in this submission. None is reserved for
    # direct callers that construct a batch without passing through Scheduler.
    loop_count: int | None = None

    def __post_init__(self):
        if self.loop_count is not None and (
            type(self.loop_count) is not int or self.loop_count < 0
        ):
            raise ValueError("loop_count must be a nonnegative integer")

    @property
    def is_speculative(self) -> bool:
        """Execution-mode flag for this batch, not the engine-wide configuration."""
        return self.phase == RequestPhase.SPECULATIVE

    @property
    def is_mixed(self) -> bool:
        return self.phase is None

    def phase_for(self, scheduled_request) -> RequestPhase:
        phase = scheduled_request.phase or self.phase
        if phase is None:
            raise ValueError("mixed batches require a phase on every scheduled request")
        return phase

    @property
    def num_tokens(self) -> int:
        return sum(scheduled_request.token_count for scheduled_request in self.requests)


@dataclass(frozen=True)
class PrefillProgress:
    """Logical host-side progress for a submitted split-prefill request.

    These flags describe request frontiers, not GPU completion. Device work may
    still be pending when the chunk or prompt becomes logically complete.
    """

    chunk_complete: bool
    prompt_complete: bool
    transfer_start: int | None = None


@dataclass
class SchedulerUpdate:
    outputs: list[Request] = field(default_factory=list)
    exited: list[Request] = field(default_factory=list)
    finished: list[Request] = field(default_factory=list)
    prefill: list[tuple[ScheduledRequest, RequestPhase, PrefillProgress]] = field(
        default_factory=list
    )
    next_prelude: list[int] = field(default_factory=list)
    retain_signal: list[int] = field(default_factory=list)
    committed_tokens: int = 0
    accepted_tokens: int = 0


class ResumeResult(Enum):
    NOT_PREEMPTED = auto()
    RESTORED = auto()
    BLOCKED = auto()


@dataclass(frozen=True)
class AdmissionPlan:
    """One candidate's allocation inputs and admission budget, not a reservation."""

    capacity_tokens: int
    initial_tokens: int
    prefix_blocks: tuple[tuple[int, ...], ...]
    cached_tokens: int
    required_blocks: int
    reserved_growth_blocks: int
    admission_headroom_blocks: int
    cached_claim_blocks: int

    @property
    def total_budget_blocks(self) -> int:
        return (
            self.required_blocks
            + self.reserved_growth_blocks
            + self.admission_headroom_blocks
            + self.cached_claim_blocks
        )


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        cache_manager,
        speculative_config=None,
        *,
        produces_tokens: bool = True,
        total_ut_steps: int | None = None,
        eos_token_id=(),
        exit_mode: str = "ouro",
    ):
        self.config = config
        self.cache_manager = cache_manager
        self.speculative_config = speculative_config
        self.total_ut_steps = (
            total_ut_steps if total_ut_steps is not None else getattr(cache_manager, "max_loops", 1)
        )
        if eos_token_id is None:
            self.eos_token_ids = ()
        elif isinstance(eos_token_id, (tuple, list)):
            self.eos_token_ids = tuple(eos_token_id)
        else:
            self.eos_token_ids = (eos_token_id,)
        self.exit_mode = exit_mode
        self.requests: dict[str, Request] = {}
        self.queues: dict[RequestPhase, deque[str]] = {s: deque() for s in RequestPhase}
        # Bound by Engine when preemption is enabled. These callbacks can copy
        # device state; queue transitions remain owned by this scheduler.
        self.suspend_callback = None
        self.resume_callback = None
        # Optional connector readiness and export predicates. Never mutate queues or blocks.
        self.execution_ready = None
        self.exports_request = None
        # Staged prompt state is required by wavefront regrouping and mixed
        # recurrent batches. Other prompt chunks execute all depths atomically.
        self.wavefront_chunked_prefill = config.wavefront_chunked_prefill
        self.staged_prefill = requires_staged_prefill(config, produces_tokens=produces_tokens)
        self.policy = create_scheduling_policy(
            config,
            speculative_config,
            produces_tokens=produces_tokens,
        )

    def add_request(self, request: Request):
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request ID: {request.request_id}")
        self.requests[request.request_id] = request
        self.queues[RequestPhase.WAITING].append(request.request_id)

    def enqueue(self, request: Request, phase: RequestPhase):
        request.phase = phase
        self.queues[phase].append(request.request_id)

    def suspend_for_transfer(self, request):
        """Stop scheduling without freeing resources that a transfer still borrows."""
        if self.requests.get(request.request_id) is not request:
            return False
        for queue in self.queues.values():
            while request.request_id in queue:
                queue.remove(request.request_id)
        request.phase = RequestPhase.TRANSFERRING
        return True

    def finish(self, request: Request, reason: FinishReason):
        # A delayed EOS can stop a request already queued for its next phase.
        for queue in self.queues.values():
            while request.request_id in queue:
                queue.remove(request.request_id)
        self.cache_manager.free(request.request_id)
        request.phase = RequestPhase.FINISHED
        request.finish_reason = reason
        request.hidden_state = None
        request.input_token_tensor = None
        request.num_output_placeholders = 0
        request.num_speculative_rounds = 0
        self.requests.pop(request.request_id)

    def abort(self, request_id: str) -> Request:
        request = self.requests[request_id]
        self.finish(request, FinishReason.ABORT)
        return request

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.requests)

    def _order_waiting_requests(self) -> deque[str]:
        """Return the live waiting queue in admission order.

        Priority is supplied by the caller, not estimated from prompt length.
        Lower values go first: -5, 0, 10. Equal values retain current queue order.
        FCFS preserves queue order, but admission may still bypass blocked requests.
        """
        waiting = self.queues[RequestPhase.WAITING]
        if self.config.policy == "priority":
            ordered = sorted(waiting, key=lambda rid: self.requests[rid].sampling_params.priority)
            waiting.clear()
            waiting.extend(ordered)
        return waiting

    def _is_preemption_candidate(self, request, requesting_request, *, priority_only, excluded):
        """Check safety before considering a victim's priority.

        Pending outputs, transfer leases and the current batch's selected
        requests must remain alive. priority_only additionally forbids evicting
        equal/higher priority requests to admit a new arrival.
        """
        if (
            request is requesting_request
            or request.phase == RequestPhase.RECEIVING
            or (
                request.phase == RequestPhase.WAITING
                and (priority_only or not self.cache_manager.has_allocation(request.request_id))
            )
            or request.num_output_placeholders
            or request.num_speculative_rounds
            or request.request_id in excluded
        ):
            return False
        if self.cache_manager.has_transfer_lease(request.request_id):
            return False
        return not priority_only or (
            request.sampling_params.priority > requesting_request.sampling_params.priority
        )

    def _select_preemption_victim(
        self, requesting_request, *, priority_only, excluded, waiting_only=False
    ):
        """Prefer lower priority, then a larger current-position KV footprint.

        Priority 10 loses to priority 0. The footprint tie-breaker uses
        required_blocks(position + 1), not exact reclaimable physical pages.
        Capacity-pressure preemption retains this ranking even in FCFS mode;
        priority_only restricts eligibility only for priority admission.
        """
        candidates = (
            request
            for request in self.requests.values()
            if (not waiting_only or request.phase == RequestPhase.WAITING)
            and self._is_preemption_candidate(
                request, requesting_request, priority_only=priority_only, excluded=excluded
            )
        )
        return max(
            candidates,
            key=lambda request: (
                request.sampling_params.priority,
                self.cache_manager.required_blocks(request.position + 1),
            ),
            default=None,
        )

    def preempt(
        self, requesting_request, *, priority_only=False, excluded=frozenset(), waiting_only=False
    ):
        """Yield a priority slot or reclaim KV from one safe victim.

        Priority admission keeps KV resident. Capacity pressure may also select
        an already waiting resident request; offloading it preserves queue order.
        Admission allocation retries only reclaim such waiting allocations.
        """
        if self.suspend_callback is None:
            return False
        victim = self._select_preemption_victim(
            requesting_request,
            priority_only=priority_only,
            excluded=excluded,
            waiting_only=waiting_only,
        )
        if victim is None:
            return False
        self.suspend_callback(victim, offload_kv=not priority_only)
        if victim.phase == RequestPhase.WAITING:
            return True
        for queue in self.queues.values():
            while victim.request_id in queue:
                queue.remove(victim.request_id)
        self.enqueue(victim, RequestPhase.WAITING)
        return True

    def _ensure_active_slot(self, request: Request, active_count: int) -> int | None:
        """Return the updated active count, or None when admission must stop.

        A full engine may make room for a higher-priority arrival. Scheduler
        suspends a selected victim and moves it to WAITING before admitting the
        arrival. The victim retains KV until an allocation actually needs it.
        Admission runs before any batch is selected, so no request is excluded
        from preemption here.
        """
        if active_count < self.config.max_num_seqs:
            return active_count
        if (
            self.config.policy == "priority"
            and self.suspend_callback is not None
            and self.preempt(request, priority_only=True, excluded=frozenset())
        ):
            return active_count - 1
        return None

    def _try_resume(self, request: Request) -> ResumeResult:
        """Translate the existing restoration callback's three-way result.

        None: no snapshot, continue fresh admission.
        RequestPhase: resources/state restored; enqueue the saved phase.
        False: snapshot exists, but restoration cannot obtain capacity yet.
        """
        restored = self.resume_callback(request) if self.resume_callback is not None else None
        while restored is False and self.preempt(request, waiting_only=True):
            restored = self.resume_callback(request)
        if restored is None:
            return ResumeResult.NOT_PREEMPTED
        if restored is False:
            return ResumeResult.BLOCKED
        self.enqueue(request, restored)
        return ResumeResult.RESTORED

    def _reserved_growth_blocks(self, *, reserve_outputs: bool) -> int:
        """Budget future growth of existing requests without allocating it.

        Example: a request needs 12 blocks eventually but owns 4. Without
        preemption, reserve the remaining 8 even though they still look free.
        With preemption, protect in-progress prompts; decode growth may instead
        reclaim capacity by suspending another request later.
        """
        cache = self.cache_manager
        reserved = 0
        for request in self.requests.values():
            # Imported/reserved requests already own pages before compute admission.
            # Without preemption their future decode growth must remain affordable too.
            has_allocation = cache.has_allocation(request.request_id)
            if request.phase not in (
                RequestPhase.PREFILL,
                RequestPhase.PREFILLING_PRELUDE,
                RequestPhase.PREFILLING_RECURRENT,
            ) and not (reserve_outputs and has_allocation):
                continue
            tokens = len(request.prompt_token_ids)
            if reserve_outputs:
                tokens += request.sampling_params.max_tokens - 1
            allocated = cache.allocated_blocks(request.request_id)
            reserved += max(0, cache.required_blocks(tokens) - allocated)
        return reserved

    def _plan_admission(self, request: Request, active_count: int) -> AdmissionPlan:
        """Describe prefix reuse, physical allocation and logical admission cost.

        With 8 prompt tokens and 5 outputs, capacity is 12: the last sampled
        output is never fed back into the model. Incremental allocation may
        acquire fewer pages now, while budgeting the remaining growth.

        Prefix lookup can publish completed prefixes and update cache LRU order;
        planning does not allocate request pages, but is not a pure cache read.
        All block counts include the cache's storage-depth multiplier.
        """
        cache = self.cache_manager
        capacity = len(request.prompt_token_ids) + request.sampling_params.max_tokens - 1
        prefix = cache.lookup_prefix(request.prompt_token_ids)
        cached_tokens = len(prefix) * cache.block_size
        initial_tokens = min(
            len(request.prompt_token_ids), cached_tokens + self.config.prefill_chunk_size
        )
        if not cache.incremental_allocation:
            initial_tokens = capacity
        reserve_outputs = cache.incremental_allocation and self.suspend_callback is None
        budget_tokens = (
            len(request.prompt_token_ids)
            if cache.incremental_allocation and not reserve_outputs
            else capacity
        )
        required = cache.required_blocks(budget_tokens) - len(prefix) * cache.storage_depths
        reserved = self._reserved_growth_blocks(reserve_outputs=reserve_outputs)
        # Allow an otherwise idle engine to start a large request.
        admission_headroom_blocks = cache.watermark_blocks if active_count else 0
        # num_free_blocks includes evictable prefix-only blocks. Reusing those
        # blocks makes them non-evictable: do not count the same capacity twice.
        # Example: free=8 fresh + 4 cached, demand=12 with a 4-block prefix.
        # New demand is 8, but the cached claim is 4: admission costs 12, not 8.
        cached_claim = cache.exclusive_prefix_blocks(prefix)
        return AdmissionPlan(
            capacity_tokens=capacity,
            initial_tokens=initial_tokens,
            prefix_blocks=prefix,
            cached_tokens=cached_tokens,
            required_blocks=required,
            reserved_growth_blocks=reserved,
            admission_headroom_blocks=admission_headroom_blocks,
            cached_claim_blocks=cached_claim,
        )

    def reserve_transfer(self, request: Request) -> AdmissionPlan | None:
        """Reserve transport KV under the same growth budget as local admission.

        A transfer needs its whole prompt mapped before compute starts. Reserving
        it must not spend pages already promised to other requests' growth.
        This operation grants no compute slot and leaves queue ownership unchanged.
        """
        occupied = any(self.cache_manager.has_allocation(rid) for rid in self.requests)
        plan = self._plan_admission(request, int(occupied))
        if self.cache_manager.incremental_allocation:
            plan = replace(plan, initial_tokens=len(request.prompt_token_ids))
        if not self._allocate(request, plan):
            return None
        self.cache_manager.pin_transfer(request.request_id, request.request_id)
        return plan

    def _allocate(self, request: Request, plan: AdmissionPlan) -> bool:
        cache = self.cache_manager
        if plan.total_budget_blocks > cache.num_free_blocks:
            return False
        return cache.allocate(
            request.request_id,
            plan.capacity_tokens,
            initial_tokens=plan.initial_tokens,
            prefix=plan.prefix_blocks,
        )

    def _commit_admission(self, request: Request, plan: AdmissionPlan) -> bool:
        """Allocate pages and enter PREFILL only when the whole budget fits."""
        if not self._allocate(request, plan):
            return False
        request.num_prefilled_tokens = plan.cached_tokens
        # A reused prompt prefix is already prefilled at every depth, so the
        # staged window starts closed at that frontier and opens for the rest.
        request.num_preluded_tokens = plan.cached_tokens
        request.prefill_window_origin = plan.cached_tokens
        request.prefill_reclaimable_tokens = plan.cached_tokens
        if request.prefill_depth_ranges:
            request.prefill_depth_ranges[:] = Request.prefill_boundaries(
                len(request.prompt_token_ids),
                len(request.prefill_depth_ranges),
                start=plan.cached_tokens,
            )
        self._enter_prefill(request)
        return True

    def _enter_prefill(self, request):
        if self.staged_prefill:
            self.enqueue(request, RequestPhase.PREFILLING_PRELUDE)
        else:
            self.enqueue(request, RequestPhase.PREFILL)

    def _admit(self):
        """Admit waiting requests: order -> slot -> restore/allocate -> defer/enter.

        A blocked long request may let a short one pass. Each successful fresh
        admission increments earlier blocked requests' bypass counts; reaching
        the limit stops further bypasses so active requests can drain.
        """
        if not self.queues[RequestPhase.WAITING]:
            return
        active_count = sum(
            r.phase not in (RequestPhase.WAITING, RequestPhase.RECEIVING) and not r.compute_released
            for r in self.requests.values()
        )
        if active_count >= self.config.max_num_seqs and (
            self.config.policy != "priority" or self.suspend_callback is None
        ):
            return
        waiting = self._order_waiting_requests()
        deferred = []
        scan_count = min(len(waiting), self.config.admission_scan_limit)
        for _ in range(scan_count):
            if not waiting:
                break
            # Peek before popping: preemption may append its victim to WAITING.
            request = self.requests[waiting[0]]
            available_count = self._ensure_active_slot(request, active_count)
            if available_count is None:
                break
            active_count = available_count
            request_id = waiting.popleft()
            request = self.requests[request_id]

            if request.phase == RequestPhase.RECEIVING:
                # Transfer completion has already made KV and hidden valid.
                # Imported and local requests compete in this same admission queue.
                self.cache_manager.unpin_transfer(request_id, request_id)
                self.enqueue(request, RequestPhase.CODA)
            elif self.cache_manager.has_allocation(
                request_id
            ) and self.cache_manager.has_transfer_lease(request_id):
                # Outgoing transfer reservation already allocated the prompt KV.
                self._enter_prefill(request)
            else:
                resume_result = self._try_resume(request)
                if resume_result == ResumeResult.RESTORED:
                    active_count += 1
                    continue
                if resume_result == ResumeResult.BLOCKED:
                    deferred.append(request_id)
                    continue

                plan = self._plan_admission(request, active_count)
                admitted = self._commit_admission(request, plan)
                while not admitted and self.preempt(request, waiting_only=True):
                    # Recompute prefix claims and growth reservations after
                    # releasing a resident allocation; the previous plan is stale.
                    plan = self._plan_admission(request, active_count)
                    admitted = self._commit_admission(request, plan)
                if not admitted:
                    deferred.append(request_id)
                    if request.admission_bypasses >= self.config.max_admission_bypasses:
                        break
                    continue

            active_count += 1
            for blocked_id in deferred:
                self.requests[blocked_id].admission_bypasses += 1
            if any(
                self.requests[rid].admission_bypasses >= self.config.max_admission_bypasses
                for rid in deferred
            ):
                break
        # [A, B] deferred must precede the untouched tail in the same order.
        waiting.extendleft(reversed(deferred))

    def _make_scheduled_request(
        self, request: Request, phase: RequestPhase, token_budget: int
    ) -> ScheduledRequest:
        """Snapshot one execution range without advancing request progress.

        Prompt length 10, prefilled 4, chunk 4, budget 6 -> range [4, 8).
        Wavefront expands a queued request into multiple ranges in take_phases().
        Decode phases contribute one position; its loop may differ by request.
        """
        if phase == RequestPhase.PREFILL:
            start = request.num_prefilled_tokens
            count = min(
                token_budget, self.config.prefill_chunk_size, len(request.prompt_token_ids) - start
            )
            return ScheduledRequest(request, start, count)
        # The two split phases are unreachable while atomic prefill is enabled.
        if phase == RequestPhase.PREFILLING_PRELUDE:
            start = request.num_preluded_tokens
            count = min(
                token_budget,
                self.config.prefill_chunk_size,
                len(request.prompt_token_ids) - start,
            )
            return ScheduledRequest(request, start, count, chunk_start=start)
        if phase == RequestPhase.PREFILLING_RECURRENT:
            start = request.num_prefilled_tokens
            if start >= request.num_preluded_tokens:
                return ScheduledRequest(
                    request, start, 0, depth=0, chunk_start=request.prefill_reclaimable_tokens
                )
            first_token_depth = request.prefill_depth(start)
            # One run per depth frontier: every token of the run already completed
            # this depth, and no earlier token is missing a shallower one, so the
            # plane's causal prefix covers the run. Tokens past num_preluded_tokens
            # have no resident prelude and cannot run yet.
            boundary = request.prefill_depth_ranges[first_token_depth]
            count = min(
                token_budget,
                self.config.prefill_chunk_size,
                request.num_preluded_tokens - start,
                boundary[1] - start,
                len(request.prompt_token_ids) - start,
            )
            return ScheduledRequest(
                request,
                start,
                count,
                depth=first_token_depth,
                chunk_start=request.prefill_reclaimable_tokens,
                completes_prompt=(
                    first_token_depth == len(request.prefill_depth_ranges) - 1
                    and start + count == len(request.prompt_token_ids)
                ),
            )
        if phase == RequestPhase.SPECULATIVE:
            # K candidates plus one bonus distribution. At the output limit,
            # K=0 is an ordinary fixed-depth step and needs no extra KV slot.
            remaining = request.sampling_params.max_tokens - request.num_scheduled_outputs
            count = min(self.speculative_config.num_speculative_tokens + 1, remaining, token_budget)
            return ScheduledRequest(request, request.position, count)
        if phase == RequestPhase.RECURRENT:
            return ScheduledRequest(request, request.position, depth=request.loops_done)
        return ScheduledRequest(request)

    def _make_wavefront_task(
        self, request, phase, token_budget, *, start, depth=None
    ) -> ScheduledRequest | None:
        """Snapshot one chunk within the bounded per-request wavefront ring."""
        chunk_size = self.config.prefill_chunk_size
        origin = request.prefill_window_origin
        if start < origin:
            raise RuntimeError("wavefront work precedes its cached-prefix origin")
        chunk_start = origin + (start - origin) // chunk_size * chunk_size
        chunk_offset = start - chunk_start
        chunk_end = min(len(request.prompt_token_ids), chunk_start + chunk_size)
        total_depths = len(request.prefill_depth_ranges)
        slot = ((chunk_start - origin) // chunk_size) % total_depths
        prefill_hidden_row_offset = slot * chunk_size + chunk_offset

        if phase == RequestPhase.PREFILLING_PRELUDE:
            window_end = min(
                len(request.prompt_token_ids),
                request.prefill_reclaimable_tokens + total_depths * chunk_size,
            )
            end = min(chunk_end, window_end)
        else:
            if depth is None:
                raise ValueError("wavefront recurrent work requires an explicit depth")
            left, right = request.prefill_depth_ranges[depth]
            if left != start:
                raise RuntimeError("wavefront recurrent task is not at its depth frontier")
            end = min(chunk_end, right, request.num_preluded_tokens)

        count = min(token_budget, end - start)
        if count <= 0:
            return None
        return ScheduledRequest(
            request,
            start,
            count,
            depth=depth,
            chunk_start=chunk_start,
            completes_prompt=(
                phase == RequestPhase.PREFILLING_RECURRENT
                and depth == total_depths - 1
                and start + count == len(request.prompt_token_ids)
            ),
            prefill_hidden_row_offset=prefill_hidden_row_offset,
        )

    def _ensure_execution_capacity(
        self, request: Request, frontier: int, *, selected: set[str]
    ) -> bool:
        """Grow KV for this step, reclaiming safe victims only while it is short.

        frontier is an exclusive token count, not a loop count. Another loop at
        the same token position normally uses existing capacity. selected holds
        the requests already chosen for this batch; preemption must not suspend
        them, so it is handed a frozen snapshot of that batch-local selection
        instead of reading shared Scheduler state.
        """
        cache = self.cache_manager
        while not cache.ensure_capacity(request.request_id, frontier):
            if not self.preempt(request, excluded=frozenset(selected)):
                return False
        return True

    def _prelude_window_open(self, request: Request) -> bool:
        """Whether the next prompt chunk may be staged for this request.

        Split prefill keeps one staged chunk. WCPB opens a bounded ring window,
        and retires one ring slot after each contiguous final-depth chunk.
        """
        if self.wavefront_chunked_prefill:
            window_end = min(
                len(request.prompt_token_ids),
                request.prefill_reclaimable_tokens
                + self.config.prefill_chunk_size * len(request.prefill_depth_ranges),
            )
            return request.num_preluded_tokens < window_end
        return request.num_preluded_tokens == request.num_prefilled_tokens

    def has_ready(self, phase):
        """A blocked transport queue must not hide another runnable phase."""
        queue = self.queues[phase]
        if self.execution_ready is None:
            return bool(queue)
        return any(self.execution_ready(self.requests[rid]) for rid in queue)

    def _take(self, phase: RequestPhase) -> SchedulerOutput | None:
        return self.take_phases((phase,))

    def take_phases(self, phases, *, eligible=None):
        """Select compatible requests across queues under one budget/reservation set."""
        eligible = eligible if eligible is not None else self.execution_ready
        token_budget = self.config.max_num_batched_tokens
        scheduled_requests = []
        selected: set[str] = set()
        for phase in phases:
            queue = self.queues[phase]
            if self.wavefront_chunked_prefill and phase in (
                RequestPhase.PREFILLING_PRELUDE,
                RequestPhase.PREFILLING_RECURRENT,
            ):
                token_budget = self._take_wavefront_phase(
                    phase,
                    queue,
                    token_budget,
                    scheduled_requests,
                    selected,
                    eligible,
                )
                continue
            remaining = len(queue)
            while queue and token_budget and len(selected) < self.config.max_num_seqs and remaining:
                remaining -= 1
                request = self.requests[queue.popleft()]
                if eligible is not None and not eligible(request):
                    queue.append(request.request_id)
                    continue
                if phase == RequestPhase.PREFILLING_PRELUDE and not self._prelude_window_open(
                    request
                ):
                    queue.append(request.request_id)
                    continue
                if phase == RequestPhase.SPECULATIVE and (
                    request.num_speculative_rounds >= 2
                    or request.num_scheduled_outputs >= request.sampling_params.max_tokens
                ):
                    queue.append(request.request_id)
                    continue
                scheduled_request = self._make_scheduled_request(request, phase, token_budget)
                if (
                    phase in (RequestPhase.PREFILLING_PRELUDE, RequestPhase.PREFILLING_RECURRENT)
                    and scheduled_request.token_count <= 0
                ):
                    queue.append(request.request_id)
                    continue
                if phase in _CAPACITY_PHASES:
                    # A scheduled span declares a prompt frontier; decode phases advance the
                    # request's own position by one.
                    frontier = (
                        scheduled_request.token_start + scheduled_request.token_count
                        if phase in _SPAN_PHASES
                        else request.position + 1
                    )
                    if not self._ensure_execution_capacity(request, frontier, selected=selected):
                        queue.append(request.request_id)
                        continue
                # Selecting A must prevent B's later capacity check from evicting A.
                selected.add(request.request_id)
                scheduled_requests.append(replace(scheduled_request, phase=phase))
                token_budget -= scheduled_request.token_count
        if not scheduled_requests:
            return None
        # Fairness follows the requests actually selected, including a mixed attempt
        # whose other queue could not reserve capacity.
        selected_phases = {scheduled_request.phase for scheduled_request in scheduled_requests}
        prompt_phases = selected_phases.intersection(self.policy.PREFILL_PHASES)
        recorded = (
            RequestPhase.RECURRENT
            if RequestPhase.RECURRENT in selected_phases
            else next(iter(prompt_phases))
            if prompt_phases
            else scheduled_requests[0].phase
        )
        loop_count = self._planned_loop_count(scheduled_requests)
        self.policy.record_batch(recorded, loop_count=loop_count)
        scheduled_requests = [
            replace(
                item,
                completes_prompt=(
                    self._scheduled_prefill_completes(item, loop_count)
                    if item.phase
                    in (RequestPhase.PREFILLING_PRELUDE, RequestPhase.PREFILLING_RECURRENT)
                    else item.completes_prompt
                ),
            )
            for item in scheduled_requests
        ]
        return SchedulerOutput(
            next(iter(selected_phases)) if len(selected_phases) == 1 else None,
            scheduled_requests,
            loop_count,
        )

    def _planned_loop_count(self, scheduled_requests):
        phases = {item.phase for item in scheduled_requests}
        if phases == {RequestPhase.PREFILL}:
            return self.total_ut_steps
        if RequestPhase.SPECULATIVE in phases or RequestPhase.CODA in phases:
            return 0
        if RequestPhase.PRELUDE in phases:
            return 0
        if self.wavefront_chunked_prefill:
            if RequestPhase.PREFILLING_RECURRENT in phases:
                return 1
            if phases == {RequestPhase.RECURRENT}:
                return self._fused_decode_loops(scheduled_requests)
            return 0
        if RequestPhase.PREFILLING_PRELUDE in phases:
            # In mixed mode, stage just the embeddings when recurrent decode work
            # is ready so the next depth can join that batch. Without a reason to
            # yield, PRELUDE can execute the full prompt depth in this submission.
            if self.config.mode == "mixed" and self.has_ready(RequestPhase.RECURRENT):
                return 0
            return self.total_ut_steps
        if RequestPhase.PREFILLING_RECURRENT in phases:
            if self.config.mode == "mixed" and (
                RequestPhase.RECURRENT in phases or self.has_ready(RequestPhase.RECURRENT)
            ):
                return 1
            return min(self._max_contiguous_prefill_loops(item) for item in scheduled_requests)
        if RequestPhase.RECURRENT in phases:
            return self._fused_decode_loops(scheduled_requests)
        raise ValueError(f"cannot derive recurrent boundary for phases: {phases}")

    def _fused_decode_loops(self, scheduled_requests):
        """Loops one decode RECURRENT submission may run without a host decision.

        A row whose exit depth is already known (fixed depth, or a trace) needs no
        gate readback between loops, so its remaining loops can run back to back
        in one submission with per-loop KV metadata. Any row that still depends
        on a hazard score keeps the one-loop boundary for the whole batch. The
        batch advances every row by the same count, so the minimum remaining
        depth bounds the fusion; rows with more loops left return to the
        RECURRENT queue afterwards.
        """
        if self.cache_manager.layout != "last_exited":
            # SHARED overwrites one plane through recurrence; keep its per-loop boundaries.
            return 1
        count = None
        for item in scheduled_requests:
            request = item.request
            params = request.sampling_params
            maximum = params.max_loops or self.total_ut_steps
            if self.exit_mode == "trace":
                target = request.exit_trace[request.num_scheduled_outputs]
            elif params.exit_threshold >= 1.0:
                target = maximum
            else:
                return 1
            remaining = min(target, maximum) - request.loops_done
            if remaining < 1:
                return 1
            count = remaining if count is None else min(count, remaining)
        return count or 1

    def _batch_loop_count(self, batch):
        if batch.loop_count is not None:
            return batch.loop_count
        phases = {batch.phase_for(item) for item in batch.requests}
        if RequestPhase.PREFILL in phases:
            return self.total_ut_steps
        if phases.intersection((RequestPhase.PRELUDE, RequestPhase.CODA, RequestPhase.SPECULATIVE)):
            return 0
        if phases.intersection((RequestPhase.PREFILLING_RECURRENT, RequestPhase.RECURRENT)):
            return 1
        return 0

    def _max_contiguous_prefill_loops(self, scheduled_request):
        """Count consecutive depth frontiers that already cover this exact span."""
        start = scheduled_request.token_start
        end = start + scheduled_request.token_count
        depth = scheduled_request.depth
        loops = 0
        for left, right in scheduled_request.request.prefill_depth_ranges[depth:]:
            if left != start or right < end:
                break
            loops += 1
        return max(1, loops)

    def _scheduled_prefill_completes(self, scheduled_request, loop_count):
        phase = scheduled_request.phase
        depth = scheduled_request.depth if phase == RequestPhase.PREFILLING_RECURRENT else 0
        return (
            loop_count > 0
            and depth + loop_count == self.total_ut_steps
            and scheduled_request.token_start + scheduled_request.token_count
            == len(scheduled_request.request.prompt_token_ids)
        )

    def _take_wavefront_phase(
        self, phase, queue, token_budget, scheduled_requests, selected, eligible
    ):
        """Round-robin each request's chunks/depths into the shared token budget."""
        remaining = len(queue)
        candidates = []
        owner_limit = self.config.max_num_seqs - len(selected)
        while queue and remaining and len(candidates) < owner_limit and token_budget:
            remaining -= 1
            request = self.requests[queue.popleft()]
            if eligible is not None and not eligible(request):
                queue.append(request.request_id)
                continue
            if phase == RequestPhase.PREFILLING_PRELUDE and not self._prelude_window_open(request):
                queue.append(request.request_id)
                continue
            if phase == RequestPhase.PREFILLING_RECURRENT and not any(
                start < min(end, request.num_preluded_tokens)
                for start, end in request.prefill_depth_ranges
            ):
                queue.append(request.request_id)
                continue
            candidates.append(request)

        if not candidates:
            return token_budget

        added = set()
        blocked = set()
        total_depths = len(candidates[0].prefill_depth_ranges)

        def accept_task(request: Request, scheduled_request: ScheduledRequest) -> bool:
            """Reserve capacity and commit one wavefront task to this batch."""
            nonlocal token_budget
            request_id = request.request_id
            frontier = scheduled_request.token_start + scheduled_request.token_count
            if not self._ensure_execution_capacity(request, frontier, selected=selected):
                blocked.add(request_id)
                return False
            selected.add(request_id)
            added.add(request_id)
            token_budget -= scheduled_request.token_count
            return True

        tasks = []
        prelude_cursors = (
            {request.request_id: request.num_preluded_tokens for request in candidates}
            if phase == RequestPhase.PREFILLING_PRELUDE
            else None
        )
        for depth in range(total_depths):
            for owner_index, request in enumerate(candidates):
                request_id = request.request_id
                if token_budget <= 0:
                    break
                if request_id in blocked or request.phase != phase:
                    continue
                if phase == RequestPhase.PREFILLING_PRELUDE:
                    assert prelude_cursors is not None
                    start, end = prelude_cursors[request_id], len(request.prompt_token_ids)
                else:
                    start, end = request.prefill_depth_ranges[depth]
                    end = min(end, request.num_preluded_tokens)
                if start >= end:
                    continue
                scheduled_request = self._make_wavefront_task(
                    request,
                    phase,
                    token_budget,
                    depth=(None if phase == RequestPhase.PREFILLING_PRELUDE else depth),
                    start=start,
                )
                if scheduled_request is None or not accept_task(request, scheduled_request):
                    continue
                if phase == RequestPhase.PREFILLING_PRELUDE:
                    assert prelude_cursors is not None
                    prelude_cursors[request_id] += scheduled_request.token_count
                tasks.append((owner_index, scheduled_request))
        tasks.sort(key=lambda item: (item[0], item[1].token_start))
        scheduled_requests.extend(
            replace(scheduled_request, phase=phase) for _, scheduled_request in tasks
        )

        for request in candidates:
            # Growing another candidate may have suspended this request and put
            # it in WAITING. It must not also return to its old prompt queue.
            if request.request_id not in added and request.phase == phase:
                queue.append(request.request_id)
        return token_budget

    def schedule(self, *, prefer_recurrent=False) -> SchedulerOutput | None:
        """Choose queued requests or an admission opportunity, then build a batch.

        An unfinished request is not necessarily runnable (e.g. PD RECEIVING).
        Conversely, one long active request must not prevent new requests from
        filling available slots. The phase policy decides when to call _admit;
        merely checking whether requests exist must not allocate or preempt.
        """
        if not self.requests:
            return None
        return self.policy.schedule(self, prefer_recurrent=prefer_recurrent)

    def _prompt_destination(self, request: Request) -> RequestPhase:
        exporting = self.exports_request is not None and self.exports_request(request)
        return RequestPhase.TRANSFERRING if exporting else RequestPhase.CODA

    def _advance_prefill_depth_frontier(
        self,
        request: Request,
        *,
        depth: int,
        token_end: int,
    ) -> bool:
        """Advance one submitted prompt span through one recurrent depth."""
        request.prefill_depth_ranges[depth][0] = token_end
        if depth + 1 < self.total_ut_steps:
            request.prefill_depth_ranges[depth + 1][1] = token_end
            return False
        request.num_prefilled_tokens = token_end
        return True

    def _apply_split_prefill_progress(
        self,
        scheduled_request: ScheduledRequest,
        phase: RequestPhase,
        *,
        loop_count: int | None = None,
        event=None,
    ) -> PrefillProgress:
        """Advance host-side logical progress for a submitted prefill request.

        The device work may still be pending. The event orders prefix publication
        and consumers with that work; this transition does not wait for it.
        """
        request = scheduled_request.request
        total_depths = self.total_ut_steps
        start = scheduled_request.token_start
        end = start + scheduled_request.token_count
        if loop_count is None:
            loop_count = 0 if phase == RequestPhase.PREFILLING_PRELUDE else 1

        if phase == RequestPhase.PREFILLING_PRELUDE:
            request.prefill_reclaimable_tokens = start
            request.num_preluded_tokens = end
            depth = 0
        else:
            depth = scheduled_request.depth
        if loop_count <= 0:
            if phase != RequestPhase.PREFILLING_PRELUDE:
                raise ValueError("recurrent prefill progress requires at least one loop")
            self.enqueue(request, RequestPhase.PREFILLING_RECURRENT)
            return PrefillProgress(False, False)

        boundaries = request.prefill_depth_ranges
        if depth + loop_count > total_depths:
            raise ValueError("prefill execution range exceeds model loop depth")
        final_depth_reached = False
        for offset in range(loop_count):
            current_depth = depth + offset
            if phase == RequestPhase.PREFILLING_RECURRENT and (
                boundaries[current_depth][0] != start or boundaries[current_depth][1] < end
            ):
                raise RuntimeError("prefill execution range crosses an incomplete depth frontier")
            final_depth_reached = self._advance_prefill_depth_frontier(
                request,
                depth=current_depth,
                token_end=end,
            )

        chunk_complete = final_depth_reached and end == request.num_preluded_tokens
        if not final_depth_reached or not chunk_complete:
            self.enqueue(request, RequestPhase.PREFILLING_RECURRENT)
            return PrefillProgress(False, False)

        self.cache_manager.publish_prefix(
            request.request_id,
            request.prompt_token_ids,
            request.num_prefilled_tokens,
            event,
        )
        request.prefill_reclaimable_tokens = end
        prompt_complete = end == len(request.prompt_token_ids)
        if prompt_complete:
            request.loops_done = total_depths
            next_phase = self._prompt_destination(request)
        else:
            next_phase = RequestPhase.PREFILLING_PRELUDE
        self.enqueue(request, next_phase)
        return PrefillProgress(True, prompt_complete)

    def _apply_prefill_progress(
        self,
        scheduled_request: ScheduledRequest,
        phase: RequestPhase,
        *,
        loop_count: int | None = None,
        event=None,
    ) -> PrefillProgress:
        """One prompt frontier update for local and remotely consumed outputs."""
        if phase != RequestPhase.PREFILL:
            return self._apply_split_prefill_progress(
                scheduled_request,
                phase,
                loop_count=loop_count,
                event=event,
            )

        request = scheduled_request.request
        request.num_prefilled_tokens += scheduled_request.token_count
        final = request.num_prefilled_tokens == len(request.prompt_token_ids)
        self.cache_manager.publish_prefix(
            request.request_id, request.prompt_token_ids, request.num_prefilled_tokens, event
        )
        if final:
            request.loops_done = self.total_ut_steps
        self.enqueue(request, self._prompt_destination(request) if final else RequestPhase.PREFILL)
        return PrefillProgress(True, final)

    def _apply_wavefront_prefill_batch_progress(
        self,
        phase: RequestPhase,
        scheduled_requests: list[ScheduledRequest],
        *,
        loop_count: int = 1,
        event=None,
    ) -> tuple[ScheduledRequest, PrefillProgress]:
        """Apply all spans for one WCPB request before advancing its lifecycle."""
        if not scheduled_requests:
            raise ValueError("wavefront progress requires at least one span")
        scheduled_requests = sorted(scheduled_requests, key=lambda item: item.token_start)
        request = scheduled_requests[0].request
        if any(item.request is not request for item in scheduled_requests):
            raise ValueError("wavefront progress spans must share one request")
        if phase not in (RequestPhase.PREFILLING_PRELUDE, RequestPhase.PREFILLING_RECURRENT):
            raise ValueError(f"invalid wavefront prefill phase: {phase}")
        expected_loops = 0 if phase == RequestPhase.PREFILLING_PRELUDE else 1
        if loop_count != expected_loops:
            raise ValueError("wavefront prefill requires one recurrent depth per scheduling step")

        transfer_start = request.prefill_reclaimable_tokens
        if phase == RequestPhase.PREFILLING_PRELUDE:
            frontier = request.num_preluded_tokens
            for item in scheduled_requests:
                if item.token_start != frontier:
                    raise RuntimeError("wavefront prelude contains a prompt-range gap")
                frontier += item.token_count
            request.num_preluded_tokens = frontier
            progress = PrefillProgress(False, False, transfer_start)
            representative = scheduled_requests[0]
            next_phase = (
                RequestPhase.PREFILLING_PRELUDE
                if self._prelude_window_open(request)
                else RequestPhase.PREFILLING_RECURRENT
            )
        else:
            chunk_complete = False
            representative = scheduled_requests[0]
            for item in scheduled_requests:
                depth = item.depth
                start = item.token_start
                end = start + item.token_count
                if request.prefill_depth_ranges[depth][0] != start:
                    raise RuntimeError("stale wavefront recurrent depth frontier")
                if not self._advance_prefill_depth_frontier(
                    request,
                    depth=depth,
                    token_end=end,
                ):
                    continue

                chunk_end = min(
                    len(request.prompt_token_ids),
                    item.chunk_start + self.config.prefill_chunk_size,
                )
                if end == chunk_end:
                    if item.chunk_start != request.prefill_reclaimable_tokens:
                        raise RuntimeError("wavefront completed prompt chunks out of order")
                    request.prefill_reclaimable_tokens = chunk_end
                    chunk_complete = True
                    representative = item

            prompt_complete = request.num_prefilled_tokens == len(request.prompt_token_ids)
            progress = PrefillProgress(chunk_complete, prompt_complete, transfer_start)
            if chunk_complete:
                self.cache_manager.publish_prefix(
                    request.request_id,
                    request.prompt_token_ids,
                    request.num_prefilled_tokens,
                    event,
                )
            if prompt_complete:
                request.loops_done = self.total_ut_steps
                next_phase = self._prompt_destination(request)
            else:
                next_phase = (
                    RequestPhase.PREFILLING_PRELUDE
                    if self._prelude_window_open(request)
                    else RequestPhase.PREFILLING_RECURRENT
                )

        self.enqueue(request, next_phase)
        return representative, progress

    def _wavefront_prefill_groups(self, batch: SchedulerOutput):
        groups = {}
        if not self.wavefront_chunked_prefill:
            return groups
        for item in batch.requests:
            phase = batch.phase_for(item)
            request = item.request
            if (
                phase in (RequestPhase.PREFILLING_PRELUDE, RequestPhase.PREFILLING_RECURRENT)
                and self.requests.get(request.request_id) is request
                and request.phase != RequestPhase.FINISHED
            ):
                groups.setdefault((request.request_id, phase), []).append(item)
        return groups

    def _apply_prefill_phase_progress(
        self,
        scheduled_request: ScheduledRequest,
        phase: RequestPhase,
        wavefront_groups: dict[tuple[str, RequestPhase], list[ScheduledRequest]],
        update: SchedulerUpdate,
        *,
        loop_count: int,
        request_events: dict[str, object] | None = None,
        fallback_event: object | None = None,
    ) -> bool:
        """Apply prefill progress shared by submission and output updates.

        The caller filters stale or finished requests before calling this
        helper. Return ``True`` when prefill owns the row, including duplicate
        rows consumed by wavefront grouping; return ``False`` for phase-specific
        logic.
        """
        request = scheduled_request.request
        if self.wavefront_chunked_prefill and phase in (
            RequestPhase.PREFILLING_PRELUDE,
            RequestPhase.PREFILLING_RECURRENT,
        ):
            grouped = wavefront_groups.pop((request.request_id, phase), None)
            if grouped is not None:
                event = (
                    request_events.get(request.request_id)
                    if request_events is not None
                    else fallback_event
                )
                representative, progress = self._apply_wavefront_prefill_batch_progress(
                    phase, grouped, loop_count=loop_count, event=event
                )
                update.prefill.append((representative, phase, progress))
            return True
        if phase in (
            RequestPhase.PREFILL,
            RequestPhase.PREFILLING_PRELUDE,
            RequestPhase.PREFILLING_RECURRENT,
        ):
            event = (
                request_events.get(request.request_id)
                if request_events is not None
                else fallback_event
            )
            progress = self._apply_prefill_progress(
                scheduled_request,
                phase,
                loop_count=loop_count,
                event=event,
            )
            update.prefill.append((scheduled_request, phase, progress))
            return True
        return False

    def _should_exit(self, request: Request) -> bool:
        """Evaluate the policy after recording the completed loop's actual hazard."""
        params = request.sampling_params
        maximum = params.max_loops or self.total_ut_steps
        return request.loops_done >= maximum or (
            params.exit_threshold < 1.0
            and request.loops_done >= params.min_loops
            and 1.0 - request.remaining_probability >= params.exit_threshold
        )

    def _trace_exit(self, request: Request) -> bool:
        return request.loops_done >= request.exit_trace[request.num_scheduled_outputs]

    def _delayed_signal(self, request: Request, score: float, signal_depth: int) -> bool:
        """Consume each hazard once, with the existing policy's eligible depth."""
        params = request.sampling_params
        if self.exit_mode == "ouro_delayed":
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

    def needs_exit_signal(self, request: Request) -> bool:
        """Whether the next submitted loop needs its predecessor's readback."""
        maximum = request.sampling_params.max_loops or self.total_ut_steps
        return self.exit_mode != "trace" and request.loops_done + 1 < maximum

    def _record_token(self, request: Request, token: int, depth: int) -> bool:
        request.generated_token_ids.append(token)
        request.exit_depths.append(depth)
        params = request.sampling_params
        if token in self.eos_token_ids and not params.ignore_eos:
            request.finish_reason = FinishReason.STOP
        elif len(request.generated_token_ids) >= params.max_tokens:
            request.finish_reason = FinishReason.LENGTH
        return request.finish_reason is not None

    def update_from_output(
        self,
        batch: SchedulerOutput,
        result: ModelRunnerOutput | None = None,
        *,
        request_events: dict[str, object] | None = None,
        delivered: bool = False,
    ) -> SchedulerUpdate:
        """Apply completed sync work or delivered async results; never wait here."""
        if result is None:
            result = ModelRunnerOutput()
        update = SchedulerUpdate()
        wavefront_groups = self._wavefront_prefill_groups(batch)
        loop_count = self._batch_loop_count(batch)
        for row, scheduled_request in enumerate(batch.requests):
            request = scheduled_request.request
            # Ignore results for a request that is no longer registered: it may
            # have been cancelled, or its ID may now belong to a new Request.
            # Compare object identity so an old result cannot update the new request.
            if (
                self.requests.get(request.request_id) is not request
                or request.phase == RequestPhase.FINISHED
            ):
                continue
            phase = batch.phase_for(scheduled_request)
            if self._apply_prefill_phase_progress(
                scheduled_request,
                phase,
                wavefront_groups,
                update,
                loop_count=loop_count,
                request_events=request_events,
                fallback_event=result.completion,
            ):
                continue
            if phase == RequestPhase.PRELUDE:
                request.loops_done = 0
                request.remaining_probability = 1.0
                request.pending_exit_depth = None
                self.enqueue(request, RequestPhase.RECURRENT)
            elif phase == RequestPhase.RECURRENT:
                # A zero count is the executor's one-loop default for RECURRENT.
                request.loops_done += max(1, self._batch_loop_count(batch))
                if self.exit_mode == "trace":
                    should_exit = self._trace_exit(request)
                elif result.gate_scores is None:
                    # Fused fixed-depth loops deliver no hazard scores; the exit
                    # depth was decided before submission.
                    maximum = request.sampling_params.max_loops or self.total_ut_steps
                    should_exit = request.loops_done >= maximum
                else:
                    score = result.gate_scores[row]
                    if self.exit_mode == "ouro":
                        request.remaining_probability *= 1.0 - score
                        should_exit = self._should_exit(request)
                    else:
                        maximum = request.sampling_params.max_loops or self.total_ut_steps
                        should_exit = (
                            request.loops_done >= maximum
                            or request.pending_exit_depth == request.loops_done
                        )
                        if not should_exit and self._delayed_signal(
                            request, score, request.loops_done
                        ):
                            request.pending_exit_depth = request.loops_done + 1
                if should_exit:
                    update.exited.append(request)
                self.enqueue(request, RequestPhase.CODA if should_exit else RequestPhase.RECURRENT)
            elif phase == RequestPhase.CODA:
                if delivered:
                    if not result.output_indices or result.output_indices[row] != len(
                        request.generated_token_ids
                    ):
                        raise RuntimeError("out-of-order coda delivery")
                    if request.num_output_placeholders != 1:
                        raise RuntimeError("invalid pending output count")
                    request.num_output_placeholders -= 1
                depth = result.depths[row] if delivered else request.loops_done
                if self._record_token(request, result.token_ids[row], depth):
                    update.finished.append(request)
                elif not delivered:
                    self.enqueue(
                        request,
                        RequestPhase.SPECULATIVE
                        if self.speculative_config
                        else RequestPhase.PRELUDE,
                    )
                update.outputs.append(request)
            elif phase == RequestPhase.SPECULATIVE:
                spec = result.speculative[row]
                if result.frontiers is not None:
                    request.num_output_placeholders -= (
                        1
                        if self.speculative_config.mode == "pipeline"
                        else scheduled_request.token_count
                    )
                    request.num_speculative_rounds -= 1
                    if request.num_output_placeholders < 0 or request.num_speculative_rounds < 0:
                        raise RuntimeError("invalid speculative reservation count")
                    expected = len(request.prompt_token_ids) + len(request.generated_token_ids) - 1
                    if result.frontiers[row] != expected + len(spec.token_ids):
                        raise RuntimeError(
                            "speculative GPU frontier disagrees with committed prefix"
                        )
                emitted = 0
                for token in spec.token_ids:
                    emitted += 1
                    if self._record_token(request, token, self.speculative_config.target_loops):
                        update.finished.append(request)
                        break
                update.committed_tokens += emitted
                update.accepted_tokens += min(spec.accepted_count, emitted)
                if request.finish_reason is None:
                    # Async work may already be writing the next round: never truncate it.
                    if result.frontiers is None:
                        # The final correction/bonus token has not been forwarded.
                        self.cache_manager.truncate_suffix(
                            request.request_id,
                            spec.kv_frontier
                            if spec.kv_frontier is not None
                            else scheduled_request.token_start + emitted,
                        )
                        request.loops_done = 0
                    if request.request_id not in self.queues[RequestPhase.SPECULATIVE]:
                        self.enqueue(request, RequestPhase.SPECULATIVE)
                update.outputs.append(request)
        return update

    def update_after_submission(
        self,
        batch: SchedulerOutput,
        *,
        completion=None,
        request_events: dict[str, object] | None = None,
        signals: dict[int, ExitSignal] | None = None,
    ) -> SchedulerUpdate:
        """Advance logical submissions without requiring CPU token/score delivery."""
        update = SchedulerUpdate()
        wavefront_groups = self._wavefront_prefill_groups(batch)
        loop_count = self._batch_loop_count(batch)
        for row, scheduled_request in enumerate(batch.requests):
            request = scheduled_request.request
            if (
                self.requests.get(request.request_id) is not request
                or request.phase == RequestPhase.FINISHED
            ):
                continue
            phase = batch.phase_for(scheduled_request)
            if self._apply_prefill_phase_progress(
                scheduled_request,
                phase,
                wavefront_groups,
                update,
                loop_count=loop_count,
                request_events=request_events,
                fallback_event=completion,
            ):
                continue
            if phase == RequestPhase.PRELUDE:
                request.loops_done = 0
                request.remaining_probability = 1.0
                request.pending_exit_depth = None
                self.enqueue(request, RequestPhase.RECURRENT)
            elif phase == RequestPhase.CODA:
                request.num_output_placeholders += 1
                if request.num_scheduled_outputs < request.sampling_params.max_tokens:
                    update.next_prelude.append(row)
                    self.enqueue(request, RequestPhase.PRELUDE)
            elif phase == RequestPhase.SPECULATIVE:
                request.num_speculative_rounds += 1
                request.num_output_placeholders += (
                    1
                    if self.speculative_config.mode == "pipeline"
                    else scheduled_request.token_count
                )
                self.enqueue(request, RequestPhase.SPECULATIVE)
            elif phase == RequestPhase.RECURRENT:
                loop_count = max(1, self._batch_loop_count(batch))
                consume_signal = (
                    loop_count == 1
                    and request.loops_done > 0
                    and request.sampling_params.exit_threshold < 1
                    and self.needs_exit_signal(request)
                )
                request.loops_done += loop_count
                should_exit = (
                    self._trace_exit(request)
                    if self.exit_mode == "trace"
                    else request.loops_done
                    >= (request.sampling_params.max_loops or self.total_ut_steps)
                )
                signal = signals.get(row) if signals else None
                if consume_signal:
                    if (
                        signal is None
                        or signal.request is not request
                        or signal.position != request.position
                        or signal.depth != request.loops_done - 1
                    ):
                        raise RuntimeError("stale lookahead signal")
                    should_exit = self._delayed_signal(request, signal.score, signal.depth)
                if should_exit:
                    request.pending_exit_depth = request.loops_done
                    update.exited.append(request)
                elif (
                    self.exit_mode in ("ouro_delayed", "random_lookahead")
                    and request.sampling_params.exit_threshold < 1
                ):
                    update.retain_signal.append(row)
                self.enqueue(request, RequestPhase.CODA if should_exit else RequestPhase.RECURRENT)
        return update
