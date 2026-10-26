"""CPU scheduling at request-phase/loop boundaries, independent of model execution."""

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum, auto

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.core.execution import WorkKey
from vllm_rlt.core.scheduling_policy import create_scheduling_policy, requires_staged_prefill
from vllm_rlt.request import FinishReason, Request, RequestStatus
from vllm_rlt.worker.output import ExitSignal, ModelRunnerOutput

# Items of these works cover a contiguous span of prompt tokens, so their KV
# frontier is the span end rather than the request's decode position.
_SPAN_WORKS = frozenset(
    (
        WorkKey.PROMPT_ATOMIC,
        WorkKey.PROMPT_PRELUDE,
        WorkKey.PROMPT_RECURRENT,
        WorkKey.SPECULATIVE,
    )
)
_CAPACITY_WORKS = _SPAN_WORKS | {WorkKey.DECODE_PRELUDE, WorkKey.DECODE_RECURRENT}


@dataclass(frozen=True)
class ScheduledRequest:
    request: Request
    work: WorkKey
    # PREFILL and SPECULATIVE use a contiguous span; ordinary decode uses one row.
    token_start: int = 0
    token_count: int = 1
    # Recurrence position/depth are snapshotted for both prompt and generation.
    # The engine must not infer them from mutable request state after submission.
    depth: int | None = None
    chunk_start: int | None = None
    completes_prompt: bool = False
    # Row offset in the request's prefill hidden-state buffer. None lets the runner
    # derive token_start - chunk_start.
    prefill_hidden_row_offset: int | None = None


@dataclass(frozen=True)
class SchedulerOutput:
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
    def works(self) -> frozenset[WorkKey]:
        return frozenset(item.work for item in self.requests)

    @property
    def is_mixed(self) -> bool:
        return len(self.works) > 1

    @property
    def has_split_prefill_work(self) -> bool:
        return any(
            item.work in (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
            for item in self.requests
        )

    @property
    def is_speculative(self) -> bool:
        """Execution-mode flag for this batch, not the engine-wide configuration."""
        return bool(self.requests) and all(
            item.work == WorkKey.SPECULATIVE for item in self.requests
        )

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
    prefill: list[tuple[ScheduledRequest, WorkKey, PrefillProgress]] = field(default_factory=list)
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
        self.waiting: deque[str] = deque()
        self.work_queues: dict[WorkKey, deque[str]] = {work: deque() for work in WorkKey}
        self.admission_slots: set[str] = set()
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
        request.status = RequestStatus.WAITING
        self.requests[request.request_id] = request
        self.waiting.append(request.request_id)

    def mark_receiving(self, request: Request):
        if self.requests.get(request.request_id) is not request:
            raise ValueError("only the registered request can enter receiving")
        if request.status != RequestStatus.WAITING:
            raise ValueError("only a waiting request can enter receiving")
        self._remove_queued_request(request)
        request.status = RequestStatus.RECEIVING

    def queue_for_admission(self, request: Request):
        if self.requests.get(request.request_id) is not request:
            raise ValueError("only the registered request can wait for admission")
        if request.status not in (RequestStatus.WAITING, RequestStatus.RECEIVING):
            raise ValueError("only waiting or receiving requests can await admission")
        if request.request_id not in self.waiting:
            self.waiting.append(request.request_id)

    def activate_request(self, request: Request):
        """Mark a resource-ready request active and reserve its compute slot."""
        if self.requests.get(request.request_id) is not request:
            raise ValueError("only the registered request can be activated")
        if request.status not in (RequestStatus.WAITING, RequestStatus.RECEIVING):
            raise ValueError("only waiting or receiving requests can be activated")
        if not self.has_admission_capacity():
            raise RuntimeError("no admission slot is available")
        request.status = RequestStatus.ACTIVE
        self.admission_slots.add(request.request_id)

    def has_admission_capacity(self) -> bool:
        return len(self.admission_slots) < self.config.max_num_seqs

    def has_admission_slot(self, request: Request) -> bool:
        return (
            self.requests.get(request.request_id) is request
            and request.request_id in self.admission_slots
        )

    def release_admission_slot(self, request: Request):
        if self.requests.get(request.request_id) is not request:
            return False
        self.admission_slots.discard(request.request_id)
        return True

    def enqueue_work(self, request: Request, work: WorkKey):
        if self.requests.get(request.request_id) is not request:
            raise ValueError("only the registered request can enter a work queue")
        if request.status != RequestStatus.ACTIVE:
            raise ValueError("only active requests can enter a work queue")
        if self.pending_work(request) is not None:
            raise ValueError(f"request {request.request_id!r} already has pending work")
        self.work_queues[work].append(request.request_id)

    def pending_work(self, request: Request) -> WorkKey | None:
        for work, queue in self.work_queues.items():
            if request.request_id in queue:
                return work
        return None

    def remove_pending_work(self, request: Request) -> WorkKey | None:
        pending = None
        for work, queue in self.work_queues.items():
            if request.request_id in queue:
                pending = work if pending is None else pending
                while request.request_id in queue:
                    queue.remove(request.request_id)
        return pending

    def _remove_queued_request(self, request: Request):
        while request.request_id in self.waiting:
            self.waiting.remove(request.request_id)
        self.remove_pending_work(request)

    def park_waiting(self, request: Request):
        self._remove_queued_request(request)
        self.release_admission_slot(request)
        request.status = RequestStatus.WAITING
        self.waiting.append(request.request_id)

    def suspend_for_transfer(self, request):
        """Stop scheduling without freeing resources that a transfer still borrows."""
        if self.requests.get(request.request_id) is not request:
            return False
        self._remove_queued_request(request)
        request.status = RequestStatus.TRANSFERRING
        return True

    def finish(self, request: Request, reason: FinishReason):
        self.mark_finished(request, reason)
        self.release_admission_slot(request)
        self.cache_manager.free(request.request_id)
        request.hidden_state = None
        request.input_token_tensor = None
        request.num_output_placeholders = 0
        request.num_speculative_rounds = 0
        self.requests.pop(request.request_id, None)

    def mark_finished(self, request: Request, reason: FinishReason):
        """Mark logical completion while a runner may still own device resources."""
        self._remove_queued_request(request)
        request.status = RequestStatus.FINISHED
        request.finish_reason = reason

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
        waiting = self.waiting
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
            or request.status == RequestStatus.RECEIVING
            or (
                request.status == RequestStatus.WAITING
                and (priority_only or not self.cache_manager.has_allocation(request.request_id))
            )
            or request.status == RequestStatus.TRANSFERRING
            or request.status == RequestStatus.FINISHED
            or (
                request.status == RequestStatus.ACTIVE
                and (
                    request.request_id not in self.admission_slots
                    or self.pending_work(request) is None
                )
            )
            or request.num_output_placeholders
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
            if (not waiting_only or request.status == RequestStatus.WAITING)
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
        self.park_waiting(victim)
        return True

    def _ensure_admission_slot(self, request: Request) -> bool:
        """Make capacity for one admission, preempting only when policy allows."""
        if self.has_admission_capacity():
            return True
        preempted = (
            self.config.policy == "priority"
            and self.suspend_callback is not None
            and self.preempt(request, priority_only=True, excluded=frozenset())
        )
        return bool(preempted and self.has_admission_capacity())

    def _try_resume(self, request: Request) -> ResumeResult:
        """Translate the existing restoration callback's three-way result.

        None: no snapshot, continue fresh admission.
        WorkKey: resources/state restored; enqueue the saved work.
        False: snapshot exists, but restoration cannot obtain capacity yet.
        """
        restored = self.resume_callback(request) if self.resume_callback is not None else None
        while restored is False and self.preempt(request, waiting_only=True):
            restored = self.resume_callback(request)
        if restored is None:
            return ResumeResult.NOT_PREEMPTED
        if restored is False:
            return ResumeResult.BLOCKED
        self.activate_request(request)
        self.enqueue_work(request, restored)
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
            # Progress, not queue state, remains valid while a prompt ticket is in flight.
            active_prompt = (
                request.status == RequestStatus.ACTIVE
                and request.num_prefilled_tokens < len(request.prompt_token_ids)
            )
            has_allocation = cache.has_allocation(request.request_id)
            if not active_prompt and not (reserve_outputs and has_allocation):
                continue
            tokens = len(request.prompt_token_ids)
            if reserve_outputs:
                tokens += request.sampling_params.max_tokens - 1
            allocated = cache.allocated_blocks(request.request_id)
            reserved += max(0, cache.required_blocks(tokens) - allocated)
        return reserved

    def _plan_admission(
        self, request: Request, *, has_allocated_requests: bool | None = None
    ) -> AdmissionPlan:
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
        has_occupants = (
            bool(self.admission_slots) if has_allocated_requests is None else has_allocated_requests
        )
        admission_headroom_blocks = cache.watermark_blocks if has_occupants else 0
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
        plan = self._plan_admission(request, has_allocated_requests=occupied)
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
        self.activate_request(request)
        if self.staged_prefill:
            self.enqueue_work(request, WorkKey.PROMPT_PRELUDE)
        else:
            self.enqueue_work(request, WorkKey.PROMPT_ATOMIC)

    def _admit(self):
        """Admit waiting requests: order -> slot -> restore/allocate -> defer/enter.

        A blocked long request may let a short one pass. Each successful fresh
        admission increments earlier blocked requests' bypass counts; reaching
        the limit stops further bypasses so active requests can drain.
        """
        waiting = self._order_waiting_requests()
        deferred = []
        scan_count = min(len(waiting), self.config.admission_scan_limit)
        for _ in range(scan_count):
            if not waiting:
                break
            # Peek before popping: preemption may append its victim to WAITING.
            request = self.requests[waiting[0]]
            if not self._ensure_admission_slot(request):
                break
            request_id = waiting.popleft()
            request = self.requests[request_id]

            if request.status == RequestStatus.RECEIVING:
                # Transfer completion has already made KV and hidden valid.
                # Imported and local requests compete in this same admission queue.
                self.cache_manager.unpin_transfer(request_id, request_id)
                self.activate_request(request)
                self.enqueue_work(request, WorkKey.DECODE_CODA)
            elif self.cache_manager.has_allocation(
                request_id
            ) and self.cache_manager.has_transfer_lease(request_id):
                # Outgoing transfer reservation already allocated the prompt KV.
                self._enter_prefill(request)
            else:
                resume_result = self._try_resume(request)
                if resume_result == ResumeResult.RESTORED:
                    continue
                if resume_result == ResumeResult.BLOCKED:
                    deferred.append(request_id)
                    continue

                plan = self._plan_admission(request)
                admitted = self._commit_admission(request, plan)
                while not admitted and self.preempt(request, waiting_only=True):
                    # Recompute prefix claims and growth reservations after
                    # releasing a resident allocation; the previous plan is stale.
                    plan = self._plan_admission(request)
                    admitted = self._commit_admission(request, plan)
                if not admitted:
                    deferred.append(request_id)
                    if request.admission_bypasses >= self.config.max_admission_bypasses:
                        break
                    continue

            for blocked_id in deferred:
                self.requests[blocked_id].admission_bypasses += 1
            if any(
                self.requests[rid].admission_bypasses >= self.config.max_admission_bypasses
                for rid in deferred
            ):
                break
        # [A, B] deferred must precede the untouched tail in the same order.
        waiting.extendleft(reversed(deferred))

    def _make_scheduled_work(
        self, request: Request, work: WorkKey, token_budget: int
    ) -> ScheduledRequest:
        """Snapshot one work item's execution range without advancing progress.

        Prompt length 10, prefilled 4, chunk 4, budget 6 -> range [4, 8).
        Decode work contributes one position; recurrence depth varies per request.
        """
        if work == WorkKey.PROMPT_ATOMIC:
            start = request.num_prefilled_tokens
            count = min(
                token_budget, self.config.prefill_chunk_size, len(request.prompt_token_ids) - start
            )
            return ScheduledRequest(request, work, start, count)
        if work == WorkKey.PROMPT_PRELUDE:
            start = request.num_preluded_tokens
            count = min(
                token_budget,
                self.config.prefill_chunk_size,
                len(request.prompt_token_ids) - start,
            )
            return ScheduledRequest(request, work, start, count, chunk_start=start)
        if work == WorkKey.PROMPT_RECURRENT:
            start = request.num_prefilled_tokens
            if start >= request.num_preluded_tokens:
                return ScheduledRequest(
                    request, work, start, 0, depth=0, chunk_start=request.prefill_reclaimable_tokens
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
                work,
                start,
                count,
                depth=first_token_depth,
                chunk_start=request.prefill_reclaimable_tokens,
                completes_prompt=(
                    first_token_depth == len(request.prefill_depth_ranges) - 1
                    and start + count == len(request.prompt_token_ids)
                ),
            )
        if work == WorkKey.SPECULATIVE:
            # K candidates plus one bonus distribution. At the output limit,
            # K=0 is an ordinary fixed-depth step and needs no extra KV slot.
            remaining = request.sampling_params.max_tokens - request.num_scheduled_outputs
            count = min(self.speculative_config.num_speculative_tokens + 1, remaining, token_budget)
            return ScheduledRequest(request, work, request.position, count)
        if work == WorkKey.DECODE_RECURRENT:
            return ScheduledRequest(request, work, request.position, depth=request.loops_done)
        return ScheduledRequest(request, work)

    def _make_wavefront_task(
        self, request, work, token_budget, *, start, depth=None
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

        if work == WorkKey.PROMPT_PRELUDE:
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
            work,
            start,
            count,
            depth=depth,
            chunk_start=chunk_start,
            completes_prompt=(
                work == WorkKey.PROMPT_RECURRENT
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

    def has_ready_work(self, work: WorkKey):
        """A blocked transport queue must not hide another runnable work type."""
        queue = self.work_queues[work]
        if self.execution_ready is None:
            return bool(queue)
        return any(self.execution_ready(self.requests[rid]) for rid in queue)

    def has_ready_works(self, *works: WorkKey) -> bool:
        return any(self.has_ready_work(work) for work in works)

    def take_works(self, works, *, eligible=None):
        """Select compatible requests across queues under one budget/reservation set."""
        eligible = eligible if eligible is not None else self.execution_ready
        token_budget = self.config.max_num_batched_tokens
        scheduled_requests = []
        selected: set[str] = set()
        for work in works:
            queue = self.work_queues[work]
            if self.wavefront_chunked_prefill and work in (
                WorkKey.PROMPT_PRELUDE,
                WorkKey.PROMPT_RECURRENT,
            ):
                token_budget = self._take_wavefront_work(
                    work,
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
                if request.request_id in selected:
                    continue
                if eligible is not None and not eligible(request):
                    queue.append(request.request_id)
                    continue
                if work == WorkKey.PROMPT_PRELUDE and not self._prelude_window_open(request):
                    queue.append(request.request_id)
                    continue
                if work == WorkKey.SPECULATIVE and (
                    request.num_speculative_rounds >= 2
                    or request.num_scheduled_outputs >= request.sampling_params.max_tokens
                ):
                    queue.append(request.request_id)
                    continue
                scheduled_request = self._make_scheduled_work(request, work, token_budget)
                if (
                    work in (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
                    and scheduled_request.token_count <= 0
                ):
                    queue.append(request.request_id)
                    continue
                if work in _CAPACITY_WORKS:
                    # A scheduled span declares a prompt frontier; decode phases advance the
                    # request's own position by one.
                    frontier = (
                        scheduled_request.token_start + scheduled_request.token_count
                        if work in _SPAN_WORKS
                        else request.position + 1
                    )
                    if not self._ensure_execution_capacity(request, frontier, selected=selected):
                        queue.append(request.request_id)
                        continue
                # Selecting A must prevent B's later capacity check from evicting A.
                selected.add(request.request_id)
                scheduled_requests.append(scheduled_request)
                token_budget -= scheduled_request.token_count
        if not scheduled_requests:
            return None
        # Fairness follows the requests actually selected, including a mixed attempt
        # whose other queue could not reserve capacity.
        loop_count = self._planned_loop_count(scheduled_requests)
        scheduled_requests = [
            replace(
                item,
                completes_prompt=self._scheduled_prefill_completes(item, loop_count)
                if item.work in (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
                else item.completes_prompt,
            )
            for item in scheduled_requests
        ]
        self.policy.record_batch(item.work for item in scheduled_requests)
        return SchedulerOutput(scheduled_requests, loop_count)

    def _planned_loop_count(self, scheduled_requests):
        works = {item.work for item in scheduled_requests}
        if works == {WorkKey.PROMPT_ATOMIC}:
            return self.total_ut_steps
        if WorkKey.SPECULATIVE in works or WorkKey.DECODE_CODA in works:
            return 0
        if WorkKey.DECODE_PRELUDE in works and WorkKey.PROMPT_PRELUDE in works:
            return 0
        if WorkKey.DECODE_PRELUDE in works and not works.intersection(
            (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
        ):
            return 0
        if self.wavefront_chunked_prefill:
            return int(WorkKey.PROMPT_RECURRENT in works or WorkKey.DECODE_RECURRENT in works)
        if WorkKey.PROMPT_PRELUDE in works:
            # In mixed mode, stage just the embeddings when recurrent decode work
            # is ready so the next depth can join that batch. Without a reason to
            # yield, PRELUDE can execute the full prompt depth in this submission.
            if self.config.mode == "mixed" and self.has_ready_work(WorkKey.DECODE_RECURRENT):
                return 0
            return self.total_ut_steps
        if WorkKey.PROMPT_RECURRENT in works:
            if self.config.mode == "mixed" and (
                WorkKey.DECODE_RECURRENT in works or self.has_ready_work(WorkKey.DECODE_RECURRENT)
            ):
                return 1
            return min(self._max_contiguous_prefill_loops(item) for item in scheduled_requests)
        if WorkKey.DECODE_RECURRENT in works:
            return 1
        return 0

    def _batch_loop_count(self, batch):
        if batch.loop_count is not None:
            return batch.loop_count
        works = batch.works
        if WorkKey.PROMPT_ATOMIC in works:
            return self.total_ut_steps
        if works.intersection(
            (
                WorkKey.PROMPT_PRELUDE,
                WorkKey.DECODE_PRELUDE,
                WorkKey.DECODE_CODA,
                WorkKey.SPECULATIVE,
            )
        ):
            if WorkKey.PROMPT_RECURRENT in works or WorkKey.DECODE_RECURRENT in works:
                return 1
            return 0
        if works.intersection((WorkKey.PROMPT_RECURRENT, WorkKey.DECODE_RECURRENT)):
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
        work = scheduled_request.work
        depth = scheduled_request.depth if work == WorkKey.PROMPT_RECURRENT else 0
        return (
            loop_count > 0
            and depth + loop_count == self.total_ut_steps
            and scheduled_request.token_start + scheduled_request.token_count
            == len(scheduled_request.request.prompt_token_ids)
        )

    def _take_wavefront_work(
        self, work, queue, token_budget, scheduled_requests, selected, eligible
    ):
        """Round-robin each request's chunks/depths into the shared token budget."""
        remaining = len(queue)
        candidates = []
        owner_limit = self.config.max_num_seqs - len(selected)
        while queue and remaining and len(candidates) < owner_limit and token_budget:
            remaining -= 1
            request = self.requests[queue.popleft()]
            queue.append(request.request_id)
            if eligible is not None and not eligible(request):
                continue
            if work == WorkKey.PROMPT_PRELUDE and not self._prelude_window_open(request):
                continue
            if work == WorkKey.PROMPT_RECURRENT and not any(
                start < min(end, request.num_preluded_tokens)
                for start, end in request.prefill_depth_ranges
            ):
                continue
            candidates.append(request)

        if not candidates:
            return token_budget

        added = set()
        blocked = set()
        if work == WorkKey.PROMPT_PRELUDE:
            cursors = {request.request_id: request.num_preluded_tokens for request in candidates}
            for _ in range(len(candidates[0].prefill_depth_ranges)):
                for request in candidates:
                    request_id = request.request_id
                    if token_budget <= 0:
                        break
                    if request_id in blocked or request.status != RequestStatus.ACTIVE:
                        continue
                    start = cursors[request_id]
                    if start >= len(request.prompt_token_ids):
                        continue
                    scheduled_request = self._make_wavefront_task(
                        request, work, token_budget, start=start
                    )
                    if scheduled_request is None:
                        continue
                    frontier = scheduled_request.token_start + scheduled_request.token_count
                    if not self._ensure_execution_capacity(request, frontier, selected=selected):
                        blocked.add(request_id)
                        continue
                    scheduled_requests.append(scheduled_request)
                    selected.add(request_id)
                    added.add(request_id)
                    cursors[request_id] += scheduled_request.token_count
                    token_budget -= scheduled_request.token_count
        else:
            tasks = []
            for depth in range(len(candidates[0].prefill_depth_ranges)):
                for owner_index, request in enumerate(candidates):
                    request_id = request.request_id
                    if token_budget <= 0:
                        break
                    if request_id in blocked or request.status != RequestStatus.ACTIVE:
                        continue
                    start, end = request.prefill_depth_ranges[depth]
                    end = min(end, request.num_preluded_tokens)
                    if start >= end:
                        continue
                    scheduled_request = self._make_wavefront_task(
                        request, work, token_budget, depth=depth, start=start
                    )
                    if scheduled_request is None:
                        continue
                    frontier = scheduled_request.token_start + scheduled_request.token_count
                    if not self._ensure_execution_capacity(request, frontier, selected=selected):
                        blocked.add(request_id)
                        continue
                    tasks.append((owner_index, scheduled_request))
                    selected.add(request_id)
                    added.add(request_id)
                    token_budget -= scheduled_request.token_count

            tasks.sort(key=lambda item: (item[0], item[1].token_start))
            scheduled_requests.extend(scheduled_request for _, scheduled_request in tasks)

        for request in candidates:
            # Growing another candidate may have suspended this request and put
            # it in WAITING. It must not also return to its old prompt queue.
            if request.request_id in added or request.status != RequestStatus.ACTIVE:
                while request.request_id in queue:
                    queue.remove(request.request_id)
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

    def _prompt_destination(self, request: Request):
        exporting = self.exports_request is not None and self.exports_request(request)
        if exporting:
            self.suspend_for_transfer(request)
        else:
            self.enqueue_work(request, WorkKey.DECODE_CODA)

    def _apply_split_prefill_progress(
        self,
        request: Request,
        work: WorkKey,
        scheduled_request: ScheduledRequest,
        *,
        loop_count: int | None = None,
        event=None,
    ) -> PrefillProgress:
        """Advance host-side logical progress for a submitted prefill request.

        The device work may still be pending. The event orders prefix publication
        and consumers with that work; this transition does not wait for it.
        """
        total_depths = self.total_ut_steps
        start = scheduled_request.token_start
        end = start + scheduled_request.token_count
        if loop_count is None:
            loop_count = 0 if work == WorkKey.PROMPT_PRELUDE else 1
        if work == WorkKey.PROMPT_PRELUDE:
            request.prefill_reclaimable_tokens = min(request.prefill_reclaimable_tokens, start)
            request.num_preluded_tokens = end
            depth = 0
        else:
            depth = scheduled_request.depth
        if loop_count <= 0:
            if work != WorkKey.PROMPT_PRELUDE:
                raise ValueError("recurrent prefill progress requires at least one loop")
            self.enqueue_work(request, WorkKey.PROMPT_RECURRENT)
            return PrefillProgress(False, False)
        boundaries = request.prefill_depth_ranges
        if depth + loop_count > total_depths:
            raise ValueError("prefill execution range exceeds model loop depth")
        for offset in range(loop_count):
            current_depth = depth + offset
            if work == WorkKey.PROMPT_RECURRENT and (
                boundaries[current_depth][0] != start or boundaries[current_depth][1] < end
            ):
                raise RuntimeError("prefill execution range crosses an incomplete depth frontier")
            boundaries[current_depth][0] = end
            if current_depth + 1 < total_depths:
                boundaries[current_depth + 1][1] = end
            else:
                request.num_prefilled_tokens = end

        final_depth_reached = depth + loop_count == total_depths
        chunk_complete = final_depth_reached and end == request.num_preluded_tokens
        if not final_depth_reached or not chunk_complete:
            self.enqueue_work(request, WorkKey.PROMPT_RECURRENT)
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
            self._prompt_destination(request)
        else:
            self.enqueue_work(request, WorkKey.PROMPT_PRELUDE)
        return PrefillProgress(True, prompt_complete)

    def _apply_prefill_progress(
        self, scheduled_request: ScheduledRequest, work: WorkKey, *, loop_count=None, event=None
    ) -> PrefillProgress:
        """One prompt frontier update for local and remotely consumed outputs."""
        request = scheduled_request.request
        if work == WorkKey.PROMPT_ATOMIC:
            request.num_prefilled_tokens += scheduled_request.token_count
            final = request.num_prefilled_tokens == len(request.prompt_token_ids)
            self.cache_manager.publish_prefix(
                request.request_id, request.prompt_token_ids, request.num_prefilled_tokens, event
            )
            if final:
                request.loops_done = self.total_ut_steps
                self._prompt_destination(request)
            else:
                self.enqueue_work(request, WorkKey.PROMPT_ATOMIC)
            return PrefillProgress(True, final)
        return self._apply_split_prefill_progress(
            request, work, scheduled_request, loop_count=loop_count, event=event
        )

    def _apply_wavefront_prefill_batch_progress(
        self,
        work: WorkKey,
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
        if work not in (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT):
            raise ValueError(f"invalid wavefront prefill work: {work}")
        expected_loops = 0 if work == WorkKey.PROMPT_PRELUDE else 1
        if loop_count != expected_loops:
            raise ValueError("wavefront prefill requires one recurrent depth per scheduling step")

        transfer_start = request.prefill_reclaimable_tokens
        if work == WorkKey.PROMPT_PRELUDE:
            frontier = request.num_preluded_tokens
            for item in scheduled_requests:
                if item.token_start != frontier:
                    raise RuntimeError("wavefront prelude contains a prompt-range gap")
                frontier += item.token_count
            request.num_preluded_tokens = frontier
            progress = PrefillProgress(False, False, transfer_start)
            representative = scheduled_requests[0]
            next_work = (
                WorkKey.PROMPT_PRELUDE
                if self._prelude_window_open(request)
                else WorkKey.PROMPT_RECURRENT
            )
        else:
            chunk_complete = False
            representative = scheduled_requests[0]
            for item in scheduled_requests:
                depth = item.depth
                start = item.token_start
                end = start + item.token_count
                boundaries = request.prefill_depth_ranges
                if boundaries[depth][0] != start:
                    raise RuntimeError("stale wavefront recurrent depth frontier")
                boundaries[depth][0] = end
                if depth + 1 < self.total_ut_steps:
                    boundaries[depth + 1][1] = end
                    continue

                request.num_prefilled_tokens = end
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
                next_work = self._prompt_destination(request)
            else:
                next_work = (
                    WorkKey.PROMPT_PRELUDE
                    if self._prelude_window_open(request)
                    else WorkKey.PROMPT_RECURRENT
                )

        if next_work is not None and request.status == RequestStatus.ACTIVE:
            self.enqueue_work(request, next_work)
        return representative, progress

    def _wavefront_prefill_groups(self, batch: SchedulerOutput):
        groups = {}
        if not self.wavefront_chunked_prefill:
            return groups
        for item in batch.requests:
            work = item.work
            request = item.request
            if (
                work in (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
                and self.requests.get(request.request_id) is request
                and request.status != RequestStatus.FINISHED
            ):
                groups.setdefault((request.request_id, work), []).append(item)
        return groups

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
        for row, scheduled_request in enumerate(batch.requests):
            request = scheduled_request.request
            # Ignore results for a request that is no longer registered: it may
            # have been cancelled, or its ID may now belong to a new Request.
            # Compare object identity so an old result cannot update the new request.
            if (
                self.requests.get(request.request_id) is not request
                or request.status == RequestStatus.FINISHED
            ):
                continue
            work = scheduled_request.work
            if self.wavefront_chunked_prefill and work in (
                WorkKey.PROMPT_PRELUDE,
                WorkKey.PROMPT_RECURRENT,
            ):
                grouped = wavefront_groups.pop((request.request_id, work), None)
                if grouped is not None:
                    event = (
                        request_events.get(request.request_id)
                        if request_events is not None
                        else result.completion
                    )
                    representative, progress = self._apply_wavefront_prefill_batch_progress(
                        work, grouped, loop_count=self._batch_loop_count(batch), event=event
                    )
                    update.prefill.append((representative, work, progress))
                continue
            if work in (
                WorkKey.PROMPT_ATOMIC,
                WorkKey.PROMPT_PRELUDE,
                WorkKey.PROMPT_RECURRENT,
            ):
                event = (
                    request_events.get(request.request_id)
                    if request_events is not None
                    else result.completion
                )
                progress = self._apply_prefill_progress(
                    scheduled_request, work, loop_count=self._batch_loop_count(batch), event=event
                )
                update.prefill.append((scheduled_request, work, progress))
            elif work == WorkKey.DECODE_PRELUDE:
                request.loops_done = 0
                request.remaining_probability = 1.0
                request.pending_exit_depth = None
                self.enqueue_work(request, WorkKey.DECODE_RECURRENT)
            elif work == WorkKey.DECODE_RECURRENT:
                request.loops_done += 1
                if self.exit_mode == "trace":
                    should_exit = self._trace_exit(request)
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
                    self.enqueue_work(request, WorkKey.DECODE_CODA)
                else:
                    self.enqueue_work(request, WorkKey.DECODE_RECURRENT)
            elif work == WorkKey.DECODE_CODA:
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
                    next_work = (
                        WorkKey.SPECULATIVE if self.speculative_config else WorkKey.DECODE_PRELUDE
                    )
                    self.enqueue_work(request, next_work)
                update.outputs.append(request)
            elif work == WorkKey.SPECULATIVE:
                spec = result.speculative[row]
                if result.frontiers is not None:
                    request.num_output_placeholders -= scheduled_request.token_count
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
                            request.request_id, scheduled_request.token_start + emitted
                        )
                        request.loops_done = 0
                    if self.pending_work(request) is None:
                        self.enqueue_work(request, WorkKey.SPECULATIVE)
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
        for row, scheduled_request in enumerate(batch.requests):
            request = scheduled_request.request
            if (
                self.requests.get(request.request_id) is not request
                or request.status == RequestStatus.FINISHED
            ):
                continue
            work = scheduled_request.work
            if self.wavefront_chunked_prefill and work in (
                WorkKey.PROMPT_PRELUDE,
                WorkKey.PROMPT_RECURRENT,
            ):
                grouped = wavefront_groups.pop((request.request_id, work), None)
                if grouped is not None:
                    event = (
                        request_events.get(request.request_id)
                        if request_events is not None
                        else completion
                    )
                    representative, progress = self._apply_wavefront_prefill_batch_progress(
                        work, grouped, loop_count=self._batch_loop_count(batch), event=event
                    )
                    update.prefill.append((representative, work, progress))
                continue
            if work in (
                WorkKey.PROMPT_ATOMIC,
                WorkKey.PROMPT_PRELUDE,
                WorkKey.PROMPT_RECURRENT,
            ):
                event = (
                    request_events.get(request.request_id)
                    if request_events is not None
                    else completion
                )
                progress = self._apply_prefill_progress(
                    scheduled_request, work, loop_count=self._batch_loop_count(batch), event=event
                )
                update.prefill.append((scheduled_request, work, progress))
            elif work == WorkKey.DECODE_PRELUDE:
                request.loops_done = 0
                request.remaining_probability = 1.0
                request.pending_exit_depth = None
                self.enqueue_work(request, WorkKey.DECODE_RECURRENT)
            elif work == WorkKey.DECODE_CODA:
                request.num_output_placeholders += 1
                if request.num_scheduled_outputs < request.sampling_params.max_tokens:
                    update.next_prelude.append(row)
                    self.enqueue_work(request, WorkKey.DECODE_PRELUDE)
            elif work == WorkKey.SPECULATIVE:
                request.num_output_placeholders += scheduled_request.token_count
                request.num_speculative_rounds += 1
                self.enqueue_work(request, WorkKey.SPECULATIVE)
            elif work == WorkKey.DECODE_RECURRENT:
                consume_signal = (
                    request.loops_done > 0
                    and request.sampling_params.exit_threshold < 1
                    and self.needs_exit_signal(request)
                )
                request.loops_done += 1
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
                self.enqueue_work(
                    request,
                    WorkKey.DECODE_CODA if should_exit else WorkKey.DECODE_RECURRENT,
                )
        return update
