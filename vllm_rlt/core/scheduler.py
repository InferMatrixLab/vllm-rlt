"""CPU scheduling at request-phase/loop boundaries, independent of model execution."""

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum, auto

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.core.execution import WorkKey
from vllm_rlt.core.scheduling_policy import create_scheduling_policy
from vllm_rlt.request import FinishReason, Request, RequestStatus

# Items of these phases cover a contiguous span of prompt tokens, so their KV
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


@dataclass(frozen=True)
class SchedulerOutput:
    requests: list[ScheduledRequest]

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
        atomic_prefill: bool = True,
        *,
        produces_tokens: bool = True,
    ):
        self.config = config
        self.cache_manager = cache_manager
        self.speculative_config = speculative_config
        self.requests: dict[str, Request] = {}
        self.waiting: deque[str] = deque()
        self.work_queues: dict[WorkKey, deque[str]] = {work: deque() for work in WorkKey}
        self.admission_slots: set[str] = set()
        # Bound by Engine when preemption is enabled. These callbacks can copy
        # device state; queue transitions remain owned by this scheduler.
        self.suspend_callback = None
        self.resume_callback = None
        # Optional connector readiness predicate. Never mutates queues or blocks.
        self.execution_ready = None
        # One scheduling boundary per chunk (atomic) or per recurrence depth (split).
        # The policy is derived from the same flag so the two cannot disagree.
        self.atomic_prefill = atomic_prefill
        self.policy = create_scheduling_policy(
            config, speculative_config, atomic_prefill, produces_tokens=produces_tokens
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
            or request.status != RequestStatus.ACTIVE
            or request.request_id not in self.admission_slots
            or self.pending_work(request) is None
            or request.num_output_placeholders
            or request.request_id in excluded
        ):
            return False
        if self.cache_manager.has_transfer_lease(request.request_id):
            return False
        return not priority_only or (
            request.sampling_params.priority > requesting_request.sampling_params.priority
        )

    def _select_preemption_victim(self, requesting_request, *, priority_only, excluded):
        """Prefer lower priority, then a larger current-position KV footprint.

        Priority 10 loses to priority 0. The footprint tie-breaker uses
        required_blocks(position + 1), not exact reclaimable physical pages.
        Capacity-pressure preemption retains this ranking even in FCFS mode;
        priority_only restricts eligibility only for priority admission.
        """
        candidates = (
            request
            for request in self.requests.values()
            if self._is_preemption_candidate(
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

    def preempt(self, requesting_request, *, priority_only=False, excluded=frozenset()):
        """Select and suspend one safe victim, then move it back to admission."""
        if self.suspend_callback is None:
            return False
        victim = self._select_preemption_victim(
            requesting_request, priority_only=priority_only, excluded=excluded
        )
        if victim is None:
            return False
        self.suspend_callback(victim)
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
            bool(self.admission_slots)
            if has_allocated_requests is None
            else has_allocated_requests
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
        request.prefill_chunk_start = plan.cached_tokens
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
        if self.atomic_prefill:
            self.enqueue_work(request, WorkKey.PROMPT_ATOMIC)
        else:
            self.enqueue_work(request, WorkKey.PROMPT_PRELUDE)

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
                if not self._commit_admission(request, plan):
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
                    request, work, start, 0, depth=0, chunk_start=request.prefill_chunk_start
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
                chunk_start=request.prefill_chunk_start,
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

    def _ensure_execution_capacity(
        self, request: Request, frontier: int, *, selected: set[str]
    ) -> bool:
        """Grow KV for this step; if needed, preempt one other request and retry.

        frontier is an exclusive token count, not a loop count. Another loop at
        the same token position normally uses existing capacity. selected holds
        the requests already chosen for this batch; preemption must not suspend
        them, so it is handed a frozen snapshot of that batch-local selection
        instead of reading shared Scheduler state.
        """
        cache = self.cache_manager
        if cache.ensure_capacity(request.request_id, frontier):
            return True
        if not self.preempt(request, excluded=frozenset(selected)):
            return False
        return cache.ensure_capacity(request.request_id, frontier)

    def _prelude_window_open(self, request: Request) -> bool:
        """Whether the next prompt chunk may be staged for this request.

        Split prefill keeps one staged chunk: deeper depths of the current chunk
        still read its hidden rows, so the next chunk is staged only after the
        current one reached the last depth. Wavefront chunked prefill widens the
        window to several chunks, and then this bound becomes a chunk count.
        """
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
            remaining = len(queue)
            while (
                queue
                and token_budget
                and len(scheduled_requests) < self.config.max_num_seqs
                and remaining
            ):
                remaining -= 1
                request = self.requests[queue.popleft()]
                if request.request_id in selected:
                    continue
                if eligible is not None and not eligible(request):
                    queue.append(request.request_id)
                    continue
                if work == WorkKey.PROMPT_PRELUDE and not self._prelude_window_open(
                    request
                ):
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
        self.policy.record_batch(item.work for item in scheduled_requests)
        return SchedulerOutput(scheduled_requests)

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
