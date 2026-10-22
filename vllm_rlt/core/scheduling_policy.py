"""Work selection and fairness state for loop-level scheduling."""

from enum import Enum, auto
from typing import TYPE_CHECKING

from vllm_rlt.config import SchedulerConfig, SpeculativeConfig
from vllm_rlt.core.execution import WorkKey

if TYPE_CHECKING:
    from vllm_rlt.core.scheduler import Scheduler, SchedulerOutput


class NoRefillPhase(Enum):
    FILL = auto()
    CORE = auto()
    CODA = auto()


PROMPT_WORKS = frozenset(
    (WorkKey.PROMPT_ATOMIC, WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
)
DECODE_PROGRESS_WORKS = frozenset((WorkKey.DECODE_RECURRENT, WorkKey.SPECULATIVE))
PRELUDE_WORKS = (WorkKey.DECODE_PRELUDE, WorkKey.PROMPT_PRELUDE)
RECURRENT_WORKS = (WorkKey.DECODE_RECURRENT, WorkKey.PROMPT_RECURRENT)


class SchedulingPolicy:
    PREFILL_WORKS = (WorkKey.PROMPT_ATOMIC,)

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.prefill_batches_since_recurrent = 0

    @property
    def decode_due(self) -> bool:
        return self.prefill_batches_since_recurrent >= self.config.max_prefill_batches_before_decode

    def work_group(self, work: WorkKey) -> tuple[WorkKey, ...]:
        return (work,)

    def has_ready_exact_work(self, scheduler, work: WorkKey) -> bool:
        return scheduler.has_ready_work(work)

    def has_ready_work(self, scheduler, work: WorkKey) -> bool:
        return scheduler.has_ready_works(*self.work_group(work))

    def take_work(self, scheduler, work: WorkKey):
        return scheduler.take_works(self.work_group(work))

    def record_batch(self, works) -> None:
        works = frozenset(works)
        if works & DECODE_PROGRESS_WORKS:
            self.prefill_batches_since_recurrent = 0
        elif works & PROMPT_WORKS:
            self.prefill_batches_since_recurrent += 1


class PromptOnlyPolicy(SchedulingPolicy):
    """Select prompt work only; resource ownership and execution stay shared."""

    def __init__(self, config, *, atomic_prefill):
        super().__init__(config)
        self.PREFILL_WORKS = (
            (WorkKey.PROMPT_ATOMIC,)
            if atomic_prefill
            else (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
        )

    def schedule(self, scheduler, *, prefer_recurrent=False):
        scheduler._admit()
        for work in self.PREFILL_WORKS:
            batch = self.take_work(scheduler, work)
            if batch is not None:
                return batch
        return None


class RefillPolicy(SchedulingPolicy):
    def schedule(
        self, scheduler: "Scheduler", *, prefer_recurrent: bool = False
    ) -> "SchedulerOutput | None":
        if prefer_recurrent and self.has_ready_exact_work(scheduler, WorkKey.DECODE_RECURRENT):
            return self.take_work(scheduler, WorkKey.DECODE_RECURRENT)
        if self.has_ready_exact_work(scheduler, WorkKey.DECODE_PRELUDE):
            return self.take_work(scheduler, WorkKey.DECODE_PRELUDE)
        if self.has_ready_exact_work(scheduler, WorkKey.DECODE_CODA) and (
            len(scheduler.work_queues[WorkKey.DECODE_CODA]) >= self.config.min_coda_batch_size
            or not self.has_ready_exact_work(scheduler, WorkKey.DECODE_RECURRENT)
        ):
            return self.take_work(scheduler, WorkKey.DECODE_CODA)
        if self.decode_due and self.has_ready_exact_work(scheduler, WorkKey.DECODE_RECURRENT):
            return self.take_work(scheduler, WorkKey.DECODE_RECURRENT)

        scheduler._admit()
        for work in self.PREFILL_WORKS:
            if self.has_ready_work(scheduler, work):
                return self.take_work(scheduler, work)
        # Admission can restore a request directly to a boundary work queue.
        if self.has_ready_exact_work(scheduler, WorkKey.DECODE_PRELUDE):
            return self.take_work(scheduler, WorkKey.DECODE_PRELUDE)
        if self.has_ready_exact_work(scheduler, WorkKey.DECODE_CODA) and (
            len(scheduler.work_queues[WorkKey.DECODE_CODA]) >= self.config.min_coda_batch_size
            or not self.has_ready_exact_work(scheduler, WorkKey.DECODE_RECURRENT)
        ):
            return self.take_work(scheduler, WorkKey.DECODE_CODA)
        if self.has_ready_exact_work(scheduler, WorkKey.DECODE_RECURRENT):
            return self.take_work(scheduler, WorkKey.DECODE_RECURRENT)
        return None


class NoRefillPolicy(SchedulingPolicy):
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.phase = NoRefillPhase.FILL

    def schedule(
        self, scheduler: "Scheduler", *, prefer_recurrent: bool = False
    ) -> "SchedulerOutput | None":
        if self.phase == NoRefillPhase.CORE:
            if self.has_ready_work(scheduler, WorkKey.DECODE_RECURRENT):
                return self.take_work(scheduler, WorkKey.DECODE_RECURRENT)
            self.phase = NoRefillPhase.CODA
        if self.phase == NoRefillPhase.CODA:
            if self.has_ready_work(scheduler, WorkKey.DECODE_CODA):
                return self.take_work(scheduler, WorkKey.DECODE_CODA)
            self.phase = NoRefillPhase.FILL
        if self.decode_due:
            if self.has_ready_work(scheduler, WorkKey.DECODE_PRELUDE):
                return self.take_work(scheduler, WorkKey.DECODE_PRELUDE)
            if self.has_ready_work(scheduler, WorkKey.DECODE_RECURRENT):
                self.phase = NoRefillPhase.CORE
                return self.take_work(scheduler, WorkKey.DECODE_RECURRENT)
        scheduler._admit()
        if self.has_ready_work(scheduler, WorkKey.PROMPT_ATOMIC):
            return self.take_work(scheduler, WorkKey.PROMPT_ATOMIC)
        if self.has_ready_work(scheduler, WorkKey.DECODE_CODA):
            return self.take_work(scheduler, WorkKey.DECODE_CODA)
        if self.has_ready_work(scheduler, WorkKey.DECODE_PRELUDE):
            return self.take_work(scheduler, WorkKey.DECODE_PRELUDE)
        if self.has_ready_work(scheduler, WorkKey.DECODE_RECURRENT):
            self.phase = NoRefillPhase.CORE
            return self.take_work(scheduler, WorkKey.DECODE_RECURRENT)
        return None


class SpeculativePolicy(SchedulingPolicy):
    """Schedule draft/verify rounds with independently chunked prompt work."""

    def __init__(self, config, *, atomic_prefill=True):
        super().__init__(config)
        self.PREFILL_WORKS = (
            (WorkKey.PROMPT_ATOMIC,)
            if atomic_prefill
            else (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
        )

    def schedule(self, scheduler, *, prefer_recurrent=False):
        if self.has_ready_exact_work(scheduler, WorkKey.DECODE_CODA):
            return self.take_work(scheduler, WorkKey.DECODE_CODA)
        if self.decode_due and self.has_ready_exact_work(scheduler, WorkKey.SPECULATIVE):
            return self.take_work(scheduler, WorkKey.SPECULATIVE)
        scheduler._admit()
        if self.has_ready_exact_work(scheduler, WorkKey.DECODE_CODA):
            return self.take_work(scheduler, WorkKey.DECODE_CODA)
        for work in self.PREFILL_WORKS:
            if self.has_ready_work(scheduler, work):
                return self.take_work(scheduler, work)
        if self.has_ready_exact_work(scheduler, WorkKey.SPECULATIVE):
            return self.take_work(scheduler, WorkKey.SPECULATIVE)
        return None


class SplitPrefillPolicy(RefillPolicy):
    """Refill ordering where each prompt recurrence depth is its own batch."""

    PREFILL_WORKS = (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)


class MixedPolicy(SplitPrefillPolicy):
    """Combine compatible prompt and decode work while sharing refill ordering."""

    def work_group(self, work: WorkKey) -> tuple[WorkKey, ...]:
        if work in PRELUDE_WORKS:
            return PRELUDE_WORKS
        if work in RECURRENT_WORKS:
            return RECURRENT_WORKS
        return (work,)


def create_scheduling_policy(
    scheduler_config: SchedulerConfig,
    speculative_config: SpeculativeConfig | None,
    atomic_prefill: bool = True,
    *,
    produces_tokens: bool = True,
) -> SchedulingPolicy:
    """Pick one policy and reject combinations no policy implements."""
    if not produces_tokens:
        return PromptOnlyPolicy(scheduler_config, atomic_prefill=atomic_prefill)
    if scheduler_config.mode == "mixed":
        if atomic_prefill or speculative_config is not None:
            raise ValueError("mixed scheduling requires non-atomic, non-speculative work")
    elif not atomic_prefill and speculative_config is None and scheduler_config.mode != "refill":
        raise ValueError("split prefill currently implements refill scheduling only")

    if scheduler_config.mode == "mixed":
        return MixedPolicy(scheduler_config)
    if speculative_config is not None:
        return SpeculativePolicy(scheduler_config, atomic_prefill=atomic_prefill)
    if atomic_prefill:
        policy_cls = RefillPolicy if scheduler_config.mode == "refill" else NoRefillPolicy
        return policy_cls(scheduler_config)
    return SplitPrefillPolicy(scheduler_config)
