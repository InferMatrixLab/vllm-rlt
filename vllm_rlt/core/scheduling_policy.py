"""Request-phase selection and fairness state for loop-level scheduling."""

from enum import Enum, auto
from typing import TYPE_CHECKING

from vllm_rlt.config import SchedulerConfig, SpeculativeConfig
from vllm_rlt.request import RequestPhase

if TYPE_CHECKING:
    from vllm_rlt.core.scheduler import Scheduler, SchedulerOutput


class NoRefillPhase(Enum):
    FILL = auto()
    CORE = auto()
    CODA = auto()


class SchedulingPolicy:
    # Prompt phases supported by this policy also count against the decode bound.
    PREFILL_PHASES = (RequestPhase.PREFILL,)

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.prefill_batches_since_recurrent = 0

    @property
    def decode_due(self) -> bool:
        return self.prefill_batches_since_recurrent >= self.config.max_prefill_batches_before_decode

    def record_batch(self, phase: RequestPhase) -> None:
        """Count nonempty scheduled batches, not attempts or GPU completions."""
        if phase in self.PREFILL_PHASES:
            self.prefill_batches_since_recurrent += 1
        elif phase in (RequestPhase.RECURRENT, RequestPhase.SPECULATIVE):
            self.prefill_batches_since_recurrent = 0


class PromptOnlyPolicy(SchedulingPolicy):
    """Select prompt work only; resource ownership and execution stay shared."""

    def __init__(self, config, *, atomic_prefill):
        super().__init__(config)
        self.PREFILL_PHASES = (
            (RequestPhase.PREFILL,)
            if atomic_prefill
            else (RequestPhase.PREFILLING_PRELUDE, RequestPhase.PREFILLING_RECURRENT)
        )

    def schedule(self, scheduler, *, prefer_recurrent=False):
        scheduler._admit()
        for phase in self.PREFILL_PHASES:
            batch = scheduler.take_phases((phase,))
            if batch is not None:
                return batch
        return None


class RefillPolicy(SchedulingPolicy):
    def schedule(
        self, scheduler: "Scheduler", *, prefer_recurrent: bool = False
    ) -> "SchedulerOutput | None":
        # Give independent core work an overlap opportunity before refilling.
        if prefer_recurrent and scheduler.has_ready(RequestPhase.RECURRENT):
            return scheduler._take(RequestPhase.RECURRENT)
        if scheduler.has_ready(RequestPhase.PRELUDE):
            return scheduler._take(RequestPhase.PRELUDE)
        if scheduler.has_ready(RequestPhase.CODA) and (
            len(scheduler.queues[RequestPhase.CODA]) >= self.config.min_coda_batch_size
            or not scheduler.has_ready(RequestPhase.RECURRENT)
        ):
            return scheduler._take(RequestPhase.CODA)
        if self.decode_due and scheduler.has_ready(RequestPhase.RECURRENT):
            return scheduler._take(RequestPhase.RECURRENT)
        scheduler._admit()
        for phase in self.PREFILL_PHASES:
            if scheduler.has_ready(phase):
                return scheduler._take(phase)
        # Admission can restore a snapshot directly to a boundary phase.
        if scheduler.has_ready(RequestPhase.PRELUDE):
            return scheduler._take(RequestPhase.PRELUDE)
        if scheduler.has_ready(RequestPhase.CODA) and (
            len(scheduler.queues[RequestPhase.CODA]) >= self.config.min_coda_batch_size
            or not scheduler.has_ready(RequestPhase.RECURRENT)
        ):
            return scheduler._take(RequestPhase.CODA)
        if scheduler.has_ready(RequestPhase.RECURRENT):
            return scheduler._take(RequestPhase.RECURRENT)
        return None


class NoRefillPolicy(SchedulingPolicy):
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.phase = NoRefillPhase.FILL

    def schedule(
        self, scheduler: "Scheduler", *, prefer_recurrent: bool = False
    ) -> "SchedulerOutput | None":
        if self.phase == NoRefillPhase.CORE:
            if scheduler.has_ready(RequestPhase.RECURRENT):
                return scheduler._take(RequestPhase.RECURRENT)
            self.phase = NoRefillPhase.CODA
        if self.phase == NoRefillPhase.CODA:
            if scheduler.has_ready(RequestPhase.CODA):
                return scheduler._take(RequestPhase.CODA)
            self.phase = NoRefillPhase.FILL
        # Bound prefill runs so continuous arrivals cannot starve decode.
        if self.decode_due:
            if scheduler.has_ready(RequestPhase.PRELUDE):
                return scheduler._take(RequestPhase.PRELUDE)
            if scheduler.has_ready(RequestPhase.RECURRENT):
                self.phase = NoRefillPhase.CORE
                return scheduler._take(RequestPhase.RECURRENT)
        scheduler._admit()
        if scheduler.has_ready(RequestPhase.PREFILL):
            return scheduler._take(RequestPhase.PREFILL)
        if scheduler.has_ready(RequestPhase.CODA):  # First output after full-depth prompt prefill.
            return scheduler._take(RequestPhase.CODA)
        if scheduler.has_ready(RequestPhase.PRELUDE):
            return scheduler._take(RequestPhase.PRELUDE)
        if scheduler.has_ready(RequestPhase.RECURRENT):
            self.phase = NoRefillPhase.CORE
            return scheduler._take(RequestPhase.RECURRENT)
        return None


class SpeculativePolicy(SchedulingPolicy):
    """Schedule draft/verify rounds with independently chunked prompt work."""

    def __init__(self, config, *, atomic_prefill=True):
        super().__init__(config)
        self.PREFILL_PHASES = (
            (RequestPhase.PREFILL,)
            if atomic_prefill
            else (RequestPhase.PREFILLING_PRELUDE, RequestPhase.PREFILLING_RECURRENT)
        )

    def schedule(self, scheduler, *, prefer_recurrent=False):
        if scheduler.has_ready(RequestPhase.CODA):
            return scheduler._take(RequestPhase.CODA)
        if self.decode_due and scheduler.has_ready(RequestPhase.SPECULATIVE):
            return scheduler._take(RequestPhase.SPECULATIVE)
        scheduler._admit()
        # Admission can import a complete prompt directly into CODA.
        if scheduler.has_ready(RequestPhase.CODA):
            return scheduler._take(RequestPhase.CODA)
        for phase in self.PREFILL_PHASES:
            if scheduler.has_ready(phase):
                return scheduler._take(phase)
        if scheduler.has_ready(RequestPhase.SPECULATIVE):
            return scheduler._take(RequestPhase.SPECULATIVE)
        return None


class SplitPrefillPolicy(RefillPolicy):
    """Refill ordering where each recurrence depth is its own batch boundary."""

    PREFILL_PHASES = (RequestPhase.PREFILLING_PRELUDE, RequestPhase.PREFILLING_RECURRENT)


class MixedPolicy(SplitPrefillPolicy):
    """Choose compatible prompt/generation loops; execution is policy-agnostic."""

    def schedule(self, scheduler, *, prefer_recurrent=False):
        if scheduler.has_ready(RequestPhase.RECURRENT) and scheduler.has_ready(
            RequestPhase.PREFILLING_RECURRENT
        ):
            return scheduler.take_phases(
                (RequestPhase.RECURRENT, RequestPhase.PREFILLING_RECURRENT)
            )
        return super().schedule(scheduler, prefer_recurrent=prefer_recurrent)


def create_scheduling_policy(
    scheduler_config: SchedulerConfig,
    speculative_config: SpeculativeConfig | None,
    atomic_prefill: bool = True,
    *,
    produces_tokens: bool = True,
) -> SchedulingPolicy:
    """Pick one policy; reject combinations no policy implements yet.

    Failing here keeps an unimplemented mode from silently changing
    semantics or returning an empty schedule forever.
    """
    # Prompt-only phases do not use prompt/generation interleaving rules.
    if not produces_tokens:
        return PromptOnlyPolicy(scheduler_config, atomic_prefill=atomic_prefill)

    # The remaining compatibility checks apply to generation scheduling.
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
