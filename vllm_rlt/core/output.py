"""Loop execution plans with a vLLM SchedulerOutput view of the work."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput

from vllm_rlt.core.loop import LoopOp, LoopSpan


@dataclass
class LoopSchedulerOutput(SchedulerOutput):
    op: LoopOp | None = None
    spans: tuple[LoopSpan, ...] = ()
    discarded_states: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @classmethod
    def for_step(cls, op: LoopOp, spans: Iterable[LoopSpan]) -> "LoopSchedulerOutput":
        output = cls.make_empty()
        output.spans = tuple(spans)
        if not output.spans:
            raise ValueError("a loop op requires at least one span")
        output.op = op
        counts = output.num_scheduled_tokens
        for span in output.spans:
            counts[span.request_id] = counts.get(span.request_id, 0) + span.token_count
        output.total_num_scheduled_tokens = sum(counts.values())
        return output


@dataclass(slots=True)
class LoopModelRunnerOutput(ModelRunnerOutput):
    # Ordered by span, with None for spans that do not exit on this step.
    exit_depths: tuple[int | None, ...] = ()
    draft_tokens: dict[str, int] | None = None
    logprobs_by_request: dict[str, LogprobsLists] = field(default_factory=dict)
