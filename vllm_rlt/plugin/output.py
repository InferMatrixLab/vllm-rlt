"""Scheduler output and runner output carrying loop information.

Both subclass vLLM's types, so every consumer that only knows vLLM's fields
(executor, worker, EngineCore) keeps working. Following vllm-omni, the extra
fields are plain dataclass attributes with defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_rlt.plugin.program import LoopStep


@dataclass
class LoopSchedulerOutput(SchedulerOutput):
    """``num_scheduled_tokens`` covers only the rows of ``loop_step``.

    ``exit_params`` maps each gated row to (threshold, min_loops, limit): the
    worker has no request objects, so the scheduler sends what the runner needs
    to evaluate the exit gate. ``candidates`` is what this rank could run next,
    which the workers use to agree on one op across data-parallel ranks.
    ``span_starts`` is each row's ``num_computed_tokens`` before its span: with
    WCPB several chunks of one prompt are in flight, so the worker must not
    infer a chunk's position from its own counter.
    """

    loop_step: LoopStep | None = None
    exit_params: dict[str, tuple[float, int, int]] = field(default_factory=dict)
    candidates: dict[tuple[int, int], int] = field(default_factory=dict)
    step_key: tuple[int, int] | None = None  # the (op, depth) group this step is
    span_starts: dict[str, int] = field(default_factory=dict)


@dataclass
class LoopModelRunnerOutput(ModelRunnerOutput):
    """Runner output for one loop step. ``exit_depths`` maps a request to the
    core depth at which its exit gate fired (only rows with ``can_exit``).
    ``skipped`` lists rows the worker did not run because the data-parallel
    ranks agreed on a different op; the scheduler leaves them where they were."""

    exit_depths: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
