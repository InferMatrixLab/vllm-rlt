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
    to evaluate the exit gate.
    """

    loop_step: LoopStep | None = None
    exit_params: dict[str, tuple[float, int, int]] = field(default_factory=dict)


@dataclass
class LoopModelRunnerOutput(ModelRunnerOutput):
    """Runner output for one loop step. ``exit_depths`` maps a request to the
    core depth at which its exit gate fired (only rows with ``can_exit``)."""

    exit_depths: dict[str, int] = field(default_factory=dict)
