"""Pure-logical schedule, lane tree, and driver for pipelined self-speculative decoding.

Decoupled from PyTorch tensors, CUDA graphs, and physical KV allocation.
A mock executor can fully drive and verify this logic deterministically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class PipelineLane:
    """One input token progressing through repeated executions of the recurrent core.

    stage: Next zero-based recurrent iteration to execute (not layer ID).
    index: Zero-based completion index of the prediction.
    token: Input token being processed; prediction is the next token.
    state: Integer slot ID in the state bank (assigned and owned by executor).
    branch: Logical branch ID, local to the generation call.
    parent_branch: Branch ID whose readout spawned this lane.
    origin_stage: Parent's recurrent iteration when spawning this lane.
    first_token: Candidate token ID proposed at q1.
    first_state: Optional proposal distribution/state (for sampling).
    second_token: Optional alternative candidate token ID proposed at q2.
    second_state: Optional second proposal distribution/state.
    """

    stage: int
    index: int
    token: int
    state: int | None = None
    branch: int = 0
    parent_branch: int | None = None
    origin_stage: int | None = None
    first_token: int | None = None
    first_state: Any = None
    second_token: int | None = None
    second_state: Any = None


@dataclass(frozen=True)
class PipelineResult:
    """Result for one lane produced by an executor forward pass."""

    state: int
    prediction: int
    proposed: bool = False
    proposal_state: Any = None
    accepted_stage: int | None = None
    proposal_action: str | None = None


@dataclass(frozen=True)
class PipelineSchedule:
    """Schedule configuration and derived topology for looped recurrent pipelining.

    stages: Total recurrent iterations S per token (e.g. 4 for Ouro).
    step_size: Recurrent iterations K executed per cycle (q1 readout happens after K).
    second_step: Optional recurrent multiple X for q2 readout (logical scheduler only;
        currently unsupported in PipelinedSpeculativeRunner).
    """

    stages: int
    step_size: int
    second_step: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stages, int) or self.stages <= 0:
            raise ValueError(f"stages must be a positive integer, got {self.stages}")
        if (
            not isinstance(self.step_size, int)
            or not 1 <= self.step_size < self.stages
            or self.stages % self.step_size != 0
        ):
            raise ValueError(
                f"step_size must divide stages with 1 <= step_size < stages; "
                f"got step_size={self.step_size}, stages={self.stages}"
            )
        if self.second_step is not None:
            if not isinstance(self.second_step, int) or not (1 < self.second_step < self.depth):
                raise ValueError(
                    f"second_step must satisfy 1 < second_step < depth; "
                    f"got second_step={self.second_step}, depth={self.depth}"
                )

    @property
    def depth(self) -> int:
        """Total cycles needed for a lane to reach final depth: S / K."""
        return self.stages // self.step_size

    @property
    def proposal_stage(self) -> int:
        """q1's zero-based last recurrent iteration: K - 1."""
        return self.step_size - 1

    @property
    def second_stage(self) -> int | None:
        """q2's zero-based last iteration: K * X - 1, or None if disabled."""
        if self.second_step is None:
            return None
        return self.step_size * self.second_step - 1

    @property
    def endpoint_kinds(self) -> dict[int, str]:
        """Map readout iteration indices to 'q1', 'q2', and 'final' roles."""
        endpoints = {self.proposal_stage: "q1"}
        if self.second_stage is not None:
            endpoints[self.second_stage] = "q2"
        endpoints[self.stages - 1] = "final"
        return endpoints

    @property
    def endpoint_stages(self) -> tuple[int, ...]:
        """Readout iteration indices in ascending order."""
        return tuple(sorted(self.endpoint_kinds.keys()))

    @property
    def row_birth_sizes(self) -> list[int]:
        """Maximum number of new stage-zero lanes born across cycles 0 through depth - 1."""
        if self.second_step is None:
            return [1] * self.depth
        births = [1] * self.second_step
        for step in range(self.second_step, self.depth):
            births.append(births[step - 1] + births[step - self.second_step])
        return births

    @property
    def max_rows(self) -> int:
        """Upper bound on concurrent live lanes."""
        return sum(self.row_birth_sizes)

    def role_max_batch_sizes(self) -> dict[str, int]:
        """Upper bounds for pre, recurrent core, and post readout batch sizes."""
        births = self.row_birth_sizes
        post_steps = {0, self.depth - 1}
        if self.second_step is not None:
            post_steps.add(self.second_step - 1)
        return {
            "pre": births[-1],
            "recurrent": sum(births),
            "post": sum(births[self.depth - 1 - step] for step in post_steps),
        }


class PipelineExecutor(Protocol):
    """Protocol connecting the logical pipeline driver to the model execution runtime."""

    @property
    def schedule(self) -> PipelineSchedule: ...

    def forward(self, lanes: Sequence[PipelineLane]) -> Sequence[PipelineResult]: ...

    def retain_branches(
        self,
        lanes: Sequence[PipelineLane],
        accepted_stage: int | None = None,
    ) -> None: ...

    def should_stop(self, tokens: Sequence[int]) -> bool: ...


def _descends_from(
    branch: int | None,
    ancestor: int,
    parents: Mapping[int, int | None],
) -> bool:
    """Check whether branch equals ancestor or descends from it."""
    while branch is not None and branch != ancestor:
        branch = parents.get(branch)
    return branch == ancestor


def _prune_ancestry(
    lanes: Sequence[PipelineLane],
    root: int,
    parents: Mapping[int, int | None],
) -> dict[int, int | None]:
    """Return an updated parent map retaining only live branches rooted at root."""
    live = {root}
    for lane in lanes:
        branch = lane.branch
        while branch != root:
            if branch is None or branch not in parents:
                raise RuntimeError("pipeline branch escaped the selected ancestry")
            live.add(branch)
            branch = parents[branch]
    return {branch: None if branch == root else parents[branch] for branch in live}


def continue_pipelined(
    first_token: int,
    max_new_tokens: int,
    executor: PipelineExecutor,
) -> list[int]:
    """Drive pipelined self-speculative decoding from first_token up to max_new_tokens.

    Returns the ordered list of committed completion token IDs (including first_token).
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    schedule = executor.schedule
    proposal_stage = schedule.proposal_stage
    second_stage = schedule.second_stage
    proposal_stages = {proposal_stage, second_stage} - {None}
    stride = schedule.step_size
    final_stage = schedule.stages - 1
    max_rows = schedule.max_rows
    committed = [first_token]

    if max_new_tokens == 1:
        executor.retain_branches(())
        return committed

    next_branch = 1
    parents: dict[int, int | None] = {0: None}
    lanes = [
        PipelineLane(
            stage=0,
            index=1,
            token=first_token,
        )
    ]

    while len(committed) < max_new_tokens:
        results = list(executor.forward(lanes))
        if len(results) != len(lanes):
            raise ValueError(f"executor returned {len(results)} results for {len(lanes)} lanes")

        following: list[PipelineLane] = []
        children: list[PipelineLane] = []
        final: tuple[PipelineLane, PipelineResult] | None = None
        accepted_stage: int | None = None

        for lane, result in zip(lanes, results):
            end_stage = lane.stage + stride - 1
            if end_stage == final_stage:
                if final is not None:
                    raise RuntimeError("multiple branches reached the final stage")
                final = (lane, result)
                continue

            lane.stage = end_stage + 1
            lane.state = result.state
            continued = lane

            if end_stage in proposal_stages and result.proposed:
                if end_stage == proposal_stage:
                    continued.first_token = result.prediction
                    continued.first_state = result.proposal_state
                elif result.proposal_action in ("residual", "argmax"):
                    continued.second_token = result.prediction
                    continued.second_state = result.proposal_state
                else:
                    raise RuntimeError("q2 proposal missing valid branch action")

                if lane.index + 1 < max_new_tokens:
                    children.append(
                        PipelineLane(
                            stage=0,
                            index=lane.index + 1,
                            token=result.prediction,
                            branch=next_branch,
                            parent_branch=lane.branch,
                            origin_stage=end_stage,
                        )
                    )
                    parents[next_branch] = lane.branch
                    next_branch += 1

            following.append(continued)

        following.extend(children)

        if final is not None:
            lane, result = final
            if lane.index != len(committed):
                raise RuntimeError("pipeline attempted to commit out of order")
            committed.append(result.prediction)
            if executor.should_stop(committed):
                executor.retain_branches(())
                return committed

            if lane.index + 1 < max_new_tokens:
                selected = next(
                    (
                        child
                        for child in following
                        if child.parent_branch == lane.branch
                        and child.token == result.prediction
                        and child.origin_stage == result.accepted_stage
                    ),
                    None,
                )

                if selected is None:
                    selected = PipelineLane(
                        stage=0,
                        index=lane.index + 1,
                        token=result.prediction,
                        branch=next_branch,
                        parent_branch=lane.branch,
                        origin_stage=final_stage,
                    )
                    parents[next_branch] = lane.branch
                    next_branch += 1
                    following.append(selected)

                accepted_stage = result.accepted_stage
                selected_branch = selected.branch
                following = [
                    child
                    for child in following
                    if _descends_from(child.branch, selected_branch, parents)
                ]
                parents = _prune_ancestry(following, selected_branch, parents)
            else:
                following = []

        if len(following) > max_rows:
            raise RuntimeError(
                f"speculative tree exceeded branch row limit ({len(following)} > {max_rows})"
            )

        lanes = following
        executor.retain_branches(tuple(lanes), accepted_stage)
        if not lanes and len(committed) < max_new_tokens:
            raise RuntimeError("pipeline drained before generation completed")

    return committed
