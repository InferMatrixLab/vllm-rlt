"""Deterministic unit tests for pure-logical speculative pipeline scheduling."""

from __future__ import annotations

import pytest

from vllm_rlt.worker.speculative_pipeline import (
    PipelineLane,
    PipelineResult,
    PipelineSchedule,
    continue_pipelined,
)


def test_schedule_validation():
    # Valid schedules
    s1 = PipelineSchedule(stages=4, step_size=1, second_step=2)
    assert s1.depth == 4
    assert s1.proposal_stage == 0
    assert s1.second_stage == 1
    assert s1.endpoint_kinds == {0: "q1", 1: "q2", 3: "final"}

    s2 = PipelineSchedule(stages=4, step_size=1, second_step=None)
    assert s2.second_stage is None
    assert s2.endpoint_kinds == {0: "q1", 3: "final"}
    assert s2.max_rows == 4

    # Invalid schedules
    with pytest.raises(ValueError, match="stages must be a positive integer"):
        PipelineSchedule(stages=0, step_size=1)

    with pytest.raises(ValueError, match="step_size must divide stages"):
        PipelineSchedule(stages=4, step_size=3)

    with pytest.raises(ValueError, match="step_size must divide stages"):
        PipelineSchedule(stages=4, step_size=4)

    with pytest.raises(ValueError, match="second_step must satisfy"):
        PipelineSchedule(stages=4, step_size=1, second_step=4)


def test_schedule_role_max_batch_sizes():
    sched = PipelineSchedule(stages=4, step_size=1, second_step=2)
    sizes = sched.role_max_batch_sizes()
    assert sizes["pre"] >= 1
    assert sizes["recurrent"] >= sched.depth
    assert sizes["post"] >= 1


class DeterministicMockExecutor:
    """Mock executor returning scripted ground-truth completions."""

    def __init__(
        self,
        schedule: PipelineSchedule,
        target_tokens: list[int],
        proposal_accuracy: float = 1.0,
        eos_token: int | None = None,
    ):
        self._schedule = schedule
        self._target_tokens = target_tokens
        self._proposal_accuracy = proposal_accuracy
        self._eos_token = eos_token
        self.forward_calls = 0
        self.retained_calls: list[tuple[tuple[PipelineLane, ...], int | None]] = []

    @property
    def schedule(self) -> PipelineSchedule:
        return self._schedule

    def forward(self, lanes: list[PipelineLane]) -> list[PipelineResult]:
        self.forward_calls += 1
        results = []
        proposal_stages = {self._schedule.proposal_stage, self._schedule.second_stage} - {None}
        final_stage = self._schedule.stages - 1

        for slot, lane in enumerate(lanes):
            end_stage = lane.stage + self._schedule.step_size - 1
            idx = lane.index
            correct_token = self._target_tokens[idx] if idx < len(self._target_tokens) else 9999

            if end_stage == final_stage:
                # Final readout gives the true target token
                accepted_stage = None
                if lane.first_token is not None and lane.first_token == correct_token:
                    accepted_stage = self._schedule.proposal_stage
                elif lane.second_token is not None and lane.second_token == correct_token:
                    accepted_stage = self._schedule.second_stage

                results.append(
                    PipelineResult(
                        state=slot,
                        prediction=correct_token,
                        proposed=False,
                        accepted_stage=accepted_stage,
                    )
                )
            elif end_stage in proposal_stages:
                # Propose next token
                if self._proposal_accuracy == 1.0:
                    cand = correct_token
                elif self._proposal_accuracy == 0.0:
                    cand = correct_token + 1000
                else:
                    cand = correct_token if (idx % 2 == 0) else correct_token + 1000

                action = "residual" if end_stage == self._schedule.second_stage else None
                results.append(
                    PipelineResult(
                        state=slot,
                        prediction=cand,
                        proposed=True,
                        proposal_action=action,
                    )
                )
            else:
                results.append(
                    PipelineResult(
                        state=slot,
                        prediction=0,
                        proposed=False,
                    )
                )
        return results

    def retain_branches(
        self,
        lanes: tuple[PipelineLane, ...],
        accepted_stage: int | None = None,
    ) -> None:
        self.retained_calls.append((lanes, accepted_stage))

    def should_stop(self, tokens: list[int]) -> bool:
        if self._eos_token is not None and tokens[-1] == self._eos_token:
            return True
        return False


def test_continue_pipelined_perfect_proposals():
    target = [10, 11, 12, 13, 14, 15, 16]
    sched = PipelineSchedule(stages=4, step_size=1, second_step=2)
    mock = DeterministicMockExecutor(sched, target, proposal_accuracy=1.0)

    committed = continue_pipelined(first_token=10, max_new_tokens=len(target), executor=mock)
    assert committed == target
    assert mock.forward_calls > 0
    # Every token after the initial ramp reaches acceptance
    last_retained = mock.retained_calls[-1][0]
    assert len(last_retained) <= sched.max_rows


def test_continue_pipelined_complete_rejections():
    target = [100, 101, 102, 103, 104]
    sched = PipelineSchedule(stages=4, step_size=1, second_step=2)
    mock = DeterministicMockExecutor(sched, target, proposal_accuracy=0.0)

    committed = continue_pipelined(first_token=100, max_new_tokens=len(target), executor=mock)
    # Output must still strictly equal ground truth even when all proposals are rejected
    assert committed == target


def test_continue_pipelined_alternating_proposals():
    target = [1000, 1001, 1002, 1003, 1004, 1005, 1006]
    sched = PipelineSchedule(stages=4, step_size=1, second_step=None)
    mock = DeterministicMockExecutor(sched, target, proposal_accuracy=0.5)

    committed = continue_pipelined(first_token=1000, max_new_tokens=len(target), executor=mock)
    assert committed == target


def test_continue_pipelined_early_stopping():
    target = [50, 51, 52, 999, 53, 54]  # 999 is EOS
    sched = PipelineSchedule(stages=4, step_size=1, second_step=2)
    mock = DeterministicMockExecutor(sched, target, proposal_accuracy=1.0, eos_token=999)

    committed = continue_pipelined(first_token=50, max_new_tokens=10, executor=mock)
    assert committed == [50, 51, 52, 999]
    # Resources must be released upon early stop
    assert mock.retained_calls[-1][0] == ()


def test_continue_pipelined_single_token():
    sched = PipelineSchedule(stages=4, step_size=1)
    mock = DeterministicMockExecutor(sched, [42])

    committed = continue_pipelined(first_token=42, max_new_tokens=1, executor=mock)
    assert committed == [42]
    assert mock.retained_calls[-1][0] == ()
