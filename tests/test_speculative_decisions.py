"""Unit tests for GPU decision policies, pinned readback, and slot management."""

import pytest
import torch

from vllm_rlt.worker.speculative_decisions import (
    DecisionReadback,
    GreedyDecisionPolicy,
    StateSlotPool,
)
from vllm_rlt.worker.speculative_pipeline import PipelineLane, PipelineSchedule


def test_state_slot_pool():
    pool = StateSlotPool(capacity=3)
    s0 = pool.acquire()
    s1 = pool.acquire()
    s2 = pool.acquire()
    assert [s0, s1, s2] == [0, 1, 2]

    with pytest.raises(RuntimeError, match="pipeline requires more recurrent state slots"):
        pool.acquire()

    # Retain only slot 0 and 2; slot 1 should be recycled
    pool.retain([0, 2])
    # Next acquire should reuse slot 1
    assert pool.acquire() == 1


def test_decision_readback_cpu():
    readback = DecisionReadback(max_rows=4, device="cpu")
    decisions = torch.tensor([[101], [202], [303]], dtype=torch.int64)
    staged = readback.stage(decisions)
    recovered = readback.read(staged)
    assert recovered == [[101], [202], [303]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decision_readback_cuda():
    readback = DecisionReadback(max_rows=4, device="cuda")
    decisions = torch.tensor([[55], [66]], dtype=torch.int64, device="cuda")
    staged = readback.stage(decisions)
    recovered = readback.read(staged)
    assert recovered == [[55], [66]]


def test_greedy_decision_policy_resolve():
    sched = PipelineSchedule(stages=4, step_size=1, second_step=2)
    policy = GreedyDecisionPolicy(max_rows=4, device="cpu")

    lane = PipelineLane(stage=0, index=1, token=10)

    # 1. q1 proposes argmax
    out_q1 = policy.resolve(lane, "q1", [100], None, sched)
    assert out_q1.prediction == 100
    assert out_q1.proposed is True
    assert out_q1.accepted_stage is None

    # Save q1 proposal onto lane
    lane.first_token = 100

    # 2. q2 proposes only if different from q1
    out_q2_same = policy.resolve(lane, "q2", [100], None, sched)
    assert out_q2_same.proposed is False

    out_q2_diff = policy.resolve(lane, "q2", [200], None, sched)
    assert out_q2_diff.prediction == 200
    assert out_q2_diff.proposed is True
    assert out_q2_diff.proposal_action == "argmax"

    lane.second_token = 200

    # 3. Final matches q1
    out_final_accept_q1 = policy.resolve(lane, "final", [100], None, sched)
    assert out_final_accept_q1.prediction == 100
    assert out_final_accept_q1.first_rejected is False
    assert out_final_accept_q1.accepted_stage == sched.proposal_stage  # stage 0

    # 4. Final rejects q1 but matches q2
    out_final_accept_q2 = policy.resolve(lane, "final", [200], None, sched)
    assert out_final_accept_q2.prediction == 200
    assert out_final_accept_q2.first_rejected is True
    assert out_final_accept_q2.second_rejected is False
    assert out_final_accept_q2.accepted_stage == sched.second_stage  # stage 1

    # 5. Final rejects both
    out_final_reject_all = policy.resolve(lane, "final", [999], None, sched)
    assert out_final_reject_all.prediction == 999
    assert out_final_reject_all.first_rejected is True
    assert out_final_reject_all.second_rejected is True
    assert out_final_reject_all.accepted_stage is None
