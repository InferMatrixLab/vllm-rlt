"""Pure-Python tests: no vLLM, no GPU. These pin down the loop program, the
planner, the exit policy and DP alignment, which hold the plugin's decisions."""

from __future__ import annotations

import math

import pytest

from vllm_rlt.plugin.exit import ExitParams, should_exit, update_remaining
from vllm_rlt.plugin.parallel import NoAlign, choose_op
from vllm_rlt.plugin.plan import Ready, candidates, group_key, take_rows
from vllm_rlt.plugin.program import LoopLayout, LoopStep, Op

INKLING = LoopLayout(num_passes=3)
OURO = LoopLayout(num_passes=4, has_prelude=False, has_gate=True)


def walk(layout: LoopLayout, last_depth: int, exit_at: int | None = None):
    op, depth = layout.first()
    seq = [(op, depth)]
    while True:
        nxt = layout.next_op(op, depth, last_depth, exited=(op == Op.CORE and depth == exit_at))
        if nxt is None:
            return seq
        op, depth = nxt
        seq.append(nxt)


def test_inkling_program_is_prelude_core_coda():
    expected = [(Op.PRELUDE, 0), (Op.CORE, 0), (Op.CORE, 1), (Op.CORE, 2), (Op.CODA, 2)]
    assert walk(INKLING, 2) == expected


def test_ouro_has_no_prelude():
    assert walk(OURO, 3)[0] == (Op.CORE, 0)
    assert walk(OURO, 3)[-1][0] == Op.CODA


def test_early_exit_skips_remaining_cores_and_keeps_exit_depth():
    seq = walk(OURO, 3, exit_at=1)
    assert seq == [(Op.CORE, 0), (Op.CORE, 1), (Op.CODA, 1)]


def test_full_only_is_one_op():
    full = LoopLayout(num_passes=3, full_only=True)
    assert full.first() == (Op.FULL, 0)
    assert full.next_op(Op.FULL, 0, 2) is None


def test_exit_params_parse_and_validate():
    p = ExitParams.from_extra_args({"rlt_exit_threshold": 0.9, "rlt_min_loops": 2})
    assert p.uses_gate and p.min_loops == 2 and p.limit(4) == 4
    assert not ExitParams.from_extra_args(None).uses_gate
    fixed = {"rlt_target_loops": 2, "rlt_exit_threshold": 0.5}
    assert ExitParams.from_extra_args(fixed).limit(4) == 2
    for bad in ({"rlt_exit_threshold": 1.5}, {"rlt_min_loops": 0}):
        with pytest.raises(ValueError):
            ExitParams.from_extra_args(bad)


def test_exit_gate_matches_ouro_cumulative_threshold():
    """remaining = prod(1 - sigmoid(g)); exit when 1 - remaining >= threshold."""
    params = ExitParams(threshold=0.8, min_loops=1)
    remaining, gates = 1.0, [-1.0, 0.0, 2.0, 5.0]
    exits = []
    for i, g in enumerate(gates, start=1):
        remaining = update_remaining(remaining, g)
        exits.append(should_exit(params, i, remaining, num_passes=4))
    cumulative = 1 - math.prod(1 - 1 / (1 + math.exp(-g)) for g in gates[:3])
    assert cumulative >= 0.8 and exits == [False, False, True, True]


def test_exit_respects_min_and_max_loops():
    params = ExitParams(threshold=0.1, min_loops=3, max_loops=3)
    assert not should_exit(params, 2, remaining=0.01, num_passes=4)
    assert should_exit(params, 3, remaining=0.99, num_passes=4)  # hits max_loops


def ready(rid, op, depth=0, waited=0, tokens=1):
    return Ready(rid, op, depth, tokens, waited)


def test_planner_takes_the_largest_single_op_group():
    rows = [
        ready("a", Op.CORE, 1),
        ready("b", Op.CORE, 1),
        ready("c", Op.CORE, 0),
        ready("d", Op.PRELUDE),
    ]
    local = candidates(rows, allow_mixed_depth=False, wait_limit=8)
    assert choose_op([local]) == (int(Op.CORE), 1)
    picked = take_rows(rows, (int(Op.CORE), 1), max_rows=8, token_budget=100)
    assert [r.req_id for r in picked] == ["a", "b"]
    assert len({r.op for r in picked}) == 1


def test_planner_never_mixes_depths_unless_enabled():
    rows = [ready("a", Op.CORE, 0), ready("b", Op.CORE, 2)]
    assert group_key(rows[0], False) != group_key(rows[1], False)
    assert group_key(rows[0], True) == group_key(rows[1], True)


def test_coda_rows_share_a_group_regardless_of_exit_depth():
    assert group_key(ready("a", Op.CODA, 0), False) == group_key(ready("b", Op.CODA, 2), False)


def test_starvation_overrides_group_size():
    rows = [ready(f"c{i}", Op.CORE, 0) for i in range(5)] + [ready("old", Op.CODA, 0, waited=9)]
    assert choose_op([candidates(rows, False, wait_limit=8)]) == (int(Op.CODA), -1)


def test_take_rows_respects_row_and_token_limits_but_always_takes_one():
    rows = [ready(f"r{i}", Op.CORE, 0, tokens=300) for i in range(4)]
    assert len(take_rows(rows, (int(Op.CORE), 0), max_rows=8, token_budget=700)) == 2
    assert len(take_rows(rows, (int(Op.CORE), 0), max_rows=8, token_budget=10)) == 1
    assert len(take_rows(rows, (int(Op.CORE), 0), max_rows=3, token_budget=10**6)) == 3


def test_dp_alignment_is_deterministic_and_rank_symmetric():
    a = {(int(Op.CORE), 1): 3, (int(Op.PRELUDE), -1): 1}
    b = {(int(Op.CORE), 1): 1, (int(Op.CODA), -1): 2}
    assert choose_op([a, b]) == choose_op([b, a]) == (int(Op.CORE), 1)
    assert choose_op([{}, {}]) is None
    assert NoAlign().align(a) == (int(Op.CORE), 1)


def test_dp_tie_goes_to_earlier_op_then_shallower_depth():
    tie = [{(int(Op.CORE), 2): 2}, {(int(Op.PRELUDE), -1): 2}, {(int(Op.CORE), 1): 2}]
    assert choose_op(tie) == (int(Op.PRELUDE), -1)


def test_dummy_step_has_no_rows():
    step = LoopStep.dummy(Op.CORE, depth=1, rows=4)
    assert step.is_dummy and step.req_ids == () and step.dummy_rows == 4 and not step.is_final
