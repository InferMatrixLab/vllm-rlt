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
    assert walk(INKLING, 2) == [
        (Op.PRELUDE, 0),
        (Op.CORE, 0),
        (Op.CORE, 1),
        (Op.CORE, 2),
        (Op.CODA, 2),
    ]


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
    assert (
        ExitParams.from_extra_args({"rlt_target_loops": 2, "rlt_exit_threshold": 0.5}).limit(4) == 2
    )
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


def _config_for(arch: str, extra: dict | None = None, **hf):
    from types import SimpleNamespace

    hf_config = SimpleNamespace(architectures=[arch], **hf)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config), additional_config=extra or {}
    )


def test_layouts_by_architecture():
    from vllm_rlt.plugin.models import loop_layout

    text = type("T", (), {})()
    text.loop_config, text.num_loop_passes = {"num_passes": 3}, 3
    inkling = _config_for("Inkling2LoopForConditionalGeneration", text_config=text)
    assert loop_layout(inkling).full_only and loop_layout(inkling).num_passes == 3
    split = _config_for(
        "Inkling2LoopForConditionalGeneration", {"rlt_split_ops": True}, text_config=text
    )
    assert not loop_layout(split).full_only and loop_layout(split).has_prelude
    ouro = loop_layout(_config_for("OuroForCausalLM", total_ut_steps=4))
    assert not ouro.has_prelude and ouro.has_gate and ouro.num_passes == 4
    forced = _config_for("Anything", {"rlt_loop": {"num_passes": 2, "has_prelude": False}})
    assert loop_layout(forced).num_passes == 2
    with pytest.raises(ValueError, match="no loop layout"):
        loop_layout(_config_for("LlamaForCausalLM"))


# -- VarLoop vs Fixed, spans, ordering ------------------------------------------


def _inkling_cfg(*, varloop, share=False, passes=3, split=True):
    from types import SimpleNamespace

    text = SimpleNamespace(
        loop_config={"num_passes": passes, "is_varloop": varloop},
        num_loop_passes=passes,
        loop_share_cache=share,
    )
    hf = SimpleNamespace(architectures=["Inkling2LoopForConditionalGeneration"], text_config=text)
    extra = {"rlt_split_ops": True} if split else {}
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf), additional_config=extra)


def test_fixed_inkling_always_runs_every_pass():
    from vllm_rlt.plugin.models import loop_layout

    layout = loop_layout(_inkling_cfg(varloop=False))
    assert not layout.varloop and layout.num_passes == 3
    assert layout.limit(ExitParams.from_extra_args({"rlt_target_loops": 1})) == 3


def test_varloop_inkling_exits_by_pass_count_only_with_a_shared_cache():
    from vllm_rlt.plugin.models import loop_layout

    shared = loop_layout(_inkling_cfg(varloop=True, share=True))
    separate = loop_layout(_inkling_cfg(varloop=True, share=False))
    want = ExitParams.from_extra_args({"rlt_target_loops": 2})
    assert shared.varloop and shared.limit(want) == 2
    assert separate.limit(want) == 3  # a skipped pass would leave a hole in its own cache
    assert shared.limit(ExitParams()) == 3 and not shared.has_gate


def test_loop_features_turn_the_split_on():
    from vllm_rlt.plugin.models import loop_layout

    assert loop_layout(_inkling_cfg(varloop=False, split=False)).full_only
    cfg = _inkling_cfg(varloop=True, split=False)
    for feature in ("rlt_wcpb", "rlt_graphs"):
        cfg.additional_config = {feature: True}
        assert not loop_layout(cfg).full_only


def test_program_rank_orders_ops():
    layout = LoopLayout(num_passes=3)
    ranks = [layout.rank(Op.PRELUDE, 0)] + [layout.rank(Op.CORE, d) for d in range(3)]
    ranks.append(layout.rank(Op.CODA, 2))
    assert ranks == sorted(set(ranks)) and ranks == [0, 1, 2, 3, 4]


def test_a_chunk_waits_for_the_chunk_before_it():
    from vllm_rlt.plugin.plan import unblocked
    from vllm_rlt.plugin.request import Span

    layout = LoopLayout(num_passes=3)
    first = Span(0, 4, 0, Op.CORE, 0)
    second = Span(1, 4, 4, Op.CORE, 0)
    rank = lambda s: layout.rank(s.op, s.depth)  # noqa: E731
    assert unblocked([first, second], rank) == [first]  # same position: second waits
    first.depth = 1
    assert unblocked([first, second], rank) == [first, second]  # first moved on
    first.in_flight = True
    assert unblocked([first, second], rank) == [second]


def test_a_step_takes_one_span_per_request():
    rows = [
        Ready("a", Op.CORE, 0, 4, sid=0),
        Ready("a", Op.CORE, 0, 4, sid=1),
        Ready("b", Op.CORE, 0, 4, sid=0),
    ]
    picked = take_rows(rows, (int(Op.CORE), 0), max_rows=8, token_budget=100)
    assert [(r.req_id, r.sid) for r in picked] == [("a", 0), ("b", 0)]


def test_self_spec_alone_keeps_the_native_full_path():
    from vllm_rlt.plugin.models import loop_layout

    cfg = _inkling_cfg(varloop=True, split=False)
    cfg.additional_config = {"rlt_self_spec": True}
    assert loop_layout(cfg).full_only


def test_state_helpers_keep_the_token_axis():
    import torch

    from vllm_rlt.plugin.context import (
        Sink,
        TokenState,
        state_copy_,
        state_like,
        state_view,
    )

    state = TokenState(
        {"stack": torch.arange(2 * 3 * 4.0).reshape(2, 3, 4), "partial": torch.ones(3, 4)},
        {"stack": 1, "partial": 0},
        {"num_blocks": 1},
    )
    static = state_like(state, 8)
    assert static["stack"].shape == (2, 8, 4) and static["partial"].shape == (8, 4)
    state_copy_(static, state, 3)
    assert torch.equal(state_view(static, 3)["stack"], state["stack"])
    assert static.meta == {"num_blocks": 1}
    sink = Sink(8)
    sink.write(state, 3)
    assert torch.equal(sink.state["partial"][:3], state["partial"])
    plain = Sink(5)
    plain.write(torch.ones(2, 4), 2)
    assert plain.state.shape == (5, 4) and plain.state[:2].sum() == 8


def test_self_spec_method_registers_and_resolves():
    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        get_custom_speculator_config_class,
    )

    import vllm_rlt.plugin as plugin

    plugin.register()
    cls = get_custom_speculator_config_class("rlt_self")
    assert cls is not None and cls.speculator_class.endswith("LoopSpeculator")
    assert "speculator_options" in cls.supported_options


def test_kv_fill_copies_one_slot_across_passes_in_the_logical_layout():
    from types import SimpleNamespace

    import torch

    from vllm_rlt.plugin.kvfill import fill_skipped_passes

    blocks, heads, tokens, content = 4, 2, 8, 6
    caches = [torch.randn(blocks, heads, tokens, content) for _ in range(3)]
    attention = [[SimpleNamespace(kv_cache=c, layer_name=f"p{i}")] for i, c in enumerate(caches)]
    slots = torch.tensor([9, 18, -1])  # block 1 offset 1, block 2 offset 2, a pad row
    want = caches[0][[1, 2], :, [1, 2], :].clone()
    before = [c.clone() for c in caches]
    assert fill_skipped_passes(attention, lambda name: slots, exit_depth=0) == 2
    for p in (1, 2):
        assert torch.equal(caches[p][[1, 2], :, [1, 2], :], want)
        untouched = torch.ones_like(caches[p], dtype=torch.bool)
        untouched[[1, 2], :, [1, 2], :] = False
        assert torch.equal(caches[p][untouched], before[p][untouched])
    assert torch.equal(caches[0], before[0])
    assert fill_skipped_passes(attention, lambda name: slots, exit_depth=2) == 0


def test_mixed_step_keeps_rows_without_state_in_place():
    """A depth-0 row of a mixed CORE step has no carried state. The merged state
    must still hold one block per row in batch order (zeros for that row), or
    every later row reads another row's hidden states. Seen on a GPU: with mixed
    depths the third prompt's first token came out wrong."""
    import types

    import torch

    pytest.importorskip("vllm.v1.worker.gpu.model_runner")
    from vllm_rlt.plugin.runner import LoopModelRunner

    a = torch.full((2, 3), 1.0)
    c = torch.full((1, 3), 3.0)
    fake = types.SimpleNamespace(
        loop_state={("a", 0): a, ("c", 0): c},
        loop_layout=LoopLayout(num_passes=4, has_prelude=False, has_gate=True),
        device=torch.device("cpu"),
    )
    step = LoopStep(
        op=Op.CORE,
        depths=(2, 0, 1),
        req_ids=("a", "b", "c"),
        num_tokens=(2, 4, 1),
        can_exit=(False,) * 3,
        advanced=(False,) * 3,
        spans=(0, 0, 0),
    )
    out = types.SimpleNamespace(num_scheduled_tokens={"a": 2, "b": 4, "c": 1})
    ctx = LoopModelRunner._make_context(fake, out, step, ["a", "b", "c"], {"a": 0, "b": 0, "c": 0})
    assert ctx.state.shape == (7, 3)
    assert ctx.state[:2].eq(1).all() and ctx.state[2:6].eq(0).all() and ctx.state[6:].eq(3).all()
    assert ctx.row_depth.tolist() == [2, 2, 0, 0, 0, 0, 1]
