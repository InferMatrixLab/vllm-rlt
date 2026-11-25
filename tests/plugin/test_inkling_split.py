"""The Inkling op split must visit exactly the layers vLLM's single forward does,
in the same order. This is checked on the host against vLLM's own ``forward``
visit construction; the GPU numerics are not (see the module docstring)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
inkling = pytest.importorskip("vllm_rlt.plugin.inkling")

from vllm_rlt.plugin.context import TokenState, state_cat, state_slice  # noqa: E402
from vllm_rlt.plugin.program import Op  # noqa: E402

Split = inkling.SplitInkling2LoopModel


def fake(layers=24, core=(6, 18), passes=3):
    m = SimpleNamespace(
        layers=[None] * layers,
        loop_core=range(*core),
        config=SimpleNamespace(num_loop_passes=passes),
    )
    m._visits = lambda: Split._visits(m)  # the real model has these as methods
    return m


def vllm_visits(m):
    """The visits construction from vLLM's Inkling2LoopModel.forward."""
    core, n = m.loop_core, m.config.num_loop_passes
    v = [(i, 0) for i in range(core.start)]
    v += [(i, p) for p in range(n) for i in core]
    v += [(i, 0) for i in range(core.stop, len(m.layers))]
    return v


def split_ops(m):
    return (
        [(Op.PRELUDE, 0)] + [(Op.CORE, d) for d in range(m.config.num_loop_passes)] + [(Op.CODA, 0)]
    )


@pytest.mark.parametrize("layers,core,passes", [(24, (6, 18), 3), (24, (6, 18), 1), (8, (2, 6), 4)])
def test_ops_partition_vllm_visits_exactly_once_in_order(layers, core, passes):
    m = fake(layers, core, passes)
    visits = Split._visits(m)
    assert visits == vllm_visits(m)
    assert len(visits) == layers - (core[1] - core[0]) + passes * (core[1] - core[0])
    covered = []
    for op, depth in split_ops(m):
        covered.extend(Split._op_range(m, op, depth))
    assert covered == list(range(len(visits)))


def test_each_op_runs_the_expected_layers():
    m = fake()
    visits = Split._visits(m)
    layers = lambda op, d: [visits[i] for i in Split._op_range(m, op, d)]  # noqa: E731
    assert layers(Op.PRELUDE, 0) == [(i, 0) for i in range(6)]
    assert layers(Op.CORE, 1) == [(i, 1) for i in range(6, 18)]
    assert layers(Op.CODA, 0) == [(i, 0) for i in range(18, 24)]


def test_token_state_with_missing_tensors_merges_and_slices():
    import torch

    def state(rows, with_bank):
        t = {
            "stack": torch.randn(5, rows, 4),
            "inputs": torch.randn(2, rows, 4),
            "prelude_partial": torch.randn(rows, 4),
        }
        if with_bank:
            t["bank"] = torch.randn(2, rows, 4)
        return TokenState(t, {k: inkling.DIMS[k] for k in t}, {"num_blocks": 3})

    both = state_cat([state(2, True), state(3, True)])
    assert both["stack"].shape == (5, 5, 4) and both["bank"].shape == (2, 5, 4)
    assert state_slice(both, 2, 3)["inputs"].shape == (2, 3, 4)
    no_bank = state_cat([state(2, False), state(1, False)])  # a CORE depth-0 state
    assert "bank" not in no_bank.tensors


def test_states_at_different_block_counts_cannot_share_a_step():
    import torch

    mk = lambda n: TokenState(  # noqa: E731
        {"stack": torch.zeros(2, 1, 4)}, {"stack": 1}, {"num_blocks": n}
    )
    with pytest.raises(ValueError, match="one op at one depth"):
        state_cat([mk(1), mk(2)])
