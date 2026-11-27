from vllm_rlt.core.loop import LoopOp
from vllm_rlt.parallel import DummyLoopStep, choose_op, dummy_for_idle_rank


def test_choose_op_sums_rank_rows_and_retains_fixed_tie_break():
    assert choose_op([]) is None
    assert choose_op([(LoopOp.CORE, 0)]) is None
    assert choose_op([(LoopOp.CORE, 2), (LoopOp.CODA, 3), (LoopOp.CORE, 2)]) is LoopOp.CORE
    assert choose_op([(LoopOp.CODA, 3), (LoopOp.PRELUDE, 3)]) is LoopOp.PRELUDE


def test_idle_rank_uses_agreed_op_and_padded_rows():
    chosen = choose_op([(LoopOp.CORE, 32), (LoopOp.CODA, 4)])
    dummy = dummy_for_idle_rank(chosen, ready_rows=0, padded_rows=32)
    assert dummy == DummyLoopStep(LoopOp.CORE, 32)
    assert dummy_for_idle_rank(chosen, ready_rows=4, padded_rows=32) is None
    assert dummy_for_idle_rank(None, ready_rows=0, padded_rows=0) is None
