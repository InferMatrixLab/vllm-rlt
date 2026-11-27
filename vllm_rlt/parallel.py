"""Operation agreement for data-parallel loop steps."""

from collections.abc import Iterable
from dataclasses import dataclass

from vllm_rlt.core.loop import LoopOp


@dataclass(frozen=True, slots=True)
class DummyLoopStep:
    op: LoopOp
    rows: int

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError("a dummy loop step requires at least one row")


def choose_op(proposals: Iterable[tuple[LoopOp, int]]) -> LoopOp | None:
    """Choose among rank proposals, breaking ties in Prelude/Core/Coda order."""
    counts = dict.fromkeys(LoopOp, 0)
    for op, rows in proposals:
        if rows < 0:
            raise ValueError("a rank proposal cannot have a negative row count")
        counts[op] += rows
    chosen = max(counts, key=counts.__getitem__)
    return chosen if counts[chosen] else None


def dummy_for_idle_rank(
    op: LoopOp | None, ready_rows: int, padded_rows: int
) -> DummyLoopStep | None:
    """Describe the agreed op on a rank that has no rows for it."""
    if op is None or ready_rows:
        return None
    return DummyLoopStep(op, padded_rows)
