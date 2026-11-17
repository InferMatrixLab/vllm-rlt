"""Shared prefix proofs must detach under individual mutation or snapshots."""

from copy import deepcopy
from types import SimpleNamespace

from vllm_rlt.core.kv_cache_manager import _CohortWrittenPositions, _WrittenPositions


def test_cohort_frontier_mutation_and_snapshot_isolation():
    frontier = SimpleNamespace(end=3)
    left = _CohortWrittenPositions(_WrittenPositions(prefix=2), frontier)
    right = _CohortWrittenPositions(_WrittenPositions(prefix=2), frontier)
    snapshot = deepcopy([left, right])
    assert left.prefix == right.prefix == 3
    left.add(7)
    frontier.end = 4
    assert left.prefix == 3 and left.pending == {7}
    assert right.prefix == 4
    assert [x.prefix for x in snapshot] == [3, 3]
    right.prefix = 1
    frontier.end = 8
    assert right.prefix == 1 and left.prefix == 3
    left.add_range(3, 7)
    assert left.prefix == 8 and not left.pending


def test_rebinding_keeps_previous_cohort_rows_independent():
    old = SimpleNamespace(end=5)
    moved = _CohortWrittenPositions(_WrittenPositions(), old)
    survivor = _CohortWrittenPositions(_WrittenPositions(), old)
    new = SimpleNamespace(end=6)
    moved = _CohortWrittenPositions(moved, new)
    old.end = 7
    assert moved.prefix == 6 and survivor.prefix == 7
    moved.add_range(6, 8)
    new.end = 20
    assert moved.prefix == 8 and survivor.prefix == 7
