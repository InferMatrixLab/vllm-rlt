"""Final KV capacity arithmetic for startup sizing (CPU-only)."""

import pytest

from vllm_rlt.core.memory import budget_blocks, kv_capacity

MIB = 1024**2


def test_kv_capacity_subtracts_every_overhead_from_the_tighter_free_bound():
    overheads = {"peak": 3 * MIB, "graph": 5 * MIB, "metadata": 1 * MIB}
    # Free memory is the binding bound here: min(64, 100 - 20) = 64 MiB.
    blocks, budget = kv_capacity(
        free_bytes=64 * MIB,
        cap_bytes=100 * MIB,
        reserved_bytes=20 * MIB,
        per_block=MIB,
        overheads=overheads,
        useful_blocks=10**6,
    )
    assert budget == 64 * MIB - 9 * MIB
    assert blocks == 55


def test_kv_capacity_uses_utilization_cap_when_it_is_tighter():
    blocks, budget = kv_capacity(
        free_bytes=512 * MIB,
        cap_bytes=100 * MIB,
        reserved_bytes=20 * MIB,
        per_block=2 * MIB,
        overheads={"graph": 10 * MIB},
        useful_blocks=10**6,
    )
    assert budget == 70 * MIB
    assert blocks == 35


def test_kv_capacity_is_capped_by_useful_blocks_without_changing_budget():
    blocks, budget = kv_capacity(
        free_bytes=100 * MIB,
        cap_bytes=100 * MIB,
        reserved_bytes=0,
        per_block=MIB,
        overheads={},
        useful_blocks=7,
    )
    assert blocks == 7
    assert budget == 100 * MIB


def test_kv_capacity_raises_when_no_block_fits():
    with pytest.raises(ValueError, match="no memory remains for KV blocks"):
        kv_capacity(
            free_bytes=4 * MIB,
            cap_bytes=100 * MIB,
            reserved_bytes=0,
            per_block=MIB,
            overheads={"graph": 4 * MIB},
            useful_blocks=10,
        )


def test_budget_blocks_floors_partial_blocks():
    assert budget_blocks(5 * MIB + 1, MIB) == 5
    with pytest.raises(ValueError):
        budget_blocks(MIB - 1, MIB)
