"""Measured KV capacity (RFC #192 step 2 handoff) and the publish gate.

``measured_kv_capacity`` consumes the step-1 preparation report; the expected
values below are hand-computed from the KV layout and the stated margins, not
from the implementation.
"""

import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import measured_kv_capacity

MIB = 1 << 20
GIB = 1024 * MIB
LAYERS, KV_HEADS, HEAD_DIM, BLOCK = 2, 1, 4, 2


def report(*, free, total, gaps=()):
    """One drained, post-cleanup sample as ``prepare_runtime`` reports it."""
    return dict(
        phases=[
            dict(
                name="prepared",
                driver_free_bytes=free,
                device_total_bytes=total,
                allocated_bytes=0,
                reserved_bytes=0,
            )
        ],
        coverage_gaps=tuple(gaps),
    )


def plan(**overrides):
    parameters = dict(
        report=report(free=60 * GIB, total=80 * GIB),
        bytes_per_block=3 * MIB,
        useful_blocks=10**6,
        gpu_memory_utilization=0.9,
        transient_bytes=4 * GIB,
        safety_bytes=256 * MIB,
    )
    parameters.update(overrides)
    return measured_kv_capacity(**parameters)


def test_blocks_follow_the_measured_budget_without_double_counting():
    blocks, record = plan()
    # resident = 80 - 60 = 20 GiB; cap = 72 GiB; remaining = min(60, 72 - 20) = 52 GiB
    # budget = 52 GiB - 4 GiB - 256 MiB = 48896 MiB; 48896 // 3 = 16298 blocks
    assert (record["resident_bytes"], record["utilization_cap_bytes"]) == (20 * GIB, int(72 * GIB))
    assert record["remaining_instance_bytes"] == 52 * GIB
    assert record["kv_budget_bytes"] == 53248 * MIB - 4096 * MIB - 256 * MIB
    assert blocks == 16298
    assert record["source"] == "measured_warmup"
    assert blocks * record["bytes_per_block"] <= record["kv_budget_bytes"]


def test_driver_free_memory_is_bounded_by_the_utilization_cap():
    # Only 1 GiB is resident, so the 0.5 cap (40 GiB) binds before free memory.
    blocks, record = plan(
        report=report(free=79 * GIB, total=80 * GIB),
        gpu_memory_utilization=0.5,
        bytes_per_block=MIB,
        transient_bytes=0,
        safety_bytes=0,
    )
    assert record["remaining_instance_bytes"] == 39 * GIB
    assert blocks == 39 * 1024


def test_each_margin_reduces_capacity_exactly():
    free = 20 * GIB
    base, _ = plan(
        report=report(free=free, total=80 * GIB),
        transient_bytes=0,
        safety_bytes=0,
        gpu_memory_utilization=1.0,
    )
    transient, _ = plan(
        report=report(free=free, total=80 * GIB),
        transient_bytes=1024 * 3 * MIB,
        safety_bytes=0,
        gpu_memory_utilization=1.0,
    )
    safety, _ = plan(
        report=report(free=free, total=80 * GIB),
        transient_bytes=0,
        safety_bytes=1024 * 3 * MIB,
        gpu_memory_utilization=1.0,
    )
    assert base - transient == 1024
    assert base - safety == 1024


def test_useful_blocks_cap_the_measured_budget():
    blocks, record = plan(useful_blocks=100)
    assert blocks == 100
    assert record["max_useful_blocks"] == 100


def test_coverage_gaps_travel_with_the_plan():
    gaps = ("contexts beyond the reported workloads retain the estimated reserve",)
    _, record = plan(report=report(free=60 * GIB, total=80 * GIB, gaps=gaps))
    assert record["coverage_gaps"] == gaps


def test_a_budget_below_one_block_is_rejected_with_the_margins():
    with pytest.raises(ValueError, match="remaining=.*transient=.*safety=.*block="):
        plan(
            report=report(free=8 * MIB, total=80 * GIB),
            transient_bytes=0,
            safety_bytes=0,
        )


@pytest.mark.parametrize(
    "bad_report",
    [None, {}, dict(phases=[]), dict(phases=[dict(driver_free_bytes=1)])],
)
def test_a_missing_or_incomplete_report_is_rejected(bad_report):
    with pytest.raises(ValueError, match="report|phase"):
        plan(report=bad_report)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(bytes_per_block=0), "bytes_per_block"),
        (dict(useful_blocks=0), "useful_blocks"),
        (dict(transient_bytes=-1), "transient_bytes"),
        (dict(safety_bytes=-1), "safety_bytes"),
        (dict(gpu_memory_utilization=0.0), "gpu_memory_utilization"),
    ],
)
def test_invalid_inputs_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        plan(**overrides)


class FakeStorage:
    """Mapped storage stand-in: only ``committed_blocks`` has backing."""

    def __init__(self, *, max_blocks, committed=0, dtype=torch.float32):
        element = torch.empty((), dtype=dtype).element_size()
        self.bytes_per_block = 2 * LAYERS * BLOCK * KV_HEADS * HEAD_DIM * element
        shape = (max_blocks, LAYERS, BLOCK, KV_HEADS, HEAD_DIM)
        self.tensors = (
            torch.zeros(shape, dtype=dtype),
            torch.zeros(shape, dtype=dtype),
        )
        self.max_blocks = max_blocks
        self.dtype = dtype
        self.committed_blocks = 0
        self.committed_bytes = 0
        self.commit(committed)

    def commit(self, blocks):
        self.committed_blocks = max(self.committed_blocks, blocks)
        self.committed_bytes = self.committed_blocks * self.bytes_per_block
        return self.committed_blocks


def make_cache(storage=None, num_blocks=4, **kwargs):
    parameters = dict(
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_blocks=num_blocks,
        block_size=BLOCK,
        max_loops=1,
    )
    parameters.update(kwargs)
    return KVCacheManager(**parameters, storage=storage)


def test_storage_backed_cache_starts_at_its_committed_capacity():
    storage = FakeStorage(max_blocks=10, committed=4)
    cache = make_cache(storage, num_blocks=4)
    assert (cache.published_blocks, cache.committed_blocks, cache.max_blocks) == (4, 4, 10)
    assert cache.num_blocks == 4
    assert cache.num_free_blocks == 4
    assert cache.key_cache.shape == storage.tensors[0].shape


def test_publish_requires_committed_memory_and_grows_the_free_pool():
    storage = FakeStorage(max_blocks=10, committed=2)
    cache = make_cache(storage, num_blocks=2)
    assert cache.allocate("a", 4)  # two published blocks
    assert cache.allocate("b", 2) is False
    with pytest.raises(ValueError, match="committed 2"):
        cache.publish_capacity(8)
    assert (cache.published_blocks, cache.num_free_blocks) == (2, 0)

    storage.commit(10)
    assert cache.publish_capacity(10) == 10
    assert cache.num_free_blocks == 8
    assert cache.allocate("b", 2)
    blocks = cache.get_block_table("b", 0)
    assert set(blocks) & set(range(2, 10)), "growth must hand out the newly backed blocks"
    key = torch.full((1, KV_HEADS, HEAD_DIM), 3.0)
    cache.write(0, ["b"], [0], [0], key, key + 1)
    assert torch.equal(cache.key_cache[blocks[0], 0, 0], key[0])


def test_publish_is_monotonic_and_bounded():
    storage = FakeStorage(max_blocks=10, committed=10)
    cache = make_cache(storage, num_blocks=2)
    assert cache.publish_capacity(6) == 6
    with pytest.raises(ValueError, match="only grow"):
        cache.publish_capacity(4)
    with pytest.raises(ValueError, match="reservation holds"):
        cache.publish_capacity(11)
    assert cache.publish_capacity(6) == 6  # idempotent
    assert cache.num_blocks == 6


def test_publish_requires_storage():
    cache = make_cache(None)
    with pytest.raises(RuntimeError, match="mapped storage"):
        cache.publish_capacity(8)


def test_plain_cache_keeps_fixed_capacity():
    cache = make_cache(None, num_blocks=6)
    assert cache.storage is None
    assert (cache.max_blocks, cache.committed_blocks, cache.published_blocks) == (6, 6, 6)


def test_storage_must_be_committed_before_the_cache_is_built():
    storage = FakeStorage(max_blocks=10, committed=2)
    with pytest.raises(ValueError, match="commit the pool"):
        make_cache(storage, num_blocks=4)


def test_storage_must_cover_the_requested_blocks_and_layout():
    coarse = FakeStorage(max_blocks=2, committed=2)
    with pytest.raises(ValueError, match="reserves"):
        make_cache(coarse, num_blocks=4)
    wrong_dtype = FakeStorage(max_blocks=10, committed=10, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="do not match the KV layout"):
        make_cache(wrong_dtype, num_blocks=4)


def test_existing_bootstrap_blocks_are_counted_once_in_the_total():
    # The bootstrap prefix is already mapped, so it sits outside `free` and must
    # extend the measured delta rather than being sized from free again.
    capacity, record = plan(existing_blocks=100)
    assert record["additional_blocks"] == 16298
    assert capacity == 16398
    assert record["existing_blocks"] == 100
    # Only the new part spends budget.
    assert (capacity - 100) * record["bytes_per_block"] <= record["kv_budget_bytes"]


def test_bootstrap_only_capacity_is_usable_without_growth():
    capacity, record = plan(
        report=report(free=8 * MIB, total=80 * GIB),
        existing_blocks=8,
        transient_bytes=0,
        safety_bytes=0,
    )
    assert capacity == 8
    assert record["additional_blocks"] == 0
    assert record["limited_by"] == "budget"


def test_the_reservation_truncates_the_measured_capacity():
    capacity, record = plan(reservation_blocks=1000)
    assert capacity == 1000
    assert record["limited_by"] == "reservation"
    assert any("exceeds the reservation" in gap for gap in record["coverage_gaps"])


def test_the_useful_block_cap_binds_before_the_reservation():
    capacity, record = plan(useful_blocks=100, reservation_blocks=1000)
    assert capacity == 100
    assert record["limited_by"] == "useful_blocks"
    assert not any("exceeds the reservation" in gap for gap in record["coverage_gaps"])


def test_coverage_gaps_reject_a_zero_transient_allowance():
    gaps = ("random block speculation retains the estimated transient reserve",)
    with pytest.raises(ValueError, match="coverage gaps require an explicit transient"):
        plan(
            report=report(free=60 * GIB, total=80 * GIB, gaps=gaps),
            transient_bytes=0,
            safety_bytes=0,
        )
    # The same report is accepted once an allowance is stated, and the gap travels with it.
    _, record = plan(report=report(free=60 * GIB, total=80 * GIB, gaps=gaps))
    assert record["coverage_gaps"] == gaps


def test_storage_may_not_report_more_committed_blocks_than_it_reserves():
    storage = FakeStorage(max_blocks=4, committed=4)
    storage.committed_blocks = 5  # a storage-side accounting bug
    with pytest.raises(ValueError, match="reserves only"):
        make_cache(storage, num_blocks=4)
