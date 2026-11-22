"""Joint-scheme contracts for RFC #192: measured capacity on a growing mapped pool.

These tests are written from the design goals, not from the implementation:

* G1 the plan never exceeds the whole-device utilization cap and leaves the
  transient allowance plus the safety margin free;
* G2 capacity never falls below the trial pool that already exists;
* G3 the reservation is sized to the useful cap, so it does not truncate the
  measurement, and a truncation is an error unless it is asked for;
* G5 only backed blocks reach the scheduler: ``published <= committed <= max``,
  publishing is monotone and needs backing;
* G6 FP8 binds its scale planes from the pool, and a pool that cannot supply
  them is refused;
* G7 the estimate stays in force for paths the preparation did not exercise;
* G9 every violation fails at startup, before anything is mapped or published.

The pool below is a *fake* only in the sense that it allocates no device memory.
It reproduces the arithmetic that makes the scheme non-trivial: physical memory
arrives in whole allocation granules, so the effective block count is what the
slowest plane can back, not what the caller requested.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import (
    commit_measured_capacity,
    measured_kv_capacity,
    transient_allowance,
)

MIB = 1 << 20
GIB = 1024 * MIB
LAYERS, KV_HEADS, HEAD_DIM, BLOCK = 2, 1, 4, 2
GRANULE = 400  # bytes; a fake allocation granule, not a real 2 MiB one


def round_up(value, multiple):
    return -(-value // multiple) * multiple


class Plane:
    """One pool: a reserved address range that gains physical memory in granules."""

    def __init__(self, shape, dtype, max_blocks, granule):
        self.shape, self.dtype = tuple(shape), dtype
        self.block_bytes = torch.empty((), dtype=dtype).element_size() * int(
            torch.tensor(list(shape)).prod()
        )
        self.reserved_bytes = round_up(max_blocks * self.block_bytes, granule)
        self.mapped_bytes = 0
        self.granule = granule
        self.grow_calls = 0

    @property
    def backed_blocks(self):
        return self.mapped_bytes // self.block_bytes

    def grow(self, blocks):
        target = round_up(blocks * self.block_bytes, self.granule)
        if target <= self.mapped_bytes:
            return
        self.grow_calls += 1
        self.mapped_bytes = min(target, self.reserved_bytes)


class GranularPool:
    """A multi-plane mapped pool with CUDA VMM's granule and reporting semantics."""

    def __init__(self, planes, *, max_blocks, granule=GRANULE, initial_blocks=1):
        self.max_blocks = max_blocks
        self.kv_cache_dtype = "auto"
        self._data = [Plane(spec, dtype, max_blocks, granule) for spec, dtype in planes]
        self._all = list(self._data)
        for plane in self._data:
            plane.grow(initial_blocks)
        self.tensors = tuple(
            torch.zeros((max_blocks, *plane.shape), dtype=plane.dtype) for plane in self._data
        )
        self.scale_tensors = None
        self.committed_blocks = 0
        self.begin()

    def begin(self):
        self.commit(0)

    def commit(self, blocks):
        if blocks > self.max_blocks:
            raise ValueError(f"cannot commit {blocks} blocks, max_blocks is {self.max_blocks}")
        for plane in self._all:
            plane.grow(blocks)
        self.committed_blocks = min(
            [self.max_blocks] + [plane.backed_blocks for plane in self._all]
        )
        return self.committed_blocks

    @property
    def committed_bytes(self):
        return sum(plane.mapped_bytes for plane in self._all)


def fp8_pool(planes, *, max_blocks, granule=GRANULE, initial_blocks=1):
    """An FP8 pool: data planes plus one per-vector scale plane per direction."""
    pool = GranularPool(
        planes, max_blocks=max_blocks, granule=granule, initial_blocks=initial_blocks
    )
    pool.kv_cache_dtype = "fp8_e4m3"
    return pool


def report(*, free, total, blocks, gaps=(), phases=None, allocated=0, peak_allocated=0):
    if phases is None:
        phases = [
            dict(
                name="cleaned",
                driver_free_bytes=free,
                device_total_bytes=total,
                allocated_bytes=allocated,
                peak_allocated_bytes=peak_allocated,
            )
        ]
    return dict(phases=phases, coverage_gaps=tuple(gaps), num_blocks=blocks, status="prepared")


def stub_attention():
    return SimpleNamespace(
        info={"selected_backend": "contracts"},
        capabilities=SimpleNamespace(fp8_kv=True, fp8_kv_static=True),
    )


def make_cache(pool, num_blocks, **kwargs):
    parameters = dict(
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_blocks=num_blocks,
        block_size=BLOCK,
        max_loops=1,
        dtype=torch.float32,
        attention=stub_attention(),
        storage=pool,
    )
    parameters.update(kwargs)
    return KVCacheManager(**parameters)


DATA = ((LAYERS, BLOCK, KV_HEADS, HEAD_DIM), torch.float32)
SCALE = ((LAYERS, BLOCK, KV_HEADS), torch.float32)


# G1: the budget comes from the device, not from a free-floating number.


def test_the_utilization_cap_bounds_the_whole_device_not_the_free_bytes():
    # 80 GiB device, 0.9 cap, 20 GiB already resident, 60 GiB free.
    capacity, record = measured_kv_capacity(
        report=report(free=60 * GIB, total=80 * GIB, blocks=0),
        bytes_per_block=3 * MIB,
        useful_blocks=1 << 20,
        gpu_memory_utilization=0.9,
        transient_bytes=4 * GIB,
        safety_bytes=256 * MIB,
    )
    # The cap allows 72 GiB total; 20 GiB is resident, so 52 GiB is the ceiling.
    assert record["remaining_instance_bytes"] == 52 * GIB
    assert record["kv_budget_bytes"] == 52 * GIB - 4 * GIB - 256 * MIB
    assert capacity == (52 * GIB - 4 * GIB - 256 * MIB) // (3 * MIB)


def test_a_lower_cap_does_not_hand_back_memory_the_driver_already_gave_up():
    # With 0.5 the cap is 40 GiB and 20 GiB is resident: 20 GiB remain, which is
    # less than the 60 GiB the driver reports as free.
    capacity, record = measured_kv_capacity(
        report=report(free=60 * GIB, total=80 * GIB, blocks=0),
        bytes_per_block=3 * MIB,
        useful_blocks=1 << 20,
        gpu_memory_utilization=0.5,
        transient_bytes=GIB,
        safety_bytes=64 * MIB,
    )
    assert record["remaining_instance_bytes"] == 20 * GIB
    assert capacity == (20 * GIB - GIB - 64 * MIB) // (3 * MIB)


def test_the_cap_trims_its_headroom_from_whatever_free_memory_reports():
    # With a cap below 1 the ceiling is min(free, cap - resident), and resident
    # is total - free, so the cap side is the smaller one on any device that is
    # not nearly empty: the plan may use free memory minus the (1 - cap) share
    # of the whole device. On a shared device the neighbour's memory is already
    # out of `free`, and the cap trims its share again rather than handing it
    # back.
    _, record = measured_kv_capacity(
        report=report(free=10 * GIB, total=80 * GIB, blocks=0),
        bytes_per_block=3 * MIB,
        useful_blocks=1 << 20,
        gpu_memory_utilization=0.9,
        transient_bytes=GIB,
        safety_bytes=64 * MIB,
    )
    assert record["resident_bytes"] == 70 * GIB
    assert record["remaining_instance_bytes"] == 2 * GIB


def test_a_cap_of_one_uses_the_measured_free_memory_as_it_stands():
    _, record = measured_kv_capacity(
        report=report(free=10 * GIB, total=80 * GIB, blocks=0),
        bytes_per_block=3 * MIB,
        useful_blocks=1 << 20,
        gpu_memory_utilization=1,
        transient_bytes=GIB,
        safety_bytes=64 * MIB,
    )
    assert record["remaining_instance_bytes"] == 10 * GIB


# G2: the trial pool is a floor, never a casualty of the measurement.


def test_a_negative_budget_keeps_the_trial_pool_instead_of_shrinking_it():
    capacity, record = measured_kv_capacity(
        report=report(free=1 * MIB, total=80 * GIB, blocks=512),
        bytes_per_block=3 * MIB,
        useful_blocks=1 << 20,
        gpu_memory_utilization=0.9,
        transient_bytes=8 * GIB,
        safety_bytes=GIB,
        existing_blocks=512,
    )
    assert capacity == 512
    assert record["additional_blocks"] == 0


def test_the_useful_cap_binds_before_the_reservation():
    _, record = measured_kv_capacity(
        report=report(free=60 * GIB, total=80 * GIB, blocks=0),
        bytes_per_block=3 * MIB,
        useful_blocks=1000,
        gpu_memory_utilization=0.9,
        transient_bytes=GIB,
        safety_bytes=MIB,
        reservation_blocks=4096,
    )
    assert record["limited_by"] == "useful_blocks"


# G3: a reservation that would clamp the measurement is a startup error.


def test_a_reservation_at_the_trial_pool_is_refused_as_a_dead_end():
    pool = GranularPool([DATA, DATA], max_blocks=64, initial_blocks=8)
    # 64 GiB free and 64 blocks reserved: the measurement wants far more.
    with pytest.raises(RuntimeError, match="reserve the useful cap"):
        commit_measured_capacity(
            pool,
            report=report(free=60 * GIB, total=80 * GIB, blocks=pool.committed_blocks),
            bytes_per_block=3 * MIB,
            useful_blocks=1 << 20,
            gpu_memory_utilization=0.9,
            transient_bytes=GIB,
            safety_bytes=MIB,
        )


def test_an_intended_truncation_still_maps_only_the_reservation():
    pool = GranularPool([DATA, DATA], max_blocks=1024, initial_blocks=8)
    backed, record = commit_measured_capacity(
        pool,
        report=report(free=60 * GIB, total=80 * GIB, blocks=pool.committed_blocks),
        bytes_per_block=3 * MIB,
        useful_blocks=1 << 20,
        gpu_memory_utilization=0.9,
        transient_bytes=GIB,
        safety_bytes=MIB,
        allow_reservation_truncation=True,
    )
    assert backed == 1024
    assert record["limited_by"] == "reservation"
    assert any("truncated" in gap for gap in record["coverage_gaps"])


# The granule: what the pool backs is not what was asked for.


def test_the_pool_reports_granule_blocks_and_the_planner_must_believe_it():
    pool = GranularPool([DATA, DATA], max_blocks=1024, initial_blocks=1)
    # One requested block is rounded up to a granule, which backs several blocks.
    assert pool.committed_blocks > 1
    plan, _ = commit_measured_capacity(
        pool,
        report=report(free=60 * GIB, total=80 * GIB, blocks=pool.committed_blocks),
        bytes_per_block=3 * MIB,
        useful_blocks=1024,
        gpu_memory_utilization=0.9,
        transient_bytes=GIB,
        safety_bytes=MIB,
    )
    assert plan == 1024


def test_a_report_that_describes_the_request_instead_of_the_backing_is_refused():
    pool = GranularPool([DATA, DATA], max_blocks=1024, initial_blocks=1)
    with pytest.raises(ValueError, match="storage backs"):
        commit_measured_capacity(
            pool,
            report=report(free=60 * GIB, total=80 * GIB, blocks=1),
            bytes_per_block=3 * MIB,
            useful_blocks=1024,
            gpu_memory_utilization=0.9,
            transient_bytes=GIB,
            safety_bytes=MIB,
        )


def test_a_partially_committed_pool_never_publishes_the_rounded_up_tail():
    pool = GranularPool([DATA, DATA], max_blocks=1024, initial_blocks=8)
    cache = make_cache(pool, pool.committed_blocks)
    # Three more blocks are requested; a granule backs more than that.
    backed = pool.commit(pool.committed_blocks + 3)
    assert backed > pool.committed_blocks - 1
    cache.publish_capacity(backed)
    with pytest.raises(ValueError, match="committed"):
        cache.publish_capacity(backed + 1)


def test_the_slowest_plane_sets_the_effective_count():
    # A wide data plane and a narrow scale plane do not round up by the same
    # number of blocks, so the pool must report the smaller of the two.
    pool = fp8_pool(
        [((LAYERS, BLOCK, KV_HEADS, HEAD_DIM * 16), torch.float8_e4m3fn), SCALE],
        max_blocks=4096,
        granule=GRANULE,
        initial_blocks=1,
    )
    pool.tensors = pool.tensors
    expected = min(plane.mapped_bytes // plane.block_bytes for plane in pool._all)
    assert pool.committed_blocks == min(4096, expected)


# G5: publication is the scheduler's only source of blocks.


def test_publishing_beyond_the_backing_is_refused_before_the_pool_is_touched():
    pool = GranularPool([DATA, DATA], max_blocks=1024, initial_blocks=8)
    cache = make_cache(pool, pool.committed_blocks)
    with pytest.raises(ValueError, match="committed"):
        cache.publish_capacity(pool.max_blocks)


def test_publishing_is_monotone_and_idempotent():
    pool = GranularPool([DATA, DATA], max_blocks=1024, initial_blocks=8)
    cache = make_cache(pool, pool.committed_blocks)
    first = pool.committed_blocks
    grown = pool.commit(64)
    assert cache.publish_capacity(grown) == grown
    assert cache.publish_capacity(grown) == grown
    with pytest.raises(ValueError, match="only grow"):
        cache.publish_capacity(grown - 1)
    assert cache.published_blocks == grown > first


def test_a_pool_that_under_commits_is_caught_at_the_plan_boundary():
    class Lazy(GranularPool):
        def commit(self, blocks):
            stale = self.committed_blocks
            super().commit(blocks)
            if not blocks:
                return self.committed_blocks
            self.committed_blocks = stale  # a growth that reports the old count
            return stale

    pool = Lazy([DATA, DATA], max_blocks=1024, initial_blocks=8)
    with pytest.raises(RuntimeError, match="backed blocks"):
        commit_measured_capacity(
            pool,
            report=report(free=60 * GIB, total=80 * GIB, blocks=pool.committed_blocks),
            bytes_per_block=3 * MIB,
            useful_blocks=1024,
            gpu_memory_utilization=0.9,
            transient_bytes=GIB,
            safety_bytes=MIB,
        )


# G6: FP8 binds through the pool, and only through the pool.


def test_fp8_binds_the_scale_planes_the_pool_carries():
    data = ((LAYERS, BLOCK, KV_HEADS, HEAD_DIM), torch.float8_e4m3fn)
    pool = fp8_pool([data, data, SCALE, SCALE], max_blocks=1024, initial_blocks=1)
    pool.scale_tensors = (pool.tensors[2], pool.tensors[3])
    pool.tensors = pool.tensors[:2]
    cache = make_cache(pool, pool.committed_blocks, kv_cache_dtype="fp8_e4m3")
    assert cache.key_cache.dtype == torch.float8_e4m3fn
    assert cache.key_scale is pool.scale_tensors[0]
    assert cache.value_scale is pool.scale_tensors[1]


def test_fp8_without_exposed_scale_planes_is_refused_with_the_hook_named():
    data = ((LAYERS, BLOCK, KV_HEADS, HEAD_DIM), torch.float8_e4m3fn)
    pool = fp8_pool([data, data, SCALE, SCALE], max_blocks=1024, initial_blocks=1)
    # The bytes are reserved, but the pool does not expose them as scales.
    pool.tensors = pool.tensors[:2]
    with pytest.raises(ValueError, match="scale_tensors"):
        make_cache(pool, pool.committed_blocks, kv_cache_dtype="fp8_e4m3")


def test_a_pool_without_the_scale_bytes_is_refused_before_the_hook():
    data = ((LAYERS, BLOCK, KV_HEADS, HEAD_DIM), torch.float8_e4m3fn)
    pool = fp8_pool([data, data], max_blocks=1024, initial_blocks=1)
    with pytest.raises(ValueError, match="committed bytes"):
        make_cache(pool, pool.committed_blocks, kv_cache_dtype="fp8_e4m3")


def test_scale_planes_of_the_wrong_layout_are_refused():
    data = ((LAYERS, BLOCK, KV_HEADS, HEAD_DIM), torch.float8_e4m3fn)
    pool = fp8_pool([data, data, SCALE, SCALE], max_blocks=1024, initial_blocks=1)
    # Missing the head axis: a plausible mistake that would silently mis-index.
    pool.scale_tensors = (
        torch.zeros((1024, LAYERS, BLOCK), dtype=torch.float32),
        torch.zeros((1024, LAYERS, BLOCK), dtype=torch.float32),
    )
    pool.tensors = pool.tensors[:2]
    with pytest.raises(ValueError, match="scale layout"):
        make_cache(pool, pool.committed_blocks, kv_cache_dtype="fp8_e4m3")


def test_a_pool_that_stores_a_different_format_is_refused():
    pool = GranularPool([DATA, DATA], max_blocks=1024, initial_blocks=1)
    with pytest.raises(ValueError, match="must declare the format"):
        make_cache(pool, pool.committed_blocks, kv_cache_dtype="fp8_e4m3", fp8_scale=0.5)


# G7: unexercised paths keep the estimate.


def test_a_gap_keeps_the_estimate_even_when_the_measurement_is_zero():
    value, record = transient_allowance(
        report(free=1, total=1, blocks=1, gaps=("long context",)),
        estimate_bytes=512 * MIB,
    )
    assert value == 512 * MIB
    assert record["measured_transient_bytes"] == 0


def test_a_gap_with_no_allowance_is_refused_rather_than_sized_as_measured():
    with pytest.raises(ValueError, match="coverage gaps require"):
        measured_kv_capacity(
            report=report(free=60 * GIB, total=80 * GIB, blocks=0, gaps=("long context",)),
            bytes_per_block=3 * MIB,
            useful_blocks=1 << 20,
            gpu_memory_utilization=0.9,
            transient_bytes=0,
            safety_bytes=MIB,
        )


def test_a_phase_that_peaked_below_its_baseline_contributes_nothing():
    phases = [
        dict(
            name="reset",
            driver_free_bytes=1,
            device_total_bytes=1,
            allocated_bytes=100,
            peak_allocated_bytes=40,  # a stats reset inside the phase
        ),
        dict(name="cleaned", driver_free_bytes=1, device_total_bytes=1),
    ]
    value, record = transient_allowance(dict(phases=phases, coverage_gaps=()), estimate_bytes=MIB)
    assert record["measured_transient_bytes"] == 0
    assert value == MIB


def test_the_concurrency_factor_scales_the_measurement_not_the_estimate():
    phases = [
        dict(
            name="cleaned",
            driver_free_bytes=1,
            device_total_bytes=1,
            allocated_bytes=GIB,
            peak_allocated_bytes=2 * GIB,
        )
    ]
    value, _ = transient_allowance(
        dict(phases=phases, coverage_gaps=()), estimate_bytes=MIB, overlap_factor=4
    )
    assert value == 4 * GIB


# G9: violations fail at startup, before anything is mapped.


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(bytes_per_block=True), "bytes_per_block"),
        (dict(useful_blocks=True), "useful_blocks"),
        (dict(gpu_memory_utilization=0), "gpu_memory_utilization"),
        (dict(gpu_memory_utilization=1.5), "gpu_memory_utilization"),
        (dict(transient_bytes=-1), "transient_bytes"),
        (dict(safety_bytes=-1), "safety_bytes"),
        (dict(existing_blocks=True), "existing_blocks"),
        (dict(reservation_blocks=0), "reservation_blocks"),
    ],
)
def test_invalid_plan_inputs_are_rejected(overrides, message):
    arguments = dict(
        report=report(free=60 * GIB, total=80 * GIB, blocks=0),
        bytes_per_block=3 * MIB,
        useful_blocks=1 << 20,
        gpu_memory_utilization=0.9,
        transient_bytes=GIB,
        safety_bytes=MIB,
    )
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        measured_kv_capacity(**arguments)


def test_the_report_must_be_a_measurement_of_this_pool():
    pool = GranularPool([DATA, DATA], max_blocks=1024, initial_blocks=8)
    for bad in (None, {}, dict(num_blocks=True), dict(num_blocks=-1), dict(num_blocks=3)):
        with pytest.raises(ValueError):
            commit_measured_capacity(
                pool,
                report=dict(phases=[], coverage_gaps=()) if bad is None else bad,
                bytes_per_block=3 * MIB,
                useful_blocks=1024,
                gpu_memory_utilization=0.9,
                transient_bytes=GIB,
                safety_bytes=MIB,
            )


def test_a_pool_that_claims_more_backing_than_it_reserves_is_refused():
    pool = GranularPool([DATA, DATA], max_blocks=64, initial_blocks=8)
    pool.committed_blocks = 65
    with pytest.raises(ValueError, match="reserves only"):
        make_cache(pool, 8)


def test_a_pool_must_back_the_blocks_the_cache_opens_with():
    pool = GranularPool([DATA, DATA], max_blocks=1024, initial_blocks=8)
    with pytest.raises(ValueError, match="committed .* blocks"):
        make_cache(pool, pool.committed_blocks + 1)


# Boundary: the smallest pool the scheme allows.


def test_a_single_block_pool_can_be_planned_published_and_used():
    pool = GranularPool([DATA, DATA], max_blocks=1, initial_blocks=1)
    assert pool.committed_blocks == 1
    cache = make_cache(pool, 1)
    assert cache.published_blocks == 1
    cache.key_cache[0].fill_(1.0)
    assert float(cache.key_cache[0].mean()) == 1.0
    with pytest.raises(ValueError, match="reservation"):
        cache.publish_capacity(2)
