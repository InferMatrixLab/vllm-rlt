"""Joint-scheme behaviour on a real device: measured capacity over a VMM KV pool.

Run with an explicitly selected device::

    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_rfc192_joint_gpu.py --run-gpu

Every test states a design goal and then measures it against the composed scheme
(measured planner + mapped storage + published capacity):

* G1 the growth is bounded by the device cap and the driver sees it;
* G4 addresses, strides and captured graph bounds never change when capacity
  grows, and a graph captured on the trial pool replays on the new blocks;
* G5 only backed blocks are visible, and the reservation beyond them is not
  mapped;
* G6 FP8 binds its scale planes and its byte accounting matches the pool;
* G9 a contract violation fails before anything is used, and a failed growth
  leaves a pool the next call can still use.

The driver is the instrument, not the torch allocator: the point of the scheme
is that this memory never appears in the allocator counters.
"""

import gc
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm_rlt.core.kv_storage")

from vllm_rlt.core.kv_storage import PoolSpec, StableKVStorage, cuda_driver  # noqa: E402

from vllm_rlt.core.kv_cache_manager import KVCacheManager, kv_bytes_per_block  # noqa: E402
from vllm_rlt.core.memory import (  # noqa: E402
    commit_measured_capacity,
    torch_attention_scratch_bytes,
)
from vllm_rlt.kernels.paged_attention import torch_paged_attention  # noqa: E402

MIB = 1 << 20
LAYERS, HEADS, DIM, BLOCK = 2, 8, 128, 16
PLANE_BYTES = LAYERS * BLOCK * HEADS * DIM * 2  # one plane, one block, bf16
BLOCK_BYTES = 2 * PLANE_BYTES
DEVICE = torch.device("cuda:0")
USEFUL = 4096
SAFETY = 256 * MIB
ESTIMATE = 512 * MIB

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def tidy():
    yield
    gc.collect()
    torch.cuda.synchronize(DEVICE)
    torch.cuda.empty_cache()


def sample(name="cleaned"):
    free, total = torch.cuda.mem_get_info(DEVICE)
    return dict(
        name=name,
        elapsed_seconds=0.0,
        driver_free_bytes=free,
        device_total_bytes=total,
        allocated_bytes=torch.cuda.memory_allocated(DEVICE),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(DEVICE),
    )


def report(pool):
    return dict(
        status="prepared",
        phases=[sample("initialized"), sample("cleaned")],
        coverage_gaps=(),
        num_blocks=pool.committed_blocks,
    )


def data_spec(dtype=torch.bfloat16):
    return PoolSpec((LAYERS, BLOCK, HEADS, DIM), dtype)


def make_cache(pool, num_blocks, **kwargs):
    parameters = dict(
        num_layers=LAYERS,
        num_kv_heads=HEADS,
        head_dim=DIM,
        num_blocks=num_blocks,
        block_size=BLOCK,
        max_loops=1,
        device=DEVICE,
        dtype=torch.bfloat16,
        storage=pool,
    )
    parameters.update(kwargs)
    return KVCacheManager(**parameters)


def plan_and_grow(pool, *, useful=USEFUL, bytes_per_block=BLOCK_BYTES, transient=ESTIMATE):
    free_before, _ = torch.cuda.mem_get_info(DEVICE)
    allocated_before = torch.cuda.memory_allocated(DEVICE)
    layout = [(t.data_ptr(), tuple(t.shape), tuple(t.stride())) for t in pool.tensors]
    backed, record = commit_measured_capacity(
        pool,
        report=report(pool),
        bytes_per_block=bytes_per_block,
        useful_blocks=useful,
        gpu_memory_utilization=0.9,
        transient_bytes=transient,
        safety_bytes=SAFETY,
    )
    free_after, _ = torch.cuda.mem_get_info(DEVICE)
    return dict(
        backed=backed,
        record=record,
        layout_stable=layout
        == [(t.data_ptr(), tuple(t.shape), tuple(t.stride())) for t in pool.tensors],
        driver_delta=free_before - free_after,
        allocated_delta=torch.cuda.memory_allocated(DEVICE) - allocated_before,
    )


def touch_block(block, committed, max_blocks=USEFUL):
    """Touch one block in a child process: an unmapped page poisons the context."""
    script = f"""
import torch
from vllm_rlt.core.kv_storage import PoolSpec, StableKVStorage
spec = PoolSpec(({LAYERS}, {BLOCK}, {HEADS}, {DIM}), torch.bfloat16)
pool = StableKVStorage(
    [spec, spec], max_blocks={max_blocks}, initial_blocks=1, device=torch.device("cuda:0")
)
pool.commit({committed})
pool.tensors[0][{block}].fill_(1.0)
torch.cuda.synchronize()
print("TOUCHED", pool.committed_blocks)
"""
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=os.environ
    )


# G1 + G4: the measurement becomes physical memory, and nothing moves.


def test_the_measured_plan_grows_the_pool_to_the_useful_cap_in_place():
    pool = StableKVStorage(
        [data_spec(), data_spec()], max_blocks=USEFUL, initial_blocks=1, device=DEVICE
    )
    trial = pool.committed_blocks
    assert trial < USEFUL
    growth = plan_and_grow(pool)
    assert growth["backed"] == USEFUL
    assert growth["record"]["limited_by"] == "useful_blocks"
    assert growth["layout_stable"]
    expected = (USEFUL - trial) * BLOCK_BYTES
    assert abs(growth["driver_delta"] - expected) <= 32 * MIB
    assert growth["allocated_delta"] == 0


def test_the_grown_blocks_hold_data_and_the_rest_of_the_reservation_is_unmapped():
    usable = 512
    pool = StableKVStorage(
        [data_spec(), data_spec()], max_blocks=USEFUL, initial_blocks=1, device=DEVICE
    )
    growth = plan_and_grow(pool, useful=usable)
    assert growth["backed"] == usable
    cache = make_cache(pool, pool.committed_blocks)
    cache.publish_capacity(growth["backed"])
    last = cache.published_blocks - 1
    cache.key_cache[last].fill_(1.0)
    cache.value_cache[last].fill_(-1.0)
    torch.cuda.synchronize(DEVICE)
    assert float(cache.key_cache[last].float().mean()) == 1.0
    assert float(cache.value_cache[last].float().mean()) == -1.0
    # A block the plan never paid for is address space without memory.
    fault = touch_block(USEFUL - 1, usable)
    assert fault.returncode != 0, f"uncommitted block was writable: {fault.stdout}"
    assert "CUDA" in fault.stderr


# G4: a graph captured on the trial pool still describes the grown pool.


def test_a_graph_captured_before_the_growth_writes_the_new_blocks():
    pool = StableKVStorage(
        [data_spec(), data_spec()], max_blocks=USEFUL, initial_blocks=1, device=DEVICE
    )
    keys = pool.tensors[0]
    target = torch.tensor([USEFUL - 1], device=DEVICE)
    payload = torch.ones((1, LAYERS, BLOCK, HEADS, DIM), dtype=torch.bfloat16, device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize(DEVICE)
    with torch.cuda.graph(graph):
        keys.index_put_((target,), payload)
    plan_and_grow(pool)
    graph.replay()
    torch.cuda.synchronize(DEVICE)
    assert bool((keys[USEFUL - 1] == 1).all())
    assert float(keys[0].float().abs().sum()) == 0.0


def test_a_gathered_read_of_new_blocks_matches_the_eager_answer():
    pool = StableKVStorage(
        [data_spec(), data_spec()], max_blocks=USEFUL, initial_blocks=1, device=DEVICE
    )
    keys = pool.tensors[0]
    table = torch.tensor([0, USEFUL - 1], device=DEVICE)
    query = torch.randn(HEADS, DIM, dtype=torch.bfloat16, device=DEVICE)

    def gathered():
        flat = keys.index_select(0, table).reshape(2 * LAYERS * BLOCK, HEADS, DIM).float()
        scores = (flat * query.float()).sum(-1)
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * flat).sum(0)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize(DEVICE)
    with torch.cuda.graph(graph):
        captured = gathered()
    plan_and_grow(pool)
    keys[0] = 1.0
    keys[USEFUL - 1] = 2.0
    graph.replay()
    torch.cuda.synchronize(DEVICE)
    # The same read, run eagerly on the same data, must agree with the capture.
    eager = gathered()
    assert torch.allclose(captured, eager, atol=1e-3)
    # And it reads live memory: changing the far block changes the answer, which
    # a graph holding a copy of the trial pool could not do.
    keys[USEFUL - 1] = 0.0
    graph.replay()
    torch.cuda.synchronize(DEVICE)
    assert torch.allclose(captured, gathered(), atol=1e-3)
    assert not torch.allclose(captured, eager, atol=1e-3)


# G6: FP8 keeps its scales on the pool and its byte accounting matches.


class Fp8Source:
    """The adapter an integration needs: the storage exposes only ``tensors``."""

    kv_cache_dtype = "fp8_e4m3"

    def __init__(self, storage):
        self._storage = storage
        self.tensors = storage.tensors[:2]
        self.scale_tensors = storage.tensors[2:]

    @property
    def max_blocks(self):
        return self._storage.max_blocks

    @property
    def committed_blocks(self):
        return self._storage.committed_blocks

    @property
    def committed_bytes(self):
        return self._storage.committed_bytes

    def commit(self, blocks):
        return self._storage.commit(blocks)


def test_fp8_grows_with_scale_planes_and_matches_the_block_accounting():
    data = PoolSpec((LAYERS, BLOCK, HEADS, DIM), torch.float8_e4m3fn)
    scale = PoolSpec((LAYERS, BLOCK, HEADS), torch.float32)
    usable = 1024
    storage = StableKVStorage(
        [data, data, scale, scale], max_blocks=usable, initial_blocks=1, device=DEVICE
    )
    source = Fp8Source(storage)
    per_block = kv_bytes_per_block(LAYERS, BLOCK, HEADS, DIM, 1, "fp8_e4m3")
    growth = plan_and_grow(source, useful=usable, bytes_per_block=per_block, transient=MIB)
    assert growth["backed"] == usable
    mapped = storage.committed_bytes
    assert mapped >= usable * per_block
    assert mapped - usable * per_block <= 8 * MIB  # four planes, one granule each

    cache = make_cache(source, source.committed_blocks, kv_cache_dtype="fp8_e4m3")
    cache.publish_capacity(growth["backed"])
    assert cache.key_scale is storage.tensors[2]
    assert cache.value_scale is storage.tensors[3]
    last = cache.published_blocks - 1
    cache.key_cache[last].fill_(1.0)
    cache.key_scale[last].fill_(0.5)
    torch.cuda.synchronize(DEVICE)
    assert float(cache.key_scale[last].mean()) == 0.5


# G1 from the other side: the reservation is address space only.


def test_a_reservation_larger_than_the_card_is_address_space_not_memory():
    huge = 262144  # 16 GiB per plane, more than the card holds for KV here
    free_before, _ = torch.cuda.mem_get_info(DEVICE)
    storage = StableKVStorage(
        [data_spec(), data_spec()], max_blocks=huge, initial_blocks=1, device=DEVICE
    )
    free_after, _ = torch.cuda.mem_get_info(DEVICE)
    assert storage.tensors[0].shape[0] == huge
    assert free_before - free_after < 64 * MIB
    committed = storage.commit(64)
    assert committed >= 64
    assert storage.committed_bytes <= 64 * BLOCK_BYTES + 64 * MIB


# G9: violations fail before use, and a failed growth stays recoverable.


@pytest.mark.xfail(strict=True, reason="open in #194: commit() accepts a negative count")
def test_commit_rejects_counts_outside_the_reservation():
    pool = StableKVStorage(
        [data_spec(), data_spec()], max_blocks=64, initial_blocks=1, device=DEVICE
    )
    with pytest.raises(ValueError, match="max_blocks"):
        pool.commit(65)
    with pytest.raises(ValueError):
        pool.commit(-1)


def test_a_failed_growth_keeps_the_pool_usable(monkeypatch):
    pool = StableKVStorage(
        [data_spec(), data_spec()], max_blocks=4096, initial_blocks=1, device=DEVICE
    )
    before = pool.committed_blocks
    driver = cuda_driver()
    real_map = driver.map
    calls = {"n": 0}

    def flaky(address, size, handle):
        calls["n"] += 1
        if calls["n"] == 2:  # the second plane fails after the first one grew
            raise RuntimeError("CUDA driver error 2: out of memory")
        return real_map(address, size, handle)

    monkeypatch.setattr(driver, "map", flaky)
    with pytest.raises(RuntimeError, match="out of memory"):
        pool.commit(2048)
    monkeypatch.setattr(driver, "map", real_map)
    assert pool.committed_blocks == before
    assert pool.commit(before + 8) >= before


@pytest.mark.xfail(strict=True, reason="open in #194: a failed growth strands the partial mapping")
def test_a_failed_growth_does_not_hold_memory_its_capacity_cannot_use(monkeypatch):
    pool = StableKVStorage(
        [data_spec(), data_spec()], max_blocks=4096, initial_blocks=1, device=DEVICE
    )
    driver = cuda_driver()
    real_map = driver.map
    calls = {"n": 0}

    def flaky(address, size, handle):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("CUDA driver error 2: out of memory")
        return real_map(address, size, handle)

    monkeypatch.setattr(driver, "map", flaky)
    with pytest.raises(RuntimeError, match="out of memory"):
        pool.commit(2048)
    monkeypatch.setattr(driver, "map", real_map)
    # Physical memory the reported block count cannot use is stranded: the pool
    # cannot shrink, and the mapping only comes back when the pool is released.
    stranded = pool.committed_bytes - pool.committed_blocks * BLOCK_BYTES
    assert stranded <= 8 * MIB, f"{stranded / MIB:.1f} MiB are stranded after a failed growth"


def test_the_plan_refuses_a_report_that_does_not_describe_this_pool():
    pool = StableKVStorage(
        [data_spec(), data_spec()], max_blocks=1024, initial_blocks=1, device=DEVICE
    )
    with pytest.raises(ValueError, match="storage backs"):
        commit_measured_capacity(
            pool,
            report=dict(num_blocks=1, phases=[sample()], coverage_gaps=()),
            bytes_per_block=BLOCK_BYTES,
            useful_blocks=1024,
            gpu_memory_utilization=0.9,
            transient_bytes=MIB,
            safety_bytes=MIB,
        )
    assert pool.committed_blocks > 1


# The estimate that stays in force for uncovered paths has to be honest about
# what the kernel it stands for actually allocates.


def test_the_reference_attention_reserve_bounds_the_kernel_it_sizes():
    length = 4096
    blocks = length // BLOCK
    key = torch.zeros((blocks, BLOCK, HEADS, DIM), dtype=torch.bfloat16, device=DEVICE)
    value = torch.zeros_like(key)
    query = torch.randn(1, HEADS, DIM, dtype=torch.bfloat16, device=DEVICE)
    table = torch.arange(blocks, device=DEVICE).unsqueeze(0)
    lengths = torch.tensor([length], device=DEVICE)
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    baseline = torch.cuda.memory_allocated(DEVICE)
    torch_paged_attention(query, key, value, table, lengths)
    torch.cuda.synchronize(DEVICE)
    measured = torch.cuda.max_memory_allocated(DEVICE) - baseline
    config = SimpleNamespace(
        max_position_embeddings=length,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        head_dim=DIM,
    )
    reserve = torch_attention_scratch_bytes(config, pool_element_size=2)
    # Cover the kernel, but not by a blanket multiple: the previous estimate was
    # 2.7 times this measurement, and that excess came out of the KV budget.
    assert measured <= reserve <= 1.1 * measured
