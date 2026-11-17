"""CUDA VMM KV storage: stable address, growable backing, publishable blocks.

Run with an explicitly selected device:
    CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_vmm_pool.py --run-gpu

Commits are padded to the driver allocation granularity, so a case that must
leave blocks uncommitted needs block bytes far above that granularity.
"""

import gc

import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.vmm_pool import GPUKVStorage

pytestmark = pytest.mark.gpu

MIB = 1 << 20
# Small blocks keep the bookkeeping tests cheap.
SMALL = dict(num_layers=2, num_kv_heads=1, head_dim=4, block_size=2)
# ~16 MiB per block, so granularity padding cannot cover the whole reservation.
WIDE = dict(num_layers=2, num_kv_heads=8, head_dim=1024, block_size=128)
# 256 MiB of K+V per block: eight blocks map 2 GiB, enough signal to observe a
# release on a device shared with other tenants.
HEAVY = dict(num_layers=2, num_kv_heads=8, head_dim=2048, block_size=1024)


def build(*, max_blocks=64, dims=SMALL, committed=0, published=None, device=0):
    """Return (storage, cache) with ``committed`` blocks mapped and ``published`` live."""
    element = torch.empty((), dtype=torch.float32).element_size()
    storage = GPUKVStorage(
        max_blocks=max_blocks,
        bytes_per_block=(
            2
            * dims["num_layers"]
            * dims["block_size"]
            * dims["num_kv_heads"]
            * dims["head_dim"]
            * element
        ),
        device=f"cuda:{device}",
        dtype=torch.float32,
    )
    if committed:
        storage.commit_blocks(committed)
    published = published or storage.committed_blocks
    if not published:
        # No physical memory yet, so there is nothing a cache could publish.
        return storage, None
    cache = KVCacheManager(
        num_layers=dims["num_layers"],
        num_kv_heads=dims["num_kv_heads"],
        head_dim=dims["head_dim"],
        num_blocks=published or storage.committed_blocks,
        block_size=dims["block_size"],
        max_loops=1,
        device=storage.device,
        dtype=storage.dtype,
        storage=storage,
    )
    return storage, cache


def free_bytes():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


def test_reserve_costs_nothing_and_commit_costs_physical_memory():
    base = free_bytes()
    storage, _ = build(max_blocks=8, dims=HEAVY)
    try:
        # Reserving only takes address space; nothing is mapped yet.
        assert storage.committed_blocks == 0
        assert storage.committed_bytes == 0
        assert base - free_bytes() <= 8 * MIB, "reserving VA must not pin physical memory"
        storage.commit_blocks(2)
        assert storage.committed_blocks == 2
        # Expected cost derived from the KV layout, not from the pool's own
        # counter: a plane sized as K+V instead of half would double this.
        element = torch.empty((), dtype=torch.float32).element_size()
        per_block = (
            2
            * HEAVY["num_layers"]
            * HEAVY["block_size"]
            * HEAVY["num_kv_heads"]
            * HEAVY["head_dim"]
            * element
        )
        plane = -(-(2 * (per_block // 2)) // storage.allocation_granularity)
        expected = 2 * plane * storage.allocation_granularity
        assert expected == 512 * MIB  # a signal a shared device cannot mask
        assert storage.committed_bytes == expected
        assert abs((base - free_bytes()) - expected) <= 64 * MIB
    finally:
        storage.close()
    torch.cuda.empty_cache()


def test_committed_bytes_follow_the_kv_layout():
    dims = WIDE
    element = torch.empty((), dtype=torch.float32).element_size()
    k_and_v_per_block = (
        2
        * dims["num_layers"]
        * dims["block_size"]
        * dims["num_kv_heads"]
        * dims["head_dim"]
        * element
    )
    storage, _ = build(max_blocks=8, dims=dims)
    try:
        # bytes_per_block counts K and V together, so each plane holds half.
        assert storage.bytes_per_block == k_and_v_per_block
        assert 2 * storage.plane_bytes_per_block == k_and_v_per_block
        storage.commit_blocks(4)
        plane = storage.plane_bytes_per_block * 4
        granularity = storage.allocation_granularity
        mapped_plane = -(-plane // granularity) * granularity
        assert storage.committed_bytes == 2 * mapped_plane
        assert storage.reserve_bytes == 2 * storage.plane_bytes
        assert storage.plane_bytes >= mapped_plane
    finally:
        storage.close()


def test_tensors_keep_address_shape_and_stride_across_commit():
    storage, cache = build(max_blocks=8, dims=WIDE, committed=2)
    try:
        before = [
            (t.data_ptr(), tuple(t.shape), t.stride()) for t in (cache.key_cache, cache.value_cache)
        ]
        storage.commit_blocks(8)
        cache.publish_capacity(8)
        after = [
            (t.data_ptr(), tuple(t.shape), t.stride()) for t in (cache.key_cache, cache.value_cache)
        ]
        assert before == after
        # The two planes must not overlap.
        keys_end = cache.key_cache.data_ptr() + cache.key_cache.numel() * 4
        assert keys_end <= cache.value_cache.data_ptr()
    finally:
        storage.close()


def test_existing_blocks_survive_growth_and_new_blocks_are_writable():
    # WIDE blocks are far bigger than the map granularity, so the commit really
    # starts small and the growth below is a genuine second mapping.
    storage, cache = build(max_blocks=8, dims=WIDE, committed=2)
    try:
        assert storage.committed_blocks == 2
        assert cache.allocate("a", WIDE["block_size"])
        shape = (1, WIDE["num_kv_heads"], WIDE["head_dim"])
        key = torch.full(shape, 5.0, device=storage.device)
        cache.write(0, ["a"], [0], [0], key, key + 1)
        torch.cuda.synchronize()
        # Only the committed prefix is mapped, so snapshot that much: cloning the
        # whole view would copy the unallocated tail and fault.
        mapped = storage.committed_blocks
        old = cache.key_cache[:mapped].clone()

        storage.commit_blocks(8)
        cache.publish_capacity(8)
        assert torch.equal(cache.key_cache[:mapped], old), "growth must not disturb mapped blocks"

        assert cache.allocate("b", WIDE["block_size"])
        blocks = cache.get_block_table("b", 0)
        assert max(blocks) >= 2, "the new allocation must come from the newly published blocks"
        fresh = torch.full(shape, 7.0, device=storage.device)
        cache.write(0, ["b"], [0], [0], fresh, fresh + 1)
        keys, values = cache.read(0, "b", 0, length=1)
        assert torch.equal(keys[0], fresh[0])
        assert torch.equal(values[0], fresh[0] + 1)
    finally:
        storage.close()


def test_published_blocks_never_exceed_committed_memory():
    storage, _ = build(max_blocks=8, dims=WIDE, committed=2)
    try:
        committed = storage.committed_blocks
        assert committed < 8, "granularity padding must leave blocks uncommitted here"
        cache = KVCacheManager(
            num_layers=WIDE["num_layers"],
            num_kv_heads=WIDE["num_kv_heads"],
            head_dim=WIDE["head_dim"],
            num_blocks=committed,
            block_size=WIDE["block_size"],
            max_loops=1,
            device=storage.device,
            dtype=storage.dtype,
            storage=storage,
        )
        with pytest.raises(ValueError, match="committed"):
            cache.publish_capacity(8)
        assert cache.published_blocks == committed
        assert cache.num_free_blocks == committed
        storage.commit_blocks(8)
        assert cache.publish_capacity(8) == 8
        assert cache.num_free_blocks == 8
    finally:
        storage.close()


def test_graph_captured_before_growth_replays_on_the_kv_plane():
    storage, cache = build(max_blocks=8, dims=WIDE, committed=4)
    try:
        assert cache.allocate("a", WIDE["block_size"])
        block = cache.get_block_table("a", 0)[0]
        window = cache.key_cache[block, 0, 0]
        window.zero_()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            window.add_(1.0)  # warm outside capture
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            window.mul_(2.0).add_(3.0)
        window.zero_()
        graph.replay()
        torch.cuda.synchronize()
        expected = window.clone()

        storage.commit_blocks(8)  # append physical memory after capture
        cache.publish_capacity(8)
        window.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(window, expected)
    finally:
        storage.close()


def test_close_releases_the_reservation_and_is_idempotent():
    # A 2 GiB mapping gives a release signal far above neighbour noise on a shared
    # device, and the window is measured around close() only.
    storage, _ = build(max_blocks=8, dims=HEAVY)
    storage.commit_blocks(8)
    torch.cuda.synchronize()
    committed = storage.committed_bytes
    assert committed == 2 << 30
    held = free_bytes()
    storage.close()
    storage.close()  # idempotent
    torch.cuda.empty_cache()
    assert storage.committed_bytes == 0
    assert storage.committed_blocks == 0
    assert free_bytes() - held >= committed - 64 * MIB, "close must return the mapping"
    with pytest.raises(RuntimeError, match="closed"):
        storage.commit_blocks(4)


def test_views_do_not_free_the_mapping():
    storage, cache = build(committed=4)
    try:
        address = cache.key_cache.data_ptr()
        probe = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=storage.device)
        cache.key_cache[0, 0, 0].copy_(probe)
        torch.cuda.synchronize()

        del cache
        gc.collect()
        torch.cuda.empty_cache()
        # A behavioural oracle: if the allocator had released the VA, the rebuilt
        # view would fault or lose the data. Free-memory deltas are not usable
        # here because other tests' cached blocks also return to the driver.
        # Granularity padding may map the whole reservation; it must not shrink.
        assert storage.committed_blocks >= 4
        rebuilt = KVCacheManager(
            num_layers=SMALL["num_layers"],
            num_kv_heads=SMALL["num_kv_heads"],
            head_dim=SMALL["head_dim"],
            num_blocks=storage.committed_blocks,
            block_size=SMALL["block_size"],
            max_loops=1,
            device=storage.device,
            dtype=storage.dtype,
            storage=storage,
        )
        assert rebuilt.key_cache.data_ptr() == address
        assert torch.equal(rebuilt.key_cache[0, 0, 0], probe)
    finally:
        storage.close()


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_engine_publishes_planned_capacity_after_bootstrap(backend):
    from tests.helpers import LocalLLM, tiny_ouro_config
    from vllm_rlt import CacheConfig, SamplingParams
    from vllm_rlt.models import OuroForCausalLM

    torch.manual_seed(123)
    base = OuroForCausalLM(tiny_ouro_config()).to(device="cuda", dtype=torch.bfloat16)
    prompts = [[2, 3, 4, 5], [6, 7]]
    params = [SamplingParams(max_tokens=4, ignore_eos=True, temperature=0.0) for _ in prompts]

    expected = LocalLLM(base, cache_config=CacheConfig(64, 2), attention_backend=backend).generate(
        prompts, params
    )

    engine = LocalLLM(
        base,
        cache_config=CacheConfig(block_size=2, vmm_bootstrap_blocks=2),
        attention_backend=backend,
    )
    stage = engine.engine
    try:
        assert stage.kv_storage is not None
        plan = stage.memory_plan
        assert plan["vmm"] is True
        assert plan["vmm_bootstrap_blocks"] == 2
        assert plan["vmm_max_blocks"] == stage.cache_manager.max_blocks
        # Bootstrap is mapped first, then the planned capacity is published.
        assert stage.cache_manager.published_blocks == plan["vmm_max_blocks"]
        assert stage.cache_manager.committed_blocks >= stage.cache_manager.published_blocks
        actual = engine.generate(prompts, params)
    finally:
        stage.close()
    assert [(o.token_ids, o.exit_depths) for o in actual] == [
        (o.token_ids, o.exit_depths) for o in expected
    ]
    assert stage.kv_storage is None, "close must release the reservation"


def test_connector_is_rejected_with_vmm_storage():
    from tests.helpers import tiny_ouro_model
    from vllm_rlt import CacheConfig
    from vllm_rlt.engine.stage_engine import StageEngine

    model = tiny_ouro_model("cuda", torch.bfloat16)
    with pytest.raises(ValueError, match="PD connector"):
        StageEngine(
            model,
            cache_config=CacheConfig(block_size=2, vmm_bootstrap_blocks=2),
            connector=object(),
        )
