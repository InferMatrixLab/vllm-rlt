"""Capacity publishing for VMM-backed KV storage (RFC #192 stage 2).

These run on CPU against a stand-in pool: they cover the bookkeeping contract
(published <= committed <= max, growth only, no publishing before a commit).
The CUDA mapping itself is exercised in test_vmm_pool.py.
"""

import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager


class FakeStorage:
    """Stand-in for GPUKVStorage that keeps the K/V planes on the CPU."""

    def __init__(self, *, max_blocks, num_layers, num_kv_heads, head_dim, block_size):
        self.max_blocks = max_blocks
        self.committed_blocks = 0
        self.dtype = torch.float32
        element = torch.empty((), dtype=self.dtype).element_size()
        self.bytes_per_block = 2 * num_layers * block_size * num_kv_heads * head_dim * element
        self.shape = (max_blocks, num_layers, block_size, num_kv_heads, head_dim)
        self.keys_tensor = torch.zeros(self.shape, dtype=self.dtype)
        self.values_tensor = torch.zeros(self.shape, dtype=self.dtype)
        self.commits = []

    def commit(self, blocks):
        self.committed_blocks = max(self.committed_blocks, blocks)
        self.commits.append(blocks)
        return self.committed_blocks

    def keys(self, shape):
        assert tuple(shape) == self.shape
        return self.keys_tensor

    def values(self, shape):
        assert tuple(shape) == self.shape
        return self.values_tensor


def make_storage(max_blocks):
    return FakeStorage(
        max_blocks=max_blocks, num_layers=2, num_kv_heads=1, head_dim=4, block_size=2
    )


def make_cache(storage=None, num_blocks=4, **kwargs):
    parameters = dict(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        num_blocks=num_blocks,
        block_size=2,
        max_loops=1,
    )
    parameters.update(kwargs)
    return KVCacheManager(**parameters, storage=storage)


def test_storage_backed_cache_starts_at_bootstrap_capacity():
    storage = make_storage(10)
    storage.commit(4)
    cache = make_cache(storage, num_blocks=4)
    assert cache.published_blocks == 4
    assert cache.committed_blocks == 4
    assert cache.max_blocks == 10
    assert cache.num_blocks == 4
    assert cache.num_free_blocks == 4
    # The planes span the whole reservation, not the bootstrap prefix.
    assert cache.key_cache.shape == storage.shape
    assert cache.value_cache.shape == storage.shape


def test_publish_capacity_extends_the_free_pool():
    storage = make_storage(10)
    storage.commit(4)
    cache = make_cache(storage, num_blocks=4)
    assert cache.allocate("a", 8)  # 4 pages: exactly the published blocks
    assert cache.num_free_blocks == 0
    assert cache.allocate("b", 2) is False  # no free blocks; admission must not raise
    storage.commit(8)
    assert cache.publish_capacity(8) == 8
    assert cache.num_blocks == 8
    assert cache.published_blocks == 8
    assert cache.committed_blocks == 8
    assert cache.num_free_blocks == 4
    assert cache.allocate("b", 2)
    blocks = {page for table in cache.plane_block_tables("b") for page in table}
    assert blocks <= set(range(8))
    assert blocks & set(range(4, 8)), "published blocks are the ones handed out"


def test_publish_never_exceeds_committed_memory():
    storage = make_storage(10)
    storage.commit(4)
    cache = make_cache(storage, num_blocks=4)
    free_before, published_before = cache.num_free_blocks, cache.num_blocks
    with pytest.raises(ValueError, match="committed"):
        cache.publish_capacity(6)
    assert cache.num_free_blocks == free_before
    assert cache.num_blocks == published_before
    assert cache.published_blocks == published_before


def test_publish_rejects_growth_past_the_reservation():
    storage = make_storage(10)
    storage.commit(4)
    cache = make_cache(storage, num_blocks=4)
    storage.commit(10)
    with pytest.raises(ValueError, match="reservation holds"):
        cache.publish_capacity(11)
    assert cache.num_blocks == 4


def test_publish_is_monotonic():
    storage = make_storage(10)
    storage.commit(8)
    cache = make_cache(storage, num_blocks=4)
    assert cache.publish_capacity(8) == 8
    with pytest.raises(ValueError, match="only grow"):
        cache.publish_capacity(6)
    assert cache.num_blocks == 8
    assert cache.publish_capacity(8) == 8  # idempotent


def test_publish_requires_storage():
    cache = make_cache(None)
    with pytest.raises(RuntimeError, match="VMM-backed"):
        cache.publish_capacity(8)


def test_plain_cache_keeps_fixed_capacity():
    cache = make_cache(None, num_blocks=6)
    assert cache.storage is None
    assert cache.max_blocks == cache.committed_blocks == cache.published_blocks == 6
    assert cache.num_blocks == 6


def test_storage_must_be_committed_before_the_cache_is_built():
    storage = make_storage(10)
    storage.commit(2)
    with pytest.raises(ValueError, match="commit the pool"):
        make_cache(storage, num_blocks=4)


def test_storage_must_cover_the_requested_blocks():
    storage = make_storage(2)
    storage.commit(2)
    with pytest.raises(ValueError, match="reserves"):
        make_cache(storage, num_blocks=4)


def test_storage_layout_must_match_the_cache():
    storage = make_storage(10)
    storage.bytes_per_block += 1
    storage.commit(4)
    with pytest.raises(ValueError, match="bytes per block"):
        make_cache(storage, num_blocks=4)


def test_published_blocks_are_writable_after_growth():
    storage = make_storage(10)
    storage.commit(2)
    cache = make_cache(storage, num_blocks=2)
    assert cache.allocate("a", 4)  # exactly the two published blocks
    assert cache.allocate("b", 2) is False  # nothing free, and admission must not raise
    storage.commit(10)
    cache.publish_capacity(10)
    assert cache.allocate("b", 2)
    blocks = cache.get_block_table("b", 0)
    assert set(blocks) & set(range(2, 10)), "growth must hand out the newly published blocks"
    key = torch.full((1, 1, 4), 3.0)
    cache.write(0, ["b"], [0], [0], key, key + 1)
    written = cache.key_cache[blocks[0], 0, 0]
    assert torch.equal(written, key[0])
    assert torch.equal(cache.value_cache[blocks[0], 0, 0], key[0] + 1)


def test_watermark_tracks_published_capacity():
    storage = make_storage(20)
    storage.commit(4)
    cache = make_cache(storage, num_blocks=4, incremental_allocation=True, watermark_ratio=0.5)
    assert cache.watermark_blocks == 2
    storage.commit(20)
    cache.publish_capacity(20)
    assert cache.watermark_blocks == 10


def test_vmm_bootstrap_requires_automatic_planning():
    from vllm_rlt import CacheConfig

    with pytest.raises(ValueError, match="automatic KV planning"):
        CacheConfig(num_blocks=8, vmm_bootstrap_blocks=2)
    with pytest.raises(ValueError, match="automatic KV planning"):
        CacheConfig(kv_cache_memory_bytes=1024, vmm_bootstrap_blocks=2)
    with pytest.raises(ValueError, match="vmm_bootstrap_blocks"):
        CacheConfig(vmm_bootstrap_blocks=0)


def test_vmm_storage_requires_cuda():
    from tests.helpers import tiny_ouro_model
    from vllm_rlt import CacheConfig
    from vllm_rlt.engine.stage_engine import StageEngine

    model = tiny_ouro_model()
    with pytest.raises(ValueError, match="CUDA device"):
        StageEngine(model, cache_config=CacheConfig(block_size=2, vmm_bootstrap_blocks=2))


def test_runtime_args_expose_the_vmm_bootstrap_flag():
    import argparse

    from vllm_rlt.entrypoints.runtime_args import add_runtime_args

    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    assert parser.parse_args([]).kv_vmm_bootstrap_blocks is None
    parsed = parser.parse_args(["--kv-vmm-bootstrap-blocks", "64"])
    assert parsed.kv_vmm_bootstrap_blocks == 64
