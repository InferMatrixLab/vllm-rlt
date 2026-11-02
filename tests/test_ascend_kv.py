"""Assigned-device regressions for native KV copies and descriptor lifetime."""

import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.device import resolve_device

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("rows", [1, 3])
@torch.inference_mode()
def test_native_scatter_preserves_inputs_other_layers_and_reused_ids(rows):
    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    cache = KVCacheManager(2, 2, 64, 32, 16, 4, device=device, dtype=torch.bfloat16)
    cache.key_cache.zero_()
    cache.value_cache.zero_()
    assert cache.allocate("a", 17)
    assert cache.allocate("b", 3)
    ids, depths, positions = ["a", "b", "a"][:rows], [0, 2, 3][:rows], [15, 0, 16][:rows]
    batch = cache._prepare_batch(ids, depths, positions)
    torch.manual_seed(99)
    k = torch.randn(rows, 2, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    saved_k, saved_v = k.clone(), v.clone()
    expected_k, expected_v = cache.key_cache.clone(), cache.value_cache.clone()
    for layer in range(2):
        expected_k[batch.write_blocks, layer, batch.write_offsets] = k
        expected_v[batch.write_blocks, layer, batch.write_offsets] = v
        cache._write_prepared(layer, batch, k, v)
    torch.testing.assert_close(cache.key_cache, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(cache.value_cache, expected_v, atol=0, rtol=0)
    torch.testing.assert_close(k, saved_k, atol=0, rtol=0)
    torch.testing.assert_close(v, saved_v, atol=0, rtol=0)
    cache.free("a")
    assert cache.allocate("a", 17)
    with pytest.raises((ValueError, RuntimeError)):
        cache._write_prepared(0, batch, k, v)
    new_batch = cache._prepare_batch(["a"], [1], [0])
    expected_k, expected_v = cache.key_cache.clone(), cache.value_cache.clone()
    expected_k[new_batch.write_blocks, 1, new_batch.write_offsets] = k[:1]
    expected_v[new_batch.write_blocks, 1, new_batch.write_offsets] = v[:1]
    cache._write_prepared(1, new_batch, k[:1], v[:1])
    torch.testing.assert_close(cache.key_cache, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(cache.value_cache, expected_v, atol=0, rtol=0)
    cache.free("a")
    cache.free("b")
    assert cache.num_used_blocks == 0
    torch.npu.synchronize()
