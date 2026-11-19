"""Ascend attention consumes the common packed KV descriptor."""

import pytest
import torch

from vllm_rlt.attention.backend import AttentionBackend, BackendCapabilities
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.device import resolve_device
from vllm_rlt.kernels.ascend_attention import AscendAttention


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("npu:0", marks=pytest.mark.npu)])
@torch.inference_mode()
def test_packed_prefill_mixes_cached_chunks_and_single_queries(device_name):
    device = resolve_device(device_name)
    if device.type == "npu":
        torch.npu.set_device(device)
    attention = AttentionBackend(
        AscendAttention(),
        BackendCapabilities(packed_prefill=True),
        {"backend": "ascend_eager", "selected_backend": "ascend_eager"},
    )
    cache = KVCacheManager(
        1, 2, 64, 16, 16, 2, device=device, dtype=torch.bfloat16, attention=attention
    )
    for request in ("a", "b", "c"):
        assert cache.allocate(request, 3)

    def values(items):
        return (
            torch.tensor(items, device=device, dtype=cache.dtype)
            .view(-1, 1, 1)
            .expand(-1, cache.num_kv_heads, cache.head_dim)
            .contiguous()
        )

    history = values([1, 10, 20, 100])
    cache.write(
        0, ["a", "b", "b", "c"], [0, 1, 1, 0], [0, 0, 1, 0], torch.zeros_like(history), history
    )
    ids, depths, positions = ["a", "a", "b", "c"], [0, 0, 1, 0], [1, 2, 2, 1]
    packed = cache._prepare_batch(ids, depths, positions, packed_prefill=True)
    assert packed.cu_seqlens_q.tolist() == [0, 2, 3, 4]
    assert packed.context_lengths.tolist() == [3, 3, 2]
    assert packed.block_tables.shape[0] == 3

    new = values([3, 8, 30, 120])
    query = torch.zeros_like(new)
    cache._write_prepared(0, packed, torch.zeros_like(new), new)
    actual = cache._attend_prepared(0, packed, query)
    expected = values([2, 4, 20, 110])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    per_query = cache._prepare_batch(ids, depths, positions, for_write=False)
    reference = cache._attend_prepared(0, per_query, query)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
