import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.kernels.kv_quantization.triton_attention_packed import packed_paged_attention
from vllm_rlt.kernels.triton_attention import paged_attention

# (format, group_size, block_size, q_heads, kv_heads, head_dim)
CASES = [
    ("int4", 64, 64, 16, 2, 128),
    ("int4", 2, 4, 4, 2, 8),
    ("int4", 16, 32, 6, 3, 65),
    ("nvfp4", 16, 16, 4, 2, 8),
    ("nvfp4", 16, 64, 16, 2, 128),
    ("nvfp4", 16, 32, 6, 3, 65),
]


@pytest.mark.gpu
@pytest.mark.parametrize("format,group,block,heads,kv_heads,dim", CASES)
def test_packed_attention_matches_decode_then_attend(format, group, block, heads, kv_heads, dim):
    torch.manual_seed(0)
    layers, num_blocks, slots = 2, 48, 12
    cache = KVCacheManager(
        num_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=dim,
        num_blocks=num_blocks,
        block_size=block,
        max_loops=1,
        dtype=torch.bfloat16,
        device="cuda",
        kv_quantization=format,
        kv_quantization_group_size=group,
        kv_quantization_window=block,
    )
    packed = torch.randperm(num_blocks, device="cuda")[:20]
    for kind in ("key", "value"):
        pages = (torch.randn(len(packed), layers, block, kv_heads, dim, device="cuda") * 2).to(
            torch.bfloat16
        )
        cache._encode_pages(kind, pages, packed.tolist())
    staged_slots = torch.randperm(cache.key_cache.shape[0], device="cuda")[:slots]
    cache.key_cache.normal_()
    cache.value_cache.normal_()
    # Remapped pages: packed first, then staged; sources follow the module contract.
    sources = torch.cat([-(packed + 1), staged_slots]).to(torch.int32)
    pages_total = len(sources)
    rows = 5
    pages_per_row = 7
    tables = torch.stack(
        [torch.randperm(pages_total, device="cuda")[:pages_per_row] for _ in range(rows)]
    ).to(torch.int32)
    # Lengths: empty, partial page, one token, full table, mid-page.
    lengths = torch.tensor(
        [0, block // 2 + 1, 1, pages_per_row * block, 3 * block + 1], device="cuda"
    ).to(torch.int32)
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    for layer in range(layers):
        keys = torch.empty(pages_total, block, kv_heads, dim, dtype=torch.bfloat16, device="cuda")
        values = torch.empty_like(keys)
        keys[: len(packed)] = cache._decode_pages("key", layer, packed)
        values[: len(packed)] = cache._decode_pages("value", layer, packed)
        keys[len(packed) :] = cache.key_cache[staged_slots, layer]
        values[len(packed) :] = cache.value_cache[staged_slots, layer]
        expected = paged_attention(q, keys, values, tables, lengths)
        actual = packed_paged_attention(
            q,
            cache.key_cache[:, layer],
            cache.value_cache[:, layer],
            cache._packed_pools["key"],
            cache._packed_pools["value"],
            layer,
            tables,
            sources,
            lengths,
            format=format,
            group_size=group,
        )
        assert torch.equal(actual[0], torch.zeros_like(actual[0]))
        if format == "int4":
            assert torch.equal(actual, expected)
        else:
            # One ulp from a different score reduction order (module docstring).
            torch.testing.assert_close(actual, expected, atol=0, rtol=2**-7)
