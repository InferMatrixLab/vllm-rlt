import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.kernels.kv_quantization.triton_decode import decode_pages_into

# (format, group_size, block_size, heads, head_dim): odd and sub-group head_dim included.
CASES = [
    ("int4", 64, 64, 16, 128),
    ("int4", 2, 4, 2, 8),
    ("int4", 64, 64, 2, 8),
    ("int4", 16, 32, 3, 65),
    ("nvfp4", 16, 16, 2, 8),
    ("nvfp4", 16, 64, 16, 128),
    ("nvfp4", 16, 32, 3, 65),
]


@pytest.mark.gpu
@pytest.mark.parametrize("format,group,block,heads,dim", CASES)
@pytest.mark.parametrize("kind", ["key", "value"])
def test_triton_decode_matches_torch_reference_bit_for_bit(format, group, block, heads, dim, kind):
    torch.manual_seed(0)
    layers, num_blocks = 3, 40
    cache = KVCacheManager(
        num_layers=layers,
        num_kv_heads=heads,
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
    blocks = torch.randperm(num_blocks, device="cuda")[:17]
    pages = (torch.randn(len(blocks), layers, block, heads, dim, device="cuda") * 3).to(
        torch.bfloat16
    )
    # A constant page exercises zero spans and signed zeros.
    pages[0] = 0
    cache._encode_pages(kind, pages, blocks.tolist())
    for layer in range(layers):
        reference = cache._decode_pages(kind, layer, blocks)
        # NaN-filled output and shuffled rows catch unwritten or misplaced pages.
        out = torch.full(
            (len(blocks) + 3, block, heads, dim), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        rows = torch.randperm(len(blocks) + 3, device="cuda")[: len(blocks)]
        decode_pages_into(
            out,
            rows,
            blocks,
            cache._packed_pools[kind],
            layer,
            format=format,
            kind=kind,
            group_size=group,
        )
        assert torch.equal(out[rows].view(torch.int16), reference.view(torch.int16))
