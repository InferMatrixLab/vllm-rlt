import pytest
import torch

from vllm_rlt.kernels.kv_quantization import nvfp4
from vllm_rlt.kernels.kv_quantization.nvfp4_batch import pack_batch


@pytest.mark.gpu
@pytest.mark.parametrize("block,heads,dim", [(16, 2, 8), (64, 16, 128), (32, 3, 65), (16, 1, 5)])
@pytest.mark.parametrize("kind", ["key", "value"])
def test_batched_nvfp4_pack_matches_per_page_bit_for_bit(block, heads, dim, kind):
    torch.manual_seed(0)
    x = (torch.randn(11, block, heads, dim, device="cuda") * 3).to(torch.bfloat16)
    # All-zero, tiny and huge pages exercise the per-entry global scale.
    x[0] = 0
    x[1] *= 1e-4
    x[2] *= 300
    payload, scale, global_scale = pack_batch(x, kind=kind)
    for i in range(len(x)):
        reference = nvfp4.pack(x[i], kind=kind)
        assert torch.equal(payload[i], reference.payload)
        assert torch.equal(scale[i].view(torch.uint8), reference.scale.view(torch.uint8))
        assert torch.equal(global_scale[i], reference.global_scale)
