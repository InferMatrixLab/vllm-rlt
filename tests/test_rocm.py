"""ROCm platform behavior that does not require AMD hardware."""

import pytest
import torch

from vllm_rlt.attention import create_backend
from vllm_rlt.platform import is_cuda, is_rocm, supports_flash_attention
from vllm_rlt.worker.tensors import use_uva_on


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(torch.version, "cuda", None)


def test_platform_flags(rocm):
    assert is_rocm()
    assert not is_cuda()
    assert not supports_flash_attention()


def test_auto_prefers_triton_on_rocm(rocm):
    selected = create_backend("auto", torch.device("cuda"), torch.bfloat16, 128, 16)
    assert selected.info["selected_backend"] == "triton"
    assert "FlashAttention is NVIDIA-only" in selected.info["selection_reason"]


def test_explicit_flash_rejected_on_rocm(rocm):
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        create_backend("flash_attn", torch.device("cuda"), torch.bfloat16, 128, 16)


def test_uva_override_and_cpu_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_RLT_UVA", "1")
    assert use_uva_on(torch.device("cuda", 0))
    monkeypatch.setenv("VLLM_RLT_UVA", "0")
    assert not use_uva_on(torch.device("cuda", 0))
    assert not use_uva_on(torch.device("cpu"))
