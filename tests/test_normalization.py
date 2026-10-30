# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared RMSNorm layer."""

import pytest
import torch

from vllm_rlt.layers import RMSNorm
from vllm_rlt.models.nanbeige import NanbeigeRMSNorm
from vllm_rlt.models.ouro import OuroRMSNorm


def test_backward_compatible_aliases():
    assert OuroRMSNorm is RMSNorm
    assert NanbeigeRMSNorm is RMSNorm


def test_rmsnorm_weight_parameter_and_state_dict():
    hidden_size = 128
    norm = RMSNorm(hidden_size, eps=1e-5)

    assert isinstance(norm.weight, torch.nn.Parameter)
    assert norm.weight.shape == (hidden_size,)
    assert torch.all(norm.weight == 1.0)
    assert set(norm.state_dict().keys()) == {"weight"}

    # Loading a standard state_dict
    state = {"weight": torch.randn(hidden_size)}
    norm.load_state_dict(state)
    assert torch.allclose(norm.weight, state["weight"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_forward_numerics_and_dtypes(dtype):
    torch.manual_seed(42)
    hidden_size = 64
    eps = 1e-5
    norm = RMSNorm(hidden_size, eps=eps).to(dtype)

    # Random weights and input
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden_size))

    x = torch.randn(2, 8, hidden_size, dtype=dtype)
    out = norm(x)

    assert out.shape == x.shape
    assert out.dtype == dtype

    # Verify exact reference computation matching forward()
    x_fp32 = x.float()
    variance = x_fp32.square().mean(-1, keepdim=True)
    normalized = (x_fp32 * torch.rsqrt(variance + eps)).to(dtype)
    ref = norm.weight * normalized

    assert torch.equal(out, ref)


def test_rmsnorm_explicit_epsilon():
    hidden_size = 16
    x = torch.ones(1, 4, hidden_size)

    norm_small_eps = RMSNorm(hidden_size, eps=1e-8)
    norm_large_eps = RMSNorm(hidden_size, eps=1.0)

    out_small = norm_small_eps(x)
    out_large = norm_large_eps(x)

    assert not torch.allclose(out_small, out_large)
    # Variance of ones is 1.0; 1.0 / sqrt(1.0 + 1.0) = 1.0 / sqrt(2) ≈ 0.7071
    expected_large = 1.0 / (2.0**0.5)
    assert torch.allclose(out_large, torch.full_like(x, expected_large), atol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [2048, 5280, 10560])
def test_fused_rmsnorm_matches_reference_within_rounding(dtype, width):
    from vllm_rlt.kernels.norm import fused_rmsnorm, fused_rmsnorm_supported

    torch.manual_seed(0)
    norm = RMSNorm(width).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(width, device="cuda") * 0.5 + 1.0)
    hidden = (torch.randn(37, width, device="cuda") * 3).to(dtype)
    assert fused_rmsnorm_supported(hidden)
    fused = fused_rmsnorm(hidden, norm.weight, norm.variance_epsilon)
    reference = norm.reference(hidden)
    assert fused.dtype == dtype and fused.shape == reference.shape
    # Same arithmetic and rounding points; only the FP32 reduction order differs. A
    # flipped normalized value can move the weighted output by up to two ulps
    # (two chained roundings); the vast majority of elements are identical.
    if dtype == torch.float32:
        assert torch.allclose(fused, reference, rtol=1e-5, atol=1e-6)
    else:
        diff = (fused.float() - reference.float()).abs()
        ulp = torch.finfo(dtype).eps * reference.float().abs().clamp_min(torch.finfo(dtype).tiny)
        assert (diff <= 2.01 * ulp).all()
        assert (diff == 0).float().mean().item() > 0.99


@pytest.mark.gpu
def test_fused_rmsnorm_residual_epilogue_rounds_once():
    """The approx kernel's residual epilogue computes round(residual + w * normalized).

    torch's separate add rounds the weight product first. The two oracles differ
    in ~18% of elements, by more than one ulp of the result under cancellation, so
    the check is membership: every element equals the single-rounded oracle (or,
    in rare fp16 cases, the double-rounded one).
    """
    from vllm_rlt.kernels.norm import fused_rmsnorm

    torch.manual_seed(1)
    norm = RMSNorm(2048).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        # Unit weights would make the product exact and hide the epilogue's rounding.
        norm.weight.copy_(torch.randn(2048, device="cuda") * 0.5 + 1.0)
    hidden = (torch.randn(9, 2048, device="cuda") * 3).to(torch.bfloat16)
    residual = (torch.randn(9, 2048, device="cuda") * 3).to(torch.bfloat16)
    ones = torch.ones_like(norm.weight)
    normalized = fused_rmsnorm(hidden, ones, norm.variance_epsilon)  # round(x * inv), exact weight
    plain = fused_rmsnorm(hidden, norm.weight, norm.variance_epsilon)
    fused = fused_rmsnorm(hidden, norm.weight, norm.variance_epsilon, residual)
    product = norm.weight.float() * normalized.float()
    single = (residual.float() + product).to(torch.bfloat16)
    double = residual + plain
    assert torch.equal(plain, product.to(torch.bfloat16))
    assert ((fused == single) | (fused == double)).all()
    assert (fused == single).float().mean().item() > 0.99
    assert not torch.equal(single, double)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [2048, 5280, 10560])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("fuse_inverse", [False, True])
def test_fused_rmsnorm_exact_is_bitwise_identical_to_reference(
    dtype, width, with_residual, fuse_inverse
):
    from vllm_rlt.kernels.norm import fused_rmsnorm_exact

    torch.manual_seed(2)
    norm = RMSNorm(width).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(width, device="cuda") * 0.5 + 1.0)
    hidden = (torch.randn(37, width, device="cuda") * 3).to(dtype)
    residual = (torch.randn(37, width, device="cuda") * 3).to(dtype) if with_residual else None
    reference = norm.reference(hidden)
    if residual is not None:
        reference = residual + reference
    fused = fused_rmsnorm_exact(
        hidden, norm.weight, norm.variance_epsilon, residual, fuse_inverse=fuse_inverse
    )
    assert fused.dtype == dtype and fused.shape == reference.shape
    assert torch.equal(fused, reference)
    # The row statistic is torch's own, so the exact mode stays bitwise under any width.
    assert torch.equal(
        fused_rmsnorm_exact(
            hidden[:1], norm.weight, norm.variance_epsilon, fuse_inverse=fuse_inverse
        ),
        norm.reference(hidden[:1]),
    )


def test_fused_scope_graph_uses_reference_outside_capture(monkeypatch):
    """Default scope: fused kernels only during CUDA graph capture; eager uses the reference."""
    import vllm_rlt.kernels.dispatch as dispatch
    import vllm_rlt.layers.normalization as normalization

    calls = []
    monkeypatch.setattr(normalization, "FUSED_NORM", "exact")
    monkeypatch.setattr(
        normalization.RMSNorm, "reference", lambda self, h: calls.append("ref") or h
    )
    norm = RMSNorm(8)
    hidden = torch.ones(2, 8)
    monkeypatch.setattr(dispatch, "FUSED_SCOPE", "graph")
    assert dispatch.fused_allowed() is False  # no capture on CPU
    norm(hidden)
    assert calls == ["ref"]
    monkeypatch.setattr(dispatch, "FUSED_SCOPE", "always")
    assert dispatch.fused_allowed() is True
