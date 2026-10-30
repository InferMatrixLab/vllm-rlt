# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared RotaryEmbedding and RoPE helpers."""

import pytest
import torch

from vllm_rlt.layers import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from vllm_rlt.models.nanbeige import (
    NanbeigeRotaryEmbedding,
)
from vllm_rlt.models.nanbeige import (
    _rotate_half as nanbeige_rotate_half,
)
from vllm_rlt.models.ouro import (
    OuroRotaryEmbedding,
)
from vllm_rlt.models.ouro import (
    _rotate_half as ouro_rotate_half,
)


def test_backward_compatible_aliases():
    assert OuroRotaryEmbedding is RotaryEmbedding
    assert NanbeigeRotaryEmbedding is RotaryEmbedding
    assert ouro_rotate_half is rotate_half
    assert nanbeige_rotate_half is rotate_half


def test_rotate_half_mathematics():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rotated = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    torch.testing.assert_close(rotated, expected)


def test_apply_rotary_pos_emb():
    b, heads, dim = 2, 4, 8
    q = torch.randn(b, heads, dim)
    k = torch.randn(b, heads, dim)
    cos = torch.randn(b, 1, dim)
    sin = torch.randn(b, 1, dim)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    expected_q = q * cos + rotate_half(q) * sin
    expected_k = k * cos + rotate_half(k) * sin

    torch.testing.assert_close(q_rot, expected_q)
    torch.testing.assert_close(k_rot, expected_k)


def test_rotary_embedding_initialization_and_direct_arguments():
    head_dim = 64
    rope_theta = 500000.0
    rotary = RotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta)

    assert rotary.head_dim == head_dim
    assert rotary.rope_theta == rope_theta
    assert rotary.inv_freq.shape == (head_dim // 2,)
    assert rotary.inv_freq.dtype == torch.float32

    expected_freqs = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    torch.testing.assert_close(rotary.inv_freq, expected_freqs)


@pytest.mark.parametrize("target_dtype", [torch.float16, torch.bfloat16])
def test_rotary_embedding_dtype_conversion_preserves_float32(target_dtype):
    rotary = RotaryEmbedding(head_dim=32, rope_theta=10000.0)
    expected_freqs = rotary.inv_freq.clone()

    rotary.to(target_dtype)
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, expected_freqs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_embedding_forward_precision_and_shapes(dtype):
    head_dim = 16
    rotary = RotaryEmbedding(head_dim=head_dim, rope_theta=10000.0).to(dtype)
    positions = torch.tensor([0, 1, 5, 10])
    hidden = torch.randn(len(positions), 32, dtype=dtype)

    cos, sin = rotary(hidden, positions)
    assert cos.shape == (len(positions), 1, head_dim)
    assert sin.shape == (len(positions), 1, head_dim)
    assert cos.dtype == dtype
    assert sin.dtype == dtype


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads,dim", [(16, 128), (55, 96)])
def test_fused_rope_approx_rounds_the_sum_once(dtype, heads, dim):
    from vllm_rlt.kernels.rope import fused_rope_, fused_rope_supported
    from vllm_rlt.layers.rotary_embedding import apply_rotary_pos_emb_reference

    if (dim // 2) & (dim // 2 - 1):
        pytest.skip("fused RoPE needs a power-of-two half dimension")
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=dim).to("cuda")
    rows = 37
    hidden = torch.zeros(rows, 8, device="cuda", dtype=dtype)
    positions = torch.randint(0, 60000, (rows,), device="cuda")
    cos, sin = rope(hidden, positions)
    q = (torch.randn(rows, heads, dim, device="cuda") * 2).to(dtype)
    k = (torch.randn(rows, heads, dim, device="cuda") * 2).to(dtype)
    assert fused_rope_supported(q, cos)
    expected_q, expected_k = apply_rotary_pos_emb_reference(q, k, cos, sin)
    fused_q, fused_k = fused_rope_(q.clone(), cos, sin), fused_rope_(k.clone(), cos, sin)
    # The compiler is free to keep either product unrounded (FP32, or fused into an FMA
    # with the other term), so each element is one of four roundings of
    # q*cos + rotate_half(q)*sin: both products exact, either one rounded first, or
    # both rounded (the reference). GH200 showed ~9% of elements outside the first and
    # last, i.e. a mixed form. Under cancellation these differ by more than any fixed
    # number of result ulps, so the contract is membership in that set.
    from vllm_rlt.layers.rotary_embedding import rotate_half

    for value, fused, expected in ((q, fused_q, expected_q), (k, fused_k, expected_k)):
        p1 = value.float() * cos.float()
        p2 = rotate_half(value).float() * sin.float()
        r1, r2 = p1.to(dtype).float(), p2.to(dtype).float()
        candidates = [(p1 + p2), (r1 + p2), (p1 + r2)]
        member = fused == expected
        for c in candidates:
            member |= fused == c.to(dtype)
        assert member.float().mean().item() > 0.999
        assert (fused == expected).float().mean().item() > 0.7


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("heads,dim", [(16, 128), (4, 64)])
def test_fused_rope_exact_is_bitwise_identical_to_reference(dtype, heads, dim):
    from vllm_rlt.kernels.rope import fused_rope_, fused_rope_supported
    from vllm_rlt.layers.rotary_embedding import apply_rotary_pos_emb_reference

    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=dim).to("cuda")
    rows = 37
    hidden = torch.zeros(rows, 8, device="cuda", dtype=dtype)
    positions = torch.randint(0, 60000, (rows,), device="cuda")
    cos, sin = rope(hidden, positions)
    q = (torch.randn(rows, heads, dim, device="cuda") * 2).to(dtype)
    k = (torch.randn(rows, heads, dim, device="cuda") * 2).to(dtype)
    assert fused_rope_supported(q, cos)
    expected_q, expected_k = apply_rotary_pos_emb_reference(q, k, cos, sin)
    fused_q = fused_rope_(q.clone(), cos, sin, exact=True)
    fused_k = fused_rope_(k.clone(), cos, sin, exact=True)
    # Forced rounding points reproduce the reference's three roundings exactly.
    assert torch.equal(fused_q, expected_q)
    assert torch.equal(fused_k, expected_k)
