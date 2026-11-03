# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared RMSNorm layer."""

import pytest
import torch

from vllm_rlt.kernels.normalization import rms_norm, rms_norm_reference
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


def test_opt_in_residual_preserves_cpu_autograd_and_mixed_dtype():
    norm = RMSNorm(5, use_native=True)
    x = torch.randn(2, 5, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(2, 5, dtype=torch.float32, requires_grad=True)
    actual = norm(x, residual=residual)
    expected = rms_norm_reference(x, norm.weight, norm.variance_epsilon, residual)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert x.grad is not None and residual.grad is not None and norm.weight.grad is not None


@pytest.fixture
def native_rms_norm():
    from vllm_rlt.kernels._extension import load_extension

    load_extension()
    assert torch._C._dispatch_has_kernel_for_dispatch_key("vllm_rlt::rms_norm", "CUDA")
    return rms_norm


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [8, 17, 128, 256, 257, 768, 2048, 2049, 4096, 8193])
@pytest.mark.parametrize("add_residual", [False, True])
@torch.inference_mode()
def test_rmsnorm_cuda_values_tail_and_staged_rounding(native_rms_norm, dtype, width, add_residual):
    torch.manual_seed(42)
    x = torch.randn(3, width, device="cuda", dtype=dtype)
    weight = (torch.randn(width, device="cuda") * 0.1 + 1).to(dtype)
    residual = torch.randn_like(x) if add_residual else None
    x_before, w_before = x.clone(), weight.clone()
    r_before = residual.clone() if residual is not None else None
    result = native_rms_norm(x, weight, 1e-6, residual)
    expected = rms_norm_reference(x, weight, 1e-6, residual)
    # CUDA's reduction order differs from torch.mean; keep the dtype rounding
    # boundaries and independently check the residual's input is untouched.
    tolerance = {
        torch.float32: (2e-6, 2e-6),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]
    torch.testing.assert_close(result, expected, rtol=tolerance[0], atol=tolerance[1])
    assert result.shape == x.shape and result.dtype == x.dtype and result.is_contiguous()
    assert result.data_ptr() not in (x.data_ptr(), weight.data_ptr())
    assert torch.equal(x, x_before) and torch.equal(weight, w_before)
    if residual is not None:
        assert result.data_ptr() != residual.data_ptr()
        assert torch.equal(residual, r_before)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 15, 16, 17, 128, 1024])
@pytest.mark.parametrize("add_residual", [False, True])
@torch.inference_mode()
def test_rmsnorm_cuda_torch_order_is_exact(native_rms_norm, dtype, rows, add_residual):
    # The row count changes torch.mean's reduction tree. Cover the boundaries
    # where the compatibility kernel changes its block width.
    from vllm_rlt.kernels.normalization import _torch_order_validated

    if not _torch_order_validated(torch.cuda.current_device()):
        pytest.skip("the Torch-order CUDA kernel is not enabled on this Torch/GPU combination")
    x = torch.randn(rows, 2048, dtype=dtype, device="cuda")
    weight = torch.randn(2048, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if add_residual else None
    actual = native_rms_norm(x, weight, 1e-6, residual, match_torch=True)
    expected = rms_norm_reference(x, weight, 1e-6, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_rmsnorm_cuda_torch_order_strides_and_fallback(native_rms_norm, dtype):
    # An aligned SP-style slice can use the exact kernel, while an offset that
    # cannot safely use its vector loads falls back to the Torch reference.
    rows, width = 4, 2048
    base = torch.randn(rows, width * 2 + 8, dtype=dtype, device="cuda")
    residual_base = torch.randn_like(base)
    weight = torch.randn(width + 1, dtype=dtype, device="cuda")
    for offset in (0, 1):
        x = base[:, offset : offset + width]
        residual = residual_base[:, offset : offset + width]
        w = weight[offset : offset + width]
        actual = native_rms_norm(x, w, 1e-6, residual, match_torch=True)
        expected = rms_norm_reference(x, w, 1e-6, residual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_rmsnorm_cuda_torch_order_nonfinite(native_rms_norm, dtype):
    x = torch.zeros(4, 2048, dtype=dtype, device="cuda")
    x[1, 0] = float("inf")
    x[2, 0] = float("-inf")
    x[3, 0] = float("nan")
    weight = torch.randn(2048, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    actual = native_rms_norm(x, weight, 1e-6, residual, match_torch=True)
    expected = rms_norm_reference(x, weight, 1e-6, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.gpu
@torch.inference_mode()
def test_rmsnorm_cuda_torch_order_graph_replay(native_rms_norm):
    dtype = torch.bfloat16
    x = torch.randn(17, 2048, dtype=dtype, device="cuda")
    weight = torch.randn(2048, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    native_rms_norm(x, weight, 1e-6, residual, match_torch=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = native_rms_norm(x, weight, 1e-6, residual, match_torch=True)
    x.fill_(0.25)
    residual.fill_(0.5)
    graph.replay()
    expected = rms_norm_reference(x, weight, 1e-6, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("add_residual", [False, True])
@torch.inference_mode()
def test_rmsnorm_cuda_strides_and_independent_offsets(native_rms_norm, dtype, rank, add_residual):
    torch.manual_seed(123)
    width = 33
    shape = (width,) if rank == 1 else (*((2,) * (rank - 1)), width)
    base_shape = (*shape[:-1], width + 3)
    x = torch.randn(base_shape, device="cuda", dtype=dtype)[..., 1 : width + 1]
    residual = torch.randn(base_shape, device="cuda", dtype=dtype)[..., 2 : width + 2]
    weight = torch.randn(width + 1, device="cuda", dtype=dtype)[1:]
    if not add_residual:
        residual = None
    actual = native_rms_norm(x, weight, 1e-6, residual)
    expected = rms_norm_reference(x, weight, 1e-6, residual)
    tol = 2e-6 if dtype == torch.float32 else 2e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    assert actual.is_contiguous() and actual.shape == x.shape


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_rmsnorm_cuda_contiguous_with_arbitrary_singleton_stride(native_rms_norm, dtype):
    # PyTorch ignores singleton strides in is_contiguous(); flattened rows must
    # still advance by width, not by the arbitrary penultimate stride.
    width = 33
    x = torch.randn(4, width, device="cuda", dtype=dtype).as_strided(
        (4, 1, width), (width, 7 * width, 1)
    )
    residual = torch.randn(4, width, device="cuda", dtype=dtype).as_strided(
        (4, 1, width), (width, 5 * width, 1)
    )
    assert x.is_contiguous() and residual.is_contiguous()
    weight = torch.randn(width, device="cuda", dtype=dtype)
    actual = native_rms_norm(x, weight, 1e-6, residual)
    expected = rms_norm_reference(x, weight, 1e-6, residual)
    tol = 2e-6 if dtype == torch.float32 else 2e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_rmsnorm_cuda_all_alignment_combinations(native_rms_norm, dtype):
    # Each row begins at a different offset; input, residual and weight offsets
    # vary independently across every possible 16-byte alignment.
    torch.manual_seed(7)
    elements = 16 // torch.empty((), dtype=dtype).element_size()
    width = 2 * elements + 1
    x_base = torch.randn(2, width + elements, device="cuda", dtype=dtype)
    r_base = torch.randn_like(x_base)
    w_base = torch.randn(width + elements, device="cuda", dtype=dtype)
    tol = 2e-6 if dtype == torch.float32 else 2e-3 if dtype == torch.float16 else 1e-2
    for x_offset in range(elements):
        x = x_base[:, x_offset : x_offset + width]
        for w_offset in range(elements):
            weight = w_base[w_offset : w_offset + width]
            for r_offset in range(elements):
                residual = r_base[:, r_offset : r_offset + width]
                actual = native_rms_norm(x, weight, 1e-6, residual)
                expected = rms_norm_reference(x, weight, 1e-6, residual)
                torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_rmsnorm_cuda_nonfinite_and_zero_rows(native_rms_norm, dtype):
    x = torch.zeros(4, 33, dtype=dtype, device="cuda")
    x[1, 0] = float("inf")
    x[2, 0] = float("-inf")
    x[3, 0] = float("nan")
    weight = torch.randn(33, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    actual = native_rms_norm(x, weight, 1e-6, residual)
    expected = rms_norm_reference(x, weight, 1e-6, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.gpu
@torch.inference_mode()
def test_rmsnorm_cuda_falls_back_for_other_layouts_and_mixed_dtypes(native_rms_norm):
    width = 33
    inputs = (
        torch.randn(2, width * 2, device="cuda")[..., ::2],
        torch.randn(2, 1, 2, 1, width, device="cuda"),
        torch.randn(2, width, device="cuda", dtype=torch.bfloat16),
    )
    for x in inputs:
        # The last case intentionally has FP32 weight and BF16 activations.
        weight = torch.randn(width, device="cuda", dtype=torch.float32)
        residual = torch.randn_like(x)
        actual = native_rms_norm(x, weight, 1e-6, residual)
        expected = rms_norm_reference(x, weight, 1e-6, residual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(0, 2048), (2, 0)])
@torch.inference_mode()
def test_rmsnorm_cuda_empty(native_rms_norm, dtype, shape):
    x = torch.empty(shape, device="cuda", dtype=dtype)
    weight = torch.empty(shape[-1], device="cuda", dtype=dtype)
    actual = native_rms_norm(x, weight, 1e-6)
    assert actual.shape == x.shape and actual.dtype == dtype and actual.numel() == 0


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("add_residual", [False, True])
@torch.inference_mode()
def test_rmsnorm_cuda_current_stream_and_graph_replay(native_rms_norm, dtype, add_residual):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        x = torch.randn(7, 2049, device="cuda", dtype=dtype)
        weight = torch.randn(2049, device="cuda", dtype=dtype)
        residual = torch.randn_like(x) if add_residual else None
        for _ in range(3):
            native_rms_norm(x, weight, 1e-6, residual)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
        actual = native_rms_norm(x, weight, 1e-6, residual)
    x.fill_(0.5)
    if residual is not None:
        residual.fill_(0.25)
    graph.replay()
    expected = rms_norm_reference(x, weight, 1e-6, residual)
    tol = 2e-6 if dtype == torch.float32 else 2e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
