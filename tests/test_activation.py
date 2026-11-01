# SPDX-License-Identifier: Apache-2.0
"""Numerics, tails, layouts and stream contracts for SiLU-and-Mul."""

import math
import subprocess
import sys
import textwrap

import pytest
import torch

from vllm_rlt.kernels.activation import silu_and_mul, silu_and_mul_reference


def test_missing_native_extension_allows_optional_selection_but_not_explicit_load(monkeypatch):
    from vllm_rlt.kernels import _extension

    def missing(name):
        raise ModuleNotFoundError(name=name)

    _extension.try_load_extension.cache_clear()
    monkeypatch.setattr(_extension, "import_module", missing)
    try:
        assert not _extension.try_load_extension()
        with pytest.raises(RuntimeError, match="Native CUDA kernels are not built"):
            _extension.load_extension()
    finally:
        _extension.try_load_extension.cache_clear()


def test_broken_native_extension_dependency_is_not_hidden(monkeypatch):
    from vllm_rlt.kernels import _extension

    def broken(name):
        raise ModuleNotFoundError(name="missing_internal_dependency")

    _extension.try_load_extension.cache_clear()
    monkeypatch.setattr(_extension, "import_module", broken)
    try:
        with pytest.raises(ModuleNotFoundError) as exc:
            _extension.try_load_extension()
        assert exc.value.name == "missing_internal_dependency"
    finally:
        _extension.try_load_extension.cache_clear()


def test_silu_and_mul_known_values():
    gate = torch.tensor([0.0, 1.0, -1.0])
    up = torch.tensor([2.0, 3.0, 4.0])
    expected = torch.tensor([0.0, 2.1931758, -1.0757657])
    torch.testing.assert_close(silu_and_mul_reference(gate, up), expected)


@pytest.mark.parametrize("shape", [(7,), (2, 7), (2, 3, 9), (0, 7), (2, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_silu_and_mul_reference_contract(shape, dtype):
    generator = torch.Generator().manual_seed(42)
    gate = torch.randn(shape, dtype=dtype, generator=generator)
    up = torch.randn(shape, dtype=dtype, generator=generator)
    gate_before, up_before = gate.clone(), up.clone()

    out = silu_and_mul_reference(gate, up)
    # Preserve the intermediate SiLU rounding used by the existing MLPs.
    activated = (gate.double() * torch.sigmoid(gate.double())).to(dtype)
    expected = (activated.double() * up.double()).to(dtype)
    torch.testing.assert_close(out, expected)
    assert out.shape == gate.shape
    assert out.dtype == gate.dtype
    assert out.device == gate.device
    torch.testing.assert_close(gate, gate_before, rtol=0, atol=0)
    torch.testing.assert_close(up, up_before, rtol=0, atol=0)
    if out.numel():
        assert out.data_ptr() not in (gate.data_ptr(), up.data_ptr())


def test_silu_and_mul_reference_accepts_split_projection_views():
    packed = torch.arange(20, dtype=torch.float32).reshape(2, 10) / 10
    gate, up = packed.chunk(2, dim=-1)
    assert not gate.is_contiguous()
    out = silu_and_mul_reference(gate, up)
    expected = silu_and_mul_reference(gate.contiguous(), up.contiguous())
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("operator", [silu_and_mul_reference, silu_and_mul])
def test_silu_and_mul_rejects_broadcasting(operator):
    with pytest.raises(ValueError, match="same shape"):
        operator(torch.ones(2, 7), torch.ones(1, 7))


@pytest.mark.parametrize("operator", [silu_and_mul_reference, silu_and_mul])
def test_silu_and_mul_rejects_mixed_dtypes(operator):
    with pytest.raises(TypeError, match="same dtype"):
        operator(torch.ones(7), torch.ones(7, dtype=torch.float16))


@pytest.mark.parametrize("operator", [silu_and_mul_reference, silu_and_mul])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float64])
def test_silu_and_mul_rejects_unsupported_dtypes(operator, dtype):
    with pytest.raises(TypeError, match="supports float32"):
        operator(torch.ones(7, dtype=dtype), torch.ones(7, dtype=dtype))


def test_silu_and_mul_dispatches_cpu_to_torch_reference():
    gate, up = torch.randn(3, 7), torch.randn(3, 7)
    torch.testing.assert_close(silu_and_mul(gate, up), silu_and_mul_reference(gate, up))


@pytest.mark.parametrize("missing_extension", [False, True])
def test_kernel_initialization_and_cpu_fallback_do_not_initialize_cuda(missing_extension):
    script = textwrap.dedent(f"""
        import importlib
        import torch
        original_import = importlib.import_module
        attempts = []
        def import_module(name, *args, **kwargs):
            if name == 'vllm_rlt._C':
                attempts.append(name)
                if {missing_extension}:
                    raise ModuleNotFoundError(name=name)
            return original_import(name, *args, **kwargs)
        importlib.import_module = import_module
        from vllm_rlt.kernels import silu_and_mul
        for _ in range(3):
            gate, up = torch.randn(3, 7), torch.randn(3, 7)
            torch.testing.assert_close(silu_and_mul(gate, up), torch.nn.functional.silu(gate)*up)
        assert len(attempts) == 1, attempts
        assert not torch.cuda.is_initialized()
    """)
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, timeout=60)


@pytest.fixture(autouse=True)
def require_native_extension_for_gpu_tests(request):
    if request.node.get_closest_marker("gpu"):
        from vllm_rlt.kernels._extension import load_extension

        load_extension()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(), (0, 7), (2, 0), (1,), (7,), (8,), (9,), (3, 7), (2, 3, 9), (3, 5633), (128, 5632)]
)
@pytest.mark.parametrize("offsets", [(0, 0), (1, 0), (0, 1)])
def test_silu_and_mul_cuda_vectors_tails_and_storage_offsets(dtype, shape, offsets):
    count = math.prod(shape)
    gate = torch.randn(count + offsets[0], device="cuda", dtype=dtype)[offsets[0] :].view(shape)
    up = torch.randn(count + offsets[1], device="cuda", dtype=dtype)[offsets[1] :].view(shape)
    gate_before, up_before = gate.clone(), up.clone()
    out = silu_and_mul(gate, up)
    torch.testing.assert_close(out, silu_and_mul_reference(gate, up))
    assert out.shape == gate.shape and out.dtype == dtype and out.device == gate.device
    torch.testing.assert_close(gate, gate_before, rtol=0, atol=0)
    torch.testing.assert_close(up, up_before, rtol=0, atol=0)
    if count:
        assert out.data_ptr() not in (gate.data_ptr(), up.data_ptr())


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_silu_and_mul_cuda_special_values_and_underflow(dtype):
    info = torch.finfo(dtype)
    values = [
        float("nan"),
        float("inf"),
        -float("inf"),
        -100,
        -12,
        -1,
        -0.0,
        0.0,
        1,
        12,
        100,
        info.tiny,
        info.tiny / 128,
        info.max,
    ]
    gate, up = torch.meshgrid(
        torch.tensor(values, device="cuda", dtype=dtype),
        torch.tensor(values, device="cuda", dtype=dtype),
        indexing="ij",
    )
    gate, up = gate.contiguous(), up.contiguous()
    torch.testing.assert_close(
        silu_and_mul(gate, up), silu_and_mul_reference(gate, up), equal_nan=True
    )


@pytest.mark.gpu
def test_silu_and_mul_cuda_rejects_noncontiguous_input_without_copying():
    gate, up = torch.randn(2, 14, device="cuda").chunk(2, dim=-1)
    with pytest.raises(RuntimeError, match="contiguous inputs"):
        silu_and_mul(gate, up)


@pytest.mark.gpu
def test_silu_and_mul_cuda_rejects_autograd_but_allows_inference():
    gate = torch.randn(16, device="cuda", requires_grad=True)
    up = torch.randn_like(gate)
    with pytest.raises(RuntimeError, match="inference-only"):
        silu_and_mul(gate, up)
    with torch.no_grad():
        out = silu_and_mul(gate, up)
        torch.testing.assert_close(out, silu_and_mul_reference(gate, up))
        assert not out.requires_grad


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_silu_and_mul_cuda_current_stream_and_graph_replay(dtype):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gate = torch.randn(4097, device="cuda", dtype=dtype)
        up = torch.randn_like(gate)
        for _ in range(3):
            silu_and_mul(gate, up)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = silu_and_mul(gate, up)
    gate.fill_(3)
    up.fill_(2)
    graph.replay()
    torch.testing.assert_close(out, silu_and_mul_reference(gate, up))


@pytest.mark.gpu
@pytest.mark.parametrize("count", [(1 << 24) - 1, 1 << 24, (1 << 24) + 1])
def test_silu_and_mul_cuda_fp32_streaming_grid_boundary(count):
    # Four FP32 elements/vector: this boundary fills 16384 blocks of 256 threads.
    gate = torch.randn(count, device="cuda")
    up = torch.randn_like(gate)
    torch.testing.assert_close(silu_and_mul(gate, up), silu_and_mul_reference(gate, up))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset", [0, 1])
def test_silu_and_mul_cuda_low_precision_storage_patterns(dtype, offset):
    # Exercise all gate bit patterns with both vector/tail and scalar-only access.
    values = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(dtype)
    info = torch.finfo(dtype)
    subnormal = info.tiny / (1024 if dtype == torch.float16 else 128)
    factors = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            info.tiny,
            subnormal,
            info.max,
            -info.max,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        device="cuda",
        dtype=dtype,
    )
    gate = torch.empty(values.numel() * factors.numel() + 1 + offset, device="cuda", dtype=dtype)[
        offset:
    ]
    up = torch.empty_like(gate)
    gate[:-1].copy_(values.repeat(factors.numel()))
    up[:-1].copy_(factors.repeat_interleave(values.numel()))
    # The aligned path ends with one scalar multiplication yielding a subnormal.
    gate[-1], up[-1] = 1.0, subnormal
    actual = silu_and_mul(gate, up)
    expected = silu_and_mul_reference(gate, up)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    valid = ~torch.isnan(expected)
    assert torch.equal(actual[valid].view(torch.int16), expected[valid].view(torch.int16))
