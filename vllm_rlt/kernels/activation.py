"""Fused SiLU(gate) * up (Triton).

The reference ``F.silu(gate) * up`` is two kernels, each rounding to the activation dtype.
This kernel computes ``silu`` in FP32 as ``x / (1 + exp(-x))`` with libdevice ``exp`` and
IEEE division (the same formula PyTorch's CUDA kernel uses), rounds, multiplies by ``up``
and rounds again. The approximate kernel (VLLM_RLT_FUSED_ACT=1) lets the compiler elide the
intermediate rounding; the exact kernel (VLLM_RLT_FUSED_ACT=exact) forces it through inline
PTX so the only remaining difference could be the transcendental's last fp32 ulp (libdevice
``exp`` is the CUDA math library PyTorch links). The GPU test reports the identical fraction.
"""

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    from vllm_rlt.kernels.rounding import mul_round, round_to
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


if triton is not None:

    @triton.jit
    def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, numel, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        dtype = out_ptr.dtype.element_ty
        g = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        act = tl.math.div_rn(g, 1.0 + libdevice.exp(-g)).to(dtype).to(tl.float32)
        tl.store(out_ptr + offsets, (act * u).to(dtype), mask=mask)

    @triton.jit
    def _silu_mul_exact_kernel(gate_ptr, up_ptr, out_ptr, numel, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        dtype = out_ptr.dtype.element_ty
        g = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # silu rounded to dtype (PyTorch's F.silu output), then the dtype product.
        act = round_to(tl.math.div_rn(g, 1.0 + libdevice.exp(-g)), dtype)
        tl.store(out_ptr + offsets, mul_round(act, u, dtype).to(dtype), mask=mask)


def fused_silu_mul_supported(gate: torch.Tensor, up: torch.Tensor) -> bool:
    return (
        triton is not None
        and gate.is_cuda
        and gate.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and gate.dtype == up.dtype
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
    )


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor, *, exact: bool = False) -> torch.Tensor:
    out = torch.empty_like(gate)
    numel = gate.numel()
    if numel == 0:
        return out
    block = 1024
    kernel = _silu_mul_exact_kernel if exact else _silu_mul_kernel
    kernel[((numel + block - 1) // block,)](gate, up, out, numel, BLOCK=block, num_warps=4)
    return out
