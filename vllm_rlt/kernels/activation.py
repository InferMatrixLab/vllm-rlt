"""MLP activation operators and their unfused correctness reference.

``silu_and_mul`` dispatches to the optional native CUDA kernel or the Torch
reference. ``fused_silu_mul`` is the fused Triton SiLU(gate) * up used while
fusion is allowed (see ``vllm_rlt.kernels.dispatch``).

The reference ``F.silu(gate) * up`` is two kernels, each rounding to the activation dtype.
This kernel computes ``silu`` in FP32 as ``x / (1 + exp(-x))`` with libdevice ``exp`` and
IEEE division (the same formula PyTorch's CUDA kernel uses), rounds, multiplies by ``up``
and rounds again. The approximate kernel (VLLM_RLT_FUSED_ACT=1) lets the compiler elide the
intermediate rounding; the exact kernel (VLLM_RLT_FUSED_ACT=exact) forces it through inline
PTX so the only remaining difference could be the transcendental's last fp32 ulp (libdevice
``exp`` is the CUDA math library PyTorch links). The GPU test reports the identical fraction.
"""

import torch
import torch.nn.functional as F

from vllm_rlt.kernels._extension import try_load_extension


def _validate_inputs(gate: torch.Tensor, up: torch.Tensor) -> None:
    if gate.shape != up.shape:
        raise ValueError("silu_and_mul expects inputs with the same shape")
    if gate.device != up.device:
        raise ValueError("silu_and_mul expects inputs on the same device")
    if gate.dtype != up.dtype:
        raise TypeError("silu_and_mul expects inputs with the same dtype")
    if gate.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("silu_and_mul supports float32, float16 and bfloat16")


def silu_and_mul_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute ``F.silu(gate) * up`` without broadcasting or modifying either input.

    Inputs have equal shape, dtype and device. The result has the same shape,
    dtype and device and owns new storage. Noncontiguous inputs are supported
    by this reference; the native entry requires contiguous inputs.
    """
    _validate_inputs(gate, up)
    return F.silu(gate) * up


# Keep the library alive: Python registrations are removed when it is destroyed.
# Native CUDA takes precedence over the composite Torch implementation; CPU and
# installations without the optional extension use the same schema and fallback.
_library = torch.library.Library("vllm_rlt", "FRAGMENT")
if not try_load_extension():
    _library.define("silu_and_mul(Tensor gate, Tensor up) -> Tensor")
_library.impl("silu_and_mul", silu_and_mul_reference, "CompositeExplicitAutograd")

silu_and_mul = torch.ops.vllm_rlt.silu_and_mul

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
