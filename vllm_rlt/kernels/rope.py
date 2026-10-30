"""Fused rotary embedding (Triton) for ``apply_rotary_pos_emb``.

The reference computes ``q * cos + rotate_half(q) * sin`` as separate half-precision ops,
rounding each product and the sum. The approximate kernel (VLLM_RLT_FUSED_ROPE=1) writes
the same casts, but the compiler keeps the products in FP32 before the add (measured on an
RTX 5090 and GH200), so outputs agree with the reference to within two ulps rather than
bitwise. The exact kernel (VLLM_RLT_FUSED_ROPE=exact) forces each product's rounding
through inline PTX and is bitwise with the reference. One launch per tensor instead of
about five either way.
"""

import torch

try:
    import triton
    import triton.language as tl

    from vllm_rlt.kernels.rounding import mul_round
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


if triton is not None:

    @triton.jit
    def _rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        heads,
        x_stride_row,
        x_stride_head,
        cs_stride_row,
        HALF: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, HALF)
        base = x_ptr + row * x_stride_row + head * x_stride_head
        dtype = x_ptr.dtype.element_ty
        first = tl.load(base + offsets).to(tl.float32)
        second = tl.load(base + HALF + offsets).to(tl.float32)
        cos_a = tl.load(cos_ptr + row * cs_stride_row + offsets).to(tl.float32)
        cos_b = tl.load(cos_ptr + row * cs_stride_row + HALF + offsets).to(tl.float32)
        sin_a = tl.load(sin_ptr + row * cs_stride_row + offsets).to(tl.float32)
        sin_b = tl.load(sin_ptr + row * cs_stride_row + HALF + offsets).to(tl.float32)
        # rotate_half(x) = cat(-second, first)
        prod_a = (first * cos_a).to(dtype).to(tl.float32)
        rot_a = ((-second) * sin_a).to(dtype).to(tl.float32)
        prod_b = (second * cos_b).to(dtype).to(tl.float32)
        rot_b = (first * sin_b).to(dtype).to(tl.float32)
        tl.store(base + offsets, (prod_a + rot_a).to(dtype))
        tl.store(base + HALF + offsets, (prod_b + rot_b).to(dtype))

    @triton.jit
    def _rope_exact_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        heads,
        x_stride_row,
        x_stride_head,
        cs_stride_row,
        HALF: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, HALF)
        base = x_ptr + row * x_stride_row + head * x_stride_head
        dtype = x_ptr.dtype.element_ty
        first = tl.load(base + offsets).to(tl.float32)
        second = tl.load(base + HALF + offsets).to(tl.float32)
        cos_a = tl.load(cos_ptr + row * cs_stride_row + offsets).to(tl.float32)
        cos_b = tl.load(cos_ptr + row * cs_stride_row + HALF + offsets).to(tl.float32)
        sin_a = tl.load(sin_ptr + row * cs_stride_row + offsets).to(tl.float32)
        sin_b = tl.load(sin_ptr + row * cs_stride_row + HALF + offsets).to(tl.float32)
        # Each product rounded to dtype exactly as the reference's separate multiplies;
        # the fp32 sum of two rounded terms is then rounded once by the store cast.
        prod_a = mul_round(first, cos_a, dtype)
        rot_a = mul_round(-second, sin_a, dtype)
        prod_b = mul_round(second, cos_b, dtype)
        rot_b = mul_round(first, sin_b, dtype)
        tl.store(base + offsets, (prod_a + rot_a).to(dtype))
        tl.store(base + HALF + offsets, (prod_b + rot_b).to(dtype))


def fused_rope_supported(x: torch.Tensor, cos: torch.Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and x.ndim == 3
        and x.stride(2) == 1
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and cos.dtype == x.dtype
        and cos.ndim == 3
        and cos.shape[1] == 1
        and cos.shape[0] == x.shape[0]
        and cos.stride(2) == 1
        and x.shape[2] % 2 == 0
        and (x.shape[2] // 2) & (x.shape[2] // 2 - 1) == 0
    )


def fused_rope_(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, exact: bool = False
) -> torch.Tensor:
    """Apply rotary embedding to ``x`` [rows, heads, dim] in place; returns ``x``."""
    rows, heads, dim = x.shape
    if rows == 0:
        return x
    kernel = _rope_exact_kernel if exact else _rope_kernel
    kernel[(rows, heads)](
        x,
        cos,
        sin,
        heads,
        x.stride(0),
        x.stride(1),
        cos.stride(0),
        HALF=dim // 2,
        num_warps=1 if dim <= 128 else 2,
    )
    return x
