"""Fused RMSNorm (Triton) for CUDA rows.

The reference ``RMSNorm.forward`` is eight small kernels per call (cast, square, mean,
add, rsqrt, mul, cast, weight mul). ``fused_rmsnorm`` does the same arithmetic in one
launch: FP32 reduction, ``value * rsqrt(mean + eps)`` in FP32, round to the activation
dtype, then the weight multiply. Its FP32 reduction order differs (a block tree sum
instead of torch's reduction) and the compiler may merge roundings, so outputs can differ
in the last ulp; it is opt-in (VLLM_RLT_FUSED_NORM=approx).

``fused_rmsnorm_exact`` (the default) keeps torch's own ``mean`` for the statistic, so the
reduction order is the reference's, and fuses the elementwise parts around it: cast+square
before, ``rsqrt(mean + eps)``/scale/cast/weight after. Three launches instead of eight (four
instead of nine with a residual, which torch adds). Every remaining operation is one
correctly rounded FP32 operation followed by the same cast, and the casts are forced
rounding points (``kernels/rounding.py``) that the compiler cannot fold, so the result is
bitwise identical to the reference.
"""

import torch

try:
    import triton
    import triton.language as tl

    from vllm_rlt.kernels.rounding import mul_round, round_to
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        w_ptr,
        res_ptr,
        out_ptr,
        hidden,
        eps,
        x_stride,
        res_stride,
        out_stride,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < hidden
        x = tl.load(x_ptr + row * x_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / hidden
        inv = 1.0 / tl.sqrt(variance + eps)
        normalized = (x * inv).to(out_ptr.dtype.element_ty)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
        out = (w.to(tl.float32) * normalized.to(tl.float32)).to(out_ptr.dtype.element_ty)
        if HAS_RESIDUAL:
            # residual + norm(x). The compiler keeps the weight product in fp32 across
            # the cast (bf16/fp16) or contracts it into an FMA (fp32), so this is a
            # single rounding of residual + w * normalized, not torch's two roundings:
            # about 18% of elements differ from `residual + reference` by one ulp
            # (measured on GH200). Exact mode adds the residual with torch instead.
            res = tl.load(res_ptr + row * res_stride + offsets, mask=mask, other=0.0)
            out = (res.to(tl.float32) + out.to(tl.float32)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * out_stride + offsets, out, mask=mask)

    @triton.jit
    def _square_kernel(x_ptr, out_ptr, hidden, x_stride, out_stride, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < hidden
        x = tl.load(x_ptr + row * x_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * out_stride + offsets, x * x, mask=mask)

    @triton.jit
    def _rmsnorm_tail_kernel(
        x_ptr,
        inv_ptr,
        w_ptr,
        res_ptr,
        out_ptr,
        hidden,
        eps,
        x_stride,
        res_stride,
        out_stride,
        HAS_RESIDUAL: tl.constexpr,
        FUSE_INVERSE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Same rounding points as the reference: fp32 scale -> dtype, fp32 weight
        # product -> dtype. Products of two half-precision values are exact in fp32,
        # so each step rounds once. With FUSE_INVERSE the row statistic arrives as the
        # mean of squares and rsqrt(mean + eps) is computed here: the add is one IEEE
        # rounding and rsqrt lowers to the same approximate instruction torch uses.
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < hidden
        dtype = out_ptr.dtype.element_ty
        x = tl.load(x_ptr + row * x_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        inv = tl.load(inv_ptr + row)
        if FUSE_INVERSE:
            inv = tl.math.rsqrt(inv + eps)
        # Forced rounding points (inline PTX): the compiler may otherwise keep the
        # fp32 products across the casts, as it does in the approx residual epilogue.
        normalized = round_to(x * inv, dtype)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
        out = mul_round(w.to(tl.float32), normalized, dtype).to(dtype)
        if HAS_RESIDUAL:
            res = tl.load(res_ptr + row * res_stride + offsets, mask=mask, other=0.0)
            out = (res.to(tl.float32) + out.to(tl.float32)).to(dtype)
        tl.store(out_ptr + row * out_stride + offsets, out, mask=mask)


def fused_rmsnorm_supported(hidden: torch.Tensor) -> bool:
    return (
        triton is not None
        and hidden.is_cuda
        and hidden.ndim == 2
        and hidden.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and hidden.stride(1) == 1
        and hidden.shape[1] <= 16384
    )


def fused_rmsnorm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> torch.Tensor:
    """``weight * rmsnorm(hidden)`` or ``residual + weight * rmsnorm(hidden)`` in one launch."""
    rows, width = hidden.shape
    out = torch.empty_like(hidden)
    if rows == 0:
        return out
    if residual is not None and (residual.shape != hidden.shape or residual.stride(1) != 1):
        raise ValueError("residual must match hidden's shape with unit inner stride")
    block = triton.next_power_of_2(width)
    warps = 4 if block <= 2048 else (8 if block <= 8192 else 16)
    _rmsnorm_kernel[(rows,)](
        hidden,
        weight,
        residual if residual is not None else hidden,
        out,
        width,
        eps,
        hidden.stride(0),
        residual.stride(0) if residual is not None else 0,
        out.stride(0),
        HAS_RESIDUAL=residual is not None,
        BLOCK=block,
        num_warps=warps,
    )
    return out


# rsqrt(mean + eps) inside the tail kernel saves two launches per norm. It is
# bitwise with torch when Triton's rsqrt lowers to the same instruction as
# torch's; the GPU test checks both settings.
FUSE_INVERSE = True


def fused_rmsnorm_exact(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    *,
    fuse_inverse: bool | None = None,
) -> torch.Tensor:
    """Bitwise ``RMSNorm.reference`` (plus optional residual) in 3-5 (4-6) launches, not 8 (9).

    torch computes the row mean of squares exactly as the reference does; Triton
    fuses the elementwise work around it (and, by default, ``rsqrt(mean + eps)``).
    """
    if fuse_inverse is None:
        fuse_inverse = FUSE_INVERSE
    rows, width = hidden.shape
    out = torch.empty_like(hidden)
    if rows == 0:
        return out
    if residual is not None and (residual.shape != hidden.shape or residual.stride(1) != 1):
        raise ValueError("residual must match hidden's shape with unit inner stride")
    block = triton.next_power_of_2(width)
    warps = 4 if block <= 2048 else (8 if block <= 8192 else 16)
    squares = torch.empty((rows, width), dtype=torch.float32, device=hidden.device)
    _square_kernel[(rows,)](
        hidden, squares, width, hidden.stride(0), squares.stride(0), BLOCK=block, num_warps=warps
    )
    statistic = squares.mean(-1)
    if not fuse_inverse:
        statistic = torch.rsqrt(statistic + eps)
    _rmsnorm_tail_kernel[(rows,)](
        hidden,
        statistic,
        weight,
        hidden,
        out,
        width,
        eps,
        hidden.stride(0),
        0,
        out.stride(0),
        HAS_RESIDUAL=False,
        FUSE_INVERSE=fuse_inverse,
        BLOCK=block,
        num_warps=warps,
    )
    # An in-kernel residual epilogue rounds residual + w * normalized once (the
    # compiler keeps the product unrounded across the cast, or forms an FMA in
    # fp32), which differs from torch's `residual + out` in ~18% of elements by
    # one ulp. torch's own add keeps this mode exact for one more launch (6 vs 9).
    return out if residual is None else residual + out
