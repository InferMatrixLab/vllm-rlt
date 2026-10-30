"""Rounding points the compiler cannot elide (Triton device helpers).

Triton/LLVM keeps ``(a * b).to(bf16).to(f32)`` in fp32 and contracts fp32 ``a * b + c``
into an FMA, so a kernel that mirrors a PyTorch op sequence with explicit casts is not
automatically bitwise with it (measured on GH200 for RMSNorm's residual epilogue and for
RoPE). These helpers go through inline PTX, which is opaque to those passes: ``round_to``
rounds an fp32 value to ``dtype`` and widens it back, ``mul_rn`` is a plain IEEE fp32
multiply that cannot be fused into a following add.
"""

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


if triton is not None:

    @triton.jit
    def round_to(x, dtype: tl.constexpr):
        """fp32 -> ``dtype`` -> fp32 with the rounding guaranteed to happen."""
        if dtype == tl.bfloat16:
            y = tl.inline_asm_elementwise(
                "cvt.rn.bf16.f32 $0, $1;", "=h,r", [x], dtype=tl.bfloat16, is_pure=True, pack=1
            )
            return y.to(tl.float32)
        elif dtype == tl.float16:
            y = tl.inline_asm_elementwise(
                "cvt.rn.f16.f32 $0, $1;", "=h,r", [x], dtype=tl.float16, is_pure=True, pack=1
            )
            return y.to(tl.float32)
        else:
            return x

    @triton.jit
    def mul_rn(a, b):
        """IEEE fp32 multiply that the compiler cannot contract into an FMA."""
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def mul_round(a, b, dtype: tl.constexpr):
        """``(a * b)`` rounded to ``dtype`` exactly once, as PyTorch rounds a ``dtype`` product.

        For bf16/fp16 operands the fp32 product is exact, so rounding it to ``dtype`` is
        the one rounding PyTorch applies; for fp32 the multiply itself is the rounding.
        """
        if dtype == tl.float32:
            return mul_rn(a, b)
        else:
            return round_to(a * b, dtype)
