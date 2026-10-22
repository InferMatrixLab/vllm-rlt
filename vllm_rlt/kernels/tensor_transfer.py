"""Typed tensor movement for mapped host inputs and device-resident segments."""

import triton
import triton.language as tl


@triton.jit
def tensor_transfer_kernel(
    Descriptors,
    Source,
    Target,
    Secondary,
    N,
    SOURCE_TYPE: tl.constexpr,
    TARGET_TYPE: tl.constexpr,
    SEGMENTED: tl.constexpr,
    SECONDARY: tl.constexpr,
    ROWS: tl.constexpr,
    TILE: tl.constexpr,
):
    if SEGMENTED:
        # Each descriptor is a non-overlapping, strided 2D copy. Addresses are
        # int64 throughout; never round-trip pointers or token IDs through floats.
        desc = Descriptors + tl.program_id(0) * 8
        src = tl.load(desc).to(tl.pointer_type(SOURCE_TYPE))
        dst = tl.load(desc + 1).to(tl.pointer_type(TARGET_TYPE))
        rows = tl.load(desc + 2)
        cols = tl.load(desc + 3)
        sr, sc = tl.load(desc + 4), tl.load(desc + 5)
        dr, dc = tl.load(desc + 6), tl.load(desc + 7)
        r = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        c = tl.program_id(2) * TILE + tl.arange(0, TILE)
        mask = (r[:, None] < rows) & (c[None, :] < cols)
        value = tl.load(src + r[:, None] * sr + c[None, :] * sc, mask, 0)
        tl.store(dst + r[:, None] * dr + c[None, :] * dc, value, mask)
    else:
        i = tl.program_id(0) * TILE + tl.arange(0, TILE)
        value = tl.load(Source + i, i < N, 0)
        tl.store(Target + i, value, i < N)
        if SECONDARY:
            # Packed attention needs both int64 token/position and int32 tables.
            tl.store(Secondary + i, value, i < N)
