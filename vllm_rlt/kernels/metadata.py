"""Expand compact query metadata into KV write addresses."""

import triton
import triton.language as tl


@triton.jit
def expand_kv_addresses(
    Positions,
    Sequences,
    Tables,
    Blocks,
    Offsets,
    N,
    WIDTH,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    pos = tl.load(Positions + i, i < N, 0)
    seq = tl.load(Sequences + i, i < N, 0)
    block = tl.load(Tables + seq * WIDTH + pos // PAGE, i < N, 0)
    tl.store(Blocks + i, block, i < N)
    tl.store(Offsets + i, pos % PAGE, i < N)
