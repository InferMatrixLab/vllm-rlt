"""Scatter graph KV rows while skipping inactive bucket padding entirely."""

import triton
import triton.language as tl


@triton.jit
def _write_kv(
    K,
    V,
    KeyCache,
    ValueCache,
    Blocks,
    Offsets,
    Active,
    WriteMask,
    k_row: tl.constexpr,
    k_head: tl.constexpr,
    k_dim: tl.constexpr,
    v_row: tl.constexpr,
    v_head: tl.constexpr,
    v_dim: tl.constexpr,
    cache_block: tl.constexpr,
    cache_token: tl.constexpr,
    cache_head: tl.constexpr,
    cache_dim: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    enabled = row < tl.load(Active)
    if WriteMask is not None:
        enabled = enabled & tl.load(WriteMask + row)
    if enabled:
        index = tl.program_id(1) * TILE + tl.arange(0, TILE)
        head, dim = index // DIM, index % DIM
        valid = index < HEADS * DIM
        # Promote before multiplying: large cache offsets exceed int32.
        block = tl.load(Blocks + row).to(tl.int64)
        offset = tl.load(Offsets + row).to(tl.int64)
        target = block * cache_block + offset * cache_token + head * cache_head + dim * cache_dim
        k = tl.load(K + row * k_row + head * k_head + dim * k_dim, valid, 0)
        v = tl.load(V + row * v_row + head * v_head + dim * v_dim, valid, 0)
        tl.store(KeyCache + target, k, valid)
        tl.store(ValueCache + target, v, valid)


def write_graph_kv(key_cache, value_cache, batch, k, v):
    _write_kv[(len(k), triton.cdiv(k.shape[1] * k.shape[2], 256))](
        k,
        v,
        key_cache,
        value_cache,
        batch.write_blocks,
        batch.write_offsets,
        batch.active_rows,
        batch.write_mask,
        *k.stride(),
        *v.stride(),
        *key_cache.stride(),
        HEADS=k.shape[1],
        DIM=k.shape[2],
        TILE=256,
    )
