"""Scatter graph KV rows while skipping inactive bucket padding entirely."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_TRITON_AVAILABLE: bool | None = None
_UNQUANTIZED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _triton_available() -> bool:
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        _TRITON_AVAILABLE = triton is not None
    return _TRITON_AVAILABLE


def can_use_graph_kv_write(cache) -> bool:
    """Return True when unquantized NVIDIA CUDA graph KV writes can use Triton scatter."""
    if not _triton_available() or torch.version.hip:
        return False
    if getattr(cache, "key_scale", None) is not None:
        return False
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if not isinstance(key_cache, torch.Tensor) or not isinstance(value_cache, torch.Tensor):
        return False
    if not (key_cache.is_cuda and value_cache.is_cuda):
        return False
    return key_cache.dtype == value_cache.dtype and key_cache.dtype in _UNQUANTIZED_DTYPES


if triton is not None:

    @triton.jit
    def _write_kv_contiguous(
        K,
        V,
        KeyCache,
        ValueCache,
        Blocks,
        Offsets,
        Active,
        WriteMask,
        k_row: tl.constexpr,
        v_row: tl.constexpr,
        cache_block: tl.constexpr,
        cache_token: tl.constexpr,
        WIDTH: tl.constexpr,
        TILE: tl.constexpr,
    ):
        row = tl.program_id(0)
        enabled = True
        if Active is not None:
            enabled = row < tl.load(Active)
        if WriteMask is not None:
            enabled = enabled & tl.load(WriteMask + row)
        if enabled:
            row64 = row.to(tl.int64)
            col = tl.program_id(1) * TILE + tl.arange(0, TILE)
            valid = col < WIDTH
            # Promote before multiplying: large cache offsets exceed int32.
            block = tl.load(Blocks + row64).to(tl.int64)
            offset = tl.load(Offsets + row64).to(tl.int64)
            target = block * cache_block + offset * cache_token + col
            k = tl.load(K + row64 * k_row + col, mask=valid)
            v = tl.load(V + row64 * v_row + col, mask=valid)
            tl.store(KeyCache + target, k, mask=valid)
            tl.store(ValueCache + target, v, mask=valid)

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
        k_cache_block: tl.constexpr,
        k_cache_token: tl.constexpr,
        k_cache_head: tl.constexpr,
        k_cache_dim: tl.constexpr,
        v_cache_block: tl.constexpr,
        v_cache_token: tl.constexpr,
        v_cache_head: tl.constexpr,
        v_cache_dim: tl.constexpr,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        TILE: tl.constexpr,
    ):
        row = tl.program_id(0)
        enabled = True
        if Active is not None:
            enabled = row < tl.load(Active)
        if WriteMask is not None:
            enabled = enabled & tl.load(WriteMask + row)
        if enabled:
            row64 = row.to(tl.int64)
            index = tl.program_id(1) * TILE + tl.arange(0, TILE)
            head, dim = index // DIM, index % DIM
            valid = index < HEADS * DIM
            # Promote before multiplying: large cache offsets exceed int32.
            block = tl.load(Blocks + row64).to(tl.int64)
            offset = tl.load(Offsets + row64).to(tl.int64)
            k_target = (
                block * k_cache_block
                + offset * k_cache_token
                + head * k_cache_head
                + dim * k_cache_dim
            )
            v_target = (
                block * v_cache_block
                + offset * v_cache_token
                + head * v_cache_head
                + dim * v_cache_dim
            )
            k = tl.load(K + row64 * k_row + head * k_head + dim * k_dim, mask=valid)
            v = tl.load(V + row64 * v_row + head * v_head + dim * v_dim, mask=valid)
            tl.store(KeyCache + k_target, k, mask=valid)
            tl.store(ValueCache + v_target, v, mask=valid)


def write_graph_kv(key_cache, value_cache, batch, k, v):
    if k.shape[0] == 0:
        return
    heads, dim = k.shape[1], k.shape[2]
    width = heads * dim
    if width == 0:
        return
    contiguous = (
        k.stride(1) == dim
        and k.stride(2) == 1
        and v.stride(1) == dim
        and v.stride(2) == 1
        and key_cache.stride(2) == dim
        and key_cache.stride(3) == 1
        and value_cache.stride() == key_cache.stride()
    )
    if contiguous:
        tile = min(triton.next_power_of_2(max(width, 64)), 1024)
        _write_kv_contiguous[(len(k), triton.cdiv(width, tile))](
            k,
            v,
            key_cache,
            value_cache,
            batch.write_blocks,
            batch.write_offsets,
            batch.active_rows,
            batch.write_mask,
            k.stride(0),
            v.stride(0),
            key_cache.stride(0),
            key_cache.stride(1),
            WIDTH=width,
            TILE=tile,
        )
        return
    _write_kv[(len(k), triton.cdiv(width, 256))](
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
        *value_cache.stride(),
        HEADS=heads,
        DIM=dim,
        TILE=256,
    )
