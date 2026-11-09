"""Fused gather-and-accumulate for the sparse-decode consume step.

A consume row only touches its ``k`` retained keys, but the Torch path gathers
them into a dense ``[rows, heads, k, dim]`` tensor twice (K then V) and runs one
cuBLAS call per row. At 32k context with ``k = 3277`` that is 8.2 ms of K gather
plus 8.1 ms of V gather through three advanced indices on a strided
``[blocks, block_size, kv, dim]`` cache, which measures about 45 GB/s.

These two kernels walk the retained indices in place: the first writes one FP32
score per ``(row, head, retained)``, the second accumulates ``p * V`` into one
output row. Nothing dense is materialised, and the softmax in between stays in
Torch because ``k`` is small.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - exercised only without Triton
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _selected_scores_kernel(
        q_ptr,
        k_ptr,
        idx_ptr,
        t_ptr,
        out_ptr,
        retained,
        k_stride_block,
        DIM: tl.constexpr,
        KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        PAGES: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        block = tl.program_id(2)
        heads: tl.constexpr = KV * GROUPS
        kv_head = head // GROUPS

        slot = block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = slot < retained
        index = tl.load(idx_ptr + (row * heads + head) * retained + slot, mask=mask, other=0)
        # Promote the page ID before multiplying by the physical block stride.
        page = tl.load(t_ptr + row * PAGES + index // BLOCK_SIZE, mask=mask, other=0).to(tl.int64)
        offset = index % BLOCK_SIZE
        dim = tl.arange(0, DIM)
        keys = tl.load(
            k_ptr
            + page[:, None] * k_stride_block
            + (offset[:, None] * KV + kv_head) * DIM
            + dim[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        query = tl.load(q_ptr + (row * heads + head) * DIM + dim).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1)
        tl.store(out_ptr + (row * heads + head) * retained + slot, scores, mask=mask)

    @triton.jit
    def _selected_values_kernel(
        p_ptr,
        v_ptr,
        idx_ptr,
        t_ptr,
        out_ptr,
        retained,
        v_stride_block,
        DIM: tl.constexpr,
        KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        PAGES: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        heads: tl.constexpr = KV * GROUPS
        kv_head = head // GROUPS
        dim = tl.arange(0, DIM)
        accumulator = tl.zeros([DIM], dtype=tl.float32)
        for start in range(0, retained, BLOCK_K):
            slot = start + tl.arange(0, BLOCK_K)
            mask = slot < retained
            index = tl.load(idx_ptr + (row * heads + head) * retained + slot, mask=mask, other=0)
            weight = tl.load(
                p_ptr + (row * heads + head) * retained + slot, mask=mask, other=0.0
            ).to(tl.float32)
            page = tl.load(t_ptr + row * PAGES + index // BLOCK_SIZE, mask=mask, other=0).to(
                tl.int64
            )
            offset = index % BLOCK_SIZE
            values = tl.load(
                v_ptr
                + page[:, None] * v_stride_block
                + (offset[:, None] * KV + kv_head) * DIM
                + dim[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(weight[:, None] * values, axis=0)
        tl.store(out_ptr + (row * heads + head) * DIM + dim, accumulator)


def _cache_view(cache: torch.Tensor, layer: int) -> tuple[torch.Tensor, int]:
    """Return the per-layer cache view and its KV head count."""
    if cache.dim() == 5:
        return cache[:, layer], cache.shape[3]
    return cache, cache.shape[2]


def selected_scores(
    queries: torch.Tensor,
    key_cache: torch.Tensor,
    selected: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    layer: int,
    block_size: int,
    block_k: int = 128,
) -> torch.Tensor:
    """``q . K[S]`` for every retained key, without materialising K.

    ``queries`` is ``[rows, heads, dim]``, ``selected`` is ``[rows, heads, k]``
    of absolute positions and the result is ``[rows, heads, k]`` FP32.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("fused consume needs Triton")
    rows, heads, dim = queries.shape
    cache, kv_heads = _cache_view(key_cache, layer)
    if heads % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    groups = heads // kv_heads
    retained = selected.shape[-1]
    out = torch.empty(rows, heads, retained, device=queries.device, dtype=torch.float32)
    if retained == 0:
        return out
    grid = (rows, heads, triton.cdiv(retained, block_k))
    _selected_scores_kernel[grid](
        queries.contiguous(),
        cache,
        selected.contiguous().to(torch.int32),
        block_tables,
        out,
        retained,
        cache.stride(0),
        DIM=dim,
        KV=kv_heads,
        GROUPS=groups,
        BLOCK_SIZE=block_size,
        PAGES=block_tables.shape[1],
        BLOCK_K=block_k,
    )
    return out


def selected_values(
    probabilities: torch.Tensor,
    value_cache: torch.Tensor,
    selected: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    layer: int,
    block_size: int,
    block_k: int = 128,
) -> torch.Tensor:
    """``sum_j p[j] * V[S[j]]`` for every row and head.

    ``probabilities`` is ``[rows, heads, k]`` and the result is
    ``[rows, heads, dim]`` FP32.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("fused consume needs Triton")
    rows, heads, retained = probabilities.shape
    cache, kv_heads = _cache_view(value_cache, layer)
    if heads % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    groups = heads // kv_heads
    dim = cache.shape[-1]
    out = torch.empty(rows, heads, dim, device=probabilities.device, dtype=torch.float32)
    if retained == 0:
        return out
    grid = (rows, heads)
    _selected_values_kernel[grid](
        probabilities.contiguous(),
        cache,
        selected.contiguous().to(torch.int32),
        block_tables,
        out,
        retained,
        cache.stride(0),
        DIM=dim,
        KV=kv_heads,
        GROUPS=groups,
        BLOCK_SIZE=block_size,
        PAGES=block_tables.shape[1],
        BLOCK_K=block_k,
    )
    return out
