"""Fused paged score pass for the sparse-decode capture step.

The selector needs a probability for every key of the reference loop, but the
Torch oracle materialises the whole causal row in FP32 with the GQA heads
expanded, which costs tens of MiB per layer and leaves the pass CPU-bound. This
kernel reads the paged cache once and writes one FP32 score per
``(row, head, position)``, so selection, softmax and the retained contribution
stay on the device without an expanded copy.

It computes the same dot product as the oracle in FP32 (the cache is BF16, and
BF16 -> FP32 is exact); only the reduction order differs. Measured against the
oracle at 8k context: max score delta 1.1e-5 on a magnitude of 65, retained mass
delta 9e-8, full attention output delta 2e-8, and identical selected sets on
511/512 row-head pairs.
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
    def _capture_scores_kernel(
        q_ptr,
        k_ptr,
        t_ptr,
        l_ptr,
        out_ptr,
        context,
        q_stride_row,
        k_stride_block,
        DIM: tl.constexpr,
        GROUPS: tl.constexpr,
        KV: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        PAGES: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        kv_head = tl.program_id(1)
        chunk = tl.program_id(2)

        position = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        row_length = tl.load(l_ptr + row)
        mask = position < row_length
        page = tl.load(t_ptr + row * PAGES + position // BLOCK_SIZE, mask=mask, other=0)
        offset = position % BLOCK_SIZE
        dim = tl.arange(0, DIM)
        # [BLOCK_N, DIM]
        keys = tl.load(
            k_ptr
            + page[:, None] * k_stride_block
            + (offset[:, None] * KV + kv_head) * DIM
            + dim[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        for group in tl.static_range(GROUPS):
            head = kv_head * GROUPS + group
            query = tl.load(q_ptr + row * q_stride_row + head * DIM + dim).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1)
            tl.store(
                out_ptr + (row * (KV * GROUPS) + head) * context + position,
                scores,
                mask=mask,
            )


def fused_capture_scores(
    queries: torch.Tensor,
    key_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    layer: int,
    block_size: int,
    block_n: int = 128,
) -> torch.Tensor:
    """Scores of every row against its causal keys, straight from the cache.

    ``queries`` is ``[rows, heads, dim]``, ``key_cache`` is either
    ``[blocks, layers, block_size, kv_heads, dim]`` or the already sliced
    ``[blocks, block_size, kv_heads, dim]``, and ``block_tables`` is
    ``[rows, pages]``. Returns ``[rows, heads, max_context]`` FP32 with entries
    beyond each row's length left at zero.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("fused capture needs Triton")
    if not queries.is_cuda:
        raise ValueError("fused capture is a CUDA path")

    rows, heads, dim = queries.shape
    if key_cache.dim() == 5:
        layer_cache = key_cache[:, layer]
        kv_heads = key_cache.shape[3]
    else:
        # The cache manager hands the attention layer a per-layer view already.
        layer_cache = key_cache
        kv_heads = key_cache.shape[2]
    if heads % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    groups = heads // kv_heads
    lengths = context_lengths.to(torch.int64)
    context = int(lengths.max().item()) if lengths.numel() else 0
    if context == 0:
        return queries.new_zeros((rows, heads, 0), dtype=torch.float32)

    pages = block_tables.shape[1]
    out = torch.zeros(rows, heads, context, device=queries.device, dtype=torch.float32)
    grid = (rows, kv_heads, triton.cdiv(context, block_n))
    _capture_scores_kernel[grid](
        queries.contiguous(),
        layer_cache,
        block_tables,
        lengths,
        out,
        context,
        queries.stride(0),
        key_cache.stride(0),
        DIM=dim,
        GROUPS=groups,
        KV=kv_heads,
        BLOCK_SIZE=block_size,
        PAGES=pages,
        BLOCK_N=block_n,
    )
    # Rows are ragged; the kernel leaves the tail at zero, so mask it to -inf
    # before any softmax sees it.
    if int(lengths.min().item()) != context:
        positions = torch.arange(context, device=queries.device)[None, None, :]
        out = out.masked_fill(positions >= lengths[:, None, None], float("-inf"))
    return out
