"""Single-query paged attention with independent block tables for every row.

Rows may represent different requests, positions, or recurrence depths. The
manager selects each row's depth-specific block table before dispatching here.
"""

import math

import torch


def torch_paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
) -> torch.Tensor:
    """Reference attention over [physical block, token, KV head, dimension]."""
    output = torch.empty_like(q)
    block_size = key_cache.shape[1]
    head_groups = q.shape[1] // key_cache.shape[2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    for row, length in enumerate(context_lengths.tolist()):
        if length == 0:
            output[row].zero_()
            continue
        token_positions = torch.arange(length, device=q.device)
        blocks = block_tables[row, token_positions // block_size]
        offsets = token_positions % block_size
        keys = key_cache[blocks, offsets].repeat_interleave(head_groups, dim=1)
        values = value_cache[blocks, offsets].repeat_interleave(head_groups, dim=1)
        scores = torch.einsum("hd,thd->ht", q[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum("ht,thd->hd", probabilities, values.float()).to(q.dtype)
    return output


def torch_paged_attention_windowed(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Reference sliding-window attention over the same layout.

    Row ``row`` attends only its last ``window`` keys, including the query's own
    position, matching per-layer sliding-window semantics. History outside the
    window stays stored: this is masking, not reclamation.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    output = torch.empty_like(q)
    block_size = key_cache.shape[1]
    head_groups = q.shape[1] // key_cache.shape[2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    for row, length in enumerate(context_lengths.tolist()):
        if length == 0:
            output[row].zero_()
            continue
        token_positions = torch.arange(max(0, length - window), length, device=q.device)
        blocks = block_tables[row, token_positions // block_size]
        offsets = token_positions % block_size
        keys = key_cache[blocks, offsets].repeat_interleave(head_groups, dim=1)
        values = value_cache[blocks, offsets].repeat_interleave(head_groups, dim=1)
        scores = torch.einsum("hd,thd->ht", q[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum("ht,thd->hd", probabilities, values.float()).to(q.dtype)
    return output


class TorchPagedAttention:
    """Reference single-query attention, with and without sliding windows."""

    def __call__(self, q, key_cache, value_cache, block_tables, context_lengths):
        return torch_paged_attention(q, key_cache, value_cache, block_tables, context_lengths)

    def windowed(self, q, key_cache, value_cache, block_tables, context_lengths, window):
        return torch_paged_attention_windowed(
            q, key_cache, value_cache, block_tables, context_lengths, window
        )


def triton_paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    window: int = 0,
) -> torch.Tensor:
    """Dispatch explicitly to Triton, with no silent reference fallback."""
    if q.device.type != "cuda":
        raise ValueError("the Triton attention backend requires a CUDA or ROCm device")
    if q.shape[-1] > 256:
        raise ValueError("the Triton attention backend supports head_dim <= 256")
    # Keep Triton optional and avoid loading its runtime for CPU-only callers.
    from vllm_rlt.kernels.triton_attention import paged_attention

    return paged_attention(q, key_cache, value_cache, block_tables, context_lengths, window=window)


def triton_paged_prefill(
    q, key_cache, value_cache, block_tables, context_lengths, cu_seqlens_q, max_seqlen_q
):
    """Packed causal prefill with one block table per sequence (see triton_attention)."""
    from vllm_rlt.kernels.triton_attention import paged_prefill_attention

    return paged_prefill_attention(
        q, key_cache, value_cache, block_tables, context_lengths, cu_seqlens_q, max_seqlen_q
    )


class TritonPagedAttention:
    """Triton single-query decode attention plus packed causal prefill."""

    def __call__(self, q, key_cache, value_cache, block_tables, context_lengths):
        return triton_paged_attention(q, key_cache, value_cache, block_tables, context_lengths)

    def windowed(self, q, key_cache, value_cache, block_tables, context_lengths, window):
        return triton_paged_attention(
            q, key_cache, value_cache, block_tables, context_lengths, window=window
        )

    def prefill(self, *args, **kwargs):
        return triton_paged_prefill(*args, **kwargs)
