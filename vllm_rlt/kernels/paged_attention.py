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
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference attention over [physical block, token, KV head, dimension].

    FP8 pools pass per-vector scales shaped [physical block, token, KV head].
    """
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
        keys = key_cache[blocks, offsets].float()
        values = value_cache[blocks, offsets].float()
        if key_scale is not None:
            keys = keys * key_scale[blocks, offsets].unsqueeze(-1)
            values = values * value_scale[blocks, offsets].unsqueeze(-1)
        keys = keys.repeat_interleave(head_groups, dim=1)
        values = values.repeat_interleave(head_groups, dim=1)
        scores = torch.einsum("hd,thd->ht", q[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum("ht,thd->hd", probabilities, values.float()).to(q.dtype)
    return output


def triton_paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch explicitly to Triton, with no silent reference fallback."""
    if q.device.type != "cuda":
        raise ValueError("the Triton attention backend requires a CUDA or ROCm device")
    if q.shape[-1] > 256:
        raise ValueError("the Triton attention backend supports head_dim <= 256")
    # Keep Triton optional and avoid loading its runtime for CPU-only callers.
    from vllm_rlt.kernels.triton_attention import paged_attention

    return paged_attention(
        q, key_cache, value_cache, block_tables, context_lengths, key_scale, value_scale
    )
