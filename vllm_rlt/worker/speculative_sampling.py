"""Batched, host-sync-free sampling helpers for asynchronous speculative decoding."""

import torch


def batched_probabilities(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    *,
    max_top_k: int,
    apply_top_p: bool,
) -> torch.Tensor:
    """Return the temperature/top-k/top-p distribution of every row as ``[n, V]`` float32.

    Row ``i`` equals ``Sampler.probabilities(logits[i], SamplingParams(temperature=t_i,
    top_k=k_i, top_p=p_i))``: top-k keeps exactly ``min(k, V)`` candidates by index (ties
    resolve to an unspecified index), top-p uses the strict ``>`` cumulative test and keeps
    the token that crosses the threshold. Rows with ``top_k <= 0`` or ``top_k >= V`` are not
    truncated, and rows with ``top_p == 1`` are never trimmed.

    Only non-greedy rows may be passed (``temperature > 0``). No host synchronization
    happens here: ``max_top_k`` (the largest effective ``k`` over truncating rows, 0 if no
    row truncates) and ``apply_top_p`` (whether any row has ``top_p < 1``) come from the
    host-side plan. Inputs are never modified.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be a 2-D [n, V] tensor")
    n, vocab = logits.shape
    for name, value in (("temperature", temperature), ("top_k", top_k), ("top_p", top_p)):
        if value.shape != (n,):
            raise ValueError(f"{name} must have shape [{n}], got {tuple(value.shape)}")
    if max_top_k < 0:
        raise ValueError("max_top_k must be nonnegative")

    x = logits.float() / temperature.float()[:, None]
    if max_top_k > 0:
        k_eff = torch.where((top_k > 0) & (top_k < vocab), top_k, vocab)
        truncates = k_eff < vocab
        values, indices = x.topk(min(max_top_k, vocab), dim=-1)
        keep = torch.arange(values.shape[-1], device=x.device) < k_eff[:, None]
        masked = torch.full_like(x, -torch.inf).scatter_(
            1, indices, values.masked_fill(~keep, -torch.inf)
        )
        x = torch.where(truncates[:, None], masked, x)
    if apply_top_p:
        p = top_p.float()[:, None]
        sorted_x, indices = x.sort(dim=-1, descending=True)
        remove = (sorted_x.softmax(-1).cumsum(-1) > p) & (p < 1)
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        x = x.scatter(1, indices, sorted_x.masked_fill(remove, -torch.inf))
    return x.softmax(-1)
