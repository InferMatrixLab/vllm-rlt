"""Batched, host-sync-free sampling helpers for asynchronous speculative decoding."""

from typing import NamedTuple

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


def categorical(weights: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-CDF draw from every row of nonnegative, possibly unnormalized ``[m, V]`` weights.

    ``u`` is ``[m]`` uniforms in ``[0, 1)``. Returns ``(index [m] int64, valid [m] bool)``; a
    row is valid iff its float64 mass is finite and positive. Invalid rows return the legal
    placeholder index 0. Every valid draw lands on a token with a real cumulative increment
    (so never on a zero-weight token). No host synchronization happens here.
    """
    if weights.ndim != 2 or weights.shape[-1] < 1:
        raise ValueError("weights must be a 2-D [m, V] tensor with V >= 1")
    if u.shape != weights.shape[:1]:
        raise ValueError(f"u must have shape [{weights.shape[0]}], got {tuple(u.shape)}")
    vocab = weights.shape[-1]
    # Accumulate in float64: each token's selection probability then equals its weight up to
    # float64 rounding (an FP32 CDF quantizes small tokens after a large mass).
    c = weights.double().cumsum(-1)
    mass = c[:, -1]
    valid = torch.isfinite(mass) & (mass > 0)
    # Scale by the actual mass and clamp the threshold below it (not the index). With
    # right=True the chosen i satisfies c[i] > t >= c[i-1], so it lies in an interval with a
    # real cumulative increment: never a zero-weight token and never a positive-weight tail
    # token whose increment was rounded away (a plateau). t < mass guarantees i <= V - 1.
    t = u.double() * mass
    t = torch.minimum(t, torch.nextafter(mass, torch.zeros_like(mass)))
    index = torch.searchsorted(c, t[:, None], right=True).squeeze(-1)
    index = torch.where(valid, index, torch.zeros_like(index)).clamp(0, vocab - 1)
    return index, valid


class AcceptResult(NamedTuple):
    """Device-side speculative verification result; consume only after checking ``error``.

    ``tokens`` ``[n, W1 + 1]``: ``draft_tokens[:s]``, then the final token, then zeros.
    ``accepted`` ``[n]``: number of leading accepted drafts ``s``. ``final`` ``[n]``: the
    residual/bonus token. ``error`` ``[n]`` bool: the stop point was numerically invalid.
    On an error row, ``final`` is 0; ``accepted`` and the leading draft tokens may still
    be retained. Other fields are not committable results when ``error`` is true.
    """

    tokens: torch.Tensor
    accepted: torch.Tensor
    final: torch.Tensor
    error: torch.Tensor


def speculative_accept(
    draft_tokens: torch.Tensor,
    drafts: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor,
    u_accept: torch.Tensor,
    u_final: torch.Tensor,
) -> AcceptResult:
    """Verify sampled drafts with speculative rejection sampling, batched and host-sync-free.

    ``draft_tokens`` ``[n, W1]`` int64 were sampled from ``q`` ``[n, W1, V]``; ``drafts`` ``[n]``
    is the real draft count ``d_i <= W1`` (columns ``j >= d_i`` are padding and are ignored,
    even if they hold NaN). ``p`` ``[n, W1 + 1, V]`` are target weights: row ``j`` verifies
    draft ``j`` and row ``d_i`` is the bonus row. ``u_accept`` ``[n, W1]`` and ``u_final``
    ``[n]`` are uniforms in ``[0, 1)``. W1 may be 0.

    ``p`` and ``q`` rows are normalized per row here (``w / w.sum()``) and every comparison,
    the residual and the bonus use these normalized values; a row is valid iff its raw mass
    is finite and positive. Each position ``j < d_i`` is a valid accept, a valid reject, or
    invalid (invalid row, ``q(x) == 0``, non-finite value at ``x``, or an out-of-range
    token). The stop point ``s`` is the number of leading valid accepts. Then ``final`` is a
    residual draw if position ``s`` is a valid reject, a bonus draw from ``p[s]`` if
    ``s == d_i``, and an error if position ``s`` is invalid or the final distribution has no
    valid mass. Positions after ``s`` never influence the result. Inputs are not modified.
    """
    n, w1 = draft_tokens.shape
    vocab = p.shape[-1]
    if drafts.shape != (n,) or u_final.shape != (n,) or u_accept.shape != (n, w1):
        raise ValueError("drafts, u_accept and u_final must match draft_tokens [n, W1]")
    if q.shape != (n, w1, vocab) or p.shape != (n, w1 + 1, vocab):
        raise ValueError("q must be [n, W1, V] and p must be [n, W1 + 1, V]")

    rows = torch.arange(n, device=p.device)
    pos = torch.arange(w1 + 1, device=p.device)
    p_mass = p.sum(-1, dtype=torch.float32)  # [n, W1 + 1]
    p_ok = torch.isfinite(p_mass) & (p_mass > 0)
    d = drafts.clamp(0, w1)
    error = drafts != d  # out-of-range counts cannot be honored; flag instead of guessing
    x = draft_tokens.clamp(0, vocab - 1)  # legal gather/output indices even for bad drafts

    if w1 > 0:
        in_range = (draft_tokens >= 0) & (draft_tokens < vocab)
        q_mass = q.sum(-1, dtype=torch.float32)  # [n, W1]
        q_ok = torch.isfinite(q_mass) & (q_mass > 0)
        # Elementwise w / mass gives exactly the entries of the normalized rows.
        px = p[:, :w1].gather(2, x[..., None]).squeeze(-1).float() / p_mass[:, :w1]
        qx = q.gather(2, x[..., None]).squeeze(-1).float() / q_mass
        invalid = ~(
            p_ok[:, :w1] & q_ok & in_range & torch.isfinite(px) & torch.isfinite(qx) & (qx > 0)
        )
        # p >= q ignores u: u * q can round back to q for tiny q and falsely reject p == q.
        accept = ~invalid & ((px >= qx) | (u_accept.float() * qx < px))
        accept &= pos[None, :w1] < d[:, None]
        s = accept.long().cumprod(1).sum(1)  # leading valid accepts, s <= d
        at_end = s == d
        s_q = s.clamp(max=w1 - 1)
        invalid_at_s = invalid[rows, s_q] & ~at_end

        p_row = p[rows, s].float() / p_mass[rows, s][:, None]
        q_row = q[rows, s_q].float() / q_mass[rows, s_q][:, None]
        weights = torch.where(at_end[:, None], p_row, (p_row - q_row).clamp_min(0))
        stop_ok = torch.where(at_end, p_ok[rows, s], ~invalid_at_s)
    else:
        s = torch.zeros_like(d)
        weights = p[:, 0].float() / p_mass[:, :1]
        stop_ok = p_ok[:, 0]

    final, final_valid = categorical(weights, u_final)
    error = error | ~(stop_ok & final_valid)
    final = torch.where(error, torch.zeros_like(final), final)

    drafts_padded = torch.nn.functional.pad(x, (0, 1))
    tokens = torch.where(
        pos[None] < s[:, None],
        drafts_padded,
        torch.where(pos[None] == s[:, None], final[:, None], torch.zeros_like(drafts_padded)),
    )
    return AcceptResult(tokens, s, final, error)
