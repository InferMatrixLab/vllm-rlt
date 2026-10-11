"""Shared distribution construction and exact speculative rejection sampling."""

import torch

from vllm_rlt.worker.sampler import (
    apply_repetition_penalty,
    probabilities,
)

__all__ = [
    "apply_repetition_penalty",
    "draw",
    "generator_for",
    "probabilities",
    "rejection_sample",
]


def generator_for(request, device):
    if request.generator is None:
        request.generator = torch.Generator(device=device).manual_seed(request.sampling_params.seed)
    return request.generator


def draw(probs, generator):
    return torch.multinomial(probs, 1, generator=generator).squeeze(0)


def rejection_sample(candidate, target, proposal, generator):
    """Return (token, accepted) for normalized, actually sampled p and q.

    Comparing u*q to p avoids division by tiny q. A zero residual after a real
    rejection is a numerical error, never a reason to silently sample from p.
    """
    mass = proposal[candidate]
    if not bool(mass > 0):
        raise ValueError("candidate has zero proposal probability")
    uniform = torch.rand((), device=target.device, generator=generator)
    if bool(uniform * mass < target[candidate]):
        return candidate, True
    residual = (target - proposal).clamp_min(0)
    total = residual.sum()
    if not bool(torch.isfinite(total) & (total > 0)):
        raise RuntimeError("invalid speculative residual probability mass")
    return int(draw(residual / total, generator).item()), False
