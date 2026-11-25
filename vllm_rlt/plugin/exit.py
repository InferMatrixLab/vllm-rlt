"""Early-exit policy, following Ouro's reference implementation.

After each core pass the model produces a gate logit. With
``lambda_i = sigmoid(gate_i)`` the probability of still running is
``remaining = prod(1 - lambda_i)``, and a token exits once
``1 - remaining >= threshold`` (HF Ouro: ``cumsum(pdf) >= threshold``, which is
the same quantity). ``min_loops``/``max_loops``/``target_loops`` bound it.

Per-request parameters arrive through ``SamplingParams.extra_args`` (the
``vllm_xargs`` field of /v1/completions) so no engine request type changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KEYS = ("rlt_exit_threshold", "rlt_min_loops", "rlt_max_loops", "rlt_target_loops")


@dataclass(frozen=True)
class ExitParams:
    threshold: float = 1.0  # >= 1 disables the gate
    min_loops: int = 1
    max_loops: int | None = None
    target_loops: int | None = None  # fixed depth, no gate

    @classmethod
    def from_extra_args(cls, extra: dict | None) -> ExitParams:
        extra = extra or {}
        params = cls(
            threshold=float(extra.get("rlt_exit_threshold", 1.0)),
            min_loops=int(extra.get("rlt_min_loops", 1)),
            max_loops=_opt_int(extra.get("rlt_max_loops")),
            target_loops=_opt_int(extra.get("rlt_target_loops")),
        )
        if not (math.isfinite(params.threshold) and 0.0 <= params.threshold <= 1.0):
            raise ValueError("rlt_exit_threshold must be in [0, 1]")
        for name in ("min_loops", "max_loops", "target_loops"):
            value = getattr(params, name)
            if value is not None and value < 1:
                raise ValueError(f"rlt_{name} must be a positive integer")
        return params

    @property
    def uses_gate(self) -> bool:
        return self.target_loops is None and self.threshold < 1.0

    def limit(self, num_passes: int) -> int:
        bound = self.target_loops or self.max_loops or num_passes
        return max(1, min(bound, num_passes))


def _opt_int(value) -> int | None:
    return None if value is None else int(value)


def update_remaining(remaining: float, gate_logit: float) -> float:
    """Fold one pass's hazard into the probability of still running."""
    return remaining * (1.0 - 1.0 / (1.0 + math.exp(-gate_logit)))


def should_exit(params: ExitParams, passes_done: int, remaining: float, num_passes: int) -> bool:
    """Decision after ``passes_done`` core passes (1-based) with ``remaining``
    already updated for the pass that just finished."""
    if passes_done >= params.limit(num_passes):
        return True
    if not params.uses_gate or passes_done < params.min_loops:
        return False
    return 1.0 - remaining >= params.threshold


def exit_mask(gate_logits, remaining, passes_done, thresholds, min_loops, limits):
    """Vectorized version for the runner. All arguments are 1-D tensors on the
    same device; ``passes_done`` is each row's pass count including the one that
    just ran (rows may be at different depths). ``remaining`` is updated in
    place and the boolean exit mask is returned."""
    import torch

    remaining.mul_(1.0 - torch.sigmoid(gate_logits.float()))
    gated = (1.0 - remaining >= thresholds) & (passes_done >= min_loops)
    return gated | (passes_done >= limits)
