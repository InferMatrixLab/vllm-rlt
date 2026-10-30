# SPDX-License-Identifier: Apache-2.0
"""Shared normalization layers across models."""

import os

import torch
from torch import nn

from vllm_rlt.kernels.dispatch import fused_allowed

# Fused Triton RMSNorm on CUDA (see vllm_rlt/kernels/norm.py):
# "exact" (default): torch computes the mean, Triton the elementwise parts; bitwise identical
# to ``reference`` with 3 launches per norm instead of 8. "1"/"approx": one launch with its
# own fp32 reduction order (last-ulp differences, changes greedy outputs). "0": reference.
FUSED_NORM = os.environ.get("VLLM_RLT_FUSED_NORM", "exact")
FUSED_NORM = {"1": "approx", "approx": "approx", "exact": "exact"}.get(FUSED_NORM)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Retains FP32 precision during reduction to preserve numerical stability
    across all input activation dtypes (BF16/FP16/FP32).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """``weight * norm(hidden)``, plus ``residual`` when given (``residual + norm``)."""
        if FUSED_NORM and self.weight.dtype == hidden.dtype and fused_allowed():
            from vllm_rlt.kernels.norm import (
                fused_rmsnorm,
                fused_rmsnorm_exact,
                fused_rmsnorm_supported,
            )

            if fused_rmsnorm_supported(hidden) and (
                residual is None or residual.dtype == hidden.dtype
            ):
                kernel = fused_rmsnorm_exact if FUSED_NORM == "exact" else fused_rmsnorm
                return kernel(hidden, self.weight, self.variance_epsilon, residual)
        out = self.reference(hidden)
        return out if residual is None else residual + out

    def reference(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(hidden.dtype)
