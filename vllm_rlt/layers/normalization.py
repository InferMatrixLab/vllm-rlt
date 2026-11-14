# SPDX-License-Identifier: Apache-2.0
"""Shared normalization layers across models."""

import torch
from torch import nn

from vllm_rlt.kernels.normalization import rms_norm, rms_norm_reference


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Retains FP32 precision during reduction to preserve numerical stability
    across all input activation dtypes (BF16/FP16/FP32).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, *, use_native: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_native = use_native

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_native:
            return rms_norm(hidden, self.weight, self.variance_epsilon, residual)
        return rms_norm_reference(hidden, self.weight, self.variance_epsilon, residual)
