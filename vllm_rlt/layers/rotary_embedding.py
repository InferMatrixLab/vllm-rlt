# SPDX-License-Identifier: Apache-2.0
"""Shared rotary position embedding (RoPE) layers across models."""

import os

import torch
from torch import nn

from vllm_rlt.kernels.dispatch import fused_allowed

# Fused Triton RoPE on CUDA: "exact" (default) is bitwise with the reference (forced
# rounding points), "1"/"approx" lets the compiler keep products in fp32, "0" is eager.
FUSED_ROPE = {"1": "approx", "approx": "approx", "exact": "exact"}.get(
    os.environ.get("VLLM_RLT_FUSED_ROPE", "exact")
)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input tensor."""
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to Query and Key states."""
    if FUSED_ROPE and fused_allowed():
        from vllm_rlt.kernels.rope import fused_rope_, fused_rope_supported

        if fused_rope_supported(q, cos) and fused_rope_supported(k, cos):
            # q and k are fresh projection outputs here, so in-place is safe.
            exact = FUSED_ROPE == "exact"
            return fused_rope_(q, cos, sin, exact=exact), fused_rope_(k, cos, sin, exact=exact)
    return apply_rotary_pos_emb_reference(q, k, cos, sin)


def apply_rotary_pos_emb_reference(q, k, cos, sin):
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with precision-preserving buffer management.

    Accepts `head_dim` and `rope_theta` directly without depending on a specific
    model configuration dataclass.
    """

    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.register_buffer("inv_freq", self.frequencies(), persistent=False)

    def frequencies(self, device: torch.device | str | None = None) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Module.to(dtype) otherwise rounds these constants before forward's
        # float32 cast, changing long-context rotary angles irreversibly.
        self.inv_freq = self.frequencies(device=self.inv_freq.device)
        return self

    def forward(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Force FP32 calculation to maintain accuracy regardless of model dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            angles = positions.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1)
        return (
            angles.cos().to(hidden.dtype).unsqueeze(1),
            angles.sin().to(hidden.dtype).unsqueeze(1),
        )
