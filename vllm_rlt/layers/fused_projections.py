# SPDX-License-Identifier: Apache-2.0
"""One linear weight with disjoint row slices for legacy checkpoint keys."""

import torch
from torch import nn
from torch.nn import functional as F


class FusedProjectionLayer(nn.Module):
    """Base for Ouro layers whose input projections share a single parameter."""

    def __init__(self, in_features: int, projections: tuple[tuple[str, int], ...]) -> None:
        super().__init__()
        self.projection_sizes = tuple(width for _, width in projections)
        self.checkpoint_slices = {}
        offset = 0
        for name, width in projections:
            self.checkpoint_slices[f"{name}.weight"] = slice(offset, offset + width)
            offset += width
        self.input_weight = nn.Parameter(torch.empty(offset, in_features))
        for rows in self.checkpoint_slices.values():
            nn.init.kaiming_uniform_(self.input_weight[rows], a=5**0.5)

    def project(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return F.linear(hidden, self.input_weight).split(self.projection_sizes, dim=-1)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        weight = destination.pop(prefix + "input_weight")
        for name, rows in self.checkpoint_slices.items():
            destination[prefix + name] = weight[rows]

    @torch.no_grad()
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        packed_key = prefix + "input_weight"
        if packed_key in state_dict:
            unexpected_keys.append(packed_key)
        pieces = {}
        for name, rows in self.checkpoint_slices.items():
            key = prefix + name
            if key not in state_dict:
                missing_keys.append(key)
                continue
            piece = state_dict.pop(key)
            expected = self.input_weight[rows].shape
            if not isinstance(piece, torch.Tensor) or piece.shape != expected:
                error_msgs.append(f"size or tensor type mismatch for {key}: expected {expected}")
                continue
            pieces[name] = piece

        assign = local_metadata.get("assign_to_params_buffers", False)
        weight = self.input_weight.detach()
        if pieces and assign:
            first = next(iter(pieces.values()))
            compatible = all(
                part.device == first.device and part.dtype == first.dtype
                for part in pieces.values()
            )
            partial_meta = (
                weight.is_meta and not first.is_meta and len(pieces) != len(self.checkpoint_slices)
            )
            if not compatible or partial_meta:
                error_msgs.append(f"incompatible slices for assigning fused weight at {prefix}")
                pieces.clear()
            else:
                weight = torch.empty_like(weight, device=first.device, dtype=first.dtype)
                for name, rows in self.checkpoint_slices.items():
                    if name not in pieces:
                        weight[rows].copy_(self.input_weight.detach()[rows])

        for name, piece in pieces.items():
            rows = self.checkpoint_slices[name]
            if not assign and weight.is_meta and not piece.is_meta:
                # Let PyTorch report its normal no-op warning for meta parameters.
                if len(pieces) == len(self.checkpoint_slices):
                    weight = torch.cat(tuple(pieces.values()), dim=0)
                break
            try:
                weight[rows].copy_(piece)
            except Exception as exc:
                error_msgs.append(f"copying {prefix + name} into the fused weight: {exc}")
        state_dict[packed_key] = weight
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
