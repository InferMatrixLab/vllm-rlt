"""Shared logical KV validation and nibble packing.

Adapted from FlashLoop packing.py at aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6.
Copyright (c) 2026 FlashLoop contributors. MIT License; see NOTICE.
"""

from typing import Literal

import torch

KVKind = Literal["key", "value"]
_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _validate_inputs(values: torch.Tensor, *, kind: KVKind, group_size: int) -> None:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if kind not in {"key", "value"}:
        raise ValueError("kind must be 'key' or 'value'")
    if values.ndim != 3 or values.shape[2] <= 0:
        raise ValueError("KV values must have shape [tokens, heads, positive channels]")
    if values.dtype not in _DTYPES:
        raise ValueError("KV values must use a floating-point dtype")


def _pack_channel_nibbles(codes: torch.Tensor) -> torch.Tensor:
    payload = codes[..., 0::2].clone(memory_format=torch.contiguous_format)
    payload[..., : codes.shape[-1] // 2] |= codes[..., 1::2] << 4
    return payload


def _unpack_channel_nibbles(
    payload: torch.Tensor, channels: int, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    codes = torch.empty((*payload.shape[:-1], channels), dtype=dtype, device=payload.device)
    codes[..., 0::2] = payload & 0x0F
    codes[..., 1::2] = (payload >> 4)[..., : channels // 2]
    return codes
