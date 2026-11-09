"""Shared logical KV validation and nibble packing.

Adapted from FlashLoop packing.py at aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6.
Copyright (c) 2026 FlashLoop contributors. MIT License; see NOTICE.
"""

from typing import Literal, Protocol

import torch

KVKind = Literal["key", "value"]
_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


class PackedGroupKV(Protocol):
    """Logical tensor encoded in groups along one axis.

    ``group_axis`` and ``scale_shape`` let callers size and place parameters
    without branching on format. Concrete formats carry their own quantization
    parameters, and physical page ownership is managed separately.
    """

    @property
    def format(self) -> str: ...

    @property
    def payload(self) -> torch.Tensor: ...

    @property
    def original_shape(self) -> tuple[int, int, int]: ...

    @property
    def original_dtype(self) -> torch.dtype: ...

    @property
    def group_size(self) -> int: ...

    @property
    def group_axis(self) -> int: ...

    @property
    def scale_shape(self) -> tuple[int, ...]: ...

    @property
    def storage_bytes(self) -> int: ...


def group_axis(kind: KVKind) -> int:
    return 0 if kind == "key" else 2


def scale_shape(shape: tuple[int, int, int], group_size: int, *, kind: KVKind) -> tuple[int, ...]:
    dimensions = list(shape)
    axis = group_axis(kind)
    dimensions[axis] = (dimensions[axis] + group_size - 1) // group_size
    return tuple(dimensions)


def _validate_inputs(values: torch.Tensor, *, kind: KVKind, group_size: int) -> None:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if kind not in {"key", "value"}:
        raise ValueError("kind must be 'key' or 'value'")
    if values.ndim != 3 or values.shape[2] <= 0:
        raise ValueError("KV values must have shape [tokens, heads, positive channels]")
    if values.dtype not in _DTYPES:
        raise ValueError("KV values must use a floating-point dtype")


def _pad_repeat(values: torch.Tensor, axis: int, multiple: int) -> torch.Tensor:
    """Pad ``axis`` to a multiple by repeating the last slice (keeps extremes)."""
    length = values.shape[axis]
    pad = (-length) % multiple
    if pad == 0:
        return values
    index = [slice(None)] * values.ndim
    index[axis] = slice(length - 1, length)
    tail = values[tuple(index)]
    shape = list(values.shape)
    shape[axis] = pad
    return torch.cat([values, tail.expand(shape)], dim=axis)


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
