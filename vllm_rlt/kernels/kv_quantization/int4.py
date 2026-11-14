"""Affine INT4 reference codec for logical [token, KV head, channel] tensors.

Adapted from FlashLoop packing.py at aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6.
Copyright (c) 2026 FlashLoop contributors. MIT License; see NOTICE.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from . import _common
from ._common import (
    KVKind,
    _pack_channel_nibbles,
    _pad_repeat,
    _unpack_channel_nibbles,
    _validate_inputs,
    group_axis,
)

format = "int4"
storage_dtype = torch.uint8
default_group_size = 64


def scale_shape(shape: tuple[int, int, int], group_size: int, *, kind: KVKind) -> tuple[int, ...]:
    """Protocol hook: metadata shape for ``shape`` at ``group_size``."""
    return _common.scale_shape(shape, group_size, kind=kind)


def _codes_for_group(
    values: torch.Tensor, *, reduce_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    work = values.float()
    minimum = work.amin(dim=reduce_dim, keepdim=True)
    span = work.amax(dim=reduce_dim, keepdim=True) - minimum
    scale = torch.where(span > 0, span / 15.0, torch.ones_like(span))
    codes = (work - minimum).div_(scale).round_().clamp_(0, 15).to(torch.uint8)
    return codes, minimum.to(torch.float16), scale.to(torch.float16)


@dataclass(frozen=True)
class PackedInt4KV:
    """Encoded tensor and FP16 affine parameters; treat its tensors as immutable.

    ``payload`` is uint8 ``[T, H, ceil(D/2)]``. For K, parameters are
    ``[ceil(T/G), H, D]``; for V, ``[T, H, ceil(D/G)]``. An odd last channel
    leaves the high nibble zero. Constant groups use code zero and scale one.
    """

    format: ClassVar[Literal["int4"]] = "int4"

    payload: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    original_shape: tuple[int, int, int]
    original_dtype: torch.dtype
    kind: KVKind
    group_size: int

    @property
    def axis(self) -> int:
        return group_axis(self.kind)

    @property
    def group_axis(self) -> int:
        return self.axis

    @property
    def scale_shape(self) -> tuple[int, ...]:
        return tuple(self.scale.shape)

    @property
    def storage_bytes(self) -> int:
        """Payload and parameter bytes, excluding allocator overhead and decode scratch."""
        return sum(t.numel() * t.element_size() for t in (self.payload, self.minimum, self.scale))


def storage_bytes(packed: PackedInt4KV) -> int:
    return packed.storage_bytes


@torch.no_grad()
def pack(
    values: torch.Tensor, *, kind: KVKind, group_size: int = default_group_size
) -> PackedInt4KV:
    """Quantize logical KV in FP32, storing uint8 payload and FP16 parameters.

    Groups are formed by one padded reshape along the group axis; the padded
    tail repeats the last slice so padding never moves an extreme.
    Parameters are stored as FP16: a group whose minimum magnitude or
    ``span / 15`` exceeds 65504 overflows to infinity, so float32/float64
    inputs must keep both within FP16 range.
    """
    group_size = int(group_size)
    _validate_inputs(values, kind=kind, group_size=group_size)
    tokens, heads, channels = values.shape
    shape = (tokens, heads, channels)
    axis = group_axis(kind)
    work = values.float()
    padded = _pad_repeat(work, axis, group_size)
    if kind == "key":
        grouped = padded.view(padded.shape[0] // group_size, group_size, *shape[1:])
        codes, minimum, scale = _codes_for_group(grouped, reduce_dim=1)
        codes = codes.view(padded.shape)[: shape[0]]
        minimum, scale = minimum.squeeze(1), scale.squeeze(1)
    else:
        grouped = padded.view(shape[0], shape[1], padded.shape[2] // group_size, group_size)
        codes, minimum, scale = _codes_for_group(grouped, reduce_dim=3)
        codes = codes.view(padded.shape)[..., : shape[2]]
        minimum, scale = minimum.squeeze(3), scale.squeeze(3)
    return PackedInt4KV(
        _pack_channel_nibbles(codes), minimum, scale, shape, values.dtype, kind, group_size
    )


@torch.no_grad()
def unpack(packed: PackedInt4KV) -> torch.Tensor:
    """Reference/debug full decode; serving kernels should decode tiles in-register."""
    codes = _unpack_channel_nibbles(packed.payload, packed.original_shape[2])
    axis = packed.axis
    length = packed.original_shape[axis]
    group_size = min(packed.group_size, length)
    minimum = packed.minimum.repeat_interleave(group_size, axis).narrow(axis, 0, length)
    scale = packed.scale.repeat_interleave(group_size, axis).narrow(axis, 0, length)
    return codes.mul_(scale).add_(minimum).to(packed.original_dtype)
