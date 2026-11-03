"""Affine INT4 KV packing and cross-loop residual reconstruction.

Inputs are logical ``[token, KV head, channel]`` tensors for one request/layer.
K is grouped across stored token rows per channel; V across channels per token.

Adapted from FlashLoop packing.py and cache.py at aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6.
Copyright (c) 2026 FlashLoop contributors. MIT License.
"""

from dataclasses import dataclass
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


def _codes_for_group(
    values: torch.Tensor, *, reduce_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    work = values.float()
    minimum = work.amin(dim=reduce_dim, keepdim=True)
    span = work.amax(dim=reduce_dim, keepdim=True) - minimum
    scale = torch.where(span > 0, span / 15.0, torch.ones_like(span))
    codes = (work - minimum).div_(scale).round_().clamp_(0, 15).to(torch.uint8)
    return codes, minimum.to(torch.float16), scale.to(torch.float16)


def _pack_channel_nibbles(codes: torch.Tensor) -> torch.Tensor:
    payload = codes[..., 0::2].clone(memory_format=torch.contiguous_format)
    payload[..., : codes.shape[-1] // 2] |= codes[..., 1::2] << 4
    return payload


def _unpack_channel_nibbles(payload: torch.Tensor, channels: int) -> torch.Tensor:
    codes = torch.empty((*payload.shape[:-1], channels), dtype=torch.float32, device=payload.device)
    codes[..., 0::2] = payload & 0x0F
    codes[..., 1::2] = (payload >> 4)[..., : channels // 2]
    return codes


@dataclass(frozen=True)
class PackedInt4KV:
    """Encoded tensor and FP16 affine parameters; treat its tensors as immutable.

    ``payload`` is uint8 ``[T, H, ceil(D/2)]``. For K, parameters are
    ``[ceil(T/G), H, D]``; for V, ``[T, H, ceil(D/G)]``. An odd last channel
    leaves the high nibble zero. Constant groups use code zero and scale one.
    """

    payload: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    original_shape: tuple[int, int, int]
    original_dtype: torch.dtype
    kind: KVKind
    group_size: int

    @property
    def storage_bytes(self) -> int:
        """Payload and parameter bytes, excluding allocator overhead and decode scratch."""
        return sum(t.numel() * t.element_size() for t in (self.payload, self.minimum, self.scale))


@torch.no_grad()
def pack_kv(values: torch.Tensor, *, kind: KVKind, group_size: int = 64) -> PackedInt4KV:
    """Quantize logical KV in FP32, storing uint8 payload and FP16 parameters."""
    group_size = int(group_size)
    _validate_inputs(values, kind=kind, group_size=group_size)
    shape = tuple(values.shape)
    axis = 0 if kind == "key" else 2
    codes = torch.empty_like(values, dtype=torch.uint8)
    minima, scales = [], []
    for start in range(0, shape[axis], group_size):
        length = min(group_size, shape[axis] - start)
        group_codes, minimum, scale = _codes_for_group(
            values.narrow(axis, start, length), reduce_dim=axis
        )
        codes.narrow(axis, start, length).copy_(group_codes)
        minima.append(minimum)
        scales.append(scale)
    metadata_shape = list(shape)
    metadata_shape[axis] = (shape[axis] + group_size - 1) // group_size
    minimum = (
        torch.cat(minima, dim=axis)
        if minima
        else torch.empty(metadata_shape, device=values.device, dtype=torch.float16)
    )
    scale = torch.cat(scales, dim=axis) if scales else torch.empty_like(minimum)
    payload = _pack_channel_nibbles(codes)
    return PackedInt4KV(payload, minimum, scale, shape, values.dtype, kind, group_size)


@torch.no_grad()
def unpack_kv(packed: PackedInt4KV) -> torch.Tensor:
    """Reconstruct with stored parameters in FP32, then cast to compute dtype."""
    codes = _unpack_channel_nibbles(packed.payload, packed.original_shape[2])
    axis = 0 if packed.kind == "key" else 2
    length = packed.original_shape[axis]
    group_size = min(packed.group_size, length)
    minimum = packed.minimum.repeat_interleave(group_size, axis).narrow(axis, 0, length)
    scale = packed.scale.repeat_interleave(group_size, axis).narrow(axis, 0, length)
    return codes.mul_(scale).add_(minimum).to(packed.original_dtype)


def _validate_positions(positions: torch.Tensor, rows: int, tokens: int) -> None:
    if positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if positions.numel() != rows:
        raise ValueError("positions must identify every compact residual row")
    if (
        rows
        and (
            (positions[0] < 0) | (positions[-1] >= tokens) | (positions[1:] <= positions[:-1]).any()
        ).item()
    ):
        raise ValueError("positions must be strictly increasing and within the logical token range")


@dataclass(frozen=True)
class Int4Residual:
    """One depth's compact residual and its logical scatter map.

    ``positions=None`` denotes a dense update; an empty map denotes inheritance
    at every position. Positions affect scatter only: K groups follow compact
    row order, as in the pinned upstream cache.
    """

    data: PackedInt4KV
    positions: torch.Tensor | None
    num_tokens: int

    @property
    def storage_bytes(self) -> int:
        mapping_bytes = (
            0 if self.positions is None else self.positions.numel() * self.positions.element_size()
        )
        return self.data.storage_bytes + mapping_bytes


@torch.no_grad()
def encode_residual(
    current: torch.Tensor,
    reconstructed_previous: torch.Tensor,
    *,
    kind: KVKind,
    positions: torch.Tensor | None = None,
    group_size: int = 64,
) -> Int4Residual:
    """Encode ``current - reconstructed_previous`` in the input dtype.

    Optional positions map compact rows into the full predecessor and are copied
    as contiguous int32 on the input device, matching FlashLoop's index format.
    """
    if (
        current.dtype != reconstructed_previous.dtype
        or current.device != reconstructed_previous.device
    ):
        raise ValueError("current and predecessor must have matching dtype and device")
    if positions is None:
        if current.shape != reconstructed_previous.shape:
            raise ValueError("dense residual requires matching current and predecessor shapes")
        predecessor = reconstructed_previous
    else:
        if current.shape[1:] != reconstructed_previous.shape[1:]:
            raise ValueError("current and predecessor must have matching heads and channels")
        positions = positions.to(
            device=current.device,
            dtype=torch.int32,
            copy=True,
            memory_format=torch.contiguous_format,
        )
        _validate_positions(positions, current.shape[0], reconstructed_previous.shape[0])
        predecessor = reconstructed_previous.index_select(0, positions)
    data = pack_kv(current - predecessor, kind=kind, group_size=group_size)
    return Int4Residual(data, positions, reconstructed_previous.shape[0])


@torch.no_grad()
def decode_residual(residual: Int4Residual, previous: torch.Tensor) -> torch.Tensor:
    """Return a new full tensor; absent rows retain their previous values."""
    expected_shape = (residual.num_tokens, *residual.data.original_shape[1:])
    if previous.shape != expected_shape:
        raise ValueError("predecessor shape does not match residual logical extent")
    if (
        previous.dtype != residual.data.original_dtype
        or previous.device != residual.data.payload.device
    ):
        raise ValueError("predecessor dtype and device must match residual")
    delta = unpack_kv(residual.data)
    if residual.positions is None:
        return previous + delta
    result = previous.clone()
    indices = residual.positions.long()
    result.index_copy_(0, indices, previous.index_select(0, indices).add_(delta))
    return result
