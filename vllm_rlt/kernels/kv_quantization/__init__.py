"""Logical INT4/NVFP4 KV codecs and cross-loop residual reconstruction.

Inputs are [token, KV head, channel] tensors for one request/layer.
Residual arithmetic adapts FlashLoop cache.py at
aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6.
Copyright (c) 2026 FlashLoop contributors. MIT License; see NOTICE.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from . import int4, nvfp4
from ._common import KVKind
from .int4 import PackedInt4KV
from .nvfp4 import PackedNVFP4KV

KVFormat = Literal["int4", "nvfp4"]
PackedKV = PackedInt4KV | PackedNVFP4KV


def pack_kv(
    values: torch.Tensor,
    *,
    kind: KVKind,
    format: KVFormat = "int4",
    group_size: int | None = None,
) -> PackedKV:
    """Encode logical KV; defaults are INT4 G=64 or NVFP4 G=16."""
    if format == "int4":
        return int4.pack(values, kind=kind, group_size=64 if group_size is None else group_size)
    if format == "nvfp4":
        return nvfp4.pack(values, kind=kind, group_size=16 if group_size is None else group_size)
    raise ValueError("format must be 'int4' or 'nvfp4'")


def unpack_kv(packed: PackedKV) -> torch.Tensor:
    """Decode using the format carried by the packed tensor."""
    if isinstance(packed, PackedInt4KV):
        return int4.unpack(packed)
    if isinstance(packed, PackedNVFP4KV):
        return nvfp4.unpack(packed)
    raise TypeError("expected PackedInt4KV or PackedNVFP4KV")


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
class KVResidual:
    """One depth's compact residual and its logical scatter map.

    ``positions=None`` denotes a dense update; an empty map denotes inheritance
    at every position. K groups follow compact row order for both formats;
    V groups remain within each compact row, along channels.
    """

    data: PackedKV
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
    format: KVFormat = "int4",
    group_size: int | None = None,
) -> KVResidual:
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
    data = pack_kv(current - predecessor, kind=kind, format=format, group_size=group_size)
    return KVResidual(data, positions, reconstructed_previous.shape[0])


@torch.no_grad()
def decode_residual(residual: KVResidual, previous: torch.Tensor) -> torch.Tensor:
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
