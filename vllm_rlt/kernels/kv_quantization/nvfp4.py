# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 reference for logical [token, KV head, channel] tensors.

Adapted from vLLM ref_nvfp4_quant / ref_nvfp4_quant_dequant at
c7a760751977e4e3396d586956d8767a9bc44a82 in
vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py.
Unlike the hardware layout, this logical format also supports partial groups.
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

format = "nvfp4"
storage_dtype = torch.uint8
default_group_size = 16


def scale_shape(shape: tuple[int, int, int], group_size: int, *, kind: KVKind) -> tuple[int, ...]:
    """Protocol hook: metadata shape for ``shape`` at ``group_size``."""
    return _common.scale_shape(shape, group_size, kind=kind)


@dataclass(frozen=True)
class PackedNVFP4KV:
    """Uint8 payload, E4M3 block scales and an FP32 global encoding multiplier.

    K groups 16 tokens per channel; V groups 16 channels per token.
    ``scale`` is [ceil(T/16), H, D] for K or [T, H, ceil(D/16)] for V.
    ``global_scale`` is shared by this encoded tensor. Decode multiplies
    E2M1 values by ``scale / global_scale``. Treat tensors as immutable.
    """

    format: ClassVar[Literal["nvfp4"]] = "nvfp4"

    payload: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    original_shape: tuple[int, int, int]
    original_dtype: torch.dtype
    kind: KVKind
    group_size: int = 16

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
        return sum(
            t.numel() * t.element_size() for t in (self.payload, self.scale, self.global_scale)
        )


def storage_bytes(packed: PackedNVFP4KV) -> int:
    return packed.storage_bytes


@torch.no_grad()
def pack(
    values: torch.Tensor, *, kind: KVKind, group_size: int = default_group_size
) -> PackedNVFP4KV:
    """Encode E2M1 with nearest-even rounding and FP8-rounded block scales.

    Groups are formed by one padded reshape along the group axis. ``global_scale``
    is derived from the whole tensor, so it stays a single FP32 scalar.
    """
    group_size = int(group_size)
    _validate_inputs(values, kind=kind, group_size=group_size)
    if group_size != 16:
        raise ValueError("NVFP4 requires group_size=16")
    tokens, heads, channels = values.shape
    shape = (tokens, heads, channels)
    axis = group_axis(kind)
    work = values.float()
    global_scale = torch.ones((), device=values.device, dtype=torch.float32)
    if values.numel():
        amax = work.abs().amax()
        global_scale = torch.where(amax > 0, 2688.0 / amax, global_scale).clamp_max_(
            torch.finfo(torch.float32).max
        )
    inverse_global = global_scale.reciprocal()
    # bucketize keeps equality in the lower bin. Move odd lower-code boundaries
    # down one FP32 ULP so midpoint ties choose the even upper code instead.
    boundaries = torch.tensor(
        [0.25, 0.75 - 2**-24, 1.25, 1.75 - 2**-23, 2.5, 3.5 - 2**-22, 5.0],
        device=values.device,
        dtype=torch.float32,
    )
    padded = _pad_repeat(work, axis, 16)
    if kind == "key":
        grouped = padded.view(padded.shape[0] // 16, 16, *shape[1:])
        reduce_dim = 1
    else:
        grouped = padded.view(shape[0], shape[1], padded.shape[2] // 16, 16)
        reduce_dim = 3
    amax = grouped.abs().amax(dim=reduce_dim, keepdim=True)
    scale = (global_scale * (amax * (1.0 / 6))).clamp_(0, 448).to(torch.float8_e4m3fn)
    effective = scale.float() * inverse_global
    multiplier = effective.reciprocal()
    finite = torch.isfinite(multiplier)
    normalized = grouped * torch.where(finite, multiplier, 0.0)
    # Preserve tiny values when only the FP32 reciprocal overflows.
    normalized = torch.where((effective > 0) & ~finite, grouped / effective, normalized).clamp_(
        -6, 6
    )
    codes = torch.bucketize(normalized.abs().contiguous(), boundaries, out_int32=True).to(
        torch.uint8
    )
    codes |= torch.signbit(normalized).to(torch.uint8) << 3
    scales = scale.squeeze(reduce_dim)
    if kind == "key":
        codes = codes.view(padded.shape)[: shape[0]]
    else:
        codes = codes.view(padded.shape)[..., : shape[2]]
    return PackedNVFP4KV(
        _pack_channel_nibbles(codes), scales, global_scale, shape, values.dtype, kind
    )


@torch.no_grad()
def unpack(packed: PackedNVFP4KV) -> torch.Tensor:
    """Reference/debug full decode; serving kernels should decode tiles in-register."""
    channels = packed.original_shape[2]
    codes = _unpack_channel_nibbles(packed.payload, channels, dtype=torch.uint8)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.payload.device,
        dtype=torch.float32,
    )
    values = torch.index_select(magnitudes, 0, (codes & 7).flatten().to(torch.int32)).reshape(
        codes.shape
    )
    values.copysign_((codes << 4).view(torch.int8))
    axis = packed.axis
    scale = (packed.scale.float() / packed.global_scale).repeat_interleave(16, dim=axis)
    return values.mul_(scale.narrow(axis, 0, packed.original_shape[axis])).to(packed.original_dtype)
