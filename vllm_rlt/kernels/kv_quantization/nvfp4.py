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

from ._common import KVKind, _pack_channel_nibbles, _unpack_channel_nibbles, _validate_inputs


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
        return 0 if self.kind == "key" else 2

    @property
    def storage_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for t in (self.payload, self.scale, self.global_scale)
        )


@torch.no_grad()
def pack(values: torch.Tensor, *, kind: KVKind, group_size: int = 16) -> PackedNVFP4KV:
    """Encode E2M1 with nearest-even rounding and FP8-rounded block scales."""
    group_size = int(group_size)
    _validate_inputs(values, kind=kind, group_size=group_size)
    if group_size != 16:
        raise ValueError("NVFP4 requires group_size=16")
    shape = tuple(values.shape)
    axis = 0 if kind == "key" else 2
    global_scale = torch.ones((), device=values.device, dtype=torch.float32)
    if values.numel():
        amax = values.abs().amax().float()
        global_scale = torch.where(amax > 0, 2688.0 / amax, global_scale).clamp_max_(
            torch.finfo(torch.float32).max
        )
    inverse_global = global_scale.reciprocal()
    codes = torch.empty_like(values, dtype=torch.uint8)
    scale_shape = list(shape)
    scale_shape[axis] = (shape[axis] + 15) // 16
    scales = torch.empty(
        scale_shape,
        dtype=torch.float8_e4m3fn,
        device=values.device,
    )
    # bucketize keeps equality in the lower bin. Move odd lower-code boundaries
    # down one FP32 ULP so midpoint ties choose the even upper code instead.
    boundaries = torch.tensor(
        [0.25, 0.75 - 2**-24, 1.25, 1.75 - 2**-23, 2.5, 3.5 - 2**-22, 5.0],
        device=values.device,
        dtype=torch.float32,
    )
    for start in range(0, shape[axis], 16):
        length = min(16, shape[axis] - start)
        work = values.narrow(axis, start, length).float()
        scale = (global_scale * (work.abs().amax(dim=axis, keepdim=True) * (1.0 / 6))).clamp_(
            0, 448
        )
        scale = scale.to(torch.float8_e4m3fn)
        effective_scale = scale.float() * inverse_global
        multiplier = effective_scale.reciprocal()
        finite_multiplier = torch.isfinite(multiplier)
        normalized = work * torch.where(finite_multiplier, multiplier, 0.0)
        # Preserve tiny values when only the FP32 reciprocal overflows.
        normalized = torch.where(
            (effective_scale > 0) & ~finite_multiplier, work / effective_scale, normalized
        ).clamp_(-6, 6)
        magnitude = normalized.abs().contiguous()
        group_codes = torch.bucketize(magnitude, boundaries, out_int32=True).to(torch.uint8)
        group_codes |= torch.signbit(normalized).to(torch.uint8) << 3
        codes.narrow(axis, start, length).copy_(group_codes)
        scales.select(axis, start // 16).copy_(scale.squeeze(axis))
    return PackedNVFP4KV(
        _pack_channel_nibbles(codes), scales, global_scale, shape, values.dtype, kind
    )


@torch.no_grad()
def unpack(packed: PackedNVFP4KV) -> torch.Tensor:
    """Decode with stored block/global scales in FP32, then cast to compute dtype."""
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
