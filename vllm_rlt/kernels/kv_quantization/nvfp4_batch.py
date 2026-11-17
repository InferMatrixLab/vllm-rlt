"""Batched NVFP4 encode for many [tokens, heads, dim] pages in one call.

Each batch entry keeps its own FP32 global scale. Every operation mirrors
nvfp4.pack with a leading batch axis, so the result matches packing each
entry on its own bit for bit.
"""

import torch

from ._common import KVKind, _pack_channel_nibbles, _pad_repeat, group_axis

_BOUNDARIES = [0.25, 0.75 - 2**-24, 1.25, 1.75 - 2**-23, 2.5, 3.5 - 2**-22, 5.0]


@torch.no_grad()
def pack_batch(
    values: torch.Tensor, *, kind: KVKind
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode ``[N, tokens, heads, dim]`` into (payload, scale, global_scale[N])."""
    count, tokens, heads, channels = values.shape
    axis = group_axis(kind) + 1
    work = values.float()
    ones = torch.ones((count,), device=values.device, dtype=torch.float32)
    global_scale = ones
    if values.numel():
        amax = work.abs().flatten(1).amax(dim=1)
        global_scale = torch.where(amax > 0, 2688.0 / amax, ones).clamp_max_(
            torch.finfo(torch.float32).max
        )
    inverse_global = global_scale.reciprocal().view(count, 1, 1, 1, 1)
    scale_global = global_scale.view(count, 1, 1, 1, 1)
    boundaries = torch.tensor(_BOUNDARIES, device=values.device, dtype=torch.float32)
    padded = _pad_repeat(work, axis, 16)
    if kind == "key":
        grouped = padded.view(count, padded.shape[1] // 16, 16, heads, channels)
        reduce_dim = 2
    else:
        grouped = padded.view(count, tokens, heads, padded.shape[3] // 16, 16)
        reduce_dim = 4
    amax = grouped.abs().amax(dim=reduce_dim, keepdim=True)
    scale = (scale_global * (amax * (1.0 / 6))).clamp_(0, 448).to(torch.float8_e4m3fn)
    effective = scale.float() * inverse_global
    multiplier = effective.reciprocal()
    finite = torch.isfinite(multiplier)
    normalized = grouped * torch.where(finite, multiplier, 0.0)
    normalized = torch.where((effective > 0) & ~finite, grouped / effective, normalized).clamp_(
        -6, 6
    )
    codes = torch.bucketize(normalized.abs().contiguous(), boundaries, out_int32=True).to(
        torch.uint8
    )
    codes |= torch.signbit(normalized).to(torch.uint8) << 3
    scales = scale.squeeze(reduce_dim)
    codes = codes.view(padded.shape)
    codes = codes[:, :tokens] if kind == "key" else codes[..., :channels]
    return _pack_channel_nibbles(codes), scales, global_scale
