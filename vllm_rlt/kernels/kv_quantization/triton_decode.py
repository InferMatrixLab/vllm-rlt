"""Gather packed KV pages of one layer and decode them straight into BF16 pages.

The Torch reference (``unpack_kv``) materializes FP32 codes and expanded
parameters, then the caller copies the result into the attention input. This
kernel reads each page's payload and group parameters once and writes the
decoded value into its destination row, matching the reference bit for bit:
arithmetic stays in FP32 with the same operation order and no FMA contraction.
"""

import torch
import triton
import triton.language as tl

_TILE = 1024


@triton.jit
def _e2m1(code):
    # E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 as FP32 bits: m = 2..7 is
    # (m + 252) << 22, m = 1 is 252 << 22. Bit 3 is the sign, so 0b1000 is -0.
    m = code & 7
    magnitude = tl.where(m >= 2, m + 252, tl.where(m == 1, 252, 0)) << 22
    return (magnitude | ((code & 8) << 28)).to(tl.float32, bitcast=True)


def _pool_args(pool, format):
    """(payload, first, second, global) tensors plus their [block, layer] strides.

    INT4 reads minimum and scale as first and second; NVFP4 reads its E4M3
    scale as first and its per-page FP32 global scale as global.
    """
    payload = pool["payload"]
    if format == "nvfp4":
        first = second = pool["scale"]
        global_scale = pool["global_scale"]
        global_strides = global_scale.stride()[:2]
    elif format == "int4":
        first, second = pool["minimum"], pool["scale"]
        global_scale, global_strides = second, (0, 0)
    else:
        raise ValueError(f"unknown KV format: {format!r}")
    for tensor in (payload, first, second):
        if not tensor.is_contiguous():
            raise ValueError("packed KV pools must be contiguous")
    if first.stride() != second.stride():
        raise ValueError("INT4 minimum and scale pools must share a layout")
    tensors = (payload, first, second, global_scale)
    return tensors, (*payload.stride()[:2], *first.stride()[:2], *global_strides)


@triton.jit
def _decode_pages(
    Out,
    Rows,
    Blocks,
    Payload,
    First,
    Second,
    Global,
    out_page,
    pool_block,
    pool_layer,
    param_block,
    param_layer,
    global_block,
    global_layer,
    layer,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    IS_KEY: tl.constexpr,
    IS_NVFP4: tl.constexpr,
    TILE: tl.constexpr,
):
    page = tl.program_id(0)
    token = tl.program_id(1)
    index = tl.program_id(2) * TILE + tl.arange(0, TILE)
    valid = index < HEADS * DIM
    head, dim = index // DIM, index % DIM
    # Promote before multiplying: pool offsets exceed int32 for large caches.
    block = tl.load(Blocks + page).to(tl.int64)
    row = tl.load(Rows + page).to(tl.int64)
    packed_dim = (DIM + 1) // 2
    payload = Payload + block * pool_block + layer * pool_layer
    byte = tl.load(payload + (token * HEADS + head) * packed_dim + dim // 2, mask=valid, other=0)
    code = ((byte >> ((dim % 2) * 4).to(tl.uint8)) & 0x0F).to(tl.int32)
    if IS_KEY:
        param = ((token // GROUP) * HEADS + head) * DIM + dim
    else:
        groups = (DIM + GROUP - 1) // GROUP
        param = (token * HEADS + head) * groups + dim // GROUP
    base = block * param_block + layer * param_layer
    if IS_NVFP4:
        scale = tl.load(First + base + param, mask=valid, other=0.0).to(tl.float32)
        global_scale = tl.load(Global + block * global_block + layer * global_layer)
        value = _e2m1(code) * tl.div_rn(scale, global_scale)
    else:
        scale = tl.load(Second + base + param, mask=valid, other=0.0).to(tl.float32)
        minimum = tl.load(First + base + param, mask=valid, other=0.0).to(tl.float32)
        value = code.to(tl.float32) * scale + minimum
    out = Out + row * out_page + token * HEADS * DIM + index
    tl.store(out, value.to(Out.dtype.element_ty), mask=valid)


def decode_pages_into(
    out: torch.Tensor,
    rows: torch.Tensor,
    blocks: torch.Tensor,
    pool: dict[str, torch.Tensor],
    layer: int,
    *,
    format: str,
    kind: str,
    group_size: int,
) -> None:
    """Write decoded ``pool[blocks, layer]`` pages into ``out[rows]``.

    ``out`` is contiguous ``[pages, tokens, heads, dim]``; ``pool`` holds the
    manager's ``[blocks, layers, ...]`` storages, each contiguous.
    """
    count = blocks.numel()
    if not count:
        return
    _, tokens, heads, dim = out.shape
    if not out.is_contiguous():
        raise ValueError("decode_pages_into requires a contiguous output")
    tensors, strides = _pool_args(pool, format)
    grid = (count, tokens, triton.cdiv(heads * dim, _TILE))
    _decode_pages[grid](
        out,
        rows,
        blocks,
        *tensors,
        out.stride(0),
        *strides,
        layer,
        HEADS=heads,
        DIM=dim,
        GROUP=group_size,
        IS_KEY=kind == "key",
        IS_NVFP4=format == "nvfp4",
        TILE=min(_TILE, triton.next_power_of_2(heads * dim)),
        enable_fp_fusion=False,
    )
