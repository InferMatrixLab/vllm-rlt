"""Single-query paged attention that reads INT4/NVFP4 KV pages in place.

The reference path decodes every page a batch reads into BF16 pages and then
runs ``paged_attention``, so each decode step rewrites the whole history. This
kernel decodes each [BLOCK_T, head_dim] tile in registers right before its dot
product, as vLLM's TurboQuant decode does. A page is either still staged in
the BF16 cache or packed; ``SOURCES`` says which, per page of the remapped
table: a staging slot (>= 0) or ``-(block + 1)`` for a packed block. Tiles may
span pages, so each token selects its source with masked loads.

Decode uses the same FP32 operations as ``triton_decode`` (no FMA contraction)
and rounds to the cache dtype; the softmax reuses the reference tiling, split
layout and merge. INT4 results equal decode-then-``paged_attention`` bit for
bit. NVFP4 does too except for rare single outputs (1 in 20480 at head_dim 128,
none at 8 or 65), because its E4M3 scale is one byte where the BF16 key tile is
two: Triton fills a 16-byte vector from the scale load and gives the tile
``sizePerThread`` [1, 16] instead of the reference kernel's [1, 8], so the FP32
score dot product associates as a different tree. The decoded values themselves
are identical; only that reduction order differs. ``tl.max_contiguous`` on the
scale index does not change the choice, and an explicit two-level sum in the
reference's tree did not either, so this is left as a documented caveat rather
than paid for with a layout-forcing workaround.
"""

import torch
import triton
import triton.language as tl

from vllm_rlt.kernels.kv_quantization.triton_decode import _e2m1, _pool_args
from vllm_rlt.kernels.triton_attention import NUM_SPLITS, _paged_attention_merge_kernel


@triton.jit
def _mul_rn(a, b, EXACT: tl.constexpr):
    # PTX never contracts mul.rn into an FMA, so a later add rounds separately,
    # as in triton_decode (compiled with enable_fp_fusion=False).
    if EXACT:
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )
    return a * b


@triton.jit
def _packed_tile(
    Payload,
    First,
    Second,
    Global,
    block,
    offsets,
    dims,
    mask,
    kv_head,
    layer,
    pool_block,
    pool_layer,
    param_block,
    param_layer,
    global_block,
    global_layer,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    IS_KEY: tl.constexpr,
    IS_NVFP4: tl.constexpr,
    EXACT: tl.constexpr,
):
    """Decode one [tokens, dims] tile of a packed page into FP32."""
    packed_dim = (DIM + 1) // 2
    payload = Payload + block[:, None] * pool_block + layer * pool_layer
    byte = tl.load(
        payload + (offsets[:, None] * HEADS + kv_head) * packed_dim + dims[None, :] // 2,
        mask=mask,
        other=0,
    )
    code = ((byte >> ((dims[None, :] % 2) * 4).to(tl.uint8)) & 0x0F).to(tl.int32)
    if IS_KEY:
        param = ((offsets[:, None] // GROUP) * HEADS + kv_head) * DIM + dims[None, :]
    else:
        groups = (DIM + GROUP - 1) // GROUP
        param = (offsets[:, None] * HEADS + kv_head) * groups + dims[None, :] // GROUP
    base = block[:, None] * param_block + layer * param_layer
    if IS_NVFP4:
        scale = tl.load(First + base + param, mask=mask, other=0.0).to(tl.float32)
        # One global scale per page and layer: load it once per token.
        tokens = tl.max(mask.to(tl.int32), axis=1) > 0
        global_scale = tl.load(
            Global + block * global_block + layer * global_layer, mask=tokens, other=1.0
        )
        ratio = tl.div_rn(scale, global_scale[:, None])
        value = _e2m1(code) * ratio
    else:
        minimum = tl.load(First + base + param, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(Second + base + param, mask=mask, other=0.0).to(tl.float32)
        value = _mul_rn(code.to(tl.float32), scale, EXACT) + minimum
    return value


@triton.jit
def _packed_attention_kernel(
    Q,
    K,
    V,
    KPayload,
    KFirst,
    KSecond,
    KGlobal,
    VPayload,
    VFirst,
    VSecond,
    VGlobal,
    TABLES,
    SOURCES,
    LENGTHS,
    PART_M,
    PART_L,
    PART_O,
    layer,
    q_batch_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    q_dim_stride: tl.constexpr,
    k_block_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    v_block_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    table_stride: tl.constexpr,
    kp_block: tl.constexpr,
    kp_layer: tl.constexpr,
    kq_block: tl.constexpr,
    kq_layer: tl.constexpr,
    kg_block: tl.constexpr,
    kg_layer: tl.constexpr,
    vp_block: tl.constexpr,
    vp_layer: tl.constexpr,
    vq_block: tl.constexpr,
    vq_layer: tl.constexpr,
    vg_block: tl.constexpr,
    vg_layer: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GROUP: tl.constexpr,
    IS_NVFP4: tl.constexpr,
    EXACT: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = head // HEAD_GROUPS
    length = tl.load(LENGTHS + row)
    split_tokens = ((length + SPLITS - 1) // SPLITS + BLOCK_T - 1) // BLOCK_T * BLOCK_T
    begin = split * split_tokens
    end = tl.minimum(length, begin + split_tokens)
    dims = tl.arange(0, BLOCK_D)
    valid_dims = dims < HEAD_DIM
    query = tl.load(
        Q + row * q_batch_stride + head * q_head_stride + dims * q_dim_stride,
        mask=valid_dims,
        other=0,
    ).to(tl.float32)
    maximum = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start in range(begin, end, BLOCK_T):
        positions = start + tl.arange(0, BLOCK_T)
        valid_tokens = positions < end
        mask = valid_tokens[:, None] & valid_dims[None, :]
        offsets = positions % PAGE_SIZE
        pages = tl.load(
            TABLES + row * table_stride + positions // PAGE_SIZE, mask=valid_tokens, other=0
        )
        sources = tl.load(SOURCES + pages, mask=valid_tokens, other=0).to(tl.int64)
        staged = mask & (sources[:, None] >= 0)
        packed = mask & (sources[:, None] < 0)
        slot = tl.maximum(sources, 0)
        block = tl.maximum(-sources - 1, 0)
        keys = tl.load(
            K
            + slot[:, None] * k_block_stride
            + offsets[:, None] * k_token_stride
            + kv_head * k_head_stride
            + dims[None, :],
            mask=staged,
            other=0,
        )
        values = tl.load(
            V
            + slot[:, None] * v_block_stride
            + offsets[:, None] * v_token_stride
            + kv_head * v_head_stride
            + dims[None, :],
            mask=staged,
            other=0,
        )
        if tl.max(tl.max(packed.to(tl.int32), axis=1), axis=0) > 0:
            decoded_keys = _packed_tile(
                KPayload, KFirst, KSecond, KGlobal, block, offsets, dims, packed, kv_head,
                layer, kp_block, kp_layer, kq_block, kq_layer, kg_block, kg_layer,
                KV_HEADS, HEAD_DIM, GROUP, True, IS_NVFP4, EXACT,
            )
            decoded_values = _packed_tile(
                VPayload, VFirst, VSecond, VGlobal, block, offsets, dims, packed, kv_head,
                layer, vp_block, vp_layer, vq_block, vq_layer, vg_block, vg_layer,
                KV_HEADS, HEAD_DIM, GROUP, False, IS_NVFP4, EXACT,
            )
            # Round like the decoded BF16 pages the reference attends over.
            keys = tl.where(packed, decoded_keys.to(keys.dtype), keys)
            values = tl.where(packed, decoded_values.to(values.dtype), values)
        keys = keys.to(tl.float32)
        values = values.to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid_tokens, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        correction = tl.exp(maximum - next_maximum)
        weights = tl.exp(scores - next_maximum)
        accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
        normalizer = normalizer * correction + tl.sum(weights, axis=0)
        maximum = next_maximum
    part = (row * NUM_HEADS + head) * SPLITS + split
    tl.store(PART_M + part, maximum)
    tl.store(PART_L + part, normalizer)
    tl.store(PART_O + part * BLOCK_D + dims, accumulator)


def packed_paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_pool: dict[str, torch.Tensor],
    value_pool: dict[str, torch.Tensor],
    layer: int,
    block_tables: torch.Tensor,
    sources: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    format: str,
    group_size: int,
) -> torch.Tensor:
    """Decode attention over staged BF16 and packed pages of one layer.

    ``key_cache``/``value_cache`` are the staging pages of this layer,
    ``[slots, block_size, kv_heads, dim]`` with unit dim stride. ``block_tables``
    index ``sources`` (see the module docstring).
    """
    output = torch.empty_like(q)
    if q.shape[0] == 0:
        return output
    rows, heads = q.shape[0], q.shape[1]
    _, page_size, kv_heads, dim = key_cache.shape
    if key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError("staging caches need a unit dim stride")
    block_d = triton.next_power_of_2(dim)
    part_m = torch.empty((rows, heads, NUM_SPLITS), device=q.device, dtype=torch.float32)
    part_l = torch.empty_like(part_m)
    part_o = torch.empty((rows, heads, NUM_SPLITS, block_d), device=q.device, dtype=torch.float32)
    k_tensors, k_strides = _pool_args(key_pool, format)
    v_tensors, v_strides = _pool_args(value_pool, format)
    _packed_attention_kernel[(rows, heads, NUM_SPLITS)](
        q,
        key_cache,
        value_cache,
        *k_tensors,
        *v_tensors,
        block_tables,
        sources,
        context_lengths,
        part_m,
        part_l,
        part_o,
        layer,
        *q.stride(),
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        block_tables.stride(0),
        *k_strides,
        *v_strides,
        HEAD_DIM=dim,
        HEAD_GROUPS=heads // kv_heads,
        NUM_HEADS=heads,
        KV_HEADS=kv_heads,
        PAGE_SIZE=page_size,
        GROUP=group_size,
        IS_NVFP4=format == "nvfp4",
        EXACT=torch.version.hip is None,
        SCALE=dim**-0.5,
        BLOCK_D=block_d,
        BLOCK_T=32,
        SPLITS=NUM_SPLITS,
        num_warps=4,
    )
    _paged_attention_merge_kernel[(rows, heads)](
        part_m,
        part_l,
        part_o,
        output,
        *output.stride(),
        HEAD_DIM=dim,
        NUM_HEADS=heads,
        BLOCK_D=block_d,
        SPLITS=NUM_SPLITS,
        num_warps=4,
    )
    return output
