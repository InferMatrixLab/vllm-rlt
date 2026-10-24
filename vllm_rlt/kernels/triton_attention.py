"""Paged GQA kernel using bounded-memory online softmax.

One Triton program computes one query head. Unlike a fixed-depth batch, every
query row receives its own physical block table. This is a correctness-first
kernel without a performance claim or autotuning. FP8 e4m3 pools are read with
one float32 scale per (block, token, KV head) vector.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    Q,
    K,
    V,
    TABLES,
    LENGTHS,
    OUT,
    KS,
    VS,
    q_batch_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    q_dim_stride: tl.constexpr,
    k_block_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    k_dim_stride: tl.constexpr,
    v_block_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    v_dim_stride: tl.constexpr,
    table_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    out_head_stride: tl.constexpr,
    out_dim_stride: tl.constexpr,
    s_block_stride: tl.constexpr,
    s_token_stride: tl.constexpr,
    s_head_stride: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // HEAD_GROUPS
    length = tl.load(LENGTHS + row)
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        Q + row * q_batch_stride + head * q_head_stride + dims * q_dim_stride,
        mask=dims < HEAD_DIM,
        other=0,
    ).to(tl.float32)
    maximum = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start in range(0, length, BLOCK_T):
        positions = start + tl.arange(0, BLOCK_T)
        valid_tokens = positions < length
        blocks = tl.load(
            TABLES + row * table_stride + positions // PAGE_SIZE,
            mask=valid_tokens,
            other=0,
        ).to(tl.int64)
        # Physical IDs fit in int32, but ID * block_stride can exceed 2**31
        # elements in a large KV pool. Promote BEFORE multiplying, for K and V.
        offsets = positions % PAGE_SIZE
        kv_mask = valid_tokens[:, None] & (dims[None, :] < HEAD_DIM)
        key_pointers = (
            K
            + blocks[:, None] * k_block_stride
            + offsets[:, None] * k_token_stride
            + kv_head * k_head_stride
            + dims[None, :] * k_dim_stride
        )
        if HAS_SCALE:
            # An integer fill value cannot be cast to fp8: zero masked lanes after widening.
            scale_offsets = (
                blocks * s_block_stride + offsets * s_token_stride + kv_head * s_head_stride
            )
            keys = tl.where(kv_mask, tl.load(key_pointers, mask=kv_mask).to(tl.float32), 0.0)
            keys = keys * tl.load(KS + scale_offsets, mask=valid_tokens, other=0.0)[:, None]
        else:
            keys = tl.load(key_pointers, mask=kv_mask, other=0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid_tokens, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        correction = tl.exp(maximum - next_maximum)
        weights = tl.exp(scores - next_maximum)
        value_pointers = (
            V
            + blocks[:, None] * v_block_stride
            + offsets[:, None] * v_token_stride
            + kv_head * v_head_stride
            + dims[None, :] * v_dim_stride
        )
        if HAS_SCALE:
            values = tl.where(kv_mask, tl.load(value_pointers, mask=kv_mask).to(tl.float32), 0.0)
            values = values * tl.load(VS + scale_offsets, mask=valid_tokens, other=0.0)[:, None]
        else:
            values = tl.load(value_pointers, mask=kv_mask, other=0).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
        normalizer = normalizer * correction + tl.sum(weights, axis=0)
        maximum = next_maximum
    tl.store(
        OUT + row * out_batch_stride + head * out_head_stride + dims * out_dim_stride,
        accumulator / tl.maximum(normalizer, 1.0e-30),
        mask=dims < HEAD_DIM,
    )


def paged_attention(
    q, key_cache, value_cache, block_tables, context_lengths, key_scale=None, value_scale=None
):
    """Launch over [batch row, query head]; inputs are validated by the manager."""
    output = torch.empty_like(q)
    if q.shape[0] == 0:
        return output
    has_scale = key_scale is not None
    # Unused pointer arguments when the pool is not quantized.
    key_scale = key_scale if has_scale else context_lengths
    value_scale = value_scale if has_scale else context_lengths
    scale_strides = key_scale.stride() if has_scale else (0, 0, 0)
    _paged_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        output,
        key_scale,
        value_scale,
        *q.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        block_tables.stride(0),
        *output.stride(),
        *scale_strides,
        HAS_SCALE=has_scale,
        HEAD_DIM=q.shape[-1],
        HEAD_GROUPS=q.shape[1] // key_cache.shape[2],
        PAGE_SIZE=key_cache.shape[1],
        SCALE=q.shape[-1] ** -0.5,
        BLOCK_D=triton.next_power_of_2(q.shape[-1]),
        BLOCK_T=32,
        num_warps=4,
    )
    return output


@triton.jit
def _fp8_store_kernel(
    K,
    V,
    K_CODES,
    V_CODES,
    K_SCALES,
    V_SCALES,
    BLOCKS,
    OFFSETS,
    k_row_stride,
    k_head_stride,
    v_row_stride,
    v_head_stride,
    c_block_stride,
    c_token_stride,
    c_head_stride,
    s_block_stride,
    s_token_stride,
    s_head_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
    INV_FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    valid = dims < HEAD_DIM
    block = tl.load(BLOCKS + row).to(tl.int64)
    offset = tl.load(OFFSETS + row).to(tl.int64)
    codes_at = block * c_block_stride + offset * c_token_stride + head * c_head_stride + dims
    scale_at = block * s_block_stride + offset * s_token_stride + head * s_head_stride
    # The same float32 operations as kv_cache_manager.quantize_fp8, in the same order.
    k = tl.load(K + row * k_row_stride + head * k_head_stride + dims, mask=valid, other=0.0)
    k = k.to(tl.float32)
    k_scale = tl.maximum(tl.max(tl.abs(k), axis=0) * INV_FP8_MAX, TINY)
    k_codes = tl.minimum(tl.maximum(tl.div_rn(k, k_scale), -FP8_MAX), FP8_MAX)
    tl.store(K_CODES + codes_at, k_codes.to(K_CODES.dtype.element_ty), mask=valid)
    tl.store(K_SCALES + scale_at, k_scale)
    v = tl.load(V + row * v_row_stride + head * v_head_stride + dims, mask=valid, other=0.0)
    v = v.to(tl.float32)
    v_scale = tl.maximum(tl.max(tl.abs(v), axis=0) * INV_FP8_MAX, TINY)
    v_codes = tl.minimum(tl.maximum(tl.div_rn(v, v_scale), -FP8_MAX), FP8_MAX)
    tl.store(V_CODES + codes_at, v_codes.to(V_CODES.dtype.element_ty), mask=valid)
    tl.store(V_SCALES + scale_at, v_scale)


def store_fp8(k, v, key_codes, value_codes, key_scale, value_scale, blocks, offsets, fp8_max):
    """Quantize rows k, v [row, KV head, dim] and scatter codes and scales in one launch.

    Row i goes to (blocks[i], offsets[i]) of one layer's code pools
    [physical block, token, KV head, dim] and scale pools [physical block, token, KV head].
    """
    if k.shape[0] == 0:
        return
    k = k if k.stride(-1) == 1 else k.contiguous()
    v = v if v.stride(-1) == 1 else v.contiguous()
    if key_codes.stride() != value_codes.stride() or key_scale.stride() != value_scale.stride():
        raise ValueError("FP8 key and value pools must share a layout")
    _fp8_store_kernel[(k.shape[0], k.shape[1])](
        k,
        v,
        key_codes,
        value_codes,
        key_scale,
        value_scale,
        blocks,
        offsets,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        *key_codes.stride()[:3],
        *key_scale.stride(),
        HEAD_DIM=k.shape[-1],
        BLOCK_D=triton.next_power_of_2(k.shape[-1]),
        FP8_MAX=fp8_max,
        INV_FP8_MAX=1.0 / fp8_max,
        TINY=torch.finfo(torch.float32).tiny,
        num_warps=1,
    )
