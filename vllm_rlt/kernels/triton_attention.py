"""Paged GQA kernels using bounded-memory online softmax.

Every query row receives its own physical block table. Single-query attention
splits each row's context into NUM_SPLITS equal parts (split-KV): one program
per (row, head, split) produces a partial softmax state and a second kernel
merges the parts. A row's partition depends only on its own length, so its
result does not depend on the batch it runs in. Without splitting, one long
row serialized its whole context in one program and set the decode step time.

FP8 e4m3 pools are read with one float32 scale per (block, token, KV head)
vector; both kernels dequantize on load.
"""

import torch
import triton
import triton.language as tl

# Splits per row for single-query attention. A power of two (tl.arange) and a
# constant, so the grid is static under CUDA graphs and independent of batch.
NUM_SPLITS = 16


@triton.jit
def _paged_attention_kernel(
    Q,
    K,
    V,
    TABLES,
    LENGTHS,
    PART_M,
    PART_L,
    PART_O,
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
    s_block_stride: tl.constexpr,
    s_token_stride: tl.constexpr,
    s_head_stride: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
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
    # Equal tile-aligned parts of this row's context; trailing parts may be empty.
    split_tokens = ((length + SPLITS - 1) // SPLITS + BLOCK_T - 1) // BLOCK_T * BLOCK_T
    begin = split * split_tokens
    end = tl.minimum(length, begin + split_tokens)
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        Q + row * q_batch_stride + head * q_head_stride + dims * q_dim_stride,
        mask=dims < HEAD_DIM,
        other=0,
    ).to(tl.float32)
    maximum = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start in range(begin, end, BLOCK_T):
        positions = start + tl.arange(0, BLOCK_T)
        valid_tokens = positions < end
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
    # An empty part stores (-inf, 0, 0), which the merge weights by zero.
    part = (row * NUM_HEADS + head) * SPLITS + split
    tl.store(PART_M + part, maximum)
    tl.store(PART_L + part, normalizer)
    tl.store(PART_O + part * BLOCK_D + dims, accumulator)


@triton.jit
def _paged_attention_merge_kernel(
    PART_M,
    PART_L,
    PART_O,
    OUT,
    out_batch_stride: tl.constexpr,
    out_head_stride: tl.constexpr,
    out_dim_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    parts = (row * NUM_HEADS + head) * SPLITS + tl.arange(0, SPLITS)
    dims = tl.arange(0, BLOCK_D)
    maxima = tl.load(PART_M + parts)
    top = tl.max(maxima, axis=0)
    # All parts empty (length 0) leaves top = -inf; zero weights give a zero row.
    scale = tl.where(maxima > -float("inf"), tl.exp(maxima - top), 0.0)
    normalizer = tl.sum(tl.load(PART_L + parts) * scale, axis=0)
    partial = tl.load(PART_O + parts[:, None] * BLOCK_D + dims[None, :])
    accumulator = tl.sum(partial * scale[:, None], axis=0)
    tl.store(
        OUT + row * out_batch_stride + head * out_head_stride + dims * out_dim_stride,
        accumulator / tl.maximum(normalizer, 1.0e-30),
        mask=dims < HEAD_DIM,
    )


def paged_attention(
    q, key_cache, value_cache, block_tables, context_lengths, key_scale=None, value_scale=None
):
    """Launch split-KV parts over [row, head, split], then merge per [row, head]."""
    output = torch.empty_like(q)
    if q.shape[0] == 0:
        return output
    has_scale = key_scale is not None
    # Unused pointer arguments when the pool is not quantized.
    key_scale = key_scale if has_scale else context_lengths
    value_scale = value_scale if has_scale else context_lengths
    scale_strides = key_scale.stride() if has_scale else (0, 0, 0)
    rows, heads = q.shape[0], q.shape[1]
    block_d = triton.next_power_of_2(q.shape[-1])
    part_m = torch.empty((rows, heads, NUM_SPLITS), device=q.device, dtype=torch.float32)
    part_l = torch.empty_like(part_m)
    part_o = torch.empty((rows, heads, NUM_SPLITS, block_d), device=q.device, dtype=torch.float32)
    _paged_attention_kernel[(rows, heads, NUM_SPLITS)](
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        part_m,
        part_l,
        part_o,
        key_scale,
        value_scale,
        *q.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        block_tables.stride(0),
        *scale_strides,
        HAS_SCALE=has_scale,
        HEAD_DIM=q.shape[-1],
        HEAD_GROUPS=heads // key_cache.shape[2],
        NUM_HEADS=heads,
        PAGE_SIZE=key_cache.shape[1],
        SCALE=q.shape[-1] ** -0.5,
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
        HEAD_DIM=q.shape[-1],
        NUM_HEADS=heads,
        BLOCK_D=block_d,
        SPLITS=NUM_SPLITS,
        num_warps=4,
    )
    return output


@triton.jit
def _paged_prefill_kernel(
    Q,
    K,
    V,
    TABLES,
    LENGTHS,
    CU_Q,
    OUT,
    KS,
    VS,
    q_token_stride,
    q_head_stride,
    k_block_stride,
    k_token_stride,
    k_head_stride,
    v_block_stride,
    v_token_stride,
    v_head_stride,
    table_stride,
    out_token_stride,
    out_head_stride,
    s_block_stride,
    s_token_stride,
    s_head_stride,
    HAS_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRECISION: tl.constexpr,
    max_q_blocks,
):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    seq = pid // max_q_blocks
    q_block = pid % max_q_blocks
    q_begin = tl.load(CU_Q + seq)
    q_len = tl.load(CU_Q + seq + 1) - q_begin
    if q_block * BLOCK_M >= q_len:
        return
    kv_len = tl.load(LENGTHS + seq)
    prefix = kv_len - q_len
    kv_head = head // HEAD_GROUPS
    rows = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_valid = rows < q_len
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_DIM
    query = tl.load(
        Q
        + (q_begin + rows)[:, None].to(tl.int64) * q_token_stride
        + head * q_head_stride
        + dims[None, :],
        mask=row_valid[:, None] & dim_valid[None, :],
        other=0.0,
    )
    positions = prefix + rows
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    # Bottom-right causal alignment: the last query row sees every key.
    end = tl.minimum(kv_len, prefix + (q_block + 1) * BLOCK_M)
    for start in range(0, end, BLOCK_N):
        keys_pos = start + tl.arange(0, BLOCK_N)
        key_valid = keys_pos < end
        blocks = tl.load(
            TABLES + seq * table_stride + keys_pos // PAGE_SIZE, mask=key_valid, other=0
        ).to(tl.int64)
        offsets = keys_pos % PAGE_SIZE
        kv_mask = key_valid[:, None] & dim_valid[None, :]
        key_pointers = (
            K
            + blocks[:, None] * k_block_stride
            + offsets[:, None] * k_token_stride
            + kv_head * k_head_stride
            + dims[None, :]
        )
        if HAS_SCALE:
            # Dequantize to the query dtype for tl.dot. Masked lanes are zeroed
            # after widening: an unfilled fp8 lane may hold a NaN code.
            scale_offsets = (
                blocks * s_block_stride + offsets * s_token_stride + kv_head * s_head_stride
            )
            keys = tl.where(kv_mask, tl.load(key_pointers, mask=kv_mask).to(tl.float32), 0.0)
            keys = keys * tl.load(KS + scale_offsets, mask=key_valid, other=0.0)[:, None]
            keys = keys.to(Q.dtype.element_ty)
        else:
            keys = tl.load(key_pointers, mask=kv_mask, other=0.0)
        scores = tl.dot(query, tl.trans(keys), input_precision=PRECISION) * SCALE
        visible = key_valid[None, :] & (keys_pos[None, :] <= positions[:, None])
        scores = tl.where(visible, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        # Rows with no visible key in this tile keep -inf; avoid inf - inf.
        safe = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp(maximum - safe)
        weights = tl.exp(scores - safe[:, None])
        value_pointers = (
            V
            + blocks[:, None] * v_block_stride
            + offsets[:, None] * v_token_stride
            + kv_head * v_head_stride
            + dims[None, :]
        )
        if HAS_SCALE:
            values = tl.where(kv_mask, tl.load(value_pointers, mask=kv_mask).to(tl.float32), 0.0)
            values = values * tl.load(VS + scale_offsets, mask=key_valid, other=0.0)[:, None]
            values = values.to(Q.dtype.element_ty)
        else:
            values = tl.load(value_pointers, mask=kv_mask, other=0.0)
        accumulator = accumulator * correction[:, None] + tl.dot(
            weights.to(values.dtype), values, input_precision=PRECISION
        )
        normalizer = normalizer * correction + tl.sum(weights, axis=1)
        maximum = next_maximum
    out = accumulator / tl.maximum(normalizer, 1.0e-30)[:, None]
    tl.store(
        OUT
        + (q_begin + rows)[:, None].to(tl.int64) * out_token_stride
        + head * out_head_stride
        + dims[None, :],
        out,
        mask=row_valid[:, None] & dim_valid[None, :],
    )


def paged_prefill_attention(
    q,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    cu_seqlens_q,
    max_seqlen_q,
    key_scale=None,
    value_scale=None,
):
    """Packed causal prefill over paged K/V, bottom-right aligned per sequence.

    q is [total query tokens, heads, dim]; sequence s owns rows
    cu_seqlens_q[s]:cu_seqlens_q[s + 1] and keys [0, context_lengths[s]) through
    block_tables[s]. Query i of a sequence sits at position
    context_length - query_length + i and attends keys up to that position.
    FP8 pools pass per-vector scales [physical block, token, KV head].
    """
    output = torch.empty_like(q)
    sequences = block_tables.shape[0]
    if q.shape[0] == 0 or sequences == 0:
        return output
    block_m, block_n = 64, 64
    q_blocks = triton.cdiv(int(max_seqlen_q), block_m)
    for tensor in (key_cache, value_cache):
        if tensor.stride(3) != 1:
            raise ValueError("paged prefill requires a contiguous head dimension")
    if q.stride(2) != 1:
        raise ValueError("paged prefill requires contiguous query rows")
    has_scale = key_scale is not None
    # Unused pointer arguments when the pool is not quantized.
    key_scale = key_scale if has_scale else context_lengths
    value_scale = value_scale if has_scale else context_lengths
    scale_strides = key_scale.stride() if has_scale else (0, 0, 0)
    _paged_prefill_kernel[(sequences * q_blocks, q.shape[1])](
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        cu_seqlens_q,
        output,
        key_scale,
        value_scale,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_tables.stride(0),
        output.stride(0),
        output.stride(1),
        *scale_strides,
        HAS_SCALE=has_scale,
        HEAD_DIM=q.shape[-1],
        HEAD_GROUPS=q.shape[1] // key_cache.shape[2],
        PAGE_SIZE=key_cache.shape[1],
        SCALE=q.shape[-1] ** -0.5,
        BLOCK_D=max(16, triton.next_power_of_2(q.shape[-1])),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        # FP32 callers (reference checks) need exact products, not TF32.
        PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
        max_q_blocks=q_blocks,
        num_warps=4,
        num_stages=2,
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
    ACTIVE,
    MASK,
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
    # Optional device-side row skipping: rows at or past *ACTIVE (bucketed graph
    # padding) and rows whose MASK entry is false write neither codes nor scales.
    enabled = row >= 0
    if ACTIVE is not None:
        enabled = enabled & (row < tl.load(ACTIVE))
    if MASK is not None:
        enabled = enabled & tl.load(MASK + row)
    if enabled:
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


def store_fp8(
    k,
    v,
    key_codes,
    value_codes,
    key_scale,
    value_scale,
    blocks,
    offsets,
    fp8_max,
    *,
    mask=None,
    active=None,
):
    """Quantize rows k, v [row, KV head, dim] and scatter codes and scales in one launch.

    Row i goes to (blocks[i], offsets[i]) of one layer's code pools
    [physical block, token, KV head, dim] and scale pools [physical block, token, KV head].
    Optional ``mask`` (bool [row]) and ``active`` (int32 scalar row count) skip rows
    on the device, as the unquantized masked and graph KV writers do.
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
        active,
        mask,
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
