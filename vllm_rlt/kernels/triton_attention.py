"""Paged GQA kernels using bounded-memory online softmax.

Every query row receives its own physical block table. Single-query attention
splits each row's context into equal parts (split-KV): one program per
(row, head, split) produces a partial softmax state and a second kernel merges
the parts. The split count comes from the batch shape (see `_split_count`), so
it is still constant under CUDA graphs. A row's partition depends only on its
own length, so its result does not depend on the batch it runs in. Without
splitting, one long row serialized its whole context in one program and set the
decode step time.

FP8 e4m3 pools are read with one float32 scale per (block, token, KV head)
vector; both kernels dequantize on load.
"""

from functools import lru_cache, partial

import torch
import triton
import triton.language as tl
from triton.errors import TritonError
from triton.runtime import driver

_PACKED_WINDOW_TILE = (32, 32)

# Split-KV exists to fill an otherwise idle GPU on long single-row contexts. The split axis
# only helps while the other two axes underfill the GPU: measured on 108 SMs, throughput peaks
# near 512 programs, and 16 splits then costs 1.20x at rows=128 (length 2048) and 1.6x at
# length 128. The bounds are per kernel because the axes carry different work: the grouped
# kernel's second axis is one program per KV head and covers the whole query group, while the
# per-head kernel's is one program per query head. The floor matters too: the per-head kernel
# needs real splits even at rows=128 (1 vs 4 splits measured 1.37 vs 1.27 ms at length 2048),
# while the grouped kernel prefers a single part there (4 splits cost 1.11x).
_GROUPED_SPLITS = (1, 512)
_PER_HEAD_SPLITS = (4, 2048)
_MAX_SPLITS = 16


def _split_count(rows, head_axis, bounds):
    """Splits per row chosen from the batch shape alone, so a captured graph stays valid.

    Depends only on tensor shapes, not on the context lengths of individual rows, so it needs
    no host readback and no device sync.
    """
    low, target = bounds
    # tl.arange in the merge kernel needs a power of two.
    spare = max(1, target // max(1, rows * head_axis))
    return max(low, min(_MAX_SPLITS, triton.next_power_of_2(spare)))


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
    WINDOW: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = head // HEAD_GROUPS
    length = tl.load(LENGTHS + row)
    # WINDOW=0 covers the full context; otherwise only the last WINDOW keys are
    # read, which is a sliding sublayer's mask. Out-of-window keys stay stored.
    covered = length if WINDOW == 0 else tl.minimum(length, WINDOW)
    # Equal tile-aligned parts of the covered range; trailing parts may be empty.
    split_tokens = ((covered + SPLITS - 1) // SPLITS + BLOCK_T - 1) // BLOCK_T * BLOCK_T
    window_start = length - covered
    begin = window_start + split * split_tokens
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


@triton.jit
def _paged_attention_group_kernel(
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
    q_batch_stride,
    q_head_stride,
    q_dim_stride,
    k_block_stride,
    k_token_stride,
    k_head_stride,
    k_dim_stride,
    v_block_stride,
    v_token_stride,
    v_head_stride,
    v_dim_stride,
    table_stride,
    s_block_stride,
    s_token_stride,
    s_head_stride,
    HAS_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SPLITS: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PRECISION: tl.constexpr,
):
    """GQA decode built on the paged-prefill kernel's shape.

    `_paged_prefill_kernel` already solves "many query rows share one KV tile": it keeps a
    per-row online-softmax state and drives the tile through two `tl.dot`s. Decode needs the
    same thing with the query rows being the query heads of one KV group, so this kernel
    reuses that structure verbatim (same accumulator/correction/rescale order, same FP8
    dequantize-then-narrow path) and changes only what a "row" is and drops the causal mask.

    It targets the measured defect of the per-head kernel: with HEAD_GROUPS query heads per
    KV head, that kernel launched HEAD_GROUPS programs which each walked the same tile, so
    instructions (and loads) were repeated per head. Here one program owns a
    [BLOCK_H, BLOCK_T] score tile, so the KV tile is loaded once and a tensor-core dot
    serves the whole group.

    The program id names the KV head, not a block of query heads: a query-head block like
    `[head_block * BLOCK_H, ...)` does not line up with KV groups when HEAD_GROUPS does not
    divide BLOCK_H, which silently scores heads against the wrong KV head. Deriving `heads`
    from `kv_head` makes the mapping exact by construction, and every query head then has
    exactly one owning program, which is also what keeps the partial stores race-free.
    """
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    lane = tl.arange(0, BLOCK_H)
    heads = kv_head * HEAD_GROUPS + lane
    # BLOCK_H is padded up to what `tl.dot` wants, so most lanes can be outside the group.
    in_group = (lane < HEAD_GROUPS) & (heads < NUM_HEADS)
    length = tl.load(LENGTHS + row)
    # WINDOW=0 covers the full context, matching the per-head kernel's semantics.
    covered = length if WINDOW == 0 else tl.minimum(length, WINDOW)
    window_start = length - covered
    # Tile-aligned equal parts of the covered range, as the per-head kernel splits them, so a
    # row's result does not depend on the batch it ran in.
    split_tokens = ((covered + SPLITS - 1) // SPLITS + BLOCK_T - 1) // BLOCK_T * BLOCK_T
    begin = window_start + split * split_tokens
    end = tl.minimum(length, begin + split_tokens)
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_DIM
    query = tl.load(
        Q + row * q_batch_stride + heads[:, None] * q_head_stride + dims[None, :] * q_dim_stride,
        mask=in_group[:, None] & dim_valid[None, :],
        other=0.0,
    )
    maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BLOCK_H,), tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    if end > begin:
        for start in range(begin, end, BLOCK_T):
            keys_pos = start + tl.arange(0, BLOCK_T)
            key_valid = keys_pos < end
            blocks = tl.load(
                TABLES + row * table_stride + keys_pos // PAGE_SIZE,
                mask=key_valid,
                other=0,
            ).to(tl.int64)
            offsets = keys_pos % PAGE_SIZE
            kv_mask = key_valid[:, None] & dim_valid[None, :]
            key_pointers = (
                K
                + blocks[:, None] * k_block_stride
                + offsets[:, None] * k_token_stride
                + kv_head * k_head_stride
                + dims[None, :] * k_dim_stride
            )
            if HAS_SCALE:
                scale_offsets = (
                    blocks * s_block_stride + offsets * s_token_stride + kv_head * s_head_stride
                )
                keys = tl.where(kv_mask, tl.load(key_pointers, mask=kv_mask).to(tl.float32), 0.0)
                keys = keys * tl.load(KS + scale_offsets, mask=key_valid, other=0.0)[:, None]
                keys = keys.to(Q.dtype.element_ty)
            else:
                keys = tl.load(key_pointers, mask=kv_mask, other=0.0)
            scores = tl.dot(query, tl.trans(keys), input_precision=PRECISION) * SCALE
            # Decode has one query position per row, which sits at the end of the context, so
            # there is no causal mask; only padding lanes outside the group are excluded.
            visible = key_valid[None, :] & in_group[:, None]
            scores = tl.where(visible, scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            safe = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
            correction = tl.exp(maximum - safe)
            weights = tl.exp(scores - safe[:, None])
            value_pointers = (
                V
                + blocks[:, None] * v_block_stride
                + offsets[:, None] * v_token_stride
                + kv_head * v_head_stride
                + dims[None, :] * v_dim_stride
            )
            if HAS_SCALE:
                values = tl.where(
                    kv_mask, tl.load(value_pointers, mask=kv_mask).to(tl.float32), 0.0
                )
                values = values * tl.load(VS + scale_offsets, mask=key_valid, other=0.0)[:, None]
                values = values.to(Q.dtype.element_ty)
            else:
                values = tl.load(value_pointers, mask=kv_mask, other=0.0)
            rounded = weights.to(values.dtype)
            partial = tl.dot(rounded, values, input_precision=PRECISION)
            accumulator = accumulator * correction[:, None] + partial
            normalizer = normalizer * correction + tl.sum(weights, axis=1)
            maximum = next_maximum
    # The merge kernel weights the parts by their normalizers, so it needs the unnormalised
    # numerator: the per-head kernel stores `accumulator` here, and dividing first would make
    # the merge divide by the normalizer a second time.
    # An empty part (or a padding lane) stores (-inf, 0, 0), which the merge weights by zero.
    maximum = tl.where(normalizer > 0, maximum, -float("inf"))
    part = (row * NUM_HEADS + heads) * SPLITS + split
    tl.store(PART_M + part, maximum, mask=in_group)
    tl.store(PART_L + part, normalizer, mask=in_group)
    tl.store(
        PART_O + part[:, None] * BLOCK_D + dims[None, :],
        accumulator,
        mask=in_group[:, None] & dim_valid[None, :],
    )


def paged_attention(
    q,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    key_scale=None,
    value_scale=None,
    window=0,
):
    """Launch split-KV parts over [row, head, split], then merge per [row, head].

    ``window`` masks each row to its last ``window`` keys (0 keeps the full
    context); it never reclaims storage.
    """
    if type(window) is not int or window < 0:
        raise ValueError("window must be a nonnegative integer")
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
    head_groups = heads // key_cache.shape[2]
    # The grouped kernel spends one program per KV head, the per-head kernel one per head.
    splits = _split_count(
        rows,
        key_cache.shape[2] if head_groups > 1 else heads,
        _GROUPED_SPLITS if head_groups > 1 else _PER_HEAD_SPLITS,
    )
    part_m = torch.empty((rows, heads, splits), device=q.device, dtype=torch.float32)
    part_l = torch.empty_like(part_m)
    part_o = torch.empty((rows, heads, splits, block_d), device=q.device, dtype=torch.float32)
    if head_groups > 1:
        # GQA: reuse the prefill kernel's many-rows-one-tile structure with the query heads
        # of a KV group as the rows, so the tile is loaded once per group instead of once
        # per head. MHA keeps the per-head kernel, which measures faster there.
        # tl.dot needs at least 16 rows; groups of 4 or 6 pad up and mask the extra lanes.
        block_h = max(16, triton.next_power_of_2(min(head_groups, 16)))
        _paged_attention_group_kernel[(rows, key_cache.shape[2], splits)](
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
            HEAD_GROUPS=head_groups,
            NUM_HEADS=heads,
            PAGE_SIZE=key_cache.shape[1],
            SCALE=q.shape[-1] ** -0.5,
            BLOCK_D=block_d,
            BLOCK_T=32,
            SPLITS=splits,
            WINDOW=window,
            BLOCK_H=block_h,
            PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
            num_warps=4,
            num_stages=2,
        )
        _merge_parts(part_m, part_l, part_o, output, q, rows, heads, block_d, splits)
        return output
    _paged_attention_kernel[(rows, heads, splits)](
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
        SPLITS=splits,
        WINDOW=window,
        num_warps=4,
    )
    _merge_parts(part_m, part_l, part_o, output, q, rows, heads, block_d, splits)
    return output


def _merge_parts(part_m, part_l, part_o, output, q, rows, heads, block_d, splits):
    """Collapse the split-KV partials into the output rows (shared by both decode kernels)."""
    _paged_attention_merge_kernel[(rows, heads)](
        part_m,
        part_l,
        part_o,
        output,
        *output.stride(),
        HEAD_DIM=q.shape[-1],
        NUM_HEADS=heads,
        BLOCK_D=block_d,
        SPLITS=splits,
        num_warps=4,
    )


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
    WINDOW: tl.constexpr,
    COMPENSATE: tl.constexpr,
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
    first = tl.maximum(0, prefix + q_block * BLOCK_M - WINDOW + 1) if WINDOW else 0
    for start in range(first // BLOCK_N * BLOCK_N, end, BLOCK_N):
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
        if WINDOW:
            visible = visible & (keys_pos[None, :] >= positions[:, None] - WINDOW + 1)
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
        rounded = weights.to(values.dtype)
        partial = tl.dot(rounded, values, input_precision=PRECISION)
        if COMPENSATE:
            # Recover the leading bits lost when the FP32 weight is cast for MMA.
            residual = (weights - rounded.to(tl.float32)).to(values.dtype)
            partial = tl.dot(residual, values, partial, input_precision=PRECISION)
        accumulator = accumulator * correction[:, None] + partial
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
    *,
    window=0,
    compensate=False,
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
    if type(window) is not int or window < 0:
        raise ValueError("window must be a nonnegative integer")
    for tensor in (key_cache, value_cache):
        if tensor.stride(3) != 1:
            raise ValueError("paged prefill requires a contiguous head dimension")
    if q.stride(2) != 1:
        raise ValueError("paged prefill requires contiguous query rows")
    _launch_paged_prefill(
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        cu_seqlens_q,
        output,
        max_seqlen_q,
        key_scale=key_scale,
        value_scale=value_scale,
        window=window,
        compensate=compensate,
    )
    return output


def _launch_paged_prefill(
    q,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    cu_seqlens_q,
    output,
    max_seqlen_q,
    *,
    window,
    compensate,
    key_scale=None,
    value_scale=None,
    compile_only=False,
):
    block_m, block_n = _PACKED_WINDOW_TILE if compensate else (64, 64)
    q_blocks = triton.cdiv(int(max_seqlen_q), block_m)
    grid = (block_tables.shape[0] * q_blocks, q.shape[1])
    launch = (
        partial(_paged_prefill_kernel.warmup, grid=grid)
        if compile_only
        else _paged_prefill_kernel[grid]
    )
    has_scale = key_scale is not None
    # Unused pointer arguments when the pool is not quantized.
    key_scale = key_scale if has_scale else context_lengths
    value_scale = value_scale if has_scale else context_lengths
    scale_strides = key_scale.stride() if has_scale else (0, 0, 0)
    return launch(
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
        WINDOW=window,
        COMPENSATE=compensate,
        max_q_blocks=q_blocks,
        num_warps=4,
        num_stages=1 if compensate else 2,
    )


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
    FIXED_SCALE: tl.constexpr,
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
        if FIXED_SCALE is None:
            k_scale = tl.maximum(tl.max(tl.abs(k), axis=0) * INV_FP8_MAX, TINY)
        else:
            k_scale = tl.full((), FIXED_SCALE, tl.float32)
        k_codes = tl.minimum(tl.maximum(tl.div_rn(k, k_scale), -FP8_MAX), FP8_MAX)
        tl.store(K_CODES + codes_at, k_codes.to(K_CODES.dtype.element_ty), mask=valid)
        tl.store(K_SCALES + scale_at, k_scale)
        v = tl.load(V + row * v_row_stride + head * v_head_stride + dims, mask=valid, other=0.0)
        v = v.to(tl.float32)
        if FIXED_SCALE is None:
            v_scale = tl.maximum(tl.max(tl.abs(v), axis=0) * INV_FP8_MAX, TINY)
        else:
            v_scale = tl.full((), FIXED_SCALE, tl.float32)
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
    fixed_scale=None,
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
        FIXED_SCALE=fixed_scale,
        num_warps=1,
    )


def select_packed_window_tile(device, dtype, head_dim, block_size, shape):
    if torch.version.hip or dtype != torch.bfloat16:
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        return _select_packed_window_tile(index, head_dim, block_size, shape)
    except torch.OutOfMemoryError:
        # A transient failure must not be cached for other engines.
        return None


@lru_cache(maxsize=16)
def _select_packed_window_tile(index, head_dim, block_size, shape):
    # Warmup only needs tensor metadata. Preserve the KV pool's interleaved layer strides.
    q = torch.empty((256, shape.query_heads, head_dim), device="meta", dtype=torch.bfloat16)
    kv_shape = (1, shape.kv_layers, block_size, shape.kv_heads)
    kv_dtype = torch.float8_e4m3fn if shape.fp8_kv else q.dtype
    kv = q.new_empty((*kv_shape, head_dim), dtype=kv_dtype)[:, 0]
    scale = q.new_empty(kv_shape, dtype=torch.float32)[:, 0] if shape.fp8_kv else None
    table = q.new_empty((1, 16), dtype=torch.int32)
    lengths = q.new_empty(1, dtype=torch.int32)
    offsets = q.new_empty(2, dtype=torch.int32)
    windows = (0,) if shape.window is None else (0, shape.window)
    with torch.cuda.device(index):
        limit = driver.active.utils.get_device_properties(index)["max_shared_mem"]
        try:
            for window in windows:
                compiled = _launch_paged_prefill(
                    q,
                    kv,
                    kv,
                    table,
                    lengths,
                    offsets,
                    q,
                    len(q),
                    window=window,
                    compensate=True,
                    key_scale=scale,
                    value_scale=scale,
                    compile_only=True,
                )
                if compiled.metadata.shared > limit:
                    return None
        except TritonError:
            return None
    return _PACKED_WINDOW_TILE
