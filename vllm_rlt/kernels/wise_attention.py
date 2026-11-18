"""Split-KV paged decode attention with WISE working-set support (arXiv 2609.27373).

Two launches per layer call:

* ``_wise_split_kernel`` over [row, query head, split]: each split attends a
  contiguous share of the row's 32-token tiles (one WISE block per tile) and
  writes a partial online-softmax state. Per row mode:

  - ``MODE_DENSE``: every tile;
  - ``MODE_DISCOVER`` / ``MODE_DISCOVER_UNION``: every tile, also recording each
    tile's (max, sum exp);
  - ``MODE_SPARSE``: only tiles whose support bit is set, compacted first so the
    attention loop is branch-free. Splits divide the selected tiles, so a sparse
    row's work, not just its bytes, shrinks with density.

* ``_wise_finish_kernel`` over [row, query head]: merges the partial states and,
  for discovery rows, rebuilds per-block mass and selects the smallest stable
  descending-mass prefix reaching ``eta`` (the authors' direct_block_support rule
  for one query row). The first discovery depth overwrites the support bitmask,
  later ones OR into it.

Splitting keeps one long dense row from setting the latency of a launch whose
other rows are sparse. With every support bit set a sparse row uses the same
tiles, splits and arithmetic as MODE_DENSE, so its output is bitwise identical.
"""

import math

import torch
import triton
import triton.language as tl

MODE_DENSE, MODE_DISCOVER, MODE_SPARSE, MODE_DISCOVER_UNION = 0, 1, 2, 3
WISE_BLOCK = 32
MAX_SPLITS = 32


def num_splits(rows, heads, device, max_tiles):
    """Splits per [row, head]: about eight 32-token tiles per program for the longest
    row, and at least four programs per SM. A dense program scanning thousands of
    tokens is latency-bound and sets the time of a launch whose other rows are sparse."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    wanted = max(math.ceil(max_tiles / 8), math.ceil(4 * sms / max(1, rows * heads)), 1)
    return min(MAX_SPLITS, triton.next_power_of_2(wanted))


@triton.jit
def _wise_split_kernel(
    Q,
    K,
    V,
    TABLES,
    LENGTHS,
    MODES,
    SLOTS,
    SUPPORT,
    MASS_M,
    MASS_S,
    SELECTED,
    PART_M,
    PART_L,
    PART_ACC,
    layer,
    q_batch_stride,
    q_head_stride,
    k_block_stride,
    k_token_stride,
    k_head_stride,
    v_block_stride,
    v_token_stride,
    v_head_stride,
    table_stride,
    support_slot_stride,
    support_layer_stride,
    support_head_stride,
    mass_row_stride,
    mass_head_stride,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = head // HEAD_GROUPS
    length = tl.load(LENGTHS + row)
    mode = tl.load(MODES + row)
    mass = row.to(tl.int64) * mass_row_stride + head * mass_head_stride
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        Q + row * q_batch_stride + head * q_head_stride + dims, mask=dims < HEAD_DIM, other=0
    ).to(tl.float32)
    count = (length + BLOCK_T - 1) // BLOCK_T
    total = count
    if mode == 2:
        slot = tl.load(SLOTS + row).to(tl.int64)
        support = (
            SUPPORT
            + slot * support_slot_stride
            + layer * support_layer_stride
            + head * support_head_stride
        )
        index = tl.arange(0, MAX_BLOCKS)
        bits = tl.load(support + index, mask=index < count, other=0) != 0
        target = tl.cumsum(bits.to(tl.int32), axis=0) - 1
        # Every split writes the same list for its [row, head], then reads it back.
        tl.store(SELECTED + mass + target, index, mask=bits)
        total = tl.sum(bits.to(tl.int32), axis=0)
        tl.debug_barrier()
    share = (total + SPLITS - 1) // SPLITS
    first = split * share
    last = tl.minimum(first + share, total)
    maximum = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    sparse = mode == 2
    record = (mode == 1) | (mode == 3)
    for item in range(first, last):
        # Masked loads/stores instead of branches keep the loop pipelinable.
        tile = tl.where(sparse, tl.load(SELECTED + mass + item, mask=sparse, other=0), item)
        positions = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        valid_tokens = positions < length
        blocks = tl.load(
            TABLES + row * table_stride + positions // PAGE_SIZE, mask=valid_tokens, other=0
        ).to(tl.int64)
        offsets = positions % PAGE_SIZE
        keys = tl.load(
            K
            + blocks[:, None] * k_block_stride
            + offsets[:, None] * k_token_stride
            + kv_head * k_head_stride
            + dims[None, :],
            mask=valid_tokens[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid_tokens, scores, -float("inf"))
        tile_maximum = tl.max(scores, axis=0)
        next_maximum = tl.maximum(maximum, tile_maximum)
        correction = tl.exp(maximum - next_maximum)
        weights = tl.exp(scores - next_maximum)
        tl.store(MASS_M + mass + tile, tile_maximum, mask=record)
        tl.store(MASS_S + mass + tile, tl.sum(tl.exp(scores - tile_maximum), axis=0), mask=record)
        values = tl.load(
            V
            + blocks[:, None] * v_block_stride
            + offsets[:, None] * v_token_stride
            + kv_head * v_head_stride
            + dims[None, :],
            mask=valid_tokens[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
        normalizer = normalizer * correction + tl.sum(weights, axis=0)
        maximum = next_maximum
    part = (row.to(tl.int64) * HEADS + head) * SPLITS + split
    tl.store(PART_M + part, maximum)
    tl.store(PART_L + part, normalizer)
    tl.store(PART_ACC + part * BLOCK_D + dims, accumulator)


@triton.jit
def _wise_finish_kernel(
    LENGTHS,
    MODES,
    SLOTS,
    SUPPORT,
    MASS_M,
    MASS_S,
    STATS,
    PART_M,
    PART_L,
    PART_ACC,
    OUT,
    layer,
    out_batch_stride,
    out_head_stride,
    support_slot_stride,
    support_layer_stride,
    support_head_stride,
    mass_row_stride,
    mass_head_stride,
    HEAD_DIM: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    SPLITS: tl.constexpr,
    ETA: tl.constexpr,
    COMBINE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    mode = tl.load(MODES + row)
    if COMBINE:
        dims = tl.arange(0, BLOCK_D)
        splits = tl.arange(0, SPLITS)
        part = (row.to(tl.int64) * HEADS + head) * SPLITS
        maxima = tl.load(PART_M + part + splits)
        normalizers = tl.load(PART_L + part + splits)
        accumulators = tl.load(PART_ACC + (part + splits)[:, None] * BLOCK_D + dims[None, :])
        maximum = tl.max(maxima, axis=0)
        scale = tl.where(maxima == -float("inf"), 0.0, tl.exp(maxima - maximum))
        normalizer = tl.sum(normalizers * scale, axis=0)
        accumulator = tl.sum(accumulators * scale[:, None], axis=0)
        tl.store(
            OUT + row * out_batch_stride + head * out_head_stride + dims,
            accumulator / tl.maximum(normalizer, 1.0e-30),
            mask=dims < HEAD_DIM,
        )
    if (mode == 1) | (mode == 3):
        length = tl.load(LENGTHS + row)
        slot = tl.load(SLOTS + row).to(tl.int64)
        support = (
            SUPPORT
            + slot * support_slot_stride
            + layer * support_layer_stride
            + head * support_head_stride
        )
        mass = row.to(tl.int64) * mass_row_stride + head * mass_head_stride
        index = tl.arange(0, MAX_BLOCKS)
        count = (length + BLOCK_T - 1) // BLOCK_T
        valid = index < count
        tile_m = tl.load(MASS_M + mass + index, mask=valid, other=-float("inf"))
        tile_s = tl.load(MASS_S + mass + index, mask=valid, other=0.0)
        weight = tile_s * tl.exp(tile_m - tl.max(tile_m, axis=0))
        weight = tl.where(valid, weight, 0.0)
        probability = weight / tl.sum(weight, axis=0)
        # Stable descending sort: nonnegative float bits are monotone as int32;
        # the low half breaks ties toward the lower block index. Blocks past
        # the row sort last (negative keys) but keep their index in the low
        # half, so the store below also clears their stale bits.
        bits = probability.to(tl.int32, bitcast=True).to(tl.int64)
        low = (MAX_BLOCKS - 1 - index).to(tl.int64)
        key = tl.where(valid, (bits << 32) | low, low - (1 << 62))
        ranked = tl.sort(key, descending=True)
        order = MAX_BLOCKS - 1 - (ranked & 0xFFFFFFFF)
        ranked_mass = tl.where(
            ranked >= 0, ((ranked >> 32).to(tl.int32)).to(tl.float32, bitcast=True), 0.0
        )
        total = tl.sum(ranked_mass, axis=0)
        below = tl.sum((tl.cumsum(ranked_mass, axis=0) < ETA * total).to(tl.int32), axis=0)
        keep_count = tl.maximum(tl.minimum(below + 1, count), 1)
        chosen = (index < keep_count).to(tl.uint8)
        # STATS counts the support the reuse depths attend: the first discovery
        # adds its selection and causal blocks, a union adds only the blocks it sets.
        if mode == 3:
            previous = tl.load(support + order)
            chosen = chosen | previous
            tl.atomic_add(STATS, tl.sum((chosen - previous).to(tl.int64), axis=0))
        else:
            tl.atomic_add(STATS, keep_count.to(tl.int64))
            tl.atomic_add(STATS + 1, count.to(tl.int64))
        tl.store(support + order, chosen)


def wise_scratch(rows, heads, head_dim, max_blocks, device):
    """Per-traversal scratch: tile mass (also the compacted tile lists) and split partials."""
    width = triton.next_power_of_2(head_dim)
    return {
        "mass": torch.empty((2, rows, heads, max_blocks), device=device, dtype=torch.float32),
        "part_m": torch.empty((rows, heads, MAX_SPLITS), device=device, dtype=torch.float32),
        "part_l": torch.empty((rows, heads, MAX_SPLITS), device=device, dtype=torch.float32),
        "part_acc": torch.empty(
            (rows, heads, MAX_SPLITS, width), device=device, dtype=torch.float32
        ),
    }


def select_from_mass(
    context_lengths, modes, slots, support, layer, mass_m, mass_s, stats, eta, heads
):
    """Discovery selection only (FlashAttention path): masses are already recorded."""
    rows = context_lengths.shape[0]
    if rows == 0:
        return
    _wise_finish_kernel[(rows, heads)](
        context_lengths,
        modes,
        slots,
        support,
        mass_m,
        mass_s,
        stats,
        mass_m,
        mass_m,
        mass_m,
        mass_m,
        layer,
        0,
        0,
        support.stride(0),
        support.stride(1),
        support.stride(2),
        mass_m.stride(0),
        mass_m.stride(1),
        HEAD_DIM=1,
        HEADS=heads,
        BLOCK_D=1,
        BLOCK_T=WISE_BLOCK,
        MAX_BLOCKS=support.shape[-1],
        SPLITS=1,
        ETA=eta,
        COMBINE=False,
        num_warps=4,
    )


def wise_paged_attention(
    q,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    modes,
    slots,
    support,
    layer,
    scratch,
    stats,
    eta=0.95,
    splits=None,
    num_warps=1,
    num_stages=None,
):
    """Split-KV WISE decode over [row, query head].

    One warp per split program measured fastest on GH200 (4K x 20 dense: 207 us at
    1 warp, 312 us at 4, FlashAttention-3 211 us).

    ``support`` is [slots, layers, query heads, MAX_BLOCKS] uint8, MAX_BLOCKS a power
    of two covering every row. ``scratch`` comes from ``wise_scratch``; ``stats`` is
    int64 [2] accumulating (selected, causal) block counts over discovery rows.
    """
    output = torch.empty_like(q)
    rows, heads = q.shape[:2]
    if rows == 0:
        return output
    splits = splits or num_splits(rows, heads, q.device, support.shape[-1])
    mass = scratch["mass"]
    block_d = triton.next_power_of_2(q.shape[-1])
    _wise_split_kernel[(rows, heads, splits)](
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        modes,
        slots,
        support,
        mass[0],
        mass[1],
        mass[1].view(torch.int32),
        scratch["part_m"],
        scratch["part_l"],
        scratch["part_acc"],
        layer,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_tables.stride(0),
        support.stride(0),
        support.stride(1),
        support.stride(2),
        mass.stride(1),
        mass.stride(2),
        HEAD_DIM=q.shape[-1],
        HEAD_GROUPS=heads // key_cache.shape[2],
        HEADS=heads,
        PAGE_SIZE=key_cache.shape[1],
        SCALE=q.shape[-1] ** -0.5,
        BLOCK_D=block_d,
        BLOCK_T=WISE_BLOCK,
        MAX_BLOCKS=support.shape[-1],
        SPLITS=splits,
        num_warps=num_warps,
        **({"num_stages": num_stages} if num_stages else {}),
    )
    _wise_finish_kernel[(rows, heads)](
        context_lengths,
        modes,
        slots,
        support,
        mass[0],
        mass[1],
        stats,
        scratch["part_m"],
        scratch["part_l"],
        scratch["part_acc"],
        output,
        layer,
        output.stride(0),
        output.stride(1),
        support.stride(0),
        support.stride(1),
        support.stride(2),
        mass.stride(1),
        mass.stride(2),
        HEAD_DIM=q.shape[-1],
        HEADS=heads,
        BLOCK_D=block_d,
        BLOCK_T=WISE_BLOCK,
        MAX_BLOCKS=support.shape[-1],
        SPLITS=splits,
        ETA=eta,
        COMBINE=True,
        num_warps=4,
    )
    return output


def block_mass(mass_m, mass_s, lengths):
    """Reference per-block attention probability mass [rows, heads, blocks] from tile records."""
    blocks = torch.arange(mass_m.shape[-1], device=mass_m.device)
    valid = blocks[None, :] * WISE_BLOCK < lengths[:, None]
    m = mass_m.masked_fill(~valid[:, None, :], -float("inf"))
    s = mass_s.masked_fill(~valid[:, None, :], 0.0)
    weights = s * torch.exp(m - m.amax(-1, keepdim=True))
    return weights / weights.sum(-1, keepdim=True), valid


def select_support(mass, valid, eta):
    """Reference rule: smallest stable descending-mass prefix reaching ``eta`` of the causal mass.

    Same rule as the pinned author ``direct_block_support`` for one query row.
    """
    ranked, order = torch.sort(
        mass.masked_fill(~valid[:, None, :], -float("inf")), descending=True, stable=True
    )
    total = mass.sum(-1, keepdim=True)
    keep = ranked.clamp_min(0).cumsum(-1).lt(eta * total).sum(-1) + 1
    keep = torch.minimum(keep, valid.sum(-1, keepdim=True).expand_as(keep)).clamp_min(1)
    chosen = torch.arange(mass.shape[-1], device=mass.device) < keep.unsqueeze(-1)
    return torch.zeros_like(chosen).scatter_(-1, order, chosen) & valid[:, None, :]
