"""WISE working-set decode on top of FlashAttention 2/3/4 paged kernels.

FlashAttention takes one page table per row for all heads, so sparse rows
attend the union over heads of their WISE block support, at page granularity
(a page is kept if any of its 32-token blocks is selected by any head).
``compact_page_tables`` rewrites each sparse row's page table to only those
pages, keeping order so that the partially filled newest page stays last;
dense and discovery rows keep their table. With every block selected the
compacted table and length equal the original ones, so FlashAttention's output
is unchanged bit for bit.

Discovery rows run dense FlashAttention with its log-sum-exp; ``discover_support``
then rereads K once (split across programs like the attention), rebuilds each
32-token block's probability mass from the LSE, and applies the same stable
smallest-prefix selection as the Triton kernel.
"""

import triton
import triton.language as tl

from vllm_rlt.kernels.wise_attention import WISE_BLOCK, num_splits, select_from_mass


@triton.jit
def _compact_kernel(
    TABLES,
    LENGTHS,
    MODES,
    SLOTS,
    SUPPORT,
    OUT_TABLES,
    OUT_LENGTHS,
    layer,
    table_stride,
    out_table_stride,
    support_slot_stride,
    support_layer_stride,
    support_head_stride,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCKS_PER_PAGE: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(LENGTHS + row)
    mode = tl.load(MODES + row)
    pages = tl.arange(0, MAX_PAGES)
    count = (length + PAGE_SIZE - 1) // PAGE_SIZE
    live = pages < count
    table = tl.load(TABLES + row * table_stride + pages, mask=live, other=0)
    if mode == 2:
        slot = tl.load(SLOTS + row).to(tl.int64)
        base = SUPPORT + slot * support_slot_stride + layer * support_layer_stride
        keep = tl.zeros((MAX_PAGES,), dtype=tl.int32)
        for head in range(HEADS):
            for part in range(BLOCKS_PER_PAGE):
                blocks = (pages * PAGE_SIZE) // BLOCK + part
                bits = tl.load(base + head * support_head_stride + blocks, mask=live, other=0)
                keep = keep | bits.to(tl.int32)
        keep = keep & live.to(tl.int32)
        target = tl.cumsum(keep, axis=0) - 1
        tl.store(OUT_TABLES + row * out_table_stride + target, table, mask=keep != 0)
        kept = tl.sum(keep, axis=0)
        newest = tl.sum(tl.where(pages == count - 1, keep, 0), axis=0)
        tail = length - (count - 1) * PAGE_SIZE
        tl.store(OUT_LENGTHS + row, (kept - newest) * PAGE_SIZE + newest * tail)
    else:
        tl.store(OUT_TABLES + row * out_table_stride + pages, table, mask=live)
        tl.store(OUT_LENGTHS + row, length)


def compact_page_tables(
    block_tables, context_lengths, modes, slots, support, layer, page_size, out_tables, out_lengths
):
    rows = block_tables.shape[0]
    if rows == 0:
        return out_tables[:0], out_lengths[:0]
    max_pages = triton.next_power_of_2(block_tables.shape[1])
    _compact_kernel[(rows,)](
        block_tables,
        context_lengths,
        modes,
        slots,
        support,
        out_tables,
        out_lengths,
        layer,
        block_tables.stride(0),
        out_tables.stride(0),
        support.stride(0),
        support.stride(1),
        support.stride(2),
        HEADS=support.shape[2],
        PAGE_SIZE=page_size,
        BLOCK=WISE_BLOCK,
        MAX_PAGES=max_pages,
        BLOCKS_PER_PAGE=max(1, page_size // WISE_BLOCK),
        num_warps=4,
    )
    return out_tables[:rows], out_lengths[:rows]


@triton.jit
def _flash_mass_kernel(
    Q,
    K,
    TABLES,
    LENGTHS,
    LSE,
    MODES,
    MASS_M,
    MASS_S,
    q_batch_stride,
    q_head_stride,
    k_block_stride,
    k_token_stride,
    k_head_stride,
    table_stride,
    lse_batch_stride,
    lse_head_stride,
    mass_row_stride,
    mass_head_stride,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Discovery rows: each split records exp(score - LSE) summed per tile (max 0)."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    mode = tl.load(MODES + row)
    if (mode == 1) | (mode == 3):
        kv_head = head // HEAD_GROUPS
        length = tl.load(LENGTHS + row)
        lse = tl.load(LSE + row * lse_batch_stride + head * lse_head_stride)
        mass = row.to(tl.int64) * mass_row_stride + head * mass_head_stride
        dims = tl.arange(0, BLOCK_D)
        query = tl.load(
            Q + row * q_batch_stride + head * q_head_stride + dims, mask=dims < HEAD_DIM, other=0
        ).to(tl.float32)
        count = (length + BLOCK_T - 1) // BLOCK_T
        share = (count + SPLITS - 1) // SPLITS
        for tile in range(split * share, tl.minimum(split * share + share, count)):
            positions = tile * BLOCK_T + tl.arange(0, BLOCK_T)
            valid_tokens = positions < length
            blocks = tl.load(
                TABLES + row * table_stride + positions // PAGE_SIZE, mask=valid_tokens, other=0
            ).to(tl.int64)
            keys = tl.load(
                K
                + blocks[:, None] * k_block_stride
                + (positions % PAGE_SIZE)[:, None] * k_token_stride
                + kv_head * k_head_stride
                + dims[None, :],
                mask=valid_tokens[:, None] & (dims[None, :] < HEAD_DIM),
                other=0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * SCALE
            weights = tl.where(valid_tokens, tl.exp(scores - lse), 0.0)
            tl.store(MASS_M + mass + tile, 0.0)
            tl.store(MASS_S + mass + tile, tl.sum(weights, axis=0))


def discover_support(
    q,
    key_cache,
    block_tables,
    context_lengths,
    lse,
    modes,
    slots,
    support,
    layer,
    scratch,
    stats,
    eta=0.95,
):
    """Selection for discovery rows from FlashAttention's LSE; other rows are untouched.

    ``lse`` is [rows, heads] natural-log normalizers of the dense attention just run.
    ``scratch`` comes from ``wise_scratch``; its mass planes are left holding the masses.
    """
    rows, heads = q.shape[:2]
    if rows == 0:
        return
    mass = scratch["mass"]
    splits = num_splits(rows, heads, q.device, support.shape[-1])
    _flash_mass_kernel[(rows, heads, splits)](
        q,
        key_cache,
        block_tables,
        context_lengths,
        lse,
        modes,
        mass[0],
        mass[1],
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        block_tables.stride(0),
        lse.stride(0),
        lse.stride(1),
        mass.stride(1),
        mass.stride(2),
        HEAD_DIM=q.shape[-1],
        HEAD_GROUPS=heads // key_cache.shape[2],
        PAGE_SIZE=key_cache.shape[1],
        SCALE=q.shape[-1] ** -0.5,
        BLOCK_D=triton.next_power_of_2(q.shape[-1]),
        BLOCK_T=WISE_BLOCK,
        SPLITS=splits,
        num_warps=1,
    )
    select_from_mass(
        context_lengths, modes, slots, support, layer, mass[0], mass[1], stats, eta, heads
    )
