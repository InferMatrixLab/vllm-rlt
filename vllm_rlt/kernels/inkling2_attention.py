"""Paged GQA with Inkling 2 Mini's learned distance bias and attention sinks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention(
    Q,
    K,
    V,
    Bias,
    Sink,
    Table,
    Positions,
    Out,
    stride_page: tl.constexpr,
    stride_token: tl.constexpr,
    stride_table: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    EXTENT: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    pos = tl.load(Positions + row).to(tl.int32)
    table = Table + row * stride_table
    d = tl.arange(0, DIM)
    t = tl.arange(0, BLOCK)
    vector = (row * HEADS + head) * DIM + d
    q = tl.load(Q + vector).to(tl.float32)
    bias = Bias + (row * HEADS + head) * EXTENT
    # A zero value belongs to the sink. Start the online softmax with its logit.
    maximum = tl.load(Sink + row * HEADS + head).to(tl.float32)
    denominator = tl.full((), 1.0, tl.float32)
    output = tl.full((DIM,), 0.0, tl.float32)
    start = tl.maximum(pos - WINDOW + 1, 0) if WINDOW else 0
    for offset in range(start, pos + 1, BLOCK):
        idx = offset + t
        valid = idx <= pos
        page = tl.load(table + idx // PAGE, valid, other=-1)
        address = page[:, None].to(tl.int64) * stride_page
        address += (idx % PAGE)[:, None] * stride_token + (head // GROUP) * DIM + d[None, :]
        k = tl.load(K + address, (page >= 0)[:, None], other=0).to(tl.float32)
        v = tl.load(V + address, (page >= 0)[:, None], other=0).to(tl.float32)
        distance = pos - idx
        rel = tl.load(bias + distance, valid & (distance < EXTENT), other=0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * (DIM**-0.5) + rel
        scores = tl.where(valid & (page >= 0), scores, float("-inf"))
        new_max = tl.maximum(maximum, tl.max(scores, axis=0))
        old_scale = tl.exp(maximum - new_max)
        weights = tl.exp(scores - new_max)
        denominator = denominator * old_scale + tl.sum(weights, axis=0)
        output = output * old_scale + tl.sum(weights[:, None] * v, axis=0)
        maximum = new_max
    tl.store(Out + vector, output / denominator)


def paged_attention(q, keys, values, bias, sinks, batch, window):
    q = q.contiguous()
    rows, heads, dim = q.shape
    output = torch.empty_like(q)
    _attention[(rows, heads)](
        q,
        keys,
        values,
        bias.contiguous(),
        sinks.contiguous(),
        batch.block_tables,
        batch.position_ids,
        output,
        keys.stride(0),
        keys.stride(1),
        batch.block_tables.stride(0),
        heads,
        heads // keys.shape[2],
        dim,
        keys.shape[1],
        bias.shape[-1],
        window or 0,
        32,
    )
    return output
