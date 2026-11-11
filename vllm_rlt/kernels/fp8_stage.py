"""Stage FP8 paged KV as dense BF16/FP16 for backends without FP8 paged decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _stage_kernel(
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_ptr,
    table_ptr,
    out_key_ptr,
    out_value_ptr,
    page: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    block_tokens: tl.constexpr,
    kb: tl.constexpr,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kd: tl.constexpr,
    vb: tl.constexpr,
    vt: tl.constexpr,
    vh: tl.constexpr,
    vd: tl.constexpr,
    ksb: tl.constexpr,
    kst: tl.constexpr,
    ksh: tl.constexpr,
    vsb: tl.constexpr,
    vst: tl.constexpr,
    vsh: tl.constexpr,
):
    entry = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.load(table_ptr + entry)
    valid = block >= 0
    block = tl.maximum(block, 0)
    dims = tl.arange(0, dim)
    for start in range(0, page, block_tokens):
        tokens = start + tl.arange(0, block_tokens)
        source_k = block * kb + tokens[:, None] * kt + head * kh + dims[None, :] * kd
        source_v = block * vb + tokens[:, None] * vt + head * vh + dims[None, :] * vd
        target = (entry * page + tokens)[:, None] * heads * dim + head * dim + dims[None, :]
        scale = tl.load(key_scale_ptr + block * ksb + tokens * kst + head * ksh) * valid
        value_scale = tl.load(value_scale_ptr + block * vsb + tokens * vst + head * vsh) * valid
        key = tl.load(key_ptr + source_k).to(tl.float32) * scale[:, None]
        value = tl.load(value_ptr + source_v).to(tl.float32) * value_scale[:, None]
        tl.store(out_key_ptr + target, key.to(out_key_ptr.dtype.element_ty))
        tl.store(out_value_ptr + target, value.to(out_value_ptr.dtype.element_ty))


_BUFFERS: dict = {}


def _buffers(entries, page, heads, dim, dtype, device):
    key = (entries, page, heads, dim, dtype, str(device))
    pair = _BUFFERS.get(key)
    if pair is None:
        pair = (
            torch.empty((entries, page, heads, dim), dtype=dtype, device=device),
            torch.empty((entries, page, heads, dim), dtype=dtype, device=device),
        )
        _BUFFERS[key] = pair
    return pair


def stage_fp8_kv(key_cache, value_cache, key_scale, value_scale, table, dtype):
    """Widen ``table``'s pages into compact dense K/V with one Triton launch.

    ``table`` is the flattened list of valid physical blocks (one entry per
    row); duplicates are staged separately, so the caller's page table becomes a
    simple arange over entries. Strides are explicit because the cache is often a
    layer slice of a wider pool.
    """
    page, heads, dim = key_cache.shape[1:]
    entries = table.shape[0]
    out_key, out_value = _buffers(entries, page, heads, dim, dtype, key_cache.device)
    _stage_kernel[(entries, heads)](
        key_cache,
        value_cache,
        key_scale,
        value_scale,
        table,
        out_key,
        out_value,
        page=page,
        heads=heads,
        dim=dim,
        block_tokens=128,
        kb=key_cache.stride(0),
        kt=key_cache.stride(1),
        kh=key_cache.stride(2),
        kd=key_cache.stride(3),
        vb=value_cache.stride(0),
        vt=value_cache.stride(1),
        vh=value_cache.stride(2),
        vd=value_cache.stride(3),
        ksb=key_scale.stride(0),
        kst=key_scale.stride(1),
        ksh=key_scale.stride(2),
        vsb=value_scale.stride(0),
        vst=value_scale.stride(1),
        vsh=value_scale.stride(2),
    )
    return out_key, out_value
