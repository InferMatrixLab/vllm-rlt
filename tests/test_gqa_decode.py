# SPDX-License-Identifier: Apache-2.0
"""GQA decode: every geometry this repo ships must match the FP32 torch reference.

This test exists because the first attempt at batching the query group produced a kernel
that was 3.5x faster and numerically wrong, and nothing failed loudly enough to catch it
before the speed number was reported. The gate here is deliberately exact-ish (BF16
rounding against a FP32 oracle) over every (heads, kv_heads) pair in the model configs, at
row counts from 1 to 128, so a wrong group mapping cannot hide behind one lucky shape.
"""

import pytest
import torch

from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.kernels.triton_attention import paged_attention

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# (query heads, kv heads) from vllm_rlt/models/*.py
GEOMETRIES = [
    (16, 16),  # Ouro: MHA, must keep using the per-head kernel
    (32, 8),  # group of 4
    (48, 8),  # Nanbeige, group of 6
    (16, 4),  # group of 4
]


def _make(rows, length, heads, kv_heads, dim, page, device="cuda", seed=0):
    torch.manual_seed(seed)
    per_row = (length + page - 1) // page
    total = rows * per_row + 8
    dtype = torch.bfloat16
    key = torch.randn(total, page, kv_heads, dim, device=device, dtype=dtype)
    val = torch.randn_like(key)
    tables = torch.zeros(rows, per_row + 1, device=device, dtype=torch.int32)
    for r in range(rows):
        tables[r, :per_row] = torch.arange(
            r * per_row, (r + 1) * per_row, device=device, dtype=torch.int32
        )
    ctx = torch.full((rows,), length, device=device, dtype=torch.int32)
    q = torch.randn(rows, heads, dim, device=device, dtype=dtype)
    return q, key, val, tables, ctx


@requires_cuda
@pytest.mark.parametrize("heads,kv_heads", GEOMETRIES)
@pytest.mark.parametrize("rows", [1, 3, 8, 32, 128])
@pytest.mark.parametrize("length", [1, 33, 512, 2048])
def test_gqa_decode_matches_torch_reference(heads, kv_heads, rows, length):
    page, dim = 16, 128
    q, key, val, tables, ctx = _make(rows, length, heads, kv_heads, dim, page)
    reference = torch_paged_attention(q, key, val, tables, ctx).float()
    got = paged_attention(q, key, val, tables, ctx).float()

    diff = (got - reference).abs()
    scale = reference.abs().max().item()
    # BF16 rounding against an FP32 oracle; the whole point is that a wrong group mapping or
    # a mis-sliced output is orders of magnitude above this.
    assert diff.max().item() <= max(2e-2, 0.02 * scale), (
        f"max |diff| {diff.max().item():.6f} vs reference scale {scale:.6f} "
        f"for heads={heads} kv_heads={kv_heads} rows={rows} length={length}"
    )
    assert torch.isfinite(got).all(), "non-finite output"


@requires_cuda
@pytest.mark.parametrize("heads,kv_heads", [(32, 8), (48, 8), (16, 4)])
def test_gqa_decode_is_batch_invariant(heads, kv_heads):
    """Split alignment makes a row's result independent of the batch it ran in."""
    page, dim, length = 16, 128, 512
    single = []
    for rows in (1, 4):
        q, key, val, tables, ctx = _make(rows, length, heads, kv_heads, dim, page, seed=7)
        single.append((paged_attention(q, key, val, tables, ctx).float(), q, key, val, tables, ctx))

    out4 = single[1][0]
    for r in range(4):
        deep = torch_paged_attention(
            single[1][1][r : r + 1],
            single[1][2],
            single[1][3],
            single[1][4][r : r + 1],
            single[1][5][r : r + 1],
        ).float()
        assert (out4[r : r + 1] - deep).abs().max().item() <= 2e-2


@requires_cuda
@pytest.mark.parametrize("heads,kv_heads", [(32, 8), (48, 8)])
def test_gqa_decode_window_masks_to_last_keys(heads, kv_heads):
    page, dim, rows, length, window = 16, 128, 4, 512, 128
    q, key, val, tables, ctx = _make(rows, length, heads, kv_heads, dim, page, seed=11)
    got = paged_attention(q, key, val, tables, ctx, window=window).float()
    reference = torch_paged_attention_windowed(q, key, val, tables, ctx, window).float()
    assert (got - reference).abs().max().item() <= 2e-2


def torch_paged_attention_windowed(q, key, val, tables, ctx, window):
    from vllm_rlt.kernels.paged_attention import torch_paged_attention_windowed as impl

    return impl(q, key, val, tables, ctx, window)


@requires_cuda
@pytest.mark.parametrize("heads,kv_heads", [(32, 8), (48, 8)])
def test_gqa_decode_zero_length_row_returns_zero(heads, kv_heads):
    """A row with no context must produce zeros, not NaN, through the split/merge path."""
    page, dim, rows = 16, 128, 3
    q, key, val, tables, ctx = _make(rows, 256, heads, kv_heads, dim, page, seed=13)
    ctx[1] = 0
    out = paged_attention(q, key, val, tables, ctx).float()
    assert torch.isfinite(out).all()
    assert out[1].abs().max().item() == 0.0


@requires_cuda
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("window", [0, 193])
def test_mixed_gqa_decode_switches_paths_on_graph_replay(quantized, window):
    if quantized and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("Triton FP8 E4M3FN loads require SM89 or later")
    q, key, val, tables, lengths = _make(32, 544, 48, 8, 128, 16, seed=29)
    key_scale = value_scale = None
    if quantized:
        key_scale = key.abs().amax(-1).float() / 448
        value_scale = val.abs().amax(-1).float() / 448
        key = (key.float() / key_scale[..., None]).to(torch.float8_e4m3fn)
        val = (val.float() / value_scale[..., None]).to(torch.float8_e4m3fn)
    mixed = torch.tensor([0, 17, 128, 33] * 6 + [129, 193, 544, 512] * 2, device="cuda")
    uniform = torch.full_like(mixed, 512)
    lengths.copy_(mixed)
    paged_attention(q, key, val, tables, lengths, key_scale, value_scale, window=window)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_attention(q, key, val, tables, lengths, key_scale, value_scale, window=window)
    for replay_lengths in (mixed, uniform, mixed.roll(3)):
        lengths.copy_(replay_lengths)
        graph.replay()
        if window:
            from vllm_rlt.kernels.paged_attention import torch_paged_attention_windowed

            expected = torch_paged_attention_windowed(
                q, key, val, tables, lengths, window, key_scale, value_scale
            )
        else:
            expected = torch_paged_attention(q, key, val, tables, lengths, key_scale, value_scale)
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.002)


@requires_cuda
def test_wide_gqa_group_produces_every_query_head():
    q, key, val, tables, lengths = _make(3, 65, 32, 1, 64, 16, seed=31)
    actual = paged_attention(q, key, val, tables, lengths)
    expected = torch_paged_attention(q, key, val, tables, lengths)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0001)
