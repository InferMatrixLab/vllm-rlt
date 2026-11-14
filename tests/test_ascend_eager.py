"""Causal/GQA semantics for the separately selected BF16 eager contract."""

import pytest
import torch

from vllm_rlt.kernels.ascend_eager import AscendEagerAttention, _flat_kv_rows, _prefill_geometry


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_eager_groups_match_independent_causal_queries(kv_heads):
    torch.manual_seed(730)
    keys = torch.randn(8, 16, kv_heads, 64, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    q = torch.randn(7, 4, 64, dtype=torch.bfloat16)
    lengths = torch.tensor([0, 1, 15, 16, 17, 3, 31], dtype=torch.int32)
    tables = torch.tensor([[4, 1], [4, 1], [4, 1], [4, 1], [4, 1], [7, 2], [7, 2]])
    actual = AscendEagerAttention()(q, keys, values, tables, lengths)
    expected = torch.zeros_like(q)
    for row, length in enumerate(lengths.tolist()):
        if not length:
            continue
        pos = torch.arange(length)
        k = keys[tables[row, pos // 16], pos % 16].repeat_interleave(4 // kv_heads, dim=1)
        v = values[tables[row, pos // 16], pos % 16].repeat_interleave(4 // kv_heads, dim=1)
        scores = torch.einsum("hd,thd->ht", q[row], k) * (64**-0.5)
        probs = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        expected[row] = torch.einsum("ht,thd->hd", probs, v)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0]) == 0


@pytest.mark.npu
@torch.inference_mode()
def test_pooled_kv_reads_match_layer_views_across_pages_and_depths():
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.device import resolve_device

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(99)
    cache = KVCacheManager(
        3, 2, 64, 16, 16, 2, device=device, dtype=torch.bfloat16, backend="ascend_eager"
    )
    assert cache.allocate("a", 17)
    assert cache.allocate("b", 3)
    batch = cache._prepare_batch(["a"] * 17 + ["b"] * 3, [0] * 17 + [1] * 3, [*range(17), 0, 1, 2])
    for layer in range(3):
        k = torch.randn(20, 2, 64, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cache._write_prepared(layer, batch, k, v)

    q = torch.randn(20, 4, 64, device=device, dtype=torch.bfloat16)
    for layer in range(3):
        expected = cache.attention(
            q,
            cache.key_cache[:, layer],
            cache.value_cache[:, layer],
            batch.block_tables,
            batch.context_lengths,
        )
        actual = cache._attend_prepared(layer, batch, q)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.npu
@pytest.mark.parametrize("chunk_size", [4096, 128])
@torch.inference_mode()
def test_engine_matches_the_same_manual_prefill_chunks(chunk_size):
    from tests.helpers import tiny_ouro_config
    from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.device import resolve_device
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.models import OuroForCausalLM

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(41)
    model = OuroForCausalLM(
        tiny_ouro_config(
            hidden_size=256, intermediate_size=512, head_dim=64, max_position_embeddings=256
        )
    ).to(device=device, dtype=torch.bfloat16)
    c = model.config
    tokens = ([3, 7, 2, 11, 5, 17, 8] * 22)[:149]
    cache = KVCacheManager(
        c.num_hidden_layers,
        c.num_key_value_heads,
        c.head_dim,
        128,
        16,
        4,
        device=device,
        dtype=torch.bfloat16,
        backend="ascend_eager",
    )
    assert cache.allocate("manual", len(tokens) + 3)
    expected, manual_states = [], []
    handle = model.model.norm.register_forward_hook(
        lambda _m, _i, output: manual_states.append(output.detach().cpu().clone())
    )
    try:
        for start in range(0, len(tokens), chunk_size):
            end = min(start + chunk_size, len(tokens))
            hidden = model.prelude(torch.tensor(tokens[start:end], device=device))
            for depth in range(4):
                hidden, _ = model.recurrent(
                    hidden,
                    ["manual"] * (end - start),
                    [depth] * (end - start),
                    list(range(start, end)),
                    cache,
                    compute_gate=False,
                )
        for step in range(3):
            expected.append(int(model.coda(hidden[-1:])[0].argmax()))
            if step < 2:
                pos = len(tokens) + step
                hidden = model.prelude(torch.tensor([expected[-1]], device=device))
                for depth in range(4):
                    hidden, _ = model.recurrent(
                        hidden, ["manual"], [depth], [pos], cache, compute_gate=False
                    )
    finally:
        handle.remove()
        cache.free("manual")
    actual_states = []
    handle = model.model.norm.register_forward_hook(
        lambda _m, _i, output: actual_states.append(output.detach().cpu().clone())
    )
    engine = StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(num_blocks=128),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=4096, prefill_chunk_size=chunk_size
        ),
    )
    try:
        engine.add_request(
            "engine",
            tokens,
            SamplingParams(max_tokens=3, temperature=0, min_loops=4, max_loops=4, ignore_eos=True),
        )
        result = None
        while engine.has_unfinished_requests():
            for out in engine.step():
                if out.finished:
                    result = out
        assert result is not None
        assert result.token_ids == expected
        assert result.exit_depths == [4] * 3
        assert len(actual_states) == len(manual_states)
        for actual, reference in zip(actual_states, manual_states, strict=True):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        handle.remove()
        engine.close()


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@torch.inference_mode()
def test_bf16_probability_rounding_is_part_of_the_eager_contract(device_name):
    from vllm_rlt.device import resolve_device
    from vllm_rlt.kernels.paged_attention import torch_paged_attention

    device = resolve_device(device_name)
    if device_name == "npu":
        torch.npu.set_device(0)
    keys = torch.zeros(1, 16, 1, 64, dtype=torch.bfloat16, device=device)
    values = torch.zeros_like(keys)
    keys[0, 0].fill_(1.03125)
    keys[0, 1].fill_(1)
    values[0, 0].fill_(128)
    values[0, 1].fill_(-128)
    query = torch.ones(1, 1, 64, dtype=torch.bfloat16, device=device)
    table = torch.tensor([[0]], dtype=torch.int32, device=device)
    lengths = torch.tensor([2], dtype=torch.int32, device=device)
    # QK/sqrt(64) is [8.25,8]. BF16 probabilities are [.5625,.4375],
    # so their weighted value is exactly16. FP32 probabilities retain
    # [.5621765,.4378235] and round the final value to15.9375 instead.
    actual = AscendEagerAttention()(query, keys, values, table, lengths)
    torch.testing.assert_close(actual, torch.full_like(actual, 16), atol=0, rtol=0)
    old_contract = torch_paged_attention(query, keys, values, table, lengths)
    assert not torch.equal(actual, old_contract)


class _PreparedBatch:
    """Only the host metadata the two batched attention paths consume."""

    def __init__(self, rows, tables, write_blocks, write_offsets):
        self.rows = rows
        self.block_tables = tables
        self.write_blocks = write_blocks
        self.write_offsets = write_offsets


def _batched_fixture(kind):
    layers, blocks, block_size, kv_heads, head_dim = 3, 24, 4, 2, 8
    head_groups, table_width = 2, 4
    dtype = torch.bfloat16  # the Ascend eager contract is BF16 end to end
    torch.manual_seed(913)
    pool_k = torch.randn(blocks, layers, block_size, kv_heads, head_dim, dtype=dtype)
    pool_v = torch.randn_like(pool_k)
    keys, values = pool_k[:, 1], pool_v[:, 1]

    first, second, third, fourth = (object() for _ in range(4))
    if kind == "prefill_grid":
        rows = [(a, 0, pos) for a in (first, second) for pos in range(4)]
    elif kind == "prefill_single":
        rows = [(first, 0, pos) for pos in range(4)]
    elif kind == "ragged_decode":
        rows = [(a, 0, 5) for a in (first, second, third, fourth)]
    elif kind == "ragged_padded_tables":
        # Mixed natural lengths: short rows have -1 in the high page slots,
        # exactly as plan_attention_metadata pads a shorter block table.
        rows = [(first, 0, 9), (second, 0, 1), (third, 0, 5), (fourth, 0, 2)]
    elif kind == "single_decode":
        rows = [(first, 0, 5)]
    else:  # irregular: a prefill group with a missing position
        rows = [(first, 0, 0), (first, 0, 2), (first, 0, 3)]

    page_tables = {
        id(first): [2, 5, 7, 9],
        id(second): [3, 6, 8, 10],
        id(third): [4, 11, 12, 13],
        id(fourth): [14, 15, 16, 17],
    }
    if kind == "ragged_padded_tables":
        page_tables = {
            id(first): [2, 5, 7, 9],
            id(second): [3, -1, -1, -1],
            id(third): [4, 11, -1, -1],
            id(fourth): [14, -1, -1, -1],
        }
    tables, write_blocks, write_offsets = [], [], []
    for allocation, _depth, position in rows:
        table = page_tables[id(allocation)]
        tables.append(table)
        write_blocks.append(table[position // block_size])
        write_offsets.append(position % block_size)
    batch = _PreparedBatch(
        rows,
        torch.tensor(tables, dtype=torch.int32),
        torch.tensor(write_blocks, dtype=torch.int64),
        torch.tensor(write_offsets, dtype=torch.int64),
    )
    q = torch.randn(len(rows), kv_heads * head_groups, head_dim, dtype=dtype)
    return q, keys, values, batch


@pytest.mark.parametrize(
    "kind,expected_prefill,expected_ragged",
    [
        ("prefill_grid", 1, 0),
        ("prefill_single", 1, 0),
        ("ragged_decode", 0, 1),
        ("ragged_padded_tables", 0, 1),
        ("single_decode", 0, 0),
        ("irregular", 0, 0),
    ],
)
@torch.inference_mode()
def test_batched_attention_paths_match_per_group_eager(kind, expected_prefill, expected_ragged):
    """Batching may remove launches but must never change the arithmetic."""
    q, keys, values, batch = _batched_fixture(kind)
    kernel = AscendEagerAttention()
    groups = kernel._plan(
        [(id(allocation), depth) for allocation, depth, _ in batch.rows],
        [position + 1 for _, _, position in batch.rows],
        q.device,
    )
    expected = kernel._forward(
        q, keys, values, batch.block_tables, groups,
        flat_rows=_flat_kv_rows(keys, values),
    )
    actual = kernel.attend_metadata(q, keys, values, batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert kernel.batched_prefill_calls == expected_prefill
    assert kernel.ragged_calls == expected_ragged


def test_prefill_geometry_accepts_one_request_but_rejects_irregular_grids():
    first, second = object(), object()
    assert _prefill_geometry([(first, 0, pos) for pos in range(4)]) == (1, 4)
    assert _prefill_geometry(
        [(first, 0, pos) for pos in range(4)] + [(second, 0, pos) for pos in range(4)]
    ) == (2, 4)
    assert _prefill_geometry([(first, 0, 0), (first, 0, 2)]) is None
    assert _prefill_geometry([(first, 0, 1)]) is None
