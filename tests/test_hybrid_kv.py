# SPDX-License-Identifier: Apache-2.0
"""Native Dory storage: numerical dataflow, ownership and physical-page budgets."""

import pytest
import torch

from tests.reference import dory as reference
from tests.test_dory import (
    assert_matches_reference,
    run_all_positions_at_once,
    run_one_token_per_step,
    tiny_dory_config,
)
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.config import ExecutionConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import plan_cache
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import DoryForCausalLM
from vllm_rlt.worker.metadata import GroupedKVMetadata, MetadataBuilder


def native_cache(model, blocks=48):
    config = model.config
    return KVCacheManager(
        num_layers=config.num_kv_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=blocks,
        block_size=2,
        max_loops=config.total_ut_steps,
        layout=config.kv_cache_layout,
        recurrent_layers=model.recurrent_kv_layers,
        layer_windows=model.kv_layer_windows,
    )


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_native_dory_matches_unchanged_dense_reference_and_kv(monkeypatch, mode):
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=mode)).eval()
    tokens = torch.tensor([4, 7, 9, 3, 12, 5])
    histories = []
    original = reference._History

    class ObservedHistory(original):
        def __init__(self):
            super().__init__()
            histories.append(self)

    # Observe the existing oracle's dense K/V without changing its operations.
    monkeypatch.setattr(reference, "_History", ObservedHistory)
    expected = reference.dense_dory_reference(model, tokens)
    cache = native_cache(model)
    actual, cache = run_one_token_per_step(model, tokens, cache=cache)
    assert_matches_reference(actual, expected)
    for (physical_layer, plane), entries in histories[0].entries.items():
        layer = model.config.kv_layer_index(physical_layer)
        keys, values = cache.read(layer, "request", plane)
        torch.testing.assert_close(
            keys, torch.stack([entries[p][0] for p in range(6)]), atol=3e-6, rtol=3e-5
        )
        torch.testing.assert_close(
            values, torch.stack([entries[p][1] for p in range(6)]), atol=3e-6, rtol=3e-5
        )
    slots = 6 if mode == "per_loop" else 4
    assert cache.allocated_blocks("request") == cache.num_used_blocks == 3 * slots
    tables = []
    for layer in model.kv_layer_windows:
        depths = range(2) if layer in model.recurrent_kv_layers and mode == "per_loop" else (0,)
        for depth in depths:
            tables.extend(cache.layer_block_table("request", layer, depth))
    assert len(set(tables)) == 3 * slots
    assert set(cache.layer_groups) == {0, 1, 2, 3}
    assert model.config.attention_layer_ids == (0, 2, 4, 6)
    assert len(cache.groups) == 3  # SWA boundary, FA recurrent, SWA recurrent.
    with pytest.raises(ValueError, match="no attention KV"):
        model.config.kv_layer_index(1)
    with pytest.raises(ValueError, match="layer must be"):
        cache.read(4, "request", 0)
    with pytest.raises(ValueError, match="depth zero"):
        cache.read(0, "request", 1)
    if mode == "last_loop":
        assert cache.layer_block_table("request", 2, 0) == cache.layer_block_table("request", 2, 1)
    assert cache.token_written("request", 5, 1)
    cache.free("request")
    assert cache.num_used_blocks == 0


def test_native_per_loop_batched_queries_and_sliced_metadata():
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config()).eval()
    tokens = torch.tensor([4, 7, 9, 3, 12, 5])
    cache = native_cache(model)
    boundary_id, boundary = cache.layer_group(0)
    assert cache.layer_group(3) == (boundary_id, boundary)
    assert boundary.layers == (0, 3)
    assert (boundary.window, boundary.recurrent, boundary.planes) == (4, False, 1)
    swa_id, swa = cache.layer_group(2)
    assert swa_id != boundary_id
    assert (swa.window, swa.recurrent, swa.planes) == (4, True, 2)
    assert model.config.kv_layer_index(4) == 2
    assert cache.layer_group(1)[1].window is None
    actual, cache = run_all_positions_at_once(model, tokens, cache=cache)
    assert_matches_reference(actual, reference.dense_dory_reference(model, tokens))
    batch = cache._prepare_batch(["request"] * 2, [0, 0], [4, 5], padded_rows=3)
    assert isinstance(batch, GroupedKVMetadata)
    assert not hasattr(batch, "write_blocks")
    view = cache.layer_view(batch, 1)
    assert cache.layer_view(batch, 1) is view
    assert view.write_blocks.tolist() == [cache.layer_block_table("request", 1, 0)[2]] * 2
    assert view.write_offsets.tolist() == [0, 1]
    assert view.context_lengths.tolist() == [5, 6, 0]
    assert view.position_ids.tolist() == [4, 5, 0]
    windowed = cache.layer_view(batch, 0)
    table = cache.layer_block_table("request", 0, 0)
    assert windowed.context_lengths.tolist() == [5, 4, 0]
    assert windowed.position_ids.tolist() == [4, 5, 0]
    assert windowed.block_tables.tolist() == [list(table[:3]), [*table[1:3], -1], [0, 0, 0]]
    assert windowed.write_blocks.tolist() == [table[2], table[2]]
    assert windowed.write_offsets.tolist() == [0, 1]
    sliced = MetadataBuilder.slice_rows(batch, 1, 2)
    assert not sliced.layer_views and len(sliced.rows) == 1
    assert cache.layer_view(sliced, 2).write_blocks.tolist() == [
        cache.layer_block_table("request", 2, 0)[2]
    ]
    assert len(MetadataBuilder.without_padding(batch).position_ids) == 2
    with pytest.raises(NotImplementedError, match="ordinary layer views"):
        cache.get_block_table("request", 0)
    cache.free("request")
    assert cache.allocate("request", 6)
    with pytest.raises(RuntimeError, match="stale prepared"):
        cache.layer_view(batch, 1)


def test_native_atomic_growth_failure_free_and_early_exit_copy():
    model = DoryForCausalLM(tiny_dory_config()).eval()
    cache = native_cache(model, blocks=18)
    assert cache.required_blocks(6) == 18
    assert cache.allocate("a", 6, initial_tokens=2)
    assert cache.allocate("b", 6, initial_tokens=2)
    assert cache.num_used_blocks == 12
    before = tuple(cache.layer_block_table("a", layer, 0) for layer in model.kv_layer_windows)
    assert not cache.ensure_capacity("a", 6)
    assert cache.num_used_blocks == 12
    assert before == tuple(
        cache.layer_block_table("a", layer, 0) for layer in model.kv_layer_windows
    )
    assert cache.ensure_capacity("a", 4)
    assert cache.num_used_blocks == 18
    assert not cache.allocate("c", 6, initial_tokens=2)
    assert not cache.has_allocation("c")
    cache.free("b")
    assert cache.ensure_capacity("a", 6)
    for layer in model.kv_layer_windows:
        values = torch.full((6, 2, 8), float(layer + 1))
        cache.write(layer, ["a"] * 6, [0] * 6, list(range(6)), values, values + 10)
    for position in range(6):
        cache.finalize_token("a", position, 0)
    for layer in model.recurrent_kv_layers:
        k0, v0 = cache.read(layer, "a", 0)
        k1, v1 = cache.read(layer, "a", 1)
        torch.testing.assert_close(k0, k1, atol=0, rtol=0)
        torch.testing.assert_close(v0, v1, atol=0, rtol=0)
    assert cache.token_written("a", 5, 1)
    cache.free("a")
    assert cache.num_used_blocks == 0 and cache.num_free_blocks == 18
    assert cache.allocate("c", 6)
    assert not cache.token_written("c", 0, 0)
    cache.free("c")
    assert cache.num_free_blocks == 18


def test_native_planner_preserves_configured_all_layer_byte_budget():
    model = DoryForCausalLM(tiny_dory_config()).eval()
    scheduler, execution = SchedulerConfig(), ExecutionConfig()
    native = native_cache(model)
    page_bytes = 2 * 2 * 2 * 8 * 4
    assert native.bytes_per_block == page_bytes
    blocks, plan = plan_cache(
        model, CacheConfig(num_blocks=3, block_size=2), scheduler, execution, "torch"
    )
    assert model.config.num_kv_layers == 4 < model.config.num_hidden_layers
    assert blocks == 3 * model.config.num_kv_layers
    assert blocks * plan["bytes_per_block"] == 3 * 4 * page_bytes
    assert plan["block_unit"] == "physical_layer_page" and plan["configured_num_blocks"] == 3
    blocks, plan = plan_cache(
        model,
        CacheConfig(kv_cache_memory_bytes=24 * page_bytes, block_size=2),
        scheduler,
        execution,
        "torch",
    )
    assert blocks == 24 and plan["bytes_per_block"] == page_bytes
    blocks, _ = plan_cache(model, CacheConfig(block_size=2), scheduler, execution, "torch")
    assert blocks == 256 * 4


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_native_engine_generation_uses_grouped_storage(mode):
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=mode)).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=8, block_size=2, layout=model.config.kv_cache_layout),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
    )
    cache = engine.cache_manager
    assert cache.grouped and cache.num_blocks == 8 * 4
    assert cache.key_cache.shape[1] == 1
    prompt = [4, 7, 3]
    params = SamplingParams(max_tokens=3, min_loops=2, max_loops=2, ignore_eos=True)
    engine.add_request("request", prompt, params)
    result = None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                result = output
    assert result is not None
    tokens = list(prompt)
    for _ in range(3):
        logits = reference.dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(logits[-1].argmax()))
    assert result.token_ids == tokens[len(prompt) :]
    assert cache.num_used_blocks == 0


def test_native_engine_refuses_preemption_before_running():
    model = DoryForCausalLM(tiny_dory_config()).eval()
    with pytest.raises(NotImplementedError, match="preemption snapshots"):
        LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=8, block_size=2),
            scheduler_config=SchedulerConfig(enable_preemption=True),
        )
