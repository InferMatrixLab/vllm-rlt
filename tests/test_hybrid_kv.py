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


@pytest.mark.parametrize("loops", [1, 2, 4])
@pytest.mark.parametrize("tail_pages", [0, 1, 3])
def test_grouped_prefix_import_preserves_page_order_and_shared_ownership(loops, tail_pages):
    cached_pages, block_size = 64, 2
    recurrent = (2, 3, 4, 5)
    # Every group has multiple layers; recurrent groups also have multiple
    # planes. Distinct data per layer/plane detects any flattening mismatch.
    windows = {layer: (4 if layer % 2 == 0 else None) for layer in range(8)}
    pages_per_block = 4 + 4 * loops
    pool_pages = max(cached_pages + 1, cached_pages + 2 * tail_pages)
    cache = KVCacheManager(
        num_layers=8,
        num_kv_heads=1,
        head_dim=2,
        num_blocks=pool_pages * pages_per_block,
        block_size=block_size,
        max_loops=loops,
        recurrent_layers=recurrent,
        layer_windows=windows,
        enable_prefix_caching=True,
    )
    tokens = list(range(cached_pages * block_size + 1))
    assert cache.allocate("source", len(tokens))
    expected_tables = {}
    for layer in windows:
        for depth in range(loops) if layer in recurrent else (0,):
            values = torch.full((len(tokens), 1, 2), float(layer * loops + depth))
            cache.write(
                layer, ["source"] * len(tokens), [depth] * len(tokens), tokens, values, values + 100
            )
            expected_tables[layer, depth] = cache.layer_block_table("source", layer, depth)[
                :cached_pages
            ]
    cache.publish_prefix("source", tokens, len(tokens))
    prefix = cache.lookup_prefix(tokens)
    assert len(prefix) == cached_pages
    cache.free("source")

    capacity = (cached_pages + tail_pages) * block_size
    fresh_pages = set()
    for request_id in ("hit", "sharer"):
        assert cache.allocate(request_id, capacity, prefix=prefix)
        for (layer, depth), expected in expected_tables.items():
            table = cache.layer_block_table(request_id, layer, depth)
            assert table[:cached_pages] == expected
            assert len(table) == cached_pages + tail_pages
            suffix = set(table[cached_pages:])
            assert not suffix & fresh_pages
            fresh_pages.update(suffix)
            keys, values = cache.read(layer, request_id, depth)
            expected_keys = torch.full_like(keys, float(layer * loops + depth))
            torch.testing.assert_close(keys, expected_keys)
            torch.testing.assert_close(values, expected_keys + 100)
    assert cache.num_used_blocks == (cached_pages + 2 * tail_pages) * pages_per_block
    if tail_pages:
        # A failed third import must roll back its shared-page references.
        assert not cache.allocate("blocked", capacity, prefix=prefix)
        assert not cache.has_allocation("blocked")
    cache.free("hit")
    cache.free("sharer")
    assert cache.num_used_blocks == 0
    # Evict every cached page to prove no import left a reference behind.
    assert cache.allocate("evict", pool_pages * block_size)
    cache.free("evict")
    assert cache.num_free_blocks == cache.num_blocks


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
    windows = model.kv_layer_windows
    blocks, plan = plan_cache(
        model,
        CacheConfig(num_blocks=3, block_size=2),
        scheduler,
        execution,
        "torch",
        layer_windows=windows,
        reclaim_swa=False,
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
        layer_windows=windows,
        reclaim_swa=False,
    )
    assert blocks == 24 and plan["bytes_per_block"] == page_bytes
    blocks, _ = plan_cache(
        model,
        CacheConfig(block_size=2),
        scheduler,
        execution,
        "torch",
        layer_windows=windows,
        reclaim_swa=False,
    )
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


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
@pytest.mark.parametrize("loops", [1, 2, 4])
def test_grouped_static_buffers_with_three_padded_rows_match_dense_reference(mode, loops):
    torch.manual_seed(42)
    model = DoryForCausalLM(
        tiny_dory_config(recurrent_kv_cache_mode=mode, n_recurrent_loops=loops)
    ).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=2, layout=model.kv_cache_layout),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=3),
        execution_config=ExecutionConfig(static_buffers=True, pad_to_power_of_two=True),
    )
    prompts = {"a": [4], "b": [7], "c": [3]}
    try:
        cache = engine.cache_manager
        assert cache.grouped
        assert engine.model_runner.requestspaces
        for request_id, prompt in prompts.items():
            engine.add_request(
                request_id,
                prompt,
                SamplingParams(max_tokens=3, min_loops=loops, max_loops=loops, ignore_eos=True),
            )
        completed = {}
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    completed[output.request_id] = output.token_ids
        assert set(completed) == set(prompts)
        for request_id, prompt in prompts.items():
            tokens = list(prompt)
            for _ in range(3):
                logits = reference.dense_dory_reference(model, torch.tensor(tokens))["logits"]
                tokens.append(int(logits[-1].argmax()))
            assert completed[request_id] == tokens[len(prompt) :]
        assert cache.num_used_blocks == 0
    finally:
        engine.close()


def test_preemption_selects_all_layer_kv_and_restores_a_prefix_hit():
    model = DoryForCausalLM(tiny_dory_config()).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=2, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(enable_preemption=True),
    )
    cache = engine.cache_manager
    greedy = SamplingParams(max_tokens=6, temperature=0, ignore_eos=True)
    prompt = [3, 1, 4, 1, 5, 9, 2, 6, 5]

    def finish():
        outputs = {}
        while engine.has_unfinished_requests():
            for output in engine.step():
                outputs[output.request_id] = output.token_ids
        return outputs

    try:
        assert not cache.grouped
        engine.add_request("cold", prompt, greedy)
        cold = finish()["cold"]
        engine.add_request("hit", prompt, greedy)
        hit = engine.scheduler.requests["hit"]
        while len(hit.generated_token_ids) < 2:
            engine.step()
        assert cache.prefix_hits > 0
        engine.add_request("other", [7], SamplingParams(max_tokens=1))
        assert engine.scheduler.preempt(engine.scheduler.requests["other"])
        assert engine.preemption.snapshots["hit"]["kv"] is not None
        assert finish()["hit"] == cold
        assert engine.preemption.resumptions == 1
        assert cache.num_used_blocks == 0
    finally:
        engine.close()


def test_non_ouro_exit_keeps_grouped_storage_without_reclamation():
    from vllm_rlt.config import ExitConfig

    model = DoryForCausalLM(tiny_dory_config()).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=2),
        exit_config=ExitConfig(mode="ouro_delayed"),
    )
    try:
        assert engine.cache_manager.grouped and not engine.cache_manager.reclaim_swa
    finally:
        engine.close()
