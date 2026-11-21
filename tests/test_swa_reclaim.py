# SPDX-License-Identifier: Apache-2.0
"""Ordinary fixed-loop SWA reclamation, including shared final-depth readers."""

import math

import pytest
import torch

from tests.reference.dory import dense_dory_reference
from tests.test_dory import run_engine, tiny_dory_config
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import DoryForCausalLM
from vllm_rlt.request import RequestPhase


def live_pages(cache, model, request_id, *, sliding):
    pages = set()
    for layer, window in model.kv_layer_windows.items():
        if (window is not None) != sliding:
            continue
        depths = range(cache.storage_depths) if layer in model.recurrent_kv_layers else (0,)
        for depth in depths:
            pages.update(
                page for page in cache.layer_block_table(request_id, layer, depth) if page >= 0
            )
    return pages


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
@pytest.mark.parametrize("hit_tokens", [2, 8])
# A 6-token submission retires pages of the window-4 layers before it ends.
@pytest.mark.parametrize("submission", [4, 6])
def test_reclaim_continues_after_prefix_hit(monkeypatch, mode, hit_tokens, submission):
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=mode)).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=8 if submission == 4 else 16,
            block_size=2,
            layout=model.kv_cache_layout,
            incremental_allocation=True,
            enable_prefix_caching=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=submission, prefill_chunk_size=6
        ),
    )
    cache = engine.cache_manager
    publisher = [4, 7, 3, 9, 5, 2, 8, 6, 10]
    prompt = publisher[:hit_tokens] + [11, 12, 13, 14, 15, 16]
    params = SamplingParams(max_tokens=2, min_loops=2, max_loops=2, ignore_eos=True)
    run_engine(engine, [("publisher", publisher)], params)
    assert cache.prefix_hits == 0
    assert len(cache._prefixes) == 4
    assert cache.num_used_blocks == 0

    window_layer = next(layer for layer, window in model.kv_layer_windows.items() if window)
    full_layer = next(layer for layer, window in model.kv_layer_windows.items() if window is None)
    frontiers = []
    finish_submission = cache.finish_submission

    def record_reclamation(event=None):
        if not cache.has_allocation("hit"):
            return finish_submission(event)
        if not frontiers:
            # The hit starts retaining only what its next token can read.
            frontiers.append(cache.retained_range("hit", window_layer, 0)[0])
            assert frontiers[0] == (4 if hit_tokens == 8 else 0)
        finish_submission(event)
        frontiers.append(cache.retained_range("hit", window_layer, 0)[0])
        assert cache.retained_range("hit", full_layer, 0)[0] == 0

    observed = []
    logits = model.logits

    def record_logits(state):
        output = logits(state)
        observed.append(output.detach().clone())
        return output

    monkeypatch.setattr(cache, "finish_submission", record_reclamation)
    monkeypatch.setattr(model, "logits", record_logits)
    actual = run_engine(engine, [("hit", prompt)], params)["hit"]
    assert cache.prefix_hits == hit_tokens
    assert frontiers[-1] > frontiers[0]
    assert cache.num_used_blocks == 0
    assert len(observed) == len(actual) == 2
    tokens = list(prompt)
    for output, token in zip(observed, actual):
        expected = dense_dory_reference(model, torch.tensor(tokens))["logits"][-1:]
        torch.testing.assert_close(output, expected, atol=3e-6, rtol=3e-5)
        tokens.append(token)


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_reclaim_engine_crosses_chunks_and_matches_dense_logits(monkeypatch, mode):
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=mode)).eval()
    observed = []
    logits = model.logits

    def observe(state):
        result = logits(state)
        observed.append(result.detach().clone())
        return result

    monkeypatch.setattr(model, "logits", observe)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=8,
            block_size=2,
            layout=model.config.kv_cache_layout,
            incremental_allocation=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=3, prefill_chunk_size=5
        ),
    )
    cache = engine.cache_manager
    assert cache.max_submission_tokens == 3
    prompt = [4, 7, 9, 3, 12, 5, 6, 8, 11, 13, 14, 15]
    engine.add_request(
        "request", prompt, SamplingParams(max_tokens=4, min_loops=2, max_loops=2, ignore_eos=True)
    )
    request = engine.scheduler.requests["request"]
    assert engine.scheduler._plan_admission(request, 0).initial_tokens == 3
    occupancy = []
    finish = cache.finish_submission

    def record_finish(event=None):
        finish(event)
        if cache.has_allocation("request"):
            occupancy.append(
                (
                    len(live_pages(cache, model, "request", sliding=True)),
                    len(live_pages(cache, model, "request", sliding=False)),
                )
            )

    monkeypatch.setattr(cache, "finish_submission", record_finish)
    output = None
    while engine.has_unfinished_requests():
        for item in engine.step():
            if item.finished:
                output = item
    assert output is not None and len(observed) == 4
    tokens = list(prompt)
    for actual in observed:
        expected = dense_dory_reference(model, torch.tensor(tokens))["logits"][-1:]
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        tokens.append(int(expected[-1].argmax()))
    assert output.token_ids == tokens[len(prompt) :]
    peak_swa_pages = (2 + cache.storage_depths) * math.ceil((4 - 1 + 3 + 2 - 1) / 2)
    assert max(swa for swa, _ in occupancy) <= peak_swa_pages
    assert max(full for _, full in occupancy) >= 6 * cache.storage_depths
    assert any(swa < full for swa, full in occupancy)
    assert cache.num_used_blocks == 0


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_reclaim_waits_for_group_reader_reuses_pages_and_rebuilds_or_rejects_views(mode):
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=mode)).eval()
    pool_size = 20 if mode == "per_loop" else 13
    cache = KVCacheManager(
        num_layers=4,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=pool_size,
        layout=model.config.kv_cache_layout,
        block_size=2,
        max_loops=2,
        recurrent_layers=model.recurrent_kv_layers,
        layer_windows=model.kv_layer_windows,
        incremental_allocation=True,
        reclaim_swa=True,
        max_submission_tokens=1,
    )
    assert cache.allocate("request", 8)
    original_swa_pages = live_pages(cache, model, "request", sliding=True)
    old = cache._prepare_batch(["request"], [0], [0])
    cache.layer_view(old, 0)
    recent = recent_view = None
    waits = []

    class Completion:
        def __init__(self):
            self.used = cache.num_used_blocks

        def synchronize(self):
            # No references or physical mapping may change before completion.
            assert cache.num_used_blocks == self.used
            assert cache.layer_block_table("request", 0, 0)[0] >= 0
            waits.append(True)

    for position in range(7):
        assert cache.ensure_capacity("request", position + 1)
        if position == 4:
            rebuilt = cache.layer_view(recent, 0)
            assert rebuilt is not recent_view  # Growth epoch: live rows rebuild safely.
            torch.testing.assert_close(rebuilt.write_blocks, recent_view.write_blocks)
        if position == 6:
            # The tight pool must actually reuse at least one physically retired page.
            assert original_swa_pages & live_pages(cache, model, "request", sliding=True)
        boundary = cache._prepare_batch(["request"], [0], [position])
        hidden = model.prelude_prepared(torch.tensor([position + 2]), boundary, cache)
        for depth in range(2):
            core = cache._prepare_batch(["request"], [depth], [position])
            hidden, _ = model.recurrent_prepared(hidden, core, cache)
            if mode == "last_loop" and position == 4 and depth == 0:
                # q4's next loop still reads key1 from page0: first-loop completion
                # cannot advance the shared recurrent consumption frontier.
                cache.finish_submission(Completion())
                assert cache.layer_block_table("request", 2, 0)[0] >= 0
                assert cache.retained_range("request", 2, 0)[0] == 0
                assert not waits
        if position == 4:
            # Input consumed position 4, but output still needs position 1 in page 0.
            cache.finish_submission(Completion())
            assert cache.layer_block_table("request", 0, 0)[0] >= 0
            assert cache.layer_block_table("request", 2, 0)[0] == -1
        model.coda_prepared(hidden, boundary, cache)
        cache.finish_submission(Completion() if position == 4 else None)
        cache.finalize_token("request", position, 1)
        if position == 3:
            recent = cache._prepare_batch(["request"], [0], [3])
            recent_view = cache.layer_view(recent, 0)
        if position == 4:
            assert len(waits) == 2
            assert cache.retained_range("request", 0, 0) == (2, 5)
            assert cache.retained_range("request", 1, 0) == (0, 5)
            for layer in (*model.prelude_kv_layers, *model.coda_kv_layers):
                with pytest.raises(
                    ValueError, match="boundary attention layers only own depth zero"
                ):
                    cache.retained_range("request", layer, 1)
            assert cache.token_written("request", position, 1)
            with pytest.raises(RuntimeError, match="resident"):
                cache.layer_view(old, 0)
            with pytest.raises(ValueError, match="window must match the layer policy"):
                cache.attend(0, ["request"], [0], [4], torch.zeros(1, 4, 8))
            with pytest.raises(RuntimeError, match="prefix has been reclaimed"):
                cache.read(0, "request", 0)
    cache.free("request")
    assert cache.num_used_blocks == 0 and cache.num_free_blocks == pool_size


def test_window_one_page_one_finalization_and_swa_occupancy_plateau(monkeypatch):
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(sliding_window=1)).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=9, block_size=1, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=3, prefill_chunk_size=5
        ),
    )
    cache = engine.cache_manager
    occupancy = []
    finish = cache.finish_submission

    def record_finish(event=None):
        finish(event)
        if cache.has_allocation("request"):
            occupancy.append(
                (
                    len(live_pages(cache, model, "request", sliding=True)),
                    len(live_pages(cache, model, "request", sliding=False)),
                )
            )

    monkeypatch.setattr(cache, "finish_submission", record_finish)
    prompt = [4, 7, 9, 3, 12, 5, 6, 8, 11]
    engine.add_request(
        "request", prompt, SamplingParams(max_tokens=3, min_loops=2, max_loops=2, ignore_eos=True)
    )
    output = None
    while engine.has_unfinished_requests():
        for item in engine.step():
            if item.finished:
                output = item
    assert output is not None
    tokens = list(prompt)
    for _ in range(3):
        expected = dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(expected[-1].argmax()))
    assert output.token_ids == tokens[len(prompt) :]
    # SWA stays within its append bound while the full-attention history grows.
    assert max(swa for swa, _ in occupancy) <= 4 * 3
    assert max(full for _, full in occupancy) >= 2 * len(prompt)
    assert any(swa == 0 and full > 0 for swa, full in occupancy)
    assert cache.num_used_blocks == 0


@pytest.mark.parametrize("layout", ["last_exited", "shared"])
def test_long_reclaimed_history_is_not_copied_on_growth_or_retirement(monkeypatch, layout):
    history_tokens, extra_tokens, loops = 4096, 8, 2
    planes = loops if layout == "last_exited" else 1
    cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        num_blocks=history_tokens * planes,
        block_size=1,
        max_loops=loops,
        layout=layout,
        layer_windows={0: 1},
        incremental_allocation=True,
        reclaim_swa=True,
        max_submission_tokens=history_tokens,
    )
    assert cache.allocate("request", history_tokens + extra_tokens, initial_tokens=history_tokens)
    snapshots = tuple(cache.layer_block_table("request", 0, depth) for depth in range(planes))
    # Exercise real coverage and consumption bookkeeping without benchmarking attention.
    monkeypatch.setattr(cache.attention, "windowed", lambda query, *args: query)
    values = torch.zeros(history_tokens, 1, 2)
    for depth in range(loops):
        batch = cache._prepare_batch(
            ["request"] * history_tokens, [depth] * history_tokens, list(range(history_tokens))
        )
        cache._write_prepared(0, batch, values, values)
        cache._attend_prepared(0, batch, values, window=1)
    cache.finish_submission()
    assert cache.num_used_blocks == 0
    assert all(isinstance(snapshot, tuple) and min(snapshot) >= 0 for snapshot in snapshots)

    class NoHistoryCopy(list):
        # A one-page append or retirement must not traverse thousands of dead slots.
        def __iter__(self):
            raise AssertionError("allocator scanned the full historical page table")

        def __add__(self, other):
            raise AssertionError("allocator copied the full historical page table")

    allocation = cache._get_allocation("request")
    tables = tuple(NoHistoryCopy(table) for table in allocation.group_tables[0])
    allocation.group_tables = (tables,)
    assert cache.allocated_blocks("request") == 0
    epochs = tuple(allocation.group_epochs[0])
    for position in range(history_tokens, history_tokens + extra_tokens):
        assert cache.ensure_capacity("request", position + 1)
        assert cache.num_used_blocks == planes
        assert cache.allocated_blocks("request") == planes
        for depth in range(loops):
            batch = cache._prepare_batch(["request"], [depth], [position])
            cache._write_prepared(0, batch, values[:1], values[:1])
            cache._attend_prepared(0, batch, values[:1], window=1)

        class Completion:
            def synchronize(self):
                # Residency and references remain intact until the final reader completes.
                assert cache.num_used_blocks == planes
                assert all(table[position] is not None for table in tables)

        cache.finish_submission(Completion())
        assert cache.num_used_blocks == 0
        assert cache.allocated_blocks("request") == 0
        assert all(table[position] is None for table in tables)
        assert all(current is saved for current, saved in zip(allocation.group_tables[0], tables))
    assert allocation.group_epochs[0] == [epoch + 2 * extra_tokens for epoch in epochs]
    # Public inspection still returns immutable snapshots independent of internal mutation.
    allocation.group_tables = (tuple(table[:] for table in tables),)
    for depth in range(planes):
        assert cache.layer_block_table("request", 0, depth) == (-1,) * (
            history_tokens + extra_tokens
        )
        assert cache.retained_range("request", 0, depth) == (
            history_tokens + extra_tokens,
            history_tokens + extra_tokens,
        )
    cache.free("request")
    assert cache.num_free_blocks == cache.num_blocks


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_reclaim_multi_request_abort_and_reuse(monkeypatch, mode):
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=mode)).eval()
    reference = type(model)(model.config).eval()
    reference.load_state_dict(model.state_dict())
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=16 if mode == "per_loop" else 10,
            block_size=2,
            layout=model.config.kv_cache_layout,
            incremental_allocation=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=3,
            prefill_chunk_size=3,
        ),
    )
    cache = engine.cache_manager
    prompts = {
        "long": [4, 7, 9, 3, 12, 5, 6, 8, 11, 13, 14, 15],
        "reused": [8, 2, 5, 9, 3, 7, 12],
    }
    replacement = [21, 6, 17, 5, 14, 2, 19, 4]
    for rid, count in (("long", 7), ("reused", 4)):
        engine.add_request(
            rid,
            prompts[rid],
            SamplingParams(
                max_tokens=count,
                min_loops=2,
                max_loops=2,
                ignore_eos=True,
            ),
        )
    original_b = engine.scheduler.requests["reused"]
    observed, completed = [], {}
    original_logits = model.logits

    def observe(state):
        values = original_logits(state)
        scheduled = engine.last_schedule.requests
        assert len(values) == len(scheduled)
        for value, item in zip(values, scheduled):
            request = item.request
            context = tuple(request.prompt_token_ids + request.generated_token_ids)
            observed.append(
                (request, len(request.generated_token_ids), context, value.detach().clone())
            )
        return values

    monkeypatch.setattr(model, "logits", observe)
    first_b_pages, reused_while_b_active = set(), set()
    stale = None
    aborted = stale_checked = False
    for _ in range(128):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step():
            if output.finished:
                completed[output.request_id] = output
        owned = {
            rid: live_pages(cache, model, rid, sliding=True)
            | live_pages(cache, model, rid, sliding=False)
            for rid in ("long", "reused")
            if cache.has_allocation(rid)
        }
        assert sum(map(len, owned.values())) == cache.num_used_blocks
        if len(owned) == 2:
            assert owned["long"].isdisjoint(owned["reused"])
        current_b = engine.scheduler.requests.get("reused")
        if current_b is original_b and "reused" in owned:
            if not first_b_pages:
                first_b_pages = {
                    cache.layer_block_table("reused", layer, depth)[0]
                    for layer, window in model.kv_layer_windows.items()
                    if window is not None
                    for depth in (
                        range(cache.storage_depths) if layer in model.recurrent_kv_layers else (0,)
                    )
                }
                assert min(first_b_pages) >= 0
            reused_while_b_active |= first_b_pages & owned.get("long", set())
        if stale is not None and "reused" in owned and not stale_checked:
            assert current_b is not original_b
            with pytest.raises(RuntimeError, match="stale prepared KV batch"):
                cache.layer_view(stale, 0)
            stale_checked = True
        a = engine.scheduler.requests.get("long")
        if (
            not aborted
            and reused_while_b_active
            and a is not None
            and current_b is original_b
            and a.num_prefilled_tokens >= 9
            and current_b.num_prefilled_tokens >= 6
            and len(current_b.generated_token_ids) == 1
            and current_b.phase == RequestPhase.RECURRENT
            and current_b.loops_done == 0
        ):
            position = cache.retained_range("reused", 0, 0)[1] - 1
            stale = cache._prepare_batch(["reused"], [0], [position], for_write=False)
            cache.layer_view(stale, 0)  # A valid, resident descriptor before cancellation.
            engine.abort_request("reused")
            engine.add_request(
                "reused",
                replacement,
                SamplingParams(
                    max_tokens=4,
                    min_loops=2,
                    max_loops=2,
                    ignore_eos=True,
                ),
            )
            assert engine.scheduler.requests["reused"].num_prefilled_tokens == 0
            prompts["reused"] = replacement
            aborted = True
    assert not engine.has_unfinished_requests(), "small-pool admission/growth did not complete"
    assert aborted and stale_checked and reused_while_b_active
    assert set(completed) == {"long", "reused"}
    for request, index, context, actual in observed:
        assert index == len(context) - len(request.prompt_token_ids)
        expected = dense_dory_reference(reference, torch.tensor(context))["logits"][-1]
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-6, rtol=3e-5)
    assert any(request is original_b for request, _, _, _ in observed)
    assert any(
        request is not original_b and request.request_id == "reused"
        for request, _, _, _ in observed
    )
    for rid, count in (("long", 7), ("reused", 4)):
        tokens = list(prompts[rid])
        for _ in range(count):
            logits = dense_dory_reference(reference, torch.tensor(tokens))["logits"]
            tokens.append(int(logits[-1].argmax()))
        assert completed[rid].token_ids == tokens[len(prompts[rid]) :]
    assert cache.num_used_blocks == 0 and cache.num_free_blocks == cache.num_blocks


def test_fp8_kv_falls_back_to_all_layer_blocks():
    model = DoryForCausalLM(tiny_dory_config()).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=64, block_size=2, kv_cache_dtype="fp8_e4m3", enable_prefix_caching=True
        ),
    )
    assert not engine.cache_manager.grouped


def test_grouped_prefill_ignores_advertised_packing(monkeypatch):
    """A natively packing backend must not get packed metadata for grouped pages."""
    from dataclasses import replace

    from vllm_rlt import ExecutionConfig
    from vllm_rlt.attention import AttentionBackend
    from vllm_rlt.engine import stage_engine

    create_backend = stage_engine.create_backend

    def packing_backend(*args, **kwargs):
        backend = create_backend(*args, **kwargs)
        capabilities = replace(
            backend.capabilities, packed_prefill=True, packed_sliding_window=True
        )
        return AttentionBackend(backend.kernel, capabilities, backend.info)

    monkeypatch.setattr(stage_engine, "create_backend", packing_backend)
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(max_position_embeddings=128)).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=64,
            block_size=2,
            incremental_allocation=True,
        ),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=16),
        execution_config=ExecutionConfig(packed_prefill=False),
    )
    assert engine.cache_manager.grouped
    assert engine.cache_manager.attention_capabilities.packed_prefill
    assert not engine.execution_config.packed_prefill
    # Runner packing applies only to prompts longer than 64 tokens.
    prompt = [(index * 7) % 50 + 2 for index in range(65)]
    params = SamplingParams(max_tokens=2, min_loops=2, max_loops=2, ignore_eos=True)
    tokens = list(prompt)
    for _ in range(2):
        logits = dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(logits[-1].argmax()))
    assert run_engine(engine, [("long", prompt)], params)["long"] == tokens[len(prompt) :]
    assert engine.cache_manager.num_used_blocks == 0


def test_shared_fp8_prefix_hits_ignore_advertised_packing(monkeypatch):
    """last_loop + FP8 keeps shared all-layer pages, which have no packed metadata."""
    from dataclasses import replace

    from vllm_rlt.attention import AttentionBackend
    from vllm_rlt.engine import stage_engine

    create_backend = stage_engine.create_backend

    def packing_backend(*args, **kwargs):
        backend = create_backend(*args, **kwargs)
        capabilities = replace(backend.capabilities, packed_prefill=True)
        return AttentionBackend(backend.kernel, capabilities, backend.info)

    monkeypatch.setattr(stage_engine, "create_backend", packing_backend)
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode="last_loop")).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=256,
            block_size=2,
            layout="shared",
            kv_cache_dtype="fp8_e4m3",
            enable_prefix_caching=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=12, prefill_chunk_size=4
        ),
    )
    cache = engine.cache_manager
    assert cache.layout == "shared" and not cache.grouped
    assert cache.attention_capabilities.packed_prefill
    params = SamplingParams(max_tokens=3, min_loops=2, max_loops=2, ignore_eos=True)
    long = [4, 7, 3, 9, 5, 2, 8, 6, 3, 10]
    # "short" ends its prompt inside the first chunk while the 10-token prompts
    # on either side continue, so CODA rows 0 and 2 are rebuilt non-contiguously.
    requests = [("first", long), ("short", long[:4]), ("last", long[:9] + [11])]
    cold = run_engine(engine, requests, params)
    assert cache.prefix_hits == 0
    hits = run_engine(engine, [(f"hit-{rid}", prompt) for rid, prompt in requests], params)
    assert {rid: hits[f"hit-{rid}"] for rid, _ in requests} == cold
    assert cache.prefix_hits > 0
    assert cache.num_used_blocks == 0


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_preallocation_returns_expired_swa_pages(monkeypatch, mode):
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=mode)).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=16, block_size=2, layout=model.kv_cache_layout),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=3, prefill_chunk_size=3
        ),
    )
    cache = engine.cache_manager
    assert cache.grouped and cache.reclaim_swa and not cache.incremental_allocation
    prompt = [4, 7, 9, 3, 12, 5, 6, 8, 11, 13, 14, 15]
    params = SamplingParams(max_tokens=4, min_loops=2, max_loops=2, ignore_eos=True)
    frontiers = []
    finish = cache.finish_submission

    def record(event=None):
        finish(event)
        if cache.has_allocation("request"):
            allocation = cache._allocations["request"]
            frontiers.append(max(v for planes in allocation.retired_pages for v in planes))

    monkeypatch.setattr(cache, "finish_submission", record)
    reserved = []
    step = engine.step

    def record_reserved():
        outputs = step()
        reserved.append(engine.scheduler._reserved_growth_blocks(reserve_outputs=False))
        return outputs

    monkeypatch.setattr(engine, "step", record_reserved)
    tokens = list(prompt)
    for _ in range(4):
        logits = dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(logits[-1].argmax()))
    assert run_engine(engine, [("request", prompt)], params)["request"] == tokens[len(prompt) :]
    assert frontiers[0] == 0 and frontiers[-1] > 0
    # Retired pages are not budgeted again as growth.
    assert set(reserved) == {0}
    assert cache.num_used_blocks == 0


@pytest.mark.parametrize(("mode", "blocks"), [("per_loop", 17), ("last_loop", 11)])
def test_prefix_hit_admission_skips_expired_cached_pages(mode, blocks):
    # A hit attaches only in-window SWA pages; expired cached pages stay
    # evictable, so admission must not charge them.
    torch.manual_seed(42)
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=mode)).eval()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=blocks,
            block_size=2,
            layout=model.kv_cache_layout,
            enable_prefix_caching=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=4
        ),
    )
    cache = engine.cache_manager
    params = SamplingParams(max_tokens=2, min_loops=2, max_loops=2, ignore_eos=True)
    publisher = list(range(3, 23))
    run_engine(engine, [("publisher", publisher)], params)
    # Leave only prefix-only pages free: enough for the hit, not for the
    # hit plus its expired cached pages.
    entries = len(cache._prefixes)
    occupied = cache._claim(len(cache._free_blocks))
    assert len(cache._prefixes) == entries
    prompt = publisher[:16] + [30, 31]
    actual = run_engine(engine, [("hit", prompt)], params)["hit"]
    tokens = list(prompt)
    for _ in range(2):
        logits = dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(logits[-1].argmax()))
    assert actual == tokens[len(prompt) :]
    assert cache.prefix_hits == 16
    cache._drop_refs(occupied)
    assert cache.num_used_blocks == 0
