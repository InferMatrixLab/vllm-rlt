# SPDX-License-Identifier: Apache-2.0
"""Ordinary fixed-loop SWA reclamation, including shared final-depth readers."""

import math

import pytest
import torch

from tests.reference.dory import dense_dory_reference
from tests.test_dory import tiny_dory_config
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
            reclaim_swa=True,
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
        cache_config=CacheConfig(
            num_blocks=9, block_size=1, incremental_allocation=True, reclaim_swa=True
        ),
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
            reclaim_swa=True,
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
    for step in range(128):
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
    # Device snapshots stay on-device until the whole cancel/reuse scenario ends.
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
    print(
        f"{mode}: steps={step + 1}, live_cross_request_reused_pages="
        f"{sorted(reused_while_b_active)}, stale_rejected={stale_checked}, "
        f"logits_checked={len(observed)}, cleanup=0"
    )
