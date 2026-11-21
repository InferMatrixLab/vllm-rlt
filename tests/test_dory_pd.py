# SPDX-License-Identifier: Apache-2.0
"""Dory same-host PD against the dense oracle and a single-stage engine.

P runs the input stage, every recurrent loop and the output stage of every
prompt token except the last. D receives that KV and the last token's carried
state, runs the last output stage and samples. Neither side repeats or skips a
stage of the prompt.
"""

import time

import pytest
import torch

from tests.helpers import RecordingConnector
from tests.reference.dory import dense_dory_reference
from tests.test_dory import make_cache, tiny_dory_config
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import DoryForCausalLM
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.request import RequestPhase

LOOPS = 2
SEED = 123
KV_MODES = ["per_loop", "last_loop"]
# Prompts longer than the four-token window cross reclaimable SWA pages.
PROMPTS = {"0": [4, 7, 3, 9, 5, 6, 2], "1": [8, 1, 6], "2": [3] * 9}
EXECUTION_MODES = {
    "sync_eager": dict(),
    "sync_graph": dict(cuda_graphs=True),
    "async_eager": dict(async_scheduling=True),
    "async_graph": dict(async_scheduling=True, cuda_graphs=True),
}


def config(kv_mode):
    return tiny_dory_config(n_recurrent_loops=LOOPS, recurrent_kv_cache_mode=kv_mode)


def params(max_tokens, **kwargs):
    kwargs.setdefault("ignore_eos", True)
    return SamplingParams(max_tokens=max_tokens, min_loops=LOOPS, max_loops=LOOPS, **kwargs)


def oracle(kv_mode):
    """The CPU model that each PD worker builds from the same seed."""
    torch.manual_seed(SEED)
    return DoryForCausalLM(config(kv_mode))


def greedy_reference(model, prompt, count):
    tokens = list(prompt)
    for _ in range(count):
        logits = dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(logits[-1].argmax()))
    return tokens[len(prompt) :]


def drain(engine):
    outputs = {}
    deadline = time.monotonic() + 90
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
        assert time.monotonic() < deadline
    return outputs


def tokens(outputs):
    return {rid: output.token_ids for rid, output in outputs.items()}


def cache(kv_mode, prefix_caching=False):
    return CacheConfig(
        num_blocks=128,
        block_size=2,
        layout=config(kv_mode).kv_cache_layout,
        enable_prefix_caching=prefix_caching,
        incremental_allocation=True,
    )


def create_pd(kv_mode, mode, *, p_cache=False, d_cache=False):
    pytest.importorskip("nixl")
    from vllm_rlt.pd.engine import PDEngine

    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 visible GPUs")
    return PDEngine(
        config(kv_mode),
        pd_config=PDConfig(
            prefill_devices=(0,),
            decode_devices=(1,),
            max_receiving_requests=3,
            max_draining_requests=2,
            transfer_chunk_bytes=4096,
            max_inflight_bytes=16384,
            request_timeout=90,
            startup_timeout=90,
        ),
        prefill_cache_config=cache(kv_mode, p_cache),
        decode_cache_config=cache(kv_mode, d_cache),
        prefill_scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=5, prefill_chunk_size=3
        ),
        decode_scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2),
        exit_config=ExitConfig("ouro"),
        execution_config=ExecutionConfig(**EXECUTION_MODES[mode]),
        attention_backend="triton",
        dtype="float32",
        seed=SEED,
    )


def local_engine(kv_mode):
    model = oracle(kv_mode).to("cuda:0")
    return LLMEngine(
        model,
        attention_backend="triton",
        cache_config=cache(kv_mode),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=5, prefill_chunk_size=3
        ),
        exit_config=ExitConfig("ouro"),
    )


def test_pd_rejects_dory_speculation_before_spawning():
    from vllm_rlt.pd.engine import PDEngine

    with pytest.raises(ValueError, match="boundary-KV speculative decoding"):
        PDEngine(config("per_loop"), speculative_config=SpeculativeConfig(3))


def test_import_after_a_prefix_hit_marks_only_the_owned_planes():
    model = oracle("per_loop")
    cache = make_cache(model, enable_prefix_caching=True)
    prompt = list(range(9))
    assert cache.allocate("source", 12)
    cache.mark_imported_prefix("source", len(prompt))
    cache.publish_prefix("source", prompt, len(prompt))
    prefix = cache.lookup_prefix(prompt)
    assert len(prefix) * cache.block_size == 8
    # D's own prefix hit leaves boundary layers unwritten in deeper planes.
    assert cache.allocate("hit", 12, prefix=prefix)
    cache.mark_imported_prefix("hit", len(prompt), start=8)
    for plane, layers in enumerate(cache._get_allocation("hit").written):
        for layer, written in enumerate(layers):
            owned = plane == 0 or layer in model.recurrent_kv_layers
            assert written.prefix == (len(prompt) if owned else 0)


def planes(model):
    boundary = (*model.prelude_kv_layers, *model.coda_kv_layers)
    for layer in range(model.config.num_kv_layers):
        for depth in (0,) if layer in boundary else range(model.config.total_ut_steps):
            yield layer, depth


def receiver_cache(model, num_blocks=64):
    config = model.config
    return KVCacheManager(
        num_layers=config.num_kv_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=num_blocks,
        block_size=2,
        max_loops=config.total_ut_steps,
        layout=config.kv_cache_layout,
        recurrent_layers=model.recurrent_kv_layers,
        layer_windows=model.kv_layer_windows,
        enable_prefix_caching=True,
        incremental_allocation=True,
        reclaim_swa=True,
        max_submission_tokens=3,
    )


def test_import_reserves_only_the_swa_pages_that_decode_reads():
    model = oracle("per_loop")
    cache = receiver_cache(model)
    prompt = 13
    # A local allocation would compute these tokens, so the bound rejects it.
    with pytest.raises(ValueError, match="submission bound"):
        cache.allocate("local", 20, initial_tokens=prompt)
    assert cache.allocate_import("r", 20, prompt, initial_tokens=prompt)
    tables = cache.transfer_tables("r")
    assert len(tables) == cache.pages_per_token_block
    held = sum(block >= 0 for table in tables for block in table)
    assert held == cache.allocated_blocks("r") == cache.num_used_blocks
    # The output stage of token 12 reads positions 9 to 12, so pages 0 to 3 stay empty.
    for layer, depth in planes(model):
        window = model.kv_layer_windows[layer]
        assert cache.retained_range("r", layer, depth)[0] == (8 if window else 0)
    cache.mark_imported_prefix("r", prompt, output_layers=model.coda_kv_layers)
    for plane in cache._get_allocation("r").consumed:
        for layer, consumed in enumerate(plane):
            assert consumed.prefix == (prompt - 1 if layer in model.coda_kv_layers else prompt)
    cache.free("r")
    assert cache.num_used_blocks == 0


def test_short_pool_import_drops_the_attached_prefix_references():
    model = oracle("per_loop")
    cache = receiver_cache(model, num_blocks=48)
    tokens = [5, 6, 7, 8]
    assert cache.allocate("source", 4, initial_tokens=3)
    cache.mark_imported_prefix("source", 3)
    cache.publish_prefix("source", tokens, 3)
    cache.free("source")
    [page] = cache.lookup_prefix(tokens)
    free = cache.num_free_blocks
    # Thirty token pages need more fresh pages than the pool holds.
    assert not cache.allocate_import("r", 60, 4, initial_tokens=60, prefix=(page,))
    assert not cache.has_allocation("r") and cache.num_free_blocks == free
    assert cache.lookup_prefix(tokens) == (page,)
    assert cache.allocate_import("r", 8, 4, initial_tokens=4, prefix=(page,))
    assert [table[0] for table in cache.transfer_tables("r")] == list(page)
    cache.free("r")
    assert cache.num_free_blocks == cache.num_blocks


def stage(kv_mode, role):
    return StageEngine(
        oracle(kv_mode),
        cache_config=cache(kv_mode),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=5, prefill_chunk_size=3
        ),
        stage_config=StageConfig(stage_id=role, role=role),
        exit_config=ExitConfig("ouro"),
    )


@pytest.mark.parametrize("kv_mode", KV_MODES)
def test_cpu_handoff_copies_only_the_retained_swa_pages(kv_mode):
    """Copy the pages that both stages hold, as the connector sends them."""
    p, d = stage(kv_mode, "prefill"), stage(kv_mode, "decode")
    p.connector = RecordingConnector()
    prompt = list(range(1, 14))
    source = p.reserve_transfer("r", prompt, params(6), receiving=False)
    target = d.reserve_transfer("r", prompt, params(6), receiving=True)
    for _ in range(100):
        p.step()
        if source.phase == RequestPhase.TRANSFERRING:
            break
    else:
        pytest.fail("source prompt did not complete")
    assert d.cache_manager.allocated_blocks("r") < p.cache_manager.allocated_blocks("r")
    sent = 0
    for left, right in zip(
        p.cache_manager.transfer_tables("r"), d.cache_manager.transfer_tables("r")
    ):
        for src, dst in zip(left, right):
            if dst >= 0:
                d.cache_manager.key_cache[dst].copy_(p.cache_manager.key_cache[src])
                d.cache_manager.value_cache[dst].copy_(p.cache_manager.value_cache[src])
                sent += 1
    assert sent < p.cache_manager.allocated_blocks("r")
    assert d.accept_import(target, source.hidden_state.clone())
    assert p.release_transfer(source)
    windowed = [layer for layer, window in d.model.kv_layer_windows.items() if window]
    starts = {layer: d.cache_manager.retained_range("r", layer, 0)[0] for layer in windowed}
    final = {}
    while len(target.generated_token_ids) < 4:
        final.update((o.request_id, o) for o in d.step() if o.finished)
    # Reclamation continues from the imported read positions.
    for layer in windowed:
        assert d.cache_manager.retained_range("r", layer, 0)[0] > starts[layer]
    final.update(drain(d))
    assert final["r"].token_ids == greedy_reference(oracle(kv_mode), prompt, 6)
    assert p.cache_manager.num_used_blocks == d.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_stages_keep_grouped_kv_and_only_decode_reclaims(role):
    """P holds a prompt only until its transfer completes, so it reclaims nothing."""
    stage = StageEngine(
        oracle("per_loop"),
        cache_config=cache("per_loop"),
        stage_config=StageConfig(stage_id=role, role=role),
        exit_config=ExitConfig("ouro"),
    )
    assert stage.cache_manager.grouped
    assert stage.cache_manager.reclaim_swa is (role == "decode")


@pytest.mark.gpu
@pytest.mark.parametrize("kv_mode", KV_MODES)
@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_dory_pd_matches_the_oracle_and_a_local_engine(kv_mode, mode):
    model = oracle(kv_mode)
    expected = {rid: greedy_reference(model, prompt, 5) for rid, prompt in PROMPTS.items()}
    stochastic = params(5, temperature=0.8, top_k=16, seed=947)
    local = local_engine(kv_mode)
    for rid, prompt in PROMPTS.items():
        local.add_request(rid, prompt, params(5))
    local.add_request("seeded", PROMPTS["0"], stochastic)
    reference = tokens(drain(local))
    with create_pd(kv_mode, mode) as e:
        for rid, prompt in PROMPTS.items():
            e.add_request(rid, prompt, params(5))
        e.add_request("seeded", PROMPTS["0"], stochastic)
        final = drain(e)
        assert tokens(final) == reference
        for rid in PROMPTS:
            assert final[rid].token_ids == expected[rid]
            assert final[rid].exit_depths == [LOOPS] * 5
        # Cancel while a reservation can still be in flight, then reuse the ID.
        e.add_request("reuse", PROMPTS["0"], params(5))
        e.step()
        e.abort_request("reuse")
        e.add_request("reuse", PROMPTS["1"], params(5))
        assert drain(e)["reuse"].token_ids == expected["1"]
        assert e.stats()["reserved_blocks"] == 0
        assert all(p.slots == 0 and p.reserved_blocks == 0 for p in e.peers.values())
    assert all(m["used_blocks"] == 0 for m in e.worker_metrics.values())
    assert sum(m["bytes_sent"] for m in e.worker_metrics.values()) > 0


@pytest.mark.gpu
@pytest.mark.parametrize("p_cache,d_cache", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("mode", ["sync_eager", "async_graph"])
def test_dory_pd_prefix_reuse_skips_work_and_keeps_outputs(p_cache, d_cache, mode):
    prompt = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    expected = greedy_reference(oracle("per_loop"), prompt, 6)
    with create_pd("per_loop", mode, p_cache=p_cache, d_cache=d_cache) as e:
        for round_index in range(2):
            for i in range(2):
                e.add_request(f"{round_index}-{i}", prompt, params(6))
            assert all(t == expected for t in tokens(drain(e)).values())
        assert all(p.slots == 0 and p.reserved_blocks == 0 for p in e.peers.values())
    stats = list(e.worker_metrics.values())
    p = next(x for x in stats if x["prefill_tokens"])
    assert (p["prefill_tokens"] < 4 * len(prompt)) == p_cache
    assert all(x["used_blocks"] == 0 for x in stats)


@pytest.mark.gpu
def test_grouped_pd_sends_less_kv_than_all_layer_pd():
    """Grouped PD skips boundary layers in deeper planes and expired SWA pages."""
    prompt = list(range(1, 41))
    expected = greedy_reference(oracle("per_loop"), prompt, 4)
    sent = {}
    for mode in ("sync_eager", "async_eager"):
        with create_pd("per_loop", mode) as e:
            e.add_request("long", prompt, params(4))
            assert drain(e)["long"].token_ids == expected
        sent[mode] = sum(m["bytes_sent"] for m in e.worker_metrics.values())
    assert 0 < sent["sync_eager"] < sent["async_eager"]


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["sync_eager", "async_eager"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_dory_pd_cancel_releases_both_stages(phase, mode):
    with create_pd("per_loop", mode) as e:
        e.add_request("cancel", [1] * 40, params(20))
        deadline = time.monotonic() + 30
        while True:
            e.step()
            w = next(iter(e.transfers.values()))
            if (phase == "prefill" and "prefill_started" in w.timings) or (
                phase == "decode" and w.phase == "decode"
            ):
                break
            assert time.monotonic() < deadline
        e.abort_request("cancel")
        assert drain(e) == {}
        assert all(p.slots == 0 and p.reserved_blocks == 0 for p in e.peers.values())
        e.add_request("after", PROMPTS["0"], params(5))
        assert drain(e)["after"].token_ids == greedy_reference(oracle("per_loop"), PROMPTS["0"], 5)
    assert all(m["used_blocks"] == 0 for m in e.worker_metrics.values())
