# SPDX-License-Identifier: Apache-2.0
"""Huginn numerical, checkpoint and request-lifecycle regressions."""

import json

import pytest
import torch

from tests.reference.huginn import dense_huginn_reference
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, HuginnConfig, HuginnForCausalLM


def tiny_huginn_config(**overrides):
    values = dict(
        n_embd=32,
        n_heads=4,
        n_layers=4,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=1,
        intermediate_size=64,
        mean_recurrence=3,
        block_size=64,
        vocab_size=64,
        padded_vocab_size=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    values.update(overrides)
    return HuginnConfig(**values)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return HuginnForCausalLM(tiny_huginn_config()).eval()


def make_cache(model, device="cpu", backend="torch"):
    config = model.config
    return KVCacheManager(
        num_layers=config.n_layers,
        num_kv_heads=config.n_heads,
        head_dim=config.head_dim,
        num_blocks=128,
        block_size=2,
        max_loops=config.mean_recurrence,
        device=device,
        dtype=next(model.parameters()).dtype,
        backend=backend,
        recurrent_layers=model.recurrent_kv_layers,
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_prelude_core_coda_and_kv_match_independent_dense(model, parallel):
    tokens = torch.tensor([4, 7, 9, 3, 12])
    state = torch.linspace(-0.2, 0.2, len(tokens) * model.config.n_embd).reshape(len(tokens), -1)
    expected_depths, expected_logits = dense_huginn_reference(model, tokens, state)
    cache = make_cache(model)
    assert cache.allocate("request", len(tokens))
    chunks = [list(range(len(tokens)))] if parallel else [[i] for i in range(len(tokens))]
    for positions in chunks:
        batch = cache._prepare_batch(["request"] * len(positions), [0] * len(positions), positions)
        hidden = model.prelude_prepared(tokens[positions], batch, cache)
        hidden = torch.cat((state[positions], hidden[:, model.config.n_embd :]), -1)
        for depth, expected in enumerate(expected_depths):
            core = cache._prepare_batch(
                ["request"] * len(positions),
                [depth] * len(positions),
                positions,
            )
            hidden, _ = model.recurrent_prepared(hidden, core, cache)
            torch.testing.assert_close(
                hidden[:, : model.config.n_embd], expected[positions], atol=3e-6, rtol=3e-5
            )
        logits = model.coda_prepared(hidden, batch, cache)
        torch.testing.assert_close(logits, expected_logits[positions], atol=3e-6, rtol=3e-5)
        for position in positions:
            cache.finalize_token("request", position, model.config.mean_recurrence - 1)
    allocation = cache._get_allocation("request")
    assert all(
        all(position in allocation.written[0][layer] for position in range(len(tokens)))
        for layer in (0, model.config.n_layers - 1)
    )
    cache.free("request")
    assert cache.num_used_blocks == 0


@pytest.mark.parametrize("sharded", [False, True])
def test_strict_checkpoint_and_auto_dispatch_preserve_tied_head(tmp_path, model, sharded):
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    weights = {name: value.clone().contiguous() for name, value in model.state_dict().items()}
    names = list(weights)
    groups = [names[: len(names) // 2], names[len(names) // 2 :]] if sharded else [names]
    mapping = {}
    for i, group in enumerate(groups):
        filename = f"model-{i}.safetensors" if sharded else "model.safetensors"
        save_file({name: weights[name] for name in group}, tmp_path / filename)
        mapping.update(dict.fromkeys(group, filename))
    if sharded:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert isinstance(loaded, HuginnForCausalLM)
    assert loaded.lm_head.weight is loaded.transformer.wte.weight
    for name, expected in weights.items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)


def test_boundary_writes_survive_shallow_recurrent_finalization(model):
    cache = make_cache(model)
    assert cache.allocate("request", 2)
    allocation = cache._get_allocation("request")
    for layer in range(model.config.n_layers):
        cache.write(layer, ["request"], [0], [0], torch.ones(1, 4, 8), torch.ones(1, 4, 8))
    cache.finalize_token("request", 0, 0)
    assert 0 in allocation.written[0][0]
    assert 0 not in allocation.written[1][0]
    assert all(0 in allocation.written[2][layer] for layer in model.recurrent_kv_layers)


def test_request_reuse_releases_boundary_and_recurrent_kv(model):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=128, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
    )
    rounds = []
    for _ in range(2):
        torch.manual_seed(123)
        for name, prompt in (("first", [4, 7, 3]), ("second", [9, 2])):
            engine.add_request(
                name,
                prompt,
                SamplingParams(
                    max_tokens=3,
                    min_loops=3,
                    max_loops=3,
                    ignore_eos=True,
                ),
            )
        finished = {}
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = (output.token_ids, output.exit_depths)
        assert set(finished) == {"first", "second"}
        assert all(depths == [3] * 3 for _, depths in finished.values())
        assert engine.cache_manager.num_used_blocks == 0
        rounds.append(finished)
    assert rounds[0] == rounds[1]


@pytest.mark.parametrize("temperature", [0, 0.7])
@pytest.mark.parametrize("padded", [False, True])
def test_request_seed_is_independent_of_batching_and_global_rng(model, temperature, padded):
    prompts = {"first": [4, 7, 3], "second": [9, 2, 5, 6]}
    seeds = {"first": 17, "second": 83}

    def run(names, global_seed):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=128, block_size=2),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=3
            ),
            execution_config=ExecutionConfig(static_buffers=padded, pad_to_power_of_two=padded),
        )
        torch.manual_seed(global_seed)
        before = torch.random.get_rng_state().clone()
        for name in names:
            engine.add_request(
                name,
                prompts[name],
                SamplingParams(
                    max_tokens=4,
                    min_loops=3,
                    max_loops=3,
                    ignore_eos=True,
                    seed=seeds[name],
                    temperature=temperature,
                ),
            )
        result = {}
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    result[output.request_id] = (output.token_ids, output.exit_depths)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        assert engine.cache_manager.num_used_blocks == 0
        assert not engine.rng.generators
        return result

    separate = run(["first"], 1) | run(["second"], 999)
    assert run(["second", "first"], 42) == separate


@pytest.mark.parametrize("padded", [False, True])
def test_prefill_populates_history_without_projecting_unused_logits(model, padded):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=128, block_size=2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=4, prefill_chunk_size=3
        ),
        execution_config=ExecutionConfig(static_buffers=padded, pad_to_power_of_two=padded),
    )
    coda_rows, head_rows = [], []
    hooks = [
        model.transformer.coda[0].register_forward_pre_hook(
            lambda _module, inputs: coda_rows.append(len(inputs[0]))
        ),
        model.lm_head.register_forward_pre_hook(
            lambda _module, inputs: head_rows.append(len(inputs[0]))
        ),
    ]
    prompts = ([4, 7, 3, 2, 6], [9, 2, 5], [3])
    for index, prompt in enumerate(prompts):
        engine.add_request(
            str(index),
            prompt,
            SamplingParams(max_tokens=1, min_loops=3, max_loops=3, ignore_eos=True),
        )
    try:
        while engine.has_unfinished_requests():
            engine.step()
    finally:
        for hook in hooks:
            hook.remove()
    assert sum(coda_rows) == sum(map(len, prompts))
    assert sum(head_rows) == len(prompts)
    assert engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize(
    "cache,execution,scheduler,error",
    [
        (CacheConfig(layout="shared"), ExecutionConfig(), SchedulerConfig(), "last_exited"),
        (
            CacheConfig(enable_prefix_caching=True),
            ExecutionConfig(),
            SchedulerConfig(),
            "prefix caching",
        ),
        (CacheConfig(), ExecutionConfig(), SchedulerConfig(mode="mixed"), "staged prefill"),
        (
            CacheConfig(),
            ExecutionConfig(async_scheduling=True),
            SchedulerConfig(),
            "synchronous",
        ),
    ],
)
def test_unsupported_execution_rejected_before_cache_allocation(
    model, cache, execution, scheduler, error
):
    with pytest.raises(ValueError, match=error):
        LLMEngine(
            model,
            cache_config=cache,
            execution_config=execution,
            scheduler_config=scheduler,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("max_graphs", [1, 16])
def test_huginn_graph_replay_matches_eager_with_reuse_and_fallback(max_graphs):
    torch.manual_seed(42)
    model = (
        HuginnForCausalLM(
            tiny_huginn_config(
                n_embd=384,
                n_heads=4,
                intermediate_size=768,
            )
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    results = []
    for graph in (False, True):
        engine = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=256, block_size=4),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4,
                max_num_batched_tokens=4,
                prefill_chunk_size=2,
            ),
            execution_config=ExecutionConfig(cuda_graphs=graph, cuda_graph_max_graphs=max_graphs),
        )
        rounds = []
        for _ in range(2):
            torch.manual_seed(123)
            for i, prompt in enumerate(([4, 7, 3], [9, 2], [3, 8, 7, 6], [4])):
                engine.add_request(
                    str(i),
                    prompt,
                    SamplingParams(
                        max_tokens=4 + i,
                        min_loops=3,
                        max_loops=3,
                        ignore_eos=True,
                    ),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        finished[output.request_id] = (output.token_ids, output.exit_depths)
            assert len(finished) == 4
            assert engine.cache_manager.num_used_blocks == 0
            rounds.append(finished)
        if graph:
            stats = engine.model_runner.graphs
            assert stats.captures > 0 and stats.replays > stats.captures
            coda = engine.model_runner.coda_graphs
            assert coda.captures > 0 and coda.replays > coda.captures
            if max_graphs == 1:
                assert stats.fallbacks > 0
                assert coda.fallbacks > 0
        results.append(rounds)
    assert results[0] == results[1]


@pytest.mark.gpu
@torch.inference_mode()
def test_huginn_graph_states_logits_and_kv_match_eager():
    from vllm_rlt.worker.cuda_graph import RecurrentGraphs

    torch.manual_seed(42)
    model = (
        HuginnForCausalLM(tiny_huginn_config(n_embd=384, n_heads=4, intermediate_size=768))
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    caches = [make_cache(model, "cuda", "triton") for _ in range(2)]
    graphs = RecurrentGraphs(model, caches[1], ExecutionConfig(cuda_graphs=True), False)
    for cache in caches:
        assert cache.allocate("request", 5)
    for position in range(5):
        states = []
        for cache in caches:
            torch.manual_seed(123 + position)
            batch = cache._prepare_batch(["request"], [0], [position])
            hidden = model.prelude_prepared(
                torch.tensor([4 + position], device="cuda"), batch, cache
            )
            states.append(hidden)
        torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
        for depth in range(3):
            batches = [cache._prepare_batch(["request"], [depth], [position]) for cache in caches]
            states[0], _ = model.recurrent_prepared(
                states[0], batches[0], caches[0], compute_gate=False
            )
            call = graphs.select(states[1], batches[1])
            graphs.prepare_inputs(call, states[1])
            states[1], _ = graphs.execute(call, states[1])
            torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
            for layer in model.recurrent_kv_layers:
                for eager, replay in zip(
                    caches[0].read(layer, "request", depth), caches[1].read(layer, "request", depth)
                ):
                    torch.testing.assert_close(eager, replay, rtol=0, atol=0)
        logits = [
            model.coda_prepared(hidden, cache._prepare_batch(["request"], [0], [position]), cache)
            for hidden, cache in zip(states, caches)
        ]
        torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
        for cache in caches:
            cache.finalize_token("request", position, 2)
    assert graphs.captures == 1 and graphs.replays == 15 and graphs.fallbacks == 0
    for cache in caches:
        cache.free("request")
        assert cache.num_used_blocks == 0


def test_plane_packing_geometry_and_live_transfer_slots():
    from vllm_rlt.core.kv_cache_manager import physical_planes, plane_packing

    # Huginn-0125: 2 prelude + 4 recurrent + 2 coda layers, 32 recurrences.
    assert plane_packing(8, range(2, 6), 32) == 2
    assert physical_planes(8, range(2, 6), 32) == 17
    assert physical_planes(8, range(2, 6), 32, enabled=False) == 32
    assert physical_planes(8, range(2, 6), 32, layout="shared") == 1
    # Fully recurrent models (Ouro, Nanbeige) and non-contiguous cores never pack.
    assert plane_packing(24, None, 4) == 1
    assert plane_packing(8, (1, 2, 4, 5), 32) == 1
    assert plane_packing(8, range(1, 6), 32) == 1  # Five of eight layers recur.
    assert physical_planes(9, range(3, 6), 8) == 1 + 3  # Three planes per block.

    def cache(pack, loops=5):
        return KVCacheManager(4, 1, 8, 64, 2, loops, recurrent_layers=(1, 2), **pack)

    packed, unpacked = cache(dict(pack_recurrent_planes=True)), cache({})
    assert (packed.plane_packing, packed.physical_planes) == (2, 3)
    assert packed.transfer_slots == ((0, 4), (0, 4), (0, 4))
    assert unpacked.transfer_slots == ((0, 4),) + ((1, 3),) * 4
    assert cache(dict(pack_recurrent_planes=True), loops=4).transfer_slots == (
        (0, 4),
        (0, 4),
        (0, 2),  # The last group holds one plane.
    )
    assert packed.required_blocks(5) == 3 * 3 and unpacked.required_blocks(5) == 3 * 5
    # The finalize kernel copies recurrent slots of flattened pages when packed.
    assert packed.finalize_layers == (1, 2) and unpacked.finalize_layers == (0, 4)
    assert packed.entry_stride == packed.key_cache.stride(1)
    assert unpacked.entry_stride == unpacked.key_cache.stride(0)


def test_packed_entries_address_disjoint_live_slots():
    cache = KVCacheManager(4, 1, 8, 64, 2, 5, recurrent_layers=(1, 2), pack_recurrent_planes=True)
    assert cache.allocate("a", 5) and cache.allocate("b", 3)
    assert cache.num_used_blocks == 3 * 3 + 2 * 3
    addresses = {}
    for rid in ("a", "b"):
        for depth in range(5):
            layers = range(4) if depth == 0 else (1, 2)
            for entry in cache.get_block_table(rid, depth):
                for layer in layers:
                    page = entry + layer
                    assert 0 <= page < cache.num_blocks * cache.num_layers
                    assert page not in addresses, "two live (depth, layer) slots alias"
                    addresses[page] = (rid, depth, layer)
    physical = {b for rid in ("a", "b") for table in cache.plane_block_tables(rid) for b in table}
    assert {page // cache.num_layers for page in addresses} == physical
    # Growth keeps entries in step with the physical tables.
    assert cache.allocate("c", 9, initial_tokens=2) and cache.ensure_capacity("c", 9)
    grown = cache._get_allocation("c")
    assert len(grown.block_tables[0]) == 5 and all(len(t) == 5 for t in grown.entry_tables)
    assert grown.entry_tables[0] == tuple(b * 4 for b in grown.block_tables[0])
    assert grown.entry_tables[2] == tuple(b * 4 + 2 - 1 for b in grown.block_tables[1])
    batch = cache._prepare_batch(["a"], [1], [0])
    with pytest.raises(ValueError, match="depth zero"):
        cache._write_prepared(0, batch, torch.zeros(1, 1, 8), torch.zeros(1, 1, 8))
    for rid in ("a", "b", "c"):
        cache.free(rid)
    assert cache.num_used_blocks == 0


@pytest.mark.parametrize("max_loops", [None, 2])
@pytest.mark.parametrize("temperature", [0, 0.7])
def test_packed_planes_match_unpacked_outputs_and_kv_bitwise(model, max_loops, temperature):
    """Packing changes addresses only: tokens, exit copies and every live KV byte agree."""
    model = HuginnForCausalLM(tiny_huginn_config(mean_recurrence=5)).eval()
    prompts = {"first": [4, 7, 3, 9, 2], "second": [9, 2, 5]}
    runs = []
    for pack in (False, True):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=128, block_size=2, pack_recurrent_planes=pack),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2, max_num_batched_tokens=8, prefill_chunk_size=3
            ),
        )
        cache = engine.cache_manager
        assert cache.plane_packing == (2 if pack else 1)
        params = SamplingParams(
            max_tokens=4, max_loops=max_loops, ignore_eos=True, seed=11, temperature=temperature
        )
        for name, prompt in prompts.items():
            engine.add_request(name, prompt, params)
        outputs, kv, peak = {}, {}, 0
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    outputs[output.request_id] = (output.token_ids, output.exit_depths)
            peak = max(peak, cache.num_used_blocks)
            for name in prompts:
                if not cache.has_allocation(name):
                    continue
                allocation = cache._get_allocation(name)
                for depth in range(5):
                    for layer in range(4) if depth == 0 else model.recurrent_kv_layers:
                        length = allocation.written[depth][layer].prefix
                        if length:
                            kv[name, depth, layer, length] = cache.read(layer, name, depth, length)
        assert cache.num_used_blocks == 0
        runs.append((outputs, kv, peak))
    (unpacked, unpacked_kv, unpacked_peak), (packed, packed_kv, packed_peak) = runs
    assert packed == unpacked
    if max_loops is not None:
        # Decode exits early, so finalization copies depth-2 KV into packed planes.
        assert all(depths[1:] == [2] * 3 for _, depths in packed.values())
    assert packed_kv.keys() == unpacked_kv.keys() and packed_kv
    for key, (k, v) in packed_kv.items():
        torch.testing.assert_close(k, unpacked_kv[key][0], rtol=0, atol=0)
        torch.testing.assert_close(v, unpacked_kv[key][1], rtol=0, atol=0)
    # Five planes need three physical blocks per page instead of five.
    assert packed_peak * 5 == unpacked_peak * 3


@pytest.mark.gpu
def test_packed_planes_match_unpacked_on_gpu():
    """CUDA graphs and exit copies address packed planes (Huginn runs synchronously)."""
    torch.manual_seed(42)
    model = (
        HuginnForCausalLM(
            tiny_huginn_config(n_embd=384, n_heads=4, intermediate_size=768, mean_recurrence=5)
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    prompts = ([4, 7, 3], [9, 2], [3, 8, 7, 6], [4])
    runs = []
    for pack in (False, True):
        engine = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=256, block_size=4, pack_recurrent_planes=pack),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4, max_num_batched_tokens=8, prefill_chunk_size=2
            ),
            execution_config=ExecutionConfig(cuda_graphs=True, cuda_graph_max_graphs=16),
        )
        cache = engine.cache_manager
        assert cache.plane_packing == (2 if pack else 1)
        torch.manual_seed(123)
        for i, prompt in enumerate(prompts):
            # Exit at depth 2 of 5: finalization copies into three deeper planes.
            engine.add_request(
                str(i),
                prompt,
                SamplingParams(max_tokens=4 + i, min_loops=2, max_loops=2, ignore_eos=True),
            )
        finished, peak = {}, 0
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = (output.token_ids, output.exit_depths)
            peak = max(peak, cache.num_used_blocks)
        assert len(finished) == 4 and cache.num_used_blocks == 0
        runs.append((finished, peak))
    (unpacked, unpacked_peak), (packed, packed_peak) = runs
    assert packed == unpacked
    assert packed_peak * 5 == unpacked_peak * 3
