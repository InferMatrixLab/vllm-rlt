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
