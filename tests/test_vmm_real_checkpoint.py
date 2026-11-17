# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint parity for VMM-backed KV storage (RFC #192 stage 2).

The storage tests use synthetic blocks; this one runs the pinned Ouro snapshot so
real layer/head shapes, real tokenizer ids and real generated tokens go through
the reservation. Set ``OURO_CHECKPOINT`` to the snapshot directory:

    OURO_CHECKPOINT=/data/rlt-perf/weights/Ouro-1.4B CUDA_VISIBLE_DEVICES=0 \
        python -m pytest tests/test_vmm_real_checkpoint.py -q --run-gpu

Oracle provenance: the baseline is the ordinary ``torch.empty`` pool on the same
weights, prompts and sampling parameters. A VMM mapping that shifted, doubled or
truncated a plane changes either the generated tokens or the KV bytes read back
for the same positions. The two engines run sequentially because two automatic
plans cannot share one device budget.
"""

import gc
import os

import pytest
import torch

from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM, load_tokenizer

CHECKPOINT = os.environ.get("OURO_CHECKPOINT")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not CHECKPOINT, reason="set OURO_CHECKPOINT to the pinned Ouro snapshot directory"
    ),
]

PROMPTS = ["The capital of France is", "In a distant future, humans have learned to"]
# Long enough to exceed two 16-token pages, so the prefix cache has something to
# publish. The text is real input, tokenized by the checkpoint's own tokenizer.
LONG_TEXT = (
    "The capital of France is Paris, a city on the Seine with the Eiffel Tower, "
    "the Louvre, Notre-Dame and centuries of documented history."
)
MAX_TOKENS = 8
# The France prompt tokenizes to exactly these ids on the pinned snapshot, which
# pins the real tokenizer as the source of the input.
FRANCE_IDS = [504, 3575, 282, 4649, 314]


@pytest.fixture(scope="module")
def model():
    return OuroForCausalLM.from_pretrained(CHECKPOINT, device="cuda:0", dtype=torch.bfloat16)


@pytest.fixture(scope="module")
def prompts():
    tokenizer = load_tokenizer(CHECKPOINT)
    encoded = [tokenizer.encode(text) for text in PROMPTS]
    assert encoded[0] == FRANCE_IDS, "the pinned snapshot must tokenize the prompt this way"
    return encoded


@pytest.fixture(scope="module")
def long_prompt():
    ids = load_tokenizer(CHECKPOINT).encode(LONG_TEXT)
    assert len(ids) >= 33, "prefix caching needs at least two full pages"
    return ids


def free_bytes():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


def run(
    model,
    prompt_ids,
    *,
    cache_config,
    execution_config=None,
    layers=(0,),
    backend="torch",
    sequential=False,
):
    """Drive one engine to completion; return outputs, plan, cache stats and KV bytes.

    ``sequential`` admits one request per wave, which is what the prefix cache
    needs: a page is only publishable after the request that wrote it finishes.
    """
    assert layers, "capture at least one layer so the KV bytes are compared"
    stage = StageEngine(
        model,
        cache_config=cache_config,
        attention_backend=backend,
        execution_config=execution_config,
    )
    waves = [[ids] for ids in prompt_ids] if sequential else [list(prompt_ids)]
    tokens, captured, used = {}, {}, 0
    request_ids = []
    try:
        for wave in waves:
            for ids in wave:
                rid = str(len(request_ids))
                request_ids.append(rid)
                stage.add_request(
                    rid,
                    list(ids),
                    SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
                )
            while stage.has_unfinished_requests():
                for output in stage.step():
                    if output.finished:
                        tokens[output.request_id] = (
                            tuple(output.token_ids),
                            tuple(output.exit_depths),
                        )
                cache = stage.cache_manager
                used = max(used, cache.num_used_blocks)
                for rid in request_ids:
                    if not cache.has_allocation(rid):
                        continue
                    # depth 0 is the plane every loop writes, so it is comparable.
                    keys, _ = cache.read(layers[0], rid, 0)
                    if keys.shape[0] == 0:
                        continue
                    captured[rid] = [(layer, cache.read(layer, rid, 0)) for layer in layers]
        plan = dict(stage.memory_plan)
        stats = dict(
            published=stage.cache_manager.published_blocks,
            committed=stage.cache_manager.committed_blocks,
            maximum=stage.cache_manager.max_blocks,
            used=used,
            prefix_hits=stage.cache_manager.prefix_hits,
            prefix_queries=stage.cache_manager.prefix_queries,
        )
        captures = 0
        for runner in (stage.model_runner, stage.speculative_runner):
            for name in ("graphs", "coda_graphs"):
                graphs = getattr(runner, name, None)
                captures += int(getattr(graphs, "captures", 0) or 0)
    finally:
        stage.close()
    return tokens, plan, stats, captured, captures


def assert_same_tokens_and_kv(baseline, vmm):
    base_tokens, _, base_kv, _ = baseline[0], baseline[1], baseline[3], baseline[4]
    vmm_tokens, vmm_kv = vmm[0], vmm[3]
    assert base_tokens, "the baseline must have produced outputs"
    assert vmm_tokens == base_tokens, "generated tokens must not depend on the pool"
    assert base_kv, "the baseline must have captured KV before the request finished"
    assert set(vmm_kv) == set(base_kv), "both runs must capture the same requests"
    for rid, entries in base_kv.items():
        assert len(vmm_kv[rid]) == len(entries)
        for (layer, (keys, values)), (other_layer, (other_keys, other_values)) in zip(
            entries, vmm_kv[rid]
        ):
            assert layer == other_layer
            assert keys.shape == other_keys.shape, f"layer {layer} prefix length differs"
            assert torch.equal(keys, other_keys), f"layer {layer} K bytes differ"
            assert torch.equal(values, other_values), f"layer {layer} V bytes differ"


def test_real_weights_growth_keeps_tokens_and_kv_bytes(model, prompts):
    layers = (0, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1)
    baseline = run(model, prompts, cache_config=CacheConfig(), layers=layers)
    gc.collect()
    torch.cuda.empty_cache()
    vmm = run(
        model,
        prompts,
        cache_config=CacheConfig(vmm_bootstrap_blocks=1),
        layers=layers,
    )
    assert_same_tokens_and_kv(baseline, vmm)

    base_plan, base_stats = baseline[1], baseline[2]
    vmm_plan, vmm_stats = vmm[1], vmm[2]
    # The option really split the pool, and serving used grown memory.
    assert vmm_plan["vmm"] is True
    assert vmm_plan["vmm_bootstrap_blocks"] == 1
    assert vmm_plan["vmm_max_blocks"] > 1, "the device budget must allow a real growth"
    assert vmm_stats["published"] == vmm_plan["vmm_max_blocks"] == vmm_stats["maximum"]
    assert vmm_stats["committed"] >= vmm_stats["published"]
    assert vmm_stats["used"] > 1, "requests must have used blocks the growth mapped"
    # The baseline keeps its fixed-capacity bookkeeping.
    assert "vmm" not in base_plan
    assert base_stats["maximum"] == base_stats["published"] == base_stats["committed"]


def test_real_weights_growth_with_cuda_graphs(model, prompts):
    # The torch backend has no graph capability; Triton is the graph path.
    execution = ExecutionConfig(cuda_graphs=True)
    baseline = run(
        model, prompts, cache_config=CacheConfig(), execution_config=execution, backend="triton"
    )
    gc.collect()
    torch.cuda.empty_cache()
    vmm = run(
        model,
        prompts,
        cache_config=CacheConfig(vmm_bootstrap_blocks=1),
        execution_config=execution,
        backend="triton",
    )
    assert_same_tokens_and_kv(baseline, vmm)
    assert baseline[4] > 0, "the baseline must have captured a graph"
    assert vmm[4] > 0, "the graph path must have captured at least one entry"


def test_real_weights_growth_with_prefix_caching(model, long_prompt):
    # One page becomes publishable only after the request that wrote it finished,
    # so the second identical request must be admitted after the first drains.
    repeats = [long_prompt, long_prompt]
    baseline = run(
        model,
        repeats,
        cache_config=CacheConfig(enable_prefix_caching=True),
        sequential=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    vmm = run(
        model,
        repeats,
        cache_config=CacheConfig(enable_prefix_caching=True, vmm_bootstrap_blocks=1),
        sequential=True,
    )
    assert_same_tokens_and_kv(baseline, vmm)
    base_stats, vmm_stats = baseline[2], vmm[2]
    assert base_stats["prefix_hits"] > 0, "the repeated prompt must hit published pages"
    assert vmm_stats["prefix_hits"] == base_stats["prefix_hits"]
    assert vmm_stats["prefix_queries"] == base_stats["prefix_queries"]


def test_real_weights_bootstrap_above_the_plan_clamps(model, prompts):
    baseline = run(model, prompts, cache_config=CacheConfig())
    gc.collect()
    torch.cuda.empty_cache()
    vmm = run(model, prompts, cache_config=CacheConfig(vmm_bootstrap_blocks=10**7))
    assert_same_tokens_and_kv(baseline, vmm)
    plan, stats = vmm[1], vmm[2]
    assert plan["vmm_bootstrap_blocks"] == plan["vmm_max_blocks"], "clamped to the plan"
    assert stats["published"] == stats["maximum"]
    assert stats["committed"] >= stats["published"]


def test_real_weights_repeated_create_close_leaks_nothing(model, prompts):
    # Driver free memory is not a usable oracle here: this device is shared with
    # other tenants, and their allocations move between measurements. Assert on
    # what this engine owns instead: every mapped byte returns, and no tensor
    # outlives the cycle.
    gc.collect()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    for cycle in range(3):
        stage = StageEngine(
            model,
            cache_config=CacheConfig(vmm_bootstrap_blocks=1),
            attention_backend="torch",
        )
        storage = stage.kv_storage
        try:
            assert storage is not None, "the pool must exist for this cycle"
            stage.add_request("0", list(prompts[0]), SamplingParams(max_tokens=2, temperature=0.0))
            while stage.has_unfinished_requests():
                stage.step()
            assert storage.committed_bytes > 0
        finally:
            stage.close()
        assert stage.kv_storage is None
        assert storage.committed_bytes == 0, (
            f"cycle {cycle} left {storage.committed_bytes} mapped bytes"
        )
        assert storage.committed_blocks == 0
        storage.close()  # idempotent after the engine already closed it
        gc.collect()
        leaked = torch.cuda.memory_allocated() - allocated_before
        assert leaked <= 64 * (1 << 20), f"cycle {cycle} left tensors alive: {leaked} bytes"
