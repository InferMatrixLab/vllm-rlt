# SPDX-License-Identifier: Apache-2.0
"""Dory preemption: suspend at every stage boundary, resume, match an uninterrupted run.

A suspended Dory request must keep its boundary (input- and output-stage) KV,
its recurrent planes, its carried state with the injection input, its stage and
loop progress, and its sampling RNG. Resident suspension keeps the KV in the
cache; offload copies it to the CPU and frees the allocation.
"""

import pytest
import torch

from tests.reference.dory import dense_dory_reference
from tests.test_dory import make_cache, tiny_dory_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import DoryForCausalLM
from vllm_rlt.request import RequestPhase

LOOPS = 2
PROMPT = [4, 7, 3, 9, 5]
# Steps from admission to each stage boundary with two-token prefill chunks:
# three prefill chunks, the prompt's output stage, then four stages per token.
STAGES = {
    "prefill chunk": (1, RequestPhase.PREFILL, 0),
    "prompt output stage": (3, RequestPhase.CODA, LOOPS),
    "input stage": (4, RequestPhase.PRELUDE, LOOPS),
    "first loop": (5, RequestPhase.RECURRENT, 0),
    "between loops": (6, RequestPhase.RECURRENT, 1),
    "output stage": (7, RequestPhase.CODA, LOOPS),
}
KV_MODES = ["per_loop", "last_loop"]


def dory(kv_mode, seed=42):
    torch.manual_seed(seed)
    config = tiny_dory_config(n_recurrent_loops=LOOPS, recurrent_kv_cache_mode=kv_mode)
    return DoryForCausalLM(config).eval()


def engine(model, *, num_blocks=256, incremental=False, max_num_seqs=2):
    return LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=num_blocks,
            block_size=2,
            layout=model.kv_cache_layout,
            incremental_allocation=incremental,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs, prefill_chunk_size=2, enable_preemption=True
        ),
    )


def params(max_tokens, **kwargs):
    kwargs.setdefault("ignore_eos", True)
    return SamplingParams(max_tokens=max_tokens, min_loops=LOOPS, max_loops=LOOPS, **kwargs)


def drain(e):
    final = {}
    for _ in range(2000):
        if not e.has_unfinished_requests():
            return final
        for output in e.step():
            if output.finished:
                final[output.request_id] = output
    pytest.fail("engine did not drain")


def greedy_reference(model, prompt, count):
    tokens = list(prompt)
    for _ in range(count):
        logits = dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(logits[-1].argmax()))
    return tokens[len(prompt) :]


def serial_cache(model, tokens):
    """KV of the reference schedule: one token per step at the full depth."""
    cache = make_cache(model)
    tokens = torch.tensor(tokens)
    assert cache.allocate("request", len(tokens))
    for position in range(len(tokens)):
        boundary = cache._prepare_batch(["request"], [0], [position])
        hidden = model.prelude_prepared(tokens[position : position + 1], boundary, cache)
        for depth in range(model.config.total_ut_steps):
            core = cache._prepare_batch(["request"], [depth], [position])
            hidden, _ = model.recurrent_prepared(hidden, core, cache)
        model.coda_prepared(hidden, boundary, cache)
    return cache


def assert_kv_is_the_serial_history(e, model, request_id, tokens):
    """Every plane of every position below the request frontier holds serial KV."""
    frontier = len(tokens) - 1
    expected = serial_cache(model, tokens[:frontier])
    boundary = (*model.prelude_kv_layers, *model.coda_kv_layers)
    for layer in range(model.config.num_kv_layers):
        depths = (0,) if layer in boundary else range(model.config.total_ut_steps)
        for depth in depths:
            actual = e.cache_manager.read(layer, request_id, depth, frontier)
            wanted = expected.read(layer, "request", depth, frontier)
            for a, b in zip(actual, wanted):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


def step_to(e, request, stage):
    steps, phase, loops = STAGES[stage]
    for _ in range(steps):
        e.step()
    assert (request.phase, request.loops_done) == (phase, loops)


def run(model, stage, sampling, *, preempt, resident):
    """Serve "a" to completion; "b" arrives at ``stage`` and may preempt "a"."""
    e = engine(model)
    e.add_request("a", PROMPT, params(8, priority=10, **sampling))
    a = e.scheduler.requests["a"]
    step_to(e, a, stage)
    e.add_request("b", [6, 2], params(3, priority=0, **sampling))
    if preempt:
        saved = (a.phase, a.loops_done, a.num_prefilled_tokens)
        state = e.rng.peek("a").get_state().clone() if sampling else None
        assert e.scheduler.preempt(e.scheduler.requests["b"], priority_only=resident)
        snapshot = e.preemption.snapshots["a"]
        assert snapshot["phase"] is saved[0]
        assert (a.loops_done, a.num_prefilled_tokens) == saved[1:]
        assert (snapshot["kv"] is None) is resident
        assert e.cache_manager.has_allocation("a") is resident
        if sampling:
            # Suspension keeps the generator; it does not reseed it.
            assert torch.equal(e.rng.peek("a").get_state(), state)
    final = drain(e)
    if preempt:
        assert e.preemption.preemptions == 1 and e.preemption.resumptions == 1
    assert not e.preemption.snapshots
    assert e.cache_manager.num_used_blocks == 0
    return final


@pytest.mark.parametrize("kv_mode", KV_MODES)
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("resident", [True, False], ids=["resident", "offload"])
def test_greedy_preemption_at_every_stage_matches_the_oracle(kv_mode, stage, resident):
    model = dory(kv_mode)
    final = run(model, stage, {}, preempt=True, resident=resident)
    assert final["a"].token_ids == greedy_reference(model, PROMPT, 8)
    assert final["a"].exit_depths == [LOOPS] * 8
    assert final["b"].token_ids == greedy_reference(model, [6, 2], 3)


@pytest.mark.parametrize("kv_mode", KV_MODES)
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("resident", [True, False], ids=["resident", "offload"])
def test_sampled_preemption_keeps_the_rng_stream(kv_mode, stage, resident):
    model = dory(kv_mode)
    sampling = dict(temperature=0.9, top_k=8, seed=7)
    uninterrupted = run(model, stage, sampling, preempt=False, resident=resident)
    resumed = run(model, stage, sampling, preempt=True, resident=resident)
    assert resumed["a"].token_ids == uninterrupted["a"].token_ids
    assert resumed["b"].token_ids == uninterrupted["b"].token_ids


@pytest.mark.parametrize("kv_mode", KV_MODES)
@pytest.mark.parametrize("stage", ["prefill chunk", "between loops", "output stage"])
def test_restored_kv_is_the_serial_history_in_every_plane(kv_mode, stage):
    model = dory(kv_mode)
    e = engine(model)
    e.add_request("a", PROMPT, params(8, priority=10))
    a = e.scheduler.requests["a"]
    step_to(e, a, stage)
    e.add_request("b", [6, 2], params(1, priority=0))
    assert e.scheduler.preempt(e.scheduler.requests["b"])
    assert not e.cache_manager.has_allocation("a")
    for _ in range(200):
        e.step()
        if a.phase is RequestPhase.PRELUDE and len(a.generated_token_ids) >= 3:
            break
    else:
        pytest.fail("request did not resume")
    assert_kv_is_the_serial_history(e, model, "a", PROMPT + list(a.generated_token_ids))
    drain(e)
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("kv_mode", KV_MODES)
def test_resident_suspension_later_offloaded_resumes_losslessly(kv_mode):
    model = dory(kv_mode)
    e = engine(model)
    e.add_request("a", PROMPT, params(8, priority=10))
    a = e.scheduler.requests["a"]
    step_to(e, a, "between loops")
    e.add_request("b", [6, 2], params(3, priority=0))
    assert e.scheduler.preempt(e.scheduler.requests["b"], priority_only=True)
    assert e.preemption.snapshots["a"]["kv"] is None
    # Capacity pressure converts the resident state to a CPU snapshot without a
    # second logical suspension.
    e.preemption.suspend(a, offload_kv=True)
    assert e.preemption.snapshots["a"]["kv"] is not None
    assert not e.cache_manager.has_allocation("a")
    assert e.preemption.preemptions == 1
    final = drain(e)
    assert final["a"].token_ids == greedy_reference(model, PROMPT, 8)
    assert final["b"].token_ids == greedy_reference(model, [6, 2], 3)
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("kv_mode", KV_MODES)
def test_capacity_pressure_preemption_matches_the_oracle(kv_mode):
    model = dory(kv_mode)
    prompts = {"0": [4, 7, 3], "1": [9, 5, 6], "2": [2, 8, 1]}
    # Each request needs seven two-token pages in every plane; the tight pool
    # holds two finished requests, so the third forces a preemption.
    planes = LOOPS if kv_mode == "per_loop" else 1
    tight = 16 * planes
    results = {}
    for blocks in (256, tight):
        e = engine(model, num_blocks=blocks, incremental=True, max_num_seqs=3)
        for rid, prompt in prompts.items():
            e.add_request(rid, prompt, params(10))
        results[blocks] = {rid: output.token_ids for rid, output in drain(e).items()}
        assert not e.preemption.snapshots and e.cache_manager.num_used_blocks == 0
        if blocks == tight:
            assert e.preemption.preemptions > 0
    for rid, prompt in prompts.items():
        expected = greedy_reference(model, prompt, 10)
        assert results[256][rid] == expected
        assert results[tight][rid] == expected


EXECUTION_MODES = {
    "sync_eager": dict(),
    "sync_graph": dict(cuda_graphs=True),
    "async_eager": dict(async_scheduling=True),
    "async_graph": dict(async_scheduling=True, cuda_graphs=True),
}


def cuda_pair(kv_mode):
    """A CPU oracle model and a CUDA model with the same FP32 weights."""
    reference = dory(kv_mode)
    model = DoryForCausalLM(reference.config)
    model.load_state_dict(reference.state_dict())
    return reference, model.to("cuda").eval()


def cuda_engine(model, mode, *, num_blocks, max_num_seqs, policy="fcfs"):
    return LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(
            num_blocks=num_blocks,
            block_size=2,
            layout=model.kv_cache_layout,
            incremental_allocation=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            prefill_chunk_size=2,
            policy=policy,
            enable_preemption=True,
        ),
        execution_config=ExecutionConfig(**EXECUTION_MODES[mode]),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("kv_mode", KV_MODES)
@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_cuda_capacity_pressure_preemption_matches_the_oracle(kv_mode, mode):
    reference, model = cuda_pair(kv_mode)
    prompts = {"0": [4, 7, 3], "1": [9, 5, 6], "2": [2, 8, 1]}
    planes = LOOPS if kv_mode == "per_loop" else 1
    e = cuda_engine(model, mode, num_blocks=16 * planes, max_num_seqs=3)
    for rid, prompt in prompts.items():
        e.add_request(rid, prompt, params(10))
    final = drain(e)
    assert e.preemption.preemptions > 0
    assert not e.preemption.snapshots and e.cache_manager.num_used_blocks == 0
    for rid, prompt in prompts.items():
        assert final[rid].token_ids == greedy_reference(reference, prompt, 10)


@pytest.mark.gpu
@pytest.mark.parametrize("kv_mode", KV_MODES)
@pytest.mark.parametrize("mode", EXECUTION_MODES)
@pytest.mark.parametrize("after", [1, 3, 5, 6, 9])
def test_cuda_priority_preemption_matches_the_oracle(kv_mode, mode, after):
    """A higher-priority arrival suspends the running request at a safe boundary."""
    reference, model = cuda_pair(kv_mode)
    e = cuda_engine(model, mode, num_blocks=256, max_num_seqs=1, policy="priority")
    e.add_request("low", PROMPT, params(8, priority=10))
    for _ in range(after):
        e.step()
    e.add_request("high", [6, 2], params(3, priority=0))
    final = drain(e)
    assert e.preemption.preemptions == 1 and e.preemption.resumptions == 1
    assert final["low"].token_ids == greedy_reference(reference, PROMPT, 8)
    assert final["high"].token_ids == greedy_reference(reference, [6, 2], 3)
    assert e.cache_manager.num_used_blocks == 0
