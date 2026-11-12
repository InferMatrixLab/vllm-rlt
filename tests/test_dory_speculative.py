# SPDX-License-Identifier: Apache-2.0
"""Synchronous Dory self-speculation: block and token-pipeline rounds."""

from dataclasses import replace

import pytest
import torch

from tests.reference.dory import dense_dory_reference
from tests.test_dory import make_cache, tiny_dory_config
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import DoryForCausalLM
from vllm_rlt.worker.sampler import Sampler

DEPTH_PAIRS = [(1, 2), (1, 4), (2, 4)]


def dory(loops, seed=42, **overrides):
    torch.manual_seed(seed)
    return DoryForCausalLM(tiny_dory_config(n_recurrent_loops=loops, **overrides)).eval()


def engine(model, k, draft, mode="block", **kwargs):
    loops = model.config.total_ut_steps
    kwargs.setdefault("cache_config", CacheConfig(num_blocks=1024, block_size=2))
    kwargs.setdefault("scheduler_config", SchedulerConfig(max_num_seqs=4))
    return LLMEngine(
        model,
        speculative_config=SpeculativeConfig(k, draft_loops=draft, target_loops=loops, mode=mode),
        **kwargs,
    )


def params(model, max_tokens, **kwargs):
    loops = model.config.total_ut_steps
    kwargs.setdefault("ignore_eos", True)
    return SamplingParams(max_tokens=max_tokens, min_loops=loops, max_loops=loops, **kwargs)


def drain(e):
    final = {}
    for _ in range(4000):
        if not e.has_unfinished_requests():
            return final
        for output in e.step():
            if output.finished:
                final[output.request_id] = output
    pytest.fail("speculative scheduler did not drain")


def greedy_reference(model, prompt, count):
    tokens = list(prompt)
    for _ in range(count):
        logits = dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(logits[-1].argmax()))
    return tokens


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


def assert_committed_kv_is_the_target_history(e, model, request_id, tokens, frontier):
    """Every plane of the committed prefix holds what serial execution stores."""
    if not frontier:
        return
    expected = serial_cache(model, tokens[:frontier])
    cache = e.cache_manager
    boundary = (*model.prelude_kv_layers, *model.coda_kv_layers)
    for layer in range(model.config.num_kv_layers):
        depths = (0,) if layer in boundary else range(model.config.total_ut_steps)
        for depth in depths:
            actual = cache.read(layer, request_id, depth, frontier)
            wanted = expected.read(layer, "request", depth, frontier)
            for a, b in zip(actual, wanted):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


def force_drafts(monkeypatch, e, model, sequence, wrong):
    """Make every draft row propose the reference token, or miss where ``wrong`` says.

    ``wrong(position, index)`` receives the input position of the draft row and,
    in block mode, its index in the round. Target rows keep the model's logits.
    """
    runner = e.speculative_runner
    coda, submit_round = runner._coda, runner._submit_round
    round_state = {}

    def submit(batch, *args):
        round_state.update(
            calls=0,
            drafts=max(span.token_count for span in batch.requests) - 1,
        )
        return submit_round(batch, *args)

    def controlled(hidden, boundary=None):
        logits = coda(hidden, boundary)
        positions = [position for _, _, position in boundary.rows]
        if runner.config.mode == "block":
            index = round_state["calls"]
            round_state["calls"] += 1
            if index >= round_state["drafts"]:
                return logits
            drafts = [(row, index) for row in range(len(positions))]
        else:
            depth = {
                token.position: token.depth
                for state in runner.pipeline_states.values()
                for token in state.tokens
            }
            drafts = [
                (row, None)
                for row, position in enumerate(positions)
                if depth[position] == runner.config.draft_loops
            ]
        for row, index in drafts:
            position = positions[row]
            if position + 1 >= len(sequence):
                continue
            token = sequence[position + 1]
            if wrong(position, index):
                token = (token + 1) % model.config.vocab_size
            logits[row] = -100.0
            logits[row, token] = 100.0
        return logits

    monkeypatch.setattr(runner, "_submit_round", submit)
    monkeypatch.setattr(runner, "_coda", controlled)


@pytest.mark.parametrize("mode", ["block", "pipeline"])
@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
@pytest.mark.parametrize("k", [1, 2, 4])
def test_greedy_matches_the_dense_reference_for_ragged_requests(mode, draft, target, k):
    model = dory(target)
    e = engine(model, k, draft, mode)
    prompts = {"a": [4, 7, 3], "b": [9], "c": [5, 6, 8, 2, 11]}
    limits = {"a": 9, "b": 1, "c": 6}
    for rid, prompt in prompts.items():
        e.add_request(rid, prompt, params(model, limits[rid]))
    final = drain(e)
    for rid, prompt in prompts.items():
        expected = greedy_reference(model, prompt, limits[rid])[len(prompt) :]
        assert final[rid].token_ids == expected
        assert final[rid].finish_reason == "length"
        assert final[rid].exit_depths == [target] * limits[rid]
    assert e.cache_manager.num_used_blocks == 0
    assert not e.speculative_runner.pipeline_states


@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
@pytest.mark.parametrize("rejected", [0, 1, 2, 3])
def test_block_commits_through_each_rejection_position(draft, target, rejected, monkeypatch):
    """rejected=3 accepts all three candidates and commits the bonus token."""
    model, k, prompt, count = dory(target), 3, [4, 7, 3], 13
    sequence = greedy_reference(model, prompt, count)
    e = engine(model, k, draft)
    force_drafts(monkeypatch, e, model, sequence, lambda position, index: index == rejected)
    e.add_request("r", prompt, params(model, count))
    request = e.scheduler.requests["r"]
    stats = e.speculative_runner.stats
    rounds = 0
    while e.has_unfinished_requests():
        before = (len(request.generated_token_ids), stats.rounds, stats.accepted_tokens)
        outputs = e.step()
        generated = list(request.generated_token_ids)
        assert generated == sequence[len(prompt) : len(prompt) + len(generated)]
        if stats.rounds == before[1]:
            continue
        rounds += 1
        # First token comes from prefill, so `count - 1 - before` remain.
        remaining = count - before[0]
        drafts = min(k + 1, remaining) - 1
        accepted = min(rejected, drafts)
        assert stats.accepted_tokens - before[2] == accepted
        assert len(generated) - before[0] == accepted + 1
        if outputs and outputs[0].finished:
            break
        frontier = len(prompt) + len(generated) - 1
        allocation = e.cache_manager._get_allocation("r")
        # A rejected suffix is invisible in every plane, draft CODA KV included.
        for plane in allocation.written:
            assert all(w.prefix <= frontier and not w.pending for w in plane)
        assert_committed_kv_is_the_target_history(e, model, "r", sequence, frontier)
    assert rounds >= 3
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
@pytest.mark.parametrize("pattern", ["all", "none", "every_third"])
def test_pipeline_commits_target_history_under_forced_acceptance(
    draft, target, pattern, monkeypatch
):
    model, prompt, count = dory(target), [4, 7, 3], 12
    sequence = greedy_reference(model, prompt, count)
    wrong = {
        "all": lambda position, index: False,
        "none": lambda position, index: True,
        "every_third": lambda position, index: position % 3 == 0,
    }[pattern]
    e = engine(model, 3, draft, "pipeline")
    force_drafts(monkeypatch, e, model, sequence, wrong)
    e.add_request("r", prompt, params(model, count))
    request = e.scheduler.requests["r"]
    while e.has_unfinished_requests():
        outputs = e.step()
        generated = list(request.generated_token_ids)
        assert generated == sequence[len(prompt) : len(prompt) + len(generated)]
        if outputs and outputs[0].finished:
            break
        # Positions below the last committed input have run their target CODA.
        frontier = len(prompt) + len(generated) - 1
        assert_committed_kv_is_the_target_history(e, model, "r", sequence, frontier)
    stats = e.speculative_runner.stats
    verified = count - 1
    assert stats.committed_tokens == verified
    if pattern == "all":
        assert stats.accepted_tokens == verified
    elif pattern == "none":
        assert stats.accepted_tokens == 0
    else:
        assert 0 < stats.accepted_tokens < verified
    assert not e.speculative_runner.pipeline_states
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("mode", ["block", "pipeline"])
def test_eos_inside_an_accepted_prefix_stops_and_releases(mode, monkeypatch):
    model, prompt = dory(2), [4, 7, 3]
    sequence = greedy_reference(model, prompt, 8)
    stop = sequence[len(prompt) + 2]
    assert stop not in sequence[len(prompt) : len(prompt) + 2]
    model.config = replace(model.config, eos_token_id=stop)
    e = engine(model, 4, 1, mode)
    force_drafts(monkeypatch, e, model, sequence, lambda position, index: False)
    e.add_request("r", prompt, params(model, 8, ignore_eos=False))
    result = drain(e)["r"]
    assert result.finish_reason == "stop"
    assert result.token_ids == sequence[len(prompt) : len(prompt) + 3]
    assert e.cache_manager.num_used_blocks == 0
    assert not e.speculative_runner.pipeline_states


@pytest.mark.parametrize("mode", ["block", "pipeline"])
@pytest.mark.parametrize("count", [1, 2, 5])
def test_output_and_context_limits_bound_the_last_round(mode, count):
    model = dory(2)
    limit = model.config.max_position_embeddings
    # The last output token is never an input, so it needs no position.
    prompt = [(3 * i + 2) % model.config.vocab_size for i in range(limit - count + 1)]
    e = engine(model, 4, 1, mode)
    e.add_request("r", prompt, params(model, count))
    result = drain(e)["r"]
    assert result.token_ids == greedy_reference(model, prompt, count)[len(prompt) :]
    assert e.cache_manager.num_used_blocks == 0
    with pytest.raises(ValueError, match="context length"):
        e.add_request("r", prompt, params(model, count + 1))


@pytest.mark.parametrize("mode", ["block", "pipeline"])
def test_cancellation_releases_boundary_state_and_allows_id_reuse(mode):
    model = dory(4)
    e = engine(model, 3, 2, mode)
    e.add_request("a", [4, 7, 3], params(model, 12))
    e.add_request("b", [9, 5], params(model, 6))
    final = {}
    for _ in range(3):
        final.update({output.request_id: output for output in e.step() if output.finished})
    # Mid-generation: prefill and at least one speculative submission have run.
    assert "a" not in final and e.speculative_runner.stats.rounds
    assert e.abort_request("a").finish_reason == "abort"
    assert "a" not in e.speculative_runner.pipeline_states
    e.add_request("a", [6, 2], params(model, 5))
    final.update(drain(e))
    assert final["a"].token_ids == greedy_reference(model, [6, 2], 5)[2:]
    assert final["b"].token_ids == greedy_reference(model, [9, 5], 6)[2:]
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("mode", ["block", "pipeline"])
def test_failed_round_releases_requests_and_allows_id_reuse(mode, monkeypatch):
    model = dory(2)
    e = engine(model, 3, 1, mode)
    e.add_request("a", [4, 7, 3], params(model, 8))
    runner = e.speculative_runner
    original = runner._coda

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected coda failure")

    monkeypatch.setattr(runner, "_coda", fail)
    with pytest.raises(RuntimeError, match="injected coda failure"):
        drain(e)
    assert not runner.pipeline_states
    assert e.cache_manager.num_used_blocks == 0
    monkeypatch.setattr(runner, "_coda", original)
    e.add_request("a", [5], params(model, 3))
    assert drain(e)["a"].token_ids == greedy_reference(model, [5], 3)[1:]


def test_block_sampling_is_seeded_and_isolated_between_requests():
    model = dory(2)

    def run(batch):
        e = engine(model, 3, 1)
        for rid, seed in batch:
            e.add_request(rid, [4, 7, 3], params(model, 9, temperature=0.8, top_k=8, seed=seed))
        final = drain(e)
        assert e.cache_manager.num_used_blocks == 0
        return {rid: final[rid].token_ids for rid, _ in batch}

    together = run([("a", 11), ("b", 12)])
    assert together == run([("a", 11), ("b", 12)])
    assert together["a"] == run([("a", 11)])["a"]
    assert together["a"] != together["b"]


def test_block_sampling_recovers_the_target_distribution():
    """Rejection sampling over draft-depth proposals must sample the target model.

    The second output token is the first one a speculative round commits, by
    accepting the draft candidate or sampling the residual. Its exact law is
    the mixture over the first token of the full-depth distributions, which
    the dense oracle gives in closed form.
    """
    # Large residual steps make one loop a visibly different model from two.
    model, prompt = dory(2, ngpt_alpha_init=1.0), [4, 7, 3]
    sampling = dict(temperature=1.0, top_k=3)
    probe = params(model, 3, **sampling)

    def distribution(tokens):
        logits = dense_dory_reference(model, torch.tensor(tokens))["logits"][-1]
        return Sampler.probabilities(logits, probe)

    first = distribution(prompt)
    expected = torch.zeros(model.config.vocab_size)
    for token in first.nonzero().flatten().tolist():
        expected += first[token] * distribution([*prompt, token])
    e = engine(model, 2, 1, scheduler_config=SchedulerConfig(max_num_seqs=8))
    samples = 1600
    counts = torch.zeros(model.config.vocab_size)
    for start in range(0, samples, 8):
        for seed in range(start, start + 8):
            e.add_request(str(seed), prompt, params(model, 3, seed=seed, **sampling))
        for output in drain(e).values():
            counts[output.token_ids[1]] += 1
    stats = e.speculative_runner.stats
    # The draft is a different model: both outcomes of the test must occur.
    assert 0 < stats.accepted_tokens < stats.drafted_tokens
    assert not (counts > 0)[expected == 0].any()
    sigma = (expected * (1 - expected) / samples).sqrt()
    assert ((counts / samples - expected).abs() <= 4.5 * sigma + 1e-9).all()


def test_unqualified_speculative_combinations_fail_at_construction():
    model = dory(2)
    with pytest.raises(ValueError, match="synchronous"):
        engine(model, 2, 1, execution_config=ExecutionConfig(async_scheduling=True))
    with pytest.raises(ValueError, match="CUDA graphs"):
        engine(model, 2, 1, execution_config=ExecutionConfig(cuda_graphs=True))
    with pytest.raises(ValueError, match="without CUDA graphs or preemption"):
        engine(
            model,
            2,
            1,
            scheduler_config=SchedulerConfig(max_num_seqs=4, enable_preemption=True),
        )
    with pytest.raises(ValueError, match="target_loops"):
        LLMEngine(model, speculative_config=SpeculativeConfig(2, draft_loops=1, target_loops=4))
    shared = dory(2, recurrent_kv_cache_mode="last_loop")
    with pytest.raises(ValueError, match="last_exited"):
        LLMEngine(
            shared,
            cache_config=CacheConfig(num_blocks=256, block_size=2, layout="shared"),
            speculative_config=SpeculativeConfig(2, draft_loops=1, target_loops=2),
        )
    pipeline = engine(model, 2, 1, "pipeline")
    with pytest.raises(ValueError, match="greedy"):
        pipeline.add_request("r", [4], params(model, 2, temperature=0.7))
    block = engine(model, 2, 1)
    with pytest.raises(ValueError, match="min_loops and max_loops"):
        block.add_request("r", [4], SamplingParams(max_tokens=2, min_loops=1, max_loops=2))
