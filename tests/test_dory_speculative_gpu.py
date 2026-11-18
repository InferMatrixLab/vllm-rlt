# SPDX-License-Identifier: Apache-2.0
"""Asynchronous Dory self-speculation on CUDA: block and token-pipeline rounds.

Every case needs an explicit ``CUDA_VISIBLE_DEVICES`` and ``--run-gpu``. Each
mode must serve the greedy tokens of the dense oracle. Forced drafts check that
the committed KV of every plane is the serial target history while the next
round is already submitted.
"""

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

pytestmark = pytest.mark.gpu

DEPTH_PAIRS = [(1, 2), (1, 4), (2, 4)]
# Prompts shorter than and longer than the tiny window of four.
PROMPTS = {"a": [4, 7, 3], "b": [9], "c": [5, 6, 8, 2, 11]}
LIMITS = {"a": 12, "b": 1, "c": 7}


def reference_pair(loops, **overrides):
    """A CPU oracle model and a CUDA model with the same FP32 weights."""
    torch.manual_seed(42)
    reference = DoryForCausalLM(tiny_dory_config(n_recurrent_loops=loops, **overrides)).eval()
    model = DoryForCausalLM(tiny_dory_config(n_recurrent_loops=loops, **overrides))
    model.load_state_dict(reference.state_dict())
    return reference, model.to("cuda").eval()


def engine(model, k, draft, mode="block", *, asynchronous=True):
    return LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(num_blocks=1024, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=4),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
        speculative_config=SpeculativeConfig(
            k, draft_loops=draft, target_loops=model.config.total_ut_steps, mode=mode
        ),
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


def assert_released(e):
    runner = e.speculative_runner
    assert not runner.states
    assert all(not buffer.leased for buffer in runner.buffers)
    assert e.cache_manager.num_used_blocks == 0


def serial_cache(reference, tokens):
    """KV of the reference schedule: one token per step at the full depth."""
    cache = make_cache(reference)
    tokens = torch.tensor(tokens)
    assert cache.allocate("request", len(tokens))
    for position in range(len(tokens)):
        boundary = cache._prepare_batch(["request"], [0], [position])
        hidden = reference.prelude_prepared(tokens[position : position + 1], boundary, cache)
        for depth in range(reference.config.total_ut_steps):
            core = cache._prepare_batch(["request"], [depth], [position])
            hidden, _ = reference.recurrent_prepared(hidden, core, cache)
        reference.coda_prepared(hidden, boundary, cache)
    return cache


def device_kv(cache, layer, request_id, depth, length):
    """Read KV that device-owned rounds wrote; the host written frontier lags it."""
    table = torch.tensor(cache.get_block_table(request_id, depth), dtype=torch.long)
    positions = torch.arange(length)
    blocks = table[positions // cache.block_size].to(cache.device)
    offsets = (positions % cache.block_size).to(cache.device)
    return cache.key_cache[blocks, layer, offsets], cache.value_cache[blocks, layer, offsets]


def assert_committed_kv_is_the_target_history(e, reference, request_id, tokens, frontier):
    """Every plane below ``frontier`` holds what serial execution stores."""
    e.speculative_runner.synchronize()
    expected = serial_cache(reference, tokens[:frontier])
    boundary = (*reference.prelude_kv_layers, *reference.coda_kv_layers)
    for layer in range(reference.config.num_kv_layers):
        depths = (0,) if layer in boundary else range(reference.config.total_ut_steps)
        for depth in depths:
            actual = device_kv(e.cache_manager, layer, request_id, depth, frontier)
            wanted = expected.read(layer, "request", depth, frontier)
            for a, b in zip(actual, wanted):
                torch.testing.assert_close(a.cpu(), b, atol=2e-5, rtol=2e-4)


def force_drafts(monkeypatch, e, sequence, wrong):
    """Make every draft row propose the reference token, or miss where ``wrong`` says.

    ``wrong(position, index)`` receives the input position of the draft row and,
    in block mode, its draft offset in the round. Target rows keep the model logits.
    The position read is a host synchronization that only this test adds.
    """
    runner = e.speculative_runner
    coda, draft = runner._coda, runner._draft
    vocab = e.model.config.vocab_size
    state = {"offset": None}

    def drafting(*args):
        state["offset"] = 0
        try:
            return draft(*args)
        finally:
            state["offset"] = None

    def controlled(hidden, boundary=None):
        logits = coda(hidden, boundary)
        positions = boundary.position_ids.tolist()
        if runner.config.mode == "block":
            if state["offset"] is None:
                return logits
            rows = [(row, state["offset"]) for row in range(len(positions))]
            state["offset"] += 1
        else:
            pipeline = runner.states["r"].pipeline
            width = len(positions)
            # The depths still describe the rows before this step.
            depths = pipeline.depths[:width].tolist()
            active = pipeline.active[:width].tolist()
            rows = [(row, None) for row in range(width) if active[row] and depths[row] == 0]
        for row, index in rows:
            position = positions[row]
            if position + 1 >= len(sequence):
                continue
            token = sequence[position + 1]
            if wrong(position, index):
                token = (token + 1) % vocab
            logits[row] = -100.0
            logits[row, token] = 100.0
        return logits

    monkeypatch.setattr(runner, "_draft", drafting)
    monkeypatch.setattr(runner, "_coda", controlled)


@pytest.mark.parametrize("mode", ["block", "pipeline"])
@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
@pytest.mark.parametrize("k", [1, 3])
def test_async_greedy_matches_the_dense_reference(mode, draft, target, k):
    reference, model = reference_pair(target)
    e = engine(model, k, draft, mode)
    for rid, prompt in PROMPTS.items():
        e.add_request(rid, prompt, params(model, LIMITS[rid]))
    final = drain(e)
    for rid, prompt in PROMPTS.items():
        expected = greedy_reference(reference, prompt, LIMITS[rid])[len(prompt) :]
        assert final[rid].token_ids == expected
        assert final[rid].exit_depths == [target] * LIMITS[rid]
    assert e.speculative_runner.submitted_before_collect > 0
    assert_released(e)


def bootstrap(e, model, prompt, count):
    e.add_request("r", prompt, params(model, count))
    request = e.scheduler.requests["r"]
    for _ in range(30):
        if e.step():
            return request
    pytest.fail("bootstrap did not produce a token")


@pytest.mark.parametrize("mode", ["block", "pipeline"])
def test_submission_reads_no_device_value(mode, monkeypatch):
    _, model = reference_pair(4)
    e = engine(model, 3, 1, mode)
    request = bootstrap(e, model, [4, 7, 3], 16)

    def forbidden(*args, **kwargs):
        raise AssertionError("host read in cross-round GPU submission")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        e.step()
        e.step()
    assert len(e._pending_speculative) == 2
    assert len(request.generated_token_ids) == 1
    e.abort_request("r")
    drain(e)
    assert_released(e)


@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
@pytest.mark.parametrize("rejected", [0, 1, 2, 3])
def test_block_commits_target_history_through_each_rejection(draft, target, rejected, monkeypatch):
    """rejected=3 accepts all three candidates and commits the bonus token."""
    reference, model = reference_pair(target)
    prompt, count = [4, 7, 3], 13
    sequence = greedy_reference(reference, prompt, count)
    e = engine(model, 3, draft)
    force_drafts(monkeypatch, e, sequence, lambda position, index: index == rejected)
    e.add_request("r", prompt, params(model, count))
    request = e.scheduler.requests["r"]
    overlapped = False
    while e.has_unfinished_requests():
        outputs = e.step()
        overlapped |= len(e._pending_speculative) == 2
        generated = list(request.generated_token_ids)
        assert generated == sequence[len(prompt) : len(prompt) + len(generated)]
        if outputs and outputs[0].finished:
            break
        if generated and e.cache_manager.has_allocation("r"):
            frontier = len(prompt) + len(generated) - 1
            assert_committed_kv_is_the_target_history(e, reference, "r", sequence, frontier)
    drain(e)
    assert overlapped
    stats = e.speculative_runner.stats
    if rejected:
        assert stats.accepted_tokens > 0
    else:
        assert stats.accepted_tokens == 0
    assert request.generated_token_ids == sequence[len(prompt) :]
    assert_released(e)


@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
@pytest.mark.parametrize("pattern", ["all", "none", "every_third"])
def test_pipeline_commits_target_history_under_forced_acceptance(
    draft, target, pattern, monkeypatch
):
    reference, model = reference_pair(target)
    prompt, count = [4, 7, 3], 12
    sequence = greedy_reference(reference, prompt, count)
    wrong = {
        "all": lambda position, index: False,
        "none": lambda position, index: True,
        "every_third": lambda position, index: position % 3 == 0,
    }[pattern]
    e = engine(model, 3, draft, "pipeline")
    force_drafts(monkeypatch, e, sequence, wrong)
    e.add_request("r", prompt, params(model, count))
    request = e.scheduler.requests["r"]
    while e.has_unfinished_requests():
        outputs = e.step()
        generated = list(request.generated_token_ids)
        assert generated == sequence[len(prompt) : len(prompt) + len(generated)]
        if outputs and outputs[0].finished:
            break
        if generated and e.cache_manager.has_allocation("r"):
            frontier = len(prompt) + len(generated) - 1
            assert_committed_kv_is_the_target_history(e, reference, "r", sequence, frontier)
    drain(e)
    stats = e.speculative_runner.stats
    verified = count - 1
    assert stats.committed_tokens == verified
    if pattern == "all":
        assert stats.accepted_tokens == verified
    elif pattern == "none":
        assert stats.accepted_tokens == 0
    else:
        assert 0 < stats.accepted_tokens < verified
    assert_released(e)


def stage_kv(e, layers, position):
    """KV of one position in every layer of one boundary stage."""
    return torch.stack(
        [
            torch.stack(device_kv(e.cache_manager, layer, "r", 0, position + 1))[:, position]
            for layer in layers
        ]
    )


@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
def test_pipeline_keeps_the_boundary_kv_of_rows_in_flight(draft, target, monkeypatch):
    """A pipeline step stores boundary KV only for the rows that use that stage.

    A retained row keeps the input-stage KV of its first pass. A row between
    the draft and target depths keeps the output-stage KV of its draft pass,
    which the draft rows behind it attend.
    """
    _, model = reference_pair(target, ngpt_alpha_init=1.0)
    e = engine(model, 3, draft, "pipeline")
    runner = e.speculative_runner
    coda = runner._coda
    steps = []

    def recording(hidden, boundary=None):
        # Inside the step, the device depths still describe the rows before it.
        pipeline = runner.states["r"].pipeline
        positions = boundary.position_ids.tolist()
        depths = pipeline.depths[: len(positions)].tolist()
        active = pipeline.active[: len(positions)].tolist()
        steps.append([(p, d) for p, d, a in zip(positions, depths, active) if a])
        return coda(hidden, boundary)

    monkeypatch.setattr(runner, "_coda", recording)
    e.add_request("r", [4, 7, 3], params(model, 24))
    first, drafted = {}, {}
    retained = between = 0
    while e.has_unfinished_requests():
        submitted = len(steps)
        e.step()
        if len(steps) == submitted or not e.cache_manager.has_allocation("r"):
            continue
        runner.synchronize()
        for position, depth in steps[-1]:
            prelude = stage_kv(e, model.prelude_kv_layers, position)
            output = stage_kv(e, model.coda_kv_layers, position)
            if depth == 0:
                first[position], drafted[position] = prelude, output
                continue
            assert torch.equal(prelude, first[position])
            retained += 1
            if depth + draft < target:
                assert torch.equal(output, drafted[position])
                between += 1
    assert retained
    assert bool(between) == (target - draft > draft)
    assert_released(e)


@pytest.mark.parametrize("mode", ["block", "pipeline"])
def test_cancellation_with_rounds_in_flight_allows_id_reuse(mode):
    reference, model = reference_pair(4)
    e = engine(model, 3, 2, mode)
    e.add_request("a", [4, 7, 3], params(model, 14))
    e.add_request("b", [9, 5], params(model, 8))
    final = {}
    for _ in range(30):
        final.update({output.request_id: output for output in e.step() if output.finished})
        if len(e._pending_speculative) == 2:
            break
    assert len(e._pending_speculative) == 2 and "a" not in final
    assert e.abort_request("a").finish_reason == "abort"
    e.add_request("a", [6, 2], params(model, 5))
    final.update(drain(e))
    assert final["a"].token_ids == greedy_reference(reference, [6, 2], 5)[2:]
    assert final["b"].token_ids == greedy_reference(reference, [9, 5], 8)[2:]
    assert_released(e)


@pytest.mark.parametrize("mode", ["block", "pipeline"])
def test_failed_round_releases_requests_and_allows_id_reuse(mode, monkeypatch):
    reference, model = reference_pair(2)
    e = engine(model, 3, 1, mode)
    bootstrap(e, model, [4, 7, 3], 8)
    runner = e.speculative_runner
    name = "_pipeline_step" if mode == "pipeline" else "_verify"
    original = getattr(runner, name)

    def fail(*args):
        original(*args)
        raise RuntimeError("injected failure after submission")

    monkeypatch.setattr(runner, name, fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        drain(e)
    assert not e._pending_speculative and not e._retiring_speculative
    assert not e.scheduler.requests
    assert_released(e)
    monkeypatch.setattr(runner, name, original)
    e.add_request("r", [5], params(model, 3))
    assert drain(e)["r"].token_ids == greedy_reference(reference, [5], 3)[1:]
    assert_released(e)


def test_unqualified_async_combinations_are_rejected():
    _, model = reference_pair(2)
    e = engine(model, 2, 1)
    with pytest.raises(ValueError, match="greedy sampling"):
        e.add_request("r", [4], params(model, 2, temperature=0.7))
    with pytest.raises(ValueError, match="preemption"):
        LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=256, block_size=2),
            scheduler_config=SchedulerConfig(max_num_seqs=4, enable_preemption=True),
            execution_config=ExecutionConfig(async_scheduling=True),
            speculative_config=SpeculativeConfig(2, draft_loops=1, target_loops=2),
        )
