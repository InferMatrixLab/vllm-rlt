# SPDX-License-Identifier: Apache-2.0
"""Synchronous Dory self-speculation with preemption.

Suspension happens between collected submissions. A suspended request keeps
its committed KV, the boundary and recurrent KV that in-flight pipeline tokens
stored above the committed frontier, the pipeline state (positions, depths,
carried hidden state and candidates) and its sampling RNG. Resident suspension
keeps the KV in the cache; offload copies it to the CPU and frees the pages.
"""

import pytest
import torch

from tests.reference.dory import dense_dory_reference
from tests.test_dory import tiny_dory_config
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import DoryForCausalLM
from vllm_rlt.request import RequestPhase

DEPTH_PAIRS = [(1, 2), (1, 4), (2, 4)]
MODES = ["block", "pipeline"]
PROMPT = [4, 7, 3, 9, 5]
OTHER = [6, 2]
K = 3
# Block rounds commit up to K + 1 tokens, so "a" must outlive five steps.
A_TOKENS = 20


def dory(loops, seed=42):
    torch.manual_seed(seed)
    return DoryForCausalLM(tiny_dory_config(n_recurrent_loops=loops)).eval()


def engine(
    model,
    draft,
    mode,
    *,
    num_blocks=1024,
    incremental=False,
    max_num_seqs=2,
    prefill_chunk_size=128,
    **kwargs,
):
    return LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=num_blocks, block_size=2, incremental_allocation=incremental
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            enable_preemption=True,
            prefill_chunk_size=prefill_chunk_size,
        ),
        speculative_config=SpeculativeConfig(
            K, draft_loops=draft, target_loops=model.config.total_ut_steps, mode=mode
        ),
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
    return tokens[len(prompt) :]


def written_kv(cache, request_id):
    """Every written position of every plane and layer, with its K and V."""
    allocation = cache._get_allocation(request_id)
    kv = {}
    for plane, layers in enumerate(allocation.written):
        table = allocation.block_tables[plane]
        for layer, written in enumerate(layers):
            positions = sorted({*range(written.prefix), *written.pending})
            if not positions:
                continue
            index = torch.tensor(positions, device=cache.device)
            blocks = torch.tensor(table, device=cache.device)[index // cache.block_size]
            offsets = index % cache.block_size
            kv[plane, layer] = (
                tuple(positions),
                cache.key_cache[blocks, layer, offsets].cpu(),
                cache.value_cache[blocks, layer, offsets].cpu(),
            )
    return kv


def assert_same_kv(actual, expected):
    assert actual.keys() == expected.keys()
    for key, (positions, keys, values) in expected.items():
        assert actual[key][0] == positions, key
        assert torch.equal(actual[key][1], keys), key
        assert torch.equal(actual[key][2], values), key


def trace_pipeline(e, request_id):
    """Record (position, depth) of the request's rows after each pipeline submission."""
    runner = e.speculative_runner
    submit, trace = runner._submit_pipeline, []

    def traced(batch):
        submission = submit(batch)
        for state in submission.pipeline.states:
            if state.request.request_id == request_id:
                trace.append(tuple((t.position, t.depth) for t in state.tokens))
        return submission

    runner._submit_pipeline = traced
    return trace


def committed_frontier(request):
    """Position of the newest committed token; it is the next input."""
    return len(request.prompt_token_ids) + len(request.generated_token_ids) - 1


def run(model, draft, mode, *, preempt, resident, steps, sampling=None, convert=False):
    """Serve "a"; after ``steps`` engine steps "b" arrives and may suspend "a"."""
    sampling = sampling or {}
    e = engine(model, draft, mode)
    trace = trace_pipeline(e, "a") if mode == "pipeline" else None
    e.add_request("a", PROMPT, params(model, A_TOKENS, priority=10, **sampling))
    a = e.scheduler.requests["a"]
    for _ in range(steps):
        e.step()
    assert a.phase is not RequestPhase.FINISHED and len(a.generated_token_ids) >= 1
    e.add_request("b", OTHER, params(model, 4, priority=0, **sampling))
    observed = {}
    if preempt:
        runner = e.speculative_runner
        before_kv = written_kv(e.cache_manager, "a")
        rng = e.rng.peek("a")
        rng = None if rng is None else rng.get_state().clone()
        live = runner.pipeline_states.get("a")
        assert e.scheduler.preempt(e.scheduler.requests["b"], priority_only=resident)
        snapshot = e.preemption.snapshots["a"]
        assert (snapshot["kv"] is None) is resident
        assert e.cache_manager.has_allocation("a") is resident
        assert "a" not in runner.pipeline_states
        if rng is not None:
            # Suspension keeps the generator; it does not reseed it.
            assert torch.equal(e.rng.peek("a").get_state(), rng)
        if mode == "pipeline":
            saved = snapshot["pipeline"]
            assert saved is not None and saved is not live
            target = model.config.total_ut_steps
            # In-flight rows: carried state below the full depth, and draft candidates.
            assert any(t.hidden is not None and 0 < t.depth < target for t in saved.tokens)
            assert all(t.candidate is not None for t in saved.tokens if t.depth >= draft)
            # Their KV lies at or above the committed frontier.
            frontier = committed_frontier(a)
            assert any(p >= frontier for positions, *_ in before_kv.values() for p in positions)
            observed["pipeline"] = [(t.position, t.depth, t.candidate) for t in saved.tokens]
        else:
            assert snapshot["pipeline"] is None
        if convert:
            # Capacity pressure turns resident state into a CPU snapshot without
            # a second logical suspension.
            e.preemption.suspend(a, offload_kv=True)
            assert e.preemption.snapshots["a"]["kv"] is not None
            assert not e.cache_manager.has_allocation("a")
            assert e.preemption.preemptions == 1
        resume = e.scheduler.resume_callback

        def observe(request):
            phase = resume(request)
            if phase and request.request_id == "a":
                assert_same_kv(written_kv(e.cache_manager, "a"), before_kv)
                if mode == "pipeline":
                    state = runner.pipeline_states["a"]
                    assert state.request is a
                    restored = [(t.position, t.depth, t.candidate) for t in state.tokens]
                    assert restored == observed["pipeline"]
                observed["resumed"] = True
            return phase

        e.scheduler.resume_callback = observe
    final = drain(e)
    if preempt:
        assert observed.get("resumed")
        assert e.preemption.preemptions == 1 and e.preemption.resumptions == 1
    assert not e.preemption.snapshots
    assert not e.speculative_runner.pipeline_states
    assert not e.rng.generators
    assert e.cache_manager.num_used_blocks == 0
    e.close()
    return final, trace


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
@pytest.mark.parametrize("resident", [True, False], ids=["resident", "offload"])
@pytest.mark.parametrize("steps", [3, 5])
def test_greedy_suspension_resumes_the_uninterrupted_schedule(mode, draft, target, resident, steps):
    model = dory(target)
    final, trace = run(model, draft, mode, preempt=True, resident=resident, steps=steps)
    assert final["a"].token_ids == greedy_reference(model, PROMPT, A_TOKENS)
    assert final["a"].exit_depths == [target] * A_TOKENS
    assert final["b"].token_ids == greedy_reference(model, OTHER, 4)
    if mode == "pipeline":
        # A recomputed pipeline restarts from one depth-0 row; a restored one
        # continues the same rows and depths as an uninterrupted run.
        _, uninterrupted = run(model, draft, mode, preempt=False, resident=resident, steps=steps)
        assert trace == uninterrupted


@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
@pytest.mark.parametrize("resident", [True, False], ids=["resident", "offload"])
def test_sampled_block_suspension_keeps_the_rng_stream(draft, target, resident):
    model = dory(target)
    sampling = dict(temperature=0.8, top_k=8, seed=7)
    kwargs = dict(resident=resident, steps=3, sampling=sampling)
    uninterrupted, _ = run(model, draft, "block", preempt=False, **kwargs)
    resumed, _ = run(model, draft, "block", preempt=True, **kwargs)
    assert {rid: out.token_ids for rid, out in resumed.items()} == {
        rid: out.token_ids for rid, out in uninterrupted.items()
    }


@pytest.mark.parametrize("mode", MODES)
def test_resident_suspension_later_offloaded_resumes_losslessly(mode):
    model = dory(4)
    final, trace = run(model, 2, mode, preempt=True, resident=True, steps=3, convert=True)
    assert final["a"].token_ids == greedy_reference(model, PROMPT, A_TOKENS)
    assert final["b"].token_ids == greedy_reference(model, OTHER, 4)
    if mode == "pipeline":
        _, uninterrupted = run(model, 2, mode, preempt=False, resident=True, steps=3)
        assert trace == uninterrupted


def test_dropped_pipeline_state_is_detected_by_the_schedule_check(monkeypatch):
    """Recomputation is lossless in tokens, so only the schedule exposes it."""
    model = dory(4)
    monkeypatch.setattr(
        "vllm_rlt.worker.speculative.SpeculativeRunner.restore_pipeline",
        lambda self, request, state: None,
    )
    e = engine(model, 2, "pipeline")
    trace = trace_pipeline(e, "a")
    e.add_request("a", PROMPT, params(model, A_TOKENS, priority=10))
    for _ in range(3):
        e.step()
    e.add_request("b", OTHER, params(model, 4, priority=0))
    assert e.scheduler.preempt(e.scheduler.requests["b"])
    final = drain(e)
    assert final["a"].token_ids == greedy_reference(model, PROMPT, A_TOKENS)
    monkeypatch.undo()
    _, uninterrupted = run(model, 2, "pipeline", preempt=False, resident=False, steps=3)
    assert trace != uninterrupted


def test_kv_check_detects_lost_kv_above_the_committed_frontier():
    """The resume check must see in-flight KV, not only the committed prefix."""
    model = dory(4)
    e = engine(model, 2, "pipeline")
    e.add_request("a", PROMPT, params(model, A_TOKENS, priority=10))
    a = e.scheduler.requests["a"]
    for _ in range(3):
        e.step()
    before = written_kv(e.cache_manager, "a")
    e.add_request("b", OTHER, params(model, 4, priority=0))
    assert e.scheduler.preempt(e.scheduler.requests["b"])
    snapshot = e.preemption.snapshots["a"]["kv"]
    frontier = committed_frontier(a)
    # Zero every plane of the snapshot from the committed frontier's page on.
    # Snapshot rows are plane-major, in block-table order.
    pages = snapshot.pages
    first = frontier // e.cache_manager.block_size
    for plane in range(len(snapshot.keys) // pages):
        snapshot.keys[plane * pages + first : (plane + 1) * pages].zero_()
    assert e.preemption.resume(a) is not None
    with pytest.raises(AssertionError):
        assert_same_kv(written_kv(e.cache_manager, "a"), before)
    e.close()


@pytest.mark.parametrize("mode", MODES)
def test_abort_while_suspended_allows_id_reuse(mode):
    model = dory(4)
    e = engine(model, 2, mode)
    e.add_request("a", PROMPT, params(model, A_TOKENS, priority=10))
    suspended = e.scheduler.requests["a"]
    for _ in range(3):
        e.step()
    e.add_request("b", OTHER, params(model, 4, priority=0))
    assert e.scheduler.preempt(e.scheduler.requests["b"])
    assert "a" in e.preemption.snapshots
    assert e.abort_request("a").finish_reason == "abort"
    assert "a" not in e.preemption.snapshots
    assert "a" not in e.speculative_runner.pipeline_states
    e.add_request("a", [5, 8], params(model, 6))
    assert e.scheduler.requests["a"] is not suspended
    final = drain(e)
    assert final["a"].token_ids == greedy_reference(model, [5, 8], 6)
    assert final["b"].token_ids == greedy_reference(model, OTHER, 4)
    assert not e.preemption.snapshots and not e.speculative_runner.pipeline_states
    assert e.cache_manager.num_used_blocks == 0


LONG_PROMPT = [4, 7, 3, 9, 5, 6, 2, 8, 1]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("draft", "target"), [(1, 2), (2, 4)])
@pytest.mark.parametrize("resident", [True, False], ids=["resident", "offload"])
def test_suspension_during_chunked_prefill_matches_the_oracle(mode, draft, target, resident):
    """A speculative request suspended between prefill chunks resumes on the oracle."""
    model = dory(target)
    e = engine(model, draft, mode, prefill_chunk_size=2)
    e.add_request("a", LONG_PROMPT, params(model, 8, priority=10))
    a = e.scheduler.requests["a"]
    e.step()
    assert 0 < a.num_prefilled_tokens < len(LONG_PROMPT) and not a.generated_token_ids
    e.add_request("b", OTHER, params(model, 4, priority=0))
    assert e.scheduler.preempt(e.scheduler.requests["b"], priority_only=resident)
    assert e.cache_manager.has_allocation("a") is resident
    final = drain(e)
    assert e.preemption.preemptions == 1 and e.preemption.resumptions == 1
    assert final["a"].token_ids == greedy_reference(model, LONG_PROMPT, 8)
    assert final["b"].token_ids == greedy_reference(model, OTHER, 4)
    assert not e.preemption.snapshots
    assert not e.speculative_runner.pipeline_states
    assert e.cache_manager.num_used_blocks == 0
    e.close()


PRESSURE = {"0": [4, 7, 3], "1": [9, 5, 6], "2": [2, 8, 1]}


def pressure_run(model, draft, mode, num_blocks):
    e = engine(model, draft, mode, num_blocks=num_blocks, incremental=True, max_num_seqs=3)
    for rid, prompt in PRESSURE.items():
        e.add_request(rid, prompt, params(model, 10))
    peak = 0
    final = {}
    for _ in range(4000):
        if not e.has_unfinished_requests():
            break
        for output in e.step():
            if output.finished:
                final[output.request_id] = output
        peak = max(peak, e.cache_manager.num_used_blocks)
    else:
        pytest.fail("speculative scheduler did not drain")
    for rid, prompt in PRESSURE.items():
        assert final[rid].token_ids == greedy_reference(model, prompt, 10)
    assert not e.preemption.snapshots and not e.speculative_runner.pipeline_states
    assert e.cache_manager.num_used_blocks == 0
    e.close()
    return e.preemption.preemptions, peak


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("draft", "target"), [(1, 2), (2, 4)])
@pytest.mark.parametrize("pool", ["two_requests", "one_request"])
def test_capacity_pressure_preemption_matches_the_oracle(mode, draft, target, pool):
    """Growth preemption in a tight pool keeps every request on the oracle."""
    model = dory(target)
    preemptions, peak = pressure_run(model, draft, mode, 1024)
    assert preemptions == 0
    # Two thirds of the measured three-request peak holds two requests;
    # one request's full budget plus a page forces repeated suspension.
    blocks = {
        "two_requests": peak * 2 // 3,
        "one_request": model_blocks(model, draft, mode) + 1,
    }[pool]
    preemptions, _ = pressure_run(model, draft, mode, blocks)
    assert preemptions > (1 if pool == "one_request" else 0)


def model_blocks(model, draft, mode):
    probe = engine(model, draft, mode, incremental=True)
    blocks = probe.cache_manager.required_blocks(3 + 10 - 1 + K)
    probe.close()
    return blocks


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("draft", "target"), DEPTH_PAIRS)
@pytest.mark.parametrize("resident", [True, False], ids=["resident", "offload"])
def test_cuda_suspension_matches_the_cpu_oracle(mode, draft, target, resident):
    reference = dory(target)
    model = DoryForCausalLM(reference.config)
    model.load_state_dict(reference.state_dict())
    model = model.to("cuda").eval()
    e = engine(model, draft, mode, attention_backend="triton")
    e.add_request("a", PROMPT, params(model, A_TOKENS, priority=10))
    for _ in range(3):
        e.step()
    e.add_request("b", OTHER, params(model, 4, priority=0))
    assert e.scheduler.preempt(e.scheduler.requests["b"], priority_only=resident)
    snapshot = e.preemption.snapshots["a"]
    if not resident:
        assert snapshot["kv"].keys.device.type == "cpu"
        if mode == "pipeline":
            hidden = [t.hidden for t in snapshot["pipeline"].tokens if t.hidden is not None]
            assert hidden and all(h.device.type == "cpu" for h in hidden)
    final = drain(e)
    assert e.preemption.preemptions == 1 and e.preemption.resumptions == 1
    assert final["a"].token_ids == greedy_reference(reference, PROMPT, A_TOKENS)
    assert final["b"].token_ids == greedy_reference(reference, OTHER, 4)
    assert not e.preemption.snapshots and not e.speculative_runner.pipeline_states
    assert e.cache_manager.num_used_blocks == 0
