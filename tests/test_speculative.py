"""Fixed-depth oracle, rollback, lifecycle and distribution tests for speculation."""

import warnings
from collections import Counter

import pytest
import torch

from tests.helpers import LocalLLM as LLM
from tests.helpers import tiny_ouro_config
from tests.reference import dense_reference
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.execution import WorkKey
from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request
from vllm_rlt.worker.sampler import Sampler
from vllm_rlt.worker.speculative import greedy_accept, rejection_sample


def model(seed=123, dtype=torch.float32):
    torch.manual_seed(seed)
    return OuroForCausalLM(tiny_ouro_config()).to(dtype=dtype)


def engine(m, k=3, **kwargs):
    return LLMEngine(m, speculative_config=SpeculativeConfig(k), **kwargs)


def drain(e):
    final = {}
    for _ in range(2000):
        if not e.has_unfinished_requests():
            return final
        for output in e.step():
            if output.finished:
                final[output.request_id] = output
    pytest.fail("speculative scheduler did not drain")


@pytest.mark.parametrize("generation,supports_packed", [(2, True), (4, False)])
def test_speculative_prefill_routes_by_declared_capabilities(generation, supports_packed):
    from types import SimpleNamespace

    from vllm_rlt.attention import BackendCapabilities

    e = engine(model())
    e.cache_manager.attention = SimpleNamespace(
        capabilities=BackendCapabilities(packed_prefill=supports_packed)
    )
    e.cache_manager.attention_info = {"generation": generation}
    assert e.speculative_runner._packed(True) is supports_packed
    assert e.speculative_runner._packed(False) is False


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("budget", [1, 3, 32])
def test_greedy_matches_native_with_ragged_rounds_and_limits(k, budget):
    m = model()
    prompts = [[2], [3, 4, 5, 6, 7], [9, 3]]
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in [1, 13, 8]]
    expected = LLM(m).generate(prompts, params)
    llm = LLM(
        m,
        speculative_config=SpeculativeConfig(k),
        cache_config=CacheConfig(256, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=budget, prefill_chunk_size=2),
    )
    actual = llm.generate(prompts, params)
    assert [o.token_ids for o in actual] == [o.token_ids for o in expected]
    assert [o.exit_depths for o in actual] == [o.exit_depths for o in expected]
    assert llm.engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", "torch"),
        pytest.param("cuda", "triton", marks=pytest.mark.gpu),
        pytest.param("cuda", "flash_attn_4", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reused_hidden_logits_and_all_kv_match_serial_oracle(dtype, device, backend):
    if device == "cuda" and dtype == torch.float32 and backend == "flash_attn_4":
        pytest.skip("FA4 requires 16-bit operands")
    m = model(dtype=dtype).to(device)
    if backend == "flash_attn_4":
        torch.manual_seed(123)
        m = OuroForCausalLM(tiny_ouro_config(hidden_size=256, head_dim=64)).to(
            device=device, dtype=dtype
        )
    e = engine(m, k=3, cache_config=CacheConfig(128, 2), attention_backend=backend)
    cache = e.cache_manager
    prefix = [3, 4, 5]
    current = 7
    assert cache.allocate("r", 16)
    # Native token-by-token full-depth prefix.
    for pos, token in enumerate(prefix):
        h = m.prelude(torch.tensor([token], device=device))
        for d in range(4):
            h, _ = m.recurrent(h, ["r"], [d], [pos], cache)
    r = Request("r", prefix, SamplingParams(ignore_eos=True), generated_token_ids=[current])
    runner = e.speculative_runner
    core = m.recurrent_prepared
    seen = {}
    calls_by_depth = Counter()

    def record(hidden, batch, cache, **kwargs):
        out, gate = core(hidden, batch, cache, **kwargs)
        for row, (_, depth, pos) in enumerate(batch.rows):
            calls_by_depth[depth] += 1
            seen[depth, pos] = out[row].clone()
        return out, gate

    m.recurrent_prepared = record
    head_outputs = []
    hook = m.lm_head.register_forward_hook(
        lambda module, inputs, output: head_outputs.append(output.detach().clone())
    )
    try:
        runner.submit(
            SchedulerOutput([ScheduledRequest(r, WorkKey.SPECULATIVE, 3, 4)])
        ).collect()
    finally:
        hook.remove()
        m.recurrent_prepared = core
    # Include the actual batched LM-head GEMM in the numeric comparison.
    verified_logits = head_outputs[-1]
    # The shallow head outputs determine each subsequent input in this round.
    tokens = prefix + [current]
    for pos in range(3, 6):
        tokens.append(int(m.coda(seen[1, pos]).argmax()))
    oracle = LLMEngine(m, cache_config=CacheConfig(128, 2), attention_backend=backend).cache_manager
    oracle.allocate("r", 16)
    atol, rtol = (4e-5, 4e-5) if dtype == torch.float32 else (0.08, 0.04)
    max_logit_error = 0.0
    for pos, token in enumerate(tokens):
        h = m.prelude(torch.tensor([token], device=device))
        for d in range(4):
            h, _ = m.recurrent(h, ["r"], [d], [pos], oracle)
            if pos >= 3:
                torch.testing.assert_close(seen[d, pos], h[0], atol=atol, rtol=rtol)
        if pos >= 3:
            actual, expected = verified_logits[pos - 3], m.coda(h)[0]
            max_logit_error = max(max_logit_error, float((actual - expected).abs().max()))
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            assert actual.argmax() == expected.argmax()
    for d in range(4):
        for layer in range(m.config.num_hidden_layers):
            for actual, expected in zip(
                cache.read(layer, "r", d, 7), oracle.read(layer, "r", d, 7)
            ):
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    # Every input/depth computed exactly once: no shallow recomputation in verify.
    assert calls_by_depth == Counter({0: 4, 1: 4, 2: 4, 3: 4})
    if dtype == torch.float32:
        independent = dense_reference(m, tokens, 4)
        for d in range(4):
            torch.testing.assert_close(
                torch.stack([seen[d, p] for p in range(3, 7)]),
                independent[d][0][3:],
                atol=4e-5,
                rtol=4e-5,
            )
    assert max_logit_error < (1e-4 if dtype == torch.float32 else 0.025)


@pytest.mark.parametrize("rejected", [0, 1, 2, 3])
def test_each_rejection_position_and_bonus_commit_correct_frontier(rejected, monkeypatch):
    m = model()
    e = engine(m, k=3)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    request = e.scheduler.requests["r"]
    start = request.position
    # Force draft IDs 10,11,12 and target accept up to the selected boundary.
    coda = m.coda
    calls = 0

    def controlled(h):
        nonlocal calls
        logits = torch.full((len(h), m.config.vocab_size), -100.0)
        if calls < 3:
            logits[:, 10 + calls] = 100
        else:
            for i in range(4):
                logits[i, 10 + i if i < rejected else 20 + i] = 100
        calls += 1
        return logits

    monkeypatch.setattr(m, "coda", controlled)
    out = e.step()[0]
    n = rejected + 1
    assert out.token_ids[-n:] == list(range(10, 10 + rejected)) + [20 + rejected]
    for plane in e.cache_manager._get_allocation("r").written:
        assert all(w.prefix == start + n and not w.pending for w in plane)
    monkeypatch.setattr(m, "coda", coda)
    # The next round must overwrite stale rejected slots and agree with a fresh
    # native engine on the committed prefix, including when the page is partial.
    expected = LLM(m).generate(
        [out.prompt_token_ids + out.token_ids],
        SamplingParams(max_tokens=12 - len(out.token_ids), ignore_eos=True),
    )[0]
    final = drain(e)["r"]
    assert final.token_ids[len(out.token_ids) :] == expected.token_ids


@pytest.mark.parametrize("k", [0, 1, 2, 4, 8])
def test_greedy_accept_commits_target_prefix_through_first_mismatch(k):
    candidates = list(range(100, 100 + k))
    for rejected in range(k + 1):
        # Rows before `rejected` agree; row `rejected` is the correction, or the bonus.
        targets = candidates[:rejected] + [7] + list(range(200, 200 + k - rejected))
        tokens, accepted = greedy_accept(candidates, targets)
        assert accepted == rejected
        assert tokens == candidates[:rejected] + [7]


def test_eos_in_accepted_prefix_never_delivers_following_tokens(monkeypatch):
    m = model()
    e = engine(m)
    e.add_request("r", [2, 3], SamplingParams(max_tokens=12))
    while not e.step():
        pass
    # All later proposals/target rows choose EOS. Only first EOS is committed.
    monkeypatch.setattr(m, "coda", lambda h: torch.zeros(len(h), m.config.vocab_size))
    out = e.step()[0]
    assert out.finished and out.finish_reason == "stop"
    assert len(out.token_ids) == 2 and out.token_ids[-1] == 0
    assert e.speculative_runner.stats.committed_tokens == 1
    assert e.cache_manager.num_used_blocks == 0


def test_truncation_removes_sparse_validity_without_destroying_prefix():
    e = engine(model(), cache_config=CacheConfig(64, 2))
    cache = e.cache_manager
    cache.allocate("r", 8)
    k = torch.randn(4, 2, 8)
    # Initialize [0,1,2] and a sparse stale position 6 at every layer/depth.
    for d in range(4):
        for layer in range(2):
            cache.write(layer, ["r"] * 4, [d] * 4, [0, 1, 2, 6], k, k)
    before = cache.read(0, "r", 0, 2)[0].clone()
    cache.truncate_suffix("r", 2)
    torch.testing.assert_close(cache.read(0, "r", 0, 2)[0], before, rtol=0, atol=0)
    for plane in cache._get_allocation("r").written:
        assert all(w.prefix == 2 and not w.pending for w in plane)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.read(0, "r", 0, 3)


@pytest.mark.parametrize("prefix_cache,incremental", [(False, True), (True, False), (True, True)])
def test_prefix_replay_incremental_growth_cancel_and_reuse(prefix_cache, incremental):
    m = model()
    kwargs = dict(
        cache_config=CacheConfig(
            128, 2, enable_prefix_caching=prefix_cache, incremental_allocation=incremental
        )
    )
    llm = LLM(m, speculative_config=SpeculativeConfig(4), **kwargs)
    params = SamplingParams(max_tokens=9, ignore_eos=True)
    prompts = [[2, 3, 4, 5, 6], [2, 3, 4, 5, 7]]
    expected = LLM(m).generate(prompts, params)
    for _ in range(2):
        assert [o.token_ids for o in llm.generate(prompts, params)] == [
            o.token_ids for o in expected
        ]
    e = llm.engine
    e.add_request("reuse", prompts[0], params)
    while not e.step():
        pass
    e.step()
    assert e.abort_request("reuse").finish_reason == "abort"
    e.add_request("reuse", prompts[1], params)
    assert drain(e)["reuse"].token_ids == expected[1].token_ids
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("top_k,top_p", [(-1, 1.0), (5, 1.0), (7, 0.8)])
def test_sampling_seeded_replay_and_request_isolation(top_k, top_p):
    m = model()
    prompts = [[2, 3], [7, 8, 9]]
    params = SamplingParams(
        max_tokens=10, temperature=0.8, top_k=top_k, top_p=top_p, ignore_eos=True, seed=42
    )

    def run(prompts):
        return LLM(m, speculative_config=SpeculativeConfig(3)).generate(prompts, params)

    batched = run(prompts)
    assert [o.token_ids for o in run(prompts)] == [o.token_ids for o in batched]
    assert [run([p])[0].token_ids for p in prompts] == [o.token_ids for o in batched]


def test_both_runners_share_the_engine_rng_registry():
    # The registry belongs to the engine because LLMEngine builds SpeculativeRunner
    # before ModelRunner and passes neither to the other. If the two runners held
    # separate registries, preemption and termination would stop covering the
    # speculative path's RNG.
    e = engine(model())
    assert e.model_runner.rng is e.rng
    assert e.speculative_runner.rng is e.rng

    plain = LLMEngine(model())
    assert plain.model_runner.rng is plain.rng
    assert plain.speculative_runner is None
    # The asynchronous configuration holds AsyncSpeculativeRunner instead; its
    # constructor allocates CUDA streams, so that case is covered by
    # test_async_speculative.py::test_async_speculative_engine_shares_its_registry.


def test_acquire_generator_uses_the_logits_device(monkeypatch):
    # The speculative helper must hand acquire() the logits row's device, not the
    # model's. A different runner device makes the two sources distinguishable on a
    # CPU-only machine; the spy returns a real CPU generator so the fake device is
    # never allocated.
    e = engine(model())
    runner = e.speculative_runner
    runner.device = torch.device("meta")
    recorded = []
    real_acquire = e.rng.acquire

    def spy(request_id, params, device):
        recorded.append(device)
        return real_acquire(request_id, params, torch.device("cpu"))

    monkeypatch.setattr(e.rng, "acquire", spy)
    logits = torch.linspace(1.0, -1.0, 8)
    request = Request("r", [2, 3], SamplingParams(temperature=0.7, seed=5))
    generator = runner._acquire_generator(request, logits)
    assert recorded == [logits.device]
    assert runner.device != logits.device
    assert generator.device == logits.device


def test_speculative_greedy_execution_never_acquires_a_generator(monkeypatch):
    # The CODA path has the same check in test_engine.py; this one proves the
    # contract on a request that really executes through WorkKey.SPECULATIVE, so a
    # generator that is created but unused would still be caught.
    e = engine(model())

    def forbidden(*args, **kwargs):
        raise AssertionError("greedy speculative sampling must not acquire a generator")

    monkeypatch.setattr(e.rng, "acquire", forbidden)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=6, ignore_eos=True))
    stages = []
    for _ in range(2000):
        if not e.has_unfinished_requests():
            break
        e.step()
        stages.append(e.last_schedule.works)
    else:
        pytest.fail("speculative engine did not drain")
    assert any(WorkKey.SPECULATIVE in works for works in stages)
    assert e.rng.peek("r") is None


def test_speculative_path_advances_the_registry_generator():
    # Sharing the registry is not enough: the speculative path must actually draw
    # from the stored generator. Rebuilding or reseeding one per call would keep
    # the slot frozen and silently change every sampled token after the first.
    e = engine(model())
    params = SamplingParams(
        max_tokens=24, temperature=0.8, top_k=7, top_p=0.8, seed=42, ignore_eos=True
    )
    e.add_request("r", [2, 3, 4], params)

    def step_until_speculative():
        for _ in range(10):
            e.step()
            if WorkKey.SPECULATIVE in e.last_schedule.works:
                return True
        return False

    assert step_until_speculative()
    generator = e.rng.peek("r")
    assert generator is not None
    state = generator.get_state().clone()

    assert step_until_speculative()
    assert e.has_unfinished_requests()
    assert e.rng.peek("r") is generator
    assert not torch.equal(generator.get_state(), state)


@pytest.mark.parametrize(
    "p,q",
    [
        ([0.1, 0.6, 0.3], [0.7, 0.2, 0.1]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
    ],
)
def test_rejection_sampling_recovers_target_distribution(p, q):
    p, q = torch.tensor(p), torch.tensor(q)
    g = torch.Generator().manual_seed(431)
    counts = torch.zeros(3)
    for _ in range(12000):
        candidate = int(torch.multinomial(q, 1, generator=g))
        token, _ = rejection_sample(candidate, p, q, g)
        counts[token] += 1
    torch.testing.assert_close(counts / counts.sum(), p, atol=0.016, rtol=0)


def test_probability_filters_match_expected_and_apply_strict_top_k():
    logits = torch.tensor([0.50, 0.25, 0.15, 0.10]).log()
    p = Sampler.probabilities(logits, SamplingParams(temperature=1, top_p=0.8))
    torch.testing.assert_close(p, torch.tensor([0.50, 0.25, 0.15, 0]) / 0.9)
    p = Sampler.probabilities(logits, SamplingParams(temperature=0.5, top_k=2))
    torch.testing.assert_close(p, torch.tensor([0.8, 0.2, 0, 0]))
    # All four logits are tied, so the threshold mask used to keep all of them
    # (a uniform 0.25). Strict top-k keeps exactly one; which index survives a tie
    # is not guaranteed across devices, so only the count and weight are asserted.
    p = Sampler.probabilities(torch.zeros(4), SamplingParams(temperature=1, top_k=1))
    assert int((p > 0).sum()) == 1
    torch.testing.assert_close(p[p > 0], torch.ones(1))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cache_config=CacheConfig(layout="shared")),
        dict(execution_config=ExecutionConfig(async_scheduling=True)),
        dict(scheduler_config=SchedulerConfig(enable_preemption=True)),
        dict(scheduler_config=SchedulerConfig(mode="no_refill")),
    ],
)
def test_unsupported_modes_fail_before_allocating(kwargs):
    with pytest.raises(ValueError, match="speculative"):
        engine(model(), **kwargs)


def test_rejects_adaptive_or_wrong_target_requests():
    e = engine(model())
    for params in [SamplingParams(exit_threshold=0.5), SamplingParams(max_loops=3)]:
        with pytest.raises(ValueError, match="fixed target depth"):
            e.add_request("bad", [2], params)
    assert not e.has_unfinished_requests()


def test_dynamic_arrival_budget_and_failure_reclamation(monkeypatch):
    e = engine(model(), scheduler_config=SchedulerConfig(max_num_batched_tokens=5))
    params = SamplingParams(max_tokens=9, ignore_eos=True)
    e.add_request("a", [2, 3], params)
    while not e.step():
        pass
    e.add_request("b", [4, 5, 6], params)
    for _ in range(4):
        e.step()
        assert e.last_schedule.num_tokens <= 5

    def fail(*args, **kwargs):
        raise RuntimeError("injected verify failure")

    monkeypatch.setattr(e.speculative_runner, "submit", fail)
    with pytest.raises(RuntimeError, match="injected"):
        while e.has_unfinished_requests():
            e.step()
    for rid in list(e.scheduler.requests):
        e.abort_request(rid)
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        SpeculativeConfig(k)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn_4"])
def test_gpu_sampling_and_greedy_match_replay_with_ragged_requests(backend):
    torch.manual_seed(123)
    m = OuroForCausalLM(tiny_ouro_config(hidden_size=256, head_dim=64)).to(
        device="cuda", dtype=torch.bfloat16
    )
    prompts = [[2, 3, 4], [7, 8], [9]]
    params = [
        SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.8, top_k=7, top_p=0.8, seed=42)
        for n in [8, 5, 3]
    ]

    def run(speculative, sampling):
        return LLM(
            m,
            speculative_config=SpeculativeConfig(3) if speculative else None,
            cache_config=CacheConfig(128, 16),
            attention_backend=backend,
        ).generate(prompts, sampling)

    a, b = run(True, params), run(True, params)
    assert [o.token_ids for o in a] == [o.token_ids for o in b]
    greedy = [SamplingParams(max_tokens=n, ignore_eos=True) for n in [8, 5, 3]]
    a, b = run(True, greedy), run(False, greedy)
    assert [o.token_ids for o in a] == [o.token_ids for o in b]


@pytest.mark.gpu
def test_greedy_round_reads_back_once_with_ragged_requests():
    # Draft IDs stay on the device; every speculative round, including ragged and
    # budget-shortened ones, synchronizes with the host exactly once.
    torch.manual_seed(123)
    m = OuroForCausalLM(tiny_ouro_config(hidden_size=256, head_dim=64)).to(
        device="cuda", dtype=torch.bfloat16
    )
    e = engine(m, k=3, cache_config=CacheConfig(128, 16), attention_backend="triton")
    for rid, prompt, n in [("a", [2, 3, 4], 9), ("b", [7, 8], 6), ("c", [9], 4)]:
        e.add_request(rid, prompt, SamplingParams(max_tokens=n, ignore_eos=True))
    rounds = 0
    while e.has_unfinished_requests():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                e.step()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        if WorkKey.SPECULATIVE in e.last_schedule.works:
            rounds += 1
            assert sum("synchroniz" in str(w.message) for w in caught) == 1
    assert rounds > 1


@pytest.mark.gpu
@pytest.mark.parametrize("max_graph_rows", [2, 8])
def test_speculative_cuda_graph_matches_eager_with_ragged_replay(max_graph_rows):
    m = model(dtype=torch.bfloat16).to("cuda")
    prompts = [[2, 3, 4], [7, 8]]
    params = SamplingParams(max_tokens=9, ignore_eos=True)

    def run(graphs):
        llm = LLM(
            m,
            speculative_config=SpeculativeConfig(3),
            cache_config=CacheConfig(128, 16),
            attention_backend="triton",
            execution_config=ExecutionConfig(
                cuda_graphs=graphs, cuda_graph_max_batch_size=max_graph_rows
            ),
        )
        outputs = [llm.generate(prompts, params) for _ in range(2)]
        assert llm.engine.cache_manager.num_used_blocks == 0
        return llm.engine.speculative_runner, [
            [(out.token_ids, out.exit_depths) for out in batch] for batch in outputs
        ]

    _, eager = run(False)
    runner, graphed = run(True)
    assert graphed == eager
    assert runner.graphs.captures > 0
    assert runner.graphs.replays > runner.graphs.captures
    assert runner.coda_graphs.captures > 0
    if max_graph_rows == 2:
        assert runner.graphs.fallbacks > 0
    else:
        assert any(rows > 2 for rows, _, _, _ in runner.graphs.entries)


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
@pytest.mark.parametrize("budget", [2, 7])
def test_split_prefill_and_speculation_share_forward_and_match_atomic(chunk_size, budget):
    from vllm_rlt.core.execution import Operation

    def run(atomic):
        e = engine(
            model(),
            cache_config=CacheConfig(256, 2),
            scheduler_config=SchedulerConfig(
                max_num_batched_tokens=budget,
                prefill_chunk_size=chunk_size,
            ),
            execution_config=ExecutionConfig(atomic_prefill=atomic),
        )
        assert e.speculative_runner.runtime is e.model_runner.runtime
        operations = set()
        forward = e.model_runner.runtime.forward

        def record(batch):
            operations.add(batch.operation)
            return forward(batch)

        e.model_runner.runtime.forward = record
        for i, prompt in enumerate(([2], [3, 4, 5, 6, 7], [9, 3])):
            e.add_request(str(i), prompt, SamplingParams(max_tokens=9, ignore_eos=True))
        results = drain(e)
        assert operations == {Operation.PRELUDE, Operation.RECURRENT, Operation.CODA}
        assert e.cache_manager.num_used_blocks == 0
        return {key: (value.token_ids, value.exit_depths) for key, value in results.items()}

    assert run(False) == run(True)


@pytest.mark.parametrize("atomic", [False, True])
def test_common_submit_entry_preserves_prefill_and_speculative_results(atomic, monkeypatch):
    e = engine(model(), execution_config=ExecutionConfig(atomic_prefill=atomic))
    calls = []
    submit = e.executor.submit

    def record(batch):
        calls.append((batch.works, batch.is_speculative))
        return submit(batch)

    monkeypatch.setattr(e.executor, "submit", record)
    params = SamplingParams(max_tokens=9, ignore_eos=True)
    e.add_request("r", [2, 3, 4, 5], params)
    actual = drain(e)["r"]
    expected = LLM(model()).generate([[2, 3, 4, 5]], params)[0]
    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert any(flag for _, flag in calls)
    assert any(not flag for _, flag in calls)
    assert all(flag == (works == frozenset((WorkKey.SPECULATIVE,))) for works, flag in calls)
    assert e.cache_manager.num_used_blocks == 0


def test_common_submit_rejects_speculation_without_controller():
    e = LLMEngine(model())
    request = Request("r", [2], SamplingParams())
    batch = SchedulerOutput([ScheduledRequest(request, WorkKey.SPECULATIVE, 0, 1)])
    with pytest.raises(ValueError, match="configured round controller"):
        e.executor.submit(batch).collect()


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_speculative_submit_defers_collection_and_collects_once(monkeypatch, temperature):
    e = engine(model())
    original = e.speculative_runner._collect_round
    collections = []

    def collect(submission):
        collections.append(submission)
        return original(submission)

    monkeypatch.setattr(e.speculative_runner, "_collect_round", collect)
    e.add_request(
        "r", [2, 3, 4], SamplingParams(max_tokens=9, ignore_eos=True, temperature=temperature)
    )
    try:
        while e.scheduler.pending_work(e.scheduler.requests["r"]) != WorkKey.SPECULATIVE:
            e.step()
        batch = e._schedule()
        ticket = e.executor.submit(batch)
        assert not collections
        assert ticket.has_result
        result = ticket.collect()
        assert ticket.collect() is result
        assert collections == [ticket]
        e._update(batch, result)
        assert len(drain(e)["r"].token_ids) == 9
    finally:
        e.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("use_uva", [False, True])
def test_speculative_inputs_reuse_owned_buffers_without_private_cache_staging(
    monkeypatch, device, use_uva
):
    e = engine(model().to(device), execution_config=ExecutionConfig(use_uva=use_uva))
    params = SamplingParams(max_tokens=30, ignore_eos=True)
    e.add_request("r", [2, 3, 4], params)
    try:
        while e.scheduler.pending_work(e.scheduler.requests["r"]) != WorkKey.SPECULATIVE:
            e.step()
        batch = e._schedule()
        runner = e.speculative_runner

        def forbidden(*args, **kwargs):
            raise AssertionError("runner must own staging, not allocate through the cache")

        monkeypatch.setattr(e.cache_manager, "_stage", forbidden)
        pointers = []
        for _ in range(4):
            ticket = runner.submit(batch)
            result = ticket.collect()
            assert result[0].token_ids
            pointers.append(
                {
                    dtype: tuple(
                        target.data_ptr() for target in buffer.targets if target is not None
                    )
                    for dtype, buffer in runner.input_buffers.items()
                }
            )
        assert pointers[1] == pointers[2] == pointers[3]
        e._update(batch, result)
        assert len(drain(e)["r"].token_ids) == 30
    finally:
        e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("use_uva", [False, True])
def test_speculative_input_slots_survive_three_submits_before_collection(use_uva):
    e = engine(model().cuda(), execution_config=ExecutionConfig(use_uva=use_uva))
    try:
        for i in range(3):
            e.add_request(str(i), [i + 2, i + 3], SamplingParams(max_tokens=12, ignore_eos=True))
        while any(
            e.scheduler.pending_work(request) != WorkKey.SPECULATIVE
            for request in e.scheduler.requests.values()
        ):
            e.step()
        batch = e._schedule()
        assert len(batch.requests) == 3
        batches = [SchedulerOutput([span]) for span in batch.requests]
        runner = e.speculative_runner
        expected = [runner.submit(part).collect() for part in batches]
        # The third submission wraps the two input slots before any host collection.
        # Only completion of their GPU readers permits the slots to be recycled.
        tickets = [runner.submit(part) for part in batches]
        assert [ticket.collect() for ticket in tickets] == expected
    finally:
        e.close()
