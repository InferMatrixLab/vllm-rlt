"""Rollout outputs: per-token logprobs, effective seeds and stop token IDs."""

import gc
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.helpers import tiny_ouro_config
from tests.reference import dense_reference
from tests.test_huginn import tiny_huginn_config
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.config import SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import HuginnForCausalLM, OuroForCausalLM
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine, Peer
from vllm_rlt.request import Operation, RequestOutput, RequestPhase
from vllm_rlt.serving.protocol import CompletionRequest
from vllm_rlt.worker.logprobs import token_logprob
from vllm_rlt.worker.model_runner import ReadbackSlot, Submission
from vllm_rlt.worker.output import ModelRunnerOutput, SpeculativeResult
from vllm_rlt.worker.sampler import Sampler

TRACES = {"a": [4, 1, 3, 2, 4, 1, 2, 3], "b": [4, 2, 1, 4, 3, 2, 1, 4]}  # Keyed by request ID.
EXIT_MODES = [  # (async scheduling, exit config, exit_threshold) for every CPU exit policy.
    pytest.param(False, ExitConfig("ouro"), 1.0, id="ouro-full-depth"),
    pytest.param(False, ExitConfig("ouro"), 0.5, id="ouro-early-exit"),
    pytest.param(False, ExitConfig("trace", depths_by_request=TRACES), 1.0, id="trace"),
    pytest.param(True, ExitConfig("ouro_delayed"), 0.3, id="async-ouro_delayed"),
    pytest.param(True, ExitConfig("random_lookahead"), 0.3, id="async-random_lookahead"),
    pytest.param(True, ExitConfig("trace", depths_by_request=TRACES), 1.0, id="async-trace"),
]
ASYNC_EXIT_MODES = [pytest.param(*p.values[1:], id=p.id) for p in EXIT_MODES if p.values[0]]
LOGPROBS_MODES = ["raw_logprobs", "processed_logprobs"]
# How request "a" ends: max_tokens, EOS (honored or ignored), stop IDs, both, abort.
FINISHES = "length eos stop_id ignored_eos eos_with_stop_ids stop_id_at_max eos_at_max abort"


def model(seed=123, device="cpu", dtype=torch.float32, **overrides):
    torch.manual_seed(seed)
    return OuroForCausalLM(tiny_ouro_config(**overrides)).to(device, dtype)


def rollout(**overrides):
    return SamplingParams(**{"ignore_eos": True, "logprobs": 0, **overrides})


def make_engine(m, *, asynchronous=False, exit_config=None, execution=None, **kwargs):
    """Small CPU engine by default; asynchronous engines default to ouro_delayed exits."""
    kwargs.setdefault("cache_config", CacheConfig(128, 2))
    kwargs.setdefault("scheduler_config", SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=3))
    execution = ExecutionConfig(async_scheduling=asynchronous, **(execution or {}))
    exit_config = exit_config or ExitConfig("ouro_delayed" if asynchronous else "ouro")
    return LLMEngine(m, execution_config=execution, exit_config=exit_config, **kwargs)


def check_aligned(output):
    n, logprobs = len(output.token_ids), output.logprobs
    assert len(output.exit_depths) == n
    assert (logprobs is None) == (output.sampling_params.logprobs is None)
    assert logprobs is None or len(logprobs) == n
    assert all(math.isfinite(value) and value <= 0 for value in logprobs or ())


def generate(engine, prompts, params):
    """``LLM.generate`` on an in-process engine, whose runners tests can patch (``LLM``
    runs its engine in a stage worker process)."""
    params = params if isinstance(params, list) else [params] * len(prompts)
    for i, (prompt, request_params) in enumerate(zip(prompts, params)):
        engine.add_request(str(i), prompt, request_params)
    outputs = final(stream(engine))
    return [outputs[str(i)] for i in range(len(prompts))]


def stream(engine, until=None, streamed=None):
    """Step until ``until()`` holds, else drain, checking every output's alignment. A drained
    LLMEngine (1000-step limit) holds zero used KV blocks; PD gets a 90 s wall-clock limit."""
    streamed = defaultdict(list) if streamed is None else streamed
    done = until or (lambda: not engine.has_unfinished_requests())
    local, deadline, steps = isinstance(engine, LLMEngine), time.monotonic() + 90, 0
    while not done():
        assert engine.has_unfinished_requests()
        assert (steps < 1000) if local else (time.monotonic() < deadline)
        steps += 1
        for output in engine.step():
            check_aligned(output)
            streamed[output.request_id].append(output)
    if until is None and local:
        assert engine.cache_manager.num_used_blocks == 0
    return streamed


def final(streamed):
    return {rid: outputs[-1] for rid, outputs in streamed.items()}


def assert_same(actual, expected, atol=None):
    """Same tokens and exit depths; logprobs equal, or within ``atol`` when given."""
    assert (actual.token_ids, actual.exit_depths) == (expected.token_ids, expected.exit_depths)
    logprobs = expected.logprobs
    if atol is not None and logprobs is not None:
        logprobs = pytest.approx(logprobs, abs=atol)
    assert actual.logprobs == logprobs


def hold_coda_delivery(monkeypatch):
    """Report CODA tickets unfinished, so only the one-output bound delivers them."""
    ready = Submission.ready
    monkeypatch.setattr(Submission, "ready", lambda t: t.operation != Operation.CODA and ready(t))
    return ready


def lease_cpu_logprob_slots(monkeypatch, engine, count=8):
    """Return the logprob readback pool. On CPU, which has none, install ``count`` emulated
    slots and monkeypatch ``runner.submit`` to lease one per logprob ticket (CUDA: unchanged)."""
    runner, submit = engine.model_runner, engine.model_runner.submit
    if runner.device.type == "cuda":
        return runner.logprob_slots
    runner.logprob_slots = [ReadbackSlot(torch.empty(1)) for _ in range(count)]

    def leasing_submit(*args, **kwargs):
        ticket = submit(*args, **kwargs)
        if ticket.logprobs is not None:
            ticket.logprob_slot = runner._readback_slot(ticket.logprobs, runner.logprob_slots)
        return ticket

    monkeypatch.setattr(runner, "submit", leasing_submit)
    return runner.logprob_slots


def test_valid_rollout_params_are_normalized_and_hashable():
    params = SamplingParams(seed=None, logprobs=0, stop_token_ids=[5, 3, 5])
    assert (params.seed, params.logprobs, params.stop_token_ids) == (None, 0, (5, 3, 5))
    assert {params} == {replace(params, stop_token_ids=(5, 3, 5))}
    assert SamplingParams().logprobs is None and SamplingParams(seed=2**63 - 1).seed == 2**63 - 1


def rejected(field, match, *values):
    return [pytest.param(field, value, match, id=f"{field}={value!r}") for value in values]


@pytest.mark.parametrize(
    "field,value,match",
    rejected("logprobs", "not supported yet", 1, 5)
    + rejected("logprobs", "None or 0", -1, True, 0.0, "0")
    + rejected("seed", "seed", True, -1, 1.5, "1")
    + [pytest.param("seed", 2**63, "seed", id="seed=2**63")]
    + rejected("stop_token_ids", "stop_token_ids", None, 3, "3", [True], [1.0], [-1], {3}),
)
def test_invalid_rollout_params_are_rejected(field, value, match):
    with pytest.raises(ValueError, match=match):
        SamplingParams(**{field: value})


def test_logprobs_mode_is_validated_and_forwarded_to_stage_workers(monkeypatch):
    # LLM and PDEngine run StageEngine in worker processes; capture what they would start.
    start = Mock(side_effect=RuntimeError("stop before workers"))
    monkeypatch.setattr("vllm_rlt.engine.llm_engine.start_stage_worker", start)
    for logprobs_mode in LOGPROBS_MODES:
        assert LLMEngine(model(), logprobs_mode=logprobs_mode).model_runner.logprobs_mode == (
            logprobs_mode
        )
        for build, source in ((LLM, model), (PDEngine, tiny_ouro_config)):
            with pytest.raises(RuntimeError, match="stop before workers"):
                build(source(), logprobs_mode=logprobs_mode)
    for logprobs_mode in ("raw_logits", "processed_logits", "logprobs", None):
        for build, source in ((LLMEngine, model), (LLM, model), (PDEngine, tiny_ouro_config)):
            with pytest.raises(ValueError, match="not supported"):
                build(source(), logprobs_mode=logprobs_mode)
    # Valid modes reach the worker engines (LLM, then PD); invalid ones start no worker.
    forwarded = [c.args[3]["engine"]["logprobs_mode"] for c in start.call_args_list]
    assert forwarded == [mode for mode in LOGPROBS_MODES for _ in range(2)]


def test_admission_rejects_out_of_vocabulary_stop_ids():
    engine = LLMEngine(model())
    for stop in ([64], [3, 1000]):  # The tiny vocabulary is IDs 0..63.
        with pytest.raises(ValueError, match="stop_token_ids"):
            engine.add_request("bad", [1], SamplingParams(stop_token_ids=stop))
    assert not engine.has_unfinished_requests()
    engine.add_request("ok", [1], SamplingParams(max_tokens=2, stop_token_ids=[63], logprobs=0))
    assert final(stream(engine))["ok"].finished


def test_engine_rejects_missing_logprob_for_requesting_row():
    engine = LLMEngine(model())
    engine.add_request("r", [2], SamplingParams(logprobs=0))
    with pytest.raises(RuntimeError, match="missing logprob"):
        engine.scheduler._record_token(engine.scheduler.requests["r"], 3, 4, None)


def test_stage_client_round_trips_rollout_outputs():
    """LLM runs StageEngine in a worker process: outputs cross the pipe with their logprobs
    and effective params, an abort before the first sample included, and replay exactly."""
    params = rollout(max_tokens=6, temperature=0.9, top_k=8, seed=None)
    with LLM(model(), logprobs_mode="processed_logprobs") as llm:
        client = llm.engine
        client.add_request("early", [2, 3], params)
        aborted = client.abort_request("early")
        check_aligned(aborted)
        assert aborted.finish_reason == "abort" and type(aborted.sampling_params.seed) is int
        outputs = llm.generate([[2, 3], [4]], params)
        for output in outputs:
            check_aligned(output)
            assert output.sampling_params == replace(params, seed=output.sampling_params.seed)
        assert outputs[0].sampling_params.seed != outputs[1].sampling_params.seed
        replay = llm.generate([[2, 3], [4]], [output.sampling_params for output in outputs])
        for actual, expected in zip(replay, outputs):
            assert_same(actual, expected)
        # A stop ID ends the request in the worker; the stop token keeps its logprob.
        stop = outputs[0].token_ids[0]
        params = replace(outputs[0].sampling_params, stop_token_ids=[stop])
        (stopped,) = llm.generate([[2, 3]], params)
        assert (stopped.finish_reason, stopped.token_ids) == ("stop", [stop])
        assert stopped.logprobs == outputs[0].logprobs[:1]


def test_http_request_fields_are_unchanged():
    # tests/test_serving.py covers the other rollout fields.
    for value in (0, True):
        with pytest.raises(ValueError, match="logprobs"):
            CompletionRequest.parse({"model": "m", "prompt": "a", "logprobs": value}, "m")
    assert CompletionRequest.parse({"model": "m", "prompt": "a", "seed": 7}, "m").params.seed == 7


@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODES)
@pytest.mark.parametrize("finish", FINISHES.split())
@pytest.mark.parametrize("asynchronous,exit_config,threshold", EXIT_MODES)
def test_logprobs_are_defined_and_aligned_through_each_finish(
    monkeypatch, asynchronous, exit_config, threshold, finish, logprobs_mode
):
    rows = defaultdict(list)  # CODA logits rows in token order; the baseline runs first.
    common = dict(min_loops=1, exit_threshold=threshold)
    options = dict(asynchronous=asynchronous, exit_config=exit_config, logprobs_mode=logprobs_mode)
    # The unrequested row exercises mixed batches (ignored values). Admitted first, it puts
    # "a" behind row 0 in shared batches, so each row must read its own logprob.
    unrequested = rollout(max_tokens=6, temperature=0.9, seed=5, logprobs=None, **common)

    def run(params, *, eos=0, abort=False):
        engine = make_engine(model(eos_token_id=eos), **options)
        runner = engine.model_runner
        if params.logprobs is None:  # No logprob work when no request asks.
            unexpected = AssertionError("logprobs computed without a request")
            monkeypatch.setattr(runner, "_sample_with_logprobs", Mock(side_effect=unexpected))

        def recording(sample):
            def record(row, row_params, generator):
                if row_params.seed == params.seed:  # Request "a"; "b" has seed 5.
                    rows["a"].append(row.detach().clone())
                return sample(row, row_params, generator)

            return record

        # Processed top-k/top-p rows that request logprobs draw through sample_with_logprob.
        for name in ("sample", "sample_with_logprob"):
            setattr(runner.sampler, name, recording(getattr(runner.sampler, name)))
        engine.add_request("b", [4], unrequested)
        engine.add_request("a", [2, 3], replace(params, **common))
        if not abort:
            return stream(engine)
        request = engine.scheduler.requests["a"]
        streamed = stream(engine, lambda: len(request.generated_token_ids) >= 2)
        streamed["a"].append(engine.abort_request("a"))
        return stream(engine, streamed=streamed)

    params = rollout(max_tokens=6, temperature=1.0, top_k=20, top_p=0.9, seed=7)
    baseline = final(run(params))["a"]
    tokens = baseline.token_ids
    assert baseline.finish_reason == "length" and len(rows["a"]) == len(tokens) == 6
    # Raw scores the token's CODA row; processed, the distribution the sampler draws from.
    processed = logprobs_mode == "processed_logprobs"
    expected = [
        (Sampler.probabilities(row, params).log() if processed else row.float().log_softmax(-1))[t]
        for row, t in zip(rows["a"], tokens)
    ]
    assert baseline.logprobs == pytest.approx([value.item() for value in expected], abs=1e-6)
    # Every policy except fixed full depth must exercise shallow exits here.
    shallow = threshold < 1 or exit_config.mode == "trace"
    assert any(depth < 4 for depth in baseline.exit_depths) == shallow
    # Asking for logprobs changes no token or exit depth.
    quiet = final(run(replace(params, logprobs=None)))["a"]
    assert (quiet.token_ids, quiet.exit_depths) == (tokens, baseline.exit_depths)
    # Stop on the first token that is new at index >= 2, so a stop must cut the stream.
    index = next(i for i, t in enumerate(tokens) if i >= 2 and t not in tokens[:i])
    stop, other = tokens[index], next(t for t in range(64) if t not in tokens)
    # finish -> (EOS ID, overrides, reason, length); stop IDs, unlike EOS, override ignore_eos.
    honor, cut = dict(ignore_eos=False), dict(max_tokens=index + 1)
    cases = {
        "length": (0, {}, "length", 6),
        "eos": (stop, honor, "stop", index + 1),
        "stop_id": (0, dict(stop_token_ids=[other, stop]), "stop", index + 1),
        "ignored_eos": (stop, {}, "length", 6),
        "eos_with_stop_ids": (stop, dict(honor, stop_token_ids=[other]), "stop", index + 1),
        "stop_id_at_max": (0, dict(cut, stop_token_ids=[stop]), "stop", index + 1),
        "eos_at_max": (stop, dict(honor, **cut), "stop", index + 1),
        "abort": (0, {}, "abort", None),  # After two delivered tokens.
    }
    eos, overrides, reason, length = cases[finish]
    outputs = run(replace(params, **overrides), eos=eos, abort=finish == "abort")["a"]
    n = len(outputs[-1].token_ids)
    assert outputs[-1].finish_reason == reason and (2 <= n < 6 if length is None else n == length)
    if finish != "abort":  # One token per output; a stop token keeps its depth and logprob.
        assert [len(o.token_ids) for o in outputs] == list(range(1, n + 1))
    for output in outputs:
        # Re-checked after draining: each output must own its lists, not the request's.
        check_aligned(output)
        k = len(output.token_ids)
        assert output.token_ids == tokens[:k] and output.exit_depths == baseline.exit_depths[:k]
        assert output.logprobs == baseline.logprobs[:k]


# Request ID -> overrides covering each processed-logprob path: one batched raw log_softmax
# (greedy, even with top-k, and T == 1), one per other temperature (0.7 twice, 1.3), the
# sampler's top-k/top-p logits, and a row that asks for no logprobs.
MIXED_ROWS = {
    "greedy": dict(temperature=0.0, top_k=3),
    "t1": dict(temperature=1.0),
    "t07": dict(temperature=0.7),
    "t07b": dict(temperature=0.7),
    "t13": dict(temperature=1.3),
    "top_k": dict(temperature=0.9, top_k=8),
    "top_p": dict(temperature=1.0, top_p=0.8),
    "top_k_top_p": dict(temperature=0.7, top_k=16, top_p=0.9),
    "quiet": dict(temperature=0.8, logprobs=None),
}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_grouped_processed_logprobs_equal_the_per_row_reference(monkeypatch, asynchronous):
    mode, built = "processed_logprobs", []  # built: one entry per processed_logits call.
    processed_logits = Sampler.processed_logits

    def counted(logits, params):
        built.append(params)
        return processed_logits(logits, params)

    monkeypatch.setattr(Sampler, "processed_logits", staticmethod(counted))
    batch_sizes = []

    def run(logprobs):
        scheduler = SchedulerConfig(max_num_seqs=9, max_num_batched_tokens=16)
        options = dict(logprobs_mode=mode, scheduler_config=scheduler)
        engine = make_engine(model(), asynchronous=asynchronous, **options)
        runner = engine.model_runner
        batched = runner._sample_with_logprobs

        def checked(logits, params, generators):
            before = len(built)
            token_ids, values = batched(logits, params, generators)
            # Sampling builds each sampled row's processed logits once; logprobs reuse them.
            assert len(built) - before == sum(p.temperature > 0 for p in params)
            for row, row_params in enumerate(params):  # 57d2308 ran token_logprob per row.
                if row_params.logprobs is not None:
                    reference = token_logprob(logits[row], row_params, token_ids[row], mode)
                    assert torch.equal(values[row], reference)
            batch_sizes.append(len(params))
            return token_ids, values

        monkeypatch.setattr(runner, "_sample_with_logprobs", checked)
        for i, (rid, overrides) in enumerate(MIXED_ROWS.items()):
            params = rollout(max_tokens=6, seed=20 + i, exit_threshold=1.0, **overrides)
            engine.add_request(rid, [1 + i], params if logprobs else replace(params, logprobs=None))
        return final(stream(engine))

    grouped, quiet = run(logprobs=True), run(logprobs=False)
    assert max(batch_sizes) == len(MIXED_ROWS)  # Every path shared one CODA batch.
    for rid, output in quiet.items():  # Logprobs change no token, depth or RNG draw.
        assert grouped[rid].token_ids == output.token_ids
        assert grouped[rid].exit_depths == output.exit_depths
        assert (grouped[rid].logprobs is None) == (rid == "quiet")


@pytest.mark.parametrize("layout", ["last_exited", "shared"])
def test_full_depth_raw_logprobs_vs_teacher_forced_reference(layout):
    m = model()
    engine = make_engine(m, cache_config=CacheConfig(128, 2, layout))
    prompts = {"a": [2, 3, 5], "s": [7]}
    engine.add_request("a", prompts["a"], rollout(max_tokens=6))
    engine.add_request("s", prompts["s"], rollout(max_tokens=6, temperature=0.8, seed=3))
    for rid, output in final(stream(engine)).items():
        assert output.exit_depths == [4] * 6
        tokens = prompts[rid] + output.token_ids
        rows = dense_reference(m, tokens, 4)[-1][2][len(prompts[rid]) - 1 : -1]
        expected = rows.float().log_softmax(-1)[torch.arange(6), torch.tensor(output.token_ids)]
        difference = (torch.tensor(output.logprobs) - expected).abs().max().item()
        # Shared KV reads earlier tokens' final-loop KV at every depth: no plain forward matches.
        assert difference < 2e-5 if layout == "last_exited" else difference > 1e-2


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_seed_none_is_resolved_at_admission_and_replays_exactly(asynchronous):
    params = rollout(max_tokens=8, temperature=1.0, seed=None)
    engine = make_engine(model(), asynchronous=asynchronous)
    torch_state, python_state = torch.get_rng_state(), random.getstate()
    for rid in ("x", "y", "unsampled"):
        engine.add_request(rid, [2, 3], params)
    # Seeds come from OS entropy, not from the torch or Python global RNG.
    assert torch.equal(torch.get_rng_state(), torch_state) and random.getstate() == python_state
    seeds = {rid: r.sampling_params.seed for rid, r in engine.scheduler.requests.items()}
    assert all(type(seed) is int and 0 <= seed < 2**63 for seed in seeds.values())
    # Every output, even an abort before the first sample, reports the effective params.
    outputs = {"unsampled": engine.abort_request("unsampled")}
    assert engine.rng.peek("unsampled") is None  # Aborted before its first draw.
    sampled = [engine.scheduler.requests[rid] for rid in ("x", "y")]
    streamed = stream(engine, lambda: all(r.generated_token_ids for r in sampled))
    # The engine's registry seeded each generator with exactly the seed outputs report.
    assert all(engine.rng.peek(r).initial_seed() == seeds[r] for r in ("x", "y"))
    outputs |= final(stream(engine, streamed=streamed))
    assert engine.rng.peek("x") is None and engine.rng.peek("y") is None  # Dropped at finish.
    for rid, output in outputs.items():
        check_aligned(output)
        assert output.sampling_params == replace(params, seed=seeds[rid])
    assert len(set(seeds.values())) == 3 and outputs["x"].token_ids != outputs["y"].token_ids
    # Replaying the reported params reproduces the rollout exactly.
    replay = make_engine(model(), asynchronous=asynchronous)
    for rid in ("x", "y"):
        replay.add_request(rid, [2, 3], outputs[rid].sampling_params)
    for rid, output in final(stream(replay)).items():
        assert_same(output, outputs[rid])


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_greedy_seed_none_is_resolved_at_admission(asynchronous):
    # Greedy Ouro rows never reach the RNG registry, so admission alone resolves their seed
    # (Huginn's greedy rows do reach it: see the Huginn test below).
    engine = make_engine(model(), asynchronous=asynchronous)
    engine.rng.acquire = Mock(side_effect=AssertionError("greedy Ouro rows draw no RNG"))
    engine.add_request("g", [2, 3], rollout(max_tokens=3, seed=None))
    seed = engine.scheduler.requests["g"].sampling_params.seed
    assert type(seed) is int and 0 <= seed < 2**63
    (output,) = final(stream(engine)).values()
    assert output.sampling_params == rollout(max_tokens=3, seed=seed)


@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODES)
@pytest.mark.parametrize("temperature", [0.0, 0.8], ids=["greedy", "top_k-top_p"])
def test_huginn_reports_the_seed_of_its_recurrent_state_generator(temperature, logprobs_mode):
    # Huginn draws its random recurrent-state initialization from the request's generator
    # at prefill and in every PRELUDE, greedy requests included, so its generator is
    # created before any CODA sample. That generator must still carry the reported seed.
    torch.manual_seed(42)
    m = HuginnForCausalLM(tiny_huginn_config()).eval()
    prompts = {"a": [4, 7, 3], "b": [9, 2, 5, 6]}
    scheduler = SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=3)
    options = dict(cache_config=CacheConfig(128, 2), scheduler_config=scheduler)

    def run(params):
        engine = LLMEngine(m, logprobs_mode=logprobs_mode, **options)
        acquire, created = engine.rng.acquire, {}

        def recording_acquire(request_id, request_params, device):
            generator = acquire(request_id, request_params, device)
            created.setdefault(request_id, generator.initial_seed())
            return generator

        engine.rng.acquire = recording_acquire
        for rid, prompt in prompts.items():
            engine.add_request(rid, prompt, params[rid])
        return final(stream(engine)), created

    sampling = dict(temperature=temperature, top_k=5, top_p=0.9) if temperature else {}
    params = rollout(max_tokens=5, min_loops=3, max_loops=3, seed=None, **sampling)
    outputs, created = run(dict.fromkeys(prompts, params))
    assert created == {rid: output.sampling_params.seed for rid, output in outputs.items()}
    # Requesting logprobs changes no token, and replaying the reported params is bitwise.
    quiet, _ = run({rid: replace(o.sampling_params, logprobs=None) for rid, o in outputs.items()})
    replay, _ = run({rid: output.sampling_params for rid, output in outputs.items()})
    for rid, output in outputs.items():
        assert quiet[rid].token_ids == output.token_ids
        assert quiet[rid].exit_depths == output.exit_depths
        assert_same(replay[rid], output)


PREEMPTED_PHASES = [
    pytest.param(False, RequestPhase.CODA, id="sync-coda"),
    pytest.param(False, RequestPhase.PRELUDE, id="sync-prelude"),
    pytest.param(False, RequestPhase.RECURRENT, id="sync-recurrent"),
    # Asynchronous preemption skips requests with an undelivered output. With a single
    # request, one waiting in PRELUDE still has its previous output undelivered, so this
    # construction reaches no async PRELUDE victim; the pressure test below does.
    pytest.param(True, RequestPhase.CODA, id="async-coda"),
    pytest.param(True, RequestPhase.RECURRENT, id="async-recurrent"),
]


@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODES)
@pytest.mark.parametrize("asynchronous,phase", PREEMPTED_PHASES)
def test_preemption_at_each_phase_keeps_the_rng_slot(asynchronous, phase, logprobs_mode):
    # Top-k/top-p rows: processed mode samples them through _sample_with_logprob.
    params = rollout(max_tokens=7, temperature=0.8, top_k=3, top_p=0.9, seed=45, exit_threshold=0)
    options = dict(asynchronous=asynchronous, logprobs_mode=logprobs_mode)
    scheduler = SchedulerConfig(enable_preemption=True)
    options.update(cache_config=CacheConfig(64, 2), scheduler_config=scheduler)
    results = []
    for preempt in (False, True):
        engine = make_engine(model(), **options)
        engine.add_request("a", [1, 2, 3], params)
        request = engine.scheduler.requests["a"]
        if preempt:
            stream(
                engine,
                lambda: (
                    len(request.generated_token_ids) >= 2
                    and request.phase == phase
                    and not request.num_output_placeholders
                ),
            )
            generator = engine.rng.peek("a")
            state = generator.get_state().clone()
            engine.add_request("b", [5], SamplingParams(max_tokens=1))
            assert engine.scheduler.preempt(engine.scheduler.requests["b"])
            # Suspension keeps the request's generator, at the same point of its stream.
            assert engine.rng.peek("a") is generator
            assert torch.equal(generator.get_state(), state)
        results.append(final(stream(engine))["a"])
        assert engine.preemption.resumptions == int(preempt)
    # A CODA victim resumes into "b"'s CODA batch, so its LM head runs two rows instead of
    # one. CPU GEMMs are not batch invariant, so its logprobs match numerically (up to ~5e-7).
    # PRELUDE and RECURRENT victims share no batch with "b": theirs stay bitwise.
    assert_same(results[1], results[0], atol=1e-6 if phase == RequestPhase.CODA else None)


@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODES)
def test_async_prelude_preemption_keeps_the_rng_slot(logprobs_mode):
    """Memory pressure preempts asynchronous requests waiting in PRELUDE with no undelivered
    output (the construction of tests/test_prefix_growth.py's GPU test for #109). On CPU no
    AsyncState exists, so this covers the RNG slot and logprobs, not #109's device token."""
    trace = ExitConfig("trace", depths_by_request={"t": [4, 1, 2, 3, 4, 1, 2, 3]})
    params = rollout(max_tokens=8, min_loops=1, temperature=0.9, top_k=5, top_p=0.95)
    scheduler = SchedulerConfig(
        max_num_seqs=2, max_num_batched_tokens=2, prefill_chunk_size=2, enable_preemption=True
    )
    options = dict(asynchronous=True, exit_config=trace, logprobs_mode=logprobs_mode)
    results, victims = [], []
    for blocks in (128, 24):
        cache = CacheConfig(blocks, 2, incremental_allocation=True)
        engine = make_engine(model(72), cache_config=cache, scheduler_config=scheduler, **options)
        runner, suspended = engine.model_runner, {}
        suspend, resume = runner.suspend, engine.scheduler.resume_callback

        def checked_suspend(request_id, engine=engine, suspend=suspend, suspended=suspended):
            request, generator = engine.scheduler.requests[request_id], engine.rng.peek(request_id)
            victims.append((request.phase, generator is not None))
            state = None if generator is None else generator.get_state().clone()
            suspended[request_id] = generator, state
            suspend(request_id)

        def checked_resume(request, engine=engine, resume=resume, suspended=suspended):
            if request.request_id in suspended:  # Its stream is untouched while suspended.
                generator, state = suspended[request.request_id]
                assert engine.rng.peek(request.request_id) is generator
                assert generator is None or torch.equal(generator.get_state(), state)
            return resume(request)

        runner.suspend, engine.scheduler.resume_callback = checked_suspend, checked_resume
        for i in range(4):
            prompt = [2 * i + 1, 2 * i + 2]
            engine.add_request(str(i), prompt, replace(params, seed=10 + i), trace_id="t")
        results.append(final(stream(engine)))
        assert (engine.preemption.preemptions > 0) == (blocks == 24)
        assert engine.preemption.resumptions == engine.preemption.preemptions
    assert (RequestPhase.PRELUDE, True) in victims  # A PRELUDE victim that had already sampled.
    for rid, output in results[0].items():
        # Pressure changes the batch composition, so logprobs match numerically.
        assert_same(results[1][rid], output, atol=1e-6)


def test_preemption_and_prefix_hits_keep_logprobs():
    params = rollout(max_tokens=7, temperature=0.8, seed=45, exit_threshold=0)
    results = []
    for suspend in (False, True):
        scheduler = SchedulerConfig(enable_preemption=True)
        engine = make_engine(model(), cache_config=CacheConfig(64, 2), scheduler_config=scheduler)
        engine.add_request("a", [1, 2, 3], params)
        request = engine.scheduler.requests["a"]
        stream(engine, lambda: len(request.generated_token_ids) >= 3)
        if suspend:
            before = list(request.logprobs)
            engine.add_request("b", [5], SamplingParams(max_tokens=1))
            assert engine.scheduler.preempt(engine.scheduler.requests["b"])
            assert request.logprobs == before and request.phase == RequestPhase.WAITING
        results.append(final(stream(engine))["a"])
        assert engine.preemption.resumptions == int(suspend)
    assert_same(results[1], results[0])
    # Memory-pressure preemption matches an unpressured run.
    params, results = rollout(max_tokens=8, exit_threshold=0), []
    for blocks, preempt in ((128, False), (24, True)):
        cache = CacheConfig(blocks, 2, incremental_allocation=True)
        scheduler = SchedulerConfig(max_num_seqs=3, prefill_chunk_size=2, enable_preemption=preempt)
        engine = make_engine(model(15), cache_config=cache, scheduler_config=scheduler)
        for i in range(3):
            engine.add_request(str(i), [1, 2], params)
        results.append(final(stream(engine)))
        assert (engine.preemption.preemptions > 0) == preempt
    for rid, output in results[0].items():
        assert_same(results[1][rid], output, atol=1e-6)
    # A prefix-cache hit matches the cold run.
    cache = CacheConfig(64, 2, enable_prefix_caching=True)
    scheduler = SchedulerConfig(prefill_chunk_size=2)
    engine = make_engine(model(15), cache_config=cache, scheduler_config=scheduler)
    engine.add_request("cold", [1, 2, 3, 4, 5], replace(params, max_tokens=4))
    cold = final(stream(engine))["cold"]
    engine.add_request("warm", [1, 2, 3, 4, 5], replace(params, max_tokens=4))
    engine.step()
    assert engine.last_schedule.requests[0].token_start == 4
    assert_same(final(stream(engine))["warm"], cold, atol=1e-6)


@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODES)
@pytest.mark.parametrize(
    "old,new", [(0, None), (None, 0)], ids=["logprobs-then-none", "none-then-logprobs"]
)
@pytest.mark.parametrize("exit_config,threshold", ASYNC_EXIT_MODES)
def test_lagging_delivery_abort_and_id_reuse_keep_logprobs(
    monkeypatch, exit_config, threshold, old, new, logprobs_mode
):
    sampling = dict(temperature=0.9, top_p=0.95, seed=7, min_loops=1, exit_threshold=threshold)
    params = rollout(max_tokens=6, **sampling)  # T != 1 and top-p: processed differs from raw.
    options = dict(exit_config=exit_config, logprobs_mode=logprobs_mode)

    def fresh(*, asynchronous):
        engine = make_engine(model(), asynchronous=asynchronous, **options)
        engine.add_request("a", [5, 6], replace(params, logprobs=new))
        return final(stream(engine))["a"]

    expected = fresh(asynchronous=True)
    if exit_config.mode == "trace":  # Asynchronous delivery matches the synchronous path.
        assert_same(fresh(asynchronous=False), expected, atol=1e-6)
    hold_coda_delivery(monkeypatch)
    engine = make_engine(model(), asynchronous=True, **options)
    slots = lease_cpu_logprob_slots(monkeypatch, engine)
    engine.add_request("a", [2, 3], replace(params, logprobs=old))
    engine.add_request("b", [4], replace(params, seed=5))
    request = engine.scheduler.requests["a"]
    # Step until the request has delivered a token and has one undelivered CODA output.
    stream(engine, lambda: request.generated_token_ids and request.num_output_placeholders == 1)
    delivered = len(request.generated_token_ids)
    aborted = engine.abort_request("a")
    check_aligned(aborted)
    # The undelivered token is not part of the abort output.
    assert aborted.finish_reason == "abort" and len(aborted.token_ids) == delivered
    # Reuse the ID with the opposite setting while the stale ticket (and lease) is pending.
    assert engine._pending_coda and (old is None or any(slot.leased for slot in slots))
    engine.add_request("a", [5, 6], replace(params, logprobs=new))
    assert_same(final(stream(engine))["a"], expected, atol=1e-6)
    assert not engine._pending_coda and not any(slot.leased for slot in slots)


def test_speculative_admission_rejects_logprobs():
    engine = LLMEngine(model(), speculative_config=SpeculativeConfig(3))
    with pytest.raises(ValueError, match="speculative"):
        engine.add_request("r", [2, 3], SamplingParams(logprobs=0))
    assert not engine.has_unfinished_requests()


@pytest.mark.parametrize(
    "finish,max_tokens,emitted",
    [
        ("round", 9, [10, 50, 0, 13]),
        ("stop_token_ids", 9, [10, 50]),
        ("eos", 9, [10, 50, 0]),
        ("length", 3, [10, 50]),
    ],
    ids=["continue", "stop-id-mid-round", "eos-mid-round", "length-at-round-end"],
)
def test_async_speculative_collect_uses_shared_finish_rules(finish, max_tokens, emitted):
    """Deliver one async speculative round on CPU. The runner needs CUDA, so a stub ticket
    stands in for its readback; _collect_speculative itself is the code under test."""
    engine = LLMEngine(model(), speculative_config=SpeculativeConfig(3))
    stops = [50] if finish == "stop_token_ids" else []
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=finish != "eos", stop_token_ids=stops)
    engine.add_request("r", [2, 3], params)
    request = engine.scheduler.requests["r"]
    stream(engine, lambda: request.generated_token_ids)  # The bootstrap CODA token.
    batch = engine.scheduler.schedule()
    [scheduled_request] = batch.requests
    # Apply _step_async_speculative's submission bookkeeping, then deliver the round.
    engine.scheduler.update_after_submission(batch)
    assert request.num_output_placeholders == scheduled_request.token_count
    tokens = [10, 50, 0, 13][: scheduled_request.token_count]
    frontier = len(request.prompt_token_ids) + len(request.generated_token_ids) - 1 + len(tokens)
    result = ModelRunnerOutput(
        speculative=[SpeculativeResult(tokens, len(tokens) - 1)], frontiers=[frontier]
    )
    ticket = SimpleNamespace(
        batch=batch,
        states=(SimpleNamespace(delivered=0),),
        rounds=(0,),
        ready=lambda: True,
        collect=lambda: result,
        retire=lambda: None,
    )
    stats = engine.speculative_runner.stats
    engine.speculative_runner = SimpleNamespace(
        stats=stats,
        readback_waits=0,
        discarded_rounds=0,
        request_ready=lambda r: True,
        release=lambda r: None,
    )
    engine.async_speculative, engine._pending_speculative = True, [ticket]
    [output] = engine._collect_speculative()
    reason = {"round": None, "length": "length"}.get(finish, "stop")
    assert (output.token_ids[1:], output.finish_reason) == (emitted, reason)
    assert output.exit_depths == [4] * len(output.token_ids) and output.logprobs is None
    assert stats.committed_tokens == len(emitted) and request.num_output_placeholders == 0
    if reason is None:
        assert list(engine.scheduler.queues[RequestPhase.SPECULATIVE]) == ["r"]
    else:
        assert not engine.has_unfinished_requests()
        assert engine.cache_manager.num_used_blocks == 0


def test_pd_coordinator_resolves_seeds_and_mirrors_logprobs():
    engine = object.__new__(PDEngine)  # Coordinator only: no worker processes.
    engine.closed, engine.failure = False, None
    engine.requests, engine.transfers, engine.outputs = {}, {}, deque()
    engine.config, engine.exit_config = PDConfig(), ExitConfig("ouro_delayed")
    engine.speculative_config, engine._async_speculative = None, False
    engine.model_config = tiny_ouro_config()
    info, roles = dict(block_size=4, depths=4, num_blocks=1024), ("prefill", "decode")
    engine.peers = {
        role: Peer(role, role, SimpleNamespace(info=info, send=Mock())) for role in roles
    }
    with pytest.raises(ValueError, match="stop_token_ids"):
        engine.add_request("bad", [1], SamplingParams(stop_token_ids=[64]))
    # P and D get these effective params (one resolved seed), which an early abort reports.
    for rid in ("unsampled", "r"):
        engine.add_request(rid, [1, 2], rollout(max_tokens=8, temperature=1.0, seed=None))
    unsampled = engine.requests["unsampled"].sampling_params
    assert engine.abort_request("unsampled").sampling_params == unsampled
    request = engine.requests["r"]
    assert type(unsampled.seed) is int and type(request.sampling_params.seed) is int
    assert request.logprobs == []
    w = next(iter(engine.transfers.values()))
    w.p, w.d, w.phase = "prefill", "decode", "decode"
    for n in (1, 2):
        mirrored = dict(logprobs=[-0.5] * n, sampling_params=request.sampling_params)
        output = RequestOutput(w.tid, [1, 2], [5] * n, [4] * n, False, **mirrored)
        message = dict(kind="output", request_id=w.tid, output=output)
        engine._message(engine.peers["decode"], message)
    assert [o.request_id for o in engine.outputs] == ["r", "r"]
    assert engine.outputs[-1].logprobs == [-0.5, -0.5]
    aborted = engine.abort_request("r")
    assert aborted.finish_reason == "abort" and aborted.sampling_params == request.sampling_params
    assert (aborted.token_ids, aborted.exit_depths) == ([5, 5], [4, 4])
    assert aborted.logprobs == [-0.5, -0.5]


def gpu_engine(
    m, *, asynchronous, exit_config=None, seqs=3, logprobs_mode="raw_logprobs", **execution
):
    scheduler = SchedulerConfig(max_num_seqs=seqs, max_num_batched_tokens=8, prefill_chunk_size=4)
    options = dict(cache_config=CacheConfig(128, 16), scheduler_config=scheduler)
    options |= dict(logprobs_mode=logprobs_mode, attention_backend="triton", execution=execution)
    exit_config = exit_config or ExitConfig("ouro_delayed")
    return make_engine(m, asynchronous=asynchronous, exit_config=exit_config, **options)


def gpu_params(rid, **overrides):
    values = dict(max_tokens=6 + int(rid), temperature=0.8, top_k=16, seed=40 + int(rid))
    return rollout(**(values | dict(min_loops=2, exit_threshold=0.2) | overrides))


@pytest.mark.gpu
@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODES)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_gpu_cuda_graph_logprobs_match_eager(asynchronous, logprobs_mode):
    # Static eager pads RECURRENT rows to a power of two while graph replay runs len(rows),
    # so GEMM reductions may round; FP32 keeps tokens and exits exact and bounds logprobs.
    m = model(321, "cuda", torch.float32, head_dim=64)
    results = []
    for graphs in (False, True):
        # One stream (no prefer_recurrent overlap): batches cannot depend on graph timing.
        static = dict(static_buffers=True, pad_to_power_of_two=True, multi_stream=False)
        options = dict(asynchronous=asynchronous, logprobs_mode=logprobs_mode, cuda_graphs=graphs)
        engine = gpu_engine(m, **options, **static)
        for i in range(3):
            engine.add_request(str(i), [2 + i, 3, 4] * (i + 1), gpu_params(str(i)))
        results.append(final(stream(engine)))
        assert not graphs or engine.model_runner.graphs.replays > 0
    eager, graphed = results
    assert graphed.keys() == eager.keys()
    for rid, output in eager.items():
        assert_same(graphed[rid], output, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True], ids=["one-stream", "multi-stream"])
def test_gpu_async_readback_matches_sync_and_returns_leases(monkeypatch, multi_stream):
    # FP32 keeps sampled tokens stable across sync/async batch compositions.
    m = model(321, "cuda", torch.float32, head_dim=64)
    traces = {str(i): [4, 1, 3, 2, 4, 1, 2, 3, 4, 2, 1, 3] for i in range(4)}
    options = dict(exit_config=ExitConfig("trace", depths_by_request=traces), seqs=4)

    def run(*, asynchronous):
        engine = gpu_engine(m, asynchronous=asynchronous, multi_stream=multi_stream, **options)
        for i in range(4):
            params = gpu_params(str(i), max_tokens=12, min_loops=1, exit_threshold=1.0)
            engine.add_request(str(i), [2, 3 + i], params)
        return engine, final(stream(engine))

    _, expected = run(asynchronous=False)
    # Delivery lags to the one-output bound: logprob leases overlap live exit-score tickets.
    hold_coda_delivery(monkeypatch)
    engine, actual = run(asynchronous=True)
    assert not any(slot.leased for slot in engine.model_runner.logprob_slots)
    for rid, output in expected.items():
        assert_same(actual[rid], output, atol=1e-4)


@pytest.mark.gpu
def test_gpu_delayed_exit_scores_and_logprobs_do_not_exhaust_readback_pools(monkeypatch):
    engine = gpu_engine(model(5, "cuda", torch.bfloat16, head_dim=64), asynchronous=True, seqs=4)
    assert len(engine.model_runner.logprob_slots) == 4 + 4
    # Lagging delivery holds a logprob lease per request while delayed exit scores hold theirs.
    hold_coda_delivery(monkeypatch)
    for round_index in range(3):
        for i in range(4):
            engine.add_request(f"{round_index}-{i}", [2, 3, i], gpu_params(str(i), max_tokens=16))
        assert len(final(stream(engine))) == 4
    assert not any(slot.leased for slot in engine.model_runner.logprob_slots)


@pytest.mark.gpu
def test_gpu_abort_with_pending_coda_returns_lease_and_reuses_id(monkeypatch):
    engine = gpu_engine(model(5, "cuda", torch.bfloat16, head_dim=64), asynchronous=True)
    slots = engine.model_runner.logprob_slots
    ready = hold_coda_delivery(monkeypatch)
    for i in range(2):
        engine.add_request(str(i), [2, 3, i], gpu_params(str(i), max_tokens=12))
    request = engine.scheduler.requests["0"]
    # Step until the request has delivered a token and has one undelivered CODA output.
    stream(engine, lambda: request.generated_token_ids and request.num_output_placeholders == 1)
    delivered = len(request.generated_token_ids)
    aborted = engine.abort_request("0")
    check_aligned(aborted)
    assert len(aborted.token_ids) == delivered and any(slot.leased for slot in slots)
    stale = [t for t in engine._pending_coda if any(i.request is request for i in t.batch.requests)]
    assert stale
    # release() synchronized the aborted request, so the next step collects its ticket.
    monkeypatch.setattr(Submission, "ready", ready)
    engine.add_request("0", [2, 3], gpu_params("0", logprobs=None))
    for output in engine.step():
        check_aligned(output)
    assert not any(t is s for t in engine._pending_coda for s in stale)
    outputs = final(stream(engine))
    assert outputs["0"].logprobs is None and len(outputs["0"].token_ids) == 6
    assert len(outputs["1"].logprobs) == 12
    assert not engine._pending_coda and not any(slot.leased for slot in slots)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_execution_error_returns_undelivered_logprob_leases(monkeypatch, device):
    cuda = device == "cuda"
    m = model(5, "cuda", torch.bfloat16, head_dim=64) if cuda else model(5)
    engine = gpu_engine(m, asynchronous=True, seqs=4) if cuda else make_engine(m, asynchronous=True)
    slots = lease_cpu_logprob_slots(monkeypatch, engine)
    runner = engine.model_runner
    ready, execute = hold_coda_delivery(monkeypatch), runner._execute
    held = []

    def fail(batch, prepared):
        result = execute(batch, prepared)
        if prepared.batch.operation == Operation.RECURRENT and engine._pending_coda:
            held.append(sum(slot.leased for slot in slots))
            raise RuntimeError("injected failure after CODA submission")
        return result

    monkeypatch.setattr(runner, "_execute", fail)
    for i in range(3):
        engine.add_request(str(i), [2, 3, i], gpu_params(str(i), max_tokens=8))
    with pytest.raises(RuntimeError, match="injected"):
        stream(engine)
    # The failure path drops undelivered tickets without collect(); __del__ returns leases.
    assert held and held[0] > 0 and not engine._pending_coda
    gc.collect()
    assert not any(slot.leased for slot in slots)
    monkeypatch.setattr(runner, "_execute", execute)
    monkeypatch.setattr(Submission, "ready", ready)
    engine.add_request("3", [2, 3], gpu_params("3", max_tokens=4))
    check_aligned(final(stream(engine))["3"])


@pytest.mark.gpu
def test_gpu_async_speculative_stop_token_ids_match_sync():
    """Cross-round async speculation (CUDA, Triton, greedy) stops on stop IDs like sync."""
    m = model(device="cuda")
    prompts = {"a": [2, 3, 4, 5], "b": [9], "c": [2, 3, 4, 7]}

    def run(asynchronous, **overrides):
        engine = LLMEngine(
            m,
            attention_backend="triton",
            cache_config=CacheConfig(256, 2, incremental_allocation=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4, max_num_batched_tokens=16, prefill_chunk_size=2
            ),
            execution_config=ExecutionConfig(async_scheduling=asynchronous),
            speculative_config=SpeculativeConfig(3),
        )
        for rid, prompt in prompts.items():
            params = SamplingParams(max_tokens=16, ignore_eos=True, **overrides)
            engine.add_request(rid, prompt, params)
        outputs = final(stream(engine))
        engine.close()
        return outputs

    free = run(False)
    # Stop on the latest first occurrence after the bootstrap token, so a round is cut.
    first = {
        (rid, i): t
        for rid, o in free.items()
        for i, t in enumerate(o.token_ids)
        if i > 0 and t not in o.token_ids[:i]
    }
    assert first, "greedy outputs repeat their bootstrap token; pick other prompts"
    stop = first[max(first, key=lambda key: key[1])]
    expected, actual = run(False, stop_token_ids=[stop]), run(True, stop_token_ids=[stop])
    assert any(o.finish_reason == "stop" for o in expected.values())
    for rid, output in expected.items():
        tokens = free[rid].token_ids
        cut = tokens[: tokens.index(stop) + 1] if stop in tokens else tokens
        assert output.token_ids == cut
        assert (actual[rid].token_ids, actual[rid].finish_reason) == (
            output.token_ids,
            output.finish_reason,
        )


@pytest.mark.gpu
def test_gpu_pd_forwards_logprobs_and_abort_stays_aligned():
    pytest.importorskip("nixl")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 visible GPUs")
    exit_config = ExitConfig("trace", depths_by_request={"t": [4, 2, 3, 1] * 16})
    prefill = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=5, prefill_chunk_size=3)
    transfer = dict(transfer_chunk_bytes=4096, max_inflight_bytes=16384)
    caches = dict(prefill_cache_config=CacheConfig(128, 4), decode_cache_config=CacheConfig(128, 4))
    with PDEngine(
        tiny_ouro_config(head_dim=64),
        pd_config=PDConfig(**transfer, request_timeout=90, startup_timeout=90),  # GPUs 0, 1.
        prefill_scheduler_config=prefill,
        decode_scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2),
        exit_config=exit_config,
        execution_config=ExecutionConfig(async_scheduling=True),
        attention_backend="triton",
        seed=123,
        **caches,
    ) as e:
        m = model(123, "cuda:0", torch.bfloat16, head_dim=64)
        options = dict(cache_config=CacheConfig(128, 4), scheduler_config=prefill)
        reference = make_engine(m, exit_config=exit_config, attention_backend="triton", **options)
        params = rollout(max_tokens=6, min_loops=1, temperature=0.7, top_k=32, seed=947)
        for target in (e, reference):
            target.add_request("r", [1, 2, 3, 4, 5], params, trace_id="t")
        assert_same(final(stream(e))["r"], final(stream(reference))["r"], atol=5e-2)
        e.add_request("abort", [1, 2, 3, 4, 5], replace(params, max_tokens=64), trace_id="t")
        delivered = defaultdict(list)
        stream(e, lambda: len(delivered["abort"]) >= 2, delivered)
        output = e.abort_request("abort")
        check_aligned(output)
        assert output.token_ids == delivered["abort"][-1].token_ids
        assert output.logprobs == delivered["abort"][-1].logprobs
        stream(e)
