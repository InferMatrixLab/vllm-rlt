import math
from dataclasses import replace

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    LoopCDParams,
    SamplingParams,
    SchedulerConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import FinishReason, Request, RequestOutput, RequestPhase


def tiny_model():
    torch.manual_seed(123)
    return OuroForCausalLM(tiny_ouro_config())


def drain(engine, limit=1000):
    outputs, trace = {}, []
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            return outputs, trace
        emitted = engine.step()
        batch = engine.last_schedule
        trace.append((batch.phase, tuple(i.request.request_id for i in batch.requests)))
        for output in emitted:
            if output.finished:
                outputs[output.request_id] = output
    pytest.fail("engine failed to finish within bounded scheduler steps")


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_chunked_packed_generation_matches_serial(mode):
    model = tiny_model()
    prompts = [[2, 3, 4, 5, 6], [6], [8, 9, 10]]
    params = [
        SamplingParams(max_tokens=4, exit_threshold=t, ignore_eos=True) for t in [0.0, 1.0, 0.7]
    ]
    serial = [
        LLM(model, cache_config=CacheConfig(64, 2)).generate([prompt], param)[0]
        for prompt, param in zip(prompts, params)
    ]
    llm = LLM(
        model,
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=3, mode=mode),
    )
    outputs = llm.generate(prompts, params)
    assert [o.token_ids for o in outputs] == [o.token_ids for o in serial]
    assert [o.exit_depths for o in outputs] == [o.exit_depths for o in serial]
    assert outputs[0].exit_depths == [4, 2, 2, 2]
    assert outputs[1].exit_depths == [4, 4, 4, 4]
    assert llm.engine.stats()["used_blocks"] == 0


def test_cumulative_gate_and_minimum_depth():
    model = tiny_model()
    with torch.no_grad():
        model.model.early_exit_gate.weight.zero_()
        model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
    outputs = LLM(model).generate(
        [[3, 4]], SamplingParams(max_tokens=3, exit_threshold=0.6, ignore_eos=True)
    )
    # A per-step hazard of .4 never reaches .6, but the depth-two CDF is .64.
    assert outputs[0].exit_depths == [4, 2, 2]


def test_threshold_one_never_exits_early_even_if_sigmoid_rounds_to_one():
    model = tiny_model()
    with torch.no_grad():
        model.model.early_exit_gate.weight.zero_()
        model.model.early_exit_gate.bias.fill_(100)
    output = LLM(model).generate([[2]], SamplingParams(max_tokens=2, ignore_eos=True))[0]
    assert output.exit_depths == [4, 4]


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_refill_recycles_exited_token_before_slow_token_coda(mode):
    engine = LLMEngine(tiny_model(), scheduler_config=SchedulerConfig(mode=mode))
    engine.add_request("fast", [1], SamplingParams(max_tokens=3, exit_threshold=0, ignore_eos=True))
    engine.add_request("slow", [2], SamplingParams(max_tokens=3, exit_threshold=1, ignore_eos=True))
    outputs, trace = drain(engine)
    recurrent = [ids for stage, ids in trace if stage == RequestPhase.RECURRENT]
    # In refill mode fast returns with its next token while slow is on its first.
    assert recurrent[:4] == (
        [("fast", "slow")] * 2 + [("slow", "fast")] * 2
        if mode == "refill"
        else [("fast", "slow")] * 2 + [("slow",)] * 2
    )
    assert set(outputs) == {"fast", "slow"}


def test_memory_pressure_waits_then_reuses_pages_and_abort_reclaims():
    engine = LLMEngine(tiny_model(), cache_config=CacheConfig(8, 2))
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    for name in ["a", "b", "c"]:
        engine.add_request(name, [2, 3], params)
    engine.step()
    assert engine.scheduler.requests["b"].phase == RequestPhase.WAITING
    assert engine.abort_request("b").finish_reason == "abort"
    assert engine.cache_manager.num_used_blocks == 8
    assert engine.abort_request("a").finished
    assert engine.cache_manager.num_used_blocks == 0
    outputs, _ = drain(engine)
    assert set(outputs) == {"c"}
    assert engine.cache_manager.num_used_blocks == 0


def test_large_coda_threshold_flushes_without_deadlock():
    engine = LLMEngine(tiny_model(), scheduler_config=SchedulerConfig(min_coda_batch_size=50))
    engine.add_request("one", [3], SamplingParams(max_tokens=3, exit_threshold=0, ignore_eos=True))
    outputs, _ = drain(engine)
    assert outputs["one"].exit_depths == [4, 2, 2]


def test_new_arrival_during_decode():
    engine = LLMEngine(tiny_model())
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    engine.add_request("first", [1, 2], params)
    while not engine.step():
        pass
    engine.add_request("later", [5, 6, 7], params)
    outputs, _ = drain(engine)
    assert set(outputs) == {"first", "later"}


def test_eos_finishes_without_extra_decode_and_invalid_requests_do_not_poison_queue():
    model = tiny_model()
    with torch.no_grad():
        model.lm_head.weight.zero_()  # deterministic argmax 0, Ouro EOS
    engine = LLMEngine(model, cache_config=CacheConfig(4, 2))
    for tokens, params in [
        ([], SamplingParams()),
        ([64], SamplingParams()),
        ([1], SamplingParams(max_tokens=9)),
    ]:
        with pytest.raises(ValueError):
            engine.add_request("bad", tokens, params)
    assert not engine.has_unfinished_requests()
    engine.add_request("eos", [1], SamplingParams(max_tokens=2))
    outputs, trace = drain(engine)
    assert outputs["eos"].token_ids == [0]
    assert outputs["eos"].finish_reason == "stop"
    assert [stage for stage, _ in trace] == [RequestPhase.PREFILL, RequestPhase.CODA]
    assert engine.cache_manager.num_used_blocks == 0


def test_sampled_generation_is_batch_invariant():
    model = tiny_model()
    params = SamplingParams(
        max_tokens=3, temperature=0.8, top_k=9, top_p=0.8, seed=5, ignore_eos=True
    )
    serial = LLM(model).generate([[2, 4]], params)[0]
    batched = LLM(model).generate([[6], [2, 4]], params)[1]
    assert serial.token_ids == batched.token_ids


def test_same_id_after_finish_starts_from_a_fresh_generator():
    params = SamplingParams(max_tokens=3, temperature=0.8, seed=17, ignore_eos=True)
    engine = LLMEngine(tiny_model(), cache_config=CacheConfig(64, 2))
    engine.add_request("r", [2, 4], params)
    request = engine.scheduler.requests["r"]
    for _ in range(20):
        if engine.rng.peek("r") is not None:
            break
        engine.step()
    first_generator = engine.rng.peek("r")
    assert first_generator is not None
    first, _ = drain(engine)
    # Termination drops the request's RNG slot, so a request that reuses the ID
    # cannot continue the previous stream: it gets a new generator and replays
    # the same tokens.
    assert engine.rng.peek("r") is None
    engine.add_request("r", [2, 4], params)
    assert engine.scheduler.requests["r"] is not request
    for _ in range(20):
        if engine.rng.peek("r") is not None:
            break
        engine.step()
    assert engine.rng.peek("r") is not first_generator
    second, _ = drain(engine)
    assert second["r"].token_ids == first["r"].token_ids
    assert second["r"].exit_depths == first["r"].exit_depths


def test_abort_drops_the_rng_slot_so_id_reuse_starts_over():
    params = SamplingParams(max_tokens=4, temperature=0.8, seed=23, ignore_eos=True)
    engine = LLMEngine(tiny_model(), cache_config=CacheConfig(64, 2))
    engine.add_request("r", [2, 4], params)
    for _ in range(20):
        if engine.rng.peek("r") is not None:
            break
        engine.step()
    aborted_generator = engine.rng.peek("r")
    assert aborted_generator is not None
    engine.abort_request("r")
    assert engine.rng.peek("r") is None

    engine.add_request("r", [2, 4], params)
    for _ in range(20):
        if engine.rng.peek("r") is not None:
            break
        engine.step()
    assert engine.rng.peek("r") is not aborted_generator
    reused, _ = drain(engine)

    fresh = LLMEngine(tiny_model(), cache_config=CacheConfig(64, 2))
    fresh.add_request("r", [2, 4], params)
    expected, _ = drain(fresh)
    # The rebuilt request replays a same-seed run instead of continuing the
    # aborted stream.
    assert reused["r"].token_ids == expected["r"].token_ids


def test_greedy_requests_never_create_a_generator():
    engine = LLMEngine(tiny_model(), cache_config=CacheConfig(64, 2))
    engine.add_request("g", [2, 4], SamplingParams(max_tokens=3, ignore_eos=True))
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            break
        engine.step()
        assert engine.rng.peek("g") is None
    else:
        pytest.fail("engine failed to finish within bounded scheduler steps")
    assert engine.rng.peek("g") is None


def test_greedy_sampling_never_acquires_a_generator(monkeypatch):
    # The registry is the only creator, so the strongest greedy contract is that
    # acquire() is never reached at all, not merely that no slot is left behind.
    engine = LLMEngine(tiny_model(), cache_config=CacheConfig(64, 2))

    def forbidden(*args, **kwargs):
        raise AssertionError("greedy sampling must not acquire a generator")

    monkeypatch.setattr(engine.rng, "acquire", forbidden)
    engine.add_request("g", [2, 4], SamplingParams(max_tokens=3, ignore_eos=True))
    outputs, _ = drain(engine)
    assert outputs["g"].finish_reason == "length"
    assert engine.rng.peek("g") is None


def test_coda_sampling_is_prepared_before_execution(monkeypatch):
    from vllm_rlt.request import ExecutionBatch, Operation, StateSpan

    engine = LLMEngine(tiny_model(), cache_config=CacheConfig(64, 2))
    runner = engine.model_runner
    params = SamplingParams(temperature=0.7, seed=3)
    request = Request("r", [1], params)
    request.hidden_state = torch.ones(engine.model.config.hidden_size)
    batch = ExecutionBatch(Operation.CODA, (StateSpan(request, 0, 1, 0),))
    recorded = []
    acquire = engine.rng.acquire

    def spy(request_id, sampling, device):
        recorded.append((sampling, device))
        return acquire(request_id, sampling, device)

    monkeypatch.setattr(engine.rng, "acquire", spy)
    prepared = runner.prepare_on_cpu(batch)
    request.sampling_params = SamplingParams(temperature=1.2, seed=9)
    prepared = runner.prepare_on_gpu(prepared)
    assert recorded == [(params, runner.device)]

    def unexpected(*args, **kwargs):
        pytest.fail("sampling generators must be resolved before execution")

    monkeypatch.setattr(engine.rng, "acquire", unexpected)
    token = runner._execute(None, prepared)
    assert token.shape == (1,) and token.device == runner.device
    engine.close()


def test_engines_own_independent_rng_registries():
    first = LLMEngine(tiny_model(), cache_config=CacheConfig(64, 2))
    second = LLMEngine(tiny_model(), cache_config=CacheConfig(64, 2))
    assert first.model_runner.rng is first.rng
    assert second.model_runner.rng is second.rng
    assert first.rng is not second.rng
    first.model_runner.rng.acquire("r", SamplingParams(temperature=1.0), torch.device("cpu"))
    assert second.rng.peek("r") is None


def test_execution_failure_releases_affected_requests(monkeypatch):
    engine = LLMEngine(tiny_model())
    engine.add_request("one", [2], SamplingParams(max_tokens=2))

    def fail(_):
        raise RuntimeError("injected runner failure")

    monkeypatch.setattr(engine.executor, "submit", fail)
    with pytest.raises(RuntimeError, match="injected"):
        engine.step()
    assert not engine.has_unfinished_requests()
    assert engine.cache_manager.num_used_blocks == 0


def test_generate_cleans_up_partial_admission():
    llm = LLM(tiny_model())
    with pytest.raises(ValueError):
        llm.generate([[1, 2], []])
    assert not llm.engine.has_unfinished_requests()
    assert llm.generate([[1]], SamplingParams(max_tokens=1))[0].finished


@pytest.mark.parametrize("request_id", ["", None, 1])
def test_invalid_request_id_does_not_poison_admission(request_id):
    engine = LLMEngine(tiny_model())
    with pytest.raises(ValueError, match="nonempty string"):
        engine.add_request(request_id, [1], SamplingParams(max_tokens=1))
    engine.add_request("valid", [1], SamplingParams(max_tokens=1))
    assert drain(engine)[0]["valid"].finished


@pytest.mark.parametrize("invalid", [1.5, float("nan"), float("inf"), True, 0])
def test_integer_limits_rejected_before_scheduling(invalid):
    for constructor, name in [
        (SamplingParams, "max_tokens"),
        (SamplingParams, "min_loops"),
        (SamplingParams, "max_loops"),
        (SchedulerConfig, "max_num_seqs"),
        (SchedulerConfig, "max_num_batched_tokens"),
        (CacheConfig, "num_blocks"),
    ]:
        with pytest.raises(ValueError, match="positive integer"):
            constructor(**{name: invalid})


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_repeated_short_arrivals_cannot_starve_existing_decode(mode):
    engine = LLMEngine(tiny_model(), scheduler_config=SchedulerConfig(max_num_seqs=2, mode=mode))
    engine.add_request("long", [1, 2], SamplingParams(max_tokens=20, ignore_eos=True))
    while not engine.step():
        pass
    # Leave the request at PRELUDE, also covering no_refill's fill phase.
    recurrent_steps = 0
    for index in range(12):
        request_id = f"short-{index}"
        engine.add_request(request_id, [3, 4], SamplingParams(max_tokens=1, ignore_eos=True))
        for _ in range(100):
            engine.step()
            batch = engine.last_schedule
            if batch.phase == RequestPhase.RECURRENT and any(
                scheduled_request.request.request_id == "long"
                for scheduled_request in batch.requests
            ):
                recurrent_steps += 1
            if request_id not in engine.scheduler.requests:
                break
        else:
            pytest.fail("short arrival did not finish")
    assert recurrent_steps >= 11
    drain(engine)
    assert engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_failed_prefill_batch_does_not_consume_decode_fairness_budget(monkeypatch, mode):
    engine = LLMEngine(tiny_model(), scheduler_config=SchedulerConfig(mode=mode))
    for rid in ("prefill", "decode"):
        engine.add_request(rid, [1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
    scheduler = engine.scheduler
    scheduler._admit()
    scheduler.queues[RequestPhase.PREFILL].remove("decode")
    scheduler.enqueue(scheduler.requests["decode"], RequestPhase.RECURRENT)
    ensure_capacity = engine.cache_manager.ensure_capacity
    monkeypatch.setattr(
        engine.cache_manager, "ensure_capacity", lambda rid, frontier: rid != "prefill"
    )
    assert scheduler.schedule() is None
    monkeypatch.setattr(engine.cache_manager, "ensure_capacity", ensure_capacity)
    assert scheduler.schedule().phase == RequestPhase.PREFILL
    assert scheduler.schedule().phase == RequestPhase.RECURRENT


def test_failed_recurrent_batch_keeps_decode_due(monkeypatch):
    engine = LLMEngine(tiny_model())
    for rid in ("prefill", "decode"):
        engine.add_request(rid, [1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
    scheduler = engine.scheduler
    scheduler._admit()
    scheduler.queues[RequestPhase.PREFILL].remove("decode")
    scheduler.enqueue(scheduler.requests["decode"], RequestPhase.RECURRENT)
    assert scheduler.schedule().phase == RequestPhase.PREFILL
    ensure_capacity = engine.cache_manager.ensure_capacity
    monkeypatch.setattr(engine.cache_manager, "ensure_capacity", lambda rid, frontier: False)
    assert scheduler.schedule() is None
    scheduler.enqueue(scheduler.requests["prefill"], RequestPhase.PREFILL)
    monkeypatch.setattr(engine.cache_manager, "ensure_capacity", ensure_capacity)
    assert scheduler.schedule().phase == RequestPhase.RECURRENT


def test_abort_removes_all_queued_work_and_serializes_reason():
    engine = LLMEngine(tiny_model())
    engine.add_request("r", [1, 2], SamplingParams(max_tokens=2))
    engine.scheduler._admit()
    request = engine.scheduler.requests["r"]
    # A delayed termination must remove every queued occurrence.
    engine.scheduler.queues[RequestPhase.PREFILL].append("r")
    engine.scheduler.enqueue(request, RequestPhase.RECURRENT)
    output = engine.abort_request("r")
    assert request.finish_reason is FinishReason.ABORT
    assert type(output.finish_reason) is str and output.finish_reason == "abort"
    assert all("r" not in queue for queue in engine.scheduler.queues.values())
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.has_unfinished_requests()


@pytest.mark.parametrize("reason", list(FinishReason))
def test_finish_reason_preserves_public_string_values(reason):
    engine = LLMEngine(tiny_model())
    engine.add_request("r", [1], SamplingParams(max_tokens=1))
    request = engine.scheduler.requests["r"]
    engine.scheduler.finish(request, reason)
    output = RequestOutput.from_request(request)
    assert output.finished and type(output.finish_reason) is str
    assert output.finish_reason == reason.value


@pytest.mark.parametrize("strength", [None, 0.0, 0.3])
@pytest.mark.parametrize("max_loops", [3, None])
@torch.inference_mode()
def test_decode_deeper_than_prefill_rejected_before_admission(strength, max_loops):
    guidance = None if strength is None else LoopCDParams(strength=strength)
    engine = LLMEngine(
        tiny_model(),
        cache_config=CacheConfig(num_blocks=128, block_size=4),
        execution_config=ExecutionConfig(loopcd=bool(strength), prefill_depth=2),
    )
    params = SamplingParams(
        max_tokens=3,
        min_loops=max_loops or 4,
        max_loops=max_loops,
        ignore_eos=True,
        loopcd=guidance,
    )
    with pytest.raises(ValueError, match="decode depth .* exceeds prefill_depth 2"):
        engine.add_request("r", [4, 7, 9], params)
    assert not engine.scheduler.requests
    assert not engine.has_unfinished_requests()
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.model_runner.loopcd_references

    # Rejected admission must leave the ID available for a valid continuation.
    engine.add_request("r", [4, 7, 9], replace(params, min_loops=2, max_loops=2))
    outputs, _ = drain(engine)
    assert outputs["r"].exit_depths == [2, 2, 2]
    assert len(outputs["r"].token_ids) == 3
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.model_runner.loopcd_references
    engine.close()


@pytest.mark.parametrize("strength", [None, 0.0, 0.3])
@torch.inference_mode()
def test_prefill_history_allows_shallower_decode(strength):
    engine = LLMEngine(
        tiny_model(),
        cache_config=CacheConfig(num_blocks=128, block_size=4),
        execution_config=ExecutionConfig(loopcd=bool(strength), prefill_depth=4),
    )
    engine.add_request(
        "r",
        [4, 7, 9],
        SamplingParams(
            max_tokens=3,
            min_loops=3,
            max_loops=3,
            ignore_eos=True,
            loopcd=None if strength is None else LoopCDParams(strength=strength),
        ),
    )
    outputs, _ = drain(engine)
    assert outputs["r"].exit_depths == [4, 3, 3]
    assert len(outputs["r"].token_ids) == 3
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.model_runner.loopcd_references
    engine.close()


@pytest.mark.parametrize("mode", ["ouro", "ouro_delayed", "random_lookahead"])
def test_prefill_history_checks_adaptive_decode_bound(mode):
    engine = LLMEngine(
        tiny_model(),
        execution_config=ExecutionConfig(prefill_depth=2),
        exit_config=ExitConfig(mode=mode),
    )
    params = SamplingParams(max_tokens=3, min_loops=1, max_loops=4, exit_threshold=0.7)
    with pytest.raises(ValueError, match="decode depth 4 exceeds prefill_depth 2"):
        engine.add_request("r", [4, 7, 9], params)
    assert not engine.scheduler.requests
    assert engine.cache_manager.num_used_blocks == 0
    engine.close()


@pytest.mark.parametrize("trace", [[2, 3, 2], [2, 2, 3]])
def test_prefill_history_rejects_deeper_trace_before_admission(trace):
    engine = LLMEngine(
        tiny_model(),
        execution_config=ExecutionConfig(prefill_depth=2),
        exit_config=ExitConfig(mode="trace", depths_by_request={"profile": trace}),
    )
    params = SamplingParams(max_tokens=3, min_loops=1, max_loops=4)
    with pytest.raises(ValueError, match="decode depth 3 exceeds prefill_depth 2"):
        engine.add_request("r", [4, 7, 9], params, trace_id="profile")
    assert not engine.scheduler.requests
    assert engine.cache_manager.num_used_blocks == 0
    engine.close()


@torch.inference_mode()
def test_prefill_history_uses_only_requested_trace_depths():
    engine = LLMEngine(
        tiny_model(),
        execution_config=ExecutionConfig(prefill_depth=2),
        exit_config=ExitConfig(mode="trace", depths_by_request={"profile": [2, 1, 2, 4]}),
    )
    engine.add_request(
        "r",
        [4, 7, 9],
        SamplingParams(max_tokens=3, min_loops=1, ignore_eos=True),
        trace_id="profile",
    )
    outputs, _ = drain(engine)
    assert outputs["r"].exit_depths == [2, 1, 2]
    assert len(outputs["r"].token_ids) == 3
    assert engine.cache_manager.num_used_blocks == 0
    engine.close()


@pytest.mark.parametrize("strength", [None, 0.0, 0.3])
@torch.inference_mode()
def test_prefill_only_output_does_not_require_decode_history(strength):
    engine = LLMEngine(
        tiny_model(),
        execution_config=ExecutionConfig(loopcd=bool(strength), prefill_depth=2),
    )
    engine.add_request(
        "r",
        [4, 7, 9],
        SamplingParams(
            max_tokens=1,
            min_loops=4,
            max_loops=4,
            ignore_eos=True,
            loopcd=None if strength is None else LoopCDParams(strength=strength),
        ),
    )
    outputs, _ = drain(engine)
    assert outputs["r"].exit_depths == [2]
    assert len(outputs["r"].token_ids) == 1
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.model_runner.loopcd_references
    engine.close()
