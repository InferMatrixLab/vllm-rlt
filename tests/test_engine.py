import math

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import LLM, CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import FinishReason, RequestOutput, Stage


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
        trace.append((batch.stage, tuple(i.request.request_id for i in batch.items)))
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
    assert llm.engine.cache_manager.num_used_blocks == 0


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
    recurrent = [ids for stage, ids in trace if stage == Stage.RECURRENT]
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
    assert engine.scheduler.requests["b"].stage == Stage.WAITING
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
    assert [stage for stage, _ in trace] == [Stage.PREFILL, Stage.CODA]
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
        if request.generator is not None:
            break
        engine.step()
    assert request.generator is not None
    first, _ = drain(engine)
    # finish() pops the request from scheduler.requests, but this local name is
    # still a live reference to the same object, so it can observe the released
    # RNG: reusing the ID must not continue the previous stream.
    assert request.generator is None
    engine.add_request("r", [2, 4], params)
    assert engine.scheduler.requests["r"] is not request
    second, _ = drain(engine)
    assert second["r"].token_ids == first["r"].token_ids
    assert second["r"].exit_depths == first["r"].exit_depths


def test_greedy_requests_never_create_a_generator():
    engine = LLMEngine(tiny_model(), cache_config=CacheConfig(64, 2))
    engine.add_request("g", [2, 4], SamplingParams(max_tokens=3, ignore_eos=True))
    request = engine.scheduler.requests["g"]
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            break
        engine.step()
        assert request.generator is None
    else:
        pytest.fail("engine failed to finish within bounded scheduler steps")
    assert request.generator is None


def test_execution_failure_releases_affected_requests(monkeypatch):
    engine = LLMEngine(tiny_model())
    engine.add_request("one", [2], SamplingParams(max_tokens=2))

    def fail(_):
        raise RuntimeError("injected runner failure")

    monkeypatch.setattr(engine.model_runner, "execute", fail)
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
            if batch.stage == Stage.RECURRENT and any(
                item.request.request_id == "long" for item in batch.items
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
    scheduler.queues[Stage.PREFILL].remove("decode")
    scheduler.enqueue(scheduler.requests["decode"], Stage.RECURRENT)
    ensure_capacity = engine.cache_manager.ensure_capacity
    monkeypatch.setattr(
        engine.cache_manager, "ensure_capacity", lambda rid, frontier: rid != "prefill"
    )
    assert scheduler.schedule() is None
    monkeypatch.setattr(engine.cache_manager, "ensure_capacity", ensure_capacity)
    assert scheduler.schedule().stage == Stage.PREFILL
    assert scheduler.schedule().stage == Stage.RECURRENT


def test_failed_recurrent_batch_keeps_decode_due(monkeypatch):
    engine = LLMEngine(tiny_model())
    for rid in ("prefill", "decode"):
        engine.add_request(rid, [1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
    scheduler = engine.scheduler
    scheduler._admit()
    scheduler.queues[Stage.PREFILL].remove("decode")
    scheduler.enqueue(scheduler.requests["decode"], Stage.RECURRENT)
    assert scheduler.schedule().stage == Stage.PREFILL
    ensure_capacity = engine.cache_manager.ensure_capacity
    monkeypatch.setattr(engine.cache_manager, "ensure_capacity", lambda rid, frontier: False)
    assert scheduler.schedule() is None
    scheduler.enqueue(scheduler.requests["prefill"], Stage.PREFILL)
    monkeypatch.setattr(engine.cache_manager, "ensure_capacity", ensure_capacity)
    assert scheduler.schedule().stage == Stage.RECURRENT


def test_abort_removes_all_queued_work_and_serializes_reason():
    engine = LLMEngine(tiny_model())
    engine.add_request("r", [1, 2], SamplingParams(max_tokens=2))
    engine.scheduler._admit()
    request = engine.scheduler.requests["r"]
    # A delayed termination must remove every queued occurrence.
    engine.scheduler.queues[Stage.PREFILL].append("r")
    engine.scheduler.enqueue(request, Stage.RECURRENT)
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


def test_repetition_penalty_and_dynamic_depth_temp():
    with pytest.raises(ValueError):
        SamplingParams(repetition_penalty=0.0)
    with pytest.raises(ValueError):
        SamplingParams(repetition_penalty=-1.0)
    with pytest.raises(ValueError):
        SamplingParams(repetition_penalty=0.5)
    with pytest.raises(ValueError):
        SamplingParams(dynamic_depth_temp="invalid")

    model = tiny_model()
    p_default = SamplingParams(max_tokens=4, repetition_penalty=1.0, ignore_eos=True)
    p_penalized = SamplingParams(max_tokens=4, repetition_penalty=5.0, ignore_eos=True)
    out_default = LLM(model).generate([[1, 2]], p_default)[0]
    out_penalized = LLM(model).generate([[1, 2]], p_penalized)[0]
    assert out_default.token_ids == [9, 27, 27, 39]
    assert out_penalized.token_ids == [9, 27, 15, 43]
