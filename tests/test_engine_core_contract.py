"""Scheduler result-application regression tests."""

import math
from unittest.mock import Mock

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase
from vllm_rlt.worker.model_runner import Submission


def make_engine(
    *,
    asynchronous=True,
    preemption=False,
    token_budget=16,
    mode="refill",
    prefill_chunk_size=16,
):
    torch.manual_seed(123)
    model = OuroForCausalLM(tiny_ouro_config())
    with torch.no_grad():
        model.model.early_exit_gate.weight.zero_()
        model.model.early_exit_gate.bias.fill_(math.log(0.3 / 0.7))
    return LLMEngine(
        model,
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(
            mode=mode,
            max_num_seqs=4,
            max_num_batched_tokens=token_budget,
            prefill_chunk_size=prefill_chunk_size,
            enable_preemption=preemption,
        ),
        exit_config=ExitConfig("ouro_delayed"),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
    )


def drain(engine):
    outputs = []
    for _ in range(300):
        if not engine.has_unfinished_requests():
            assert engine.cache_manager.num_used_blocks == 0
            return outputs
        outputs.extend(engine.step())
    pytest.fail("engine failed to drain")


def wait_for_signal(engine, request_id):
    for _ in range(100):
        engine.step()
        request = engine.scheduler.requests[request_id]
        if request.generated_token_ids and request.phase == RequestPhase.RECURRENT:
            if request_id in engine._pending_exit_signals:
                return request
    pytest.fail("recurrent signal was not retained")


@pytest.mark.parametrize(
    "asynchronous,threshold,expected_depths,expected_remaining,signal_depths",
    [
        (False, 0.99, [4, 4, 4], 0.343, [1, 2, 3]),
        (True, 0.99, [4, 4, 4], 0.49, [1, 2]),
        (False, 0.5, [4, 3, 3], 0.49, [1, 2]),
        (True, 0.5, [4, 3, 3], 0.49, [1, 2]),
    ],
)
def test_delayed_signal_consumption_at_max_and_threshold(
    monkeypatch, asynchronous, threshold, expected_depths, expected_remaining, signal_depths
):
    engine = make_engine(asynchronous=asynchronous)
    params = SamplingParams(
        max_tokens=3, min_loops=1, max_loops=4, exit_threshold=threshold, ignore_eos=True
    )
    engine.add_request("a", [2, 3], params)
    probabilities = []
    consumed = {}
    enqueue = engine.scheduler.enqueue
    delayed_signal = engine.scheduler._delayed_signal

    def record_enqueue(request, phase):
        if phase == RequestPhase.CODA and request.generated_token_ids:
            probabilities.append(request.remaining_probability)
        return enqueue(request, phase)

    def record_signal(request, score, depth):
        consumed.setdefault(len(request.generated_token_ids), []).append(depth)
        return delayed_signal(request, score, depth)

    monkeypatch.setattr(engine.scheduler, "enqueue", record_enqueue)
    monkeypatch.setattr(engine.scheduler, "_delayed_signal", record_signal)
    result = drain(engine)[-1]
    assert result.exit_depths == expected_depths
    assert probabilities == pytest.approx([expected_remaining] * 2)
    assert consumed == {1: signal_depths, 2: signal_depths}


def test_async_pending_signal_survives_preemption_and_resume():
    params = SamplingParams(max_tokens=4, min_loops=1, exit_threshold=0.5, ignore_eos=True)
    baseline = make_engine()
    baseline.add_request("a", [1, 2, 3], params)
    expected = drain(baseline)[-1]

    engine = make_engine(preemption=True)
    engine.add_request("a", [1, 2, 3], params)
    request = wait_for_signal(engine, "a")
    retained = engine._pending_exit_signals["a"]
    engine.add_request("new", [4], SamplingParams(max_tokens=1, ignore_eos=True))
    assert engine.scheduler.preempt(engine.scheduler.requests["new"])
    assert request.phase == RequestPhase.WAITING
    assert engine._pending_exit_signals["a"] is retained
    actual = drain(engine)
    assert [out for out in actual if out.request_id == "a"][-1] == expected
    assert engine.preemption.resumptions == 1


def test_async_abort_with_pending_signal_then_reuse_id():
    params = SamplingParams(max_tokens=3, exit_threshold=0.5, ignore_eos=True)
    expected_engine = make_engine()
    expected_engine.add_request("a", [5], params)
    expected = drain(expected_engine)[-1]

    engine = make_engine()
    engine.add_request("a", [2, 3], params)
    wait_for_signal(engine, "a")
    assert engine.abort_request("a").finish_reason == "abort"
    assert "a" not in engine._pending_exit_signals
    engine.add_request("a", [5], params)
    assert drain(engine)[-1] == expected


@pytest.mark.parametrize("snapshot", ["positions", "depths"])
def test_stale_async_signal_raises_and_reclaims_state(snapshot):
    engine = make_engine()
    engine.add_request("a", [2], SamplingParams(max_tokens=2, exit_threshold=0.5))
    wait_for_signal(engine, "a")
    handle = engine._pending_exit_signals["a"]
    values = getattr(handle.ticket, snapshot)
    setattr(handle.ticket, snapshot, (values[0] + 1,))
    with pytest.raises(RuntimeError, match="stale lookahead signal"):
        engine.step()
    assert not engine.has_unfinished_requests()
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine._pending_exit_signals and not engine._inflight


@pytest.mark.parametrize(
    "corruption,message", [("index", "out-of-order"), ("count", "invalid pending")]
)
def test_invalid_async_coda_delivery_raises_and_reclaims_state(corruption, message):
    engine = make_engine()
    engine.add_request("a", [2], SamplingParams(max_tokens=1, ignore_eos=True))
    engine.step()  # PREFILL
    engine.step()  # CODA submission
    ticket = engine._pending_coda[0]
    if corruption == "index":
        ticket.output_indices = (1,)
    else:
        engine.scheduler.requests["a"].num_output_placeholders = 0
    with pytest.raises(RuntimeError, match=message):
        engine.step()
    assert not engine.has_unfinished_requests()
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine._pending_coda


def test_sync_result_application_failure_aborts_only_selected_batch(monkeypatch):
    engine = make_engine(asynchronous=False, token_budget=1)
    for rid in ("a", "b"):
        engine.add_request(rid, [2], SamplingParams(max_tokens=1, ignore_eos=True))
    original = engine.scheduler.update_from_output

    def fail(batch, result, **kwargs):
        raise RuntimeError("result application failed")

    monkeypatch.setattr(engine.scheduler, "update_from_output", fail)
    with pytest.raises(RuntimeError, match="result application failed"):
        engine.step()
    assert "a" not in engine.scheduler.requests
    assert "b" in engine.scheduler.requests
    monkeypatch.setattr(engine.scheduler, "update_from_output", original)
    assert drain(engine)[-1].request_id == "b"


def test_async_result_application_failure_synchronizes_and_aborts_all(monkeypatch):
    engine = make_engine()
    engine.add_request("a", [2], SamplingParams(max_tokens=2))
    engine.add_request("b", [3], SamplingParams(max_tokens=2))
    engine.step()  # PREFILL
    engine.step()  # CODA submit
    synchronize = Mock(wraps=engine.model_runner.synchronize)
    monkeypatch.setattr(engine.model_runner, "synchronize", synchronize)

    def fail(batch, result, **kwargs):
        raise RuntimeError("result application failed")

    monkeypatch.setattr(engine.scheduler, "update_from_output", fail)
    with pytest.raises(RuntimeError, match="result application failed"):
        engine.step()
    synchronize.assert_called_once()
    assert not engine.has_unfinished_requests()
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine._inflight and not engine._pending_coda and not engine._pending_exit_signals


def test_later_async_coda_ticket_can_deliver_first():
    engine = make_engine()
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    engine.add_request("first", [2], params)
    engine.step()  # PREFILL
    engine.step()  # first CODA submission
    first = engine._pending_coda[0]

    class HeldEvent:
        def query(self):
            return False

        def synchronize(self):
            pass

    original_event = first.event
    first.event = HeldEvent()
    engine.add_request("second", [3], params)
    engine.step()  # second PREFILL
    engine.step()  # second CODA submission
    assert len(engine._pending_coda) == 2
    delivered = engine._collect_coda()
    assert [out.request_id for out in delivered] == ["second"]
    assert engine._pending_coda == [first]
    first.event = original_event
    assert [out.request_id for out in engine._collect_coda()] == ["first"]
    assert engine.cache_manager.num_used_blocks == 0


def test_old_request_coda_delivery_is_discarded():
    engine = make_engine()
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    engine.add_request("a", [2], params)
    engine.step()
    engine.step()
    old_ticket = engine._pending_coda[0]
    old_request = old_ticket.batch.requests[0].request
    engine.abort_request("a")
    engine.add_request("a", [3], params)
    assert engine.scheduler.requests["a"] is not old_request
    assert engine._deliver_coda(old_ticket) == []
    assert engine.scheduler.requests["a"].generated_token_ids == []
    assert drain(engine)[-1].prompt_token_ids == [3]


def test_old_request_signal_is_discarded_without_collect(monkeypatch):
    engine = make_engine()
    params = SamplingParams(max_tokens=2, exit_threshold=0.5, ignore_eos=True)
    engine.add_request("a", [2], params)
    wait_for_signal(engine, "a")
    old_handle = engine._pending_exit_signals["a"]
    engine.abort_request("a")
    engine.add_request("a", [3], params)
    engine.step()  # PREFILL
    engine.step()  # CODA submit
    engine.step()  # PRELUDE
    engine._pending_exit_signals["a"] = old_handle

    def forbidden_collect():
        pytest.fail("old-request score must not be collected")

    monkeypatch.setattr(old_handle.ticket, "collect", forbidden_collect)
    engine.step()  # first RECURRENT submission of the new request
    assert engine.scheduler.requests["a"] is not old_handle.ticket.batch.requests[0].request
    assert drain(engine)[-1].prompt_token_ids == [3]


def test_stale_lookahead_signal_raises():
    engine = make_engine()
    params = SamplingParams(max_tokens=2, exit_threshold=0.99, ignore_eos=True)
    engine.add_request("a", [2], params)
    wait_for_signal(engine, "a")
    del engine._pending_exit_signals["a"]
    with pytest.raises(RuntimeError, match="stale lookahead signal"):
        engine.step()


def test_async_recurrent_collects_each_prior_score_after_next_submit(monkeypatch):
    engine = make_engine()
    engine.add_request("a", [2], SamplingParams(max_tokens=3, exit_threshold=0.99, ignore_eos=True))
    submit = engine.model_runner.submit
    collect = Submission.collect
    observed = []
    recurrent_submissions = 0
    submitted_at = {}

    def record_submit(scheduled_batch=None, *, batch=None):
        nonlocal recurrent_submissions
        ticket = (
            submit(scheduled_batch, batch=batch) if batch is not None else submit(scheduled_batch)
        )
        if ticket.batch.phase == RequestPhase.RECURRENT:
            recurrent_submissions += 1
            submitted_at[id(ticket)] = recurrent_submissions
        return ticket

    def record_collect(ticket):
        if ticket.batch.phase == RequestPhase.RECURRENT:
            observed.append((submitted_at[id(ticket)], recurrent_submissions))
        return collect(ticket)

    monkeypatch.setattr(engine.model_runner, "submit", record_submit)
    monkeypatch.setattr(Submission, "collect", record_collect)
    assert drain(engine)[-1].exit_depths == [4, 4, 4]
    assert len(observed) == 4  # two scores for each of the two decode tokens
    assert all(source < current for source, current in observed)
    assert [source for source, _ in observed] == sorted(source for source, _ in observed)


def test_finish_waits_for_runner_then_polls_prefix_before_free(monkeypatch):
    engine = make_engine(asynchronous=False)
    engine.add_request("a", [2], SamplingParams(max_tokens=1, ignore_eos=True))
    engine.step()  # PREFILL
    order = []
    release = engine.model_runner.release
    poll = engine.cache_manager.poll_prefixes
    finish = engine.scheduler.finish

    def record_release(request_id):
        order.append("release")
        return release(request_id)

    def record_poll():
        order.append("poll")
        return poll()

    def record_finish(request, reason):
        order.append("free")
        return finish(request, reason)

    monkeypatch.setattr(engine.model_runner, "release", record_release)
    monkeypatch.setattr(engine.cache_manager, "poll_prefixes", record_poll)
    monkeypatch.setattr(engine.scheduler, "finish", record_finish)
    assert engine.step()[0].finished
    assert order == ["release", "poll", "free"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_mixed_recurrent_and_split_prefill_progression_through_scheduler(asynchronous):
    reference = make_engine(asynchronous=False)
    params = SamplingParams(max_tokens=3, min_loops=1, exit_threshold=0.5, ignore_eos=True)
    reference.add_request("decode", [2, 3], params)
    reference.add_request("prompt", [4, 5, 6, 7], params)
    expected = {out.request_id: out for out in drain(reference) if out.finished}

    engine = make_engine(
        asynchronous=asynchronous,
        mode="mixed",
        prefill_chunk_size=2,
        token_budget=4,
    )
    engine.add_request("decode", [2, 3], params)
    for _ in range(10):
        engine.step()
        if (
            engine.scheduler.requests["decode"].generated_token_ids
            and engine.scheduler.requests["decode"].phase is RequestPhase.RECURRENT
        ):
            break
    engine.add_request("prompt", [4, 5, 6, 7], params)
    saw_mixed_batch = False
    actual = {}
    for _ in range(100):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            if out.finished:
                actual[out.request_id] = out
        batch = engine.last_schedule
        if batch is not None and {batch.phase_for(item) for item in batch.requests} == {
            RequestPhase.RECURRENT,
            RequestPhase.PREFILLING_RECURRENT,
        }:
            saw_mixed_batch = True
    assert saw_mixed_batch
    assert actual["decode"].token_ids == expected["decode"].token_ids
    assert actual["decode"].exit_depths == expected["decode"].exit_depths
    assert actual["prompt"].token_ids == expected["prompt"].token_ids
    assert actual["prompt"].exit_depths == expected["prompt"].exit_depths
