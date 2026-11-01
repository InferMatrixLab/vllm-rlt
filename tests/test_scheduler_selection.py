"""Selection protection is batch-local state, not Scheduler-global state.

Preemption must not suspend a request the Scheduler has already placed in the
batch under construction. That exclusion belongs to one batch construction, so
it must not survive the batch, reach a direct `_take()` caller such as the PD
prefill worker, or outlive the request whose ID it names.
"""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase


def model():
    torch.manual_seed(123)
    return OuroForCausalLM(tiny_ouro_config())


def engine(max_num_seqs=2, **scheduler_options):
    return LLMEngine(
        model(),
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            enable_preemption=True,
            **scheduler_options,
        ),
        execution_config=ExecutionConfig(),
    )


def params(max_tokens=5):
    return SamplingParams(max_tokens=max_tokens, ignore_eos=True)


@pytest.mark.parametrize("wavefront", [False, True])
def test_batch_selection_is_never_preempted_by_a_later_item(monkeypatch, wavefront):
    """Selecting A must stop B's capacity failure from evicting A."""
    e = engine(wavefront_chunked_prefill=wavefront)
    for name in ("a", "b"):
        e.add_request(name, [1, 2], params())
    e.scheduler._admit()
    monkeypatch.setattr(e.cache_manager, "ensure_capacity", lambda rid, frontier: rid != "b")
    phase = RequestPhase.PREFILLING_PRELUDE if wavefront else RequestPhase.PREFILL
    batch = e.scheduler._take(phase)
    assert [scheduled_request.request.request_id for scheduled_request in batch.requests] == ["a"]
    assert e.preemption.preemptions == 0
    assert e.scheduler.requests["a"].phase == phase


def test_wavefront_does_not_schedule_a_candidate_preempted_during_selection(monkeypatch):
    e = engine(wavefront_chunked_prefill=True, prefill_chunk_size=2)
    for name in ("a", "b"):
        e.add_request(name, [1, 2, 3, 4], params())
    e.scheduler._admit()
    # A needs B's pages. B is already in the local candidate list when suspended.
    monkeypatch.setattr(
        e.cache_manager,
        "ensure_capacity",
        lambda rid, frontier: rid == "a" and not e.cache_manager.has_allocation("b"),
    )

    batch = e.scheduler._take(RequestPhase.PREFILLING_PRELUDE)

    assert {item.request.request_id for item in batch.requests} == {"a"}
    assert e.preemption.preemptions == 1
    assert list(e.scheduler.queues[RequestPhase.WAITING]) == ["b"]
    assert not e.scheduler.queues[RequestPhase.PREFILLING_PRELUDE]
    assert e.scheduler.requests["b"].phase == RequestPhase.WAITING
    e.close()


def test_direct_take_does_not_inherit_an_earlier_batch_selection(monkeypatch):
    """The PD prefill worker calls _take() directly, without schedule()."""
    e = engine()
    for name in ("a", "b"):
        e.add_request(name, [1, 2], params())
    e.scheduler._admit()
    # A first batch selects both requests.
    assert e.scheduler._take(RequestPhase.PREFILL) is not None

    seen = []

    def spy(requester, *, priority_only=False, excluded=frozenset()):
        seen.append((requester.request_id, excluded))
        return False

    # The Scheduler holds the callback it was given, so replace that reference.
    monkeypatch.setattr(e.scheduler, "preempt", spy)
    for name in ("a", "b"):
        e.scheduler.enqueue(e.scheduler.requests[name], RequestPhase.PREFILL)
    monkeypatch.setattr(e.cache_manager, "ensure_capacity", lambda rid, frontier: rid != "b")
    assert e.scheduler._take(RequestPhase.PREFILL) is not None
    # Only "a", selected in this batch, is excluded. The first batch's
    # selection is not still in force, and the set is handed over frozen.
    assert seen == [("b", frozenset({"a"}))]


def test_request_selected_in_an_earlier_batch_is_still_preemptable():
    """A previously scheduled request remains a legitimate preemption victim."""
    e = engine(max_num_seqs=1)
    e.add_request("victim", [1, 2], params())
    e.step()
    assert e.scheduler.requests["victim"].phase != RequestPhase.WAITING
    e.add_request("requester", [3, 4], params())
    assert e.scheduler.preempt(e.scheduler.requests["requester"])
    assert "victim" in e.preemption.snapshots


def test_reused_request_id_is_not_protected_by_the_previous_request():
    """Cancelling a request must not leave its ID protected for a new one."""
    e = engine(max_num_seqs=1)
    e.add_request("r", [1, 2], params())
    e.step()
    e.abort_request("r")
    e.add_request("r", [3, 4], params())
    e.step()
    e.add_request("requester", [5, 6], params())
    assert e.scheduler.preempt(e.scheduler.requests["requester"])
    assert "r" in e.preemption.snapshots


@pytest.mark.parametrize("boundary", [RequestPhase.PRELUDE, RequestPhase.CODA])
def test_refill_runs_boundary_restored_during_admission(boundary):
    """A sole resumed boundary must run in the admission step that restores it."""
    e = engine(max_num_seqs=1)
    e.add_request("resumed", [1, 2], params(max_tokens=3))
    request = e.scheduler.requests["resumed"]
    for _ in range(20):
        if request.phase == boundary:
            break
        e.step()
    else:
        pytest.fail(f"request did not reach {boundary.value}")

    generated_before = len(request.generated_token_ids)
    e.add_request("pressure", [3, 4], params(max_tokens=1))
    assert e.scheduler.preempt(e.scheduler.requests["pressure"])
    assert e.preemption.snapshots["resumed"]["phase"] == boundary
    assert request.phase == RequestPhase.WAITING
    e.abort_request("pressure")

    outputs = e.step()

    assert e.preemption.resumptions == 1
    assert e.last_schedule.phase == boundary
    if boundary == RequestPhase.PRELUDE:
        assert not outputs
        assert request.phase == RequestPhase.RECURRENT
        assert len(request.generated_token_ids) == generated_before
    else:
        assert [output.request_id for output in outputs] == ["resumed"]
        assert request.phase == RequestPhase.PRELUDE
        assert len(request.generated_token_ids) == generated_before + 1


def test_refill_restored_coda_waits_for_recurrent_batchmate():
    """Post-admission CODA keeps the same batching rule as an existing CODA."""
    e = engine(
        max_num_seqs=2,
        min_coda_batch_size=2,
        max_prefill_batches_before_decode=100,
    )
    e.add_request("core", [1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
    core = e.scheduler.requests["core"]
    for _ in range(20):
        if core.phase == RequestPhase.RECURRENT:
            break
        e.step()
    else:
        pytest.fail("core request did not reach recurrent execution")

    e.add_request(
        "restored-coda",
        [3, 4],
        SamplingParams(max_tokens=1, ignore_eos=True, priority=10),
    )
    e.step()
    restored = e.scheduler.requests["restored-coda"]
    assert core.phase == RequestPhase.RECURRENT
    assert restored.phase == RequestPhase.CODA

    e.add_request("pressure", [5, 6], params(max_tokens=1))
    assert e.scheduler.preempt(e.scheduler.requests["pressure"])
    assert e.preemption.snapshots["restored-coda"]["phase"] == RequestPhase.CODA
    e.abort_request("pressure")

    e.step()

    assert e.preemption.resumptions == 1
    assert e.last_schedule.phase == RequestPhase.RECURRENT
    assert restored.phase == RequestPhase.CODA

    for _ in range(20):
        e.step()
        if e.last_schedule.phase == RequestPhase.CODA:
            break
    else:
        pytest.fail("CODA requests did not form a batch")
    assert {
        scheduled_request.request.request_id for scheduled_request in e.last_schedule.requests
    } == {
        "core",
        "restored-coda",
    }
    assert not e.has_unfinished_requests()
    assert not e.preemption.snapshots
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_repeated_pressure_reclaims_earlier_batches(mode):
    """Pressure must keep reclaiming earlier batches, not only the newest one."""
    e = LLMEngine(
        model(),
        cache_config=CacheConfig(24, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, prefill_chunk_size=2, mode=mode, enable_preemption=True
        ),
    )
    for name in range(3):
        e.add_request(str(name), [1, 2], SamplingParams(max_tokens=8, ignore_eos=True))
    outputs = {}
    for _ in range(1000):
        if not e.has_unfinished_requests():
            break
        for output in e.step():
            if output.finished:
                outputs[output.request_id] = output
    else:
        pytest.fail("engine did not drain")
    assert sorted(outputs) == ["0", "1", "2"]
    assert outputs["0"].token_ids == outputs["1"].token_ids == outputs["2"].token_ids
    assert e.preemption.preemptions > 0
    assert not e.preemption.snapshots
    assert e.cache_manager.num_used_blocks == 0
