"""Ascend decode scheduling, refill and asynchronous result delivery."""

import pytest
import torch

from tests.helpers import make_cpu_cache, tiny_ouro_config
from tests.test_ascend import make_npu_model
from tests.test_npu_async import drain
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.scheduler import ScheduledRequest, Scheduler, SchedulerOutput
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.request import Request, RequestPhase


def test_decode_prelude_leaves_recurrent_work_to_scheduler():
    scheduler = Scheduler(SchedulerConfig(), None, total_ut_steps=4, exit_mode="ouro_delayed")
    request = Request("a", [1], SamplingParams(min_loops=4, max_loops=4))
    request.phase = RequestPhase.PRELUDE
    scheduler.requests["a"] = request
    item = ScheduledRequest(request, phase=RequestPhase.PRELUDE)
    assert scheduler._planned_loop_count([item]) == 0
    batch = SchedulerOutput(RequestPhase.PRELUDE, [item], 0)
    update = scheduler.update_after_submission(batch)
    assert request.loops_done == 0
    assert request.phase == RequestPhase.RECURRENT
    assert update.exited == []
    request.phase = RequestPhase.PRELUDE
    scheduler.requests.pop("a")
    scheduler.update_after_submission(batch)
    assert request.phase == RequestPhase.PRELUDE  # A stale submission cannot advance a request.


@pytest.mark.parametrize("late_arrival", [False, True])
def test_decode_preserves_refill_opportunity(late_arrival):
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=3, prefill_chunk_size=3),
        make_cpu_cache(tiny_ouro_config(), blocks=128),
        total_ut_steps=4,
        exit_mode="ouro_delayed",
    )
    params = SamplingParams(max_tokens=16, min_loops=4, max_loops=4, ignore_eos=True)
    first = Request("a", [1, 2, 3], params)
    second = Request("b", [4, 5, 6], params)
    scheduler.add_request(first)
    if not late_arrival:
        scheduler.add_request(second)
    batch = scheduler.schedule()
    assert batch.phase == RequestPhase.PREFILL
    assert [item.request for item in batch.requests] == [first]
    scheduler.update_after_submission(batch)
    batch = scheduler.schedule()
    assert batch.phase == RequestPhase.CODA
    scheduler.update_after_submission(batch)
    if late_arrival:
        scheduler.add_request(second)
    assert second.phase == (RequestPhase.WAITING if late_arrival else RequestPhase.PREFILL)

    # Every prompt consumes a complete token budget. Refill must still get a
    # turn before the first generation ends.
    for _ in range(8):
        batch = scheduler.schedule()
        assert batch is not None
        scheduler.update_after_submission(batch)
        if any(item.request is second for item in batch.requests):
            assert batch.phase == RequestPhase.PREFILL
            assert first.num_scheduled_outputs < params.max_tokens
            break
    else:
        pytest.fail("decode boundaries starved a pending prompt")
    targets = {request.request_id: request.num_scheduled_outputs + 1 for request in (first, second)}
    for _ in range(12):
        batch = scheduler.schedule()
        assert batch is not None
        scheduler.update_after_submission(batch)
        if all(
            request.num_scheduled_outputs >= targets[request.request_id]
            for request in (first, second)
        ):
            break
    else:
        pytest.fail("refilled decode starved the existing recurrent request")


def engine_for(model, multi_stream=False, token_budget=128):
    return StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(num_blocks=128, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=token_budget, prefill_chunk_size=128
        ),
        execution_config=ExecutionConfig(
            async_scheduling=True,
            multi_stream=multi_stream,
            use_uva=False,
        ),
        exit_config=ExitConfig("ouro_delayed"),
    )


@pytest.mark.parametrize("waiting", [False, True])
def test_decode_prelude_keeps_recurrent_due(waiting):
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=2, max_prefill_batches_before_decode=4),
        make_cpu_cache(tiny_ouro_config(), blocks=128),
        total_ut_steps=4,
        exit_mode="ouro_delayed",
    )
    params = SamplingParams(max_tokens=8, min_loops=4, max_loops=4, ignore_eos=True)
    scheduler.add_request(Request("a", [1, 2, 3], params))
    for expected in (RequestPhase.PREFILL, RequestPhase.CODA):
        batch = scheduler.schedule()
        assert batch.phase == expected
        scheduler.update_after_submission(batch)
    if waiting:
        scheduler.add_request(Request("b", [4, 5, 6], params))
    for _ in range(scheduler.config.max_prefill_batches_before_decode):
        scheduler.policy.record_batch(RequestPhase.PREFILL)
    assert scheduler.policy.decode_due
    batch = scheduler.schedule()
    assert batch.phase == RequestPhase.PRELUDE
    assert batch.loop_count == 0
    assert scheduler.policy.decode_due


@pytest.mark.npu
@torch.inference_mode()
@pytest.mark.parametrize("multi_stream", [False, True])
def test_ascend_decode_refills_during_generation(multi_stream):
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    params = SamplingParams(max_tokens=8, min_loops=4, max_loops=4, ignore_eos=True)
    engine = engine_for(model, multi_stream, token_budget=3)
    try:
        for name, prompt in [("a", [3, 7, 11]), ("b", [13, 17, 19])]:
            engine.add_request(name, prompt, params)
        actual, first_finished, second_started = {}, False, False
        first_tokens_at_second_start = None
        first_tokens, first_advanced = 0, False
        for _ in range(128):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                if out.request_id == "b" and not second_started:
                    assert not first_finished
                    second_started = True
                    first_tokens_at_second_start = first_tokens
                if out.request_id == "a":
                    first_tokens = len(out.token_ids)
                    if second_started and "b" not in actual:
                        first_advanced |= first_tokens > first_tokens_at_second_start
                if out.finished:
                    actual[out.request_id] = (
                        list(out.token_ids),
                        list(out.exit_depths),
                        out.finish_reason,
                    )
                    if out.request_id == "a":
                        first_finished = True
        else:
            pytest.fail("refill did not drain in 128 steps")
        assert second_started and first_advanced
        assert set(actual) == {"a", "b"}
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


@pytest.mark.npu
@torch.inference_mode()
@pytest.mark.parametrize("multi_stream", [False, True])
def test_ascend_decode_preserves_cancel_reuse_and_eos(multi_stream):
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    engine = engine_for(model, multi_stream)
    try:
        for i, prompt in enumerate([[3, 7, 11], [13, 17, 19]]):
            engine.add_request(str(i), prompt, params)
        expected = drain(engine)
        assert engine.cache_manager.num_used_blocks == 0
        engine.add_request("0", [2, 5, 9], params)
        engine.add_request("1", [13, 17, 19], params)
        engine.step()
        engine.step()
        engine.step()
        engine.abort_request("0")
        engine.add_request("0", [3, 7, 11], params)
        assert drain(engine) == expected
        eos = model.config.eos_token_id
        engine.model_runner.sampler.sample_batch = lambda logits, options, generators: (
            logits.new_full((len(options),), eos, dtype=torch.long)
        )
        engine.add_request(
            "eos", [3, 7, 11], SamplingParams(max_tokens=4, min_loops=4, max_loops=4)
        )
        assert drain(engine)["eos"] == ([eos], [4], "stop")
    finally:
        engine.close()


@pytest.mark.npu
@torch.inference_mode()
def test_ascend_prelude_uses_device_sample_before_coda_delivery():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    engine = engine_for(model)

    class HeldEvent:
        def query(self):
            return False

        def synchronize(self):
            pytest.fail("decode plan waited for host token delivery")

    try:
        engine.add_request(
            "a", [3, 7, 11], SamplingParams(max_tokens=3, min_loops=4, max_loops=4, ignore_eos=True)
        )
        engine.step()
        engine.step()
        ticket = engine._pending_coda[0]
        actual_event = ticket.event
        ticket.event = HeldEvent()
        try:
            engine.step()
            assert engine.last_schedule.phase == RequestPhase.PRELUDE
            assert engine.last_schedule.loop_count == 0
            request = engine.scheduler.requests["a"]
            assert request.loops_done == 0 and request.generated_token_ids == []
        finally:
            ticket.event = actual_event
        assert drain(engine)["a"][1] == [4, 4, 4]
    finally:
        engine.close()
