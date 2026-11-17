"""Fixed decode plans keep device inputs and preserve result retirement boundaries."""

import pytest
import torch

from tests.test_ascend import make_npu_model
from tests.test_npu_async import drain
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.scheduler import ScheduledRequest, Scheduler, SchedulerOutput
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.request import Request, RequestPhase


@pytest.mark.parametrize(
    "exit_mode,min_loops,expected",
    [("ouro_delayed", 4, 4), ("ouro_delayed", 1, 0), ("trace", 4, 0)],
)
def test_fixed_prelude_planning_and_submission_progress(exit_mode, min_loops, expected):
    scheduler = Scheduler(
        SchedulerConfig(), None, total_ut_steps=4, exit_mode=exit_mode, coalesce_fixed_prelude=True
    )
    request = Request("a", [1], SamplingParams(min_loops=min_loops, max_loops=4))
    request.phase = RequestPhase.PRELUDE
    scheduler.requests["a"] = request
    item = ScheduledRequest(request, phase=RequestPhase.PRELUDE)
    assert scheduler._planned_loop_count([item]) == expected
    update = scheduler.update_after_submission(
        SchedulerOutput(RequestPhase.PRELUDE, [item], expected)
    )
    assert request.loops_done == expected
    assert request.phase == (RequestPhase.CODA if expected else RequestPhase.RECURRENT)
    assert update.exited == ([request] if expected else [])
    request.phase = RequestPhase.PRELUDE
    scheduler.requests.pop("a")
    scheduler.update_after_submission(SchedulerOutput(RequestPhase.PRELUDE, [item], expected))
    assert request.phase == RequestPhase.PRELUDE  # Stale submissions cannot advance a reused ID.


def engine_for(model, planned, multi_stream=False):
    return StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(num_blocks=128, block_size=16, contiguous_kv=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=128, prefill_chunk_size=128
        ),
        execution_config=ExecutionConfig(
            async_scheduling=True,
            multi_stream=multi_stream,
            use_uva=False,
            coalesce_fixed_prelude=planned,
        ),
        exit_config=ExitConfig("ouro_delayed"),
    )


@pytest.mark.npu
@torch.inference_mode()
@pytest.mark.parametrize("multi_stream", [False, True])
def test_fixed_prelude_preserves_generation_cancel_reuse_and_eos(multi_stream):
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    expected = None
    for planned in [False, True]:
        engine = engine_for(model, planned, multi_stream)
        try:
            for i, prompt in enumerate([[3, 7, 11], [13, 17, 19]]):
                engine.add_request(str(i), prompt, params)
            actual = drain(engine)
            if expected is None:
                expected = actual
            else:
                assert actual == expected
            assert engine.cache_manager._key_cache is None
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
def test_fixed_prelude_uses_device_sample_before_coda_delivery():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    engine = engine_for(model, True)

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
            assert engine.last_schedule.loop_count == 4
            request = engine.scheduler.requests["a"]
            assert request.loops_done == 4 and request.generated_token_ids == []
        finally:
            ticket.event = actual_event
        assert drain(engine)["a"][1] == [4, 4, 4]
    finally:
        engine.close()
