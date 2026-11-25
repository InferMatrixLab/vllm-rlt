"""NPU event ownership and device-token lookahead preserve eager execution."""

import pytest
import torch

from tests.test_ascend import make_npu_model
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.scheduler import ScheduledRequest, Scheduler, SchedulerOutput
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.request import Request, RequestPhase


def test_fixed_depth_decode_keeps_each_loop_schedulable():
    scheduler = Scheduler(SchedulerConfig(), None, total_ut_steps=4, exit_mode="ouro_delayed")
    request = Request("a", [1], SamplingParams(min_loops=1, max_loops=4))
    request.phase = RequestPhase.RECURRENT
    scheduler.requests["a"] = request
    for depth in range(4):
        assert request.phase == RequestPhase.RECURRENT
        item = ScheduledRequest(request, depth=depth, phase=RequestPhase.RECURRENT)
        assert scheduler._planned_loop_count([item]) == 1
        scheduler.update_after_submission(SchedulerOutput(RequestPhase.RECURRENT, [item], 1))
        assert request.loops_done == depth + 1
    assert request.phase == RequestPhase.CODA


def make_engine(
    model, asynchronous, multi_stream, *, exit_mode=None, static_buffers=False, pad=False
):
    return StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(num_blocks=128, block_size=16, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=128, prefill_chunk_size=128
        ),
        execution_config=ExecutionConfig(
            async_scheduling=asynchronous,
            multi_stream=multi_stream,
            static_buffers=static_buffers,
            pad_to_power_of_two=pad,
            use_uva=False,
            cuda_graphs=False,
        ),
        exit_config=ExitConfig(exit_mode or ("ouro_delayed" if asynchronous else "ouro")),
    )


def drain(engine):
    got = {}
    for _ in range(300):
        for output in engine.step():
            if output.finished:
                got[output.request_id] = (
                    output.token_ids,
                    output.exit_depths,
                    output.finish_reason,
                )
        if not engine.has_unfinished_requests():
            assert engine.cache_manager.num_used_blocks == 0
            return got
    pytest.fail("NPU async engine did not drain")


@pytest.mark.npu
def test_npu_workspace_retires_its_consumer_before_reuse():
    from vllm_rlt.worker.buffers import Workspace

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    workspace = Workspace(4, 4, 8, device, torch.bfloat16, use_uva=False)
    consumer = torch.npu.Stream(device=device)
    consumer.wait_stream(torch.npu.current_stream(device))
    workspace.acquire()
    with torch.npu.stream(consumer):
        workspace.hidden.fill_(7)
        observed = workspace.hidden.clone()
        workspace.record_done()
    (event,) = workspace.submissions.events
    workspace.acquire()
    assert event.query()
    assert not workspace.submissions.events
    workspace.hidden.fill_(9)
    assert observed.cpu().tolist() == [[7.0] * 8] * 4


@pytest.mark.npu
@pytest.mark.parametrize("multi_stream", [False, True])
@pytest.mark.parametrize(
    "static_buffers,pad",
    [(False, False), (True, False), (True, True)],
    ids=["dynamic", "static", "static-padded"],
)
@torch.inference_mode()
def test_async_npu_matches_sync_without_device_barriers(
    monkeypatch, multi_stream, static_buffers, pad
):
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    recurrent = model.recurrent_prepared
    row_shapes = set()

    def observe_rows(hidden, batch, cache, **kwargs):
        row_shapes.add((len(batch.rows), len(hidden)))
        return recurrent(hidden, batch, cache, **kwargs)

    monkeypatch.setattr(model, "recurrent_prepared", observe_rows)
    prompts = [[3, 5, 7] * 7, [2, 11, 17], [13, 19] * 9]
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    reference = None
    for asynchronous in (False, True):
        engine = make_engine(
            model,
            asynchronous,
            multi_stream,
            static_buffers=asynchronous and static_buffers,
            pad=asynchronous and pad,
        )
        try:
            row_shapes.clear()
            assert bool(engine.model_runner.requestspaces) == (asynchronous and static_buffers)
            for repeat in range(2):
                for i, prompt in enumerate(prompts):
                    engine.add_request(str(i), prompt, params)
                with monkeypatch.context() as patch:

                    def forbidden(*args, **kwargs):
                        pytest.fail("submission called device-wide synchronize")

                    patch.setattr(torch.npu, "synchronize", forbidden)
                    actual = drain(engine)
                assert all(depths == [4] * 4 for _, depths, _ in actual.values())
                if reference is None:
                    reference = actual
                else:
                    assert actual == reference
            if asynchronous:
                assert (3, 4 if pad else 3) in row_shapes
        finally:
            engine.close()


@pytest.mark.npu
@pytest.mark.parametrize(
    "exit_mode, async_schedule, multi_stream, pad",
    [
        ("ouro", False, False, False),
        ("ouro", False, True, False),
        ("ouro_delayed", True, False, False),
        ("ouro_delayed", True, True, False),
        ("ouro_delayed", True, True, True),
    ],
    ids=[
        "ouro-sync",
        "ouro-sync-multi",
        "delayed-async-single",
        "delayed-async-multi",
        "delayed-async-padded",
    ],
)
@torch.inference_mode()
def test_npu_adaptive_exit_then_deeper_decode(
    monkeypatch, exit_mode, async_schedule, multi_stream, pad
):
    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    model = make_npu_model(device)
    recurrent = model.recurrent_prepared

    def alternate_gate(hidden, batch, cache, **kwargs):
        hidden, gate = recurrent(hidden, batch, cache, **kwargs)
        if gate is not None:
            # Decode starts at position 15. Walk into the next page at full
            # depth, which needs the KV copied from the earlier exit.
            gate = batch.position_ids.remainder(2).to(gate.dtype) * 20 - 10
        return hidden, gate

    monkeypatch.setattr(model, "recurrent_prepared", alternate_gate)
    adaptive = SamplingParams(max_tokens=5, min_loops=2, exit_threshold=0.9, ignore_eos=True)
    fixed = SamplingParams(max_tokens=5, ignore_eos=True)
    capped = SamplingParams(max_tokens=5, min_loops=1, max_loops=1, ignore_eos=True)
    early = 2 if exit_mode == "ouro" else 3

    def run(asynchronous, use_multi_stream):
        engine = make_engine(
            model,
            asynchronous,
            use_multi_stream,
            exit_mode=exit_mode,
            static_buffers=asynchronous and pad,
            pad=asynchronous and pad,
        )
        try:
            with pytest.raises(ValueError, match="exceed the model's supported depth"):
                engine.add_request("invalid", [3], SamplingParams(max_loops=5))
            with pytest.raises(ValueError, match="require greedy sampling"):
                engine.add_request("invalid", [3], SamplingParams(temperature=0.5))
            engine.add_request("adaptive", [3, 7, 11] * 5, adaptive)
            for _ in range(30):
                if any(out.token_ids for out in engine.step()):
                    break
            else:
                pytest.fail("adaptive request did not produce its first token")
            engine.add_request("fixed", [13, 17, 19], fixed)
            engine.add_request("capped", [3, 5, 7], capped)
            return drain(engine)
        finally:
            engine.close()

    expected = run(False, False)
    assert expected["adaptive"][1] == [4, early, 4, early, 4]
    assert expected["fixed"][1] == [4] * 5
    assert expected["capped"][1] == [4, 1, 1, 1, 1]
    if async_schedule or multi_stream:
        assert run(async_schedule, multi_stream) == expected


@pytest.mark.npu
@torch.inference_mode()
def test_npu_adaptive_abort_discards_pending_gate_before_request_id_reuse():
    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    model = make_npu_model(device)
    model.model.early_exit_gate.weight.zero_()
    model.model.early_exit_gate.bias.fill_(10)
    params = SamplingParams(max_tokens=4, min_loops=2, exit_threshold=0.9, ignore_eos=True)
    engine = make_engine(model, True, True)
    try:
        engine.add_request("a", [3, 7, 11], params)
        for _ in range(30):
            engine.step()
            if "a" in engine._pending_exit_signals:
                break
        else:
            pytest.fail("adaptive request did not retain a gate result")
        engine.abort_request("a")
        assert "a" not in engine._pending_exit_signals
        engine.add_request("a", [13, 17, 19], params)
        assert drain(engine)["a"][1] == [4, 3, 3, 3]
    finally:
        engine.close()


@pytest.mark.npu
@torch.inference_mode()
def test_npu_delayed_delivery_consumes_device_token_before_host_result():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    engine = make_engine(model, True, False)
    params = SamplingParams(max_tokens=3, min_loops=4, max_loops=4, ignore_eos=True)

    class HeldEvent:
        def query(self):
            return False

        def synchronize(self):
            pytest.fail("host delivery was required before the next device input")

    try:
        engine.add_request("a", [3, 7, 11], params)
        engine.step()
        engine.step()
        ticket = engine._pending_coda[0]
        real_event = ticket.event
        ticket.event = HeldEvent()
        try:
            request = engine.scheduler.requests["a"]
            assert request.generated_token_ids == []
            engine.step()
            assert engine.last_schedule.phase == RequestPhase.PRELUDE
            for depth in range(1, 5):
                engine.step()
                assert engine.last_schedule.phase == RequestPhase.RECURRENT
                assert request.loops_done == depth
                assert request.generated_token_ids == []
            assert ticket.device_values.device.type == "npu"
        finally:
            ticket.event = real_event
        assert drain(engine)["a"][1] == [4, 4, 4]
    finally:
        engine.close()


@pytest.mark.npu
@torch.inference_mode()
def test_npu_cancel_reuse_rejects_late_coda_and_eos_stops():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    params = SamplingParams(max_tokens=3, min_loops=4, max_loops=4, ignore_eos=True)
    expected_engine = make_engine(model, False, False)
    expected_engine.add_request("cancel", [13, 17, 19], params)
    expected_engine.add_request("keep", [3, 7, 11], params)
    expected = drain(expected_engine)
    expected_engine.close()
    engine = make_engine(model, True, True)
    try:
        engine.add_request("cancel", [2, 5] * 9, params)
        engine.add_request("keep", [3, 7, 11], params)
        engine.step()
        engine.step()
        engine.abort_request("cancel")
        engine.add_request("cancel", [13, 17, 19], params)
        assert drain(engine) == expected
        eos = model.config.eos_token_id
        engine.model_runner.sampler.sample_batch = lambda logits, options, generators: (
            logits.new_full((len(options),), eos, dtype=torch.long)
        )
        engine.add_request("eos", [3, 7], SamplingParams(max_tokens=4, min_loops=4, max_loops=4))
        assert drain(engine)["eos"] == ([eos], [4], "stop")
    finally:
        engine.close()


@pytest.mark.npu
def test_npu_batched_greedy_preserves_ties_and_padding():
    from vllm_rlt.worker.ascend_model_runner import AscendSampler

    device = resolve_device("npu:0")
    logits = torch.zeros((3, 49152), dtype=torch.bfloat16)
    logits[0, 1] = 3
    logits[0, 40000] = 3
    logits[1, 0] = 4
    logits[1, -1] = 4
    logits[2, 0] = 9
    logits = logits.to(device)
    params = (SamplingParams(), SamplingParams())
    actual = AscendSampler().sample_batch(logits, params, (None, None))
    expected = torch.stack([row.argmax() for row in logits[:2]])
    assert actual.cpu().tolist() == [1, 0]
    assert torch.equal(actual, expected)


@pytest.mark.npu
@torch.inference_mode()
def test_npu_hidden_gather_keeps_cross_stream_sources_alive():
    from types import SimpleNamespace

    from vllm_rlt.worker.model_runner import ModelRunner

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    producer = torch.npu.Stream(device=device)
    consumer = torch.npu.Stream(device=device)
    blocker = torch.npu.Stream(device=device)
    with torch.npu.stream(producer):
        source = torch.full((64,), 7.0, dtype=torch.bfloat16, device=device)
    producer.synchronize()
    with torch.npu.stream(consumer):
        warm = torch.stack([source])
    consumer.synchronize()
    del warm
    pointer = source.data_ptr()
    request = SimpleNamespace(request_id="a", hidden_state=source)
    batch = SimpleNamespace(spans=[SimpleNamespace(request=request)])
    matrix = torch.ones((16384, 16384), dtype=torch.bfloat16, device=device)
    product = torch.empty_like(matrix)
    blocker.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(blocker):
        for _ in range(4):
            torch.mm(matrix, matrix, out=product)
        gate = torch.npu.Event()
        gate.record(blocker)
    consumer.wait_event(gate)
    runner = SimpleNamespace(events={})
    try:
        with torch.npu.stream(consumer):
            ModelRunner._wait_for_requests(runner, batch, consumer)
            actual = ModelRunner._gather(runner, [request], None, 1)
        request.hidden_state = None
        del source
        poisoned = []
        with torch.npu.stream(producer):
            for _ in range(32):
                value = torch.empty((64,), dtype=torch.bfloat16, device=device)
                value.fill_(99.0)
                poisoned.append(value)
        assert not gate.query(), "consumer delay expired before the source reuse check"
        assert all(value.data_ptr() != pointer for value in poisoned)
        producer.synchronize()
        consumer.synchronize()
        assert actual.cpu().tolist() == [[7.0] * 64]
    finally:
        producer.synchronize()
        consumer.synchronize()
        blocker.synchronize()
