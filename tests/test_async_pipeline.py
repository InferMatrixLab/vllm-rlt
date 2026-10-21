"""Pipeline invariants: CPU delivery is not a prerequisite for GPU progress."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Operation, RequestPhase
from vllm_rlt.worker.model_runner import Submission


def make_engine(
    *, asynchronous=True, device="cpu", multi_stream=False, static=False, seqs=3, **kwargs
):
    torch.manual_seed(123)
    base = OuroForCausalLM(tiny_ouro_config()).to(device)
    return LLMEngine(
        base,
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=seqs, max_num_batched_tokens=seqs),
        execution_config=ExecutionConfig(
            async_scheduling=asynchronous,
            multi_stream=multi_stream,
            static_buffers=static,
            pad_to_power_of_two=static,
        ),
        exit_config=kwargs.pop("exit_config", ExitConfig("ouro_delayed")),
        attention_backend="triton" if device == "cuda" else "torch",
        **kwargs,
    )


def drain(engine):
    results = []
    for _ in range(300):
        if not engine.has_unfinished_requests():
            assert engine.cache_manager.num_used_blocks == 0
            assert all(not q for q in engine.scheduler.queues.values())
            return results
        results.extend(engine.step())
    pytest.fail("async pipeline failed to drain")


class HeldEvent:
    complete = False

    def query(self):
        return self.complete

    def synchronize(self):
        self.complete = True


@pytest.mark.parametrize("max_tokens", [1, 2, 5])
@pytest.mark.parametrize("static", [False, True])
def test_delayed_delivery_uses_snapshot_depths_and_bounds_placeholders(
    monkeypatch, max_tokens, static
):
    traces = {"a": [4, 1, 4, 2, 3], "b": [4, 3, 2, 4, 1]}
    options = dict(exit_config=ExitConfig("trace", depths_by_request=traces), static=static)
    params = SamplingParams(max_tokens=max_tokens, min_loops=1, ignore_eos=True)
    sync = make_engine(asynchronous=False, **options)
    engine = make_engine(**options)
    for e in (sync, engine):
        e.add_request("a", [2, 3], params)
        e.add_request("b", [4], params)
    expected = drain(sync)
    original_ready = Submission.ready

    def ready(ticket):
        # Force delivery to lag until the one-output speculation bound drains it.
        return False if ticket.batch.phase == RequestPhase.CODA else original_ready(ticket)

    monkeypatch.setattr(Submission, "ready", ready)
    actual = drain(engine)
    for rid in traces:
        left = [o for o in expected if o.request_id == rid]
        right = [o for o in actual if o.request_id == rid]
        assert right == left
        assert [len(o.token_ids) for o in right] == list(range(1, max_tokens + 1))
    assert not engine._pending_coda


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("multi_stream", [False, True])
def test_prelude_consumes_device_sample_before_cpu_delivery(monkeypatch, device, multi_stream):
    engine = make_engine(device=device, multi_stream=multi_stream)
    engine.add_request("a", [2], SamplingParams(max_tokens=3, ignore_eos=True, exit_threshold=1))
    engine.step()
    engine.step()
    ticket = engine._pending_coda[0]
    real_event, held = ticket.event, HeldEvent()
    ticket.event = held  # Hold only host delivery; preserve real GPU dependencies.
    request = engine.scheduler.requests["a"]
    assert request.generated_token_ids == []
    assert request.num_output_placeholders == 1
    assert request.position == 1
    observed = []
    original = engine.model.prelude

    def prelude(tokens):
        assert tokens.device == ticket.device_values.device
        observed.append(tokens)
        return original(tokens)

    monkeypatch.setattr(engine.model, "prelude", prelude)
    engine.step()
    assert engine.last_schedule.phase == RequestPhase.PRELUDE
    assert not held.complete and not request.generated_token_ids
    assert request.loops_done == 0
    engine.step()
    assert engine.last_schedule.phase == RequestPhase.RECURRENT
    assert not held.complete and not request.generated_token_ids
    engine.model_runner.synchronize()
    assert torch.equal(observed[0], ticket.device_values)
    ticket.event = real_event
    actual = drain(engine)
    assert actual[0].exit_depths == [4]  # Not the next token's current depth (1).
    assert actual[-1].exit_depths == [4, 4, 4]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("static", [False, True])
def test_delayed_eos_discards_next_work_and_reclaims_after_gpu_completion(
    monkeypatch, device, static
):
    engine = make_engine(device=device, static=static, multi_stream=True)
    eos = engine.model.config.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    samples = []

    def sample(logits, params, generator):
        samples.append("a")
        return logits.new_tensor(eos, dtype=torch.long)

    monkeypatch.setattr(engine.model_runner.sampler, "sample", sample)
    engine.add_request("a", [2], SamplingParams(max_tokens=5, exit_threshold=1))
    engine.step()
    engine.step()
    ticket = engine._pending_coda[0]
    real_event, held = ticket.event, HeldEvent()
    ticket.event = held
    engine.step()  # Speculative next prelude.
    engine.step()  # Speculative next core; writes KV, but produces no output.
    assert not held.complete
    assert engine.cache_manager.num_used_blocks > 0
    ticket.event = real_event
    outputs = drain(engine)
    assert len(outputs) == 1
    assert outputs[0].token_ids == [eos]
    assert outputs[0].exit_depths == [4]
    assert outputs[0].finish_reason == "stop"
    assert samples == ["a"]
    assert engine.model_runner.state_slots == {}
    engine.add_request("a", [3], SamplingParams(max_tokens=1, ignore_eos=True))
    assert drain(engine)[-1].finish_reason == "length"


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True])
def test_cpu_prepare_finishes_while_previous_gpu_forward_is_inflight(monkeypatch, multi_stream):
    if torch.version.hip:
        pytest.skip("CUDA sleep topology probe is NVIDIA-specific")
    engine = make_engine(device="cuda", multi_stream=multi_stream)
    params = SamplingParams(max_tokens=3, ignore_eos=True, exit_threshold=1)
    engine.add_request("warm", [2], params)
    drain(engine)
    runner = engine.model_runner
    execute, prepare = runner._execute, runner.prepare_on_cpu
    delayed = torch.cuda.Event()
    injected = False
    observations = []

    def execute_with_delay(batch, prepared=None):
        nonlocal injected
        if (
            injected
            and not delayed.query()
            and multi_stream
            and batch.phase == RequestPhase.RECURRENT
        ):
            # H2D completes on the copy stream while the preceding core is
            # still busy. It must not depend on core or boundary completion.
            assert prepared.kv.position_ids.is_cuda
            runner.copy_stream.synchronize()
            assert not delayed.query()
        result = execute(batch, prepared)
        if batch.phase == RequestPhase.RECURRENT and not injected:
            # Extend a forward to make the ordering test deterministic. This is
            # a topology test, not performance evidence.
            torch.cuda._sleep(150_000_000)
            delayed.record(torch.cuda.current_stream())
            injected = True
        return result

    def inspect_prepare(batch, **kwargs):
        result = prepare(batch, **kwargs)
        if injected and not observations and batch.operation == Operation.RECURRENT:
            assert not delayed.query(), "prepare waited for the preceding forward"
            assert result.kv.position_ids.device.type == "cpu"
            assert result.kv.position_ids.is_pinned()
            observations.append(True)
        return result

    monkeypatch.setattr(runner, "_execute", execute_with_delay)
    monkeypatch.setattr(runner, "prepare_on_cpu", inspect_prepare)
    engine.add_request("a", [2], params)
    drain(engine)
    assert observations == [True]
    assert not runner.requestspaces
    assert runner.states is runner.async_state.hidden
    assert not runner.async_state.owners
    assert len(runner.submissions.events) <= 3


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True])
def test_bounded_pipeline_survives_slow_gpu_and_dma_bank_reuse(monkeypatch, multi_stream):
    if torch.version.hip:
        pytest.skip("CUDA sleep topology probe is NVIDIA-specific")
    engine = make_engine(device="cuda", multi_stream=multi_stream, seqs=1)
    sync = make_engine(device="cuda", asynchronous=False, seqs=1)
    params = SamplingParams(max_tokens=12, ignore_eos=True, exit_threshold=1)
    sync.add_request("a", [2], params)
    expected = drain(sync)[-1]
    runner = engine.model_runner
    original = runner._execute
    maximum = 0

    def execute(batch, prepared=None):
        nonlocal maximum
        maximum = max(maximum, len(runner.submissions.events))
        if batch.phase == RequestPhase.RECURRENT:
            torch.cuda._sleep(10_000_000)
        return original(batch, prepared)

    monkeypatch.setattr(runner, "_execute", execute)
    engine.add_request("a", [2], params)
    actual = drain(engine)[-1]
    assert actual == expected
    assert maximum <= 2  # New event is appended after execution, total <= 3.
    assert not engine._pending_exit_signals


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True])
def test_abort_speculative_core_then_reuse_id(monkeypatch, multi_stream):
    engine = make_engine(device="cuda", multi_stream=multi_stream, static=True)
    params = SamplingParams(max_tokens=4, ignore_eos=True)
    engine.add_request("reuse", [2], params)
    engine.step()
    engine.step()
    ticket = engine._pending_coda[0]
    real_event = ticket.event
    ticket.event = HeldEvent()
    engine.step()
    engine.step()
    assert engine.last_schedule.phase == RequestPhase.RECURRENT
    output = engine.abort_request("reuse")
    assert output.finish_reason == "abort" and output.token_ids == []
    assert engine.cache_manager.num_used_blocks == 0
    ticket.event = real_event
    engine.add_request("reuse", [3], params)
    result = drain(engine)[-1]
    assert result.prompt_token_ids == [3]
    assert len(result.token_ids) == 4


@pytest.mark.gpu
def test_completed_core_refills_even_when_coda_cpu_delivery_is_pending(monkeypatch):
    engine = make_engine(device="cuda", multi_stream=True)
    engine.add_request("fast", [2], SamplingParams(max_tokens=3, exit_threshold=0, ignore_eos=True))
    engine.add_request("slow", [3], SamplingParams(max_tokens=3, exit_threshold=1, ignore_eos=True))
    original = engine.executor.submit
    held = HeldEvent()
    target = []

    def submit(batch):
        ticket = original(batch)
        if (
            batch.phase == RequestPhase.CODA
            and len(batch.requests) == 1
            and batch.requests[0].request.request_id == "fast"
        ):
            target.append((ticket, ticket.event))
            ticket.event = held
        return ticket

    monkeypatch.setattr(engine.executor, "submit", submit)
    for _ in range(30):
        engine.step()
        if target:
            break
    assert target
    engine.model_runner.synchronize()  # GPU is done; ONLY CPU delivery lags.
    engine.step()
    assert engine.last_schedule.phase == RequestPhase.PRELUDE
    assert not held.complete
    engine.step()
    assert engine.last_schedule.phase == RequestPhase.RECURRENT
    assert {i.request.request_id for i in engine.last_schedule.requests} == {"fast", "slow"}
    assert not held.complete
    target[0][0].event = target[0][1]
    monkeypatch.setattr(engine.executor, "submit", original)
    drain(engine)


@pytest.mark.gpu
def test_device_progress_does_not_wait_for_readback_stream():
    engine = make_engine(device="cuda", multi_stream=True)
    runner = engine.model_runner
    engine.add_request("a", [2], SamplingParams(max_tokens=3, ignore_eos=True, exit_threshold=1))
    try:
        engine.step()  # Prefill has no host result.
        with torch.cuda.stream(runner.readback_stream):
            torch.cuda._sleep(1_000_000_000)
        engine.step()  # Coda submits D2H behind the delay.
        ticket = engine._pending_coda[0]
        compute_done = runner.events["a"]
        assert compute_done is not ticket.event
        compute_done.synchronize()
        assert not ticket.ready()
        engine.step()  # Prelude consumes the device token, not the host result.
        assert engine.last_schedule.phase == RequestPhase.PRELUDE
        runner.events["a"].synchronize()
        assert not ticket.ready()
        assert not engine.scheduler.requests["a"].generated_token_ids
        assert drain(engine)[-1].exit_depths == [4, 4, 4]
    finally:
        engine.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_fixed_depth_async_does_not_read_back_unused_gates(monkeypatch, device):
    engine = make_engine(device=device, multi_stream=True)
    engine.add_request("a", [2], SamplingParams(max_tokens=3, ignore_eos=True, exit_threshold=1))
    submit = engine.executor.submit
    recurrent = []

    def record(batch):
        ticket = submit(batch)
        if batch.phase == RequestPhase.RECURRENT:
            recurrent.append(ticket)
            assert not ticket.has_result
            assert ticket.slot is None
        return ticket

    monkeypatch.setattr(engine.executor, "submit", record)
    try:
        assert drain(engine)[-1].exit_depths == [4, 4, 4]
        assert recurrent
    finally:
        engine.close()


@pytest.mark.gpu
def test_first_async_state_import_waits_for_prefill_producer(monkeypatch):
    if torch.version.hip:
        pytest.skip("CUDA sleep topology probe is NVIDIA-specific")
    sync = make_engine(device="cuda", asynchronous=False, seqs=1)
    engine = make_engine(device="cuda", multi_stream=True, seqs=1)
    params = SamplingParams(max_tokens=4, ignore_eos=True, exit_threshold=1)
    runner = engine.model_runner
    execute, prepare = runner._execute, runner.prepare_on_gpu
    imported = []

    def delayed_prefill(batch, prepared):
        if prepared.batch.operation == Operation.PRELUDE and prepared.batch.loop_count:
            # The following CODA imports this hidden on copy_stream. Waiting only
            # on the caller/default stream does not order it after this producer.
            torch.cuda._sleep(30_000_000)
        return execute(batch, prepared)

    def observe_import(prepared):
        if prepared.routing is not None and prepared.routing.imports:
            imported.append(True)
        return prepare(prepared)

    monkeypatch.setattr(runner, "_execute", delayed_prefill)
    monkeypatch.setattr(runner, "prepare_on_gpu", observe_import)
    try:
        sync.add_request("a", [2], params)
        expected = drain(sync)[-1]
        engine.add_request("a", [2], params)
        assert drain(engine)[-1] == expected
        assert imported
    finally:
        sync.close()
        engine.close()
