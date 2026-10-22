"""Mixed recurrent execution must preserve hidden state, gate and depth-local KV."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import SamplingParams, SchedulerConfig
from vllm_rlt.core.execution import ExecutionBatch, Operation, StateSpan, WorkKey
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.scheduler import Scheduler, SchedulerOutput
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request, RequestStatus
from vllm_rlt.worker.executor import ForwardBatch, ModelRuntime, build_execution_batch


def make_span(name, start, count, depth):
    return StateSpan(Request(name, [1], SamplingParams()), start, count, depth)


def make_cache(config, layout):
    return KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=64,
        block_size=2,
        max_loops=config.total_ut_steps,
        layout=layout,
    )


@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("compute_gate", [False, True])
@torch.inference_mode()
def test_mixed_prompt_and_generation_rows_match_serial_execution(layout, compute_gate):
    torch.manual_seed(71)
    config = tiny_ouro_config()
    model = OuroForCausalLM(config).eval()
    mixed, serial = make_cache(config, layout), make_cache(config, layout)

    def seed(cache):
        for name in ("prompt", "generation"):
            assert cache.allocate(name, 3)
        prompt = model.prelude(torch.tensor([1, 2, 3]))
        prompt, _ = model.recurrent(prompt, ["prompt"] * 3, [0] * 3, [0, 1, 2], cache)
        generation = model.prelude(torch.tensor([4, 5, 6]))
        for depth in (0, 1):
            generation, _ = model.recurrent(
                generation, ["generation"] * 3, [depth] * 3, [0, 1, 2], cache
            )
        # Prepare causal history for the single generation row at depth two.
        model.recurrent(generation[:2], ["generation"] * 2, [2] * 2, [0, 1], cache)
        return torch.cat((prompt, generation[2:]))

    hidden = seed(mixed)
    reference = seed(serial)
    execution_batch = ExecutionBatch(
        Operation.RECURRENT,
        (
            make_span("prompt", 0, 3, 1),
            make_span("generation", 2, 1, 2),
        ),
    )
    executor = ModelRuntime(model, mixed)
    forward = ForwardBatch(
        Operation.RECURRENT,
        hidden=hidden,
        metadata=(mixed.prepare_recurrent(execution_batch, packed=True),),
        compute_gate=compute_gate,
    )
    actual = executor.forward(executor.prepare(forward))

    expected_hidden, expected_gate = [], []
    offset = 0
    for span in execution_batch.spans:
        end = offset + span.token_count
        h, gate = model.recurrent(
            reference[offset:end],
            [span.request.request_id] * span.token_count,
            [span.depth] * span.token_count,
            list(range(span.token_start, span.token_start + span.token_count)),
            serial,
            compute_gate=compute_gate,
        )
        expected_hidden.append(h)
        if gate is not None:
            expected_gate.append(gate)
        offset = end
    torch.testing.assert_close(actual.hidden, torch.cat(expected_hidden))
    if compute_gate:
        torch.testing.assert_close(actual.gate_logits, torch.cat(expected_gate))
    else:
        assert actual.gate_logits is None
    for span in execution_batch.spans:
        for layer in range(config.num_hidden_layers):
            actual_kv = mixed.read(layer, span.request.request_id, span.depth)
            expected_kv = serial.read(layer, span.request.request_id, span.depth)
            for actual_tensor, expected_tensor in zip(actual_kv, expected_kv):
                torch.testing.assert_close(actual_tensor, expected_tensor)


def test_scheduled_recurrence_does_not_follow_mutated_request_progress():
    request = Request("r", [1, 2, 3], SamplingParams())
    request.loops_done = 2
    scheduler = Scheduler(SchedulerConfig(), None)
    scheduled_request = scheduler._make_scheduled_work(request, WorkKey.DECODE_RECURRENT, 1)
    batch = SchedulerOutput([scheduled_request])
    request.loops_done = 3
    request.generated_token_ids.append(4)
    assert build_execution_batch(batch, 4).addresses() == (["r"], [2], [2])


@torch.inference_mode()
def test_prepared_batch_cannot_write_to_a_reused_request_id():
    config = tiny_ouro_config()
    cache = make_cache(config, "last_exited")
    model = OuroForCausalLM(config).eval()
    executor = ModelRuntime(model, cache)
    cache.allocate("r", 1)
    kv = cache.prepare_recurrent(ExecutionBatch(Operation.RECURRENT, (make_span("r", 0, 1, 0),)))
    cache.free("r")
    cache.allocate("r", 1)
    cache.key_cache.fill_(17)
    before = cache.key_cache.clone()
    with pytest.raises(RuntimeError, match="stale"):
        forward = ForwardBatch(
            Operation.RECURRENT, hidden=model.prelude(torch.tensor([1])), metadata=(kv,)
        )
        executor.forward(executor.prepare(forward))
    torch.testing.assert_close(cache.key_cache, before)


@pytest.mark.parametrize(
    "device,graphs",
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=pytest.mark.gpu),
        pytest.param("cuda", True, marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_mixed_policy_runs_real_combined_batches_with_matching_outputs(
    asynchronous, padded, device, graphs
):
    from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig
    from vllm_rlt.engine.llm_engine import LLMEngine

    def run(mode, atomic=False):
        torch.manual_seed(123)
        engine = LLMEngine(
            OuroForCausalLM(tiny_ouro_config()).to(
                device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
            ),
            attention_backend="triton" if device == "cuda" else "torch",
            cache_config=CacheConfig(128, 2),
            scheduler_config=SchedulerConfig(
                mode=mode,
                max_num_seqs=3,
                max_num_batched_tokens=5,
                prefill_chunk_size=4,
            ),
            execution_config=ExecutionConfig(
                atomic_prefill=atomic,
                cuda_graphs=graphs,
                async_scheduling=asynchronous,
                static_buffers=padded,
                pad_to_power_of_two=padded,
            ),
            exit_config=ExitConfig(mode="ouro_delayed" if asynchronous else "ouro"),
        )
        for name, prompt in (("short", [1]), ("long", [2, 3, 4, 5, 6, 7, 8, 9, 10])):
            engine.add_request(
                name,
                prompt,
                SamplingParams(
                    max_tokens=6,
                    max_loops=3,
                    ignore_eos=True,
                    exit_threshold=0.6,
                ),
            )
        outputs = {}
        mixed = []
        calls = []
        prelude = engine.model.prelude
        recurrent = engine.model.recurrent_prepared

        def record_prelude(token_ids):
            calls.append(len(token_ids))
            return prelude(token_ids)

        def record(hidden, kv, cache, **kwargs):
            calls.append(len(hidden))
            return recurrent(hidden, kv, cache, **kwargs)

        engine.model.prelude = record_prelude
        engine.model.recurrent_prepared = record
        for _ in range(300):
            calls.clear()
            for output in engine.step():
                if output.finished:
                    outputs[output.request_id] = (output.token_ids, output.exit_depths)
            batch = engine.last_schedule
            if batch is not None and batch.is_mixed:
                works = {item.work for item in batch.requests}
                if len(works) == 2:
                    # Both storage kinds pass through a single model invocation.
                    if not graphs:
                        assert calls == [batch.num_tokens]
                    assert batch.num_tokens <= 5
                    mixed.append(batch)
            if not engine.has_unfinished_requests():
                break
        else:
            pytest.fail("mixed scheduling stalled")
        assert engine.cache_manager.num_used_blocks == 0
        return outputs, mixed

    expected, _ = run("refill", atomic=True)
    separated, _ = run("refill")
    actual, mixed = run("mixed")
    assert mixed
    assert actual == separated == expected


def mixed_ready_engine():
    from tests.helpers import make_prefill_engine

    engine = make_prefill_engine(
        scheduler_config=SchedulerConfig(
            mode="mixed",
            max_num_seqs=3,
            max_num_batched_tokens=5,
            prefill_chunk_size=4,
        )
    )
    for name, prompt in (("short", [1]), ("long", list(range(2, 11)))):
        engine.add_request(name, prompt, SamplingParams(max_tokens=6, ignore_eos=True))
    for _ in range(100):
        batch = engine.scheduler.schedule()
        works = batch.works if batch is not None else frozenset()
        if {
            WorkKey.DECODE_RECURRENT,
            WorkKey.PROMPT_RECURRENT,
        } <= works:
            return engine, batch
        if batch is not None:
            engine._update(batch, engine.executor.submit(batch).collect())
    pytest.fail("no mixed batch became ready")


@torch.inference_mode()
def test_prepared_batch_survives_progress_changes_and_contains_no_policy_label():
    reference, reference_batch = mixed_ready_engine()
    expected = reference.model_runner.submit(
        batch=build_execution_batch(reference_batch, 4)
    ).collect()
    engine, batch = mixed_ready_engine()
    prepared = engine.model_runner.prepare_on_gpu(
        engine.model_runner.prepare_on_cpu(
            build_execution_batch(batch, engine.model.config.total_ut_steps)
        )
    )
    for scheduled_request in batch.requests:
        scheduled_request.request.loops_done += 10
        scheduled_request.request.generated_token_ids.append(17)
    # The prepared batch owns its addresses and routing; later request progress cannot
    # retarget the submitted operation, even for mixed multi-row state bindings.
    actual = engine.model_runner._execute(batch, prepared)
    torch.testing.assert_close(actual, torch.tensor(expected))
    assert not hasattr(prepared.batch, "stage")
    for left, right in zip(prepared.batch.spans, build_execution_batch(reference_batch, 4).spans):
        for layer in range(engine.model.config.num_hidden_layers):
            for actual_kv, expected_kv in zip(
                engine.cache_manager.read(layer, left.request.request_id, left.depth),
                reference.cache_manager.read(layer, right.request.request_id, right.depth),
            ):
                torch.testing.assert_close(actual_kv, expected_kv)


def test_late_mixed_result_cannot_advance_a_reused_request_id():
    engine, batch = mixed_ready_engine()
    result = engine.executor.submit(batch).collect()
    old = next(
        scheduled_request.request
        for scheduled_request in batch.requests
        if scheduled_request.work == WorkKey.DECODE_RECURRENT
    )
    engine.abort_request(old.request_id)
    engine.add_request(old.request_id, [8, 9], SamplingParams(max_tokens=2))
    replacement = engine.scheduler.requests[old.request_id]
    engine._update(batch, result)
    assert replacement is not old
    assert replacement.status == RequestStatus.WAITING
    assert replacement.loops_done == 0
    assert replacement.generated_token_ids == []
    other = next(
        scheduled_request
        for scheduled_request in batch.requests
        if scheduled_request.request is not old
    )
    assert other.request.prefill_depth(other.token_start) == other.depth + 1


def test_mixed_selection_protects_requests_selected_from_another_queue(monkeypatch):
    engine, batch = mixed_ready_engine()
    for scheduled_request in batch.requests:
        engine.scheduler.enqueue_work(scheduled_request.request, scheduled_request.work)
    selected = next(
        i.request.request_id for i in batch.requests if i.work == WorkKey.DECODE_RECURRENT
    )
    blocked = next(
        i.request.request_id for i in batch.requests if i.work != WorkKey.DECODE_RECURRENT
    )
    exclusions = []
    monkeypatch.setattr(
        engine.cache_manager, "ensure_capacity", lambda rid, frontier: rid != blocked
    )

    def preempt(request, *, excluded):
        exclusions.append(excluded)
        return False

    engine.scheduler.preempt = preempt
    actual = engine.scheduler.schedule()
    assert [i.request.request_id for i in actual.requests] == [selected]
    assert exclusions == [frozenset({selected})]
    assert list(engine.scheduler.work_queues[WorkKey.PROMPT_RECURRENT]) == [blocked]


def test_runtime_closes_metadata_iterator_when_loop_execution_fails():
    config = tiny_ouro_config()
    cache = make_cache(config, "last_exited")
    model = OuroForCausalLM(config)
    runtime = ModelRuntime(model, cache)
    released = []

    def metadata():
        try:
            yield None
        finally:
            released.append(True)

    def fail(*args, **kwargs):
        raise RuntimeError("injected loop failure")

    model.recurrent_prepared = fail
    with pytest.raises(RuntimeError, match="injected"):
        forward = ForwardBatch(
            Operation.RECURRENT, hidden=torch.zeros(1, config.hidden_size), metadata=metadata()
        )
        runtime.forward(runtime.prepare(forward))
    assert released == [True]


@torch.inference_mode()
def test_input_operation_mixes_host_tokens_and_device_sample_without_stage():
    from dataclasses import replace

    engine, scheduled_batch = mixed_ready_engine()
    spans = []
    tokens = []
    for span in build_execution_batch(scheduled_batch, 4).spans:
        if span.chunk_start is not None:
            values = tuple(
                span.request.prompt_token_ids[
                    span.token_start : span.token_start + span.token_count
                ]
            )
            spans.append(replace(span, input_tokens=values, retain_chunk=True))
            tokens.extend(values)
        else:
            span.request.input_token_tensor = torch.tensor(17)
            spans.append(replace(span, input_tokens=None))
            tokens.append(17)
    batch = ExecutionBatch(Operation.PRELUDE, tuple(spans))
    expected = engine.model.prelude(torch.tensor(tokens))
    assert engine.model_runner.submit(batch=batch).collect() is None
    rows = []
    for span in batch.spans:
        if span.chunk_start is None:
            rows.append(span.request.hidden_state.unsqueeze(0))
            assert span.request.input_token_tensor is None
        else:
            slot = engine.model_runner.state_slots[span.request.request_id]
            start = span.token_start - span.chunk_start
            rows.append(engine.model_runner.prefill_states[slot, start : start + span.token_count])
    torch.testing.assert_close(torch.cat(rows), expected)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@torch.inference_mode()
def test_continuous_loop_range_matches_yielding_each_loop(chunked, padded):
    from dataclasses import replace

    from vllm_rlt import CacheConfig, ExecutionConfig
    from vllm_rlt.engine.llm_engine import LLMEngine

    def run(continuous):
        torch.manual_seed(71)
        engine = LLMEngine(
            OuroForCausalLM(tiny_ouro_config()),
            cache_config=CacheConfig(64, 2),
            execution_config=ExecutionConfig(
                atomic_prefill=False,
                static_buffers=padded,
                pad_to_power_of_two=padded,
            ),
        )
        request = Request("r", [1, 2, 3] if chunked else [1], SamplingParams())
        cache = engine.cache_manager
        assert cache.allocate("r", len(request.prompt_token_ids))
        span = StateSpan(
            request,
            0,
            len(request.prompt_token_ids),
            0,
            chunk_start=0 if chunked else None,
            retain_chunk=chunked,
            input_tokens=tuple(request.prompt_token_ids),
        )
        runner = engine.model_runner
        runner.submit(batch=ExecutionBatch(Operation.PRELUDE, (span,))).collect()
        if continuous:
            runner.submit(
                batch=ExecutionBatch(Operation.RECURRENT, (span,), loop_count=4)
            ).collect()
        else:
            for depth in range(4):
                runner.submit(
                    batch=ExecutionBatch(Operation.RECURRENT, (replace(span, depth=depth),))
                ).collect()
        assert request.loops_done == 0  # Execution does not commit scheduler progress.
        tensors = [request.hidden_state.clone()]
        for depth in range(4):
            for layer in range(engine.model.config.num_hidden_layers):
                tensors.extend(t.clone() for t in cache.read(layer, "r", depth))
        return tensors

    for actual, expected in zip(run(True), run(False)):
        torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def test_invalid_loop_range_fails_before_writing_kv():
    from vllm_rlt.worker.model_runner import ModelRunner

    config = tiny_ouro_config()
    cache = make_cache(config, "last_exited")
    cache.allocate("r", 1)
    cache.key_cache.fill_(17)
    before = cache.key_cache.clone()
    runner = ModelRunner(OuroForCausalLM(config), cache)
    with pytest.raises(ValueError, match="exceeds model loop depth"):
        runner.submit(
            batch=ExecutionBatch(Operation.RECURRENT, (make_span("r", 0, 1, 3),), 2)
        ).collect()
    torch.testing.assert_close(cache.key_cache, before)


def test_mixed_attempt_with_one_selected_phase_is_not_a_mixed_batch():
    from dataclasses import replace

    engine, batch = mixed_ready_engine()
    for item in batch.requests:
        engine.scheduler.enqueue_work(item.request, item.work)
    compatible = {WorkKey.DECODE_RECURRENT, WorkKey.PROMPT_RECURRENT}
    for work, queue in engine.scheduler.work_queues.items():
        if work not in compatible:
            queue.clear()
    engine.scheduler.config = replace(engine.scheduler.config, max_num_batched_tokens=1)
    selected = engine.scheduler.schedule()
    assert selected is not None
    assert {item.work for item in selected.requests} == {WorkKey.DECODE_RECURRENT}
    assert selected.works == frozenset((WorkKey.DECODE_RECURRENT,))
    assert not selected.is_mixed
    engine.close()
