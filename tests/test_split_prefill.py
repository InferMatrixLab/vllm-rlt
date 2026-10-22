"""Split prefill: staged chunks and one boundary per recurrence depth.

The scheduler trace tests retain a small mirror of the depth-frontier invariant;
runtime tests exercise the engine transition and synchronous runner path.

Request-phase order is an engine-side obligation, not a scheduler guard: the scheduler
trusts that a request reaches `PREFILLING_RECURRENT` only after its chunk was
staged, so a stage enqueued out of order produces no usable scheduled request rather than a
dedicated error. `test_split_prefill_steps_one_depth_per_chunk_and_keeps_the_
prompt_ordered` pins that order from admission to the last depth.
"""

import pytest
import torch

from tests.helpers import make_prefill_engine, tiny_ouro_config, tiny_ouro_model
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.execution import Operation, WorkKey
from vllm_rlt.core.scheduling_policy import (
    NoRefillPolicy,
    RefillPolicy,
    SchedulingPolicy,
    SpeculativePolicy,
    SplitPrefillPolicy,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request, RequestStatus
from vllm_rlt.worker.executor import build_execution_batch

TOTAL_DEPTHS = 4


def drain_request(e):
    for _ in range(100):
        outputs = e.step()
        if outputs and outputs[-1].finished:
            return outputs[-1]
    pytest.fail("request did not finish within bounded scheduler steps")


def run_request(e, prompt, request_id="run"):
    e.add_request(
        request_id,
        prompt,
        SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True),
    )
    return drain_request(e)


def test_cpu_prefill_segments_gather_and_scatter_contiguous_spans():
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput

    e = make_prefill_engine(
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=4
        )
    )
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    e.add_request("b", [5, 6, 7, 8], SamplingParams(max_tokens=1))
    a, b = (e.scheduler.requests[request_id] for request_id in ("a", "b"))
    batch = SchedulerOutput(
        [
            ScheduledRequest(a, WorkKey.PROMPT_PRELUDE, 1, 2, chunk_start=0),
            ScheduledRequest(b, WorkKey.PROMPT_PRELUDE, 2, 1, chunk_start=1),
        ],
    )
    prepared = e.model_runner.prepare_on_gpu(
        e.model_runner.prepare_on_cpu(build_execution_batch(batch, e.model.config.total_ut_steps))
    )
    slots = e.model_runner.state_slots
    metadata = prepared.batch_inputs
    assert metadata.buffers.segments.cpu[:2].tolist() == [
        [slots["a"], 1, 2, 0, 1],
        [slots["b"], 1, 1, 2, 1],
    ]
    assert metadata.token_ids.tolist() == [2, 3, 7]
    assert metadata.buffers.segments.cpu.data_ptr() == metadata.buffers.segment_view.data_ptr()

    try:
        hidden = torch.arange(
            3 * e.model.config.hidden_size, dtype=e.model_runner.prefill_states.dtype
        ).view(3, e.model.config.hidden_size)
        e.model_runner._scatter_prefill(hidden, prepared)
        assert torch.equal(e.model_runner._gather_prefill(prepared), hidden)
    finally:
        e.close()


def test_recurrent_prepare_uses_absolute_token_start_for_kv_rows():
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput

    e = make_prefill_engine()
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    request = e.scheduler.requests["a"]
    assert e.cache_manager.allocate("a", 4)
    batch = SchedulerOutput(
        [
            ScheduledRequest(
                request,
                WorkKey.PROMPT_RECURRENT,
                token_start=2,
                token_count=1,
                depth=1,
                chunk_start=1,
            )
        ],
    )
    try:
        prepared = e.model_runner.prepare_on_gpu(
            e.model_runner.prepare_on_cpu(
                build_execution_batch(batch, e.model.config.total_ut_steps)
            )
        )
        assert prepared.kv.position_ids.tolist() == [2]
        assert prepared.kv.rows[0][2] == 2
        slot = e.model_runner.state_slots["a"]
        assert prepared.batch_inputs.buffers.segments.cpu[0].tolist() == [slot, 1, 1, 0, 1]
    finally:
        e.close()


def test_final_depth_prefill_scatter_retains_only_coda_hidden_per_item():
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput

    e = make_prefill_engine(
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
        )
    )
    e.add_request("mid", [1, 2], SamplingParams(max_tokens=1))
    e.add_request("final", [3, 4], SamplingParams(max_tokens=1))
    for request_id in ("mid", "final"):
        assert e.cache_manager.allocate(request_id, 2)
    mid, final = (e.scheduler.requests[rid] for rid in ("mid", "final"))
    batch = SchedulerOutput(
        [
            ScheduledRequest(mid, WorkKey.PROMPT_RECURRENT, 0, 2, depth=1, chunk_start=0),
            ScheduledRequest(
                final,
                WorkKey.PROMPT_RECURRENT,
                0,
                2,
                depth=3,
                chunk_start=0,
                completes_prompt=True,
            ),
        ],
    )
    runner = e.model_runner
    prepared = None
    try:
        prepared = runner.prepare_on_gpu(
            runner.prepare_on_cpu(build_execution_batch(batch, runner.model.config.total_ut_steps))
        )
        mid_slot, final_slot = runner.state_slots["mid"], runner.state_slots["final"]
        runner.prefill_states[mid_slot, :2].fill_(9)
        runner.prefill_states[final_slot, :2].fill_(7)
        hidden = torch.arange(
            4 * e.model.config.hidden_size, dtype=runner.prefill_states.dtype
        ).view(4, e.model.config.hidden_size)

        runner._scatter_prefill(hidden, prepared)
        runner._store_split_prefill_outputs(hidden, prepared)

        assert prepared.batch_inputs.buffers.segments.cpu[:2].tolist() == [
            [mid_slot, 0, 2, 0, 1],
            [final_slot, 0, 2, 2, 0],
        ]
        assert torch.equal(runner.prefill_states[mid_slot, :2], hidden[:2])
        assert torch.equal(runner.prefill_states[final_slot, :2], torch.full_like(hidden[2:], 7))
        assert torch.equal(final.hidden_state, hidden[-1])
        assert mid.hidden_state is None
    finally:
        e.close()


@pytest.mark.gpu
def test_cuda_prefill_segments_gather_and_scatter_variable_spans():
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(atomic_prefill=False),
        attention_backend="triton",
    )
    try:
        e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
        e.add_request("b", [5, 6, 7, 8], SamplingParams(max_tokens=1))
        a, b = (e.scheduler.requests[request_id] for request_id in ("a", "b"))
        batch = SchedulerOutput(
            [
                ScheduledRequest(a, WorkKey.PROMPT_PRELUDE, 1, 2, chunk_start=0),
                ScheduledRequest(b, WorkKey.PROMPT_PRELUDE, 2, 1, chunk_start=1),
            ],
        )
        prepared = e.model_runner.prepare_on_gpu(
            e.model_runner.prepare_on_cpu(
                build_execution_batch(batch, e.model.config.total_ut_steps)
            )
        )
        assert tuple(prepared.batch_inputs.buffers.segment_view.shape) == (2, 5)
        assert prepared.batch_inputs.buffers.segments.cpu.is_pinned()
        assert prepared.batch_inputs.buffers.token_input.is_pinned()
        assert prepared.batch_inputs.token_ids.tolist() == [2, 3, 7]
        hidden = torch.arange(
            3 * e.model.config.hidden_size, device="cuda", dtype=torch.bfloat16
        ).view(3, e.model.config.hidden_size)
        e.model_runner._scatter_prefill(hidden, prepared)
        torch.cuda.synchronize()
        assert torch.equal(e.model_runner._gather_prefill(prepared), hidden)
    finally:
        if "prepared" in locals():
            torch.cuda.synchronize()
        e.close()


@pytest.mark.gpu
def test_cuda_final_depth_prefill_scatter_retains_only_coda_hidden_per_item():
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(atomic_prefill=False),
        attention_backend="triton",
    )
    prepared = None
    try:
        e.add_request("mid", [1, 2], SamplingParams(max_tokens=1))
        e.add_request("final", [3, 4], SamplingParams(max_tokens=1))
        for request_id in ("mid", "final"):
            assert e.cache_manager.allocate(request_id, 2)
        mid, final = (e.scheduler.requests[rid] for rid in ("mid", "final"))
        batch = SchedulerOutput(
            [
                ScheduledRequest(mid, WorkKey.PROMPT_RECURRENT, 0, 2, depth=1, chunk_start=0),
                ScheduledRequest(
                    final,
                    WorkKey.PROMPT_RECURRENT,
                    0,
                    2,
                    depth=3,
                    chunk_start=0,
                    completes_prompt=True,
                ),
            ],
        )
        runner = e.model_runner
        prepared = runner.prepare_on_gpu(
            runner.prepare_on_cpu(build_execution_batch(batch, runner.model.config.total_ut_steps))
        )
        mid_slot, final_slot = runner.state_slots["mid"], runner.state_slots["final"]
        runner.prefill_states[mid_slot, :2].fill_(9)
        runner.prefill_states[final_slot, :2].fill_(7)
        hidden = torch.arange(
            4 * e.model.config.hidden_size, device="cuda", dtype=torch.bfloat16
        ).view(4, e.model.config.hidden_size)

        runner._scatter_prefill(hidden, prepared)
        runner._store_split_prefill_outputs(hidden, prepared)
        torch.cuda.synchronize()

        assert torch.equal(runner.prefill_states[mid_slot, :2], hidden[:2])
        assert torch.equal(runner.prefill_states[final_slot, :2], torch.full_like(hidden[2:], 7))
        assert torch.equal(final.hidden_state, hidden[-1])
        assert mid.hidden_state is None
    finally:
        if prepared is not None:
            torch.cuda.synchronize()
        e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("use_uva", [False, True])
def test_cuda_prefill_submit_keeps_host_and_device_buffers_until_event(use_uva):
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
    from vllm_rlt.worker.model_runner import BatchInputBuffers
    from vllm_rlt.worker.submission_queue import SubmissionQueue

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(atomic_prefill=False, use_uva=use_uva),
        attention_backend="triton",
    )

    class HeldCudaEvent:
        def __init__(self, event):
            self.event = event
            self.released = False

        def query(self):
            return self.released and self.event.query()

        def synchronize(self):
            self.event.synchronize()
            self.released = True

    try:
        e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
        e.add_request("b", [5, 6, 7, 8], SamplingParams(max_tokens=1))
        a, b = (e.scheduler.requests[rid] for rid in ("a", "b"))
        batches = [
            SchedulerOutput(
                [ScheduledRequest(request, WorkKey.PROMPT_PRELUDE, 0, 2, chunk_start=0)],
            )
            for request in (a, b)
        ]
        runner = e.model_runner
        assert runner.batch_input_buffers[0].segments.use_uva is use_uva
        runner.batch_input_buffers = [
            BatchInputBuffers(2, 4, runner.device, torch.bfloat16, use_uva=use_uva)
            for _ in range(2)
        ]

        runner.submissions = SubmissionQueue(max_pending=2)
        first_ticket = e.executor.submit(batches[0])
        first_buffers = runner.batch_input_buffers[0]
        first_segments = first_buffers.segments.cpu[:1].clone()
        first_tokens = torch.tensor(a.prompt_token_ids[:2])
        first_sources = first_buffers.token_transfers.descriptors.cpu[:1].clone()
        held = HeldCudaEvent(first_ticket.event)
        runner.submissions.events[0] = held

        second_ticket = e.executor.submit(batches[1])
        second_buffers = runner.batch_input_buffers[1]
        assert second_buffers.segments.cpu.data_ptr() != first_buffers.segments.cpu.data_ptr()
        assert second_buffers.token_input.data_ptr() != first_buffers.token_input.data_ptr()
        assert torch.equal(first_buffers.segments.cpu[:1], first_segments)
        assert torch.equal(first_buffers.token_transfers.descriptors.cpu[:1], first_sources)

        second_ticket.event.synchronize()
        # Capacity is full: the execution owner retires the first consumer before
        # prepare_on_cpu can overwrite slot 0. The input storage itself never waits.
        original_prepare = runner.prepare_on_cpu

        def checked_prepare(batch, **kwargs):
            assert held.released
            assert torch.equal(first_buffers.token_output[:2].cpu(), first_tokens)
            return original_prepare(batch, **kwargs)

        runner.prepare_on_cpu = checked_prepare
        third_ticket = e.executor.submit(batches[1])
        runner.prepare_on_cpu = original_prepare
        assert held.released
        assert runner.submissions.slot == 0
        third_ticket.event.synchronize()
        fourth_ticket = e.executor.submit(batches[0])
        fourth_ticket.event.synchronize()
    finally:
        torch.cuda.synchronize()
        e.close()


class HeldEvent:
    def __init__(self, before_sync=None):
        self.complete = False
        self.before_sync = before_sync

    def query(self):
        return self.complete

    def synchronize(self):
        if self.before_sync is not None:
            self.before_sync()
        self.complete = True


@pytest.mark.gpu
def test_submit_transfer_failure_releases_only_prepared_prefill_buffers():
    from dataclasses import replace

    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend="triton",
    )
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    assert e.cache_manager.allocate("a", 4)
    request = e.scheduler.requests["a"]
    batch = SchedulerOutput(
        [
            ScheduledRequest(
                request,
                WorkKey.PROMPT_RECURRENT,
                token_start=0,
                token_count=2,
                depth=0,
                chunk_start=0,
            )
        ],
    )
    runner = e.model_runner
    original_prepare = runner.prepare_on_cpu

    class FailingRouting:
        imports = ()

        def transfer(self, batch):
            raise RuntimeError("injected metadata transfer failure")

        def record_done(self):
            return None

    def prepare_with_failing_transfer(value, **kwargs):
        prepared = original_prepare(value, **kwargs)
        return replace(prepared, routing=FailingRouting())

    runner.prepare_on_cpu = prepare_with_failing_transfer
    try:
        with pytest.raises(RuntimeError, match="injected metadata transfer failure"):
            runner.submit(
                batch, batch=build_execution_batch(batch, runner.model.config.total_ut_steps)
            )
        assert not runner.submissions.events
        runner.prepare_on_cpu = original_prepare
        runner.prepare_on_gpu(
            runner.prepare_on_cpu(build_execution_batch(batch, runner.model.config.total_ut_steps))
        )
        torch.cuda.synchronize()
    finally:
        runner.prepare_on_cpu = original_prepare
        e.close()


def test_submit_does_not_retry_failed_routing_cleanup(monkeypatch):
    from contextlib import nullcontext
    from dataclasses import replace

    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput

    e = make_prefill_engine()
    e.add_request("a", [1, 2], SamplingParams(max_tokens=1))
    request = e.scheduler.requests["a"]
    batch = SchedulerOutput(
        [
            ScheduledRequest(
                request, WorkKey.PROMPT_PRELUDE, token_start=0, token_count=2, chunk_start=0
            )
        ],
    )
    runner = e.model_runner

    class FakeStream:
        def wait_event(self, event):
            return None

    class FailingRouting:
        imports = ()
        ready_event = object()

        def __init__(self):
            self.record_done_calls = 0

        def transfer(self, value):
            return value

        def record_done(self):
            self.record_done_calls += 1
            raise RuntimeError("injected routing cleanup failure")

    routing = FailingRouting()
    original_prepare = runner.prepare_on_cpu
    original_execute = runner._execute

    def fail_execute(batch, prepared):
        raise RuntimeError("injected execute failure")

    runner.core_stream = FakeStream()
    runner.prepare_on_cpu = lambda value, **kwargs: replace(
        original_prepare(value, **kwargs), routing=routing
    )
    runner._execute = fail_execute
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    try:
        with pytest.raises(RuntimeError, match="injected routing cleanup failure"):
            runner.submit(
                batch, batch=build_execution_batch(batch, runner.model.config.total_ut_steps)
            )
        assert routing.record_done_calls == 1
        assert not runner.submissions.events
    finally:
        runner.prepare_on_cpu = original_prepare
        runner._execute = original_execute
        runner.core_stream = None
        e.close()


@pytest.mark.gpu
def test_cuda_submit_readback_failure_releases_batch_input_buffers(monkeypatch):
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(atomic_prefill=False),
        attention_backend="triton",
    )
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    request = e.scheduler.requests["a"]
    batch = SchedulerOutput(
        [
            ScheduledRequest(
                request, WorkKey.PROMPT_PRELUDE, token_start=0, token_count=2, chunk_start=0
            )
        ],
    )
    runner = e.model_runner
    original_readback = runner._readback_slot
    original_execute = runner._execute

    def execute_with_result(batch, prepared):
        original_execute(batch, prepared)
        return torch.zeros(1, device=runner.device, dtype=torch.float32)

    monkeypatch.setattr(runner, "_execute", execute_with_result)

    def fail_readback(result):
        raise RuntimeError("injected readback failure")

    runner._readback_slot = fail_readback
    try:
        with pytest.raises(RuntimeError, match="injected readback failure"):
            runner.submit(
                batch, batch=build_execution_batch(batch, runner.model.config.total_ut_steps)
            )
        torch.cuda.synchronize()
        assert not runner.submissions.events
        runner._readback_slot = original_readback
        runner.submit(batch, batch=build_execution_batch(batch, runner.model.config.total_ut_steps))
    finally:
        runner._readback_slot = original_readback
        torch.cuda.synchronize()
        e.close()


def advance_chunk(request, scheduled_request):
    """Apply one submitted depth run to the request's logical frontiers.

    The run leaves `prefill_depth_ranges[depth]` and extends the right edge of
    the next frontier, or joins the finished prefix at the last depth.
    """
    depth = request.prefill_depth(scheduled_request.token_start)
    boundary = request.prefill_depth_ranges[depth]
    assert scheduled_request.token_start == boundary[0], "a depth run must start at its frontier"
    end = scheduled_request.token_start + scheduled_request.token_count
    request.prefill_depth_ranges[depth] = [end, boundary[1]]
    if depth + 1 < TOTAL_DEPTHS:
        deeper = request.prefill_depth_ranges[depth + 1]
        request.prefill_depth_ranges[depth + 1] = [deeper[0], end]
    else:
        request.num_prefilled_tokens = end
    return depth


def schedule_stage(scheduler, stage):
    batch = scheduler.take_works((stage,))
    assert batch is not None, f"no {stage.value} batch was selected"
    return batch


def test_split_prefill_steps_one_depth_per_chunk_and_keeps_the_prompt_ordered():
    e = make_prefill_engine()
    e.add_request("a", [1, 2, 3, 4, 5], SamplingParams(max_tokens=2, ignore_eos=True))
    request = e.scheduler.requests["a"]
    prompt = request.prompt_token_ids

    # Mirror the pending engine update: each completed stage enqueues the next one.
    def apply(batch):
        scheduled_request = batch.requests[0]
        if scheduled_request.work == WorkKey.PROMPT_PRELUDE:
            request.num_preluded_tokens = (
                scheduled_request.token_start + scheduled_request.token_count
            )
            e.scheduler.enqueue_work(request, WorkKey.PROMPT_RECURRENT)
            return (scheduled_request.token_start, scheduled_request.token_count, None)
        depth = advance_chunk(request, scheduled_request)
        if depth + 1 < TOTAL_DEPTHS:
            e.scheduler.enqueue_work(request, WorkKey.PROMPT_RECURRENT)
        elif request.num_prefilled_tokens < len(prompt):
            e.scheduler.enqueue_work(request, WorkKey.PROMPT_PRELUDE)
        return (scheduled_request.token_start, scheduled_request.token_count, depth)

    trace = []
    for _ in range(40):
        batch = e.scheduler.schedule()
        if batch is None:
            break
        assert len(batch.requests) == 1
        trace.append((batch.requests[0].work, *apply(batch)))
    assert e.scheduler.schedule() is None, "prefill must drain, not schedule forever"

    assert trace == [
        (WorkKey.PROMPT_PRELUDE, 0, 2, None),
        (WorkKey.PROMPT_RECURRENT, 0, 2, 0),
        (WorkKey.PROMPT_RECURRENT, 0, 2, 1),
        (WorkKey.PROMPT_RECURRENT, 0, 2, 2),
        (WorkKey.PROMPT_RECURRENT, 0, 2, 3),
        (WorkKey.PROMPT_PRELUDE, 2, 2, None),
        (WorkKey.PROMPT_RECURRENT, 2, 2, 0),
        (WorkKey.PROMPT_RECURRENT, 2, 2, 1),
        (WorkKey.PROMPT_RECURRENT, 2, 2, 2),
        (WorkKey.PROMPT_RECURRENT, 2, 2, 3),
        (WorkKey.PROMPT_PRELUDE, 4, 1, None),
        (WorkKey.PROMPT_RECURRENT, 4, 1, 0),
        (WorkKey.PROMPT_RECURRENT, 4, 1, 1),
        (WorkKey.PROMPT_RECURRENT, 4, 1, 2),
        (WorkKey.PROMPT_RECURRENT, 4, 1, 3),
    ]
    assert request.num_prefilled_tokens == 5
    assert all(left == right for left, right in request.prefill_depth_ranges)


def test_engine_split_prefill_transition_advances_the_same_frontier_contract():
    e = make_prefill_engine()
    e.add_request("a", [1, 2, 3, 4, 5], SamplingParams(max_tokens=2, ignore_eos=True))
    request = e.scheduler.requests["a"]
    trace = []

    for _ in range(40):
        batch = e.scheduler.schedule()
        if batch is None:
            break
        assert len(batch.requests) == 1
        scheduled_request = batch.requests[0]
        progress = e._apply_split_prefill_progress(
            request, scheduled_request.work, scheduled_request
        )
        trace.append(
            (
                scheduled_request.work,
                scheduled_request.token_start,
                scheduled_request.token_count,
                scheduled_request.depth,
            )
        )
        if progress.prompt_complete:
            break

    assert trace == [
        (WorkKey.PROMPT_PRELUDE, 0, 2, None),
        (WorkKey.PROMPT_RECURRENT, 0, 2, 0),
        (WorkKey.PROMPT_RECURRENT, 0, 2, 1),
        (WorkKey.PROMPT_RECURRENT, 0, 2, 2),
        (WorkKey.PROMPT_RECURRENT, 0, 2, 3),
        (WorkKey.PROMPT_PRELUDE, 2, 2, None),
        (WorkKey.PROMPT_RECURRENT, 2, 2, 0),
        (WorkKey.PROMPT_RECURRENT, 2, 2, 1),
        (WorkKey.PROMPT_RECURRENT, 2, 2, 2),
        (WorkKey.PROMPT_RECURRENT, 2, 2, 3),
        (WorkKey.PROMPT_PRELUDE, 4, 1, None),
        (WorkKey.PROMPT_RECURRENT, 4, 1, 0),
        (WorkKey.PROMPT_RECURRENT, 4, 1, 1),
        (WorkKey.PROMPT_RECURRENT, 4, 1, 2),
        (WorkKey.PROMPT_RECURRENT, 4, 1, 3),
    ]
    assert request.num_prefilled_tokens == len(request.prompt_token_ids)
    assert e.scheduler.pending_work(request) is WorkKey.DECODE_CODA


@pytest.mark.parametrize(
    ("chunk_size", "token_budget", "prompt"),
    [(1, 3, [1]), (2, 3, [1, 2, 3, 4, 5]), (3, 4, [5, 4, 3, 2, 1, 6, 7])],
)
def test_sync_split_prefill_matches_atomic_tokens_and_exit_depths(chunk_size, token_budget, prompt):
    scheduler = SchedulerConfig(
        max_num_seqs=2,
        max_num_batched_tokens=token_budget,
        prefill_chunk_size=chunk_size,
    )
    atomic = make_prefill_engine(atomic=True, scheduler_config=scheduler)
    split = make_prefill_engine(scheduler_config=scheduler)

    expected = run_request(atomic, prompt)
    actual = run_request(split, prompt)

    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert actual.exit_depths == [TOTAL_DEPTHS, 2, 2]
    assert split.scheduler.requests == {}
    assert split.model_runner.state_slots == {}
    assert split.cache_manager.num_used_blocks == 0


def test_sync_split_prefill_resumes_after_a_cached_prefix():
    prompt = [1, 2, 3, 4, 5]
    cache = CacheConfig(64, 2, enable_prefix_caching=True)
    scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=3, prefill_chunk_size=2)
    atomic = make_prefill_engine(atomic=True, cache_config=cache, scheduler_config=scheduler)
    split = make_prefill_engine(cache_config=cache, scheduler_config=scheduler)

    expected = run_request(atomic, prompt)
    first = run_request(split, prompt)
    assert first.token_ids == expected.token_ids
    assert len(split.cache_manager.lookup_prefix(prompt)) == 2

    split.add_request(
        "cached",
        prompt,
        SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True),
    )
    split.scheduler._admit()
    assert split.scheduler.requests["cached"].num_prefilled_tokens == 4
    actual = drain_request(split)

    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert split.model_runner.state_slots == {}


def test_sync_split_prefill_abort_releases_slot_before_request_id_reuse():
    split = make_prefill_engine()
    prompt = [1, 2, 3, 4]
    split.add_request("reused", prompt, SamplingParams(max_tokens=2, ignore_eos=True))

    split.step()  # Stage the first prompt chunk.
    split.step()  # Complete depth zero, leaving the chunk resident.
    request = split.scheduler.requests["reused"]
    assert request.num_prefilled_tokens == 0
    assert request.prefill_depth_ranges[0] == [2, 4]
    assert split.model_runner.state_slots == {"reused": 0}

    split.abort_request("reused")
    assert split.model_runner.state_slots == {}
    assert split.cache_manager.num_used_blocks == 0

    actual = run_request(split, [5, 4, 3], request_id="reused")
    expected = run_request(make_prefill_engine(atomic=True), [5, 4, 3])
    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths


def test_cpu_async_split_prefill_matches_sync_split():
    prompt = [1, 2, 3, 4, 5]
    trace = ExitConfig("trace", depths_by_request={"run": [TOTAL_DEPTHS, 2, 2]})
    sync = make_prefill_engine(exit_config=trace)
    asynchronous = make_prefill_engine(
        exit_config=trace,
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
    )

    expected = run_request(sync, prompt)
    actual = run_request(asynchronous, prompt)

    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert asynchronous.cache_manager.num_used_blocks == 0


def test_cpu_async_split_frontier_advances_once_per_submission():
    prompt = [1, 2, 3, 4, 5]
    trace = ExitConfig("trace", depths_by_request={"run": [TOTAL_DEPTHS, 2, 2]})
    e = make_prefill_engine(
        exit_config=trace,
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
    )
    e.add_request("run", prompt, SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True))
    request = e.scheduler.requests["run"]
    held = []
    submit = e.executor.submit

    def hold_split_ticket(batch):
        ticket = submit(batch)
        if batch.has_split_prefill_work:
            ticket.event = HeldEvent()
            held.append(ticket.event)
            for scheduled_request in batch.requests:
                e.model_runner.events[scheduled_request.request.request_id] = ticket.event
        return ticket

    e.executor.submit = hold_split_ticket
    for _ in range(32):
        e.step()
        if e.scheduler.pending_work(request) is WorkKey.DECODE_CODA:
            break

    assert e.scheduler.pending_work(request) is WorkKey.DECODE_CODA
    assert request.num_prefilled_tokens == len(prompt)
    assert all(left == right for left, right in request.prefill_depth_ranges)
    assert len(held) == 15  # Three chunks, each with prelude plus four depths.
    assert e.model_runner.split_prefill_submissions == len(held)
    assert all(not event.complete for event in held)
    for event in held:
        event.complete = True

    output = drain_request(e)
    assert output.exit_depths == [TOTAL_DEPTHS, 2, 2]


def test_split_submission_bound_counts_and_waits_for_oldest_event():
    e = make_prefill_engine(
        exit_config=ExitConfig("ouro_delayed"),
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
    )
    e.add_request("wait", [1, 2], SamplingParams(max_tokens=1, ignore_eos=True))
    e.scheduler._admit()
    batch = e.scheduler.schedule()
    runner = e.model_runner
    events = [HeldEvent() for _ in range(runner.submissions.max_pending)]
    runner.submissions.events.extend(events)
    original_prepare = runner.prepare_on_cpu

    def checked_prepare(value, **kwargs):
        assert events[0].complete, "must retire the old reader before CPU writes"
        return original_prepare(value, **kwargs)

    runner.prepare_on_cpu = checked_prepare
    runner.submit(batch, batch=build_execution_batch(batch, runner.model.config.total_ut_steps))
    e._update(batch, None)

    assert events[0].complete
    assert len(runner.submissions.events) == runner.submissions.max_pending - 1
    assert runner.split_prefill_submissions == 1
    assert runner.split_prefill_submission_waits == 1
    e.abort_request("wait")


def test_runner_slot_release_waits_for_last_request_event():
    e = make_prefill_engine()
    e.add_request("slot", [1, 2], SamplingParams(max_tokens=1, ignore_eos=True))
    e.scheduler._admit()
    request = e.scheduler.requests["slot"]
    runner = e.model_runner
    slot = runner._ensure_request_slot(request)

    def verify_slot_is_still_owned():
        assert runner.state_slots["slot"] == slot

    event = HeldEvent(before_sync=verify_slot_is_still_owned)
    runner.events["slot"] = event

    runner.release("slot")

    assert event.complete
    assert "slot" not in runner.state_slots
    assert slot in runner.free_state_slots
    e.abort_request("slot")


@pytest.mark.parametrize("completed_depths", range(TOTAL_DEPTHS))
def test_split_preemption_rewinds_and_recomputes_the_resident_chunk(completed_depths):
    prompt = [1, 2, 3, 4]
    params = SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True)
    scheduler = SchedulerConfig(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        prefill_chunk_size=2,
        enable_preemption=True,
    )
    expected = run_request(make_prefill_engine(scheduler_config=scheduler), prompt)
    e = make_prefill_engine(scheduler_config=scheduler)
    e.add_request("victim", prompt, params)
    request = e.scheduler.requests["victim"]

    for _ in range(completed_depths + 1):
        e.step()
    assert e.scheduler.pending_work(request) is WorkKey.PROMPT_RECURRENT

    requester = Request("pressure", [9], SamplingParams(max_tokens=1))
    assert e.scheduler.preempt(requester)
    assert e.preemption.snapshots["victim"]["work"] is WorkKey.PROMPT_PRELUDE
    assert request.num_preluded_tokens == request.num_prefilled_tokens
    assert request.prefill_chunk_start == request.num_prefilled_tokens
    assert all(left == request.num_prefilled_tokens for left, _ in request.prefill_depth_ranges)
    assert e.cache_manager.num_used_blocks == 0

    e.scheduler._admit()
    assert e.scheduler.pending_work(request) is WorkKey.PROMPT_PRELUDE
    actual = drain_request(e)
    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert e.preemption.resumptions == 1
    assert not e.preemption.snapshots
    assert e.cache_manager.num_used_blocks == 0


def test_priority_admission_preempts_split_prefill_and_matches_reference():
    prompt = [1, 2, 3, 4]
    params = SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True, priority=10)
    trace = ExitConfig("trace", depths_by_request={"reference": [4, 2, 2]})
    scheduler = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=4,
        prefill_chunk_size=2,
        policy="priority",
    )
    expected = run_request(
        make_prefill_engine(scheduler_config=scheduler, exit_config=trace), prompt, "reference"
    )

    e = make_prefill_engine(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
            policy="priority",
            enable_preemption=True,
        ),
        exit_config=ExitConfig("trace", depths_by_request={"low": [4, 2, 2], "high": [4]}),
    )
    e.add_request("low", prompt, params)
    e.step()
    e.step()
    assert e.scheduler.pending_work(e.scheduler.requests["low"]) is WorkKey.PROMPT_RECURRENT
    e.add_request(
        "high",
        [5, 6],
        SamplingParams(max_tokens=1, max_loops=2, ignore_eos=True, priority=-10),
    )

    finished = {}
    for _ in range(100):
        for output in e.step():
            if output.finished:
                finished[output.request_id] = output
        if len(finished) == 2:
            break
    else:
        pytest.fail("priority preemption did not finish both requests")

    assert e.preemption.preemptions == 1 == e.preemption.resumptions
    assert finished["low"].token_ids == expected.token_ids
    assert finished["low"].exit_depths == expected.exit_depths
    assert len(finished["high"].token_ids) == 1
    assert e.cache_manager.num_used_blocks == 0
    assert e.model_runner.state_slots == {}


@pytest.mark.gpu
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("failure_site", ["prepare_on_gpu", "_execute"])
def test_cuda_split_failure_drains_writes_before_request_id_reuse(
    monkeypatch, async_scheduling, failure_site
):
    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(atomic_prefill=False, async_scheduling=async_scheduling),
        exit_config=ExitConfig("trace", depths_by_request={"reused": [4]}),
        attention_backend="triton",
    )
    runner = e.model_runner
    original = getattr(runner, failure_site)
    completed = torch.cuda.Event()

    def fail(batch, *args, **kwargs):
        result = original(batch, *args, **kwargs)
        is_loop = (
            (batch.batch.operation == Operation.RECURRENT)
            if failure_site == "prepare_on_gpu"
            else (batch.works == frozenset((WorkKey.PROMPT_RECURRENT,)))
        )
        if is_loop:
            torch.cuda._sleep(10_000_000)
            completed.record(torch.cuda.current_stream(runner.device))
            raise RuntimeError("injected split failure")
        return result

    try:
        monkeypatch.setattr(runner, failure_site, fail)
        e.add_request("reused", [1, 2, 3], SamplingParams(max_tokens=1, ignore_eos=True))
        e.step()
        with pytest.raises(RuntimeError, match="injected split failure"):
            e.step()
        assert completed.query()
        assert e.cache_manager.num_used_blocks == 0
        assert runner.state_slots == {}

        monkeypatch.setattr(runner, failure_site, original)
        e.add_request("reused", [3, 2, 1], SamplingParams(max_tokens=1, ignore_eos=True))
        result = drain_request(e)
        assert len(result.token_ids) == 1
        assert result.exit_depths == [4]
    finally:
        e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("multi_stream", [False, True])
def test_cuda_split_prefill_matches_atomic(async_scheduling, multi_stream):
    prompt = [1, 2, 3, 4, 5]
    params = SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True)
    execution = dict(
        async_scheduling=async_scheduling,
        multi_stream=multi_stream,
        static_buffers=async_scheduling,
        pad_to_power_of_two=async_scheduling,
    )
    torch.manual_seed(321)
    atomic_model = OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16)
    atomic = LLMEngine(
        atomic_model,
        cache_config=CacheConfig(128, 2, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(atomic_prefill=True, **execution),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend="triton",
    )
    atomic_output = run_request(atomic, prompt, "first")
    atomic.add_request("second", prompt, params)
    atomic.step()
    atomic_cached_output = drain_request(atomic)
    atomic.close()
    del atomic_model
    torch.cuda.empty_cache()

    torch.manual_seed(321)
    split_model = OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16)
    split = LLMEngine(
        split_model,
        cache_config=CacheConfig(128, 2, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(atomic_prefill=False, **execution),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend="triton",
    )
    split_output = run_request(split, prompt, "first")
    split.add_request("second", prompt, params)
    split.step()
    assert split.cache_manager.prefix_hits > 0
    assert split.scheduler.requests["second"].num_prefilled_tokens > 0
    split_cached_output = drain_request(split)
    split.close()

    assert split_output.token_ids == atomic_output.token_ids
    assert split_output.exit_depths == atomic_output.exit_depths
    assert split_cached_output.token_ids == atomic_cached_output.token_ids
    assert split_cached_output.exit_depths == atomic_cached_output.exit_depths


@pytest.mark.gpu
def test_cuda_split_prefill_survives_decode_graph_capture_and_replay():
    from vllm_rlt.worker.buffers import execution_buffer_bytes

    graph_reserve = 32 * 1024**2

    def run(cuda_graphs):
        torch.manual_seed(765)
        model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
        cache = CacheConfig(
            block_size=16, gpu_memory_utilization=0.5, memory_reserve_bytes=32 * 1024**2
        )
        scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2)
        execution = ExecutionConfig(
            async_scheduling=True,
            cuda_graphs=cuda_graphs,
            cuda_graph_memory_reserve_bytes=graph_reserve,
            atomic_prefill=False,
        )
        e = LLMEngine(
            model,
            cache_config=cache,
            scheduler_config=scheduler,
            execution_config=execution,
            exit_config=ExitConfig("trace", depths_by_request={"split": [4], "decode": [4, 2, 2]}),
            attention_backend="triton",
        )
        try:
            e.add_request("split", [1, 2, 3, 4], SamplingParams(max_tokens=1, ignore_eos=True))
            e.step()
            e.step()
            split_request = e.scheduler.requests["split"]
            assert e.scheduler.pending_work(split_request) is WorkKey.PROMPT_RECURRENT
            assert split_request.prefill_depth_ranges[0] == [2, 4]
            # Keep the staged chunk resident while decode triggers graph capture.
            parked_work = e.scheduler.remove_pending_work(split_request)
            assert parked_work is WorkKey.PROMPT_RECURRENT

            e.add_request(
                "decode",
                [5, 6],
                SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True),
            )
            decode_recurrent_runs = 0
            for _ in range(100):
                e.step()
                batch = e.last_schedule
                if batch is not None and batch.works == frozenset((WorkKey.DECODE_RECURRENT,)):
                    decode_recurrent_runs += sum(
                        scheduled_request.request.request_id == "decode"
                        for scheduled_request in batch.requests
                    )
                if decode_recurrent_runs == 2:
                    break
            else:
                pytest.fail("decode request did not reach its second recurrent step")
            assert e.scheduler.pending_work(split_request) is None
            assert split_request.num_prefilled_tokens == 0
            assert split_request.num_preluded_tokens == 2

            graph_counts = None
            if e.model_runner.graphs is not None:
                graph_counts = (
                    e.model_runner.graphs.captures,
                    e.model_runner.graphs.replays,
                    e.model_runner.graphs.fallbacks,
                )
            e.scheduler.enqueue_work(split_request, WorkKey.PROMPT_RECURRENT)
            assert e.scheduler.pending_work(split_request) is WorkKey.PROMPT_RECURRENT
            finished = {}
            for _ in range(200):
                for output in e.step():
                    if output.finished:
                        finished[output.request_id] = output
                if len(finished) == 2:
                    break
            else:
                pytest.fail("split-prefill graph requests did not finish")
            if graph_counts is not None:
                graph_counts = (
                    graph_counts,
                    (
                        e.model_runner.graphs.captures,
                        e.model_runner.graphs.replays,
                        e.model_runner.graphs.fallbacks,
                    ),
                )

            expected_buffers = execution_buffer_bytes(
                model.config, scheduler, cache, execution, next(model.parameters()).element_size()
            )
            return (
                {
                    "decode": (finished["decode"].token_ids, finished["decode"].exit_depths),
                    "split": (finished["split"].token_ids, finished["split"].exit_depths),
                },
                e.memory_plan,
                expected_buffers,
                graph_counts,
            )
        finally:
            e.close()

    eager, graphed = run(False), run(True)
    assert eager[0] == graphed[0]
    assert eager[1]["cuda_graph_reserve_bytes"] == 0
    assert graphed[1]["cuda_graph_reserve_bytes"] == graph_reserve
    # Independent startup profiles can differ in allocator state and measured peak.
    # The reserve deduction is tested with fixed inputs in test_cdb_runtime.py.
    assert eager[1]["static_buffer_bytes"] == eager[2] > 0
    assert graphed[1]["static_buffer_bytes"] == graphed[2] > 0
    before_resume, after_resume = graphed[3]
    assert before_resume == (1, 2, 0)
    assert after_resume == (1, 4, 0)


@pytest.mark.gpu
def test_cuda_async_split_preemption_releases_and_reuses_slot():
    prompt = [1, 2, 3, 4]
    params = SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True, priority=10)
    scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2)
    execution = ExecutionConfig(atomic_prefill=False)
    torch.manual_seed(432)
    reference_model = OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16)
    reference = LLMEngine(
        reference_model,
        cache_config=CacheConfig(128, 2),
        scheduler_config=scheduler,
        execution_config=execution,
        exit_config=ExitConfig("trace", depths_by_request={"run": [4, 2, 2]}),
        attention_backend="triton",
    )
    expected = run_request(reference, prompt)
    reference.close()
    del reference_model
    torch.cuda.empty_cache()

    torch.manual_seed(432)
    model = OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16)
    e = LLMEngine(
        model,
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
            policy="priority",
            enable_preemption=True,
        ),
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
        exit_config=ExitConfig(
            "trace",
            depths_by_request={
                "victim": [4, 2, 2],
                "pressure-a": [4],
                "pressure-b": [4],
            },
        ),
        attention_backend="triton",
    )
    try:
        e.add_request("victim", prompt, params)
        e.step()
        e.step()
        assert e.scheduler.pending_work(e.scheduler.requests["victim"]) is WorkKey.PROMPT_RECURRENT
        state = e.model_runner.async_state
        slot = state.slots["victim"]
        old_event = e.model_runner.events["victim"]
        assert e.scheduler.preempt(
            Request("pressure", [9], SamplingParams(max_tokens=1, priority=-10))
        )
        assert "victim" not in state.slots
        assert slot in state.free
        assert "victim" not in e.model_runner.events

        pressure_params = SamplingParams(max_tokens=1, priority=-10)
        e.add_request("pressure-a", [5, 6], pressure_params)
        e.add_request("pressure-b", [7, 8], pressure_params)
        e.scheduler._admit()
        e.step()
        assert e.scheduler.requests["victim"].status is RequestStatus.WAITING
        assert slot in state.slots.values()
        assert len(state.slots) == 2
        occupant = next(rid for rid, assigned in state.slots.items() if assigned == slot)
        other = "pressure-b" if occupant == "pressure-a" else "pressure-a"
        e.abort_request(occupant)
        e.abort_request(other)

        e.scheduler._admit()
        e.step()
        assert state.slots["victim"] != slot
        assert e.model_runner.events["victim"] is not old_event
        actual = drain_request(e)

        assert actual.token_ids == expected.token_ids
        assert actual.exit_depths == expected.exit_depths
        assert e.preemption.preemptions == 1 == e.preemption.resumptions
    finally:
        e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("attention_backend", ["triton", "flash_attn_4"])
def test_cuda_split_prefill_batches_mixed_depths(attention_backend):
    torch.manual_seed(654)
    config = (
        tiny_ouro_config(hidden_size=256, head_dim=64)
        if attention_backend == "flash_attn_4"
        else tiny_ouro_config()
    )
    model = OuroForCausalLM(config).to("cuda", torch.bfloat16)
    try:
        e = LLMEngine(
            model,
            cache_config=CacheConfig(128, 16, enable_prefix_caching=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
            ),
            execution_config=ExecutionConfig(atomic_prefill=False),
            exit_config=ExitConfig("ouro_delayed"),
            attention_backend=attention_backend,
        )
    except (ImportError, OSError, ValueError):
        if attention_backend == "flash_attn_4":
            pytest.skip("FlashAttention-4 is unavailable")
        raise
    try:
        params = SamplingParams(max_tokens=1, ignore_eos=True)
        e.add_request("a", [1, 2], params)
        e.add_request("b", [3, 4], params)
        e.step()  # Prelude for both requests.

        recurrent = e.scheduler.work_queues[WorkKey.PROMPT_RECURRENT]
        recurrent.clear()
        recurrent.append("a")
        e.step()  # Advance only request a to depth one.
        recurrent.append("b")
        e.step()  # Request a is depth one; request b is depth zero.

        assert e.last_schedule.works == frozenset((WorkKey.PROMPT_RECURRENT,))
        assert {scheduled_request.depth for scheduled_request in e.last_schedule.requests} == {0, 1}
    finally:
        e.close()


def test_split_prefill_holds_the_next_prelude_until_the_chunk_is_read():
    """One staged chunk per request: its hidden rows are in use until the last depth."""
    e = make_prefill_engine()
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=2, ignore_eos=True))
    e.scheduler._admit()
    request = e.scheduler.requests["a"]

    prelude = schedule_stage(e.scheduler, WorkKey.PROMPT_PRELUDE)
    scheduled_request = prelude.requests[0]
    request.num_preluded_tokens = scheduled_request.token_start + scheduled_request.token_count
    e.scheduler.enqueue_work(request, WorkKey.PROMPT_RECURRENT)
    batch = schedule_stage(e.scheduler, WorkKey.PROMPT_RECURRENT)
    advance_chunk(request, batch.requests[0])

    # Depth 0 of the first chunk is done, but its hidden rows are still the input
    # of depth 1, so staging the next chunk must wait.
    e.scheduler.enqueue_work(request, WorkKey.PROMPT_PRELUDE)
    assert e.scheduler.take_works((WorkKey.PROMPT_PRELUDE,)) is None
    assert list(e.scheduler.work_queues[WorkKey.PROMPT_PRELUDE]) == ["a"]
    e.scheduler.work_queues[WorkKey.PROMPT_PRELUDE].clear()
    assert e.scheduler._prelude_window_open(request) is False


def test_split_prefill_reports_no_depth_work_before_the_prelude():
    """Out-of-order staging yields an empty run, not a wrong span.

    The protocol keeps this state unreachable; the assertion documents what the
    scheduler does if it is ever reached, so a broken transition shows up as a
    zero-length scheduled request instead of a silently wrong token range.
    """
    e = make_prefill_engine()
    e.add_request("a", [1, 2, 3], SamplingParams(max_tokens=2, ignore_eos=True))
    e.scheduler._admit()
    request = e.scheduler.requests["a"]

    scheduled_request = e.scheduler._make_scheduled_work(
        request, WorkKey.PROMPT_RECURRENT, 4
    )
    assert scheduled_request.token_count == 0
    assert request.num_prefilled_tokens == 0


def test_split_prefill_batches_stay_within_the_token_budget():
    e = make_prefill_engine(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=3, prefill_chunk_size=2
        )
    )
    for name in ("a", "b", "c"):
        e.add_request(name, [1, 2, 3, 4], SamplingParams(max_tokens=1, ignore_eos=True))
    e.scheduler._admit()
    batch = schedule_stage(e.scheduler, WorkKey.PROMPT_PRELUDE)
    assert sum(i.token_count for i in batch.requests) <= 3
    assert len(batch.requests) == 2


def test_split_policy_honors_decode_due_before_split_prefill():
    from collections import deque
    from types import SimpleNamespace

    e = make_prefill_engine(scheduler_config=SchedulerConfig(max_prefill_batches_before_decode=1))
    policy = e.scheduler.policy
    policy.prefill_batches_since_recurrent = 1
    scheduler = SimpleNamespace(
        work_queues={
            WorkKey.DECODE_RECURRENT: deque(["decode"]),
            WorkKey.DECODE_PRELUDE: deque(),
            WorkKey.DECODE_CODA: deque(),
            WorkKey.PROMPT_PRELUDE: deque(["prefill"]),
            WorkKey.PROMPT_ATOMIC: deque(),
            WorkKey.PROMPT_RECURRENT: deque(),
            WorkKey.SPECULATIVE: deque(),
        },
        take_works=lambda works: works[0],
    )
    scheduler.has_ready_work = lambda work: bool(scheduler.work_queues[work])
    scheduler.has_ready_works = lambda *works: any(scheduler.has_ready_work(work) for work in works)
    assert policy.schedule(scheduler) is WorkKey.DECODE_RECURRENT
    e.close()


def test_split_prefill_stages_count_toward_decode_fairness():
    e = make_prefill_engine(scheduler_config=SchedulerConfig(max_prefill_batches_before_decode=2))
    policy = e.scheduler.policy
    assert policy.PREFILL_WORKS == (
        WorkKey.PROMPT_PRELUDE,
        WorkKey.PROMPT_RECURRENT,
    )

    policy.record_batch((WorkKey.PROMPT_PRELUDE,))
    assert not policy.decode_due
    policy.record_batch((WorkKey.PROMPT_RECURRENT,))
    assert policy.decode_due
    policy.record_batch((WorkKey.DECODE_RECURRENT,))
    assert not policy.decode_due
    e.close()


def test_split_prefill_stages_after_a_reused_prompt_prefix():
    """A cached prefix is prefilled at every depth: the window opens past it."""
    e = make_prefill_engine()
    request = Request(
        "a",
        [1, 2, 3, 4, 5],
        SamplingParams(max_tokens=2, ignore_eos=True),
        prefill_depth_ranges=Request.prefill_boundaries(5, TOTAL_DEPTHS),
    )
    # Admission sets both frontiers to the cached token count.
    request.num_prefilled_tokens = 2
    request.num_preluded_tokens = 2
    assert e.scheduler._prelude_window_open(request)
    scheduled_request = e.scheduler._make_scheduled_work(
        request, WorkKey.PROMPT_PRELUDE, 4
    )
    assert (scheduled_request.token_start, scheduled_request.token_count) == (2, 2)
    assert request.prefill_depth(scheduled_request.token_start) == 0


def test_prefill_depth_ranges_track_logical_depths():
    request = Request(
        "a",
        [1, 2, 3],
        SamplingParams(),
        prefill_depth_ranges=Request.prefill_boundaries(3, TOTAL_DEPTHS),
    )
    assert request.prefill_depth_ranges == [[0, 3], [0, 0], [0, 0], [0, 0]]
    assert request.prefill_depth(0) == 0
    assert request.prefill_depth(2) == 0

    request.prefill_depth_ranges[0] = [2, 3]
    request.prefill_depth_ranges[1] = [0, 2]
    assert request.prefill_depth(1) == 1
    assert request.prefill_depth(2) == 0

    with pytest.raises(RuntimeError, match="no prefill depth"):
        Request(
            "b",
            [1],
            SamplingParams(),
            prefill_depth_ranges=Request.prefill_boundaries(1, 1),
        ).prefill_depth(5)


def test_split_prefill_policy_is_selected_only_by_the_atomic_prefill_flag():
    assert RefillPolicy is type(make_prefill_engine(atomic=True).scheduler.policy)
    assert SplitPrefillPolicy is type(make_prefill_engine().scheduler.policy)
    assert NoRefillPolicy is type(
        make_prefill_engine(
            atomic=True, scheduler_config=SchedulerConfig(mode="no_refill")
        ).scheduler.policy
    )


def test_scheduling_policies_declare_their_prompt_work():
    assert SchedulingPolicy.PREFILL_WORKS == (WorkKey.PROMPT_ATOMIC,)
    assert RefillPolicy.PREFILL_WORKS == (WorkKey.PROMPT_ATOMIC,)
    assert NoRefillPolicy.PREFILL_WORKS == (WorkKey.PROMPT_ATOMIC,)
    assert SpeculativePolicy.PREFILL_WORKS == (WorkKey.PROMPT_ATOMIC,)
    assert SplitPrefillPolicy.PREFILL_WORKS == (
        WorkKey.PROMPT_PRELUDE,
        WorkKey.PROMPT_RECURRENT,
    )


@pytest.mark.parametrize(
    ("label", "kwargs", "message"),
    [
        (
            "shared layout",
            {"cache_config": CacheConfig(64, 2, "shared")},
            "split prefill requires last_exited",
        ),
        (
            "no_refill mode",
            {"scheduler_config": SchedulerConfig(mode="no_refill")},
            "split prefill currently implements refill",
        ),
    ],
)
def test_split_prefill_rejects_unimplemented_combinations(label, kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_prefill_engine(**kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("multi_stream", [False, True])
def test_chunk_streams_wait_before_hidden_gather(monkeypatch, async_scheduling, multi_stream):
    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(
            atomic_prefill=False, async_scheduling=async_scheduling, multi_stream=multi_stream
        ),
        exit_config=ExitConfig("trace", depths_by_request={"r": [4]}),
        attention_backend="triton",
    )
    runner = e.model_runner
    assert (runner.core_stream == runner.boundary_stream) is (not multi_stream)
    assert (runner.core_stream == runner.copy_stream) is (not multi_stream)
    original_execute = runner._execute
    original_scatter = runner._scatter_prefill
    original_gather = runner._gather_prefill
    observations = {}

    def execute(batch, prepared):
        operation = prepared.batch.operation
        expected = runner.core_stream if prepared.batch.loop_count else runner.boundary_stream
        assert torch.cuda.current_stream() == expected
        observations[operation] = True
        if operation == Operation.PRELUDE:
            # Make an early gather on the other stream read stale storage.
            torch.cuda._sleep(50_000_000)
        return original_execute(batch, prepared)

    def scatter(hidden, prepared):
        if prepared.batch.operation == Operation.PRELUDE:
            observations["produced"] = hidden.clone()
        return original_scatter(hidden, prepared)

    def gather(prepared):
        hidden = original_gather(prepared)
        if "consumed" not in observations:
            observations["consumed"] = hidden.clone()
        return hidden

    monkeypatch.setattr(runner, "_execute", execute)
    monkeypatch.setattr(runner, "_scatter_prefill", scatter)
    monkeypatch.setattr(runner, "_gather_prefill", gather)
    try:
        e.add_request("r", [1, 2, 3, 4], SamplingParams(max_tokens=1, ignore_eos=True))
        result = drain_request(e)
        assert len(result.token_ids) == 1
        assert all(operation in observations for operation in Operation)
        torch.testing.assert_close(observations["consumed"], observations["produced"])
        assert e.cache_manager.num_used_blocks == 0
    finally:
        e.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("use_uva", [False, True])
def test_chunk_prompt_cache_reuse_override_and_request_id_reuse(device, use_uva):
    from dataclasses import replace

    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput

    e = LLMEngine(
        tiny_ouro_model().to(device),
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(atomic_prefill=False, use_uva=use_uva),
        attention_backend="triton" if device == "cuda" else "torch",
    )
    runner = e.model_runner

    def prepare(request, start, override=None):
        scheduled = SchedulerOutput(
            [ScheduledRequest(request, WorkKey.PROMPT_PRELUDE, start, 2, chunk_start=start)],
        )
        batch = build_execution_batch(scheduled, e.model.config.total_ut_steps)
        assert batch.spans[0].input_tokens is None
        if override is not None:
            batch = replace(batch, spans=(replace(batch.spans[0], input_tokens=override),))
        prepared = runner.prepare_on_gpu(runner.prepare_on_cpu(batch))
        result = prepared.batch_inputs.token_ids.tolist()
        if device == "cuda":
            torch.cuda.synchronize()
        return result

    try:
        e.add_request("r", [1, 2, 3, 4], SamplingParams(max_tokens=1))
        request = e.scheduler.requests["r"]
        assert prepare(request, 0) == [1, 2]
        source_ptr = runner.prompt_token_buffers.data_ptr()
        assert prepare(request, 2) == [3, 4]
        assert runner.prompt_token_buffers.data_ptr() == source_ptr
        assert runner.prompt_token_owners["r"] is request
        assert prepare(request, 0, (7, 8)) == [7, 8]
        assert prepare(request, 0) == [1, 2]
        e.abort_request("r")
        assert "r" not in runner.prompt_token_owners
        e.add_request("r", [9, 10, 11, 12], SamplingParams(max_tokens=1))
        replacement = e.scheduler.requests["r"]
        assert prepare(replacement, 0) == [9, 10]
        assert runner.prompt_token_owners["r"] is replacement
    finally:
        e.close()


@pytest.mark.parametrize("multi_stream", [False, True])
@pytest.mark.gpu
@pytest.mark.parametrize("use_uva", [False, True])
def test_mixed_hidden_buffers_wait_for_strided_decode_and_chunk_producers(multi_stream, use_uva):
    from vllm_rlt.core.execution import ExecutionBatch, StateSpan

    e = LLMEngine(
        tiny_ouro_model().to("cuda"),
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(
            mode="mixed", max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(
            atomic_prefill=False, multi_stream=multi_stream, use_uva=use_uva
        ),
        attention_backend="triton",
    )
    runner = e.model_runner
    try:
        e.add_request("chunk", [1, 2, 3, 4], SamplingParams(max_tokens=2))
        e.add_request("decode", [5, 6], SamplingParams(max_tokens=2))
        chunk, decode = (e.scheduler.requests[rid] for rid in ("chunk", "decode"))
        for request in (chunk, decode):
            e.cache_manager.allocate(request.request_id, 4)
        slot = runner._ensure_request_slot(chunk)
        runner._ensure_request_slot(decode)
        h = e.model.config.hidden_size
        storage = torch.zeros(h * 2, device="cuda")
        decode.hidden_state = storage[::2]
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            torch.cuda._sleep(10_000_000)
            runner.prefill_states[slot, :2].fill_(3)
            decode.hidden_state.fill_(7)
            done = torch.cuda.Event()
            done.record()
        runner.events["chunk"] = runner.events["decode"] = done
        batch = ExecutionBatch(
            Operation.RECURRENT,
            (
                StateSpan(chunk, 0, 2, 0, chunk_start=0, retain_chunk=True, retain_last=True),
                StateSpan(decode, 1, 1, 0, emit_gate=True),
            ),
        )
        captured = []
        original = runner._execute

        def consume(scheduled, prepared):
            hidden = prepared.batch_inputs.hidden
            captured.append(hidden.clone())
            runner._store_hidden(batch, hidden + 1, prepared)
            return None

        runner._execute = consume
        try:
            ticket = runner.submit(batch=batch)
            ticket.event.synchronize()
        finally:
            runner._execute = original
        torch.testing.assert_close(
            captured[0], torch.tensor([3.0, 3.0, 7.0], device="cuda")[:, None].expand(3, h)
        )
        assert torch.all(runner.prefill_states[slot, :2] == 4)
        assert torch.all(chunk.hidden_state == 4)
        assert torch.all(decode.hidden_state == 8)
    finally:
        torch.cuda.synchronize()
        e.close()
