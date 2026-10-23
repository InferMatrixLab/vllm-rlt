"""Producer/consumer failure contracts on stubbed boundaries and real CUDA streams."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.helpers import (
    bootstrap_speculation,
    tiny_ouro_config,
)
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase
from vllm_rlt.serving.worker import EngineWorker
from vllm_rlt.worker.async_speculative import (
    AsyncSpeculativeRunner,
    SpeculativeRequestState,
    SpeculativeRoundTicket,
)


class Completion:
    def __init__(self):
        self.waits = 0

    def query(self):
        return True

    def synchronize(self):
        self.waits += 1


class InvalidHostResult:
    def __init__(self):
        self.reads = 0

    def __getitem__(self, key):
        return self

    def tolist(self):
        self.reads += 1
        raise ValueError("secondary host-result decoding failure")


class CompletionRunner:
    discard_buffers = AsyncSpeculativeRunner.discard_buffers

    def __init__(self, buffers, *, retirement_ready=True):
        self.buffers = buffers
        self.retirement_ready = retirement_ready
        self.synchronizations = 0
        self.readback_waits = 0
        self.discarded_rounds = 0
        self.stats = SimpleNamespace(committed_tokens=0, accepted_tokens=0)

    def synchronize(self):
        self.synchronizations += 1

    def request_ready(self, request):
        return self.retirement_ready

    def release(self, request):
        pass


def prepared_engine(*, bad_second=False, first_frontier=4, retirement_ready=True):
    torch.manual_seed(123)
    engine = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).eval(),
        cache_config=CacheConfig(num_blocks=128, block_size=4),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
        speculative_config=SpeculativeConfig(num_speculative_tokens=2, draft_policy="fixed"),
    )
    params = SamplingParams(max_tokens=6, min_loops=4, max_loops=4, ignore_eos=True)
    engine.add_request("r", [2], params)
    for _ in range(30):
        if engine.step():
            break
    else:
        pytest.fail("CPU bootstrap did not produce an output")
    request = engine.scheduler.requests["r"]
    state = SpeculativeRequestState(
        request,
        engine.cache_manager._get_allocation("r"),
        torch.tensor(0),
        torch.tensor(1),
        submitted=2,
    )
    tickets = []
    for round_id, (rows, frontier) in enumerate([(3, first_frontier), (2, 6)]):
        scheduled_request = ScheduledRequest(request, token_start=1 + round_id, token_count=rows)
        host = torch.tensor(
            list(range(10 + 3 * round_id, 10 + 3 * round_id + rows)) + [rows - 1, frontier]
        )
        if bad_second and round_id == 1:
            host = InvalidHostResult()
        buffer = SimpleNamespace(
            copy_done=Completion(),
            leased=True,
            keepalive=[torch.tensor([999])],
            host_result=host,
        )
        tickets.append(
            SpeculativeRoundTicket(
                SchedulerOutput(RequestPhase.SPECULATIVE, [scheduled_request]),
                buffer,
                (state,),
                (round_id,),
                rows,
            )
        )
    original_runner = engine.speculative_runner
    engine.speculative_runner = CompletionRunner(
        [ticket.buffer for ticket in tickets], retirement_ready=retirement_ready
    )
    engine.execution_config = replace(engine.execution_config, async_scheduling=True)
    engine.async_speculative = True
    request.num_output_placeholders = 5
    request.num_speculative_rounds = 2
    engine._pending_speculative = tickets.copy()
    return engine, tickets, original_runner


@torch.inference_mode()
def test_discard_waits_for_copy_without_interpreting_payload():
    engine, tickets, _ = prepared_engine(bad_second=True)
    ticket = tickets[1]
    with pytest.raises(RuntimeError, match="collect.*before retiring"):
        ticket.retire()
    assert ticket.buffer.copy_done.waits == 0 and ticket.buffer.leased
    ticket.retire(discard=True)
    assert ticket.buffer.copy_done.waits == 1 and ticket.buffer.host_result.reads == 0
    assert not ticket.buffer.leased and not ticket.buffer.keepalive
    with pytest.raises(RuntimeError, match="retired speculative ticket"):
        ticket.collect()

    # Repeated retirement must not release a subsequent user's lease.
    ticket.buffer.leased = True
    ticket.buffer.keepalive.append(torch.tensor([123]))
    ticket.retire(discard=True)
    assert ticket.buffer.leased and len(ticket.buffer.keepalive) == 1
    assert ticket.buffer.copy_done.waits == 1
    engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("decode_failure", [False, True])
@torch.inference_mode()
@pytest.mark.parametrize("graphs", [False, True])
def test_cuda_serving_two_completed_rounds_isolate_decode_failure(
    monkeypatch, decode_failure, graphs
):
    from tests.test_async_speculative import make_engine, tiny

    engine = make_engine(tiny("cuda"), k=2, graphs=graphs)
    try:
        bootstrap_speculation(engine, max_tokens=6)
        # Force full acceptance, so the two real queued rounds produce both
        # an intermediate snapshot and the terminal snapshot in the same tick.
        monkeypatch.setattr(
            engine.speculative_runner,
            "_coda",
            lambda h: torch.zeros(len(h), engine.model.config.vocab_size, device=h.device),
        )
        engine.step()
        engine.step()
        assert len(engine._pending_speculative) == 2
        engine.speculative_runner.synchronize()
        decoder = RecordingDecoder(fail=decode_failure)
        worker = EngineWorker.__new__(EngineWorker)
        worker.engine = engine
        worker.decoders = {"r": (decoder, 1)}
        failures, events, running = worker._tick([], set())
        assert not running and not worker.decoders
        assert "r" not in engine.scheduler.requests
        if decode_failure:
            assert [rid for rid, _ in failures] == ["r"] and not events
            assert decoder.calls == [(0, False)]
        else:
            assert not failures and len(events) == 2
            assert events[-1][1].finish_reason == "length"
            assert len(decoder.calls) == 5

        # Admit and finish a fresh request through the same actual CUDA engine
        # and serving adapter after the failed request has already retired.
        engine.add_request("healthy", [7, 8], SamplingParams(max_tokens=3, ignore_eos=True))
        healthy_decoder = RecordingDecoder()
        worker.decoders["healthy"] = (healthy_decoder, 0)
        healthy_events = []
        for _ in range(60):
            failures, events, running = worker._tick([], set())
            assert not failures
            healthy_events.extend(events)
            if not running:
                break
        else:
            pytest.fail("unrelated CUDA request did not finish")
        assert len(healthy_decoder.calls) == 3
        assert all(rid == "healthy" for rid, _ in healthy_events)
        assert healthy_events[-1][1].finish_reason == "length"
        assert not worker.decoders and engine.cache_manager.num_used_blocks == 0
        assert not engine.speculative_runner.states
        assert all(
            not buffer.leased and not buffer.keepalive
            for buffer in engine.speculative_runner.buffers
        )
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("bad_second", [False, True])
@torch.inference_mode()
@pytest.mark.parametrize("graphs", [False, True])
def test_cuda_failure_cleanup_preserves_error_and_releases_real_buffers(bad_second, graphs):
    from tests.test_async_speculative import make_engine, tiny

    engine = make_engine(tiny("cuda"), k=2, graphs=graphs)
    try:
        bootstrap_speculation(engine)
        engine.step()
        engine.step()
        tickets = engine._pending_speculative.copy()
        assert len(tickets) == 2
        engine.speculative_runner.synchronize()
        # Corrupt only the completed host payload; the real compute/copy
        # streams and pending buffer leases retain their production lifecycle.
        first = tickets[0]
        first.buffer.host_result[first.rows + len(first.batch.requests)] = 0
        if bad_second:
            tickets[1].buffer.host_result = InvalidHostResult()
        with pytest.raises(RuntimeError, match="frontier disagrees with committed prefix"):
            engine.step()
        assert not engine._pending_speculative and not engine._retiring_speculative
        assert not engine.scheduler.requests and engine.cache_manager.num_used_blocks == 0
        assert not engine.speculative_runner.states
        assert all(
            not buffer.leased and not buffer.keepalive
            for buffer in engine.speculative_runner.buffers
        )
        if bad_second:
            assert tickets[1].buffer.host_result.reads == 0
        engine.close()
    finally:
        engine.close()


@torch.inference_mode()
def test_collected_ticket_retirement_has_no_second_host_wait():
    engine, tickets, _ = prepared_engine()
    ticket = tickets[0]
    collected = ticket.collect()
    assert ticket.buffer.copy_done.waits == 1
    ticket.retire()
    assert ticket.collect() is collected
    ticket.buffer.leased = True
    ticket.buffer.keepalive.append(torch.tensor([123]))
    ticket.retire()
    assert ticket.buffer.leased and len(ticket.buffer.keepalive) == 1
    assert ticket.buffer.copy_done.waits == 1
    engine.close()


@pytest.mark.parametrize("bad_second", [False, True])
@torch.inference_mode()
def test_failure_cleanup_preserves_primary_error_and_retires_all_buffers(bad_second):
    engine, tickets, _ = prepared_engine(bad_second=bad_second, first_frontier=0)
    runner = engine.speculative_runner
    # A failed submission may have acquired storage without returning a ticket.
    orphan = SimpleNamespace(leased=True, keepalive=[torch.tensor([123])])
    runner.buffers.append(orphan)
    with pytest.raises(RuntimeError, match="frontier disagrees with committed prefix"):
        engine.step()
    assert runner.synchronizations == 1
    assert not orphan.leased and not orphan.keepalive
    assert not engine._pending_speculative and not engine._retiring_speculative
    assert not engine.scheduler.requests and engine.cache_manager.num_used_blocks == 0
    assert all(not t.buffer.leased and not t.buffer.keepalive for t in tickets)
    if bad_second:
        assert tickets[1].buffer.host_result.reads == 0
    engine.close()
    engine.close()


@torch.inference_mode()
def test_close_discards_uncollected_payloads_and_is_repeatable():
    engine, tickets, _ = prepared_engine(bad_second=True)
    engine.close()
    engine.close()
    assert not engine._pending_speculative and not engine._retiring_speculative
    assert not engine.scheduler.requests and engine.cache_manager.num_used_blocks == 0
    assert all(not t.buffer.leased and not t.buffer.keepalive for t in tickets)
    assert tickets[1].buffer.host_result.reads == 0


class RecordingDecoder:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def decode(self, token, finished):
        self.calls.append((token, finished))
        if self.fail:
            raise ValueError("injected tokenizer decode failure")
        return f"{token};"


@pytest.mark.parametrize("decode_failure", [False, True])
@pytest.mark.parametrize("retirement_ready", [False, True])
@torch.inference_mode()
def test_serving_delivers_two_snapshots_or_isolates_failure(
    monkeypatch, decode_failure, retirement_ready
):
    engine, tickets, original_runner = prepared_engine(retirement_ready=retirement_ready)
    snapshots = engine._step_async_speculative()
    assert [len(o.token_ids) for o in snapshots] == [4, 6]
    assert [o.finished for o in snapshots] == [False, True]
    assert all(not t.buffer.leased for t in tickets)
    if not retirement_ready:
        assert engine.scheduler.requests["r"].finish_reason == "length"

    engine.add_request(
        "healthy", [2], SamplingParams(max_tokens=3, min_loops=4, max_loops=4, ignore_eos=True)
    )
    worker = EngineWorker.__new__(EngineWorker)
    worker.engine = engine
    decoder = RecordingDecoder(fail=decode_failure)
    healthy_decoder = RecordingDecoder()
    worker.decoders = {"r": (decoder, 1), "healthy": (healthy_decoder, 0)}
    with monkeypatch.context() as patched:
        patched.setattr(engine, "step", lambda: snapshots)
        failures, events, running = worker._tick([], set())
    assert running and set(worker.decoders) == {"healthy"}
    if decode_failure:
        assert [rid for rid, _ in failures] == ["r"] and not events
        assert decoder.calls == [(10, False)]
    else:
        assert not failures
        assert [event.text for _, event in events] == ["10;11;12;", "13;14;"]
        assert decoder.calls == [(10, False), (11, False), (12, False), (13, False), (14, True)]
    if not retirement_ready:
        assert engine.scheduler.requests["r"].finish_reason == "length"

    # Restore ordinary CPU execution and let the unrelated request finish via
    # the actual serving adapter; only the GPU completion boundary was stubbed.
    engine._retire_speculative(wait=True)
    engine.async_speculative = False
    engine.execution_config = replace(engine.execution_config, async_scheduling=False)
    engine.speculative_runner = original_runner
    healthy_events = []
    for _ in range(60):
        failures, events, running = worker._tick([], set())
        assert not failures
        healthy_events.extend(events)
        if not running:
            break
    else:
        pytest.fail("unrelated CPU request did not finish")
    assert len(healthy_decoder.calls) == 3
    assert all(rid == "healthy" for rid, _ in healthy_events)
    assert healthy_events[-1][1].finish_reason == "length"
    assert not worker.decoders and engine.cache_manager.num_used_blocks == 0
    engine.close()
