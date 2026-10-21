"""PD ownership, paged transfer descriptors and real multi-process NIXL execution."""

import time
from dataclasses import replace

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.transport import kv_segments, partition_segments
from vllm_rlt.profiling import Profiler


def test_transfer_lease_defers_free_until_last_reader():
    cache = KVCacheManager(2, 1, 8, 16, 4, 4)
    assert cache.allocate("a", 8)
    cache.pin_transfer("a", "x")
    cache.pin_transfer("a", "y")
    cache.free("a")
    assert cache.num_used_blocks == 8
    with pytest.raises(RuntimeError, match="scheduled for release"):
        cache.pin_transfer("a", "z")
    cache.unpin_transfer("a", "x")
    assert cache.num_used_blocks == 8
    cache.unpin_transfer("a", "y")
    assert cache.num_used_blocks == 0
    cache.free("a")


def test_receive_publishes_all_depths_only_after_commit():
    c = KVCacheManager(2, 1, 8, 16, 4, 4)
    c.allocate("a", 8)
    with pytest.raises(RuntimeError, match="uninitialized"):
        c.read(0, "a", 0, 7)
    with pytest.raises(ValueError):
        c.mark_imported_prefix("a", 9)
    c.mark_imported_prefix("a", 7)
    for plane in c._get_allocation("a").written:
        assert all(w.prefix == 7 for w in plane)
    with pytest.raises(RuntimeError, match="already initialized"):
        c.mark_imported_prefix("a", 7)


def test_segment_ranges_cover_only_valid_tokens_and_all_layers_depths():
    c = KVCacheManager(3, 2, 8, 24, 4, 4)
    c.allocate("a", 9)
    k = c.key_cache
    info = dict(
        block_size=4,
        num_blocks=24,
        num_layers=3,
        device=0,
        block_bytes=k.stride(0) * 4,
        layer_bytes=k.stride(1) * 4,
        token_bytes=k.stride(2) * 4,
        key_ptr=0,
        value_ptr=10**9,
    )
    tables = c.plane_block_tables("a")
    expected = set()
    for table in tables:
        for token in range(2, 9):
            for layer in range(3):
                for base in (0, 10**9):
                    pos = (
                        base
                        + table[token // 4] * info["block_bytes"]
                        + layer * info["layer_bytes"]
                        + (token % 4) * info["token_bytes"]
                    )
                    expected.update(range(pos, pos + info["token_bytes"]))
    segments = list(kv_segments(info, tables, 2, 9))
    actual = set()
    for address, size, _ in segments:
        actual.update(range(address, address + size))
    assert actual == expected
    parts = list(partition_segments(segments, segments, 333, 3))
    assert sum(n for _, _, n in parts) == len(expected)
    assert all(n <= 333 and len(src) <= 3 and src == dst for src, dst, n in parts)


@pytest.mark.parametrize(
    "options",
    [
        dict(prefill_devices=(0,), decode_devices=(0,)),
        dict(prefill_devices=()),
        dict(max_inflight_bytes=0),
        dict(transfer_chunk_bytes=100, max_inflight_bytes=50),
    ],
)
def test_pd_configuration_validation(options):
    with pytest.raises(ValueError):
        PDConfig(**options)


def drain(engine):
    outputs = {}
    deadline = time.monotonic() + 90
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
        assert time.monotonic() < deadline
    return outputs


def test_pd_split_transfer_uses_chunk_origin_after_budget_contention(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm_rlt.engine.stage_config import StageConfig
    from vllm_rlt.pd.connector import PDStageConnector, RequestTransfer

    class ReadyEvent:
        def record(self, stream):
            pass

        def query(self):
            return True

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", ReadyEvent)
    config = tiny_ouro_config()
    engine = LLMEngine(
        OuroForCausalLM(config),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=5, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(atomic_prefill=False),
        stage_config=StageConfig(role="prefill"),
    )
    transport = SimpleNamespace(inflight_bytes=0, busy=lambda tid: False)
    connector = PDStageConnector(
        engine,
        torch.empty((2, config.hidden_size)),
        transport,
        PDConfig(),
        notify=lambda *a, **k: None,
    )
    engine.connector = connector
    transfers = []
    connector.queue_chunk = lambda w, start, end, event, final: transfers.append(
        (w.tid, start, end, final)
    )
    batches = []

    def step():
        engine.step()
        if engine.last_schedule is not None:
            batches.append(engine.last_schedule)
        return engine.last_schedule is not None

    params = SamplingParams(max_tokens=1)
    for request_id in ("b", "a"):
        engine.add_request(request_id, [1, 2, 3, 4], params)
        engine.scheduler._admit()
        request = engine.scheduler.requests[request_id]
        connector.transfers[request_id] = RequestTransfer(
            request_id, request, len(connector.transfers), "peer", outgoing=True
        )
        assert step() and connector.prefill_tokens == 0

    for _ in range(config.total_ut_steps):
        assert step()
    assert (engine.scheduler.requests["a"].num_prefilled_tokens, connector.prefill_tokens) == (1, 4)

    for _ in range(config.total_ut_steps):
        assert step()
    final_scheduled_request = next(
        scheduled_request
        for scheduled_request in batches[-1].requests
        if scheduled_request.request.request_id == "a"
    )
    assert (
        final_scheduled_request.token_start,
        final_scheduled_request.token_count,
        final_scheduled_request.chunk_start,
    ) == (1, 3, 0)
    assert (transfers, connector.prefill_tokens) == ([("b", 0, 4, True), ("a", 0, 4, True)], 8)


def create_pd(
    *,
    graph=False,
    layout="last_exited",
    multi=False,
    atomic_prefill=True,
    prefill_chunk_size=3,
    max_num_batched_tokens=5,
    speculative=False,
    asynchronous=True,
    prefix_caching=False,
):
    pytest.importorskip("nixl")
    from vllm_rlt.pd.engine import PDEngine

    if torch.cuda.device_count() < (4 if multi else 2):
        pytest.skip("requires 2/4 visible GPUs")
    config = tiny_ouro_config(head_dim=64)
    return PDEngine(
        config,
        pd_config=PDConfig(
            prefill_devices=(0, 1) if multi else (0,),
            decode_devices=(2, 3) if multi else (1,),
            transfer_chunk_bytes=4096,
            max_inflight_bytes=16384,
            request_timeout=90,
            startup_timeout=90,
        ),
        prefill_cache_config=CacheConfig(128, 4, layout, enable_prefix_caching=prefix_caching),
        decode_cache_config=CacheConfig(128, 4, layout, enable_prefix_caching=prefix_caching),
        prefill_scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=max_num_batched_tokens,
            prefill_chunk_size=prefill_chunk_size,
        ),
        decode_scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2),
        exit_config=ExitConfig("ouro")
        if speculative
        else ExitConfig("trace", depths_by_request={"t": [4, 2, 3, 1] * 16}),
        speculative_config=SpeculativeConfig(3) if speculative else None,
        execution_config=ExecutionConfig(
            async_scheduling=asynchronous,
            atomic_prefill=atomic_prefill,
            static_buffers=graph,
            pad_to_power_of_two=graph,
            cuda_graphs=graph,
        ),
        attention_backend="triton",
        seed=123,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "graph,layout", [(False, "last_exited"), (True, "last_exited"), (False, "shared")]
)
def test_pd_generation_refill_cancel_and_reuse_matches_local(graph, layout):
    params = SamplingParams(max_tokens=4, min_loops=1, ignore_eos=True)
    with create_pd(graph=graph, layout=layout) as e:
        torch.manual_seed(123)
        model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda:0", torch.bfloat16)
        reference = LLMEngine(
            model,
            cache_config=CacheConfig(128, 4, layout),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2, max_num_batched_tokens=5, prefill_chunk_size=3
            ),
            exit_config=ExitConfig("trace", depths_by_request={"t": [4, 2, 3, 1] * 16}),
            attention_backend="triton",
        )
        prompts = [[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3], [2] * 9]
        for i, prompt in enumerate(prompts):
            for target in (e, reference):
                target.add_request(str(i), prompt, params, trace_id="t")
        assert drain(e) == drain(reference)
        assert e.stats()["reserved_blocks"] == 0
        # Sampling belongs entirely to D; moving P must not consume its RNG.
        stochastic = replace(params, temperature=0.7, top_k=32, seed=947)
        for target in (e, reference):
            target.add_request("seeded", prompts[0], stochastic, trace_id="t")
        assert drain(e) == drain(reference)
        # Cancel immediately while a reservation may already be in flight, then reuse ID.
        e.add_request("reuse", prompts[0], params, trace_id="t")
        e.step()
        e.abort_request("reuse")
        e.add_request("reuse", prompts[1], params, trace_id="t")
        reference.add_request("reuse", prompts[1], params, trace_id="t")
        assert drain(e) == drain(reference)
        assert all(p.slots == 0 and p.reserved_blocks == 0 for p in e.peers.values())
    assert all(m["used_blocks"] == 0 for m in e.worker_metrics.values())
    assert sum(m["bytes_sent"] for m in e.worker_metrics.values()) > 0


@pytest.mark.gpu
def test_pd_split_prefill_matches_atomic_transfer_boundaries_and_overlaps():
    prompt = list(range(1, 33))
    params = SamplingParams(max_tokens=4, min_loops=1, ignore_eos=True)
    runs = {}
    for atomic_prefill in (True, False):
        with create_pd(
            atomic_prefill=atomic_prefill,
            prefill_chunk_size=16,
            max_num_batched_tokens=16,
        ) as e:
            e.add_request("pd", prompt, params, trace_id="t")
            output = drain(e)["pd"]
            timings = next(m["timings"] for m in e.metrics if m["request_id"] == "pd")
            runs[atomic_prefill] = (
                output.token_ids,
                output.exit_depths,
                timings["prefill_transfer_boundaries"],
                timings["prefill_transfer_overlaps"],
            )

    atomic, split = runs[True], runs[False]
    assert split[:2] == atomic[:2]
    assert atomic[2] == split[2] == ((0, 16), (16, 32), 2)
    assert split[3] > 0


@pytest.mark.gpu
def test_pd_multiple_workers_and_peer_failure():
    with create_pd(multi=True) as e:
        params = SamplingParams(max_tokens=4, min_loops=1, ignore_eos=True)
        for i in range(6):
            e.add_request(str(i), [1, 2, 3, 4, 5, 6], params, trace_id="t")
        assert len(drain(e)) == 6
        e.add_request("fail", [1] * 20, params, trace_id="t")
        e.step()
        peer = next(p for p in e.peers.values() if p.role == "prefill")
        peer.client.process.kill()
        peer.client.process.join()
        with pytest.raises(RuntimeError):
            e.step()
        assert e.closed and all(not p.client.process.is_alive() for p in e.peers.values())


@pytest.mark.gpu
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_pd_cancel_during_execution_and_close_with_live_request(phase):
    e = create_pd()
    try:
        params = SamplingParams(max_tokens=64, min_loops=1, ignore_eos=True)
        e.add_request("cancel", [1] * 50, params, trace_id="t")
        deadline = time.monotonic() + 30
        while True:
            e.step()
            w = next(iter(e.transfers.values()))
            if (phase == "prefill" and "prefill_started" in w.timings) or (
                phase == "decode" and w.phase == "decode"
            ):
                break
            assert time.monotonic() < deadline
        e.abort_request("cancel")
        assert drain(e) == {}
        assert all(p.slots == 0 and p.reserved_blocks == 0 for p in e.peers.values())
        e.add_request("live", [2] * 50, params, trace_id="t")
        e.step()
    finally:
        e.close()
    assert all(not p.client.process.is_alive() for p in e.peers.values())
    assert all(m["used_blocks"] == 0 for m in e.worker_metrics.values())


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_engine_yields_while_waiting_for_remote_kv(async_scheduling):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.engine.preemption import PreemptionManager
    from vllm_rlt.request import Request, RequestPhase

    engine = object.__new__(LLMEngine)
    engine.async_speculative = False
    engine.profiling = Profiler("cpu")
    engine.preemption = PreemptionManager(engine)
    engine.execution_config = ExecutionConfig(async_scheduling=async_scheduling)
    engine.scheduler = Scheduler(SchedulerConfig(), Mock())
    engine.cache_manager = engine.scheduler.cache_manager
    receiving = Request("remote", [1], SamplingParams(max_tokens=1), phase=RequestPhase.RECEIVING)
    engine.scheduler.requests[receiving.request_id] = receiving
    engine._inflight = []
    engine._pending_coda = []
    engine._overlap_boundary = False
    engine._pending_exit_signals = {}
    engine.model_runner = Mock()
    from vllm_rlt.engine.stage_config import StageConfig
    from vllm_rlt.worker.executor import Executor

    engine.stage_config = StageConfig()
    engine.connector = None
    engine.executor = Executor(engine.model_runner)
    assert engine.step() == []
    assert engine.has_unfinished_requests()
    engine.model_runner.synchronize.assert_not_called()
    if async_scheduling:
        # Reap the last local output while another request still awaits KV.
        active = Request("local", [1], SamplingParams(max_tokens=1), phase=RequestPhase.CODA)
        engine.scheduler.requests[active.request_id] = active
        active.num_output_placeholders = 1
        engine.model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))
        ticket = Mock()
        ticket.batch.requests = [SimpleNamespace(request=active)]
        ticket.ready.return_value = True
        ticket.collect.return_value = [3]
        ticket.output_indices = [0]
        ticket.depths = [4]
        engine._pending_coda = [ticket]
        outputs = engine.step()
        assert len(outputs) == 1 and outputs[0].finished
        assert outputs[0].token_ids == [3]
        assert list(engine.scheduler.requests) == ["remote"]
    # A lost runnable request must still trigger the original invariant.
    receiving.phase = RequestPhase.RECURRENT
    with pytest.raises(RuntimeError, match="scheduler made no progress"):
        engine.step()


@pytest.mark.gpu
@pytest.mark.parametrize("p_cache,d_cache", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("graph", [False, True])
def test_pd_prefix_reuse_reduces_transfers_and_preserves_outputs(p_cache, d_cache, graph):
    from vllm_rlt.pd.engine import PDEngine

    cfg = tiny_ouro_config(head_dim=64)
    params = SamplingParams(max_tokens=6, min_loops=1, ignore_eos=True)
    options = dict(
        pd_config=PDConfig(
            prefill_devices=(0,),
            decode_devices=(1,),
            max_receiving_requests=3,
            max_draining_requests=2,
            transfer_chunk_bytes=4096,
            max_inflight_bytes=16384,
        ),
        prefill_cache_config=CacheConfig(
            128, 4, enable_prefix_caching=p_cache, incremental_allocation=True
        ),
        decode_cache_config=CacheConfig(
            128, 4, enable_prefix_caching=d_cache, incremental_allocation=True
        ),
        prefill_scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=4, policy="priority"
        ),
        decode_scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=1, policy="priority", enable_preemption=True
        ),
        exit_config=ExitConfig("trace", depths_by_request={"t": [4, 2, 3, 1, 4, 2]}),
        execution_config=ExecutionConfig(async_scheduling=True, cuda_graphs=graph),
        attention_backend="triton",
        seed=123,
    )
    with PDEngine(cfg, **options) as e:
        results = []
        for round_index in range(2):
            for i in range(3):
                e.add_request(
                    f"{round_index}-{i}", [1, 2, 3, 4, 5, 6, 7, 8, 9], params, trace_id="t"
                )
            results.append(drain(e))
        for i in range(3):
            assert results[0][f"0-{i}"].token_ids == results[1][f"1-{i}"].token_ids
            assert results[0][f"0-{i}"].exit_depths == results[1][f"1-{i}"].exit_depths
        assert all(p.slots == 0 and p.reserved_blocks == 0 for p in e.peers.values())
    stats = list(e.worker_metrics.values())
    p = next(x for x in stats if x["prefill_tokens"])
    assert (p["prefill_tokens"] < 6 * 9) == p_cache
    full_bytes = (
        6
        * 9
        * cfg.num_hidden_layers
        * cfg.num_key_value_heads
        * cfg.head_dim
        * 2
        * 2
        * cfg.total_ut_steps
    )
    if d_cache:
        assert p["bytes_sent"] < full_bytes
    assert all(x["used_blocks"] == 0 for x in stats)


def test_cancelled_transfer_does_not_recycle_pending_hidden_copy():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_rlt.pd.connector import PDStageConnector, RequestTransfer
    from vllm_rlt.request import Request

    engine = SimpleNamespace(
        cache_manager=Mock(), scheduler=SimpleNamespace(config=SchedulerConfig())
    )
    transport = Mock()
    transport.poll.return_value = ([], [])
    transport.busy.return_value = False
    connector = PDStageConnector(engine, [None], transport, PDConfig(), notify=Mock())
    event = Mock()
    event.query.return_value = False
    request = Request("r", [1], SamplingParams())
    transfer = RequestTransfer(
        "r", request, 0, "d", outgoing=True, cancelled=True, last_event=event
    )
    connector.transfers = {"r": transfer}
    connector._remove = Mock()
    connector.progress()
    connector._remove.assert_not_called()
    event.synchronize.assert_not_called()
    event.query.return_value = True
    connector.progress()
    connector._remove.assert_called_once_with(transfer)


@pytest.mark.gpu
@pytest.mark.parametrize("asynchronous,graph", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("atomic_prefill", [False, True])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_pd_speculative_matches_local_and_retires_cancelled_rounds(
    asynchronous, graph, atomic_prefill, prefix_caching
):
    params = SamplingParams(max_tokens=9, ignore_eos=True)
    with create_pd(
        speculative=True,
        asynchronous=asynchronous,
        atomic_prefill=atomic_prefill,
        prefix_caching=prefix_caching,
        graph=graph,
    ) as engine:
        torch.manual_seed(123)
        reference = LLMEngine(
            OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda:0", torch.bfloat16),
            cache_config=CacheConfig(128, 4),
            attention_backend="triton",
        )
        try:
            prompts = [[1, 2, 3, 4, 5], [2, 3], [4], [5, 6, 7]]
            for index, prompt in enumerate(prompts):
                for target in (engine, reference):
                    target.add_request(str(index), prompt, params)
            assert drain(engine) == drain(reference)
            # Repeating a full-block prefix must exercise reuse on both stages,
            # with speculative verification still matching cold local decoding.
            for target in (engine, reference):
                target.add_request("warm", prompts[0], params)
            assert drain(engine) == drain(reference)
            timing = next(m["timings"] for m in engine.metrics if m["request_id"] == "warm")
            assert (timing["prefill_transfer_boundaries"][0][0] > 0) == prefix_caching
            engine.add_request("reuse", prompts[0], replace(params, max_tokens=60))
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                outputs = engine.step()
                if any(len(output.token_ids) > 1 for output in outputs):
                    break
            else:
                pytest.fail("speculative decode did not start")
            engine.abort_request("reuse")
            for target in (engine, reference):
                target.add_request("reuse", prompts[1], params)
            assert drain(engine) == drain(reference)
            assert all(
                peer.slots == 0 and peer.reserved_blocks == 0 for peer in engine.peers.values()
            )
            engine.add_request("close-live", prompts[0], params)
            engine.step()
        finally:
            reference.close()
    assert all(metrics["used_blocks"] == 0 for metrics in engine.worker_metrics.values())


def test_pd_reservations_are_reported_before_worker_acknowledgments():
    from types import SimpleNamespace

    from vllm_rlt.pd.engine import PDEngine, Peer, Pending
    from vllm_rlt.request import Request

    engine = PDEngine.__new__(PDEngine)
    engine.config = PDConfig()
    engine.scheduling_policy = "fcfs"
    request = Request("r", [1, 2, 3], SamplingParams(max_tokens=3))
    engine.requests = {"r": request}
    engine.transfers = {"t": Pending(request, "t", None)}
    info = dict(
        transfer_slots=4,
        max_active_requests=4,
        num_blocks=64,
        block_size=2,
        depths=4,
        watermark_blocks=0,
        info={"agent": "test"},
    )
    engine.peers = {
        role: Peer(role, role, SimpleNamespace(info=info)) for role in ("prefill", "decode")
    }
    messages = []
    engine._send = lambda *args, **kwargs: messages.append((args, kwargs))
    assert engine.stats() == dict(requests=1, reserved_blocks=0)
    engine._admit()
    # P reserves ceil(3/2)*4; D reserves ceil((3+3-1)/2)*4.
    assert engine.stats() == dict(requests=1, reserved_blocks=20)
    assert len(messages) == 1 and messages[0][0] == ("decode", "reserve")
    assert not hasattr(engine, "cache_manager") and not hasattr(engine, "scheduler")
