"""PD ownership, paged transfer descriptors and real multi-process NIXL execution."""

import json
import os
import time
from contextlib import closing
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, NanbeigeForCausalLM, OuroForCausalLM
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


def create_pd(*, graph=False, layout="last_exited", multi=False):
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
        prefill_cache_config=CacheConfig(128, 4, layout),
        decode_cache_config=CacheConfig(128, 4, layout),
        prefill_scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=5, prefill_chunk_size=3
        ),
        decode_scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2),
        exit_config=ExitConfig("trace", depths_by_request={"t": [4, 2, 3, 1] * 16}),
        execution_config=ExecutionConfig(
            async_scheduling=True,
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
        assert e.cache_manager.num_used_blocks == 0
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
        assert all(p.reserved_slots == 0 and p.reserved_blocks == 0 for p in e.peers.values())
    assert all(m["used_blocks"] == 0 for m in e.worker_metrics.values())
    assert sum(m["bytes_sent"] for m in e.worker_metrics.values()) > 0


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
        peer.process.kill()
        peer.process.join()
        with pytest.raises(RuntimeError):
            e.step()
        assert e.closed and all(not p.process.is_alive() for p in e.peers.values())


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
        assert all(p.reserved_slots == 0 and p.reserved_blocks == 0 for p in e.peers.values())
        e.add_request("live", [2] * 50, params, trace_id="t")
        e.step()
    finally:
        e.close()
    assert all(not p.process.is_alive() for p in e.peers.values())
    assert all(m["used_blocks"] == 0 for m in e.worker_metrics.values())


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_engine_yields_while_waiting_for_remote_kv(async_scheduling):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.engine.preemption import PreemptionManager
    from vllm_rlt.request import Request, Stage

    engine = object.__new__(LLMEngine)
    engine.async_speculative = False
    engine.profiling = Profiler("cpu")
    engine.preemption = PreemptionManager(engine)
    engine.execution_config = ExecutionConfig(async_scheduling=async_scheduling)
    engine.scheduler = Scheduler(SchedulerConfig(), Mock())
    engine.cache_manager = engine.scheduler.cache_manager
    receiving = Request("remote", [1], SamplingParams(max_tokens=1), stage=Stage.RECEIVING)
    engine.scheduler.requests[receiving.request_id] = receiving
    engine._inflight = []
    engine._pending_coda = []
    engine._overlap_boundary = False
    engine._pending_exit_signals = {}
    engine.model_runner = Mock()
    assert engine.step() == []
    assert engine.has_unfinished_requests()
    engine.model_runner.synchronize.assert_not_called()
    if async_scheduling:
        # Reap the last local output while another request still awaits KV.
        active = Request("local", [1], SamplingParams(max_tokens=1), stage=Stage.CODA)
        engine.scheduler.requests[active.request_id] = active
        active.num_output_placeholders = 1
        engine.model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))
        ticket = Mock()
        ticket.batch.items = [SimpleNamespace(request=active)]
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
    receiving.stage = Stage.RECURRENT
    with pytest.raises(RuntimeError, match="scheduler made no progress"):
        engine.step()


@pytest.mark.gpu
@pytest.mark.parametrize("p_cache,d_cache", [(True, True), (True, False), (False, True)])
def test_pd_prefix_reuse_reduces_transfers_and_preserves_outputs(p_cache, d_cache):
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
        execution_config=ExecutionConfig(async_scheduling=True),
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
        assert all(
            p.reserved_slots == 0 and p.reserved_compute_slots == 0 and p.reserved_blocks == 0
            for p in e.peers.values()
        )
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


@pytest.fixture
def nanbeige_checkpoint(tmp_path):
    # An explicit local checkpoint can qualify the same tests on official weights.
    checkpoint = os.environ.get("VLLM_RLT_NANBEIGE_CHECKPOINT")
    if checkpoint:
        from pathlib import Path

        assert Path(checkpoint).is_dir(), "provide a complete local checkpoint directory"
        return checkpoint
    torch.manual_seed(42)
    model = NanbeigeForCausalLM(tiny_nanbeige_config(head_dim=64))
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def test_pd_nanbeige_checkpoint_loading(nanbeige_checkpoint):
    model = AutoModelForCausalLM.from_pretrained(nanbeige_checkpoint, dtype=torch.float32)
    assert isinstance(model, NanbeigeForCausalLM)
    assert model.config.total_ut_steps == 2


def test_pd_coordinator_stores_typed_metadata(monkeypatch):
    from unittest.mock import Mock

    from vllm_rlt.pd.engine import PDEngine, PDModelInfo

    context = Mock()
    context.Pipe.side_effect = lambda: (Mock(), Mock())
    monkeypatch.setattr("vllm_rlt.pd.engine.mp.get_context", lambda _: context)
    config = tiny_nanbeige_config()

    def ready(engine, condition, deadline):
        for peer in engine.peers.values():
            peer.info = dict(
                fingerprint="same-weights",
                model=config.to_dict(),
                info=dict(agent=peer.name),
                block_size=4,
                depths=2,
                num_blocks=128,
            )
            peer.ready = peer.connected = True
        assert condition()

    monkeypatch.setattr(PDEngine, "_wait", ready)
    # The coordinator consumes worker metadata, not Nanbeige's full config class.
    engine = PDEngine(tiny_ouro_config(), attention_backend="triton")
    try:
        assert engine.model_info == PDModelInfo(64, 2, 128)
        assert not hasattr(engine, "model")
        with pytest.raises(ValueError, match="loop bounds"):
            engine.add_request("invalid", [2], SamplingParams(max_loops=3))
        engine.add_request("valid", [2, 3], SamplingParams(max_tokens=4, max_loops=2))
        assert engine.requests["valid"].prompt_token_ids == [2, 3]
        engine.abort_request("valid")
        assert not engine.requests and not engine.transfers
        # A late control completion for generation A must not mutate generation B.
        engine.add_request("reuse", [2], SamplingParams(max_tokens=1))
        old = next(iter(engine.transfers.values()))
        engine.abort_request("reuse")
        engine.add_request("reuse", [3], SamplingParams(max_tokens=1))
        replacement = next(iter(engine.transfers.values()))
        engine._message(next(iter(engine.peers.values())), dict(kind="activated", tid=old.tid))
        assert replacement.phase == "waiting"
        assert engine.requests["reuse"] is replacement.request
        assert replacement.request.generated_token_ids == []
        engine.abort_request("reuse")
    finally:
        engine._terminate()


def fake_pd_engine(monkeypatch):
    """Coordinator with mocked processes/pipes; workers report ready immediately."""
    from unittest.mock import Mock

    from vllm_rlt.pd.engine import PDEngine

    context = Mock()
    context.Pipe.side_effect = lambda: (Mock(), Mock())
    monkeypatch.setattr("vllm_rlt.pd.engine.mp.get_context", lambda _: context)
    config = tiny_ouro_config()

    def ready(engine, condition, deadline):
        for peer in engine.peers.values():
            peer.info = dict(
                fingerprint="same-weights",
                model=config.to_dict(),
                info=dict(agent=peer.name),
                block_size=4,
                depths=config.total_ut_steps,
                num_blocks=128,
                max_seqs=8,
                active_limit=4,
                watermark_blocks=0,
            )
            peer.ready = peer.connected = True
            peer.channel.poll.return_value = False
        assert condition()

    monkeypatch.setattr(PDEngine, "_wait", ready)
    return PDEngine(config, attention_backend="triton")


def pd_peer(engine, role):
    return next(p for p in engine.peers.values() if p.role == role)


def sent_commands(peer, kind=None):
    messages = [call.args[0] for call in peer.channel.send.call_args_list]
    return [m for m in messages if kind is None or m["kind"] == kind]


def admit_one(engine, request_id="r", tokens=(2, 3, 4, 5)):
    engine.add_request(request_id, list(tokens), SamplingParams(max_tokens=4, max_loops=4))
    engine._admit()
    return next(iter(engine.transfers.values()))


@pytest.mark.parametrize("depths", [1, 4])
@pytest.mark.parametrize(
    "prompt_tokens,max_tokens,p_blocks,d_blocks",
    [(4, 1, 1, 1), (5, 4, 2, 2), (4, 6, 1, 3)],
)
def test_pd_coordinator_reserves_full_request_budget(
    monkeypatch, depths, prompt_tokens, max_tokens, p_blocks, d_blocks
):
    engine = fake_pd_engine(monkeypatch)
    try:
        p, d = pd_peer(engine, "prefill"), pd_peer(engine, "decode")
        for peer in (p, d):
            peer.info["depths"] = depths
        assert engine._reserved_blocks_total() == 0
        engine.add_request("r", [2] * prompt_tokens, SamplingParams(max_tokens=max_tokens))
        engine._admit()
        w = next(iter(engine.transfers.values()))
        assert w.phase == "reserving"
        assert p.reserved_blocks == w.reserved_p_blocks == p_blocks * depths
        assert d.reserved_blocks == w.reserved_d_blocks == d_blocks * depths
        assert (p.reserved_slots, p.reserved_compute_slots, d.reserved_slots) == (1, 1, 1)
        assert d.reserved_compute_slots == 0
        total = (p_blocks + d_blocks) * depths
        assert engine._reserved_blocks_total() == total
        # Preserve the shim's existing refresh point: transfer feedback, not admission.
        assert engine.cache_manager.num_used_blocks == 0
        engine._message(d, dict(kind="reserved", tid=w.tid, tables=((1,),), slot=0))
        assert engine.cache_manager.num_used_blocks == total
    finally:
        engine._terminate()


@pytest.mark.parametrize(
    "role,limits",
    [
        ("prefill", {"max_seqs": 1}),
        ("decode", {"max_seqs": 1}),
        ("prefill", {"active_limit": 1}),
        ("prefill", {"num_blocks": 4}),
        ("decode", {"num_blocks": 8}),
        ("prefill", {"num_blocks": 8, "watermark_blocks": 1}),
        ("decode", {"num_blocks": 16, "watermark_blocks": 1}),
    ],
)
def test_pd_coordinator_reservations_gate_admission(monkeypatch, role, limits):
    engine = fake_pd_engine(monkeypatch)
    try:
        p, d = pd_peer(engine, "prefill"), pd_peer(engine, "decode")
        pd_peer(engine, role).info.update(limits)
        first = admit_one(engine)
        assert first.phase == "reserving"
        engine.add_request("next", [2, 3, 4, 5], SamplingParams(max_tokens=4))
        waiting = next(w for w in engine.transfers.values() if w.request.request_id == "next")
        engine._admit()
        assert waiting.phase == "waiting"
        assert len(sent_commands(d, "reserve")) == 1
        assert (p.reserved_blocks, d.reserved_blocks) == (4, 8)
        assert (p.reserved_slots, d.reserved_slots) == (1, 1)

        engine._message(d, dict(kind="reserved", tid=first.tid, tables=((1,),), slot=0))
        engine._message(p, dict(kind="prefill_complete", tid=first.tid))
        assert p.reserved_compute_slots == 0
        engine._admit()
        # Computation finishing returns only compute capacity, not blocks/request slots.
        if "active_limit" in limits:
            assert waiting.phase == "reserving"
            assert (p.reserved_blocks, d.reserved_blocks) == (8, 16)
            assert (p.reserved_slots, d.reserved_slots) == (2, 2)
        else:
            assert waiting.phase == "waiting"
            assert (p.reserved_blocks, d.reserved_blocks) == (4, 8)
            assert (p.reserved_slots, d.reserved_slots) == (1, 1)

        engine.abort_request("r")
        engine._message(p, dict(kind="released", tid=first.tid, role="prefill"))
        engine._message(d, dict(kind="released", tid=first.tid, role="decode"))
        engine._admit()
        assert waiting.phase == "reserving"
        assert len(sent_commands(d, "reserve")) == 2
        assert (p.reserved_blocks, d.reserved_blocks) == (4, 8)
        assert (p.reserved_slots, p.reserved_compute_slots, d.reserved_slots) == (1, 1, 1)
    finally:
        engine._terminate()


def test_pd_coordinator_handoff_state_machine(monkeypatch):
    """Pin the happy-path protocol: phases, forwarded commands, and credit accounting."""
    engine = fake_pd_engine(monkeypatch)
    try:
        p, d = pd_peer(engine, "prefill"), pd_peer(engine, "decode")
        w = admit_one(engine)
        assert w.phase == "reserving"
        assert (p.reserved_blocks, p.reserved_slots, p.reserved_compute_slots) == (
            w.reserved_p_blocks,
            1,
            1,
        )
        assert (d.reserved_blocks, d.reserved_slots) == (w.reserved_d_blocks, 1)
        assert sent_commands(d, "reserve")[0]["tid"] == w.tid

        engine._message(
            d, dict(kind="reserved", tid=w.tid, tables=((1,),), slot=2, cached_tokens=0)
        )
        assert w.phase == "prefill"
        prefill = sent_commands(p, "prefill")[0]
        assert prefill["kind"] == "prefill"
        assert prefill["tables"] == ((1,),) and prefill["slot"] == 2
        assert engine.cache_manager.num_used_blocks == 12

        engine._message(p, dict(kind="prefill_started", tid=w.tid))
        engine._message(p, dict(kind="prefill_complete", tid=w.tid))
        engine._message(p, dict(kind="prefill_complete", tid=w.tid))  # Duplicate is idempotent.
        assert p.reserved_compute_slots == 0 and w.p_compute_released
        assert p.reserved_blocks == w.reserved_p_blocks  # Drainage still holds the KV credits.
        assert engine.cache_manager.num_used_blocks == 12

        engine._message(p, dict(kind="commit", tid=w.tid, count=3))
        assert sent_commands(d, "commit") == [dict(kind="commit", tid=w.tid, count=3)]

        engine._message(d, dict(kind="activated", tid=w.tid))
        assert w.phase == "decode"
        assert sent_commands(p, "ack") == [dict(kind="ack", tid=w.tid)]

        from vllm_rlt.request import RequestOutput

        output = RequestOutput(w.tid, [2, 3, 4, 5], [7, 8], [4, 4], True, "length")
        engine._message(d, dict(kind="output", tid=w.tid, output=output))
        drained = engine.step()
        assert [o.request_id for o in drained] == ["r"]
        assert drained[0].token_ids == [7, 8] and drained[0].finished
        assert "r" not in engine.requests
        # Finishing below max_tokens does not shrink the reservation before release.
        assert (p.reserved_blocks, d.reserved_blocks) == (4, 8)
        assert engine.cache_manager.num_used_blocks == 12

        engine._message(p, dict(kind="released", tid=w.tid, role="prefill"))
        assert (p.reserved_blocks, p.reserved_slots) == (0, 0) and w.p_released
        assert engine.cache_manager.num_used_blocks == 8
        assert w.tid in engine.transfers  # Both sides must release before retirement.
        # A duplicate release must not return credits twice.
        engine._message(p, dict(kind="released", tid=w.tid, role="prefill"))
        assert p.reserved_blocks == 0 and p.reserved_slots == 0
        engine._message(d, dict(kind="released", tid=w.tid, role="decode"))
        assert (d.reserved_blocks, d.reserved_slots) == (0, 0)
        assert w.tid not in engine.transfers
        assert engine.metrics[0]["transfer_id"] == w.tid
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine._terminate()


@pytest.mark.parametrize("point", ["waiting", "reserving", "prefill", "decode"])
def test_pd_coordinator_cancel_message_order(monkeypatch, point):
    """Pin cancel vs cancel_safe ordering at each protocol point."""
    engine = fake_pd_engine(monkeypatch)
    try:
        p, d = pd_peer(engine, "prefill"), pd_peer(engine, "decode")
        if point == "waiting":
            engine.add_request("r", [2, 3], SamplingParams(max_tokens=2))
            w = next(iter(engine.transfers.values()))
        else:
            w = admit_one(engine)
        if point in ("prefill", "decode"):
            engine._message(
                d, dict(kind="reserved", tid=w.tid, tables=((1,),), slot=0, cached_tokens=0)
            )
        if point == "decode":
            engine._message(d, dict(kind="activated", tid=w.tid))

        engine.abort_request("r")
        d_kinds = [m["kind"] for m in sent_commands(d) if m["kind"] != "connect"]
        p_kinds = [m["kind"] for m in sent_commands(p) if m["kind"] != "connect"]
        if point == "waiting":
            assert w.tid not in engine.transfers
            assert d_kinds == [] and p_kinds == []
            assert engine._reserved_blocks_total() == 0
            assert all(x.reserved_slots == 0 and x.reserved_compute_slots == 0 for x in (p, d))
            return
        assert d_kinds[-1] == "cancel"
        assert p_kinds[-1] == "cancel"
        assert w.cancelled and w.tid in engine.transfers

        if point == "reserving":
            # A late reservation for a cancelled generation goes down the safe path.
            engine._message(
                d, dict(kind="reserved", tid=w.tid, tables=((1,),), slot=0, cached_tokens=0)
            )
            assert sent_commands(d)[-1]["kind"] == "cancel_safe"
            assert sent_commands(p, "prefill") == []  # No prefill dispatched.

        # P releases first; only then may D tear down its receiving allocation.
        engine._message(p, dict(kind="released", tid=w.tid, role="prefill"))
        assert sent_commands(d)[-1]["kind"] == "cancel_safe"
        engine._message(d, dict(kind="released", tid=w.tid, role="decode"))
        assert w.tid not in engine.transfers
        assert all(
            x.reserved_blocks == 0 and x.reserved_slots == 0 and x.reserved_compute_slots == 0
            for x in (p, d)
        )
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine._terminate()


def test_pd_wire_format_snapshot(monkeypatch):
    """Pin the field sets of coordinator commands; protocol typing must preserve them."""
    engine = fake_pd_engine(monkeypatch)
    try:
        p, d = pd_peer(engine, "prefill"), pd_peer(engine, "decode")
        w = admit_one(engine)
        engine._message(
            d, dict(kind="reserved", tid=w.tid, tables=((1,),), slot=0, cached_tokens=0)
        )
        engine._message(p, dict(kind="commit", tid=w.tid, count=1))
        engine._message(d, dict(kind="activated", tid=w.tid))
        engine.abort_request("r")

        connect = sent_commands(d, "connect")[0]
        reserve = sent_commands(d, "reserve")[0]
        prefill = sent_commands(p, "prefill")[0]
        commit = sent_commands(d, "commit")[0]
        ack = sent_commands(p, "ack")[0]
        cancel, cancel_p = sent_commands(d, "cancel")[0], sent_commands(p, "cancel")[0]
        assert set(connect) == {"kind", "peers"}
        assert set(reserve) == {"kind", "tid", "tokens", "params", "trace_id", "peer"}
        assert set(prefill) == {
            "kind",
            "tid",
            "tokens",
            "params",
            "trace_id",
            "peer",
            "tables",
            "slot",
            "cached_tokens",
        }
        assert set(commit) == {"kind", "tid", "count"}
        assert set(ack) == set(cancel_p) == set(cancel) == {"kind", "tid"}
    finally:
        engine._terminate()


def test_pd_profile_reply_bypasses_transfer_state(monkeypatch):
    """Session-scoped profile replies never enter transfer routing."""
    engine = fake_pd_engine(monkeypatch)
    try:
        w = admit_one(engine)
        d = pd_peer(engine, "decode")
        engine._message(d, dict(kind="profile_reply", control_id="x", session_id="s"))
        assert w.phase == "reserving" and not w.cancelled

        engine._profile_session = "s"
        replies = {}
        engine._profile_replies["c1"] = replies
        engine._message(
            d, dict(kind="profile_reply", control_id="c1", session_id="s", result={"ok": True})
        )
        assert list(replies) == [d.name]
        # A reply from a stale session is ignored.
        engine._message(d, dict(kind="profile_reply", control_id="c1", session_id="old"))
        assert len(replies) == 1
        assert w.phase == "reserving"
        engine._profile_session = None  # Let _terminate skip the manifest path.
    finally:
        engine._terminate()


def create_nanbeige_pd(checkpoint):
    pytest.importorskip("nixl")
    from vllm_rlt.pd.engine import PDEngine

    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 reserved visible GPUs")
    return PDEngine(
        checkpoint,
        pd_config=PDConfig(
            prefill_devices=(0,),
            decode_devices=(1,),
            transfer_chunk_bytes=4096,
            max_inflight_bytes=16384,
            request_timeout=90,
            startup_timeout=180,
        ),
        prefill_cache_config=CacheConfig(128, 4, "last_exited"),
        decode_cache_config=CacheConfig(128, 4, "last_exited"),
        prefill_scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=8, prefill_chunk_size=8
        ),
        decode_scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2),
        execution_config=ExecutionConfig(async_scheduling=True),
        attention_backend="triton",
    )


def nanbeige_reference(checkpoint):
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device="cuda:0", dtype=torch.bfloat16)
    assert isinstance(model, NanbeigeForCausalLM)
    assert model.config.total_ut_steps == 2
    return LLMEngine(
        model,
        cache_config=CacheConfig(128, 4, "last_exited"),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
        attention_backend="triton",
    )


def assert_nanbeige_idle(engine):
    assert not engine.requests and not engine.transfers
    assert engine.cache_manager.num_used_blocks == 0
    assert all(
        p.reserved_slots == 0 and p.reserved_blocks == 0 and p.reserved_compute_slots == 0
        for p in engine.peers.values()
    )


def assert_nanbeige_stopped(engine):
    assert all(not p.process.is_alive() for p in engine.peers.values())
    assert len(engine.worker_metrics) == 2
    assert all(m["used_blocks"] == 0 for m in engine.worker_metrics.values())


@pytest.mark.gpu
def test_pd_nanbeige_1p1d_matches_single_engine(nanbeige_checkpoint):
    """Real checkpoint loading and NIXL KV/hidden handoff, without forced exit traces."""
    params = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
    with create_nanbeige_pd(nanbeige_checkpoint) as engine:
        with closing(nanbeige_reference(nanbeige_checkpoint)) as reference:
            for i, prompt in enumerate(([2, 3, 4, 5, 6] * 3, [6, 5, 4])):
                for target in (engine, reference):
                    target.add_request(str(i), prompt, params)
            actual, expected = drain(engine), drain(reference)
            assert actual == expected
            assert len(actual) == 2
            for output in actual.values():
                assert output.finished and output.finish_reason == "length"
                assert len(output.token_ids) == params.max_tokens
                assert output.exit_depths == [2] * params.max_tokens
            # Single output exercises hidden-state handoff without a recurrent decode.
            for request_id, options in (
                ("first-only", replace(params, max_tokens=1)),
                ("seeded", replace(params, temperature=0.7, top_k=32, seed=947)),
            ):
                for target in (engine, reference):
                    target.add_request(request_id, [6, 5, 4], options)
                result = drain(engine)
                assert result == drain(reference)
                assert result[request_id].exit_depths == [2] * options.max_tokens
        assert_nanbeige_idle(engine)
    assert_nanbeige_stopped(engine)
    assert sum(m["bytes_sent"] for m in engine.worker_metrics.values()) > 0


@pytest.mark.gpu
@pytest.mark.parametrize("phase", ["prefill", "handoff", "decode", "decode_output"])
def test_pd_nanbeige_cancel_reuse_and_close(nanbeige_checkpoint, phase, monkeypatch):
    params = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
    engine = create_nanbeige_pd(nanbeige_checkpoint)
    try:
        engine.add_request("reuse", [2, 3, 4, 5] * 12, params)
        old = next(iter(engine.transfers.values()))
        message = engine._message
        boundary = dict(
            prefill="prefill_started", handoff="commit", decode="activated", decode_output="output"
        )[phase]
        replacement = replace(params, max_tokens=4)
        prompt = [6, 5, 4, 3, 2]
        cancelled = False

        def cancel_at_boundary(peer, event):
            nonlocal cancelled
            # Commit marks P's submitted handoff, before the coordinator activates D.
            # Observe protocol boundaries rather than racing sleeps against CUDA.
            reached = event.get("tid") == old.tid and event["kind"] == boundary
            if phase == "decode_output" and reached:
                reached = len(event["output"].token_ids) >= 2
            if reached and not cancelled:
                if phase in ("decode", "decode_output"):
                    message(peer, event)
                if phase == "decode_output":
                    assert event["output"].token_ids and not event["output"].finished
                aborted = engine.abort_request("reuse")
                assert aborted.finished and aborted.finish_reason == "abort"
                cancelled = True
                assert old.tid in engine.transfers
                engine.add_request("reuse", prompt, replacement)
                if phase in ("decode", "decode_output"):
                    return
            message(peer, event)

        monkeypatch.setattr(engine, "_message", cancel_at_boundary)
        deadline = time.monotonic() + 60
        while not cancelled:
            engine.step()
            assert time.monotonic() < deadline, f"did not reach {phase} boundary"
        new = next(w for w in engine.transfers.values() if not w.cancelled)
        assert new.tid != old.tid
        with closing(nanbeige_reference(nanbeige_checkpoint)) as reference:
            reference.add_request("reuse", prompt, replacement)
            actual = drain(engine)
            assert actual == drain(reference)
            assert actual["reuse"].exit_depths == [2] * replacement.max_tokens
        assert_nanbeige_idle(engine)
        # Closing with a live request must drain cancellation before worker exit.
        engine.add_request("live", [2, 3] * 24, params)
        engine.step()
    finally:
        engine.close()
    assert_nanbeige_idle(engine)
    assert_nanbeige_stopped(engine)


@pytest.mark.gpu
def test_pd_nanbeige_eos_after_handoff(tmp_path):
    """Deterministic EOS on the first output, through the unchanged P/D path."""
    torch.manual_seed(42)
    model = NanbeigeForCausalLM(tiny_nanbeige_config(head_dim=64, eos_token_id=0))
    with torch.no_grad():
        model.lm_head.weight.zero_()  # Greedy tie-breaking produces token 0 (EOS).
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    checkpoint = str(tmp_path)
    with create_nanbeige_pd(checkpoint) as engine:
        with closing(nanbeige_reference(checkpoint)) as reference:
            params = SamplingParams(max_tokens=4)
            for target in (engine, reference):
                target.add_request("eos", [2, 3, 4], params)
            result = drain(engine)
            assert result == drain(reference)
            assert result["eos"].token_ids == [0]
            assert result["eos"].exit_depths == [2]
            assert result["eos"].finished and result["eos"].finish_reason == "stop"
        assert_nanbeige_idle(engine)
    assert_nanbeige_stopped(engine)
