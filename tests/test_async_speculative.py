"""Cross-round device progress, reservation and lifetime contracts."""

import pytest
import torch

from tests.helpers import (
    bootstrap_speculation,
    make_speculative_engine,
    tiny_ouro_model,
)
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.scheduler import Scheduler
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.request import Request, RequestPhase

tiny = tiny_ouro_model
bootstrap = bootstrap_speculation


def make_engine(
    model,
    *,
    asynchronous=True,
    k=3,
    budget=16,
    prefix=False,
    multi_stream=True,
    graphs=False,
    max_graph_rows=128,
    max_graphs=16,
    max_num_seqs=4,
    spec_kwargs=None,
):
    return LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(
            max(256, 64 * max_num_seqs),
            2,
            enable_prefix_caching=prefix,
            incremental_allocation=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(
            async_scheduling=asynchronous,
            multi_stream=multi_stream,
            cuda_graphs=graphs,
            cuda_graph_max_batch_size=max_graph_rows,
            cuda_graph_max_graphs=max_graphs,
        ),
        # Mechanism tests pin K drafts per round unless a test opts into a policy.
        speculative_config=SpeculativeConfig(
            k, **({"draft_policy": "fixed"} | (spec_kwargs or {}))
        ),
    )


def execute_graph(graphs, hidden, metadata=None):
    call = graphs.select(hidden, metadata)
    graphs.prepare_inputs(call, hidden)
    return graphs.execute(call, hidden)


def drain(engine):
    outputs = {}
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            return outputs
        for output in engine.step():
            outputs[output.request_id] = output
    pytest.fail("async speculative engine failed to drain")


def test_scheduler_uses_upper_bound_and_releases_rejected_budget():
    scheduler = Scheduler(SchedulerConfig(), None, SpeculativeConfig(3))
    r = Request("r", [2, 3], SamplingParams(max_tokens=8), generated_token_ids=[4])
    r.num_output_placeholders = 4
    scheduled_request = scheduler._make_scheduled_request(r, RequestPhase.SPECULATIVE, 16)
    assert (scheduled_request.token_start, scheduled_request.token_count) == (6, 3)
    # One committed token instead of the four reserved: recover three slots.
    r.num_output_placeholders = 0
    r.generated_token_ids.append(5)
    scheduled_request = scheduler._make_scheduled_request(r, RequestPhase.SPECULATIVE, 16)
    assert (scheduled_request.token_start, scheduled_request.token_count) == (3, 4)


def test_async_spec_rejects_cpu_before_cache_allocation():
    with pytest.raises(ValueError, match="requires CUDA"):
        make_speculative_engine(tiny_ouro_model())


def test_adaptive_round_uses_only_ready_requests_and_skips_empty_decisions():
    from types import SimpleNamespace

    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=2),
        SimpleNamespace(ensure_capacity=lambda *_: True),
        SpeculativeConfig(3),
    )
    for rid in ("blocked", "inflight", "ready"):
        request = Request(rid, [2, 3], SamplingParams(max_tokens=8), generated_token_ids=[4])
        scheduler.requests[rid] = request
        scheduler.enqueue(request, RequestPhase.SPECULATIVE)
    scheduler.requests["inflight"].num_speculative_rounds = 2
    scheduler.execution_ready = lambda request: request.request_id != "blocked"
    decisions = []

    def choose(starts):
        decisions.append(starts)
        return 0

    scheduler.draft_tokens = choose
    batch = scheduler.take_phases((RequestPhase.SPECULATIVE,))
    assert decisions == [[2]]
    assert [(r.request.request_id, r.token_count) for r in batch.requests] == [("ready", 1)]
    assert scheduler.take_phases((RequestPhase.SPECULATIVE,)) is None
    assert decisions == [[2]]


@pytest.mark.gpu
@pytest.mark.parametrize("k,budget", [(1, 1), (2, 7), (4, 16), (8, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("graphs", [False, True])
def test_async_matches_sync_with_ragged_limits_and_prefix_reuse(k, budget, dtype, graphs):
    model = tiny("cuda", dtype)
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in (1, 13, 8)]
    prompts = [[2, 3, 4, 5], [2, 3, 4, 7], [9]]
    results = []
    for asynchronous in (False, True):
        e = make_engine(
            model, asynchronous=asynchronous, k=k, budget=budget, prefix=True, graphs=graphs
        )
        for repetition in range(2):
            for i, (prompt, p) in enumerate(zip(prompts, params)):
                e.add_request(str(i), prompt, p)
            output = drain(e)
            results.append([output[str(i)].token_ids for i in range(3)])
            assert all(o.finished for o in output.values())
        if asynchronous:
            if graphs:
                assert e.speculative_runner.graphs.replays > 0
                assert e.speculative_runner.coda_graphs.replays > 0
            assert not e.speculative_runner.states
            assert all(not buffer.leased for buffer in e.speculative_runner.buffers)
    assert all(r == results[0] for r in results)


@pytest.mark.gpu
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@pytest.mark.parametrize("graphs", [False, True])
def test_second_round_executes_before_cpu_commit_at_every_rejection(accepted, monkeypatch, graphs):
    model = tiny("cuda")
    e = make_engine(model, graphs=graphs)
    request = bootstrap(e)
    start = request.position
    calls = 0

    def controlled(hidden):
        nonlocal calls
        phase = calls % 4
        calls += 1
        logits = torch.full((len(hidden), model.config.vocab_size), -100.0, device="cuda")
        if phase < 3:
            logits[:, 10 + phase] = 100
        else:
            for row in range(4):
                logits[row, 10 + row if row < accepted else 20 + row] = 100
        return logits

    # Keep the synthetic phase policy outside capture: Python call counters
    # describe test phases, not replayable model operations. Core graphs still run.
    monkeypatch.setattr(e.speculative_runner, "_coda", controlled)

    # A CPU read of candidate/acceptance values would fail during submission.
    def forbidden(*args, **kwargs):
        raise AssertionError("host read in cross-round GPU submission")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        e.step()
        e.step()
    assert len(e._pending_speculative) == 2
    assert len(request.generated_token_ids) == 1
    assert e.speculative_runner.submitted_before_collect == 1
    state = e.speculative_runner.states["r"]
    e.speculative_runner.synchronize()
    assert int(state.position) == start + 2 * (accepted + 1)
    # Collecting the older round must not rewind the newer device state.
    out = e._collect_speculative()[0]
    assert out.token_ids[1:] == list(range(10, 10 + accepted)) + [20 + accepted]
    assert int(state.position) == start + 2 * (accepted + 1)
    e.abort_request("r")
    drain(e)
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_eos_discards_queued_round_and_request_id_can_be_reused(monkeypatch, graphs):
    model = tiny("cuda")
    e = make_engine(model, graphs=graphs)
    bootstrap(e, ignore_eos=False)
    coda = e.speculative_runner._coda
    monkeypatch.setattr(
        e.speculative_runner,
        "_coda",
        lambda h: torch.zeros(
            len(h),
            model.config.vocab_size,
            device="cuda",
        ),
    )
    e.step()
    e.step()
    out = e._collect_speculative()[0]
    assert out.finished and out.token_ids[-1] == 0 and len(out.token_ids) == 2
    # Reuse while an old ticket still exists. It must never touch the new request.
    monkeypatch.setattr(e.speculative_runner, "_coda", coda)
    e.add_request("r", [7, 8], SamplingParams(max_tokens=5, ignore_eos=True))
    output = drain(e)["r"]
    assert len(output.token_ids) == 5
    assert e.speculative_runner.discarded_rounds == 1
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_cancel_and_partial_submission_failure_reclaim_resources(monkeypatch, graphs):
    model = tiny("cuda")
    e = make_engine(model, graphs=graphs)
    bootstrap(e)
    e.step()
    e.step()
    assert e.abort_request("r").finish_reason == "abort"
    drain(e)
    assert not e.speculative_runner.states
    assert e.cache_manager.num_used_blocks == 0
    bootstrap_speculation(e)

    def fail(*args, **kwargs):
        raise RuntimeError("injected target failure")

    monkeypatch.setattr(e.speculative_runner, "_verify", fail)
    with pytest.raises(RuntimeError, match="injected"):
        e.step()
    assert not e.scheduler.requests
    assert not e.speculative_runner.states
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
def test_async_rejects_random_sampling():
    e = make_speculative_engine(tiny_ouro_model("cuda"))
    with pytest.raises(ValueError, match="greedy"):
        e.add_request("r", [2], SamplingParams(temperature=0.8))


@pytest.mark.gpu
def test_async_rejects_host_reading_attention():
    with pytest.raises(ValueError, match="device_context_lengths"):
        LLMEngine(
            tiny_ouro_model("cuda"),
            attention_backend="torch",
            execution_config=ExecutionConfig(async_scheduling=True),
            speculative_config=SpeculativeConfig(3),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_prefill_arrivals_interleave_with_inflight_rounds(graphs):
    model = tiny("cuda", torch.bfloat16)
    e = make_engine(model, budget=5, prefix=True, graphs=graphs)
    bootstrap(e, max_tokens=17)
    e.step()
    e.step()
    assert len(e._pending_speculative) == 2
    prompts = [[2, 3, 4], [2, 3, 4, 8, 9], [7]]
    lengths = [17, 7, 9]
    for i in (1, 2):
        e.add_request(str(i), prompts[i], SamplingParams(max_tokens=lengths[i], ignore_eos=True))
    actual = drain(e)
    reference = make_speculative_engine(model, asynchronous=False, budget=5, prefix=True)
    for rid, prompt, length in zip(["r", "1", "2"], prompts, lengths):
        reference.add_request(rid, prompt, SamplingParams(max_tokens=length, ignore_eos=True))
    expected = drain(reference)
    assert {rid: o.token_ids for rid, o in actual.items()} == {
        rid: o.token_ids for rid, o in expected.items()
    }
    assert all(o.finished for o in actual.values())
    assert not e.speculative_runner.states
    assert all(not buffer.leased for buffer in e.speculative_runner.buffers)
    e.close()
    reference.close()


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
def test_close_drains_both_submitted_rounds(multi_stream, graphs):
    e = make_engine(tiny("cuda"), multi_stream=multi_stream, graphs=graphs)
    bootstrap(e)
    e.step()
    e.step()
    assert len(e._pending_speculative) == 2
    e.close()
    assert not e.has_unfinished_requests()
    assert e.cache_manager.num_used_blocks == 0
    assert not e.speculative_runner.states
    assert all(not buffer.leased for buffer in e.speculative_runner.buffers)
    e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("graphs", [False, True])
def test_two_round_kv_matches_serial_replay_at_every_depth(dtype, graphs):
    model = tiny("cuda", dtype)
    e = make_engine(model, graphs=graphs)
    request = bootstrap(e)
    e.step()
    e.step()
    results = [ticket.collect().speculative[0] for ticket in e._pending_speculative]
    tokens = request.prompt_token_ids + request.generated_token_ids
    tokens += [token for result in results for token in result.token_ids]
    # Last emitted token has not been forwarded; all earlier tokens have KV.
    tokens = tokens[:-1]
    oracle = make_speculative_engine(model, asynchronous=False).cache_manager
    assert oracle.allocate("serial", len(tokens))
    with torch.inference_mode():
        for position, token in enumerate(tokens):
            hidden = model.prelude(torch.tensor([token], device="cuda"))
            for depth in range(model.config.total_ut_steps):
                hidden, _ = model.recurrent(hidden, ["serial"], [depth], [position], oracle)
    allocation = e.cache_manager._get_allocation("r")
    pos = torch.arange(len(tokens), device="cuda")
    atol, rtol = (4e-5, 4e-5) if dtype == torch.float32 else (0.08, 0.04)
    for depth in range(model.config.total_ut_steps):
        pages = torch.tensor(allocation.block_tables[depth], device="cuda")
        blocks, offsets = pages[pos // 2], pos % 2
        for layer in range(model.config.num_hidden_layers):
            expected_k, expected_v = oracle.read(layer, "serial", depth, len(tokens))
            torch.testing.assert_close(
                e.cache_manager.key_cache[blocks, layer, offsets], expected_k, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                e.cache_manager.value_cache[blocks, layer, offsets],
                expected_v,
                atol=atol,
                rtol=rtol,
            )
    e.close()
    assert not e.has_unfinished_requests()
    assert e.cache_manager.num_used_blocks == 0
    assert not e.speculative_runner.states
    assert all(not buffer.leased for buffer in e.speculative_runner.buffers)
    e.close()


def test_active_page_tables_are_contiguous_and_do_not_touch_inactive_storage():
    from types import SimpleNamespace

    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
    from vllm_rlt.worker.async_speculative import AsyncSpeculativeRunner

    cache = KVCacheManager(1, 1, 4, 64, 2, 4)
    assert cache.allocate("short", 4) and cache.allocate("long", 10)
    runner = object.__new__(AsyncSpeculativeRunner)
    runner.cache = cache
    buffer = SimpleNamespace(
        leased=False,
        host_tables=torch.full((128,), -91, dtype=torch.int32),
        tables=torch.full((128,), -92, dtype=torch.int32),
    )
    runner.buffers = [buffer]
    requests = [Request(rid, [2], SamplingParams()) for rid in ("short", "long")]
    batch = SchedulerOutput(
        RequestPhase.SPECULATIVE,
        [
            ScheduledRequest(requests[0], 1, 2),
            ScheduledRequest(requests[1], 7, 2),
        ],
    )
    _, _, host, device = runner._prepare_round(batch)
    assert host.shape == device.shape == (2, 4, 5)
    assert host.is_contiguous() and device.is_contiguous()
    assert host.data_ptr() == buffer.host_tables.data_ptr()
    for i, rid in enumerate(("short", "long")):
        pages = 2 if i == 0 else 5
        for depth in range(4):
            expected = cache._get_allocation(rid).block_tables[depth][:pages]
            assert host[i, depth, :pages].tolist() == list(expected)
    assert not host[0, :, 2:].any()
    assert (buffer.host_tables[40:] == -91).all()
    assert (buffer.tables == -92).all()  # CPU preparation does not upload.
    with pytest.raises(RuntimeError, match="exhausted"):
        runner._prepare_round(batch)
    # Simulate retirement, then shrink both dimensions using the same storage.
    buffer.leased = False
    before = buffer.host_tables.clone()
    smaller = SchedulerOutput(RequestPhase.SPECULATIVE, [ScheduledRequest(requests[0], 0, 1)])
    _, _, host, device = runner._prepare_round(smaller)
    assert host.shape == device.shape == (1, 4, 1)
    assert host.is_contiguous() and device.is_contiguous()
    assert torch.equal(buffer.host_tables[4:], before[4:])


def test_device_metadata_rejects_foreign_and_reused_allocations_without_host_reads(monkeypatch):
    from vllm_rlt.core.kv_cache_manager import KVCacheManager

    cache = KVCacheManager(1, 1, 4, 16, 2, 4)
    foreign = KVCacheManager(1, 1, 4, 16, 2, 4)
    assert cache.allocate("r", 2) and foreign.allocate("r", 2)
    allocation = cache._get_allocation("r")
    arguments = dict(
        allocations=(("r", allocation),),
        position_ids=torch.tensor([0]),
        write_blocks=torch.tensor([allocation.block_tables[0][0]]),
        write_offsets=torch.tensor([0]),
        block_tables=torch.tensor([allocation.block_tables[0]], dtype=torch.int32),
        context_lengths=torch.tensor([1], dtype=torch.int32),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata creation read device values on CPU")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        batch = cache.metadata_builder.prepare_device(**arguments)
    assert not hasattr(batch, "owner")
    with pytest.raises(RuntimeError, match="stale"):
        foreign._require_live_batch(batch)
    cache.free("r")
    assert cache.allocate("r", 2)
    with pytest.raises(RuntimeError, match="stale"):
        cache._require_live_batch(batch)
    with pytest.raises(RuntimeError, match="stale"):
        cache.metadata_builder.prepare_device(**arguments)


@pytest.mark.gpu
def test_async_greedy_execution_never_acquires_a_generator(monkeypatch):
    # Object identity does not prove the asynchronous greedy path leaves the RNG
    # alone, so run a real request with a spy that fails on any acquire. This needs
    # a reserved device because AsyncSpeculativeRunner allocates CUDA streams; the
    # contract therefore stays pending for M7 3/3.
    e = make_speculative_engine(tiny_ouro_model("cuda"))

    def forbidden(*args, **kwargs):
        raise AssertionError("greedy speculative sampling must not acquire a generator")

    monkeypatch.setattr(e.rng, "acquire", forbidden)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=6, ignore_eos=True))
    outputs = drain(e)
    assert outputs["r"].finish_reason == "length"
    assert e.rng.peek("r") is None


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True])
def test_async_speculative_engine_shares_its_registry(multi_stream):
    # AsyncSpeculativeRunner allocates CUDA streams in its constructor, so this
    # wiring is only verifiable on a reserved device; a CPU construction or mock
    # would not exercise the object the engine actually holds.
    e = make_speculative_engine(tiny_ouro_model("cuda"), multi_stream=multi_stream)
    assert e.async_speculative
    assert e.speculative_runner.rng is e.rng
    assert e.model_runner.rng is e.rng


@pytest.mark.gpu
@pytest.mark.parametrize("max_graph_rows,max_graphs", [(2, 16), (128, 1)])
def test_async_graph_fallback_preserves_ragged_outputs_and_reuses_entries(
    max_graph_rows, max_graphs
):
    model = tiny("cuda", torch.bfloat16)
    results = []
    for graphs in (False, True):
        engine = make_engine(
            model, graphs=graphs, max_graph_rows=max_graph_rows, max_graphs=max_graphs
        )
        repetitions = []
        for _ in range(2):
            for i, length in enumerate((15, 9, 6)):
                engine.add_request(
                    str(i),
                    [2 + i, 3, 4],
                    SamplingParams(max_tokens=length, ignore_eos=True),
                )
            outputs = drain(engine)
            repetitions.append({rid: (o.token_ids, o.exit_depths) for rid, o in outputs.items()})
            assert engine.cache_manager.num_used_blocks == 0
        if graphs:
            core, coda = engine.speculative_runner.graphs, engine.speculative_runner.coda_graphs
            assert core.captures > 0 and core.replays > core.captures
            assert coda.captures > 0 and coda.replays > coda.captures
            if max_graph_rows == 2:
                assert core.fallbacks > 0 and coda.fallbacks > 0
            else:
                assert core.fallbacks == coda.fallbacks == 0
            assert core.captures <= max_graphs and coda.captures <= max_graphs
            assert all(n <= max_graph_rows for n in core.entries)
        results.append(repetitions)
        engine.close()
    assert results[0] == results[1]


@pytest.mark.gpu
@torch.inference_mode()
def test_device_graph_replay_owns_outputs_and_checks_replaced_allocations(monkeypatch):
    engine = make_engine(tiny("cuda", torch.bfloat16), graphs=True)
    cache, runner = engine.cache_manager, engine.speculative_runner
    assert cache.allocate("manual", 4)
    allocation = cache._get_allocation("manual")

    def metadata(depth):
        table = torch.tensor([allocation.block_tables[depth]], device="cuda", dtype=torch.int32)
        return cache.metadata_builder.prepare_device(
            allocations=(("manual", allocation),),
            position_ids=torch.zeros(1, device="cuda", dtype=torch.int64),
            write_blocks=table[:, 0].long(),
            write_offsets=torch.zeros(1, device="cuda", dtype=torch.int64),
            block_tables=table,
            context_lengths=torch.ones(1, device="cuda", dtype=torch.int32),
        )

    batch = metadata(0)
    hidden = engine.model.prelude(torch.tensor([2], device="cuda"))
    first, _ = execute_graph(runner.graphs, hidden, batch)
    saved = first.clone()
    second_batch = metadata(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("device graph replay read a GPU position on the host")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        second, _ = execute_graph(runner.graphs, hidden + 1, second_batch)
    assert runner.graphs.captures == len(runner.graphs.capture_sizes)
    assert runner.graphs.replays == 2
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, saved)
    assert all(written.prefix == 0 for plane in allocation.written for written in plane)
    torch.cuda.synchronize()
    cache.free("manual")
    assert cache.allocate("manual", 4)
    with pytest.raises(RuntimeError, match="stale"):
        execute_graph(runner.graphs, hidden, batch)
    cache.free("manual")
    engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("phase", ["core", "coda"])
def test_async_graph_failed_warmup_drains_capture_stream_before_cleanup(monkeypatch, phase):
    engine = make_engine(tiny("cuda"), graphs=True)
    bootstrap(engine)
    runner = engine.speculative_runner
    completed = torch.cuda.Event()
    if phase == "core":
        # Force a capture failure despite normal startup pre-capture, to retain
        # the partial-failure cleanup contract if an entry needs rebuilding.
        runner.graphs.entries.clear()
        original = engine.model.recurrent_prepared

        def fail(hidden, batch, cache, **kwargs):
            result = original(hidden, batch, cache, **kwargs)
            # Only fail the graph proxy traversal, after GPU work is submitted.
            if cache is not engine.cache_manager:
                torch.cuda._sleep(10_000_000)
                completed.record(torch.cuda.current_stream())
                raise RuntimeError("injected core warmup failure")
            return result

        monkeypatch.setattr(engine.model, "recurrent_prepared", fail)
    else:
        runner.coda_graphs.entries.clear()
        original = engine.model.coda

        def fail(hidden):
            original(hidden)
            torch.cuda._sleep(10_000_000)
            completed.record(torch.cuda.current_stream())
            raise RuntimeError("injected coda warmup failure")

        monkeypatch.setattr(engine.model, "coda", fail)
    with pytest.raises(RuntimeError, match=f"injected {phase} warmup failure"):
        engine.step()
    assert completed.query()
    assert not engine.scheduler.requests and not runner.states
    assert engine.cache_manager.num_used_blocks == 0
    assert all(not b.leased and not b.keepalive for b in runner.buffers)
    engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("k,concurrency", [(2, 43), (4, 26)])
def test_default_graph_row_limit_preserves_high_concurrency_outputs(k, concurrency, caplog):
    model = tiny("cuda")
    results = []
    for graphs in (False, True):
        engine = make_engine(
            model,
            k=k,
            graphs=graphs,
            max_num_seqs=concurrency,
            budget=concurrency * (k + 1),
        )
        for i in range(concurrency):
            engine.add_request(
                str(i),
                [2, 3, 4],
                SamplingParams(max_tokens=k + 4, ignore_eos=True),
            )
        outputs = drain(engine)
        results.append({rid: output.token_ids for rid, output in outputs.items()})
        if graphs:
            assert "cannot cover" in caplog.text
            core, coda = engine.speculative_runner.graphs, engine.speculative_runner.coda_graphs
            assert core.replays > 0 and coda.replays > 0
            assert core.fallbacks > 0 and coda.fallbacks > 0
            assert all(rows <= 128 for rows in core.entries)
            assert all(rows <= 128 for rows in coda.entries)
        engine.close()
    assert results[0] == results[1]


@pytest.mark.gpu
@pytest.mark.parametrize("k,concurrency", [(2, 43), (4, 26)])
def test_unset_graph_row_limit_covers_high_concurrency_without_fallback(k, concurrency, caplog):
    model = tiny("cuda")
    results = []
    for graphs in (False, True):
        engine = make_engine(
            model,
            k=k,
            graphs=graphs,
            max_graph_rows=None,
            max_num_seqs=concurrency,
            budget=concurrency * (k + 1),
        )
        for i in range(concurrency):
            engine.add_request(
                str(i),
                [2, 3, 4],
                SamplingParams(max_tokens=k + 4, ignore_eos=True),
            )
        outputs = drain(engine)
        results.append({rid: output.token_ids for rid, output in outputs.items()})
        if graphs:
            assert "cannot cover" not in caplog.text
            assert engine.execution_config.cuda_graph_max_batch_size == concurrency * (k + 1)
            core, coda = engine.speculative_runner.graphs, engine.speculative_runner.coda_graphs
            assert max(core.capture_sizes) == concurrency * (k + 1)
            assert core.replays > 0 and coda.replays > 0
            assert core.fallbacks == 0 and coda.fallbacks == 0
        engine.close()
    assert results[0] == results[1]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_bucketed_graphs_reuse_all_shapes_and_never_write_padding(dtype):
    engine = make_engine(tiny("cuda", dtype), graphs=True, max_graph_rows=17, max_graphs=3)
    cache, runner = engine.cache_manager, engine.speculative_runner
    cache.key_cache.fill_(9)
    cache.value_cache.fill_(-7)
    # Rebuild with zero active rows against known sentinel storage: startup
    # capture/warmup itself must not write any KV either.
    runner.graphs.entries.clear()
    runner.graphs.captures = 0
    runner.graphs.prepare()
    assert (cache.key_cache == 9).all() and (cache.value_cache == -7).all()
    # Padding points at block zero: a real unrelated allocation must stay intact.
    assert cache.allocate("guard", 2)
    allocations = []
    for i in range(16):
        rid = f"manual-{i}"
        assert cache.allocate(rid, 2)
        allocations.append((rid, cache._get_allocation(rid)))
    initial_captures = (runner.graphs.captures, runner.coda_graphs.captures)
    assert initial_captures == (3, 3)
    # Fill every bucket, then use new row counts after the cache is full. Also
    # shrink a reused bucket: its former active rows must become inert.
    for count in (8, 12, 16, 1, 3, 6, 10, 2, 15, 1):
        table = torch.tensor(
            [a.block_tables[0] for _, a in allocations[:count]],
            device="cuda",
            dtype=torch.int32,
        )
        batch = cache.metadata_builder.prepare_device(
            allocations=tuple(allocations[:count]),
            position_ids=torch.zeros(count, device="cuda", dtype=torch.long),
            write_blocks=table[:, 0].long(),
            write_offsets=torch.zeros(count, device="cuda", dtype=torch.long),
            block_tables=table,
            context_lengths=torch.ones(count, device="cuda", dtype=torch.int32),
        )
        hidden = engine.model.prelude(torch.arange(2, 2 + count, device="cuda"))
        before_k, before_v = cache.key_cache.clone(), cache.value_cache.clone()
        actual, _ = execute_graph(runner.graphs, hidden, batch)
        untouched = torch.ones_like(cache.key_cache, dtype=torch.bool)
        untouched[table[:, 0].long(), :, 0] = False
        assert torch.equal(cache.key_cache[untouched], before_k[untouched])
        assert torch.equal(cache.value_cache[untouched], before_v[untouched])
        from vllm_rlt.worker.cuda_graph import _DeviceCache

        entry = next(e for size, e in runner.graphs.entries.items() if size >= count)
        padded_expected, _ = engine.model.recurrent_prepared(
            entry.hidden, entry.metadata, _DeviceCache(cache), compute_gate=False
        )
        assert torch.equal(actual, padded_expected[:count])
        expected, _ = engine.model.recurrent_prepared(hidden, batch, cache, compute_gate=False)
        logits = execute_graph(runner.coda_graphs, actual)
        head_entry = next(e for size, e in runner.coda_graphs.entries.items() if size >= count)
        assert torch.equal(logits, engine.model.coda(head_entry.hidden)[:count])
        # Padding can change GEMM rounding; these frozen tiny-model cases also
        # retain greedy equality to the original unpadded eager computation.
        assert torch.equal(logits.argmax(-1), engine.model.coda(expected).argmax(-1))
        for graphs in (runner.graphs, runner.coda_graphs):
            assert set(graphs.entries) == set(graphs.capture_sizes)
            assert graphs.captures == 3
            assert graphs.fallbacks == 0
    torch.cuda.synchronize()
    for rid, _ in allocations:
        cache.free(rid)
    cache.free("guard")
    engine.close()


@pytest.mark.gpu
def test_async_graphs_use_nondefault_device_with_two_live_engines():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a two-GPU scheduler reservation")
    previous = torch.cuda.current_device()
    engines = []
    try:
        torch.cuda.set_device(0)
        for graphs in (False, True):
            engines.append(make_engine(tiny("cuda:1"), graphs=graphs, k=4))
        for engine in engines:
            engine.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
        outputs = [drain(engine)["r"].token_ids for engine in engines]
        assert outputs[0] == outputs[1]
        runner = engines[1].speculative_runner
        assert runner.graphs.replays > 0 and runner.coda_graphs.replays > 0
        assert runner.graphs.capture_stream.device == torch.device("cuda:1")
        assert runner.coda_graphs.stream.device == torch.device("cuda:1")
        assert torch.cuda.current_device() == 0
    finally:
        for engine in engines:
            engine.close()
        torch.cuda.set_device(previous)


@pytest.mark.gpu
def test_async_graph_auto_memory_plan_includes_all_pools_and_logits():
    from vllm_rlt.core.memory import graph_memory_reserve

    model = tiny("cuda", torch.bfloat16)
    scheduler = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16)
    execution = ExecutionConfig(async_scheduling=True, cuda_graphs=True)
    speculative = SpeculativeConfig(4)
    engine = LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(block_size=2),
        scheduler_config=scheduler,
        execution_config=execution,
        speculative_config=speculative,
    )
    assert engine.memory_plan["source"] == "cuda_profile"
    expected = graph_memory_reserve(model, scheduler, execution, speculative)
    assert expected > 3 * execution.cuda_graph_memory_reserve_bytes
    assert engine.memory_plan["cuda_graph_reserve_bytes"] == expected
    engine.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    assert len(drain(engine)["r"].token_ids) == 12
    assert engine.speculative_runner.coda_graphs.replays > 0
    engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bucketed_ragged_arrivals_match_eager_tokens(k, dtype):
    results = []
    for graphs in (False, True):
        engine = make_engine(
            tiny("cuda", dtype),
            k=k,
            graphs=graphs,
            max_graph_rows=32,
            max_graphs=4,
        )
        outputs = {}
        for wave in range(5):
            for i in range(3):
                rid = f"{wave}-{i}"
                engine.add_request(
                    rid,
                    [2 + wave, 3 + i, 4][: 1 + i],
                    SamplingParams(max_tokens=5 + i * 4 + wave, ignore_eos=(i != 0)),
                )
            # Fixed arrivals relative to a drained submission boundary avoid
            # event.query timing selecting different batches in the two arms.
            for _ in range(3):
                for output in engine.step():
                    if output.finished:
                        outputs[output.request_id] = output.token_ids
                engine.speculative_runner.synchronize()
                while engine._pending_speculative:
                    for output in engine._collect_speculative():
                        if output.finished:
                            outputs[output.request_id] = output.token_ids
        outputs.update({rid: o.token_ids for rid, o in drain(engine).items()})
        assert len(outputs) == 15
        results.append(outputs)
        if graphs:
            for cache in (engine.speculative_runner.graphs, engine.speculative_runner.coda_graphs):
                assert cache.captures == len(cache.capture_sizes) == 4
                assert cache.replays > 0 and cache.fallbacks == 0
        engine.close()
    assert results[0] == results[1]


@pytest.mark.gpu
@pytest.mark.parametrize("override", [0, 1])
def test_forced_draft_count_matches_eager_and_fixed_k(override):
    model = tiny("cuda")
    results = []
    for graphs, kwargs in (
        (False, {}),
        (False, {"draft_tokens_override": override}),
        (True, {"draft_tokens_override": override}),
    ):
        engine = make_engine(
            model,
            k=3,
            graphs=graphs,
            max_graph_rows=None,
            max_num_seqs=8,
            budget=32,
            spec_kwargs=kwargs,
        )
        for i in range(8):
            engine.add_request(
                str(i), [2 + i, 3, 4], SamplingParams(max_tokens=11, ignore_eos=True)
            )
        results.append({rid: o.token_ids for rid, o in drain(engine).items()})
        if kwargs:
            assert (
                engine.speculative_runner.stats.drafted_tokens
                <= override * engine.speculative_runner.stats.rounds
            )
        engine.close()
    # FP32 greedy speculation commits the target model's tokens for every K.
    assert results[0] == results[1] == results[2]


@pytest.mark.gpu
def test_adaptive_draft_policy_learns_and_preserves_fp32_greedy_tokens():
    model = tiny("cuda")
    results, policy = [], None
    for kwargs in ({}, {"draft_policy": "adaptive"}):
        engine = make_engine(
            model,
            k=3,
            graphs=True,
            max_graph_rows=None,
            max_num_seqs=8,
            budget=32,
            spec_kwargs=kwargs,
        )
        for i in range(8):
            engine.add_request(
                str(i), [2 + i, 3, 4], SamplingParams(max_tokens=24, ignore_eos=True)
            )
        results.append({rid: o.token_ids for rid, o in drain(engine).items()})
        if kwargs:
            policy = engine.draft_policy
            summary = policy.summary()
            assert summary["observed_rounds"] > 0
            assert sum(summary["choices"]) == policy.rounds
            assert all(t == t for t in summary["theta"])  # finite fit
        engine.close()
    assert results[0] == results[1]
