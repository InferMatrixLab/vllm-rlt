"""Graph replay must update KV addresses, request identities and host progress."""

import gc

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


def test_cuda_graph_validation():
    with pytest.raises(ValueError, match="positive"):
        ExecutionConfig(cuda_graph_max_graphs=0)
    with pytest.raises(ValueError, match="boolean"):
        ExecutionConfig(cuda_graphs=1)
    with pytest.raises(ValueError, match="CUDA graphs require CUDA"):
        LLMEngine(
            OuroForCausalLM(tiny_ouro_config()), execution_config=ExecutionConfig(cuda_graphs=True)
        )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn"])
@pytest.mark.parametrize("mode", ["sync", "async", "multi"])
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("static", [False, True])
def test_graph_replay_matches_eager_and_reuses_requests(backend, mode, layout, static):
    torch.manual_seed(321)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    outputs = []
    for graphs in [False, True]:
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=64, block_size=16, layout=layout),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
            ),
            execution_config=ExecutionConfig(
                async_scheduling=mode != "sync",
                multi_stream=mode == "multi",
                static_buffers=static,
                pad_to_power_of_two=static,
                cuda_graphs=graphs,
            ),
            exit_config=ExitConfig("ouro_delayed"),
            attention_backend=backend,
        )
        rounds = []
        for reuse in range(2):
            for i in range(3):
                engine.add_request(
                    str(i),
                    [2 + i, 3, 4] * (i + 1 + reuse),
                    SamplingParams(
                        max_tokens=5 + i, min_loops=2, exit_threshold=0.2, ignore_eos=True
                    ),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for out in engine.step():
                    if out.finished:
                        finished[out.request_id] = (out.token_ids, out.exit_depths)
            assert engine.cache_manager.num_used_blocks == 0
            rounds.append(finished)
        if graphs:
            assert engine.model_runner.graphs.captures > 0
            assert engine.model_runner.graphs.replays > engine.model_runner.graphs.captures
        outputs.append(rounds)
    assert outputs[0] == outputs[1]


@pytest.mark.gpu
def test_graph_cache_limit_fallback_and_abort():
    torch.manual_seed(19)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(
            cuda_graphs=True,
            cuda_graph_max_batch_size=2,
            cuda_graph_max_graphs=1,
            async_scheduling=True,
        ),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend="triton",
    )
    for count in [1, 2, 3]:
        for i in range(count):
            engine.add_request(str(i), [2, 3, 4], SamplingParams(max_tokens=8, ignore_eos=True))
        for _ in range(12):
            engine.step()
        for rid in list(engine.scheduler.requests):
            engine.abort_request(rid)
        while engine.has_unfinished_requests():
            engine.step()
        assert engine.cache_manager.num_used_blocks == 0
    assert engine.model_runner.graphs.captures == 1
    assert engine.model_runner.graphs.fallbacks > 0


@pytest.mark.parametrize("enabled", [False, True])
def test_speculative_graph_memory_budget_scales_with_pools_and_retained_outputs(enabled):
    from dataclasses import replace

    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.core.memory import graph_memory_reserve

    model = OuroForCausalLM(tiny_ouro_config())
    scheduler = SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=4096)
    execution = ExecutionConfig(cuda_graphs=enabled, cuda_graph_max_batch_size=768)
    base = graph_memory_reserve(model, scheduler, execution)
    spec = graph_memory_reserve(model, scheduler, execution, SpeculativeConfig(2))
    if not enabled:
        assert base == spec == 0
        return
    assert base == execution.cuda_graph_memory_reserve_bytes
    logits = 16 * 768 * model.config.vocab_size * next(model.parameters()).element_size()
    assert spec == 3 * base + logits
    # Reducing token/row capacity also reduces the persistent-output allowance.
    limited = graph_memory_reserve(
        model, replace(scheduler, max_num_batched_tokens=64), execution, SpeculativeConfig(4)
    )
    assert limited == 3 * base + logits // 12


def test_async_graph_memory_accounts_for_every_precaptured_bucket():
    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.core.memory import graph_memory_reserve

    model = OuroForCausalLM(tiny_ouro_config())
    execution = ExecutionConfig(cuda_graphs=True, async_scheduling=True)
    # Even tiny admitted work has the configured capture capacity prepared.
    scheduler = SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=1)
    from vllm_rlt.worker.cuda_graph import speculative_capture_sizes

    reserve = graph_memory_reserve(model, scheduler, execution, SpeculativeConfig(2))
    rows = sum(speculative_capture_sizes(execution, scheduler, SpeculativeConfig(2)))
    retained = rows * model.config.vocab_size * next(model.parameters()).element_size()
    assert reserve == 3 * execution.cuda_graph_memory_reserve_bytes + retained
    # Preferred rows clipped to the global limit still cover the full ladder.
    wide = SchedulerConfig(max_num_seqs=1024, max_num_batched_tokens=1 << 16)
    sizes = speculative_capture_sizes(execution, wide, SpeculativeConfig(2))
    assert max(sizes) == 128 and min(sizes) == 1


def test_unset_graph_rows_cover_async_verification_and_keep_explicit_limits():
    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.config import DEFAULT_CUDA_GRAPH_ROWS, resolve_cuda_graph_rows
    from vllm_rlt.core.memory import graph_memory_reserve
    from vllm_rlt.worker.cuda_graph import graph_capture_sizes

    scheduler = SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=4096)
    unset = ExecutionConfig(async_scheduling=True, cuda_graphs=True)
    resolved = resolve_cuda_graph_rows(unset, scheduler, SpeculativeConfig(2))
    assert resolved.cuda_graph_max_batch_size == 768
    assert max(graph_capture_sizes(resolved)) == 768
    # The token budget bounds verification rows; small configs never shrink coverage.
    budget = SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=1024)
    assert (
        resolve_cuda_graph_rows(unset, budget, SpeculativeConfig(4)).cuda_graph_max_batch_size
        == 1024
    )
    small = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=128)
    assert (
        resolve_cuda_graph_rows(unset, small, SpeculativeConfig(2)).cuda_graph_max_batch_size
        == DEFAULT_CUDA_GRAPH_ROWS
    )
    # Native and synchronous paths keep the previous default.
    native = resolve_cuda_graph_rows(unset, scheduler)
    sync = resolve_cuda_graph_rows(
        ExecutionConfig(cuda_graphs=True), scheduler, SpeculativeConfig(2)
    )
    assert native.cuda_graph_max_batch_size == sync.cuda_graph_max_batch_size == 128
    explicit = ExecutionConfig(
        async_scheduling=True, cuda_graphs=True, cuda_graph_max_batch_size=128
    )
    assert resolve_cuda_graph_rows(explicit, scheduler, SpeculativeConfig(2)) is explicit
    # Memory planning reads the same resolved limit as the graph buckets.
    model = OuroForCausalLM(tiny_ouro_config())
    pinned = ExecutionConfig(async_scheduling=True, cuda_graphs=True, cuda_graph_max_batch_size=768)
    assert graph_memory_reserve(model, scheduler, resolved, SpeculativeConfig(2)) == (
        graph_memory_reserve(model, scheduler, pinned, SpeculativeConfig(2))
    )
    with pytest.raises(ValueError, match="positive"):
        ExecutionConfig(cuda_graph_max_batch_size=0)


def test_speculative_buckets_keep_full_batch_rows_exact_within_entry_limit():
    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.worker.cuda_graph import graph_capture_sizes, speculative_capture_sizes

    scheduler = SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=4096)
    rows = ExecutionConfig(async_scheduling=True, cuda_graphs=True, cuda_graph_max_batch_size=768)
    sizes = speculative_capture_sizes(rows, scheduler, SpeculativeConfig(2))
    assert {256, 512, 768} <= set(sizes) and min(sizes) == 1 and len(sizes) <= 16
    # A tight budget retains reachable preferred sizes, including the largest.
    few = ExecutionConfig(cuda_graph_max_batch_size=17, cuda_graph_max_graphs=3)
    small = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16)
    assert speculative_capture_sizes(few, small, SpeculativeConfig(3)) == (8, 12, 16)
    assert graph_capture_sizes(few) == (5, 9, 17)


@pytest.mark.parametrize(
    "budget,expected",
    [(1, (12,)), (2, (8, 12)), (3, (4, 8, 12)), (4, (2, 4, 8, 12)), (5, (1, 2, 4, 8, 12))],
)
def test_speculative_buckets_prioritize_reachable_rows(budget, expected):
    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.worker.cuda_graph import speculative_capture_sizes

    execution = ExecutionConfig(cuda_graph_max_batch_size=128, cuda_graph_max_graphs=budget)
    scheduler = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16)
    assert speculative_capture_sizes(execution, scheduler, SpeculativeConfig(2)) == expected


@pytest.mark.parametrize("startup", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_capture_startup_orders_producer_and_always_drains(monkeypatch, startup, fail):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm_rlt.worker.cuda_graph import _capture

    events = []
    producer = object()
    stream = SimpleNamespace(
        wait_stream=lambda other: events.append(("wait", other)),
        synchronize=lambda: events.append("drain"),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: producer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("device_sync"))
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())

    def run():
        events.append("run")
        if fail:
            raise RuntimeError("warmup failure")
        return "output"

    if fail:
        with pytest.raises(RuntimeError, match="warmup failure"):
            _capture(SimpleNamespace(device="cuda:1"), stream, None, run, startup=startup)
    else:
        _, output = _capture(SimpleNamespace(device="cuda:1"), stream, None, run, startup=startup)
        assert output == "output"
    assert events[0] == (("wait", producer) if startup else "device_sync")
    assert events[-1] == "drain"
    assert events.count("run") == (1 if fail else 3)
    assert events.count("drain") == (1 if fail else 2)


@pytest.mark.gpu
@torch.inference_mode()
def test_coda_startup_does_not_clone_but_serving_owns_outputs(monkeypatch):
    from tests.helpers import tiny_ouro_model
    from vllm_rlt.worker.cuda_graph import CodaGraphs

    model = tiny_ouro_model("cuda", torch.bfloat16)
    graphs = CodaGraphs(model, ExecutionConfig(), bucketed=True, capture_sizes=(1, 4))

    def forbidden(*args, **kwargs):
        raise AssertionError("startup cloned a discarded output")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", forbidden)
        graphs.prepare()
    assert graphs.captures == 2 and graphs.replays == 0
    assert graphs.last_event is not None
    hidden = torch.ones(3, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    call = graphs.select(hidden)
    graphs.prepare_inputs(call, hidden)
    first = graphs.execute(call, hidden)
    saved = first.clone()
    call = graphs.select(hidden + 1)
    graphs.prepare_inputs(call, hidden + 1)
    second = graphs.execute(call, hidden + 1)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, saved)
    torch.cuda.synchronize()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", [None, "warmup", "capture", "drain"])
def test_capture_restores_gc_state_on_success_and_failure(monkeypatch, enabled, failure):
    import gc
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    from vllm_rlt.worker.cuda_graph import _capture

    previous = gc.isenabled()
    states, drains = [], []

    def synchronize():
        drains.append(gc.isenabled())
        if failure == "drain" and len(drains) == 2:
            raise RuntimeError("drain failure")

    @contextmanager
    def capturing(*args, **kwargs):
        assert not gc.isenabled()
        yield

    def run():
        states.append(gc.isenabled())
        phase = "capture" if len(states) == 3 else "warmup"
        if failure == phase:
            raise RuntimeError(f"{phase} failure")
        return "output"

    stream = SimpleNamespace(synchronize=synchronize)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", capturing)
    try:
        (gc.enable if enabled else gc.disable)()
        if failure:
            with pytest.raises(RuntimeError, match=f"{failure} failure"):
                _capture(SimpleNamespace(device="cuda"), stream, None, run)
        else:
            _, output = _capture(SimpleNamespace(device="cuda"), stream, None, run)
            assert output == "output"
        assert gc.isenabled() == enabled
        assert states[:2] == ([enabled] if failure == "warmup" else [enabled, enabled])
        if failure != "warmup":
            assert states[2] is False
            assert drains[-1] is False  # Drain before restoring GC, even on failure.
    finally:
        (gc.enable if previous else gc.disable)()


@pytest.mark.gpu
@torch.inference_mode()
def test_capture_defers_collection_of_unreachable_old_graph():
    import gc
    import weakref

    from vllm_rlt.worker.cuda_graph import _capture

    previous, thresholds = gc.isenabled(), gc.get_threshold()
    collected_during_capture = []
    try:
        # Keep the old graph live through context setup, then make it unreachable
        # during capture and force automatic GC with a low allocation threshold.
        gc.collect()
        gc.disable()
        hidden = torch.ones(16, device="cuda")
        old = torch.cuda.CUDAGraph()
        with torch.cuda.graph(old):
            old_output = hidden + 1
        old.replay()
        torch.cuda.synchronize()
        reference = weakref.ref(
            old,
            lambda _: collected_during_capture.append(torch.cuda.is_current_stream_capturing()),
        )
        owners = [old]
        del old
        gc.set_threshold(10**9, 10**9, 10**9)
        gc.enable()
        stream = torch.cuda.Stream()
        pool = torch.cuda.graph_pool_handle()

        def run():
            if torch.cuda.is_current_stream_capturing():
                cycle = [owners.pop()]
                cycle.append(cycle)
                del cycle
                gc.set_threshold(1, 0, 0)
                for _ in range(8):
                    list(range(16))
            return hidden + 3

        graph, output = _capture(hidden, stream, pool, run)
        gc.set_threshold(*thresholds)
        assert gc.isenabled()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, torch.full_like(hidden, 4))
        assert torch.equal(old_output, torch.full_like(hidden, 2))
        gc.collect()
        assert reference() is None
        assert collected_during_capture == [False]
    finally:
        gc.set_threshold(*thresholds)
        (gc.enable if previous else gc.disable)()


@pytest.mark.gpu
def test_capture_survives_collecting_an_earlier_engines_garbage():
    # torch.cuda.graph does not collect garbage before capture by default. Reference
    # cycles left by an earlier engine (graphs, streams, pinned buffers) must not be
    # freed inside a later capture when an allocation there runs the collector.
    torch.manual_seed(7)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    recurrent = model.recurrent_prepared

    def collecting(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            gc.collect()
        return recurrent(*args, **kwargs)

    def run(collect_in_capture):
        model.recurrent_prepared = collecting if collect_in_capture else recurrent
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=64, block_size=16),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
            ),
            execution_config=ExecutionConfig(cuda_graphs=True, async_scheduling=True),
            exit_config=ExitConfig("ouro_delayed"),
            attention_backend="triton",
        )
        for i in range(3):
            engine.add_request(
                str(i),
                [2 + i, 3, 4] * (i + 1),
                SamplingParams(max_tokens=5 + i, min_loops=2, exit_threshold=0.2, ignore_eos=True),
            )
        finished = {}
        while engine.has_unfinished_requests():
            for out in engine.step():
                if out.finished:
                    finished[out.request_id] = (out.token_ids, out.exit_depths)
        assert engine.model_runner.graphs.captures > 0
        return finished

    gc.collect()
    gc.disable()  # Keep the first engine's cycles alive until the second engine captures.
    try:
        expected = run(collect_in_capture=False)
        actual = run(collect_in_capture=True)
    finally:
        gc.enable()
        model.recurrent_prepared = recurrent
    assert actual == expected


@pytest.mark.gpu
def test_bucketed_kv_graph_combines_active_rows_and_pipeline_write_mask():
    from types import SimpleNamespace

    from vllm_rlt.worker.cuda_graph import _DeviceCache

    keys = torch.full((4, 1, 2, 1, 4), -1.0, device="cuda")
    values = torch.full_like(keys, -2.0)
    cache = SimpleNamespace(key_cache=keys, value_cache=values)
    batch = SimpleNamespace(
        active_rows=torch.zeros((), dtype=torch.int32, device="cuda"),
        write_mask=torch.ones(5, dtype=torch.bool, device="cuda"),
        write_blocks=torch.tensor([1, 2, 3, 0, 1], device="cuda"),
        write_offsets=torch.tensor([0, 0, 0, 0, 1], device="cuda"),
    )
    k = torch.arange(20, dtype=torch.float32, device="cuda").view(5, 1, 4)
    v = k + 100
    proxy = _DeviceCache(cache)
    proxy._write_prepared(0, batch, k, v)  # Compile before capture; no active rows.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        proxy._write_prepared(0, batch, k, v)

    for count, mask in (
        (3, [True, False, True, True, True]),
        (4, [False, True, False, True, True]),
    ):
        keys.fill_(-1)
        values.fill_(-2)
        batch.active_rows.fill_(count)
        batch.write_mask.copy_(torch.tensor(mask, device="cuda"))
        graph.replay()
        expected_k, expected_v = torch.full_like(keys, -1), torch.full_like(values, -2)
        for row in range(count):
            if mask[row]:
                block, offset = int(batch.write_blocks[row]), int(batch.write_offsets[row])
                expected_k[block, 0, offset] = k[row]
                expected_v[block, 0, offset] = v[row]
        assert torch.equal(keys, expected_k)
        assert torch.equal(values, expected_v)
