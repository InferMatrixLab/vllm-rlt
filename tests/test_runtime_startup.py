"""Startup execution must use valid KV and leave no serving request state."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM


def make_engine(*, blocks=64, staged=False, shared=False):
    return StageEngine(
        OuroForCausalLM(tiny_ouro_config()),
        cache_config=CacheConfig(
            blocks,
            2,
            layout="shared" if shared else "last_exited",
            enable_prefix_caching=not shared,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3,
            max_num_batched_tokens=8,
            prefill_chunk_size=4,
            wavefront_chunked_prefill=staged,
        ),
        execution_config=ExecutionConfig(static_buffers=True),
    )


@pytest.mark.parametrize("staged,shared", [(False, False), (True, False), (False, True)])
def test_warmup_executes_valid_history_and_releases_all_request_state(staged, shared):
    from vllm_rlt.worker.warmup import prepare_runner

    engine = make_engine(staged=staged, shared=shared)
    rng = torch.get_rng_state().clone()
    result = prepare_runner(engine.model_runner, engine.scheduler.config)
    cache, runner = engine.cache_manager, engine.model_runner
    assert result["prefill_tokens"] == (1, 4)
    assert result["decode_batches"] == (1, 2, 3)
    assert result["history_tokens"] > cache.block_size
    assert cache.num_used_blocks == 0
    assert not cache._allocations and not cache._prefixes and not cache._pending_prefixes
    assert cache.prefix_queries == cache.prefix_hits == 0
    assert not engine.rng.generators
    assert not runner.state_slots and not runner.prompt_token_owners and not runner.events
    assert not engine.scheduler.requests
    assert all(not getattr(buffer, "allocations", ()) for buffer in runner.packed_metadata_buffers)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    engine.add_request("__warmup_0", [1, 2, 3], SamplingParams(max_tokens=2, ignore_eos=True))
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
    assert len(outputs[-1].token_ids) == 2


def test_warmup_failure_releases_scratch_before_propagating(monkeypatch):
    from vllm_rlt.worker.warmup import prepare_runner

    engine = make_engine(staged=True)
    runner = engine.model_runner
    execute = runner._execute
    calls = 0

    def fail(*args):
        nonlocal calls
        output = execute(*args)
        calls += 1
        if calls == 3:
            raise RuntimeError("startup execution failed")
        return output

    monkeypatch.setattr(runner, "_execute", fail)
    with pytest.raises(RuntimeError, match="startup execution failed"):
        prepare_runner(runner, engine.scheduler.config)
    assert engine.cache_manager.num_used_blocks == 0
    assert engine.cache_manager.prefix_queries == 0
    assert not runner.state_slots and not runner.events and not runner.prompt_token_owners
    assert not engine.rng.generators


def test_small_kv_capacity_reports_reduced_coverage():
    from vllm_rlt.worker.warmup import prepare_runner

    engine = make_engine(blocks=8)
    result = prepare_runner(engine.model_runner, engine.scheduler.config)
    assert result["decode_batches"] == (1,)
    assert result["coverage_gaps"]
    assert engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("blocks,staged", [(64, False), (64, True), (8, False), (4, False)])
def test_warmup_reports_completed_decode_batch_history_pairs(blocks, staged, monkeypatch):
    from vllm_rlt.request import Operation
    from vllm_rlt.worker.warmup import prepare_runner

    engine = make_engine(blocks=blocks, staged=staged)
    runner = engine.model_runner
    submit = runner.submit
    observed = set()

    def observe(*args, batch):
        ticket = submit(*args, batch=batch)
        if batch.operation == Operation.RECURRENT and batch.emit_gate:
            observed.add((len(batch.spans), batch.spans[0].token_start))
        return ticket

    monkeypatch.setattr(runner, "submit", observe)
    report = prepare_runner(runner, engine.scheduler.config)
    assert {
        (workload["batch_size"], workload["history_tokens"])
        for workload in report["decode_workloads"]
    } == observed
    assert any("batch/history combinations" in gap for gap in report["coverage_gaps"])
    assert not engine.cache_manager._allocations and not engine.rng.generators
    engine.close()


@pytest.mark.parametrize("mode", ["block", "pipeline"])
@pytest.mark.parametrize(
    "device,asynchronous",
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=pytest.mark.gpu),
        pytest.param("cuda", True, marks=pytest.mark.gpu),
    ],
)
def test_warmup_reports_unmeasured_random_block_speculation(
    mode, device, asynchronous, monkeypatch
):
    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.worker.warmup import prepare_runner

    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config(head_dim=64)).to(device),
        attention_backend="triton" if device == "cuda" else "torch",
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
        speculative_config=SpeculativeConfig(2, mode=mode),
        execution_config=ExecutionConfig(
            cuda_graphs=device == "cuda", async_scheduling=asynchronous
        ),
        _defer_runtime_preparation=True,
    )
    speculation = engine.speculative_runner
    submit = speculation.submit
    temperatures = []

    def observe(batch):
        temperatures.extend(span.request.sampling_params.temperature for span in batch.requests)
        return submit(batch)

    monkeypatch.setattr(speculation, "submit", observe)
    report = prepare_runner(engine.model_runner, engine.scheduler.config, speculation)
    assert temperatures and all(temperature == 0 for temperature in temperatures)
    assert any("random block speculation" in gap for gap in report["coverage_gaps"]) == (
        mode == "block"
    )
    assert not engine.cache_manager._allocations and not engine.rng.generators

    if mode == "block":
        engine.add_request(
            "sampled",
            [1, 2, 3],
            SamplingParams(max_tokens=6, ignore_eos=True, seed=42, temperature=0.8),
        )
        while engine.has_unfinished_requests():
            engine.step()
        assert any(temperature > 0 for temperature in temperatures)
    engine.close()


def test_warmup_populates_history_across_sliding_window(monkeypatch):
    from tests.test_dory import tiny_dory_config
    from vllm_rlt.models import DoryForCausalLM
    from vllm_rlt.worker.warmup import prepare_runner

    engine = StageEngine(
        DoryForCausalLM(tiny_dory_config(sliding_window=8)),
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3,
            max_num_batched_tokens=8,
            prefill_chunk_size=4,
        ),
    )
    positions = []
    submit = engine.model_runner.submit

    def observe(*args, batch):
        positions.extend(span.token_start for span in batch.spans)
        return submit(*args, batch=batch)

    monkeypatch.setattr(engine.model_runner, "submit", observe)
    result = prepare_runner(engine.model_runner, engine.scheduler.config)
    assert result["history_tokens"] == 9
    assert max(positions) >= 9
    assert not engine.cache_manager._allocations and not engine.rng.generators


def test_warmup_reports_capacity_below_sliding_window():
    from tests.test_dory import tiny_dory_config
    from vllm_rlt.models import DoryForCausalLM
    from vllm_rlt.worker.warmup import prepare_runner

    engine = StageEngine(
        DoryForCausalLM(tiny_dory_config(sliding_window=64)),
        cache_config=CacheConfig(16, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=4
        ),
    )
    report = prepare_runner(engine.model_runner, engine.scheduler.config)
    assert report["history_tokens"] > engine.cache_manager.block_size
    assert any("limits prefill/history coverage" in gap for gap in report["coverage_gaps"])
    assert not engine.cache_manager._allocations


def test_warmup_exercises_stochastic_sampling_without_rng_leak(monkeypatch):
    from vllm_rlt.worker.warmup import prepare_runner

    engine = make_engine()
    sample = engine.model_runner.sampler.sample
    params_seen = []
    state = torch.get_rng_state().clone()

    def observe(logits, params, generator):
        params_seen.append(params)
        return sample(logits, params, generator)

    monkeypatch.setattr(engine.model_runner.sampler, "sample", observe)
    prepare_runner(engine.model_runner, engine.scheduler.config)
    assert any(params.temperature > 0 for params in params_seen)
    assert any(params.top_k == engine.model.config.vocab_size for params in params_seen)
    assert not engine.rng.generators
    torch.testing.assert_close(torch.get_rng_state(), state)


@pytest.mark.parametrize("staged", [False, True])
def test_warmup_uses_serving_core_loop_boundary(staged, monkeypatch):
    from vllm_rlt.request import Operation
    from vllm_rlt.worker.warmup import prepare_runner

    engine = make_engine(staged=staged)
    loops = []
    submit = engine.model_runner.submit

    def observe(*args, batch):
        if batch.operation == Operation.RECURRENT:
            loops.append(batch.loop_count)
        return submit(*args, batch=batch)

    monkeypatch.setattr(engine.model_runner, "submit", observe)
    prepare_runner(engine.model_runner, engine.scheduler.config)
    assert loops and all(count == 1 for count in loops)


def test_warmup_covers_decode_batch_above_graph_limit(monkeypatch):
    from dataclasses import replace

    from vllm_rlt.request import Operation
    from vllm_rlt.worker.warmup import prepare_runner

    engine = make_engine()
    runner = engine.model_runner
    runner.execution_config = replace(runner.execution_config, cuda_graph_max_batch_size=1)
    submit = runner.submit
    rows = []

    def observe(*args, batch):
        if batch.operation == Operation.RECURRENT:
            rows.append(len(batch.spans))
        return submit(*args, batch=batch)

    monkeypatch.setattr(runner, "submit", observe)
    report = prepare_runner(runner, engine.scheduler.config)
    assert max(rows) == 3
    assert report["decode_batches"] == (1, 3)


@pytest.mark.gpu
@pytest.mark.parametrize("staged", [False, True])
def test_async_startup_submits_independent_streams_before_collect(staged, monkeypatch):
    from vllm_rlt.request import Operation
    from vllm_rlt.worker.model_runner import Submission

    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16),
        attention_backend="triton",
        cache_config=CacheConfig(128, 16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3,
            max_num_batched_tokens=8,
            prefill_chunk_size=4,
            wavefront_chunked_prefill=staged,
        ),
        execution_config=ExecutionConfig(async_scheduling=True, cuda_graphs=True),
        exit_config=ExitConfig("ouro_delayed"),
        _defer_runtime_preparation=True,
    )
    pending = []
    pairs = []
    decode_pairs = []
    decode_workloads = set()
    submit, collect = engine.model_runner.submit, Submission.collect

    def submitted(*args, batch):
        ids = {span.request.request_id for span in batch.spans}
        ticket = submit(*args, batch=batch)
        if batch.operation == Operation.RECURRENT and batch.emit_gate:
            decode_workloads.add((len(batch.spans), batch.spans[0].token_start))
        if batch.operation == Operation.CODA:
            pairs.extend(
                (other, ticket)
                for other, other_ids, loops, _ in pending
                if loops and ids.isdisjoint(other_ids)
            )
            decode_pairs.extend(
                (other, ticket)
                for other, other_ids, _, operation in pending
                if operation == Operation.RECURRENT and ids.isdisjoint(other_ids)
            )
        if batch.loop_count:
            pairs.extend(
                (other, ticket)
                for other, other_ids, loops, _ in pending
                if not loops and ids.isdisjoint(other_ids)
            )
        pending.append((ticket, ids, batch.loop_count, batch.operation))
        return ticket

    def collected(ticket):
        pending[:] = [item for item in pending if item[0] is not ticket]
        return collect(ticket)

    monkeypatch.setattr(engine.model_runner, "submit", submitted)
    monkeypatch.setattr(Submission, "collect", collected)
    try:
        report = engine.prepare_runtime()
        assert pairs and decode_pairs
        assert {
            (workload["batch_size"], workload["history_tokens"])
            for workload in report["decode_workloads"]
        } == decode_workloads
        assert report["concurrent_batches"] == (1, 2)
        assert "concurrent" in [phase["name"] for phase in report["phases"]]
        assert not engine.cache_manager._allocations and not engine.rng.generators
        assert not engine.model_runner.events
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("limit", ["requests", "tokens", "kv", "single_stream"])
def test_startup_reports_when_independent_overlap_is_unavailable(limit):
    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16),
        attention_backend="triton",
        cache_config=CacheConfig(4 if limit == "kv" else 128, 16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1 if limit == "requests" else 3,
            max_num_batched_tokens=1 if limit == "tokens" else 8,
            prefill_chunk_size=4,
        ),
        execution_config=ExecutionConfig(
            async_scheduling=True, multi_stream=limit != "single_stream"
        ),
        exit_config=ExitConfig("ouro_delayed"),
    )
    try:
        report = engine.startup_report
        assert report["concurrent_batches"] == ()
        if limit != "single_stream":
            assert any(
                "prevent independent stream overlap" in gap for gap in report["coverage_gaps"]
            )
        assert not engine.cache_manager._allocations and not engine.rng.generators
    finally:
        engine.close()


@pytest.mark.gpu
def test_concurrent_startup_failure_drains_before_releasing_requests(monkeypatch):
    from vllm_rlt.request import Operation
    from vllm_rlt.worker.model_runner import Submission

    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16),
        attention_backend="triton",
        cache_config=CacheConfig(128, 16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(async_scheduling=True),
        exit_config=ExitConfig("ouro_delayed"),
        _defer_runtime_preparation=True,
    )
    runner = engine.model_runner
    submit, collect = runner.submit, Submission.collect
    pending_core = []

    def submitted(*args, batch):
        ticket = submit(*args, batch=batch)
        if batch.operation == Operation.RECURRENT:
            pending_core.append(ticket)
        if pending_core and batch.operation == Operation.CODA:
            raise RuntimeError("concurrent preparation failed")
        return ticket

    def collected(ticket):
        pending_core[:] = [item for item in pending_core if item is not ticket]
        return collect(ticket)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(runner, "submit", submitted)
            patch.setattr(Submission, "collect", collected)
            with pytest.raises(RuntimeError, match="concurrent preparation failed"):
                engine.prepare_runtime()
        assert engine.startup_report is None
        assert not engine.cache_manager._allocations and not engine.rng.generators
        assert not runner.events and not runner.state_slots
        assert engine.prepare_runtime()["status"] == "prepared"
    finally:
        engine.close()


@pytest.mark.parametrize("draft_loops", [1, 2])
@pytest.mark.parametrize(
    "device,asynchronous",
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=pytest.mark.gpu),
        pytest.param("cuda", True, marks=pytest.mark.gpu),
    ],
)
def test_pipeline_warmup_reaches_target_verification(draft_loops, device, asynchronous):
    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.worker.warmup import prepare_runner

    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config(head_dim=64)).to(device),
        attention_backend="triton" if device == "cuda" else "torch",
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
        speculative_config=SpeculativeConfig(2, draft_loops=draft_loops, mode="pipeline"),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
        _defer_runtime_preparation=True,
    )
    verified = []
    try:
        prepare_runner(
            engine.model_runner,
            engine.scheduler.config,
            engine.speculative_runner,
            phase=lambda name: verified.append(engine.speculative_runner.stats.verified_rows),
        )
        assert verified and all(count > 0 for count in verified)
        states = "states" if asynchronous else "pipeline_states"
        assert not getattr(engine.speculative_runner, states)
        assert engine.speculative_runner.stats.verified_rows == 0
        assert not engine.cache_manager._allocations
    finally:
        engine.close()


@pytest.mark.parametrize("kind", ["dory", "huginn"])
def test_boundary_warmup_keeps_rng_and_valid_prefixes(kind):
    from tests.test_dory import tiny_dory_config
    from tests.test_huginn import tiny_huginn_config
    from vllm_rlt.models import DoryForCausalLM, HuginnForCausalLM
    from vllm_rlt.worker.warmup import prepare_runner

    model = (
        DoryForCausalLM(tiny_dory_config())
        if kind == "dory"
        else HuginnForCausalLM(tiny_huginn_config())
    )
    engine = StageEngine(
        model,
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
    )
    state = torch.get_rng_state().clone()
    prepare_runner(engine.model_runner, engine.scheduler.config)
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert not engine.rng.generators and not engine.cache_manager._allocations


@pytest.mark.gpu
def test_startup_keeps_direct_gpu_preparation_usable():
    from vllm_rlt.request import ExecutionBatch, Operation, Request, StateSpan

    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda"),
        attention_backend="triton",
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(
            mode="mixed", max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=2
        ),
    )
    runner = engine.model_runner
    request = Request("direct", [1, 2], SamplingParams())
    assert engine.cache_manager.allocate("direct", 2)
    try:
        batch = ExecutionBatch(
            Operation.RECURRENT, (StateSpan(request, 0, 2, 0, chunk_start=0, retain_chunk=True),)
        )
        runner.prepare_on_gpu(runner.prepare_on_cpu(batch))
    finally:
        runner.synchronize()
        runner.release("direct")
        engine.cache_manager.free("direct")
        engine.close()


def test_cpu_preparation_is_idempotent_and_does_not_execute(monkeypatch):
    engine = make_engine()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU startup must not run a synthetic workload")

    monkeypatch.setattr(engine.model_runner, "submit", forbidden)
    report = engine.prepare_runtime()
    assert report is engine.prepare_runtime()
    assert report["status"] == "cpu"
    assert report["capacity_source"] == "blocks"
    assert not engine.scheduler.requests


def test_worker_prepares_after_connector_and_before_ready(monkeypatch):
    from types import SimpleNamespace

    from vllm_rlt.engine.stage_config import StageConfig
    from vllm_rlt.engine.worker import StageWorker
    from vllm_rlt.models import ModelConfig

    events = []
    prepare = StageEngine.prepare_runtime

    def prepared(engine):
        assert engine.connector is connector
        events.append("prepare")
        return prepare(engine)

    def factory(engine, **kwargs):
        assert engine.startup_report is None
        events.append("connector")
        return connector

    connector = SimpleNamespace(startup_info=lambda: {})
    channel = SimpleNamespace(send=lambda message: events.append(message["kind"]))
    monkeypatch.setattr(StageEngine, "prepare_runtime", prepared)
    StageWorker(
        StageConfig(),
        "cpu",
        channel,
        tiny_ouro_config(),
        dict(
            seed=123,
            dtype="float32",
            revision=None,
            model_config=ModelConfig(tiny_ouro_config()),
            engine={},
        ),
        connector_factory=factory,
    )
    assert events == ["connector", "prepare", "ready"]


@pytest.mark.parametrize("failure", ["connector", "prepare"])
def test_worker_startup_failure_closes_resources_without_ready(monkeypatch, failure):
    from types import SimpleNamespace

    from vllm_rlt.engine.stage_config import StageConfig
    from vllm_rlt.engine.worker import StageWorker
    from vllm_rlt.models import ModelConfig

    events = []
    close = StageEngine.close

    def closed(engine):
        close(engine)
        events.append("engine closed")

    def prepared(engine):
        raise RuntimeError("prepare failed")

    def factory(engine, **kwargs):
        if failure == "connector":
            raise RuntimeError("connector failed")
        return SimpleNamespace(close=lambda: events.append("connector closed"))

    monkeypatch.setattr(StageEngine, "close", closed)
    monkeypatch.setattr(StageEngine, "prepare_runtime", prepared)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        StageWorker(
            StageConfig(),
            "cpu",
            SimpleNamespace(send=lambda message: events.append(message["kind"])),
            tiny_ouro_config(),
            dict(
                seed=123,
                dtype="float32",
                revision=None,
                model_config=ModelConfig(tiny_ouro_config()),
                engine={},
            ),
            connector_factory=factory,
        )
    assert events == (
        ["engine closed"] if failure == "connector" else ["engine closed", "connector closed"]
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_kind,asynchronous,speculative,staged",
    [
        ("ouro", False, False, False),
        ("ouro", True, False, False),
        ("ouro", True, False, True),
        ("ouro", False, True, False),
        ("ouro", True, True, False),
        ("dory", False, False, False),
        ("dory", True, False, False),
        ("huginn", False, False, False),
        ("nanbeige", False, False, False),
        ("nanbeige", True, False, False),
        ("nanbeige", True, False, True),
    ],
)
@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
def test_cuda_startup_replay_and_generation_match_unprepared(
    model_kind, asynchronous, speculative, staged, temperature, kv_cache_dtype
):
    from tests.helpers import tiny_nanbeige_config
    from tests.test_dory import tiny_dory_config
    from tests.test_huginn import tiny_huginn_config
    from vllm_rlt import ExitConfig, SpeculativeConfig
    from vllm_rlt.models import DoryForCausalLM, HuginnForCausalLM, NanbeigeForCausalLM

    torch.manual_seed(123)
    if model_kind == "dory":
        model = DoryForCausalLM(tiny_dory_config(head_dim=64, d_model=256))
    elif model_kind == "huginn":
        model = HuginnForCausalLM(tiny_huginn_config(n_embd=256, intermediate_size=512))
    elif model_kind == "nanbeige":
        model = NanbeigeForCausalLM(tiny_nanbeige_config(head_dim=64))
    else:
        model = OuroForCausalLM(tiny_ouro_config(head_dim=64))
    model = model.to("cuda", torch.bfloat16)
    results = []
    for prepared in (False, True):
        rng = torch.cuda.get_rng_state().clone()
        engine = StageEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(
                128,
                16,
                enable_prefix_caching=model_kind == "ouro",
                incremental_allocation=True,
                kv_cache_dtype=kv_cache_dtype,
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3,
                max_num_batched_tokens=8,
                prefill_chunk_size=4,
                wavefront_chunked_prefill=staged,
            ),
            execution_config=ExecutionConfig(
                cuda_graphs=True,
                static_buffers=True,
                async_scheduling=asynchronous,
            ),
            exit_config=ExitConfig("ouro" if speculative else "ouro_delayed"),
            speculative_config=SpeculativeConfig(2) if speculative else None,
            _defer_runtime_preparation=not prepared,
        )
        try:
            if prepared:
                report = engine.startup_report
                assert report["status"] == "prepared"
                cache = engine.cache_manager
                assert report["kv_bytes"] == cache.num_blocks * cache.bytes_per_block
                assert [p["name"] for p in report["phases"]] == [
                    "initialized",
                    "captured",
                    "replayed",
                    *(["concurrent_prepared", "concurrent"] if asynchronous else []),
                    "cleaned",
                ]
                assert all(p["allocated_bytes"] > 0 for p in report["phases"])
                assert engine.cache_manager.num_used_blocks == 0
                assert engine.cache_manager.prefix_queries == 0
                assert not engine.rng.generators
                assert not engine.model_runner.state_slots
                from vllm_rlt.worker.warmup import graph_owners

                assert all(
                    owner.captures > 0
                    for owner in graph_owners(engine.model_runner, engine.speculative_runner)
                )
                coverage = report["graph_coverage"]
                assert "runner.graphs" in coverage
                assert all(entry["keys"] and entry["replays"] > 0 for entry in coverage.values())
                assert all(entry["remaining_entries"] >= 0 for entry in coverage.values())
                torch.testing.assert_close(torch.cuda.get_rng_state(), rng)
            rounds = []
            for _ in range(2):
                for i in range(3):
                    engine.add_request(
                        str(i),
                        [2 + i, 3, 4] * (i + 1),
                        SamplingParams(
                            max_tokens=5,
                            ignore_eos=True,
                            seed=42 + i,
                            temperature=temperature,
                            top_k=7,
                            top_p=0.8,
                        ),
                    )
                outputs = {}
                while engine.has_unfinished_requests():
                    for output in engine.step():
                        if output.finished:
                            outputs[output.request_id] = (output.token_ids, output.exit_depths)
                assert len(outputs) == 3
                rounds.append(outputs)
            results.append(rounds)
        finally:
            engine.close()
    assert results[0] == results[1]


@pytest.mark.gpu
def test_cuda_failed_preparation_clears_partial_graphs_and_can_retry(monkeypatch):
    from vllm_rlt.worker.warmup import graph_owners

    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16),
        attention_backend="triton",
        cache_config=CacheConfig(64, 16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(cuda_graphs=True),
        _defer_runtime_preparation=True,
    )
    runner = engine.model_runner
    execute = runner._execute

    def failed(*args):
        output = execute(*args)
        if runner.graphs.captures:
            raise RuntimeError("failure after capture")
        return output

    try:
        with monkeypatch.context() as patch:
            patch.setattr(runner, "_execute", failed)
            with pytest.raises(RuntimeError, match="failure after capture"):
                engine.prepare_runtime()
        assert engine.startup_report is None
        assert not engine.cache_manager._allocations and not runner.state_slots
        assert not engine.rng.generators
        assert all(not owner.entries for owner in graph_owners(runner))
        assert engine.prepare_runtime()["status"] == "prepared"
        assert engine.prepare_runtime() is engine.startup_report
    finally:
        engine.close()


@pytest.mark.gpu
def test_startup_captures_with_graph_row_limit_below_scheduler_limit():
    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16),
        attention_backend="triton",
        cache_config=CacheConfig(64, 16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=4, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(
            cuda_graphs=True, cuda_graph_max_batch_size=2, cuda_graph_max_graphs=1
        ),
    )
    try:
        coverage = engine.startup_report["graph_coverage"]["runner.graphs"]
        assert coverage["keys"] and coverage["replays"] > 0
        assert all(key[0] <= 2 for key in coverage["keys"])
    finally:
        engine.close()
