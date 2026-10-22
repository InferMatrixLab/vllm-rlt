"""End-to-end checks of cache semantics, delayed exits and scheduling progress."""

from dataclasses import replace

import pytest
import torch

from tests.helpers import LocalLLM as LLM
from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.execution import WorkKey
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import budget_blocks
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestStatus
from vllm_rlt.worker.model_runner import Submission


def model():
    torch.manual_seed(123)
    return OuroForCausalLM(tiny_ouro_config())


def drain(engine):
    outputs = {}
    for _ in range(200):
        if not engine.has_unfinished_requests():
            assert engine.cache_manager.num_used_blocks == 0
            return outputs
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
    pytest.fail("engine failed to finish")


@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("mode", ["refill", "no_refill"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_padded_chunked_execution_matches_serial(layout, mode, asynchronous):
    prompts = [[2, 3, 4, 5, 6], [7], [8, 9]]
    params = [
        SamplingParams(max_tokens=4, exit_threshold=q, ignore_eos=True) for q in [0.0, 1.0, 0.5]
    ]
    base = model()
    serial = [
        LLM(
            base,
            cache_config=CacheConfig(64, 2, layout),
            exit_config=ExitConfig("random_lookahead", 31),
        ).generate([p], q)[0]
        for p, q in zip(prompts, params)
    ]
    llm = LLM(
        base,
        cache_config=CacheConfig(64, 2, layout),
        exit_config=ExitConfig("random_lookahead", 31),
        execution_config=ExecutionConfig(
            async_scheduling=asynchronous, static_buffers=True, pad_to_power_of_two=True
        ),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=3, prefill_chunk_size=1, mode=mode),
    )
    actual = llm.generate(prompts, params)
    assert [(o.token_ids, o.exit_depths) for o in actual] == [
        (o.token_ids, o.exit_depths) for o in serial
    ]
    assert actual[0].exit_depths == [4, 2, 2, 2]
    assert actual[1].exit_depths == [4, 4, 4, 4]
    assert llm.engine.model_runner.state_slots == {}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_signal_applies_after_one_more_loop_and_final_hidden_is_used(asynchronous):
    base = model()
    expected = LLM(base).generate(
        [[3, 4]], SamplingParams(max_tokens=3, min_loops=3, max_loops=3, ignore_eos=True)
    )[0]
    engine = LLMEngine(
        base,
        exit_config=ExitConfig("random_lookahead"),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
    )

    class ScriptedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden):
            # Predict at round 2; round 3 says no, but cannot revoke the old decision.
            logit = 100.0 if self.calls % 3 == 1 else -100.0
            self.calls += 1
            return hidden.new_full((hidden.shape[0], 1), logit)

    engine.model_runner.lookahead_head = ScriptedHead()
    engine.add_request(
        "a", [3, 4], SamplingParams(max_tokens=3, exit_threshold=0.5, ignore_eos=True)
    )
    actual = drain(engine)["a"]
    assert actual.exit_depths == [4, 3, 3]
    assert actual.token_ids == expected.token_ids
    assert engine.model_runner.lookahead_head.calls == 6


@pytest.mark.parametrize(
    "min_loops,max_loops,q,expected",
    [
        (1, 1, 0.0, 1),
        (1, 4, 0.0, 2),
        (3, 4, 0.0, 3),
        (4, 4, 0.0, 4),
        (2, 3, 1.0, 3),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_delayed_bounds_and_threshold_one(min_loops, max_loops, q, expected, asynchronous):
    llm = LLM(
        model(),
        exit_config=ExitConfig("random_lookahead"),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
    )
    with torch.no_grad():
        llm.engine.model_runner.lookahead_head.weight.zero_()
        llm.engine.model_runner.lookahead_head.bias.fill_(100)
    result = llm.generate(
        [[2]],
        SamplingParams(
            max_tokens=3,
            min_loops=min_loops,
            max_loops=max_loops,
            exit_threshold=q,
            ignore_eos=True,
        ),
    )[0]
    assert result.exit_depths == [4, expected, expected]


def test_random_head_is_independent_reproducible_and_does_not_change_base_weights():
    base = model()
    state = torch.get_rng_state().clone()
    names = list(base.state_dict())
    first = LLMEngine(base, exit_config=ExitConfig("random_lookahead", 9))
    assert torch.equal(torch.get_rng_state(), state)
    second = LLMEngine(base, exit_config=ExitConfig("random_lookahead", 9))
    third = LLMEngine(base, exit_config=ExitConfig("random_lookahead", 10))
    assert list(base.state_dict()) == names
    assert torch.equal(
        first.model_runner.lookahead_head.weight, second.model_runner.lookahead_head.weight
    )
    assert not torch.equal(
        first.model_runner.lookahead_head.weight, third.model_runner.lookahead_head.weight
    )


def test_shared_overwrites_one_plane_and_reclaims_it_once():
    cache = KVCacheManager(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        num_blocks=4,
        block_size=2,
        max_loops=4,
        layout="shared",
    )
    assert cache.required_blocks(3) == 2
    cache.allocate("a", 3)
    assert cache.get_block_table("a", 0) == cache.get_block_table("a", 3)
    for pos, exit_depth in [(0, 1), (1, 3), (2, 0)]:
        for depth in range(exit_depth + 1):
            for layer in range(2):
                value = torch.full((1, 1, 4), float(pos * 10 + depth + layer))
                cache.write(layer, ["a"], [depth], [pos], value, value + 100)
        cache.finalize_token("a", pos, exit_depth)
    for depth in range(4):
        assert cache.read(0, "a", depth)[0][:, 0, 0].tolist() == [1, 13, 20]
    cache.free("a")
    cache.free("a")
    assert cache.num_free_blocks == 4
    assert len(set(cache._free_blocks)) == 4


def test_short_requests_bypass_then_protected_long_request_gets_memory():
    engine = LLMEngine(
        model(),
        cache_config=CacheConfig(24, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_admission_bypasses=1),
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    engine.add_request("running", [1, 2, 3, 4], params)  # 8 blocks
    engine.scheduler._admit()
    engine.add_request("long", [1] * 9, params)  # 20 blocks, cannot fit remaining 16
    engine.add_request("short", [2], params)  # 4 blocks
    engine.add_request("later", [3], params)
    engine.scheduler._admit()
    assert (
        engine.scheduler.pending_work(engine.scheduler.requests["short"]) == WorkKey.PROMPT_ATOMIC
    )
    assert list(engine.scheduler.waiting) == ["long", "later"]
    engine.abort_request("short")
    engine.scheduler._admit()
    assert engine.scheduler.requests["later"].status == RequestStatus.WAITING
    engine.abort_request("running")
    engine.scheduler._admit()
    assert engine.scheduler.pending_work(engine.scheduler.requests["long"]) == WorkKey.PROMPT_ATOMIC
    assert set(drain(engine)) == {"long", "later"}


def test_chunk_limit_allows_short_prompt_before_long_prefill_finishes():
    engine = LLMEngine(
        model(), scheduler_config=SchedulerConfig(max_num_batched_tokens=4, prefill_chunk_size=1)
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    engine.add_request("long", [1] * 8, params)
    engine.add_request("short", [2], params)
    engine.step()
    assert engine.last_schedule.num_tokens == 2
    assert engine.scheduler.requests["long"].num_prefilled_tokens == 1
    outputs = engine.step()
    assert outputs[0].request_id == "short" and outputs[0].finished
    drain(engine)


def test_async_submits_next_core_before_collecting_previous_signal(monkeypatch):
    engine = LLMEngine(
        model(),
        exit_config=ExitConfig("random_lookahead"),
        execution_config=ExecutionConfig(async_scheduling=True),
    )
    engine.add_request("a", [1], SamplingParams(max_tokens=2, exit_threshold=0, ignore_eos=True))
    original_submit, original_collect = engine.executor.submit, Submission.collect
    events = []

    def submit(batch):
        if WorkKey.DECODE_RECURRENT in batch.works:
            events.append("submit")
        return original_submit(batch)

    def collect(ticket):
        if WorkKey.DECODE_RECURRENT in ticket.batch.works:
            events.append("collect")
        return original_collect(ticket)

    monkeypatch.setattr(engine.executor, "submit", submit)
    monkeypatch.setattr(Submission, "collect", collect)
    drain(engine)
    assert events[:3] == ["submit", "submit", "collect"]


def test_cancel_pending_coda_and_reuse_request_id():
    engine = LLMEngine(
        model(),
        exit_config=ExitConfig("random_lookahead"),
        execution_config=ExecutionConfig(async_scheduling=True, static_buffers=True),
    )
    params = SamplingParams(max_tokens=2, exit_threshold=0, ignore_eos=True)
    engine.add_request("same", [1], params)
    engine.step()  # prefill
    engine.step()  # submit coda; completion still pending in the engine
    engine.abort_request("same")
    engine.add_request("same", [2], params)
    result = drain(engine)["same"]
    assert result.prompt_token_ids == [2]
    assert len(result.token_ids) == 2


def test_explicit_bytes_and_cpu_auto_sizing():
    base = model()
    default = LLMEngine(base)
    block_bytes = default.cache_manager.bytes_per_block
    engine = LLMEngine(base, cache_config=CacheConfig(kv_cache_memory_bytes=block_bytes * 9 + 1))
    assert engine.cache_manager.num_blocks == 9
    assert default.memory_plan["source"] == "cpu_default"
    with pytest.raises(ValueError, match="no memory"):
        budget_blocks(-1, block_bytes)
    with pytest.raises(ValueError, match="not both"):
        CacheConfig(num_blocks=1, kv_cache_memory_bytes=100)


def test_padding_does_not_touch_free_blocks():
    engine = LLMEngine(
        model(),
        cache_config=CacheConfig(32, 2),
        execution_config=ExecutionConfig(static_buffers=True, pad_to_power_of_two=True),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=3),
    )
    cache = engine.cache_manager
    cache.key_cache.fill_(17)
    cache.value_cache.fill_(19)
    engine.add_request("a", [1, 2, 3], SamplingParams(max_tokens=1))
    engine.step()
    assert engine.model_runner.last_effective_size == 3
    assert engine.model_runner.last_submitted_size == 4
    free = cache._free_blocks
    assert torch.all(cache.key_cache[free] == 17)
    assert torch.all(cache.value_cache[free] == 19)
    drain(engine)


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("exit_mode", ["random_lookahead", "ouro_delayed"])
def test_cuda_async_matches_synchronous_with_padding_and_sampling(layout, exit_mode):
    base = model().to(device="cuda", dtype=torch.bfloat16)
    prompts = [[2, 3, 4, 5], [6], [7, 8]]
    params = [
        SamplingParams(
            max_tokens=5, exit_threshold=q, ignore_eos=True, temperature=0.7, top_k=8, seed=3
        )
        for q in [0.0, 1.0, 0.5]
    ]
    common = dict(
        cache_config=CacheConfig(64, 2, layout),
        exit_config=ExitConfig(exit_mode, 17),
        attention_backend="triton",
        scheduler_config=SchedulerConfig(max_num_batched_tokens=3, prefill_chunk_size=2),
    )
    execution = ExecutionConfig(static_buffers=True, pad_to_power_of_two=True)
    expected = LLM(base, execution_config=execution, **common).generate(prompts, params)
    actual = LLM(base, execution_config=replace(execution, async_scheduling=True), **common)
    result = actual.generate(prompts, params)
    assert [(o.token_ids, o.exit_depths) for o in result] == [
        (o.token_ids, o.exit_depths) for o in expected
    ]
    assert actual.engine.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
def test_cuda_auto_memory_plan_and_abort_pending_work(layout):
    base = model().to(device="cuda", dtype=torch.bfloat16)
    rng = torch.cuda.get_rng_state().clone()
    engine = LLMEngine(
        base,
        cache_config=CacheConfig(block_size=2, layout=layout, memory_reserve_bytes=0),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=3),
        exit_config=ExitConfig("random_lookahead", 17),
        execution_config=ExecutionConfig(
            async_scheduling=True, static_buffers=True, pad_to_power_of_two=True
        ),
        attention_backend="triton",
    )
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    plan = engine.memory_plan
    assert plan["source"] == "cuda_profile"
    assert plan["profile_peak_bytes"] > 0
    assert engine.cache_manager.num_blocks <= plan["max_useful_blocks"]
    params = SamplingParams(max_tokens=4, exit_threshold=0, ignore_eos=True)
    engine.add_request("reuse", [2, 3, 4], params)
    engine.step()  # asynchronous prefill
    engine.step()  # coda submitted but not applied
    engine.abort_request("reuse")
    engine.add_request("reuse", [5], params)
    assert drain(engine)["reuse"].prompt_token_ids == [5]
    assert engine.model_runner.state_slots == {}


def test_pending_coda_does_not_block_other_recurrent_work(monkeypatch):
    engine = LLMEngine(
        model(),
        exit_config=ExitConfig("random_lookahead"),
        execution_config=ExecutionConfig(async_scheduling=True),
    )
    engine.add_request("fast", [1], SamplingParams(max_tokens=3, exit_threshold=0, ignore_eos=True))
    engine.add_request("slow", [2], SamplingParams(max_tokens=3, exit_threshold=1, ignore_eos=True))
    original = engine.executor.submit

    class HeldEvent:
        complete = False

        def query(self):
            return self.complete

        def synchronize(self):
            self.complete = True

    held = HeldEvent()

    def submit(batch):
        ticket = original(batch)
        if (
            WorkKey.DECODE_CODA in batch.works
            and len(batch.requests) == 1
            and batch.requests[0].request.request_id == "fast"
        ):
            ticket.event = held
        return ticket

    monkeypatch.setattr(engine.executor, "submit", submit)
    for _ in range(20):
        engine.step()
        if any(t.event is held for t in engine._pending_coda):
            break
    else:
        pytest.fail("fast coda was not submitted")
    engine.step()
    # The pending sample can already re-enter prelude on device. Its next
    # recurrent batch includes BOTH requests, without waiting for CPU delivery.
    assert engine.last_schedule.works == frozenset((WorkKey.DECODE_PRELUDE,))
    assert engine.last_schedule.requests[0].request.request_id == "fast"
    assert not held.complete
    engine.step()
    assert engine.last_schedule.works == frozenset((WorkKey.DECODE_RECURRENT,))
    assert {i.request.request_id for i in engine.last_schedule.requests} == {"fast", "slow"}
    assert not held.complete
    held.complete = True
    drain(engine)


def test_async_failed_submission_reclaims_all_affected_state(monkeypatch):
    engine = LLMEngine(
        model(),
        exit_config=ExitConfig("random_lookahead"),
        execution_config=ExecutionConfig(async_scheduling=True, static_buffers=True),
    )
    engine.add_request("a", [2], SamplingParams(max_tokens=2))
    original = engine.executor.submit

    def fail(batch):
        original(batch)
        raise RuntimeError("failed after a partial submission")

    monkeypatch.setattr(engine.executor, "submit", fail)
    with pytest.raises(RuntimeError, match="partial submission"):
        engine.step()
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.has_unfinished_requests()
    assert engine.model_runner.state_slots == {}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_trace_replay_skips_gate_and_obeys_variable_depths(asynchronous):
    base = model()
    engine = LLMEngine(
        base,
        exit_config=ExitConfig("trace", depths_by_request={"a": [4, 2, 4]}),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
    )

    def forbidden(*args):
        pytest.fail("replay must not calculate online gate")

    base.model.early_exit_gate.register_forward_hook(forbidden)
    engine.add_request("a", [2, 3], SamplingParams(max_tokens=3, ignore_eos=True))
    assert drain(engine)["a"].exit_depths == [4, 2, 4]


def test_trace_validation_before_admission():
    engine = LLMEngine(model(), exit_config=ExitConfig("trace", depths_by_request={"a": [4, 1]}))
    with pytest.raises(ValueError, match="trace"):
        engine.add_request("a", [2], SamplingParams(max_tokens=2))
    assert not engine.has_unfinished_requests()


def test_synthetic_replay_variants_preserve_outputs():
    from benchmarks.cdb_runtime import benchmark

    trace, results = benchmark()
    assert trace["depths_by_request"]["A"] == [4, 2, 2, 2, 2, 2]
    assert len(results) == 8
    assert all(r["seconds"] > 0 for r in results)
    with pytest.raises(ValueError, match="does not match"):
        benchmark(layout="shared", trace=trace)


@pytest.mark.gpu
def test_cuda_boundary_and_core_can_overlap(monkeypatch):
    # Artificially extend core duration to test dependency topology, NOT speedup.
    # Tiny real kernels can finish before Python has time to submit other work.
    if torch.version.hip:
        pytest.skip("CUDA sleep topology probe is NVIDIA-specific")
    base = model().to(device="cuda", dtype=torch.bfloat16)
    engine = LLMEngine(
        base,
        cache_config=CacheConfig(32, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2),
        exit_config=ExitConfig("random_lookahead"),
        execution_config=ExecutionConfig(async_scheduling=True, static_buffers=True),
        attention_backend="triton",
    )
    # Warm the exact batch shapes, kernels and allocator before testing topology.
    engine.add_request(
        "warm-fast", [1], SamplingParams(max_tokens=3, exit_threshold=0, ignore_eos=True)
    )
    engine.add_request(
        "warm-slow", [2], SamplingParams(max_tokens=3, exit_threshold=1, ignore_eos=True)
    )
    drain(engine)
    # The mixed warmup can batch the slow request with the next fast token.
    # Also warm a single-row recurrent batch at the first KV page: loading an
    # unseen Triton specialization can synchronize the device during the probe.
    engine.add_request("warm-single", [1], SamplingParams(max_tokens=2, ignore_eos=True))
    drain(engine)
    original = engine.model_runner._execute
    intervals = {}
    origin = torch.cuda.Event(enable_timing=True)
    origin.record()
    origin.synchronize()

    def execute(batch, prepared=None):
        label = None
        if WorkKey.DECODE_CODA in batch.works and len(batch.requests) == 1:
            request = batch.requests[0].request
            if request.request_id == "fast" and len(request.generated_token_ids) == 1:
                label = "coda"
        if WorkKey.DECODE_RECURRENT in batch.works:
            if any(
                i.request.request_id == "slow"
                and i.request.loops_done == 2
                and len(i.request.generated_token_ids) == 1
                for i in batch.requests
            ):
                label = "core"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # Preparation is outside _execute now. Keep both measured operations
        # alive long enough to test stream independence, regardless of which
        # one the host submits first; a tiny coda can finish before core launch.
        if WorkKey.DECODE_RECURRENT in batch.works or label == "coda":
            torch.cuda._sleep(30_000_000)
        result = original(batch, prepared)
        end.record()
        if label:
            intervals[label] = start, end
        return result

    monkeypatch.setattr(engine.model_runner, "_execute", execute)
    engine.add_request("fast", [1], SamplingParams(max_tokens=3, exit_threshold=0, ignore_eos=True))
    engine.add_request("slow", [2], SamplingParams(max_tokens=3, exit_threshold=1, ignore_eos=True))
    drain(engine)
    engine.model_runner.synchronize()
    first, last = zip(
        *[
            (origin.elapsed_time(start), origin.elapsed_time(end))
            for start, end in intervals.values()
        ]
    )
    assert set(intervals) == {"core", "coda"}
    assert max(first) < min(last), "independent boundary and core streams did not overlap"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "minimum,maximum,threshold,depth",
    [
        (1, 4, 0.5, 3),
        (2, 4, 0.5, 3),
        (3, 4, 0.5, 4),
        (2, 4, 0.0, 3),
        (1, 4, 0.0, 2),
        (2, 4, 1.0, 4),
        (1, 1, 0.5, 1),
    ],
)
def test_ouro_delayed_reuses_gate_and_accumulates_hazards(
    asynchronous, minimum, maximum, threshold, depth
):
    base = model()
    # Each round has hazard 0.3, below 0.5. Two rounds accumulate to 0.51.
    with torch.no_grad():
        base.model.early_exit_gate.weight.zero_()
        base.model.early_exit_gate.bias.fill_(torch.logit(torch.tensor(0.3)).item())
    params = SamplingParams(
        max_tokens=4,
        min_loops=minimum,
        max_loops=maximum,
        exit_threshold=threshold,
        ignore_eos=True,
    )
    expected = LLM(base).generate(
        [[2, 3]], replace(params, min_loops=depth, max_loops=depth, exit_threshold=1.0)
    )[0]
    llm = LLM(
        base,
        exit_config=ExitConfig("ouro_delayed"),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
    )
    assert llm.engine.model_runner.lookahead_head is None
    actual = llm.generate([[2, 3]], params)[0]
    assert actual.exit_depths == [4, depth, depth, depth]
    assert actual.token_ids == expected.token_ids


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("multi_stream", [False, True])
@pytest.mark.parametrize("exit_mode", ["random_lookahead", "ouro_delayed"])
def test_cuda_dynamic_async_respects_buffers_and_matches_sync(layout, multi_stream, exit_mode):
    base = model().to(device="cuda", dtype=torch.bfloat16)
    common = dict(
        cache_config=CacheConfig(64, 2, layout),
        exit_config=ExitConfig(exit_mode, 17),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=3, prefill_chunk_size=2),
        attention_backend="triton",
    )
    prompts = [[2, 3, 4], [5], [6, 7]]
    params = [
        SamplingParams(max_tokens=5, exit_threshold=q, ignore_eos=True) for q in (0.0, 0.5, 1.0)
    ]
    expected = LLM(base, **common).generate(prompts, params)
    execution = ExecutionConfig(async_scheduling=True, multi_stream=multi_stream)
    actual = LLM(base, execution_config=execution, **common)
    assert actual.engine.execution_config == execution
    assert actual.engine.model_runner.requestspaces == {}
    assert actual.engine.model_runner.states is actual.engine.model_runner.async_state.hidden
    result = actual.generate(prompts, params)
    assert [(o.token_ids, o.exit_depths) for o in result] == [
        (o.token_ids, o.exit_depths) for o in expected
    ]
    assert actual.engine.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
@pytest.mark.parametrize("free_mib", [128, 512])
def test_cuda_graph_reserve_reduces_budget_with_fixed_memory_inputs(monkeypatch, free_mib):
    from vllm_rlt.core.memory import plan_cache

    base = model().to(device="cuda", dtype=torch.bfloat16)
    mib = 1024**2
    # Exercise both driver-free and utilization-limited budgets. Keep the real
    # model probe, but freeze measurements so allocator history cannot affect
    # the comparison between graph-enabled and graph-disabled plans.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_mib * mib, 512 * mib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 64 * mib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 16 * mib)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 24 * mib)
    cache = CacheConfig(block_size=2, gpu_memory_utilization=0.5, memory_reserve_bytes=0)
    scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=3)
    execution = ExecutionConfig(cuda_graph_memory_reserve_bytes=32 * mib)

    _, eager = plan_cache(base, cache, scheduler, execution, "triton")
    _, graphed = plan_cache(base, cache, scheduler, replace(execution, cuda_graphs=True), "triton")

    assert eager["cuda_graph_reserve_bytes"] == 0
    assert graphed["cuda_graph_reserve_bytes"] == 32 * mib
    assert eager["kv_budget_bytes"] - graphed["kv_budget_bytes"] == 32 * mib


@pytest.mark.gpu
def test_cuda_memory_plan_reclaims_allocator_residue_on_recreation():
    import gc

    base = model().to(device="cuda", dtype=torch.bfloat16)
    options = dict(
        cache_config=CacheConfig(block_size=2, memory_reserve_bytes=0),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=3),
        attention_backend="triton",
    )
    first = LLMEngine(base, **options)
    budget = first.memory_plan["kv_budget_bytes"]
    blocks = first.cache_manager.num_blocks
    del first
    gc.collect()
    # Leave a large unused allocator segment, as a released KV pool would.
    residue = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    del residue
    torch.cuda.synchronize()
    assert torch.cuda.memory_reserved() - torch.cuda.memory_allocated() >= 64 * 1024 * 1024
    second = LLMEngine(base, **options)
    assert second.cache_manager.num_blocks == blocks
    assert abs(second.memory_plan["kv_budget_bytes"] - budget) < 16 * 1024 * 1024
    second.add_request("recreated", [1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
    assert len(drain(second)["recreated"].token_ids) == 2
