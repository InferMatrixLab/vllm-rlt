import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.execution import WorkKey
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestStatus


def model():
    torch.manual_seed(15)
    return OuroForCausalLM(tiny_ouro_config())


def finish(engine):
    result = {}
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            return result
        for out in engine.step():
            if out.finished:
                result[out.request_id] = out
    pytest.fail("engine did not drain")


def test_prefix_reuse_matches_cold_generation_and_retains_shared_pages():
    e = LLMEngine(
        model(),
        cache_config=CacheConfig(64, 2, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(prefill_chunk_size=2),
    )
    p = SamplingParams(max_tokens=4, exit_threshold=0, ignore_eos=True)
    e.add_request("first", [1, 2, 3, 4, 5], p)
    first = finish(e)["first"]
    assert e.cache_manager.lookup_prefix([1, 2, 3, 4, 5])
    e.add_request("second", [1, 2, 3, 4, 5], p)
    e.step()
    assert e.last_schedule.requests[0].token_start == 4
    second = finish(e)["second"]
    assert second.token_ids == first.token_ids
    assert second.exit_depths == first.exit_depths
    assert e.cache_manager.num_used_blocks == 0
    # A different continuation may reuse only complete full-depth prompt blocks.
    assert len(e.cache_manager.lookup_prefix([1, 2, 9, 4, 5])) == 1


def test_prefix_refcounts_and_eviction():
    c = KVCacheManager(1, 1, 4, 12, 2, 2, enable_prefix_caching=True)
    assert c.allocate("a", 5)
    c.mark_imported_prefix("a", 5)
    c.publish_prefix("a", [1, 2, 3, 4, 5], 5)
    hit = c.lookup_prefix([1, 2, 3, 4, 9])
    assert c.allocate("b", 5, prefix=hit)
    a = c.plane_block_tables("a")
    b = c.plane_block_tables("b")
    assert a[0][:2] == b[0][:2]
    c.free("a")
    assert c._refs[b[0][0]] == 2  # cache + live request
    c.free("b")
    assert c.num_free_blocks == 12
    assert c.allocate("all", 12)
    assert not c.lookup_prefix([1, 2, 3, 4, 5])
    c.free("all")
    assert c.num_free_blocks == 12


def test_incremental_growth_matches_reserved_generation():
    m = model()
    p = SamplingParams(max_tokens=9, exit_threshold=0, ignore_eos=True)
    outputs = []
    for incremental in [False, True]:
        e = LLMEngine(
            m,
            cache_config=CacheConfig(64, 2, incremental_allocation=incremental),
            scheduler_config=SchedulerConfig(prefill_chunk_size=2),
        )
        e.add_request("r", [1, 2, 3, 4, 5], p)
        e.step()
        assert e.cache_manager.num_used_blocks == (4 if incremental else 28)
        outputs.append(finish(e)["r"])
    assert outputs[0].token_ids == outputs[1].token_ids
    assert outputs[0].exit_depths == outputs[1].exit_depths


@pytest.mark.parametrize("stage", [WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT])
def test_incremental_split_prefill_reserves_prompt_growth(stage):
    prompt = list(range(1, 18))
    e = LLMEngine(
        model(),
        cache_config=CacheConfig(64, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=2,
            prefill_chunk_size=2,
            enable_preemption=True,
            wavefront_chunked_prefill=True,
        ),
        execution_config=ExecutionConfig(),
    )
    try:
        e.add_request("split", prompt, SamplingParams(max_tokens=1, ignore_eos=True))
        e.scheduler._admit()
        request = e.scheduler.requests["split"]
        assert e.scheduler.pending_work(request) is WorkKey.PROMPT_PRELUDE
        if stage is WorkKey.PROMPT_RECURRENT:
            for _ in range(8):
                if e.scheduler.pending_work(request) is WorkKey.PROMPT_RECURRENT:
                    break
                e.step()
            assert e.scheduler.pending_work(request) is WorkKey.PROMPT_RECURRENT

        allocated = e.cache_manager.allocated_blocks(request.request_id)
        expected = max(0, e.cache_manager.required_blocks(len(prompt)) - allocated)
        assert expected > 0
        assert e.scheduler._reserved_growth_blocks(reserve_outputs=False) == expected
    finally:
        e.close()


def test_lossless_preemption_preserves_looped_history():
    m = model()
    p = SamplingParams(max_tokens=7, temperature=0.8, seed=45, exit_threshold=0, ignore_eos=True)
    results = []
    for suspend in [False, True]:
        e = LLMEngine(
            m,
            cache_config=CacheConfig(64, 2),
            scheduler_config=SchedulerConfig(enable_preemption=True),
        )
        e.add_request("a", [1, 2, 3], p)
        while len(e.scheduler.requests["a"].generated_token_ids) < 3:
            e.step()
        if suspend:
            generator = e.rng.peek("a")
            assert generator is not None
            state = generator.get_state().clone()
            e.add_request("b", [5], SamplingParams(max_tokens=1))
            assert e.scheduler.preempt(e.scheduler.requests["b"])
            assert "a" in e.preemption.snapshots
            # Preemption suspends rather than terminates, so the runner keeps the
            # request's RNG slot: suspension is not a reset.
            assert e.rng.peek("a") is generator
            assert torch.equal(generator.get_state(), state)
        results.append(finish(e)["a"])
        if suspend:
            assert e.preemption.resumptions == 1
            # Sampling continued from the preserved state instead of reseeding,
            # and termination then dropped the RNG slot.
            assert not torch.equal(generator.get_state(), state)
            assert e.rng.peek("a") is None
    assert results[0].token_ids == results[1].token_ids
    assert results[0].exit_depths == results[1].exit_depths


@pytest.mark.gpu
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("use_uva", [False, True])
def test_fa4_prefill_uva_prefix_growth_and_bank_reuse(graph, use_uva):
    from vllm_rlt import ExecutionConfig, ExitConfig

    torch.manual_seed(21)
    cfg = tiny_ouro_config(head_dim=64)
    m = OuroForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16)
    params = SamplingParams(max_tokens=8, min_loops=1, ignore_eos=True)
    engines = []
    for enabled in [False, True]:
        engines.append(
            LLMEngine(
                m,
                attention_backend="flash_attn_4",
                cache_config=CacheConfig(
                    128, 16, enable_prefix_caching=enabled, incremental_allocation=enabled
                ),
                scheduler_config=SchedulerConfig(
                    max_num_seqs=3, max_num_batched_tokens=23, prefill_chunk_size=17
                ),
                exit_config=ExitConfig("trace", depths_by_request={"t": [4, 1, 2, 3, 4, 2, 1, 3]}),
                execution_config=ExecutionConfig(
                    async_scheduling=True,
                    prefill_uva=enabled,
                    use_uva=use_uva,
                    static_buffers=graph,
                    pad_to_power_of_two=graph,
                    cuda_graphs=graph,
                ),
            )
        )
    for trial in range(3):
        results = []
        for e in engines:
            for i in range(3):
                e.add_request(str(i), [1, 2, 3, 4] * (5 + i), params, trace_id="t")
            results.append(finish(e))
        assert results[0] == results[1]
        assert engines[1].cache_manager.num_used_blocks == 0
    assert engines[1].cache_manager.prefix_hits > 0


@pytest.mark.parametrize("preempt", [False, True])
def test_incremental_memory_pressure_makes_progress(preempt):
    e = LLMEngine(
        model(),
        cache_config=CacheConfig(24, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, prefill_chunk_size=2, enable_preemption=preempt
        ),
    )
    p = SamplingParams(max_tokens=8, exit_threshold=0, ignore_eos=True)
    for i in range(3):
        e.add_request(str(i), [1, 2], p)
    outputs = finish(e)
    assert len(outputs) == 3
    assert outputs["0"].token_ids == outputs["1"].token_ids == outputs["2"].token_ids
    assert e.cache_manager.num_free_blocks == 24
    assert not e.preemption.snapshots
    if preempt:
        assert e.preemption.preemptions > 0


def test_priority_preempts_at_safe_boundary():
    e = LLMEngine(
        model(),
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=1, policy="priority", enable_preemption=True),
    )
    e.add_request("low", [1, 2], SamplingParams(max_tokens=4, priority=10, ignore_eos=True))
    e.step()
    e.add_request("high", [3, 4], SamplingParams(max_tokens=1, priority=-10, ignore_eos=True))
    result = finish(e)
    assert list(result) == ["high", "low"]
    assert e.preemption.preemptions == 1


@pytest.mark.gpu
@pytest.mark.parametrize("use_uva", [False, True])
def test_prefill_uva_metadata_matches_reference_and_waits_for_consumer(use_uva):
    from vllm_rlt.worker.buffers import PackedMetadataBuffer

    c = KVCacheManager(
        2, 2, 64, 64, 16, 4, device="cuda", dtype=torch.bfloat16, backend="flash_attn_4"
    )
    assert c.allocate("a", 16) and c.allocate("b", 32)
    from vllm_rlt.worker.submission_queue import SubmissionQueue

    queue = SubmissionQueue(max_pending=1)
    queue.begin()
    buffer = PackedMetadataBuffer(c.device, use_uva=use_uva)
    ids = ["a"] * 3 + ["b"] * 4
    positions = [0, 1, 2, 16, 17, 18, 19]
    tokens = [1, 2, 3, 4, 5, 6, 7]
    depth_sets = [[depth] * len(ids) for depth in range(4)]
    buffer.store(c.metadata_builder.build(ids, depth_sets, positions, packed=True), tokens)
    device_tokens, metadata = buffer.upload()
    for depth in range(4):
        actual = metadata[depth]
        expected = c._prepare_batch(ids, [depth] * len(ids), positions, packed_prefill=True)
        for name in [
            "position_ids",
            "write_blocks",
            "write_offsets",
            "block_tables",
            "context_lengths",
            "cu_seqlens_q",
        ]:
            assert torch.equal(getattr(actual, name), getattr(expected, name)), name
        assert actual.max_seqlen_q == expected.max_seqlen_q
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    saved = torch.empty_like(device_tokens)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        saved.copy_(device_tokens)
        event = torch.cuda.Event()
        event.record(stream)
        queue.complete(event)
    queue.begin()
    buffer.store(c.metadata_builder.build(ids, depth_sets, positions, packed=True), [9] * 7)
    device_tokens, _ = buffer.upload()
    assert saved.tolist() == tokens
    assert device_tokens.tolist() == [9] * 7
    event = torch.cuda.Event()
    event.record()
    queue.complete(event)


@pytest.mark.gpu
def test_async_pressure_preemption_with_resident_state():
    from vllm_rlt import ExecutionConfig, ExitConfig

    torch.manual_seed(71)
    cfg = tiny_ouro_config(head_dim=64)
    m = OuroForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16)
    options = dict(
        attention_backend="triton",
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=3, prefill_chunk_size=2, enable_preemption=True
        ),
        exit_config=ExitConfig("trace", depths_by_request={"t": [4, 1, 2, 3, 4, 1, 2, 3]}),
        execution_config=ExecutionConfig(
            async_scheduling=True, static_buffers=True, pad_to_power_of_two=True, cuda_graphs=True
        ),
    )
    results = []
    for blocks in [128, 24]:
        e = LLMEngine(
            m, cache_config=CacheConfig(blocks, 2, incremental_allocation=True), **options
        )
        for i in range(3):
            e.add_request(
                str(i),
                [1, 2],
                SamplingParams(max_tokens=8, min_loops=1, ignore_eos=True),
                trace_id="t",
            )
        results.append(finish(e))
        assert not e.preemption.snapshots and e.cache_manager.num_used_blocks == 0
        if blocks == 24:
            assert e.preemption.preemptions > 0
    assert results[0] == results[1]


@pytest.mark.gpu
def test_async_preemption_restores_prelude_token_in_a_new_slot():
    # A preempted request releases its async slot. A later arrival takes it
    # (LIFO), so the victim resumes in another slot whose device token belongs
    # to a different request. Distinct prompts make a stale token visible.
    from vllm_rlt import ExecutionConfig, ExitConfig

    torch.manual_seed(72)
    m = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to(device="cuda", dtype=torch.bfloat16)
    results, victim_stages = [], []
    for blocks in [128, 24]:
        e = LLMEngine(
            m,
            attention_backend="triton",
            cache_config=CacheConfig(blocks, 2, incremental_allocation=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2,
                max_num_batched_tokens=2,
                prefill_chunk_size=2,
                enable_preemption=True,
            ),
            exit_config=ExitConfig("trace", depths_by_request={"t": [4, 1, 2, 3, 4, 1, 2, 3]}),
            execution_config=ExecutionConfig(async_scheduling=True),
        )
        select = e.scheduler._select_preemption_victim

        def record_victim(*args, **kwargs):
            victim = select(*args, **kwargs)
            if victim is not None:
                victim_stages.append(e.scheduler.pending_work(victim))
            return victim

        e.scheduler._select_preemption_victim = record_victim
        for i in range(4):
            e.add_request(
                str(i),
                [2 * i + 1, 2 * i + 2],
                SamplingParams(max_tokens=8, min_loops=1, ignore_eos=True),
                trace_id="t",
            )
        results.append({rid: out.token_ids for rid, out in finish(e).items()})
        assert not e.preemption.snapshots and e.cache_manager.num_used_blocks == 0
    assert WorkKey.DECODE_PRELUDE in victim_stages
    assert results[0] == results[1]


@pytest.mark.parametrize("incremental", [False, True])
def test_watermark_only_limits_incremental_admission(incremental):
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import Request

    cache = KVCacheManager(
        1,
        1,
        4,
        16,
        2,
        4,
        incremental_allocation=incremental,
        watermark_ratio=0.5,
    )
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8, prefill_chunk_size=4),
        cache,
    )
    for request_id in ("a", "b"):
        scheduler.add_request(Request(request_id, [1, 2, 3, 4], SamplingParams(max_tokens=1)))
    scheduler._admit()
    assert cache.watermark_blocks == (8 if incremental else 0)
    assert scheduler.pending_work(scheduler.requests["a"]) == WorkKey.PROMPT_ATOMIC
    # Full allocation admits both requests into the 16-block pool despite the
    # supplied ratio; incremental allocation retains the configured headroom.
    if incremental:
        assert scheduler.requests["b"].status == RequestStatus.WAITING
    else:
        assert scheduler.pending_work(scheduler.requests["b"]) == WorkKey.PROMPT_ATOMIC
    assert cache.num_free_blocks == (8 if incremental else 0)
