from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.core.loop import LoopOp


@pytest.fixture
def scheduler_factory():
    pytest.importorskip("vllm")
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup

    from vllm_rlt.core.async_scheduler import LoopAsyncScheduler
    from vllm_rlt.core.loop_scheduler import LoopScheduler

    def make(
        async_scheduling=False,
        *,
        prefix=False,
        ouro=False,
        speculate=False,
        pipeline=False,
        prefill_lookahead=False,
        shared_kv=False,
        long_prefill_token_threshold=0,
        prefill_window=None,
        policy="fcfs",
        split_prefill_coda=False,
        merge_decode_core=False,
        decode_graph_sizes=(),
        max_num_seqs=4,
        max_num_batched_tokens=16,
        additional_config=None,
    ):
        text = SimpleNamespace(
            model_type="inkling2_model", num_loop_passes=3, loop_config={"is_varloop": True},
            loop_share_cache=shared_kv,
        )
        if ouro:
            text = SimpleNamespace(model_type="ouro", total_ut_steps=4)
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                async_scheduling=async_scheduling,
                policy=policy,
                long_prefill_token_threshold=long_prefill_token_threshold,
            ),
            model_config=SimpleNamespace(hf_text_config=text, max_model_len=64),
            cache_config=SimpleNamespace(enable_prefix_caching=prefix),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                use_ubatching=False,
            ),
            kv_transfer_config=None,
            lora_config=None,
            speculative_config=None,
            compilation_config=SimpleNamespace(
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                decode_cudagraph_capture_request_sizes=decode_graph_sizes,
                prefill_cudagraph_capture_token_sizes=(4, 8, 16),
                max_cudagraph_capture_size=max(decode_graph_sizes, default=0),
                max_decode_cudagraph_capture_tokens=None,
            ),
            additional_config={
                **({"rlt_prefill_window": prefill_window} if prefill_window else {}),
                "rlt_self_speculation": speculate,
                "rlt_decode_pipeline": pipeline,
                "rlt_async_prefill_lookahead": prefill_lookahead,
                "rlt_merge_decode_core": merge_decode_core,
                "rlt_split_prefill_coda": split_prefill_coda,
                **(additional_config or {}),
            },
        )
        spec = AttentionSpec(block_size=4, num_kv_heads=1, head_size=64, dtype=torch.float16)
        group = KVCacheGroup.from_layers({"test": FullAttentionManager.layer_config(spec)})
        scheduler_cls = LoopAsyncScheduler if async_scheduling else LoopScheduler
        return scheduler_cls(config, KVCacheConfig(64, [], [group]), None, 4, 4)

    return make


def request(
    request_id,
    loops=None,
    *,
    prompt=None,
    block_hasher=None,
    threshold=None,
    max_tokens=3,
    stop_token_ids=None,
    priority=0,
):
    from vllm import SamplingParams

    from vllm_rlt.core.request import LoopRequest

    extra_args = {}
    if loops is not None:
        extra_args["rlt_target_loops"] = loops
    if threshold is not None:
        extra_args["rlt_exit_threshold"] = threshold
    return LoopRequest(
        request_id=request_id,
        priority=priority,
        prompt_token_ids=prompt or [1, 2, 3],
        sampling_params=SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            extra_args=extra_args or None,
        ),
        pooling_params=None,
        block_hasher=block_hasher,
    )


def complete(scheduler, plan, token=42):
    from vllm.v1.outputs import ModelRunnerOutput

    return scheduler.update_from_output(
        plan,
        ModelRunnerOutput(
            req_ids=[span.request_id for span in plan.spans],
            sampled_token_ids=[[token] if span.sample else [] for span in plan.spans],
        ),
    )


def test_request_depth_applies_after_prompt_and_kv_slots_survive_core_steps(scheduler_factory):
    scheduler = scheduler_factory()
    requested = request("a", loops=1)
    scheduler.add_request(requested)

    plan = scheduler.schedule()
    assert plan.op is LoopOp.PRELUDE and plan.spans[0].prefill
    assert plan.scheduled_new_reqs[0].req_id == "a"
    blocks = scheduler.kv.get_block_ids("a")
    complete(scheduler, plan)
    for depth in range(3):
        plan = scheduler.schedule()
        assert plan.op is LoopOp.CORE and plan.spans[0].depth == depth
        assert scheduler.kv.get_block_ids("a") == blocks
        assert not complete(scheduler, plan)
        assert requested.num_computed_tokens == 0

    coda = scheduler.schedule()
    assert coda.op is LoopOp.CODA and coda.spans[0].sample
    assert complete(scheduler, coda)[0].outputs[0].new_token_ids == [42]
    assert requested.num_computed_tokens == 3
    complete(scheduler, scheduler.schedule())
    plan = scheduler.schedule()
    assert plan.op is LoopOp.CORE and plan.spans[0].last_core
    assert not plan.spans[0].prefill


def test_a_core_batch_can_include_multiple_depths(scheduler_factory):
    scheduler = scheduler_factory()
    scheduler.add_request(request("a"))
    complete(scheduler, scheduler.schedule())
    complete(scheduler, scheduler.schedule())
    pending = scheduler.schedule()
    assert pending.op is LoopOp.CORE and pending.spans[0].depth == 1
    scheduler.add_request(request("b"))
    assert (plan := scheduler.schedule()).op is LoopOp.PRELUDE
    complete(scheduler, pending)
    complete(scheduler, plan)
    plan = scheduler.schedule()
    assert plan.op is LoopOp.CORE
    assert {(span.request_id, span.depth) for span in plan.spans} == {("a", 2), ("b", 0)}


def test_long_prefill_threshold_leaves_batch_capacity_for_other_requests(scheduler_factory):
    scheduler = scheduler_factory(long_prefill_token_threshold=6)
    scheduler.add_request(request("a", prompt=[1] * 20))
    scheduler.add_request(request("b", prompt=[2] * 20))
    scheduler.add_request(request("c", prompt=[3] * 20))

    plan = scheduler.schedule()

    assert plan.op is LoopOp.PRELUDE
    assert plan.num_scheduled_tokens == {"a": 6, "b": 6, "c": 4}
    assert all(span.prefill for span in plan.spans)


@pytest.mark.parametrize("async_scheduling", (False, True))
def test_prelude_revisits_each_request_and_numbers_chunks_from_zero(
    scheduler_factory, async_scheduling,
):
    scheduler = scheduler_factory(async_scheduling, long_prefill_token_threshold=4, max_num_seqs=2)
    for req_id in ("a", "b"):
        scheduler.add_request(request(req_id, prompt=[1] * 20))

    plan = scheduler.schedule()

    assert plan.op is LoopOp.PRELUDE
    assert [(s.request_id, s.chunk_id, s.token_start, s.token_count) for s in plan.spans] == [
        ("a", 0, 0, 4), ("b", 0, 0, 4), ("a", 1, 4, 4), ("b", 1, 4, 4),
    ]
    assert plan.num_scheduled_tokens == {"a": 8, "b": 8}


def test_prelude_uses_other_candidates_after_a_kv_allocation_fails(scheduler_factory, monkeypatch):
    scheduler = scheduler_factory()
    scheduler.add_request(request("blocked", prompt=[1] * 16))
    scheduler.add_request(request("available", prompt=[2] * 4))
    allocate = scheduler.kv.allocate_slots

    def allocate_available(req, *args, **kwargs):
        if req.request_id == "blocked":
            return None
        return allocate(req, *args, **kwargs)

    monkeypatch.setattr(scheduler.kv, "allocate_slots", allocate_available)
    plan = scheduler.schedule()
    assert plan.op is LoopOp.PRELUDE and plan.num_scheduled_tokens == {"available": 4}
    assert scheduler._states["blocked"].entry_skips == 0


def test_prefill_lookahead_cannot_preempt_to_extend_a_batch(scheduler_factory, monkeypatch):
    scheduler = scheduler_factory(long_prefill_token_threshold=4)
    scheduler.add_request(request("a", prompt=[1] * 20))
    allocate = scheduler.kv.allocate_slots

    def fit_first_chunk(req, reserve, **kwargs):
        return allocate(req, reserve, **kwargs) if reserve <= 4 else None

    monkeypatch.setattr(scheduler.kv, "allocate_slots", fit_first_chunk)
    monkeypatch.setattr(scheduler, "_preempt_for", lambda *_: pytest.fail("lookahead preempted"))
    plan = scheduler.schedule()
    assert plan.num_scheduled_tokens == {"a": 4}
    complete(scheduler, plan)
    assert scheduler.schedule().op is LoopOp.CORE


@pytest.mark.parametrize("head_rows", (12, 16))
def test_core_fills_a_real_gap_and_preserves_an_existing_full_batch(scheduler_factory, head_rows):
    scheduler = scheduler_factory()
    for req_id, rows, depth in (("shallow", head_rows, 0), ("deep", 4, 2)):
        req = request(req_id, prompt=[1] * rows)
        scheduler.add_request(req)
        work = scheduler._reserve_span(scheduler._states[req_id], rows, [])
        work.next_op = LoopOp.CORE
        work.next_depth, work.completed_depth = depth, depth - 1

    plan = scheduler.schedule()

    assert plan.op is LoopOp.CORE
    assert plan.total_num_scheduled_tokens == 16
    expected = [("shallow", 0)] if head_rows == 16 else [("shallow", 0), ("deep", 2)]
    assert [(span.request_id, span.depth) for span in plan.spans] == expected


@pytest.mark.parametrize("async_scheduling", (False, True))
def test_prefill_window_charges_inflight_and_pauses_admission_age(
    scheduler_factory, async_scheduling,
):
    scheduler = scheduler_factory(async_scheduling, prefill_window=48)
    scheduler.add_request(request("first", prompt=[1] * 16))
    first = scheduler.schedule()
    assert first.op is LoopOp.PRELUDE
    assert scheduler._prefill_capacity() == 0
    scheduler.add_request(request("waiting", prompt=[2] * 16))
    complete(scheduler, first)
    for depth in range(3):
        plan = scheduler.schedule()
        assert plan.op is LoopOp.CORE and plan.spans[0].depth == depth
        assert scheduler._states["waiting"].entry_skips == 0
        complete(scheduler, plan)
    assert scheduler._prefill_capacity() == 16
    coda = scheduler.schedule()
    assert coda.op is LoopOp.CODA and coda.spans[0].sample
    complete(scheduler, coda)
    assert any(span.request_id == "waiting" for span in scheduler.schedule().spans)


@pytest.mark.parametrize("policy,expected", (("fcfs", "older"), ("priority", "higher")))
def test_request_policy_orders_candidates_in_the_same_stage(scheduler_factory, policy, expected):
    scheduler = scheduler_factory(policy=policy, max_num_seqs=1)
    scheduler.add_request(request("older", priority=10))
    scheduler.add_request(request("higher", priority=0))
    assert scheduler.schedule().spans[0].request_id == expected


def test_aged_admission_orders_by_wait_in_priority_mode(scheduler_factory):
    scheduler = scheduler_factory(policy="priority", max_num_seqs=1)
    scheduler.add_request(request("older", priority=10))
    scheduler.add_request(request("higher", priority=0))
    for since, request_id in enumerate(("older", "higher"), start=1):
        state = scheduler._states[request_id]
        state.entry_ready_since, state.entry_skips = since, 2
    assert scheduler.schedule().spans[0].request_id == "older"


@pytest.mark.parametrize("async_scheduling", (False, True))
def test_prefill_is_promoted_after_two_ready_skips(scheduler_factory, async_scheduling):
    scheduler = scheduler_factory(async_scheduling)
    decoder = request("decode", max_tokens=8)
    scheduler.add_request(decoder)
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    scheduler.add_request(request("prompt", prompt=[2] * 16))
    work = scheduler._reserve_span(scheduler._states["prompt"], 16, [])
    work.next_op = LoopOp.CORE
    work.next_depth = 1
    for skips in (1, 2):
        plan = scheduler.schedule()
        assert all(span.request_id == "decode" for span in plan.spans)
        assert work.skips == skips
        complete(scheduler, plan)
    plan = scheduler.schedule()
    assert plan.op is LoopOp.CORE and plan.spans[0].request_id == "prompt"
    assert work.skips == 0


@pytest.mark.parametrize("async_scheduling", (False, True))
def test_decode_coda_precedes_ready_core_and_prefill_first_token(
    scheduler_factory, async_scheduling,
):
    scheduler = scheduler_factory(async_scheduling)
    scheduler.add_request(request("decode"))
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    scheduler.add_request(request("prompt", prompt=list(range(16))))
    for request_id in ("decode", "prompt"):
        work = scheduler._reserve_span(scheduler._states[request_id], 16, [])
        work.next_op = LoopOp.CODA
        work.next_depth = work.completed_depth = work.depth_limit - 1

    plan = scheduler.schedule()
    assert plan.op is LoopOp.CODA
    assert [(span.request_id, span.sample, span.token_count) for span in plan.spans] == [
        ("decode", True, 1)
    ]


def test_coda_batches_rows_from_each_final_depth(scheduler_factory):
    scheduler = scheduler_factory()
    for name, loops in zip("abc", (1, 2, 3), strict=True):
        scheduler.add_request(request(name, loops=loops))
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    for name in "abc":
        work = scheduler._reserve_span(scheduler._states[name], 1, [])
        work.next_op = LoopOp.CODA
        work.next_depth = work.completed_depth = work.depth_limit - 1

    plan = scheduler.schedule()
    assert plan.op is LoopOp.CODA
    assert {(span.request_id, span.depth) for span in plan.spans} == {("a", 0), ("b", 1), ("c", 2)}


@pytest.mark.parametrize("async_scheduling", (False, True))
def test_aged_work_can_run_after_core_before_queued_coda(scheduler_factory, async_scheduling):
    scheduler = scheduler_factory(async_scheduling)
    for name, loops in zip("abc", (1, 2, 1), strict=True):
        scheduler.add_request(request(name, loops=loops))
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    for name in "abc":
        work = scheduler._reserve_span(scheduler._states[name], 1, [])
        work.next_op = LoopOp.CODA if name == "a" else LoopOp.CORE
        work.next_depth = work.depth_limit - 1

    plan = scheduler.schedule()
    assert plan.op is LoopOp.CORE
    assert set(plan.num_scheduled_tokens) == {"b"}
    complete(scheduler, plan)

    scheduler.add_request(request("prompt"))
    prompt = scheduler._states["prompt"]
    prompt.entry_ready_since = 1
    prompt.entry_skips = 2
    plan = scheduler.schedule()
    assert plan.op is LoopOp.PRELUDE
    assert "prompt" in plan.num_scheduled_tokens
    complete(scheduler, plan)
    plan = scheduler.schedule()
    assert plan.op is LoopOp.CODA
    assert set(plan.num_scheduled_tokens) == {"a", "b"}


@pytest.mark.parametrize("case", (
    "aged", "token_limit", "sequence_limit", "full_core", "lower_priority", "same_target",
    "same_graph_width",
))
def test_stage_selection_respects_coda_age_priority_and_batch_capacity(scheduler_factory, case):
    scheduler = scheduler_factory(
        decode_graph_sizes=(1, 2, 4), policy="priority" if case == "lower_priority" else "fcfs",
    )
    for name in "abcd":
        loops = 1 if name == "a" and case != "same_target" else 3
        priority = int(case == "lower_priority" and name != "a")
        scheduler.add_request(request(name, loops=loops, priority=priority))
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    waiting = {
        "same_graph_width": "abc", "token_limit": "ab", "sequence_limit": "ab",
    }.get(case, "a")
    for name in "abcd":
        work = scheduler._reserve_span(scheduler._states[name], 1, [])
        work.next_op = LoopOp.CODA if name in waiting else LoopOp.CORE
        work.next_depth = work.depth_limit - 1
        if name in waiting and case == "aged":
            work.skips = 2
    if case == "token_limit":
        scheduler.max_num_scheduled_tokens = 3
    if case == "sequence_limit":
        scheduler.max_num_seqs = 3
    if case == "full_core":
        scheduler.max_num_seqs = 2

    plan = scheduler.schedule()
    if case in ("same_target", "same_graph_width"):
        producer = "d" if case == "same_graph_width" else "bcd"
        assert plan.op is LoopOp.CORE and set(plan.num_scheduled_tokens) == set(producer)
        complete(scheduler, plan)
        plan = scheduler.schedule()
        assert plan.op is LoopOp.CODA and set(plan.num_scheduled_tokens) == set("abcd")
    elif case == "full_core":
        assert plan.op is LoopOp.CORE and set(plan.num_scheduled_tokens) == {"b", "c"}
    else:
        assert plan.op is LoopOp.CODA and set(plan.num_scheduled_tokens) == set(waiting)


def test_ready_decode_prelude_can_refill_partial_core(scheduler_factory):
    scheduler = scheduler_factory()
    for name in "ab":
        scheduler.add_request(request(name))
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    work = scheduler._reserve_span(scheduler._states["a"], 1, [])
    work.next_op = LoopOp.CORE

    plan = scheduler.schedule()
    assert plan.op is LoopOp.PRELUDE and set(plan.num_scheduled_tokens) == {"b"}
    complete(scheduler, plan)
    plan = scheduler.schedule()
    assert plan.op is LoopOp.CORE and set(plan.num_scheduled_tokens) == {"a", "b"}


@pytest.mark.parametrize("async_scheduling", (False, True))
def test_next_core_batch_can_use_newly_ready_rows(scheduler_factory, async_scheduling):
    scheduler = scheduler_factory(async_scheduling, additional_config={"rlt_mixed_core": True})
    for name, loops in zip("abcd", (1, 2, 3, 3), strict=True):
        scheduler.add_request(request(name, loops=loops))
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    for name in "abc":
        work = scheduler._reserve_span(scheduler._states[name], 1, [])
        work.next_op = LoopOp.CODA if name == "a" else LoopOp.CORE
        work.next_depth = 0 if name == "a" else 1
    newcomer = scheduler._reserve_span(scheduler._states["d"], 1, [])
    newcomer.next_op = None

    plan = scheduler.schedule()
    assert plan.op is LoopOp.CORE and set(plan.num_scheduled_tokens) == {"b", "c"}
    complete(scheduler, plan)

    newcomer.next_op = LoopOp.CORE
    plan = scheduler.schedule()
    assert plan.op is LoopOp.CORE
    assert {(span.request_id, span.depth) for span in plan.spans} == {("c", 2), ("d", 0)}
    complete(scheduler, plan)
    plan = scheduler.schedule()
    assert plan.op is LoopOp.CODA
    assert set(plan.num_scheduled_tokens) == {"a", "b", "c"}


def test_coda_never_waits_for_an_outstanding_core(scheduler_factory):
    scheduler = scheduler_factory()
    scheduler.add_request(request("a", loops=1))
    scheduler.add_request(request("b", loops=3))
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    for name in "ab":
        work = scheduler._reserve_span(scheduler._states[name], 1, [])
        work.next_op = LoopOp.CODA if name == "a" else LoopOp.CORE
        work.next_depth = work.depth_limit - 1

    producer = scheduler.schedule()
    assert producer.op is LoopOp.CORE
    plan = scheduler.schedule()
    assert plan.op is LoopOp.CODA and set(plan.num_scheduled_tokens) == {"a"}
    complete(scheduler, producer)
    complete(scheduler, plan)


def test_pure_decode_core_packs_ready_rows_from_multiple_depths(scheduler_factory):
    scheduler = scheduler_factory(additional_config={"rlt_mixed_core": True})
    for name in "abc":
        scheduler.add_request(request(name))
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    for depth, name in enumerate("abc"):
        work = scheduler._reserve_span(scheduler._states[name], 1, [])
        work.next_op = LoopOp.CORE
        work.next_depth, work.completed_depth = depth, depth - 1

    plan = scheduler.schedule()
    assert plan.op is LoopOp.CORE
    assert {(span.request_id, span.depth) for span in plan.spans} == {("a", 0), ("b", 1), ("c", 2)}


@pytest.mark.parametrize("async_scheduling", (False, True))
@pytest.mark.parametrize("cancel_in_flight", (False, True))
def test_prefill_coda_can_fill_seven_rows_and_preserve_the_final_token(
    scheduler_factory, async_scheduling, cancel_in_flight,
):
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import RequestStatus

    init_none_hash(sha256)
    hasher = get_request_block_hasher(4, sha256)
    scheduler = scheduler_factory(
        async_scheduling,
        prefix=True,
        prefill_lookahead=True,
        split_prefill_coda=True,
        max_num_batched_tokens=8,
    )
    decoder = request("decode", max_tokens=2)
    prompt = request("prompt", prompt=list(range(8)), block_hasher=hasher)
    probe = request("probe", prompt=list(range(9)), block_hasher=hasher)
    scheduler.add_request(decoder)
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    scheduler.add_request(prompt)
    for req in (decoder, prompt):
        work = scheduler._reserve_span(scheduler._states[req.request_id], 8, [])
        work.next_op = LoopOp.CODA
        work.next_depth = work.completed_depth = work.depth_limit - 1
    blocks = scheduler.kv.get_block_ids("prompt")

    prefix = scheduler.schedule()
    assert prefix.op is LoopOp.CODA
    assert prefix.total_num_scheduled_tokens == 8
    assert [
        (row.request_id, row.token_start, row.token_count, row.sample, row.coda_remainder)
        for row in prefix.spans
    ] == [("decode", 3, 1, True, 0), ("prompt", 0, 7, False, 1)]
    assert not scheduler.schedule().spans
    _, hit_before_ack, _ = scheduler.kv.get_computed_blocks(probe)
    assert hit_before_ack == 0
    if cancel_in_flight:
        scheduler.finish_requests("prompt", RequestStatus.FINISHED_ABORTED)
        assert scheduler.has_request("prompt")
        complete(scheduler, prefix)
        assert not scheduler.has_request("prompt")
        assert "prompt" in scheduler.schedule().finished_req_ids
        _, hit_after_abort, _ = scheduler.kv.get_computed_blocks(probe)
        assert hit_after_abort == 0
        return

    complete(scheduler, prefix)
    assert prompt.num_computed_tokens == 7
    assert prompt.num_output_tokens == 0
    assert scheduler.kv.get_block_ids("prompt") == blocks
    _, hit_after_prefix, _ = scheduler.kv.get_computed_blocks(probe)
    assert hit_after_prefix == 4

    suffix = scheduler.schedule()
    assert suffix.op is LoopOp.CODA
    assert [(row.token_start, row.token_count, row.sample) for row in suffix.spans] == [
        (7, 1, True),
    ]
    assert suffix.scheduled_cached_reqs.num_computed_tokens == [7]
    assert suffix.scheduled_cached_reqs.new_block_ids == [None]
    complete(scheduler, suffix)
    assert prompt.num_computed_tokens == 8
    assert prompt.num_output_tokens == 1
    _, hit_after_suffix, _ = scheduler.kv.get_computed_blocks(probe)
    assert hit_after_suffix == 8


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_full_prelude_does_not_starve_ready_core(scheduler_factory, async_scheduling):
    scheduler = scheduler_factory(async_scheduling)
    for name in ("a", "b", "c", "d"):
        scheduler.add_request(request(name, prompt=list(range(16))))
    for name in ("a", "b"):
        plan = scheduler.schedule()
        assert plan.op is LoopOp.PRELUDE and plan.num_scheduled_tokens == {name: 16}
        complete(scheduler, plan)
    plan = scheduler.schedule()
    assert plan.op is LoopOp.CORE and plan.num_scheduled_tokens == {"a": 16}


def test_waiting_prefill_cannot_preempt_running_request(scheduler_factory, monkeypatch):
    scheduler = scheduler_factory()
    scheduler.add_request(request("a"))
    for _ in range(4):
        complete(scheduler, scheduler.schedule())
    pending_coda = scheduler.schedule()
    assert pending_coda.op is LoopOp.CODA

    scheduler.add_request(request("b"))
    pending_prelude = scheduler.schedule()
    assert pending_prelude.op is LoopOp.PRELUDE
    complete(scheduler, pending_coda)
    complete(scheduler, pending_prelude)
    b_blocks = scheduler.kv.get_block_ids("b")

    allocate_slots = scheduler.kv.allocate_slots

    def allocate_unless_waiting(req, *args, **kwargs):
        if req.request_id == "c":
            return None
        return allocate_slots(req, *args, **kwargs)

    monkeypatch.setattr(scheduler.kv, "allocate_slots", allocate_unless_waiting)
    scheduler.add_request(request("c"))
    decode = scheduler.schedule()
    assert decode.op is LoopOp.PRELUDE
    assert [span.request_id for span in decode.spans] == ["a"]
    assert scheduler.get_request_counts() == (2, 1)
    assert scheduler.kv.get_block_ids("b") == b_blocks
    assert not decode.finished_req_ids


def test_prefill_coda_gets_budget_after_two_ready_skips(scheduler_factory):
    scheduler = scheduler_factory()
    scheduler.add_request(request("a", loops=1))
    for _ in range(7):
        complete(scheduler, scheduler.schedule())
    scheduler.add_request(request("b", prompt=list(range(24))))
    work = scheduler._reserve_span(scheduler._states["b"], 16, [])
    work.next_op = LoopOp.CODA
    work.next_depth = work.completed_depth = work.depth_limit - 1
    for skips, op in enumerate((LoopOp.CODA, LoopOp.PRELUDE), start=1):
        plan = scheduler.schedule()
        assert plan.op is op and "a" in plan.num_scheduled_tokens
        assert work.skips == skips
        complete(scheduler, plan)
    prefill = scheduler.schedule()
    assert prefill.op is LoopOp.CODA
    assert prefill.num_scheduled_tokens == {"b": 16}


def test_async_cancel_waits_for_every_inflight_core_step(scheduler_factory):
    from vllm.v1.request import RequestStatus

    scheduler = scheduler_factory(async_scheduling=True)
    scheduler.add_request(request("a"))
    prelude = scheduler.schedule()
    core = scheduler.schedule()
    assert prelude.op is LoopOp.PRELUDE and core.op is LoopOp.CORE
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert scheduler.has_request("a")
    assert not complete(scheduler, prelude)
    assert scheduler.has_request("a")
    assert not complete(scheduler, core)
    assert not scheduler.has_request("a")
    assert scheduler.schedule().finished_req_ids == {"a"}


def test_async_prior_result_does_not_enqueue_coda_twice(scheduler_factory):
    scheduler = scheduler_factory(async_scheduling=True)
    scheduler.add_request(request("a"))
    complete(scheduler, scheduler.schedule())
    complete(scheduler, scheduler.schedule())
    previous = scheduler.schedule()
    last = scheduler.schedule()
    assert last.spans[0].last_core
    complete(scheduler, previous)
    coda = scheduler.schedule()
    assert coda.op is LoopOp.CODA
    complete(scheduler, last)
    assert scheduler.schedule().op is None
    assert complete(scheduler, coda)[0].outputs[0].new_token_ids == [42]


@pytest.mark.parametrize("outcome", ["continue", "stop", "abort", "length"])
def test_async_decode_uses_coda_device_token_before_cpu_output(scheduler_factory, outcome):
    from vllm.v1.request import RequestStatus

    scheduler = scheduler_factory(async_scheduling=True)
    requested = request(
        "a",
        loops=1,
        max_tokens=2 if outcome == "length" else 3,
        stop_token_ids=[43] if outcome == "stop" else None,
    )
    scheduler.add_request(requested)
    for _ in range(5):
        complete(scheduler, scheduler.schedule())
    assert list(requested.output_token_ids) == [42]
    complete(scheduler, scheduler.schedule())  # Decode Prelude.
    complete(scheduler, scheduler.schedule())  # One request-level CORE.

    coda = scheduler.schedule()
    assert coda.op is LoopOp.CODA and coda.spans[0].token_start == 3
    assert requested.num_output_placeholders == 1
    prelude = scheduler.schedule()
    if outcome == "length":
        assert prelude.op is None
        assert complete(scheduler, coda)[0].outputs[0].new_token_ids == [42]
        assert not scheduler.has_request("a")
        return

    assert prelude.op is LoopOp.PRELUDE
    assert prelude.spans[0].token_start == 4 and prelude.spans[0].input_token is None
    assert prelude.scheduled_cached_reqs.num_computed_tokens == [4]
    assert prelude.scheduled_cached_reqs.new_block_ids[0][0]
    assert prelude.scheduled_cached_reqs.new_token_ids == []
    assert requested.num_computed_tokens == 3 and list(requested.output_token_ids) == [42]
    assert scheduler.schedule().op is None  # CORE must wait for Coda confirmation.
    if outcome == "abort":
        scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    outputs = complete(scheduler, coda, token=43)
    assert scheduler.has_request("a")  # The submitted Prelude still owns its KV.
    if outcome == "abort":
        assert not outputs
    else:
        assert outputs[0].outputs[0].new_token_ids == [43]
        assert requested.num_computed_tokens == 4 and requested.num_output_placeholders == 0
    assert not complete(scheduler, prelude)

    if outcome == "continue":
        core = scheduler.schedule()
        assert core.op is LoopOp.CORE and core.spans[0].last_core
        assert core.spans[0].token_start == 4
    else:
        assert not scheduler.has_request("a")
        assert scheduler.schedule().finished_req_ids == {"a"}


def test_prefix_hit_requires_the_final_op(scheduler_factory):
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    init_none_hash(sha256)
    hasher = get_request_block_hasher(4, sha256)
    prompt = list(range(8))
    scheduler = scheduler_factory(prefix=True)
    scheduler.add_request(request("a", prompt=prompt, block_hasher=hasher))
    other = request("b", prompt=prompt, block_hasher=hasher)
    for _ in range(4):
        plan = scheduler.schedule()
        assert plan.op is not LoopOp.CODA
        complete(scheduler, plan)
        assert scheduler.kv.get_computed_blocks(other)[1] == 0

    coda = scheduler.schedule()
    assert coda.op is LoopOp.CODA
    complete(scheduler, coda)
    assert scheduler.kv.get_computed_blocks(other)[1] == 4


@pytest.mark.parametrize("abort", [False, True], ids=["continue", "abort"])
def test_async_prefill_stages_next_chunk_without_committing_pending_kv(
    scheduler_factory, monkeypatch, abort
):
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import RequestStatus

    init_none_hash(sha256)
    hasher = get_request_block_hasher(4, sha256)
    prompt = list(range(24))
    scheduler = scheduler_factory(
        async_scheduling=True, prefix=True, prefill_lookahead=True, shared_kv=True,
    )
    requested = request("a", prompt=prompt, block_hasher=hasher)
    probe = request("prefix_probe", prompt=prompt, block_hasher=hasher)
    scheduler.add_request(requested)

    for op in (LoopOp.PRELUDE, LoopOp.CORE, LoopOp.CORE, LoopOp.CORE):
        plan = scheduler.schedule()
        assert plan.op is op
        complete(scheduler, plan)
    coda = scheduler.schedule()
    assert coda.op is LoopOp.CODA and not coda.spans[0].sample

    def unexpected_preemption(*args):
        pytest.fail("a prefill lookahead must not preempt another request")

    with monkeypatch.context() as patch:
        patch.setattr(scheduler.kv, "allocate_slots", lambda *args, **kwargs: None)
        patch.setattr(scheduler, "_preempt_for", unexpected_preemption)
        assert scheduler.schedule().op is None

    prelude = scheduler.schedule()
    assert prelude.op is LoopOp.PRELUDE
    assert [(s.token_start, s.token_count, s.sample) for s in prelude.spans] == [(16, 8, False)]
    assert prelude.scheduled_cached_reqs.num_computed_tokens == [16]
    assert prelude.scheduled_cached_reqs.new_block_ids[0][0]
    assert requested.num_computed_tokens == 0
    assert scheduler.kv.get_computed_blocks(probe)[1] == 0
    assert scheduler.schedule().op is None  # The next CORE waits for CODA's receipt.

    allocated = scheduler.kv.get_block_ids("a")
    if abort:
        scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert not complete(scheduler, coda)
    assert scheduler.has_request("a")  # The submitted Prelude still owns KV.
    if abort:
        assert scheduler.kv.get_block_ids("a") == allocated
        assert scheduler.kv.get_computed_blocks(probe)[1] == 0
        assert not complete(scheduler, prelude)
        assert not scheduler.has_request("a")
        assert not any(scheduler.kv.get_block_ids("a"))
        assert scheduler.schedule().finished_req_ids == {"a"}
        return

    assert requested.num_computed_tokens == 16
    assert scheduler.kv.get_computed_blocks(probe)[1] == 16
    assert not complete(scheduler, prelude)
    for depth in range(3):
        plan = scheduler.schedule()
        assert plan.op is LoopOp.CORE and plan.spans[0].depth == depth
        complete(scheduler, plan)
    final = scheduler.schedule()
    assert final.op is LoopOp.CODA and final.spans[0].sample
    assert scheduler.schedule().op is None  # The final prompt chunk needs the sampled token.
    assert complete(scheduler, final)[0].outputs[0].new_token_ids == [42]
    assert requested.num_computed_tokens == 24


@pytest.mark.parametrize("async_scheduling", (False, True))
def test_later_chunk_waits_for_previous_core_kv(scheduler_factory, async_scheduling):
    scheduler = scheduler_factory(async_scheduling, long_prefill_token_threshold=4)
    scheduler.add_request(request("a", prompt=[1] * 8))
    prelude = scheduler.schedule()
    complete(scheduler, prelude)
    first = scheduler.schedule()
    assert [(s.chunk_id, s.depth) for s in first.spans] == [(0, 0)]
    pending = scheduler.schedule()
    assert not any(s.chunk_id == 1 for s in pending.spans)
    complete(scheduler, first)
    if pending.spans:
        complete(scheduler, pending)
    assert any(s.chunk_id == 1 for s in scheduler.schedule().spans)


def test_adaptive_decode_does_not_issue_a_deeper_step_before_exit(scheduler_factory):
    from vllm_rlt.core.output import LoopModelRunnerOutput

    scheduler = scheduler_factory(async_scheduling=True, ouro=True)
    requested = request("a", threshold=0.5)
    scheduler.add_request(requested)
    for depth in range(4):
        plan = scheduler.schedule()
        assert plan.op is LoopOp.CORE
        assert plan.spans[0].depth == depth and not plan.spans[0].check_exit
        complete(scheduler, plan)
    plan = scheduler.schedule()
    assert plan.spans[0].check_exit
    assert scheduler.schedule().op is None
    result = LoopModelRunnerOutput(req_ids=["a"], sampled_token_ids=[[43]], exit_depths=(0,))
    assert scheduler.update_from_output(plan, result)[0].outputs[0].new_token_ids == [43]
    assert requested.num_computed_tokens == 4
    allocated = scheduler.kv.get_block_ids("a")
    assert not scheduler.reset_prefix_cache(reset_running_requests=True)
    assert scheduler.kv.get_block_ids("a") == allocated
    assert scheduler.schedule().spans[0].depth == 0


@pytest.mark.parametrize("candidate", [42, 99], ids=["accept", "reject"])
def test_shallow_draft_is_tentative_until_target_coda(scheduler_factory, candidate):
    from vllm_rlt.core.output import LoopModelRunnerOutput

    scheduler = scheduler_factory(async_scheduling=True, speculate=True)
    requested = request("a")
    scheduler.add_request(requested)
    for _ in range(6):
        complete(scheduler, scheduler.schedule())
    shallow = scheduler.schedule()
    assert shallow.op is LoopOp.CORE and shallow.spans[0].propose_draft
    assert scheduler.schedule().op is None
    complete(scheduler, shallow)
    draft = scheduler.schedule()
    assert draft.op is LoopOp.CODA and draft.spans[0].draft_coda
    assert not draft.spans[0].sample
    result = LoopModelRunnerOutput(
        req_ids=["a"], sampled_token_ids=[[]], draft_tokens={"a": candidate}
    )
    assert not scheduler.update_from_output(draft, result)
    assert list(requested.output_token_ids) == [42]
    assert requested.num_computed_tokens == 3
    for depth in (1, 2):
        plan = scheduler.schedule()
        assert plan.op is LoopOp.CORE and plan.spans[0].depth == depth
        assert not complete(scheduler, plan)
    target = scheduler.schedule()
    assert target.op is LoopOp.CODA and target.spans[0].sample
    assert complete(scheduler, target)[0].outputs[0].new_token_ids == [42]
    assert list(requested.output_token_ids) == [42, 42]


@pytest.mark.parametrize("grouped", [False, True], ids=["native-coda", "worker-group"])
def test_final_op_preserves_generation_logprobs(scheduler_factory, grouped):
    import numpy as np
    from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput

    from vllm_rlt.core.output import LoopModelRunnerOutput

    scheduler = scheduler_factory()
    req = request("a")
    req.sampling_params.logprobs = 0
    scheduler.add_request(req)
    for _ in range(4):
        complete(scheduler, scheduler.schedule())
    coda = scheduler.schedule()
    scores = LogprobsLists(np.array([[42]]), np.array([[-0.5]]), np.array([1]))
    kwargs = dict(req_ids=["a"], sampled_token_ids=[[42]])
    if grouped:
        output = LoopModelRunnerOutput(**kwargs, logprobs_by_request={"a": scores})
    else:
        output = ModelRunnerOutput(**kwargs, logprobs=scores)
    result = scheduler.update_from_output(coda, output)[0].outputs[0]
    assert result.new_token_ids == [42]
    assert result.new_logprobs is not None
    assert result.new_logprobs.logprobs.tolist() == [[-0.5]]


@pytest.mark.parametrize("candidate", [42, 99], ids=["retain", "rollback"])
def test_pipeline_runs_next_token_before_validation(scheduler_factory, candidate):
    from vllm_rlt.core.output import LoopModelRunnerOutput

    scheduler = scheduler_factory(async_scheduling=True, speculate=True, pipeline=True)
    req = request("a")
    scheduler.add_request(req)
    for _ in range(7):
        complete(scheduler, scheduler.schedule())
    draft = scheduler.schedule()
    assert draft.op is LoopOp.CODA and draft.spans[0].draft_coda
    scheduler.update_from_output(
        draft,
        LoopModelRunnerOutput(req_ids=["a"], sampled_token_ids=[[]], draft_tokens={"a": candidate}),
    )
    next_prelude = scheduler.schedule()
    assert next_prelude.op is LoopOp.PRELUDE
    assert next_prelude.spans[0].input_token == candidate
    complete(scheduler, next_prelude)
    core = scheduler.schedule()
    assert core.op is LoopOp.CORE
    assert [(span.token_start, span.depth) for span in core.spans] == [(3, 1), (4, 0)]
    complete(scheduler, core)
    while (target := scheduler.schedule()).op is LoopOp.CORE:
        complete(scheduler, target)
    assert target.op is LoopOp.CODA
    assert [span.token_start for span in target.spans] == [3]
    output = complete(scheduler, target)[0].outputs[0]
    assert output.new_token_ids == [42]
    assert req.num_computed_tokens == 4
    following = scheduler.schedule()
    if candidate == 42:
        assert not following.discarded_states
        assert following.op in (LoopOp.CORE, LoopOp.CODA)
    else:
        assert following.discarded_states == {"a": (4,)}
        assert following.op is LoopOp.PRELUDE and following.spans[0].input_token is None
