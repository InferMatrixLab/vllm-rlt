from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.core.loop import LoopOp


@pytest.fixture
def scheduler_factory():
    pytest.importorskip("vllm")
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup

    from vllm_rlt.core.async_scheduler import LoopAsyncScheduler
    from vllm_rlt.core.loop_scheduler import LoopScheduler

    def make(
        async_scheduling=False,
        *,
        prefix=False,
        wavefront=False,
        ouro=False,
        speculate=False,
        pipeline=False,
        prefill_lookahead=False,
        batch_decode_coda=False,
        long_prefill_token_threshold=0,
        finish_decode_token=False,
    ):
        text = SimpleNamespace(
            model_type="inkling2_model", num_loop_passes=3, loop_config={"is_varloop": True}
        )
        if ouro:
            text = SimpleNamespace(model_type="ouro", total_ut_steps=4)
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=4,
                max_num_batched_tokens=16,
                async_scheduling=async_scheduling,
                long_prefill_token_threshold=long_prefill_token_threshold,
            ),
            model_config=SimpleNamespace(hf_text_config=text, max_model_len=64),
            cache_config=SimpleNamespace(enable_prefix_caching=prefix),
            additional_config={
                "rlt_wavefront_prefill": wavefront,
                "rlt_prefill_chunk_size": 4,
                "rlt_self_speculation": speculate,
                "rlt_decode_pipeline": pipeline,
                "rlt_async_prefill_lookahead": prefill_lookahead,
                "rlt_batch_decode_coda": batch_decode_coda,
                "rlt_finish_decode_token": finish_decode_token,
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
@pytest.mark.parametrize("longer_depth", (2, 3))
def test_decode_coda_can_share_one_already_ready_terminal_core_plan(
    scheduler_factory, async_scheduling, longer_depth
):
    scheduler = scheduler_factory(async_scheduling, batch_decode_coda=True)
    scheduler.add_request(request("a", loops=1))
    scheduler.add_request(request("b", loops=longer_depth))

    for op in (LoopOp.PRELUDE, LoopOp.CORE, LoopOp.CORE, LoopOp.CORE, LoopOp.CODA):
        plan = scheduler.schedule()
        assert plan.op is op and len(plan.spans) == 2
        complete(scheduler, plan)

    prelude = scheduler.schedule()
    assert prelude.op is LoopOp.PRELUDE and all(not span.prefill for span in prelude.spans)
    complete(scheduler, prelude)
    core = scheduler.schedule()
    assert core.op is LoopOp.CORE
    assert [span.last_core for span in core.spans] == [True, False]
    complete(scheduler, core)

    if longer_depth == 2:
        terminal_core = scheduler.schedule()
        assert terminal_core.op is LoopOp.CORE
        assert [(span.request_id, span.last_core) for span in terminal_core.spans] == [("b", True)]
        complete(scheduler, terminal_core)
    coda = scheduler.schedule()
    assert coda.op is LoopOp.CODA
    assert {span.request_id for span in coda.spans} == ({"a", "b"} if longer_depth == 2 else {"a"})
    assert all(span.sample for span in coda.spans)
    complete(scheduler, coda)


@pytest.mark.parametrize("async_scheduling", (False, True))
def test_ready_decode_core_finishes_the_same_token_before_coda(scheduler_factory, async_scheduling):
    scheduler = scheduler_factory(async_scheduling, finish_decode_token=True)
    scheduler.add_request(request("a", loops=1))
    scheduler.add_request(request("b", loops=3))

    for op in (LoopOp.PRELUDE, LoopOp.CORE, LoopOp.CORE, LoopOp.CORE, LoopOp.CODA):
        plan = scheduler.schedule()
        assert plan.op is op and len(plan.spans) == 2
        complete(scheduler, plan)

    plan = scheduler.schedule()
    assert plan.op is LoopOp.PRELUDE and len(plan.spans) == 2
    complete(scheduler, plan)

    for depth in range(3):
        plan = scheduler.schedule()
        assert plan.op is LoopOp.CORE
        assert [(span.request_id, span.depth) for span in plan.spans] == (
            [("a", 0), ("b", 0)] if depth == 0 else [("b", depth)]
        )
        complete(scheduler, plan)

    plan = scheduler.schedule()
    assert plan.op is LoopOp.CODA
    assert {span.request_id for span in plan.spans} == {"a", "b"}
    assert all(span.sample for span in plan.spans)


@pytest.mark.parametrize("async_scheduling", (False, True))
def test_prefill_and_decode_can_catch_up_before_a_shared_core(scheduler_factory, async_scheduling):
    scheduler = scheduler_factory(async_scheduling, finish_decode_token=True)
    for request_id in ("ahead", "catch_up"):
        scheduler.add_request(request(request_id, loops=3))
    for op in (LoopOp.PRELUDE, LoopOp.CORE, LoopOp.CORE, LoopOp.CORE, LoopOp.CODA):
        plan = scheduler.schedule()
        assert plan.op is op
        complete(scheduler, plan)

    scheduler.add_request(request("prompt"))
    for request_id, depth in (("ahead", 2), ("catch_up", 1), ("prompt", 1)):
        state = scheduler._states[request_id]
        work = scheduler._reserve_span(state, 8, [])
        work.next_op = LoopOp.CORE
        work.next_depth = depth
        work.completed_depth = depth - 1

    catching_up = scheduler.schedule()
    assert catching_up.op is LoopOp.CORE
    assert {(span.request_id, span.depth, span.prefill) for span in catching_up.spans} == {
        ("catch_up", 1, False),
        ("prompt", 1, True),
    }
    complete(scheduler, catching_up)

    shared = scheduler.schedule()
    assert shared.op is LoopOp.CORE
    assert {span.request_id for span in shared.spans} == {"ahead", "catch_up", "prompt"}
    assert all(span.depth == 2 and span.last_core for span in shared.spans)


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_running_requests_advance_before_new_prefill(scheduler_factory, async_scheduling):
    scheduler = scheduler_factory(async_scheduling)
    scheduler.add_request(request("a", prompt=list(range(24))))
    scheduler.add_request(request("b", prompt=list(range(12))))
    scheduler.add_request(request("c", prompt=list(range(24))))

    for op in (LoopOp.PRELUDE, LoopOp.CORE, LoopOp.CORE, LoopOp.CORE, LoopOp.CODA):
        plan = scheduler.schedule()
        assert plan.op is op
        assert [(s.request_id, s.token_start, s.token_count) for s in plan.spans] == [("a", 0, 16)]
        assert not complete(scheduler, plan)

    second_chunk = scheduler.schedule()
    assert second_chunk.op is LoopOp.PRELUDE
    assert [(s.request_id, s.token_start, s.token_count) for s in second_chunk.spans] == [
        ("a", 16, 8),
        ("b", 0, 8),
    ]
    complete(scheduler, second_chunk)
    for _ in range(4):
        complete(scheduler, scheduler.schedule())

    mixed = scheduler.schedule()
    assert mixed.op is LoopOp.PRELUDE
    assert [(s.request_id, s.prefill, s.token_count) for s in mixed.spans] == [
        ("a", False, 1),
        ("b", True, 4),
        ("c", True, 11),
    ]


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


def test_reserved_prefill_coda_gets_budget_after_being_skipped(scheduler_factory):
    scheduler = scheduler_factory()
    scheduler.add_request(request("a", loops=1))
    for _ in range(6):
        complete(scheduler, scheduler.schedule())
    a_core = scheduler.schedule()
    assert a_core.op is LoopOp.CORE and a_core.spans[0].last_core

    scheduler.add_request(request("b", prompt=list(range(24))))
    for _ in range(3):
        complete(scheduler, scheduler.schedule())
    b_core = scheduler.schedule()
    assert b_core.op is LoopOp.CORE and b_core.spans[0].last_core
    complete(scheduler, a_core)
    complete(scheduler, b_core)

    decode = scheduler.schedule()
    assert decode.op is LoopOp.CODA
    assert [(span.request_id, span.sample, span.token_count) for span in decode.spans] == [
        ("a", True, 1)
    ]
    complete(scheduler, decode)
    prefill = scheduler.schedule()
    assert prefill.op is LoopOp.CODA
    assert [(span.request_id, span.sample, span.token_count) for span in prefill.spans] == [
        ("b", False, 16)
    ]


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
    scheduler = scheduler_factory(async_scheduling=True, prefix=True, prefill_lookahead=True)
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


def test_wavefront_reads_only_completed_previous_depths(scheduler_factory):
    scheduler = scheduler_factory(wavefront=True)
    requested = request("wave", prompt=list(range(12)))
    scheduler.add_request(requested)
    written = set()
    mixed_work = False
    while not requested.num_output_tokens:
        plan = scheduler.schedule()
        assert plan.spans
        if plan.op is LoopOp.CORE:
            for span in plan.spans:
                if span.token_start:
                    assert (span.token_start - 4, span.depth) in written
            mixed_work |= len({span.depth for span in plan.spans}) > 1
            written.update((span.token_start, span.depth) for span in plan.spans)
        complete(scheduler, plan)
    assert mixed_work
    assert written == {(position, depth) for position in (0, 4, 8) for depth in range(3)}
    assert requested.num_computed_tokens == 12


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
