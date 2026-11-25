"""LoopScheduler against the real vLLM Scheduler (CPU). Needs VLLM_SRC."""

from __future__ import annotations

from tests.plugin._sched_helpers import make_request, make_scheduler, run_step
from vllm_rlt.plugin.output import LoopSchedulerOutput
from vllm_rlt.plugin.program import LoopLayout, Op

INKLING = LoopLayout(num_passes=3)
OURO = LoopLayout(num_passes=4, has_prelude=False, has_gate=True)


def drain(sched, limit=200, **kw):
    """Run the scheduler to completion; return the emitted loop steps."""
    steps = []
    for _ in range(limit):
        if not sched.has_requests():
            break
        out = sched.schedule()
        if out.loop_step is None or out.total_num_scheduled_tokens == 0:
            continue
        steps.append((out.loop_step.op, out.loop_step.depths, out.loop_step.req_ids))
        run_step(sched, out, **kw)
    return steps


def test_every_step_holds_one_op_kind():
    sched = make_scheduler(INKLING)
    for i in range(3):
        sched.add_request(make_request(f"r{i}", prompt_len=4, max_tokens=2))
    kinds = []
    for op, depths, _ in drain(sched):
        kinds.append(op)
        if op == Op.CORE:
            assert len(set(depths)) == 1, "v0 never mixes depths in one step"
    assert {Op.PRELUDE, Op.CORE, Op.CODA} <= set(kinds)


def test_one_decode_token_walks_the_full_program():
    sched = make_scheduler(INKLING)
    sched.add_request(make_request("a", prompt_len=2, max_tokens=2))
    seq = [(op, d[0]) for op, d, _ in drain(sched)]
    # prompt chunk, then one decode token: each is PRELUDE, CORE 0..2, CODA
    one = [(Op.PRELUDE, 0), (Op.CORE, 0), (Op.CORE, 1), (Op.CORE, 2), (Op.CODA, 2)]
    assert seq == one + one


def test_ouro_layout_has_no_prelude_ops():
    sched = make_scheduler(OURO)
    sched.add_request(make_request("a", prompt_len=2, max_tokens=1))
    ops = [op for op, _, _ in drain(sched)]
    assert Op.PRELUDE not in ops and ops[0] == Op.CORE and ops[-1] == Op.CODA


def test_request_finishes_with_the_requested_number_of_tokens():
    sched = make_scheduler(INKLING)
    request = make_request("a", prompt_len=3, max_tokens=4)
    sched.add_request(request)
    drain(sched)
    assert not sched.has_requests() and not sched.running
    assert request.num_output_tokens == 4
    assert len(request.exit_depths) == 4  # one finished token per final op
    assert not request.in_loop


def test_token_level_metadata_reaches_the_runner_exactly_once():
    sched = make_scheduler(INKLING)
    sched.add_request(make_request("a", prompt_len=4, max_tokens=2))
    announced = 0
    for _ in range(40):
        if not sched.has_requests():
            break
        out = sched.schedule()
        announced += sum(1 for d in out.scheduled_new_reqs if d.req_id == "a")
        if out.total_num_scheduled_tokens:
            run_step(sched, out)
    assert announced == 1


def test_mid_loop_requests_are_not_scheduled_twice():
    sched = make_scheduler(INKLING)
    for i in range(2):
        sched.add_request(make_request(f"r{i}", prompt_len=4, max_tokens=3))
    in_loop_tokens = 0
    for _ in range(60):
        if not sched.has_requests():
            break
        out = sched.schedule()
        assert sum(out.num_scheduled_tokens.values()) == out.total_num_scheduled_tokens
        assert len(out.num_scheduled_tokens) <= 2
        if out.total_num_scheduled_tokens:
            in_loop_tokens += 1
            run_step(sched, out)
    assert in_loop_tokens > 0 and not sched.has_requests()


def test_vllm_limits_still_apply_to_hidden_requests():
    sched = make_scheduler(INKLING, max_num_seqs=2)
    for i in range(4):
        sched.add_request(make_request(f"r{i}", prompt_len=2, max_tokens=2))
    for _ in range(300):
        if not sched.has_requests():
            break
        out = sched.schedule()
        assert len(sched.running) <= 2
        if out.total_num_scheduled_tokens:
            run_step(sched, out)
    assert not sched.has_requests()


def test_early_exit_skips_cores_and_records_exit_depth():
    sched = make_scheduler(OURO)
    sched.add_request(
        make_request("a", prompt_len=2, max_tokens=2, rlt_exit_threshold=0.5, rlt_min_loops=1)
    )
    steps = []
    for _ in range(80):
        if not sched.has_requests():
            break
        out = sched.schedule()
        step = out.loop_step
        if step is None or not out.total_num_scheduled_tokens:
            continue
        gated = any(step.can_exit)
        depth = step.depths[0] if step.depths else -1
        steps.append((step.op, depth))
        # Report an exit at depth 1 for the (single-token) decode span.
        run_step(sched, out, exit_depths={"a": 1} if gated and depth == 1 else {})
    decode = steps[-3:]
    assert decode == [(Op.CORE, 0), (Op.CORE, 1), (Op.CODA, 1)], steps


def test_prompt_chunks_are_never_gated():
    sched = make_scheduler(OURO)
    sched.add_request(make_request("a", prompt_len=5, max_tokens=1, rlt_exit_threshold=0.5))
    first = sched.schedule()
    assert first.loop_step is not None and not any(first.loop_step.can_exit)


def test_async_scheduler_composes_with_vllm_asyncscheduler():
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    sched = make_scheduler(INKLING, asynchronous=True)
    assert isinstance(sched, AsyncScheduler) and sched.ASYNC
    names = [c.__name__ for c in type(sched).__mro__]
    # loop logic first, then vLLM's async layer, then the base Scheduler
    assert names.index("LoopScheduler") < names.index("AsyncScheduler") < names.index("Scheduler")
    sched.add_request(make_request("a", prompt_len=3, max_tokens=2))
    out = sched.schedule()
    assert isinstance(out, LoopSchedulerOutput) and out.loop_step.op == Op.PRELUDE


def test_async_rows_without_a_gate_are_advanced_at_emit_time():
    sched = make_scheduler(INKLING, asynchronous=True)
    request = make_request("a", prompt_len=3, max_tokens=2)
    sched.add_request(request)
    out = sched.schedule()
    assert out.loop_step.advanced == (True,) and not request.in_flight
    assert (request.loop_op, request.loop_depth) == (Op.CORE, 0)  # already moved on


def test_async_final_op_is_never_advanced_early():
    sched = make_scheduler(INKLING, asynchronous=True)
    request = make_request("a", prompt_len=3, max_tokens=1)
    sched.add_request(request)
    last = None
    for _ in range(10):
        out = sched.schedule()
        if out.loop_step is None or not out.total_num_scheduled_tokens:
            continue
        last = out
        if out.loop_step.op == Op.CODA:
            break
    assert last.loop_step.op == Op.CODA and last.loop_step.advanced == (False,)
    assert request.in_flight  # the span ends only when its token comes back


def test_full_only_model_is_plain_vllm():
    sched = make_scheduler(LoopLayout(num_passes=3, full_only=True))
    request = make_request("a", prompt_len=3, max_tokens=2)
    sched.add_request(request)
    produced = 0
    while sched.has_requests() and produced < 20:
        out = sched.schedule()
        assert getattr(out, "loop_step", None) is None
        if not out.total_num_scheduled_tokens:
            continue
        ids = list(out.num_scheduled_tokens)
        from vllm.v1.outputs import ModelRunnerOutput

        sched.update_from_output(
            out, ModelRunnerOutput(req_ids=ids, sampled_token_ids=[[5] for _ in ids])
        )
        produced += 1
    assert request.num_output_tokens == 2 and not request.in_loop


# -- WCPB: several chunks of one prompt in flight -------------------------------


def _wcpb_scheduler(**kwargs):
    sched = make_scheduler(INKLING, **kwargs)
    sched.wcpb, sched.wcpb_depth = True, INKLING.num_passes + 2
    return sched


def test_wcpb_keeps_more_than_one_chunk_of_a_prompt_in_flight():
    sched = _wcpb_scheduler(
        max_num_seqs=2, max_num_batched_tokens=4, long_prefill_token_threshold=4
    )
    request = make_request("a", prompt_len=12, max_tokens=1)
    sched.add_request(request)
    most = 0
    for _ in range(60):
        if not sched.has_requests():
            break
        out = sched.schedule()
        most = max(most, len(request.spans))
        if out.total_num_scheduled_tokens:
            # at most one span of a request per step: the batch is keyed by request
            assert len(out.num_scheduled_tokens) == len(set(out.num_scheduled_tokens))
            run_step(sched, out)
    assert most >= 2 and request.num_output_tokens == 1 and not request.spans


def test_wcpb_chunks_report_their_own_start():
    sched = _wcpb_scheduler(
        max_num_seqs=2, max_num_batched_tokens=4, long_prefill_token_threshold=4
    )
    sched.add_request(make_request("a", prompt_len=12, max_tokens=1))
    starts = set()
    for _ in range(60):
        if not sched.has_requests():
            break
        out = sched.schedule()
        if out.total_num_scheduled_tokens:
            starts.add(out.span_starts["a"])
            run_step(sched, out)
    assert {0, 4, 8} <= starts


def test_wcpb_rejects_a_shared_cache():
    import pytest as _pytest

    shared = LoopLayout(num_passes=3, share_cache=True)
    sched_cls = type(make_scheduler(INKLING))
    base = make_scheduler(INKLING)
    base.vllm_config.additional_config = {"rlt_wcpb": True}

    class Shared(sched_cls):
        def _resolve_layout(self):
            return shared

    with _pytest.raises(ValueError, match="KV cache per pass"):
        Shared(
            vllm_config=base.vllm_config,
            kv_cache_config=base.kv_cache_config,
            structured_output_manager=base.structured_output_manager,
            block_size=base.block_size,
        )
