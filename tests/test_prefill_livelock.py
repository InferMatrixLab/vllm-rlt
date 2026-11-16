"""Regression coverage for priority admission under staged prefill pressure.

A waiting request can be bypassed while other requests are admitted or
restored. The admission scan must count both kinds of progress against
``max_admission_bypasses``; otherwise a blocked resumption can be retried
forever while the requests already running never get a chance to drain.
"""

from collections import defaultdict

import pytest

from tests.helpers import tiny_ouro_model
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.scheduler import Scheduler
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.request import Request, RequestPhase

REQUESTS = (
    (0, "r0", [24, 25, 9, 13], 1, 1),
    (2, "r1", [6], 2, 2),
    (2, "r2", [16, 52, 33], 1, 1),
    (3, "r3", [26, 42, 2, 30], 5, 1),
    (5, "r4", [30, 25, 32, 37, 13, 58, 54, 26], 3, 1),
    (5, "r5", [32, 15], 5, 0),
    (6, "r6", [45], 3, 2),
    (8, "r7", [34, 27, 31, 58, 59], 2, 1),
    (10, "r8", [47, 8, 43, 17, 7, 53, 5], 5, 2),
    (11, "r9", [40, 54, 25, 7, 43, 4, 22], 1, 1),
)

REVIEWER_REQUESTS = (
    (0, "r0", [6], 4, 1),
    (4, "r1", [24, 54], 4, 0),
    (5, "r2", [48, 52, 43], 6, 1),
    (6, "r3", [17, 39, 14, 39, 3], 3, 1),
    (8, "r4", [44, 11, 28, 41, 26, 52, 47, 56, 33, 24], 4, 1),
    (8, "r5", [29, 33, 18, 58, 3, 56, 2, 24, 30], 6, 2),
    (9, "r6", [59, 25, 28, 58, 57, 34], 2, 1),
    (14, "r7", [36, 12, 16], 3, 2),
    (18, "r8", [2, 12, 21, 12], 6, 0),
    (18, "r9", [33, 33, 24], 3, 2),
)


def run_workload(
    requests,
    *,
    mode="refill",
    preemption=True,
    wavefront=True,
    asynchronous=False,
    delayed=False,
):
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(40, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            mode=mode,
            policy="priority",
            enable_preemption=preemption,
            max_num_seqs=4,
            max_num_batched_tokens=6,
            prefill_chunk_size=4,
            wavefront_chunked_prefill=wavefront,
        ),
        exit_config=ExitConfig("ouro_delayed" if delayed else "ouro"),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
    )
    arrivals = defaultdict(list)
    for step, request_id, prompt, max_tokens, priority in requests:
        arrivals[step].append((request_id, prompt, max_tokens, priority))
    finished = {}
    engine_steps = 0
    try:
        for step in range(500):
            for request_id, prompt, max_tokens, priority in arrivals[step]:
                engine.add_request(
                    request_id,
                    prompt,
                    SamplingParams(
                        max_tokens=max_tokens,
                        ignore_eos=True,
                        priority=priority,
                        max_loops=4,
                        exit_threshold=0.6,
                    ),
                )
            if not engine.has_unfinished_requests():
                if step > requests[-1][0]:
                    break
                continue
            outputs = engine.step()
            engine_steps += 1
            for output in outputs:
                if output.finished:
                    finished[output.request_id] = (
                        output.token_ids,
                        output.exit_depths,
                        output.finish_reason,
                    )
        assert len(finished) == len(requests), (
            f"only {len(finished)}/{len(requests)} requests finished in 500 steps; "
            f"preemptions={engine.preemption.preemptions if engine.preemption else 0}"
        )
        assert set(finished) == {request_id for _, request_id, *_ in requests}
        assert not engine.has_unfinished_requests()
        assert engine.cache_manager.num_used_blocks == 0
        if engine.preemption is not None:
            assert not engine.preemption.snapshots
        return (
            finished,
            engine.preemption.preemptions if engine.preemption else 0,
            engine_steps,
        )
    finally:
        engine.close()


@pytest.mark.parametrize("requests", [REQUESTS, REVIEWER_REQUESTS], ids=["seed-8", "reviewer"])
@pytest.mark.parametrize("mode,wavefront", [("refill", True), ("mixed", True), ("mixed", False)])
def test_priority_staged_prefill_drains_under_cache_pressure(requests, mode, wavefront):
    reference, _, _ = run_workload(requests, preemption=False)
    actual, preemptions, engine_steps = run_workload(requests, mode=mode, wavefront=wavefront)
    assert preemptions > 0
    assert engine_steps < 500
    assert actual == reference


def test_priority_wavefront_prefill_drains_with_async_scheduling():
    reference, _, _ = run_workload(REQUESTS, preemption=False, delayed=True)
    actual, preemptions, engine_steps = run_workload(REQUESTS, asynchronous=True, delayed=True)
    assert preemptions > 0
    assert engine_steps < 500
    assert actual == reference


def test_resumption_counts_as_an_admission_bypass():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(24, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            policy="fcfs",
            enable_preemption=True,
            max_num_seqs=3,
            max_admission_bypasses=1,
            prefill_chunk_size=4,
        ),
    )
    try:
        engine.add_request("resumed", [2], SamplingParams(max_tokens=1, priority=10))
        engine.add_request("running-1", [1] * 4, SamplingParams(max_tokens=1))
        engine.add_request("running-2", [1] * 4, SamplingParams(max_tokens=1))
        engine.scheduler._admit()
        engine.add_request("long", [1] * 9, SamplingParams(max_tokens=1))
        assert engine.scheduler.preempt(engine.scheduler.requests["long"])

        engine.scheduler._admit()
        assert engine.scheduler.requests["long"].phase == RequestPhase.WAITING
        assert engine.scheduler.requests["long"].admission_bypasses == 1
        assert engine.scheduler.requests["resumed"].phase == RequestPhase.PREFILL

        engine.abort_request("running-1")
        engine.scheduler._admit()
        assert engine.scheduler.requests["long"].phase == RequestPhase.WAITING
        engine.abort_request("running-2")
        engine.scheduler._admit()
        assert engine.scheduler.requests["long"].phase == RequestPhase.PREFILL
    finally:
        engine.close()


def test_blocked_restoration_respects_the_bypass_limit():
    cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        num_blocks=16,
        block_size=2,
        max_loops=2,
        incremental_allocation=True,
    )
    scheduler = Scheduler(
        SchedulerConfig(
            policy="fcfs",
            max_num_seqs=4,
            max_admission_bypasses=1,
            prefill_chunk_size=4,
        ),
        cache,
    )
    scheduler.add_request(Request("blocked", [1, 2, 3, 4], SamplingParams(max_tokens=1)))
    scheduler.add_request(Request("fresh", [5], SamplingParams(max_tokens=1)))
    scheduler.requests["blocked"].admission_bypasses = 1
    scheduler.resume_callback = lambda request: False if request.request_id == "blocked" else None

    scheduler._admit()

    assert scheduler.requests["blocked"].phase == RequestPhase.WAITING
    assert scheduler.requests["blocked"].admission_bypasses == 1
    assert scheduler.requests["fresh"].phase == RequestPhase.WAITING
    assert cache.num_used_blocks == 0
