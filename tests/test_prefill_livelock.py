"""Regression coverage for priority admission under staged prefill pressure."""

from collections import defaultdict

import pytest

from tests.helpers import tiny_ouro_model
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
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
    requests, *, mode="refill", preemption=True, wavefront=True, asynchronous=False, delayed=False
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
        return finished, engine.preemption.preemptions if engine.preemption else 0, engine_steps
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


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_impossible_priority_preemption_preserves_prefill_progress(prefix_caching):
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(
            24, 2, incremental_allocation=True, enable_prefix_caching=prefix_caching
        ),
        scheduler_config=SchedulerConfig(
            mode="refill",
            policy="priority",
            enable_preemption=True,
            max_num_seqs=2,
            max_num_batched_tokens=8,
            prefill_chunk_size=4,
            wavefront_chunked_prefill=True,
        ),
    )
    try:
        for request_id in ("a", "b"):
            engine.add_request(request_id, [1, 2, 3, 4], SamplingParams(max_tokens=1, priority=2))
        engine.step()
        before = {
            request_id: (
                engine.scheduler.requests[request_id].phase,
                engine.scheduler.requests[request_id].num_preluded_tokens,
                [row[:] for row in engine.scheduler.requests[request_id].prefill_depth_ranges],
            )
            for request_id in ("a", "b")
        }
        engine.add_request("high", [1] * 10, SamplingParams(max_tokens=1, priority=0))

        for _ in range(3):
            engine.scheduler._admit()

        assert engine.preemption.preemptions == 0
        assert engine.scheduler.requests["high"].phase == RequestPhase.WAITING
        for request_id, (phase, preluded, ranges) in before.items():
            request = engine.scheduler.requests[request_id]
            assert request.phase == phase
            assert request.num_preluded_tokens == preluded
            assert request.prefill_depth_ranges == ranges

        finished = set()
        for _ in range(100):
            finished.update(output.request_id for output in engine.step() if output.finished)
            if not engine.has_unfinished_requests():
                break
        assert finished == {"a", "b", "high"}
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


def test_nonincremental_priority_preemption_reserves_output_capacity():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(32, 2, incremental_allocation=False),
        scheduler_config=SchedulerConfig(
            policy="priority", enable_preemption=True, max_num_seqs=2, prefill_chunk_size=4
        ),
    )
    try:
        for request_id in ("a", "b"):
            engine.add_request(request_id, [1] * 4, SamplingParams(max_tokens=1, priority=2))
        engine.scheduler._admit()
        engine.add_request("high", [2] * 10, SamplingParams(max_tokens=5, priority=0))

        engine.scheduler._admit()

        assert engine.preemption.preemptions == 0
        assert engine.scheduler.requests["high"].phase == RequestPhase.WAITING
        finished = set()
        for _ in range(100):
            finished.update(output.request_id for output in engine.step() if output.finished)
            if not engine.has_unfinished_requests():
                break
        assert finished == {"a", "b", "high"}
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


def test_impossible_priority_restoration_does_not_preempt():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(32, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            mode="refill",
            policy="priority",
            enable_preemption=True,
            max_num_seqs=3,
            max_num_batched_tokens=10,
            prefill_chunk_size=10,
            wavefront_chunked_prefill=True,
        ),
    )
    try:
        engine.add_request("high", [1] * 10, SamplingParams(max_tokens=1, priority=0))
        engine.step()
        assert engine.scheduler.preempt(Request("pressure", [1], SamplingParams(max_tokens=1)))
        for request_id, priority in (("a", -1), ("b", -1), ("c", 2)):
            engine.add_request(request_id, [2] * 4, SamplingParams(max_tokens=1, priority=priority))
        engine.scheduler._admit()
        phase = engine.scheduler.requests["c"].phase

        for _ in range(3):
            engine.scheduler._admit()

        assert engine.preemption.preemptions == 1
        assert engine.scheduler.requests["high"].phase == RequestPhase.WAITING
        assert engine.scheduler.requests["c"].phase == phase
    finally:
        engine.close()


def test_resident_priority_restoration_needs_no_new_kv():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(12, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            policy="priority",
            enable_preemption=True,
            max_num_seqs=2,
            prefill_chunk_size=2,
            wavefront_chunked_prefill=True,
        ),
    )
    try:
        engine.add_request("high", [1, 2], SamplingParams(max_tokens=1, priority=0))
        engine.add_request("low", [3, 4], SamplingParams(max_tokens=1, priority=2))
        engine.scheduler._admit()
        pressure = Request("pressure", [1], SamplingParams(max_tokens=1, priority=-1))
        assert engine.scheduler.preempt(pressure, priority_only=True, excluded=frozenset({"low"}))
        assert engine.preemption.snapshots["high"]["kv"] is None
        engine.add_request("other", [5, 6], SamplingParams(max_tokens=1, priority=-1))

        engine.scheduler._admit()

        assert engine.cache_manager.num_free_blocks == 0
        assert engine.scheduler.requests["high"].phase == RequestPhase.PREFILLING_PRELUDE
        assert engine.scheduler.requests["low"].phase == RequestPhase.WAITING
        assert engine.preemption.resumptions == 1
    finally:
        engine.close()


def test_offloaded_priority_restoration_can_reclaim_victim_kv():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(20, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            policy="priority", enable_preemption=True, max_num_seqs=2, prefill_chunk_size=4
        ),
    )
    try:
        engine.add_request("high", [1] * 4, SamplingParams(max_tokens=1, priority=0))
        engine.add_request("low", [2] * 4, SamplingParams(max_tokens=1, priority=2))
        engine.scheduler._admit()
        pressure = Request("pressure", [1], SamplingParams(max_tokens=1, priority=-2))
        assert engine.scheduler.preempt(pressure, excluded=frozenset({"low"}))
        assert engine.preemption.snapshots["high"]["kv"] is not None
        engine.add_request("other", [3] * 4, SamplingParams(max_tokens=1, priority=-1))

        engine.scheduler._admit()

        assert engine.scheduler.requests["high"].phase == RequestPhase.PREFILL
        assert engine.scheduler.requests["low"].phase == RequestPhase.WAITING
        assert engine.preemption.snapshots["low"]["kv"] is not None
        assert engine.preemption.resumptions == 1
    finally:
        engine.close()


def test_priority_preemption_can_reclaim_waiting_resident_kv():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(24, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            policy="priority", enable_preemption=True, max_num_seqs=2, prefill_chunk_size=4
        ),
    )
    try:
        engine.add_request("active", [1] * 4, SamplingParams(max_tokens=1, priority=2))
        engine.add_request("resident", [2] * 4, SamplingParams(max_tokens=1, priority=3))
        engine.scheduler._admit()
        pressure = Request("pressure", [1], SamplingParams(max_tokens=1, priority=-2))
        assert engine.scheduler.preempt(
            pressure, priority_only=True, excluded=frozenset({"active"})
        )
        engine.add_request("other", [3] * 2, SamplingParams(max_tokens=1, priority=-1))
        engine.scheduler._admit()
        engine.add_request("high", [4] * 8, SamplingParams(max_tokens=1, priority=0))

        engine.scheduler._admit()

        assert engine.scheduler.requests["high"].phase == RequestPhase.PREFILL
        assert engine.preemption.snapshots["resident"]["kv"] is not None
        assert engine.preemption.snapshots["active"]["kv"] is not None
        assert engine.preemption.preemptions == 2
    finally:
        engine.close()


def test_priority_preemption_tries_next_victim_if_first_cannot_make_room():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(20, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            policy="priority", enable_preemption=True, max_num_seqs=2, prefill_chunk_size=2
        ),
    )
    try:
        engine.add_request("long", [1] * 8, SamplingParams(max_tokens=1, priority=2))
        engine.add_request("short", [2] * 2, SamplingParams(max_tokens=1, priority=3))
        engine.scheduler._admit()
        engine.add_request("high", [3] * 8, SamplingParams(max_tokens=1, priority=0))

        engine.scheduler._admit()

        assert engine.scheduler.requests["long"].phase == RequestPhase.WAITING
        assert engine.scheduler.requests["short"].phase == RequestPhase.PREFILL
        assert engine.scheduler.requests["high"].phase == RequestPhase.PREFILL
        assert engine.preemption.preemptions == 1
    finally:
        engine.close()


def test_priority_preemption_waives_watermark_when_victim_is_only_active_request():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(16, 2, incremental_allocation=True, watermark_ratio=0.25),
        scheduler_config=SchedulerConfig(
            policy="priority", enable_preemption=True, max_num_seqs=1, prefill_chunk_size=4
        ),
    )
    try:
        engine.add_request("low", [1] * 4, SamplingParams(max_tokens=1, priority=1))
        engine.scheduler._admit()
        engine.add_request("high", [2] * 8, SamplingParams(max_tokens=1, priority=0))

        engine.scheduler._admit()

        assert engine.scheduler.requests["low"].phase == RequestPhase.WAITING
        assert engine.scheduler.requests["high"].phase == RequestPhase.PREFILL
        assert engine.preemption.preemptions == 1
    finally:
        engine.close()


def test_priority_preemption_can_reuse_shared_prefix():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(20, 2, incremental_allocation=True, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            policy="priority", enable_preemption=True, max_num_seqs=2, prefill_chunk_size=4
        ),
    )
    try:
        cache = engine.cache_manager
        assert cache.allocate("seed", 5)
        cache.mark_imported_prefix("seed", 5)
        cache.publish_prefix("seed", [1, 2, 3, 4, 5], 5)
        cache.free("seed")
        engine.add_request("low", [1, 2, 3, 4, 7], SamplingParams(max_tokens=1, priority=2))
        engine.add_request("other", [8, 9], SamplingParams(max_tokens=1, priority=-1))
        engine.scheduler._admit()
        engine.add_request("high", [1, 2, 3, 4, 6], SamplingParams(max_tokens=1, priority=0))

        engine.scheduler._admit()

        assert engine.scheduler.requests["low"].phase == RequestPhase.WAITING
        assert engine.scheduler.requests["high"].phase == RequestPhase.PREFILL
        assert engine.scheduler.requests["high"].num_prefilled_tokens == 4
        assert engine.preemption.snapshots["low"]["kv"] is None
    finally:
        engine.close()


def test_cache_only_prefix_claim_prevents_futile_priority_preemption():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(24, 2, incremental_allocation=True, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            policy="priority", enable_preemption=True, max_num_seqs=2, prefill_chunk_size=4
        ),
    )
    try:
        cache = engine.cache_manager
        assert cache.allocate("seed", 5)
        cache.mark_imported_prefix("seed", 5)
        cache.publish_prefix("seed", [1, 2, 3, 4, 5], 5)
        cache.free("seed")
        engine.add_request("a", [6] * 4, SamplingParams(max_tokens=1, priority=2))
        engine.add_request("b", [7] * 4, SamplingParams(max_tokens=1, priority=2))
        engine.scheduler._admit()
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        engine.add_request("high", prompt, SamplingParams(max_tokens=1, priority=0))

        assert cache.prefix_admission_bounds(prompt) == (8, 8)
        engine.scheduler._admit()

        assert engine.preemption.preemptions == 0
        assert engine.scheduler.requests["a"].phase == RequestPhase.PREFILL
        assert engine.scheduler.requests["b"].phase == RequestPhase.PREFILL
        assert engine.scheduler.requests["high"].phase == RequestPhase.WAITING
    finally:
        engine.close()


def test_priority_preemption_avoids_victim_that_would_orphan_shared_prefix():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(24, 2, incremental_allocation=True, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            mode="refill",
            policy="priority",
            enable_preemption=True,
            max_num_seqs=2,
            max_num_batched_tokens=8,
            prefill_chunk_size=4,
            wavefront_chunked_prefill=True,
        ),
    )
    try:
        cache = engine.cache_manager
        assert cache.allocate("seed", 5)
        cache.mark_imported_prefix("seed", 5)
        cache.publish_prefix("seed", [1, 2, 3, 4, 5], 5)
        cache.free("seed")
        engine.add_request("a", [1, 2, 3, 4, 6], SamplingParams(max_tokens=1, priority=2))
        engine.add_request("b", [7] * 4, SamplingParams(max_tokens=1, priority=2))
        engine.scheduler._admit()
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        engine.add_request("high", prompt, SamplingParams(max_tokens=1, priority=0))

        assert cache.prefix_admission_bounds(prompt) == (8, 0)
        assert cache.prefix_admission_bounds(prompt, release_requests=["a"]) == (8, 8)
        engine.scheduler._admit()
        assert engine.preemption.preemptions == 1
        assert engine.scheduler.requests["a"].phase == RequestPhase.PREFILLING_PRELUDE
        assert engine.scheduler.requests["b"].phase == RequestPhase.WAITING
        assert engine.scheduler.requests["high"].phase == RequestPhase.PREFILLING_PRELUDE

        finished = set()
        for _ in range(100):
            finished.update(output.request_id for output in engine.step() if output.finished)
            if not engine.has_unfinished_requests():
                break
        assert finished == {"a", "b", "high"}
        assert cache.num_used_blocks == 0
    finally:
        engine.close()


def test_failed_priority_admission_does_not_repeat_preemption():
    class PendingEvent:
        def query(self):
            return False

    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(24, 2, incremental_allocation=True, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            mode="refill",
            policy="priority",
            enable_preemption=True,
            max_num_seqs=2,
            max_num_batched_tokens=8,
            prefill_chunk_size=4,
            wavefront_chunked_prefill=True,
        ),
    )
    try:
        engine.add_request("a", [1] * 4, SamplingParams(max_tokens=1, priority=2))
        engine.add_request("b", [2] * 4, SamplingParams(max_tokens=1, priority=2))
        engine.step()
        engine.cache_manager.publish_prefix("a", [1] * 4, 4, event=PendingEvent())
        engine.add_request("high", [1] * 10, SamplingParams(max_tokens=1, priority=0))

        engine.scheduler._admit()
        assert engine.preemption.preemptions == 1
        assert engine.scheduler.requests["high"].phase == RequestPhase.WAITING
        assert engine.scheduler.requests["b"].phase == RequestPhase.PREFILLING_RECURRENT
        for _ in range(5):
            engine.scheduler._admit()
        assert engine.preemption.preemptions == 1
        assert engine.scheduler.requests["b"].phase == RequestPhase.PREFILLING_RECURRENT

        finished = set()
        for _ in range(100):
            for output in engine.step():
                if output.finished:
                    finished.add(output.request_id)
            if not engine.has_unfinished_requests():
                break
        assert finished == {"a", "b", "high"}
        assert not engine.scheduler._blocked_priority_preemptions
        assert not engine.preemption.snapshots
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


def test_failed_priority_restoration_does_not_repeat_preemption(monkeypatch):
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(40, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            mode="refill",
            policy="priority",
            enable_preemption=True,
            max_num_seqs=3,
            max_num_batched_tokens=10,
            prefill_chunk_size=10,
            wavefront_chunked_prefill=True,
        ),
    )
    try:
        engine.add_request("high", [1] * 10, SamplingParams(max_tokens=1, priority=0))
        engine.step()
        assert engine.scheduler.preempt(Request("pressure", [1], SamplingParams(max_tokens=1)))
        resume = engine.scheduler.resume_callback
        monkeypatch.setattr(
            engine.scheduler,
            "resume_callback",
            lambda request: False if request.request_id == "high" else resume(request),
        )
        for request_id, priority in (("a", -1), ("b", -1), ("c", 2)):
            engine.add_request(request_id, [2] * 4, SamplingParams(max_tokens=1, priority=priority))

        engine.scheduler._admit()
        assert engine.scheduler.requests["high"].phase == RequestPhase.WAITING
        assert engine.scheduler.requests["high"].admission_bypasses == 1
        engine.scheduler._admit()
        assert engine.preemption.preemptions == 2
        assert engine.scheduler.requests["high"].phase == RequestPhase.WAITING
        for _ in range(5):
            engine.scheduler._admit()
        assert engine.preemption.preemptions == 2

        monkeypatch.setattr(engine.scheduler, "resume_callback", resume)
        finished = set()
        for _ in range(100):
            for output in engine.step():
                if output.finished:
                    finished.add(output.request_id)
            if not engine.has_unfinished_requests():
                break
        assert finished == {"a", "b", "c", "high"}
        assert not engine.scheduler._blocked_priority_preemptions
        assert not engine.preemption.snapshots
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


def test_priority_admission_retries_after_capacity_improves():
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(40, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            mode="refill",
            policy="priority",
            enable_preemption=True,
            max_num_seqs=2,
            max_num_batched_tokens=8,
            prefill_chunk_size=4,
            wavefront_chunked_prefill=True,
        ),
    )
    try:
        assert engine.cache_manager.allocate("external", 8)
        for request_id in ("a", "b"):
            engine.add_request(request_id, [1, 2, 3, 4], SamplingParams(max_tokens=1, priority=2))
        engine.step()
        engine.add_request("high", [1] * 16, SamplingParams(max_tokens=1, priority=0))
        engine.scheduler._admit()
        assert engine.preemption.preemptions == 0
        assert engine.scheduler.requests["high"].phase == RequestPhase.WAITING
        engine.scheduler._admit()
        assert all(
            engine.scheduler.requests[request_id].phase != RequestPhase.WAITING
            for request_id in ("a", "b")
        )

        engine.cache_manager.free("external")
        engine.scheduler._admit()
        assert engine.preemption.preemptions == 1
        assert engine.scheduler.requests["high"].phase == RequestPhase.PREFILLING_PRELUDE
    finally:
        engine.close()


def test_abort_clears_failed_priority_admission_for_reused_id(monkeypatch):
    engine = StageEngine(
        tiny_ouro_model(),
        attention_backend="torch",
        cache_config=CacheConfig(24, 2, incremental_allocation=True, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            policy="priority",
            enable_preemption=True,
            max_num_seqs=2,
            max_num_batched_tokens=8,
            prefill_chunk_size=4,
            wavefront_chunked_prefill=True,
        ),
    )
    try:
        monkeypatch.setattr(
            engine.cache_manager,
            "prefix_admission_bounds",
            lambda tokens, **kwargs: (((len(tokens) - 1) // 2) * 4, 0),
        )
        for request_id in ("a", "b"):
            engine.add_request(request_id, [1, 2, 3, 4], SamplingParams(max_tokens=1, priority=2))
        engine.step()
        engine.add_request("high", [1] * 10, SamplingParams(max_tokens=1, priority=0))
        engine.scheduler._admit()
        assert "high" in engine.scheduler._blocked_priority_preemptions

        engine.abort_request("high")
        assert not engine.scheduler._blocked_priority_preemptions
        engine.add_request("high", [3], SamplingParams(max_tokens=1, priority=0))
        engine.scheduler._admit()
        assert engine.scheduler.requests["high"].phase == RequestPhase.PREFILLING_PRELUDE
    finally:
        engine.close()


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
