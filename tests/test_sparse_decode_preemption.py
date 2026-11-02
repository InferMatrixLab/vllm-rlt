"""Preemption of ordinary sparse decode: a suspended token keeps its reference."""

import pytest
import torch

from tests.test_sparse_decode_sync import _bounded_pool, _ordinary, engine, generate
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.attention.sparse import CAPTURE, CONSUME


def _scheduler(**options):
    values = dict(
        max_num_seqs=1, max_num_batched_tokens=16, prefill_chunk_size=3, enable_preemption=True
    )
    values.update(options)
    return SchedulerConfig(**values)


def _record_modes(e, monkeypatch):
    """Record (request, mode) of every sparse plan the runner builds."""
    seen = []
    build = e.reference_store.build_plan

    def spy(owners, modes, **kwargs):
        seen.append([(None if o is None else o[0].request_id, m) for o, m in zip(owners, modes)])
        return build(owners, modes, **kwargs)

    monkeypatch.setattr(e.reference_store, "build_plan", spy)
    return seen


def _drain(e, outputs):
    for _ in range(1000):
        if not e.has_unfinished_requests():
            return outputs
        for output in e.step():
            if output.finished:
                outputs[output.request_id] = list(output.token_ids)
    raise AssertionError("engine did not drain")


@pytest.mark.parametrize("mode", ["resident", "offload", "resident_then_offload"])
def test_suspension_after_capture_resumes_with_the_same_reference(mode, monkeypatch):
    victim = SamplingParams(max_tokens=8, ignore_eos=True, priority=1)
    arrival = SamplingParams(max_tokens=4, ignore_eos=True)
    results = []
    for preempt in (False, True):
        e = _ordinary(False, 0.5, scheduler_config=_scheduler())
        seen = _record_modes(e, monkeypatch)
        try:
            e.add_request("a", [2, 3, 4, 5], victim)
            outputs, mark = {}, None
            for _ in range(1000):
                for output in e.step():
                    if output.finished:
                        outputs[output.request_id] = list(output.token_ids)
                # Stop right after a capture of a generated token: its later
                # loops have not consumed the reference yet.
                if seen and ("a", CAPTURE) in seen[-1] and e.reference_store.outstanding == 1:
                    break
            assert "a" not in outputs
            e.add_request("b", [2, 3], arrival)
            if preempt:
                waiting = e.scheduler.requests["b"]
                assert e.scheduler.preempt(waiting, priority_only=mode != "offload")
                saved = e.preemption.snapshots["a"]["references"]
                assert saved is not None and e.reference_store.outstanding == 0
                if mode == "resident_then_offload":
                    assert saved[1].selected.device == e.cache_manager.device
                    assert e.scheduler.preempt(waiting, waiting_only=True)
                    saved = e.preemption.snapshots["a"]["references"]
                expected = e.cache_manager.device if mode == "resident" else torch.device("cpu")
                assert saved[1].selected.device == expected
                mark = len(seen)
            results.append(_drain(e, outputs))
            assert e.preemption.resumptions == int(preempt)
            assert not e.preemption.snapshots
            assert e.reference_store.outstanding == 0
            assert e.cache_manager.num_used_blocks == 0
            if preempt:
                # The resumed token consumes its restored reference, not dense.
                after = next(plan for plan in seen[mark:] if any(r == "a" for r, _ in plan))
                assert ("a", CONSUME) in after
        finally:
            e.close()
    assert results[0] == results[1]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_kv_pressure_preemption_matches_unpreempted_sparse(asynchronous, monkeypatch):
    def run(cache_blocks, preemption):
        e = _ordinary(
            asynchronous,
            0.5,
            cache_config=CacheConfig(cache_blocks, 2, incremental_allocation=True),
            scheduler_config=_scheduler(max_num_seqs=3, enable_preemption=preemption),
        )
        _bounded_pool(e, monkeypatch)
        snapshot = e.model_runner.snapshot_attention_reference
        carried = []

        def counted(request, device):
            saved = snapshot(request, device)
            carried.append(saved is not None)
            return saved

        monkeypatch.setattr(e.model_runner, "snapshot_attention_reference", counted)
        try:
            outputs = generate(e, count=3)
            return outputs, e.preemption.preemptions, sum(carried)
        finally:
            e.close()

    baseline, none, _ = run(512, False)
    pressured, preemptions, carried = run(40, True)
    assert none == 0 and preemptions > 0 and carried > 0
    assert pressured == baseline


def test_abort_while_suspended_frees_reference_and_id_is_reusable(monkeypatch):
    e = _ordinary(False, 0.5, scheduler_config=_scheduler())
    seen = _record_modes(e, monkeypatch)
    try:
        e.add_request("a", [2, 3, 4], SamplingParams(max_tokens=8, ignore_eos=True, priority=1))
        for _ in range(1000):
            e.step()
            if seen and ("a", CAPTURE) in seen[-1]:
                break
        e.add_request("b", [2, 3], SamplingParams(max_tokens=2, ignore_eos=True))
        assert e.scheduler.preempt(e.scheduler.requests["b"], priority_only=True)
        assert e.preemption.snapshots["a"]["references"] is not None
        e.abort_request("a")
        assert "a" not in e.preemption.snapshots
        _drain(e, {})
        assert e.reference_store.outstanding == 0
        assert generate(e, count=2)
    finally:
        e.close()


@pytest.mark.parametrize("draft", [1, 2])
def test_speculative_preemption_stays_rejected(draft):
    with pytest.raises(ValueError, match="preemption only for ordinary decode"):
        engine(
            speculative_config=SpeculativeConfig(2, draft_loops=draft),
            scheduler_config=SchedulerConfig(enable_preemption=True),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("implementation", ["batched", "fused"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_cuda_kv_pressure_preemption_matches_unpreempted(implementation, asynchronous):
    from vllm_rlt import SparseDecodeAttentionConfig

    def run(cache_blocks, preemption):
        e = _ordinary(
            asynchronous,
            None,
            device="cuda",
            cache_config=CacheConfig(cache_blocks, 2, incremental_allocation=True),
            scheduler_config=_scheduler(max_num_seqs=3, enable_preemption=preemption),
            sparse_decode_attention_config=SparseDecodeAttentionConfig(
                enabled=True, key_retention=0.5, implementation=implementation
            ),
        )
        try:
            return generate(e, count=3), e.preemption.preemptions
        finally:
            e.close()

    baseline, _ = run(512, False)
    pressured, preemptions = run(40, True)
    torch.cuda.synchronize()
    assert preemptions > 0
    assert pressured == baseline


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("offload", [False, True])
def test_cuda_pipeline_preemption_restores_each_rows_reference(graphs, offload):
    """A token pipeline suspends with rows past their capture loop and resumes them sparse."""
    from vllm_rlt import ExecutionConfig, SparseDecodeAttentionConfig

    def make():
        return engine(
            retention=None,
            device="cuda",
            speculative_config=SpeculativeConfig(4, draft_loops=1, mode="pipeline"),
            scheduler_config=_scheduler(policy="priority"),
            execution_config=ExecutionConfig(
                async_scheduling=True,
                cuda_graphs=graphs,
                static_buffers=True,
                pad_to_power_of_two=True,
            ),
            sparse_decode_attention_config=SparseDecodeAttentionConfig(
                enabled=True, key_retention=0.5, implementation="fused"
            ),
        )

    prompt = [1, 2, 3]
    low = SamplingParams(max_tokens=64, ignore_eos=True, priority=10)
    e = make()
    try:
        e.add_request("low", prompt, low)
        expected = _drain(e, {})["low"]
    finally:
        e.close()

    e = make()
    try:
        e.add_request("low", prompt, low)
        request = e.scheduler.requests["low"]
        runner = e.speculative_runner
        outputs, deep = {}, 0

        def step():
            for output in e.step():
                if output.finished:
                    outputs[output.request_id] = list(output.token_ids)

        # A random tiny model rejects most drafts, so a pipeline often restarts
        # from its root. Suspend repeatedly, one more submission later each time
        # so the suspension point walks through the pipeline's cycle, until one
        # suspension holds a row past its capture loop.
        for attempt in range(8):
            for _ in range(200):
                step()
                state = runner.states.get("low")
                if state is not None and state.submitted >= 2 + attempt:
                    break
            else:
                pytest.fail("the low request did not run pipeline steps")
            e.add_request(
                f"high{attempt}", [5], SamplingParams(max_tokens=3, ignore_eos=True, priority=0)
            )
            for _ in range(200):
                step()
                if "low" in e.preemption.snapshots:
                    break
            else:
                pytest.fail("priority arrival did not suspend the pipeline")
            saved = e.preemption.snapshots["low"]["pipeline"]
            references, permutation = saved.references
            assert sorted(permutation) == list(range(len(references)))
            assert "low" not in runner.states
            if offload:
                e.preemption.suspend(request, offload_kv=True)
                saved = e.preemption.snapshots["low"]["pipeline"]
                assert saved.references[0].selected.device.type == "cpu"
            deep += any(t.depth > 1 for t in saved.tokens)
            if deep:
                break
        assert deep, "no suspension held a row past its capture loop"
        assert "low" not in outputs
        outputs.update(_drain(e, {}))
        assert outputs["low"] == expected
        assert e.preemption.resumptions >= 1
        assert not e.preemption.snapshots
        assert e.reference_store.outstanding == 0
        assert e.cache_manager.num_used_blocks == 0
        if graphs:
            assert runner.graphs.replays > 0
    finally:
        e.close()
