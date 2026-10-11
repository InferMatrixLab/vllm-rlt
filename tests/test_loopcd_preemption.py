"""LoopCD state preservation through synchronous Ouro preemption."""

from dataclasses import replace

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    LoopCDParams,
    SamplingParams,
    SchedulerConfig,
)
from vllm_rlt.config import ExitConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request, Stage
from vllm_rlt.worker.loopcd import validate

PROMPT = [4, 7, 9, 3, 12]
BOUNDARIES = [
    ("partial_prefill", False),
    ("first_coda", True),
    ("next_prelude", False),
    ("before_reference", False),
    ("at_reference", True),
    ("after_reference", True),
    ("decode_coda", True),
]


def model_for():
    torch.manual_seed(42)
    return OuroForCausalLM(tiny_ouro_config()).eval()


def gpu_model_for():
    torch.manual_seed(42)
    return OuroForCausalLM(tiny_ouro_config(head_dim=64)).to(device="cuda").eval()


def params_for(strength=0.3, decode=4, implementation="two_head"):
    return SamplingParams(
        min_loops=decode,
        max_loops=decode,
        max_tokens=4,
        ignore_eos=True,
        temperature=0.8,
        seed=17,
        loopcd=LoopCDParams(
            reference_loop=1 if decode == 2 else 2,
            prefill_reference_loop=2,
            strength=strength,
            implementation=implementation,
        ),
    )


def engine_for(model, mode="refill", attention_backend="torch", static_buffers=False):
    return LLMEngine(
        model,
        attention_backend=attention_backend,
        cache_config=CacheConfig(num_blocks=128, block_size=2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            prefill_chunk_size=2,
            enable_preemption=True,
            mode=mode,
        ),
        execution_config=ExecutionConfig(
            loopcd=True, prefill_depth=4, static_buffers=static_buffers
        ),
    )


def at_boundary(request, boundary):
    if boundary == "partial_prefill":
        return request.stage == Stage.PREFILL and request.num_prefilled_tokens == 2
    if boundary == "first_coda":
        return request.stage == Stage.CODA and request.num_scheduled_outputs == 0
    if request.num_scheduled_outputs != 1:
        return False
    if boundary == "next_prelude":
        return request.stage == Stage.PRELUDE
    if boundary == "decode_coda":
        return request.stage == Stage.CODA
    reference_loop = request.sampling_params.loopcd.reference_loop
    depth = {
        "before_reference": reference_loop - 1,
        "at_reference": reference_loop,
        "after_reference": reference_loop + 1,
    }[boundary]
    return request.stage == Stage.RECURRENT and request.loops_done == depth


def advance_to(engine, boundary):
    request = engine.scheduler.requests["a"]
    for _ in range(100):
        if at_boundary(request, boundary):
            return request
        engine.step()
    pytest.fail(f"boundary not reached: {boundary}")


def suspend(engine):
    # Exercise the existing victim-selection and WAITING/resume path at a
    # completed scheduler boundary, without changing subsequent batch shape.
    requester = Request("pressure", [2], SamplingParams())
    assert engine.preemption.preempt(requester)
    request = engine.scheduler.requests["a"]
    assert request.stage == Stage.WAITING
    assert request.hidden_state is None
    assert "a" not in engine.model_runner.loopcd_references
    assert engine.cache_manager.num_used_blocks == 0
    return engine.preemption.snapshots["a"]


def drain(engine):
    result = None
    for _ in range(100):
        if not engine.has_unfinished_requests():
            assert result is not None
            assert not engine.preemption.snapshots
            assert not engine.model_runner.loopcd_references
            assert engine.cache_manager.num_used_blocks == 0
            return result
        for output in engine.step():
            if output.finished:
                result = output
    pytest.fail("engine did not drain")


@pytest.mark.parametrize("boundary,has_reference", BOUNDARIES)
@pytest.mark.parametrize("mode", ["refill", "no_refill"])
@pytest.mark.parametrize("implementation", ["two_head", "linear_fused"])
@torch.inference_mode()
def test_suspend_resume_preserves_all_logits_rng_and_recurrent_work(
    boundary, has_reference, mode, implementation
):
    model = model_for()
    runs = []
    for preempt in (False, True):
        engine = engine_for(model, mode)
        seen = []
        sample = engine.model_runner._sample_tensor

        def observe(logits, request):
            seen.append(logits.clone())
            return sample(logits, request)

        engine.model_runner._sample_tensor = observe
        engine.add_request("a", PROMPT, params_for(implementation=implementation))
        request = advance_to(engine, boundary)
        if preempt:
            old = engine.model_runner.loopcd_references.get("a")
            state = suspend(engine)
            reference = state["loopcd_reference"]
            assert (reference is not None) == has_reference
            if reference is not None:
                assert reference.owner is request
                assert reference.hidden.device.type == "cpu"
                assert reference.hidden.data_ptr() != old.hidden.data_ptr()
                torch.testing.assert_close(reference.hidden, old.hidden, atol=0, rtol=0)
            engine.step()  # Native scheduler callback resumes the saved stage.
            assert engine.preemption.resumptions == 1
            assert not engine.preemption.snapshots
        output = drain(engine)
        runs.append((output, torch.stack(seen), engine.model_runner.loopcd_stats))
    assert runs[0][0].token_ids == runs[1][0].token_ids
    assert runs[0][0].exit_depths == runs[1][0].exit_depths == [4, 4, 4, 4]
    torch.testing.assert_close(runs[0][1], runs[1][1], atol=0, rtol=0)
    for key in ("captures", "prefill_core_rows", "decode_core_rows", "guided_rows"):
        assert runs[0][2][key] == runs[1][2][key]


@pytest.mark.parametrize("strength", [0.0, 0.3])
@torch.inference_mode()
def test_partial_decode_depth_and_repeated_preemption(strength):
    model = model_for()
    outputs = []
    for preempt in (False, True):
        engine = engine_for(model)
        engine.add_request("a", PROMPT, params_for(strength=strength, decode=2))
        if preempt:
            for boundary in ("partial_prefill", "first_coda", "at_reference", "decode_coda"):
                advance_to(engine, boundary)
                state = suspend(engine)
                if strength == 0:
                    assert state["loopcd_reference"] is None
                engine.step()
            assert engine.preemption.preemptions == engine.preemption.resumptions == 4
        output = drain(engine)
        outputs.append(output)
    assert outputs[0].token_ids == outputs[1].token_ids
    assert outputs[0].exit_depths == outputs[1].exit_depths == [4, 2, 2, 2]


@torch.inference_mode()
def test_blocked_restore_keeps_cpu_reference_until_capacity_returns(monkeypatch):
    engine = engine_for(model_for())
    engine.add_request("a", PROMPT, params_for())
    request = advance_to(engine, "after_reference")
    state = suspend(engine)
    allocate = engine.cache_manager.allocate
    with monkeypatch.context() as patch:
        patch.setattr(engine.cache_manager, "allocate", lambda *args, **kwargs: False)
        assert engine.preemption.resume(request) is False
    assert engine.preemption.snapshots["a"] is state
    assert state["loopcd_reference"].owner is request
    assert "a" not in engine.model_runner.loopcd_references
    assert request.hidden_state is None
    assert engine.cache_manager.num_used_blocks == 0
    assert engine.cache_manager.allocate == allocate
    drain(engine)
    assert engine.preemption.resumptions == 1


@pytest.mark.parametrize("field", ["owner", "position", "output_index", "loop", "missing"])
@torch.inference_mode()
def test_corrupt_suspended_reference_rejected_before_kv_allocation(field):
    engine = engine_for(model_for())
    engine.add_request("a", PROMPT, params_for())
    request = advance_to(engine, "after_reference")
    state = suspend(engine)
    reference = state["loopcd_reference"]
    value = Request("a", PROMPT, params_for()) if field == "owner" else 999
    state["loopcd_reference"] = None if field == "missing" else replace(reference, **{field: value})
    with pytest.raises(RuntimeError, match="LoopCD reference"):
        engine.preemption.resume(request)
    assert engine.cache_manager.num_used_blocks == 0
    assert "a" not in engine.model_runner.loopcd_references
    engine.abort_request("a")
    assert not engine.preemption.snapshots
    assert not engine.has_unfinished_requests()


@torch.inference_mode()
def test_abort_suspended_request_and_id_reuse_cannot_inherit_reference():
    model = model_for()
    engine = engine_for(model)
    engine.add_request("a", PROMPT, params_for())
    old = advance_to(engine, "after_reference")
    suspend(engine)
    engine.abort_request("a")
    assert not engine.preemption.snapshots
    assert not engine.model_runner.loopcd_references
    assert engine.cache_manager.num_used_blocks == 0
    engine.add_request("a", [12, 3, 7], params_for())
    assert engine.scheduler.requests["a"] is not old
    reused = drain(engine)
    baseline = engine_for(model)
    baseline.add_request("a", [12, 3, 7], params_for())
    expected = drain(baseline)
    assert reused.token_ids == expected.token_ids
    assert reused.exit_depths == expected.exit_depths


def test_graph_preemption_stays_rejected_until_separate_qualification():
    with pytest.raises(ValueError, match="preemption currently requires eager"):
        validate(
            model_for(),
            params_for(),
            ExecutionConfig(loopcd=True, cuda_graphs=True),
            CacheConfig(),
            SchedulerConfig(enable_preemption=True),
            ExitConfig(),
            None,
        )


DEVICE_CASES = [
    pytest.param("cpu", False, id="cpu"),
    pytest.param("cuda", False, marks=pytest.mark.gpu, id="cuda-static-off"),
    pytest.param("cuda", True, marks=pytest.mark.gpu, id="cuda-static-on"),
]


@pytest.mark.parametrize("device,static_buffers", DEVICE_CASES)
@pytest.mark.parametrize("policy", ["priority", "pressure"])
@torch.inference_mode()
def test_guided_requests_survive_scheduler_driven_preemption(policy, device, static_buffers):
    if device == "cpu":
        model = model_for()
        attention_backend = "torch"
    else:
        model = gpu_model_for()
        attention_backend = "triton"
    engine = LLMEngine(
        model,
        attention_backend=attention_backend,
        cache_config=CacheConfig(
            num_blocks=64 if policy == "priority" else 24,
            block_size=2,
            incremental_allocation=policy == "pressure",
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1 if policy == "priority" else 3,
            max_num_batched_tokens=8,
            prefill_chunk_size=2,
            policy="priority" if policy == "priority" else "fcfs",
            enable_preemption=True,
        ),
        execution_config=ExecutionConfig(
            loopcd=True, prefill_depth=4, static_buffers=static_buffers
        ),
    )
    params = params_for()
    if policy == "priority":
        engine.add_request("a", PROMPT, replace(params, priority=10))
        advance_to(engine, "first_coda")
        engine.add_request("b", [9, 2], replace(params, priority=-10))
        expected_ids = {"a", "b"}
    else:
        for name in ("a", "b", "c"):
            engine.add_request(name, PROMPT, params)
        expected_ids = {"a", "b", "c"}
    completed = {}
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step():
            for reference in engine.model_runner.loopcd_references.values():
                assert reference.owner is engine.scheduler.requests[reference.owner.request_id]
            if output.finished:
                completed[output.request_id] = output
    assert set(completed) == expected_ids
    assert engine.preemption.preemptions > 0
    assert engine.preemption.preemptions == engine.preemption.resumptions
    assert not engine.preemption.snapshots
    assert not engine.model_runner.loopcd_references
    assert engine.cache_manager.num_used_blocks == 0
    if policy == "priority":
        assert list(completed) == ["b", "a"]
    baseline = engine_for(model, attention_backend=attention_backend, static_buffers=static_buffers)
    baseline.add_request("a", PROMPT, params)
    expected = drain(baseline)
    for name in ("a",) if policy == "priority" else ("a", "b", "c"):
        assert completed[name].token_ids == expected.token_ids
        assert completed[name].exit_depths == expected.exit_depths
