"""Independent readout oracles and request-lifecycle checks for Ouro LoopCD."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from tests.helpers import tiny_ouro_config
from tests.reference import dense_reference
from vllm_rlt import CacheConfig, ExecutionConfig, LoopCDParams, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request, Stage
from vllm_rlt.worker.loopcd import Reference, extrapolate, reserved_bytes


def model_for():
    torch.manual_seed(42)
    return OuroForCausalLM(tiny_ouro_config()).eval()


def params_for(**kwargs):
    return SamplingParams(
        min_loops=4, max_loops=4, max_tokens=3, ignore_eos=True, loopcd=LoopCDParams(), **kwargs
    )


def engine_for(model, chunk=4, **kwargs):
    return LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=256, block_size=2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=8, prefill_chunk_size=chunk
        ),
        execution_config=ExecutionConfig(loopcd=True),
        **kwargs,
    )


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 3e-14), (torch.float32, 8e-6)])
@pytest.mark.parametrize("strength", [0, 0.3, 0.5, 2.0])
def test_affine_logits_folding_against_two_projections(dtype, tolerance, strength):
    generator = torch.Generator().manual_seed(17)
    final, reference = [torch.randn(8, 64, generator=generator, dtype=dtype) for _ in range(2)]
    weight = torch.randn(257, 64, generator=generator, dtype=dtype)
    bias = torch.randn(257, generator=generator, dtype=dtype)
    expected = (1 + strength) * F.linear(final, weight, bias) - strength * F.linear(
        reference, weight, bias
    )
    actual = F.linear(extrapolate(final, reference, strength), weight, bias)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "values",
    [
        {"reference_loop": 0},
        {"reference_loop": True},
        {"reference_loop": None},
        {"prefill_reference_loop": 0},
        {"strength": -1},
        {"strength": float("nan")},
        {"strength": float("inf")},
        {"strength_mode": "margin"},
        {"mode": "hidden"},
        {"implementation": "unknown"},
    ],
)
def test_invalid_config(values):
    with pytest.raises(ValueError):
        LoopCDParams(**values)


@pytest.mark.parametrize("enabled", [False, True])
def test_zero_strength_matches_unguided_readout_without_references(enabled):
    model = model_for()
    results = []
    for cd in (None, LoopCDParams(strength=0)):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=256, block_size=2),
            execution_config=ExecutionConfig(loopcd=enabled),
        )
        engine.add_request("a", [4, 7, 3], replace(params_for(), loopcd=cd))
        while engine.has_unfinished_requests():
            output = engine.step()
        results.append(output[-1].token_ids)
        assert engine.model_runner.loopcd_stats["captures"] == 0
        assert not engine.model_runner.loopcd_references
        assert engine.cache_manager.num_used_blocks == 0
    assert results[0] == results[1]


@pytest.mark.parametrize("prefill,decode", [(4, 4), (4, 2), (3, 2)])
@pytest.mark.parametrize("implementation", ["two_head", "linear_fused"])
@pytest.mark.parametrize("chunk", [1, 3, 8])
@torch.inference_mode()
def test_first_readout_matches_independent_dense_oracle(prefill, decode, implementation, chunk):
    model = model_for()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=256, block_size=2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=8, prefill_chunk_size=chunk
        ),
        execution_config=ExecutionConfig(loopcd=True, prefill_depth=prefill),
    )
    prompt = torch.tensor([4, 7, 9, 3, 12])
    params = replace(
        params_for(),
        min_loops=decode,
        max_loops=decode,
        loopcd=LoopCDParams(prefill_reference_loop=2, implementation=implementation),
    )
    depths = dense_reference(model, prompt, prefill)
    expected = extrapolate(depths[-1][2][-1], depths[1][2][-1], 0.3)
    engine.add_request("a", prompt.tolist(), params)
    request = engine.scheduler.requests["a"]
    while request.stage != Stage.CODA:
        engine.step()
    torch.testing.assert_close(request.hidden_state, depths[-1][0][-1], atol=4e-6, rtol=4e-5)
    ref = engine.model_runner.loopcd_references["a"]
    ref.check(request, 4, 0, 2)
    raw = request.hidden_state.clone()
    seen = []

    def sample(logits, request):
        seen.append(logits.clone())
        return logits.argmax()

    engine.model_runner._sample_tensor = sample
    engine.step()
    torch.testing.assert_close(seen[0], expected, atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(request.hidden_state, raw, atol=0, rtol=0)
    while engine.has_unfinished_requests():
        output = engine.step()
    assert output[-1].exit_depths == [prefill, decode, decode]
    assert not engine.model_runner.loopcd_references
    assert engine.cache_manager.num_used_blocks == 0


def test_refill_decode_abort_and_request_id_reuse():
    engine = engine_for(model_for(), 2)
    params = params_for()
    engine.add_request("a", [4, 7, 3], params)
    engine.add_request("b", [9, 2], params)
    while not engine.model_runner.loopcd_references:
        engine.step()
    old = engine.scheduler.requests["a"]
    engine.abort_request("a")
    engine.add_request("a", [12, 3, 7, 9], params)
    assert engine.scheduler.requests["a"] is not old
    completed = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            for ref in engine.model_runner.loopcd_references.values():
                assert ref.owner is engine.scheduler.requests[ref.owner.request_id]
            if output.finished:
                completed[output.request_id] = output
    assert set(completed) == {"a", "b"}
    assert all(x.exit_depths == [4, 4, 4] for x in completed.values())
    assert engine.model_runner.loopcd_stats["guided_rows"] > 0
    assert not engine.model_runner.loopcd_references
    assert engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("field", ["owner", "position", "output_index", "loop"])
def test_stale_reference_rejected(field):
    request = Request("a", [1, 2], SamplingParams())
    ref = Reference(request, 1, 0, 1, torch.ones(3))
    ref.check(request, 1, 0, 1)
    value = Request("a", [1, 2], SamplingParams()) if field == "owner" else 5
    with pytest.raises(RuntimeError, match="mismatch"):
        replace(ref, **{field: value}).check(request, 1, 0, 1)


@pytest.mark.parametrize("case", ["unreserved", "reference", "early_exit", "prefix"])
def test_unsupported_admission_rejected_before_request_allocation(case):
    engine = LLMEngine(
        model_for(),
        cache_config=CacheConfig(
            num_blocks=256, block_size=2, enable_prefix_caching=case == "prefix"
        ),
        execution_config=ExecutionConfig(loopcd=case != "unreserved"),
    )
    params = params_for()
    if case == "reference":
        params = replace(params, loopcd=replace(params.loopcd, reference_loop=4))
    elif case == "early_exit":
        params = replace(params, min_loops=2)
    with pytest.raises(ValueError, match="LoopCD"):
        engine.add_request("a", [4, 7], params)
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.scheduler.requests


@pytest.mark.parametrize("feature", ["async", "prefix", "shared"])
def test_shallow_prefill_capability_checked_before_allocation(feature):
    with pytest.raises(ValueError, match="prefill_depth override"):
        LLMEngine(
            model_for(),
            execution_config=ExecutionConfig(prefill_depth=2, async_scheduling=feature == "async"),
            cache_config=CacheConfig(
                enable_prefix_caching=feature == "prefix",
                layout="shared" if feature == "shared" else "last_exited",
            ),
        )


def test_reference_and_two_head_scratch_budgeted_before_auto_kv():
    model = model_for()
    scheduler = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=8)
    assert reserved_bytes(model.config, scheduler, ExecutionConfig(), 2) == 0
    assert reserved_bytes(model.config, scheduler, ExecutionConfig(loopcd=True), 2) == (
        (4 + 3 * 8) * model.config.hidden_size * 2 + 3 * 8 * model.config.vocab_size * 4
    )


@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize(
    "strength,implementation",
    [
        (0.0, "two_head"),
        (0.3, "two_head"),
        (0.3, "linear_fused"),
        (0.5, "two_head"),
        (0.5, "linear_fused"),
    ],
)
@torch.inference_mode()
def test_teacher_forced_continuation_against_dense_oracle(depth, strength, implementation):
    model = model_for()
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=16, prefill_chunk_size=3
        ),
        execution_config=ExecutionConfig(loopcd=True, prefill_depth=depth),
    )
    sampling = replace(
        params_for(),
        min_loops=depth,
        max_loops=depth,
        loopcd=LoopCDParams(strength=strength, implementation=implementation),
    )
    prompt, candidate = [4, 7, 9, 3, 12], [17, 8, 2]
    seen, owners = [], []

    def observe(logits, request):
        index = len(request.generated_token_ids)
        if index == 0:
            owners.append(request)
        prefix = torch.tensor(prompt + request.generated_token_ids)
        dense = dense_reference(model, prefix, depth)
        expected = extrapolate(dense[-1][2][-1], dense[0][2][-1], strength)
        torch.testing.assert_close(request.hidden_state, dense[-1][0][-1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)
        selected = candidate[index]
        torch.testing.assert_close(
            logits.float().log_softmax(-1)[selected],
            expected.float().log_softmax(-1)[selected],
            atol=1e-5,
            rtol=1e-5,
        )
        assert logits.argmax() == expected.argmax()
        seen.append(index)
        return torch.tensor(selected)

    engine.model_runner._sample_tensor = observe
    for run in range(2):
        if run:
            engine.add_request("a", [12, 3, 7], sampling)
            interrupted = engine.scheduler.requests["a"]
            while interrupted.stage != Stage.CODA:
                engine.step()
            engine.abort_request("a")
        engine.add_request("a", prompt, sampling)
        while engine.has_unfinished_requests():
            output = engine.step()
        assert output[-1].token_ids == candidate
        assert output[-1].exit_depths == [depth] * 3
        assert output[-1].finish_reason == "length"
        assert not engine.model_runner.loopcd_references
        assert engine.cache_manager.num_used_blocks == 0
    assert seen == [0, 1, 2] * 2 and owners[0] is not owners[1]
    assert not torch.cuda.is_initialized()
