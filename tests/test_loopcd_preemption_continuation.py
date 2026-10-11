"""Mixed-depth oracle checks across preemption and request-ID reuse."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from tests.reference.continuation import OuroContinuation
from tests.test_loopcd_preemption import suspend
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    LoopCDParams,
    SamplingParams,
    SchedulerConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Stage

PROMPT = [4, 7, 9, 3, 12]
CONTINUATION = [17, 8, 2, 11, 5, 7, 13, 6, 9]
REUSE_PROMPT = [2, 5, 8, 1]
REUSE_TOKENS = [11, 13]
PREFILL_DEPTH = 4
ATOL = RTOL = 1e-4


def _engine(model, attention_backend):
    return LLMEngine(
        model,
        attention_backend=attention_backend,
        cache_config=CacheConfig(num_blocks=128, block_size=2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            prefill_chunk_size=2,
            enable_preemption=True,
        ),
        execution_config=ExecutionConfig(loopcd=True, prefill_depth=PREFILL_DEPTH),
    )


def _params(depth, strength, max_tokens):
    return SamplingParams(
        max_tokens=max_tokens,
        min_loops=depth,
        max_loops=depth,
        ignore_eos=True,
        loopcd=LoopCDParams(strength=strength),
    )


def _plane_error(cache, oracle, request_id):
    """One gather per plane, not one sync per position."""
    device = cache.device
    positions = torch.arange(oracle.position, device=device)
    worst = 0.0
    for loop in range(cache.max_loops):
        table = torch.tensor(
            cache.get_block_table(request_id, loop), device=device, dtype=torch.long
        )
        blocks = table[positions // cache.block_size]
        offsets = positions % cache.block_size
        keys = cache.key_cache[blocks, :, offsets].permute(1, 0, 2, 3).float().cpu()
        values = cache.value_cache[blocks, :, offsets].permute(1, 0, 2, 3).float().cpu()
        expected_keys = (
            torch.stack([oracle.kv[(loop, layer)][0] for layer in range(cache.num_layers)])
            .float()
            .cpu()
        )
        expected_values = (
            torch.stack([oracle.kv[(loop, layer)][1] for layer in range(cache.num_layers)])
            .float()
            .cpu()
        )
        worst = max(
            worst,
            (keys - expected_keys).abs().max().item(),
            (values - expected_values).abs().max().item(),
        )
        torch.testing.assert_close(keys, expected_keys, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(values, expected_values, atol=ATOL, rtol=RTOL)
    return worst


def _check(engine, oracle, logits, request, expected, forced, strength, worst):
    """Assert independent state/KV/logits/logprob, recording maxima first."""
    state = request.hidden_state.float().cpu()
    final_hidden = expected[-1][0][-1].float().cpu()
    worst["state"] = max(worst["state"], (state - final_hidden).abs().max().item())
    torch.testing.assert_close(state, final_hidden, atol=ATOL, rtol=RTOL)
    worst["kv"] = max(worst["kv"], _plane_error(engine.cache_manager, oracle, request.request_id))
    if strength != 0:
        reference = engine.model_runner.loopcd_references.get(request.request_id)
        assert reference is not None
        assert reference.owner is request
        # The oracle gates the resulting readout; reference copy/restore is lossless.
        worst["reference"] = max(
            worst["reference"],
            (reference.hidden.float().cpu() - expected[0][0][-1].float().cpu()).abs().max().item(),
        )
    final_logits = expected[-1][1][-1].float().cpu()
    guided = final_logits + strength * (final_logits - expected[0][1][-1].float().cpu())
    row = logits.float().cpu()
    worst["logits"] = max(worst["logits"], (row - guided).abs().max().item())
    torch.testing.assert_close(row, guided, atol=ATOL, rtol=RTOL)
    chosen = row.log_softmax(-1)[forced]
    oracle_logp = guided.log_softmax(-1)[forced]
    worst["logprob"] = max(worst["logprob"], (chosen - oracle_logp).abs().item())
    torch.testing.assert_close(chosen, oracle_logp, atol=ATOL, rtol=RTOL)
    return torch.tensor(forced, device=logits.device)


def _expected(oracle, prompt, forced, index, depth, device):
    tokens = prompt if index == 0 else [forced[index - 1]]
    feed_depth = PREFILL_DEPTH if index == 0 else depth
    return oracle.forward(torch.tensor(tokens, device=device), feed_depth)


@torch.inference_mode()
def check_preempted_continuation(model, depth, length, strength=0.3, attention_backend="torch"):
    """Check independent hidden/KV/readout gates across native resumption.

    Reference-hidden error is diagnostic; snapshot copies are checked bit-exactly.
    Checks run at sampling because resumed CODA may finish in the same step.
    """
    prompt = PROMPT[:length]
    device = next(model.parameters()).device
    engine = _engine(model, attention_backend)
    worst = {"state": 0.0, "kv": 0.0, "logits": 0.0, "logprob": 0.0, "reference": 0.0}
    result = None
    try:
        oracle = OuroContinuation(model)
        expected, seen = [], []
        stops = {"first_coda": False, "next_prelude": False, "recurrent": False}

        def main_sample(logits, request):
            index = len(seen)
            sampled = _check(
                engine,
                oracle,
                logits,
                request,
                expected[index],
                CONTINUATION[index],
                strength,
                worst,
            )
            seen.append(logits.detach().clone())
            return sampled

        engine.model_runner._sample_tensor = main_sample
        engine.add_request("a", prompt, _params(depth, strength, len(CONTINUATION)))
        completed = None
        for _ in range(400):
            if not engine.has_unfinished_requests():
                break
            while len(expected) <= len(seen):
                index = len(expected)
                expected.append(_expected(oracle, prompt, CONTINUATION, index, depth, device))
            request = engine.scheduler.requests.get("a")
            if (
                request is not None
                and not stops["first_coda"]
                and (request.stage == Stage.CODA and request.num_scheduled_outputs == 0)
            ):
                live = engine.model_runner.loopcd_references.get("a")
                if strength != 0:
                    assert live is not None and live.owner is request

                state = suspend(engine)
                if strength != 0:
                    torch.testing.assert_close(
                        state["loopcd_reference"].hidden, live.hidden.cpu(), atol=0, rtol=0
                    )
                blocker = (
                    engine.cache_manager.num_blocks // engine.cache_manager.storage_depths
                ) * engine.cache_manager.block_size
                assert engine.cache_manager.allocate("blocker", blocker)
                assert engine.preemption.resume(request) is False
                assert engine.preemption.snapshots["a"] is state
                reference = state["loopcd_reference"]
                assert reference is None if strength == 0 else reference.owner is request
                assert not engine.cache_manager.has_allocation("a")
                assert request.hidden_state is None
                assert "a" not in engine.model_runner.loopcd_references
                engine.cache_manager.free("blocker")
                stops["first_coda"] = True
            elif (
                request is not None
                and not stops["next_prelude"]
                and (request.stage == Stage.PRELUDE and request.num_scheduled_outputs == 1)
            ):
                suspend(engine)
                stops["next_prelude"] = True
            elif (
                request is not None
                and not stops["recurrent"]
                and (request.stage == Stage.RECURRENT and request.loops_done == 1)
            ):
                suspend(engine)
                stops["recurrent"] = True
            for output in engine.step():
                if output.finished:
                    completed = output

        assert completed is not None
        assert completed.token_ids == CONTINUATION
        assert completed.exit_depths == [PREFILL_DEPTH] + [depth] * (len(CONTINUATION) - 1)
        assert len(seen) == len(CONTINUATION)
        assert all(stops.values())
        assert not engine.preemption.snapshots
        assert not engine.model_runner.loopcd_references
        assert engine.cache_manager.num_used_blocks == 0

        engine.model_runner._sample_tensor = lambda logits, request: torch.tensor(
            0, device=logits.device
        )
        engine.add_request("a", prompt, _params(depth, strength, len(REUSE_TOKENS)))
        old_request = engine.scheduler.requests["a"]
        for _ in range(200):
            current = engine.scheduler.requests["a"]
            if current.stage == Stage.CODA and current.num_scheduled_outputs == 0:
                break
            engine.step()
        assert current.stage == Stage.CODA and current.num_scheduled_outputs == 0
        suspend(engine)
        engine.abort_request("a")
        assert not engine.preemption.snapshots
        assert not engine.model_runner.loopcd_references
        assert engine.cache_manager.num_used_blocks == 0

        reuse_oracle = OuroContinuation(model)
        reuse_expected, reuse_seen = [], []

        def reuse_sample(logits, request):
            index = len(reuse_seen)
            sampled = _check(
                engine,
                reuse_oracle,
                logits,
                request,
                reuse_expected[index],
                REUSE_TOKENS[index],
                strength,
                worst,
            )
            reuse_seen.append(logits.detach().clone())
            return sampled

        engine.model_runner._sample_tensor = reuse_sample
        engine.add_request("a", REUSE_PROMPT, _params(depth, strength, len(REUSE_TOKENS)))
        assert engine.scheduler.requests["a"] is not old_request
        reused = None
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            while len(reuse_expected) <= len(reuse_seen):
                index = len(reuse_expected)
                reuse_expected.append(
                    _expected(reuse_oracle, REUSE_PROMPT, REUSE_TOKENS, index, depth, device)
                )
            for output in engine.step():
                if output.finished:
                    reused = output

        assert reused is not None
        assert reused.token_ids == REUSE_TOKENS
        assert reused.exit_depths == [PREFILL_DEPTH, depth]
        assert len(reuse_seen) == len(REUSE_TOKENS)
        assert not engine.preemption.snapshots
        assert not engine.model_runner.loopcd_references
        assert engine.cache_manager.num_used_blocks == 0
        result = {
            "outputs": len(seen),
            "reuse_outputs": len(reuse_seen),
            "exit_depths": completed.exit_depths,
            "preemptions": engine.preemption.preemptions,
            "resumptions": engine.preemption.resumptions,
            **{f"{key}_max_error": value for key, value in worst.items()},
        }
    finally:
        engine.close()
    return result


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("length", [3, 5])
@torch.inference_mode()
def test_preempted_continuation_cpu(depth, length):
    torch.manual_seed(1729)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    result = check_preempted_continuation(
        model, depth, length, strength=0.3, attention_backend="torch"
    )
    assert result["outputs"] == len(CONTINUATION)
    assert result["reuse_outputs"] == len(REUSE_TOKENS)
    assert result["preemptions"] == 4
    assert result["resumptions"] == 3
