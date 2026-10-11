"""Mixed-depth teacher forcing checks state, guided readout and every KV plane."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from tests.reference.continuation import OuroContinuation
from vllm_rlt import CacheConfig, ExecutionConfig, LoopCDParams, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Stage


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("length", [3, 4, 5])
@pytest.mark.parametrize("strength", [0.0, 0.3])
@torch.inference_mode()
def test_retained_p4_history(depth, length, strength):
    torch.manual_seed(1729)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    reference = OuroContinuation(model)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=128, block_size=4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(loopcd=True, prefill_depth=4),
    )
    prompt, continuation = [4, 7, 9, 3, 12][:length], [17, 8, 2, 11, 5, 7, 13, 6, 9]
    sampling = SamplingParams(
        max_tokens=len(continuation),
        min_loops=depth,
        max_loops=depth,
        ignore_eos=True,
        loopcd=LoopCDParams(strength=strength),
    )
    seen = []

    def sample(logits, request):
        seen.append(logits.detach().clone())
        return torch.tensor(continuation[len(seen) - 1], device=logits.device)

    engine.model_runner._sample_tensor = sample
    engine.add_request("reuse", prompt, sampling)
    for index, token in enumerate(continuation):
        expected = reference.forward(
            torch.tensor(prompt if index == 0 else [continuation[index - 1]]),
            4 if index == 0 else depth,
        )
        request = engine.scheduler.requests["reuse"]
        while request.stage != Stage.CODA:
            engine.step()
        torch.testing.assert_close(request.hidden_state, expected[-1][0][-1], atol=1e-4, rtol=1e-4)
        cache = engine.cache_manager
        allocation = cache._allocations["reuse"]
        for (loop, layer), (keys, values) in reference.kv.items():
            for position in range(reference.position):
                block = allocation.block_tables[loop][position // cache.block_size]
                offset = position % cache.block_size
                torch.testing.assert_close(
                    cache.key_cache[block, layer, offset], keys[position], atol=1e-4, rtol=1e-4
                )
                torch.testing.assert_close(
                    cache.value_cache[block, layer, offset], values[position], atol=1e-4, rtol=1e-4
                )
        expected_logits = expected[-1][1][-1].float()
        expected_logits = expected_logits + strength * (
            expected_logits - expected[0][1][-1].float()
        )
        engine.step()
        torch.testing.assert_close(seen[-1], expected_logits, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            seen[-1].float().log_softmax(-1)[token],
            expected_logits.log_softmax(-1)[token],
            atol=1e-4,
            rtol=1e-4,
        )
    assert not engine.has_unfinished_requests()
    assert not engine.model_runner.loopcd_references
    assert engine.cache_manager.num_used_blocks == 0
    assert reference.position == length + len(continuation) - 1
    engine.close()
