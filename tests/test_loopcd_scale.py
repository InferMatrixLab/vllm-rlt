import pytest
import torch

from benchmarks.loopcd_scale import signature, trial
from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, LoopCDParams, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


@pytest.mark.parametrize("batch,concurrency", [(16, 32), (32, 64), (64, 128)])
@pytest.mark.parametrize("capacity_rows", [8, None])
@torch.inference_mode()
def test_closed_loop_refill_drains_and_reuses_ids(batch, concurrency, capacity_rows):
    torch.manual_seed(1729)
    resident_limit = capacity_rows or batch
    engine = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).eval(),
        cache_config=CacheConfig(num_blocks=8 * resident_limit, block_size=4),
        scheduler_config=SchedulerConfig(max_num_seqs=batch, max_num_batched_tokens=batch),
        execution_config=ExecutionConfig(loopcd=True, prefill_depth=4),
    )
    prompts = [[4, 7, 3]] * (4 * concurrency)
    sampling = SamplingParams(
        max_tokens=3, min_loops=2, max_loops=2, ignore_eos=True, loopcd=LoopCDParams()
    )
    first, second = [trial(engine, prompts, sampling, concurrency) for _ in range(2)]
    assert signature(first) == signature(second)
    assert len(first["requests"]) == 4 * concurrency
    assert max(first["outstanding_histogram"]) == concurrency
    assert max(first["resident_histogram"]) == resident_limit
    assert max(first["decode_effective_rows_histogram"]) == resident_limit
    assert first["drained"] and second["drained"]
    assert first["kv_peak_blocks"] == 8 * resident_limit
    assert first["graph"] == dict(captures=0, replays=0, fallbacks=0)
    engine.close()
