"""Graph replay must update KV addresses, request identities and host progress."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


def test_cuda_graph_validation():
    with pytest.raises(ValueError, match="positive"):
        ExecutionConfig(cuda_graph_max_graphs=0)
    with pytest.raises(ValueError, match="boolean"):
        ExecutionConfig(cuda_graphs=1)
    with pytest.raises(ValueError, match="CUDA graphs require CUDA"):
        LLMEngine(
            OuroForCausalLM(tiny_ouro_config()), execution_config=ExecutionConfig(cuda_graphs=True)
        )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn", "flashinfer"])
@pytest.mark.parametrize("mode", ["sync", "async", "multi"])
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("static", [False, True])
def test_graph_replay_matches_eager_and_reuses_requests(backend, mode, layout, static):
    torch.manual_seed(321)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    outputs = []
    for graphs in [False, True]:
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=64, block_size=16, layout=layout),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
            ),
            execution_config=ExecutionConfig(
                async_scheduling=mode != "sync",
                multi_stream=mode == "multi",
                static_buffers=static,
                pad_to_power_of_two=static,
                cuda_graphs=graphs,
            ),
            exit_config=ExitConfig("ouro_delayed"),
            attention_backend=backend,
        )
        rounds = []
        for reuse in range(2):
            for i in range(3):
                engine.add_request(
                    str(i),
                    [2 + i, 3, 4] * (i + 1 + reuse),
                    SamplingParams(
                        max_tokens=5 + i, min_loops=2, exit_threshold=0.2, ignore_eos=True
                    ),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for out in engine.step():
                    if out.finished:
                        finished[out.request_id] = (out.token_ids, out.exit_depths)
            assert engine.cache_manager.num_used_blocks == 0
            rounds.append(finished)
        if graphs:
            assert engine.model_runner.graphs.captures > 0
            assert engine.model_runner.graphs.replays > engine.model_runner.graphs.captures
        outputs.append(rounds)
    assert outputs[0] == outputs[1]


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flashinfer"])
def test_graph_cache_limit_fallback_and_abort(backend):
    torch.manual_seed(19)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(
            cuda_graphs=True,
            cuda_graph_max_batch_size=2,
            cuda_graph_max_graphs=1,
            async_scheduling=True,
        ),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend=backend,
    )
    for count in [1, 2, 3]:
        for i in range(count):
            engine.add_request(str(i), [2, 3, 4], SamplingParams(max_tokens=8, ignore_eos=True))
        for _ in range(12):
            engine.step()
        for rid in list(engine.scheduler.requests):
            engine.abort_request(rid)
        while engine.has_unfinished_requests():
            engine.step()
        assert engine.cache_manager.num_used_blocks == 0
    assert engine.model_runner.graphs.captures == 1
    assert engine.model_runner.graphs.fallbacks > 0
