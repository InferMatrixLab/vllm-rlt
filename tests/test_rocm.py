"""ROCm platform behavior that does not require AMD hardware."""

import pytest
import torch

from vllm_rlt.attention import create_backend
from vllm_rlt.platform import is_cuda, is_rocm, supports_flash_attention
from vllm_rlt.worker.tensors import use_uva_on


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(torch.version, "cuda", None)


def test_platform_flags(rocm):
    assert is_rocm()
    assert not is_cuda()
    assert not supports_flash_attention()


def test_auto_prefers_triton_on_rocm(rocm):
    selected = create_backend("auto", torch.device("cuda"), torch.bfloat16, 128, 16)
    assert selected.info["selected_backend"] == "triton"
    assert "FlashAttention is NVIDIA-only" in selected.info["selection_reason"]


def test_explicit_flash_rejected_on_rocm(rocm):
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        create_backend("flash_attn", torch.device("cuda"), torch.bfloat16, 128, 16)


def test_uva_override_and_cpu_fallback(monkeypatch):
    assert not use_uva_on(torch.device("cpu"))
    monkeypatch.setenv("VLLM_RLT_UVA", "0")
    assert not use_uva_on(torch.device("cuda", 0))
    monkeypatch.setenv("VLLM_RLT_UVA", "1")
    assert use_uva_on(torch.device("cuda", 0)) is torch.cuda.is_available()


@pytest.mark.gpu
def test_packed_prefill_and_wcpb_match_default():
    from tests.helpers import tiny_ouro_config
    from vllm_rlt import (
        LLM,
        CacheConfig,
        ExecutionConfig,
        ExitConfig,
        SamplingParams,
        SchedulerConfig,
    )
    from vllm_rlt.models import OuroForCausalLM

    torch.manual_seed(9)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    prompts = [[2] * 16, [3] * 23]
    params = [SamplingParams(max_tokens=3, ignore_eos=True, temperature=0.0)] * 2

    def run(*, execution=None, scheduler=None):
        with LLM(
            model,
            device="cuda",
            dtype=torch.bfloat16,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=64, block_size=16),
            scheduler_config=scheduler
            or SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64, prefill_chunk_size=8),
            execution_config=execution or ExecutionConfig(),
            exit_config=ExitConfig("ouro"),
        ) as llm:
            return [o.token_ids for o in llm.generate(prompts, params)]

    baseline = run()
    assert run(execution=ExecutionConfig(prefill_uva=True, use_uva=True)) == baseline
    assert (
        run(
            scheduler=SchedulerConfig(
                max_num_seqs=2,
                max_num_batched_tokens=64,
                prefill_chunk_size=8,
                wavefront_chunked_prefill=True,
            )
        )
        == baseline
    )
