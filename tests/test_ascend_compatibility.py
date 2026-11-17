"""Ascend opt-ins preserve the existing CPU/CUDA/ROCm API and defaults."""

import builtins
from dataclasses import fields

import pytest
import torch

from tests.helpers import tiny_ouro_model
from vllm_rlt import CacheConfig, ExecutionConfig, SchedulerConfig
from vllm_rlt.attention.backend import BackendCapabilities
from vllm_rlt.device import stream_module
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.layers.rotary_embedding import RotaryEmbedding
from vllm_rlt.worker.executor import ModelRuntime
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.sampler import Sampler


def test_backend_capability_positional_arguments_keep_their_meaning():
    names = tuple(f.name for f in fields(BackendCapabilities))
    assert names[:6] == (
        "packed_prefill",
        "cuda_graphs",
        "async_scheduling",
        "packed_matches_decode",
        "device_context_lengths",
        "sliding_window",
    )
    caps = BackendCapabilities(False, False, False, False, False, True)
    assert caps.sliding_window and not caps.sequence_ranges and not caps.owned_contiguous_kv


@pytest.mark.parametrize("hip", [None, "6.4"])
def test_cuda_and_rocm_stream_selection_never_loads_npu(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch_npu" or name.startswith("vllm_rlt.worker.ascend"):
            pytest.fail("CUDA/ROCm selected an Ascend implementation")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert stream_module(torch.device("cuda:0")) is torch.cuda


@pytest.mark.parametrize("hip", [None, "6.4"])
def test_cuda_rope_keeps_device_math_without_host_migration(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    rotary = RotaryEmbedding(128)
    arange = torch.arange
    seen = []

    def recording(*args, **kwargs):
        seen.append(kwargs["device"])
        # CPU execution of the same arithmetic; no CUDA/ROCm hardware claim.
        return arange(*args, **dict(kwargs, device="cpu"))

    monkeypatch.setattr(torch, "arange", recording)
    actual = rotary.frequencies(torch.device("cuda:0"))
    assert seen == [torch.device("cuda:0")]
    expected = 1.0 / (10000.0 ** (arange(0, 128, 2, dtype=torch.float32) / 128))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_default_engine_keeps_upstream_runtime_and_sampler(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch_npu" or name.startswith("vllm_rlt.worker.ascend"):
            pytest.fail("default engine loaded Ascend worker code")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    engine = StageEngine(
        tiny_ouro_model(),
        cache_config=CacheConfig(num_blocks=128),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=128),
    )
    try:
        assert type(engine.model_runner) is ModelRunner
        assert type(engine.model_runner.runtime) is ModelRuntime
        assert type(engine.model_runner.sampler) is Sampler
    finally:
        engine.close()


@pytest.mark.parametrize("flag", ["contiguous_kv", "coalesce_fixed_prelude", "native_eager_plan"])
def test_ascend_options_fail_before_allocating_on_other_devices(flag):
    cache = CacheConfig(num_blocks=128, **({flag: True} if flag == "contiguous_kv" else {}))
    execution = ExecutionConfig(**({flag: True} if flag != "contiguous_kv" else {}))
    with pytest.raises(ValueError, match="Ascend execution options require an NPU"):
        StageEngine(tiny_ouro_model(), cache_config=cache, execution_config=execution)
