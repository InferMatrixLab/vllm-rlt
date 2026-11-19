"""Ascend opt-ins preserve the existing CPU/CUDA/ROCm API and defaults."""

import builtins
from dataclasses import fields

import pytest
import torch

from tests.helpers import tiny_ouro_model
from vllm_rlt import CacheConfig, SchedulerConfig
from vllm_rlt.attention.backend import BackendCapabilities
from vllm_rlt.device import stream_module
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.layers.rotary_embedding import RotaryEmbedding
from vllm_rlt.worker.executor import ModelRuntime
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.sampler import Sampler


def test_backend_capability_positional_arguments_keep_their_meaning():
    names = tuple(f.name for f in fields(BackendCapabilities))
    assert names == (
        "packed_prefill",
        "cuda_graphs",
        "async_scheduling",
        "packed_matches_decode",
        "fp8_kv",
        "fp8_kv_static",
        "device_context_lengths",
        "sliding_window",
        "packed_sliding_window",
    )
    caps = BackendCapabilities(False, False, False, False, False, True, False, True)
    assert caps.sliding_window and caps.fp8_kv_static


@pytest.mark.parametrize("hip", [None, "6.4"])
def test_cuda_and_rocm_stream_selection_never_loads_npu(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch_npu" or name.startswith("vllm_rlt.worker.ascend_model_runner"):
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
        if name == "torch_npu" or name.startswith("vllm_rlt.worker.ascend_model_runner"):
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
