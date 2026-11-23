"""Device bootstrap and fail-fast NPU scope without a device dependency."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig, SpeculativeConfig
from vllm_rlt.core.memory import plan_cache
from vllm_rlt.device import resolve_device, stream_module
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.profiling import ProfileConfig


def test_cpu_device_resolution_does_not_import_optional_backend(monkeypatch):
    monkeypatch.setattr(
        importlib, "import_module", lambda name: pytest.fail(f"unexpected import: {name}")
    )
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda:2") == torch.device("cuda:2")
    assert stream_module("cpu") is None
    assert stream_module("cuda:2") is torch.cuda


def npu_model_stub(**overrides):
    parameter = SimpleNamespace(
        device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, element_size=lambda: 2
    )
    return SimpleNamespace(
        parameters=lambda: iter([parameter]), config=tiny_ouro_config(**overrides)
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(execution_config=ExecutionConfig(cuda_graphs=True)),
        dict(execution_config=ExecutionConfig(async_scheduling=True, static_buffers=True)),
        dict(exit_config=ExitConfig(mode="random_lookahead")),
    ],
)
def test_npu_unsupported_execution_rejected_before_allocation(options):
    with pytest.raises(ValueError, match="NPU Ouro requires local execution"):
        StageEngine(npu_model_stub(), **options)


@pytest.mark.parametrize("backend", ["torch", "ascend_eager"])
def test_npu_requires_bf16_kv_before_allocation(backend):
    with pytest.raises(ValueError, match="NPU Ouro requires BF16 KV cache"):
        StageEngine(
            npu_model_stub(),
            cache_config=CacheConfig(kv_cache_dtype="fp8_e4m3"),
            attention_backend=backend,
        )


def test_npu_does_not_use_cpu_default_memory_budget():
    model = npu_model_stub()
    with pytest.raises(ValueError, match="explicit num_blocks"):
        plan_cache(model, CacheConfig(), SchedulerConfig(), ExecutionConfig(), "torch")
    blocks, info = plan_cache(
        model, CacheConfig(num_blocks=17), SchedulerConfig(), ExecutionConfig(), "torch"
    )
    assert blocks == 17 and info["source"] == "blocks"


@pytest.mark.parametrize("backend", ["triton", "flash_attn_4"])
def test_npu_rejects_incompatible_attention_before_allocation(backend):
    with pytest.raises(ValueError, match="NPU attention requires"):
        StageEngine(npu_model_stub(), attention_backend=backend)


def test_builtin_profiler_does_not_mislabel_npu_as_cpu_only(tmp_path):
    with pytest.raises(ValueError, match="torch_npu.profiler"):
        ProfileConfig(str(tmp_path)).resolve_activities("npu:0")


def test_npu_async_requires_explicit_delayed_exit_policy():
    with pytest.raises(ValueError, match="async scheduling requires ouro_delayed"):
        StageEngine(npu_model_stub(), execution_config=ExecutionConfig(async_scheduling=True))


def test_npu_rejects_async_self_speculation_before_allocation():
    with pytest.raises(ValueError, match="async speculative decoding requires CUDA"):
        StageEngine(
            npu_model_stub(),
            speculative_config=SpeculativeConfig(2),
            execution_config=ExecutionConfig(async_scheduling=True),
        )
