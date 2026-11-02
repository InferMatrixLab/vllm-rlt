"""Device bootstrap and fail-fast NPU scope without a device dependency."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.core.memory import plan_cache
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine


def test_cpu_device_resolution_does_not_import_optional_backend(monkeypatch):
    monkeypatch.setattr(
        importlib, "import_module", lambda name: pytest.fail(f"unexpected import: {name}")
    )
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda:2") == torch.device("cuda:2")


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
        dict(execution_config=ExecutionConfig(async_scheduling=True)),
        dict(execution_config=ExecutionConfig(cuda_graphs=True)),
        dict(cache_config=CacheConfig(num_blocks=32, enable_prefix_caching=True)),
        dict(scheduler_config=SchedulerConfig(enable_preemption=True)),
        dict(exit_config=ExitConfig(mode="ouro_delayed")),
    ],
)
def test_npu_unsupported_execution_rejected_before_allocation(options):
    with pytest.raises(ValueError, match="NPU Ouro requires synchronous"):
        StageEngine(npu_model_stub(), **options)


def test_npu_does_not_use_cpu_default_memory_budget():
    model = npu_model_stub()
    with pytest.raises(ValueError, match="explicit num_blocks"):
        plan_cache(model, CacheConfig(), SchedulerConfig(), ExecutionConfig(), "ascend")
    blocks, info = plan_cache(
        model, CacheConfig(num_blocks=17), SchedulerConfig(), ExecutionConfig(), "ascend"
    )
    assert blocks == 17 and info["source"] == "blocks"
