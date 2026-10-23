"""Selection contracts with hardware/package boundaries simulated on CPU."""

import logging
from types import SimpleNamespace

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams
from vllm_rlt.attention import BackendCapabilities, backend_capabilities, create_backend
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


@pytest.fixture
def packages(monkeypatch):
    from vllm_rlt.kernels import flash_attention

    installed = {"flash_attn", "flash_attn_3.flash_attn_interface", "flash_attn.cute", "triton"}
    original_import = flash_attention.importlib.import_module

    def import_module(name):
        if name == "vllm_rlt.kernels.triton_attention":
            if "triton" not in installed:
                raise ImportError("triton is not installed")
            return SimpleNamespace()
        if name in ("flash_attn", "flash_attn_3.flash_attn_interface", "flash_attn.cute"):
            if name not in installed:
                raise ImportError(f"{name} is not installed")
            return SimpleNamespace(
                flash_attn_with_kvcache=lambda *args, **kwargs: None,
                flash_attn_varlen_func=lambda *args, **kwargs: None,
            )
        return original_import(name)

    monkeypatch.setattr(flash_attention.importlib, "import_module", import_module)
    monkeypatch.setattr(flash_attention, "version", lambda name: "test")
    monkeypatch.setattr(torch.version, "hip", None)
    return installed


def test_cpu_auto_reports_torch_and_runs_reference_attention(caplog):
    caplog.set_level(logging.INFO)
    selected = create_backend("auto", torch.device("cpu"), torch.float32, 8, 2)
    assert selected.info["selected_backend"] == "torch"
    assert selected.info["selection_reason"]
    assert "torch" in caplog.text and "CPU" in caplog.text
    q = torch.randn(1, 1, 8)
    k, v = torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8)
    table, lengths = torch.tensor([[0]], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
    explicit = create_backend("torch", q.device, q.dtype, 8, 2)
    torch.testing.assert_close(selected(q, k, v, table, lengths), explicit(q, k, v, table, lengths))


@pytest.mark.parametrize(
    "sm,dtype,head_dim,block_size,missing,expected",
    [
        ((8, 9), torch.bfloat16, 128, 256, (), "flash_attn_2"),
        ((8, 9), torch.bfloat16, 128, 16, (), "triton"),
        ((8, 9), torch.float32, 128, 256, (), "triton"),
        ((8, 9), torch.bfloat16, 127, 256, (), "triton"),
        ((8, 9), torch.bfloat16, 264, 256, (), "torch"),
        ((8, 9), torch.bfloat16, 128, 256, ("flash_attn",), "triton"),
        ((8, 9), torch.bfloat16, 128, 16, ("triton",), "torch"),
        ((9, 0), torch.bfloat16, 128, 16, (), "flash_attn_4"),
        ((9, 0), torch.bfloat16, 128, 16, ("flash_attn.cute",), "flash_attn_3"),
        (
            (9, 0),
            torch.bfloat16,
            128,
            256,
            ("flash_attn.cute", "flash_attn_3.flash_attn_interface"),
            "flash_attn_2",
        ),
        ((10, 0), torch.bfloat16, 128, 16, (), "flash_attn_4"),
        ((12, 0), torch.bfloat16, 128, 16, (), "triton"),
        ((7, 5), torch.float16, 128, 256, (), "torch"),
    ],
)
def test_auto_selection_constraints(
    packages, monkeypatch, sm, dtype, head_dim, block_size, missing, expected
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: sm)
    packages.difference_update(missing)
    selected = create_backend("auto", torch.device("cuda"), dtype, head_dim, block_size)
    assert selected.info["selected_backend"] == expected
    assert selected.info["selection_reason"]
    if expected == "flash_attn_4":
        assert backend_capabilities(selected).packed_prefill


def test_rocm_auto_uses_triton(packages, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: pytest.fail("NVIDIA query")
    )
    selected = create_backend("auto", torch.device("cuda"), torch.bfloat16, 128, 16)
    assert selected.info["selected_backend"] == "triton"
    assert "NVIDIA" in selected.info["selection_reason"]


def test_auto_reports_missing_packages_and_configuration(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    packages.clear()
    selected = create_backend("auto", torch.device("cuda"), torch.bfloat16, 128, 16)
    reason = selected.info["selection_reason"]
    assert "flash-attn-4" in reason and "flash-attn-3" in reason
    assert "multiple of 256" in reason and "triton" in reason


@pytest.mark.parametrize("backend", ["flash_attn_2", "triton"])
def test_explicit_missing_package_never_falls_back(packages, monkeypatch, backend):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    packages.clear()
    with pytest.raises(ImportError):
        create_backend(backend, torch.device("cuda"), torch.bfloat16, 128, 256)


def test_explicit_flash_incompatibility_is_not_overridden(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    with pytest.raises(ValueError, match="multiple of 256"):
        create_backend("flash_attn_2", torch.device("cuda"), torch.bfloat16, 128, 16)


def test_auto_does_not_hide_unexpected_backend_failures(packages, monkeypatch):
    from vllm_rlt.attention import backend

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    def fail(*args, **kwargs):
        raise RuntimeError("implementation defect")

    monkeypatch.setattr(backend, "FlashPagedAttention", fail)
    with pytest.raises(RuntimeError, match="implementation defect"):
        create_backend("auto", torch.device("cuda"), torch.bfloat16, 128, 16)


@pytest.mark.parametrize("requirement", ["cuda_graphs", "device_context_lengths"])
def test_execution_requirements_prevent_unsafe_torch_fallback(packages, monkeypatch, requirement):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    packages.clear()
    with pytest.raises(ValueError, match="no compatible attention backend"):
        create_backend(
            "auto",
            torch.device("cuda"),
            torch.bfloat16,
            128,
            16,
            required_capabilities=BackendCapabilities(**{requirement: True}),
        )


@pytest.mark.parametrize(
    "sm,backend", [((8, 0), "flash_attn_2"), ((9, 0), "flash_attn_3"), ((10, 0), "flash_attn_4")]
)
def test_flash_device_lengths_and_graph_requirements(packages, monkeypatch, sm, backend):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: sm)
    selected = create_backend(
        backend,
        torch.device("cuda"),
        torch.bfloat16,
        64,
        256,
        required_capabilities=BackendCapabilities(
            async_scheduling=True, cuda_graphs=True, device_context_lengths=True
        ),
    )
    assert selected.info["selected_backend"] == backend


def test_packed_prefill_requires_declared_support(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    packages.remove("flash_attn.cute")
    with pytest.raises(ValueError, match="packed_prefill"):
        create_backend(
            "auto",
            torch.device("cuda"),
            torch.bfloat16,
            128,
            16,
            required_capabilities=BackendCapabilities(packed_prefill=True),
        )


def test_cpu_engine_resolves_auto_before_cache_planning():
    model = OuroForCausalLM(tiny_ouro_config())
    engine = LLMEngine(model, attention_backend="auto")
    assert engine.cache_manager.backend == "torch"
    assert engine.cache_manager.attention_info["requested_backend"] == "auto"
    assert engine.cache_manager.attention_info["selection_reason"]
    outputs = []
    for current in (engine, LLMEngine(model, attention_backend="torch")):
        current.add_request("a", [1, 2, 3], SamplingParams(max_tokens=2, ignore_eos=True))
        while current.has_unfinished_requests():
            for output in current.step():
                if output.finished:
                    outputs.append((output.token_ids, output.exit_depths))
        assert current.cache_manager.num_used_blocks == 0
    assert len(outputs) == 2 and outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "execution", [ExecutionConfig(cuda_graphs=True), ExecutionConfig(async_scheduling=True)]
)
def test_cuda_engine_resolves_auto_before_profiling(packages, monkeypatch, execution):
    from vllm_rlt.engine import stage_engine as llm_engine

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    model = SimpleNamespace(
        config=tiny_ouro_config(),
        parameters=lambda: iter(
            [SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)]
        ),
    )

    class ProfileReached(Exception):
        pass

    def plan_cache(model, cache, scheduler, execution, selected, *, speculative=None):
        assert selected == "triton"  # FA2 cannot use the configured 16-token pages.
        raise ProfileReached

    monkeypatch.setattr(llm_engine, "plan_cache", plan_cache)
    with pytest.raises(ProfileReached):
        LLMEngine(
            model,
            attention_backend="auto",
            execution_config=execution,
            exit_config=ExitConfig("ouro_delayed"),
            cache_config=CacheConfig(block_size=16),
        )
    packages.clear()
    with pytest.raises(ValueError, match="no compatible attention backend"):
        LLMEngine(
            model,
            attention_backend="auto",
            execution_config=execution,
            exit_config=ExitConfig("ouro_delayed"),
            cache_config=CacheConfig(block_size=16),
        )


def test_explicit_flash_alias_keeps_architecture_selection(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    explicit = create_backend("flash_attn", torch.device("cuda"), torch.bfloat16, 128, 16)
    assert explicit.info["selected_backend"] == "flash_attn_3"
    assert not explicit.capabilities.packed_prefill


def test_explicit_triton_rejects_unsupported_nvidia_hardware(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(ValueError, match="Compute Capability 8.0"):
        create_backend("triton", torch.device("cuda"), torch.float16, 128, 256)


def test_auto_skips_broken_binary_import(packages, monkeypatch):
    from vllm_rlt.kernels import flash_attention

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    original_import = flash_attention.importlib.import_module

    def import_module(name):
        if name == "flash_attn":
            raise OSError("incompatible CUDA extension")
        return original_import(name)

    monkeypatch.setattr(flash_attention.importlib, "import_module", import_module)
    selected = create_backend("auto", torch.device("cuda"), torch.bfloat16, 128, 256)
    assert selected.info["selected_backend"] == "triton"
    assert "incompatible CUDA extension" in selected.info["selection_reason"]
    with pytest.raises(OSError, match="incompatible CUDA extension"):
        create_backend("flash_attn_2", torch.device("cuda"), torch.bfloat16, 128, 256)
