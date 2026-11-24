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
    from vllm_rlt.kernels import flash_attention, flashinfer_attention

    installed = {
        "flash_attn",
        "flash_attn_3.flash_attn_interface",
        "flash_attn.cute",
        "TensorwiseScales",
        "triton",
        "flashinfer",
    }
    original_import = flash_attention.importlib.import_module

    def import_module(name):
        if name == "vllm_rlt.kernels.triton_attention":
            if "triton" not in installed:
                raise ImportError("triton is not installed")
            return SimpleNamespace(select_packed_window_tile=lambda *args: None)
        if name in ("flash_attn", "flash_attn_3.flash_attn_interface", "flash_attn.cute"):
            if name not in installed:
                raise ImportError(f"{name} is not installed")
            result = SimpleNamespace(
                flash_attn_with_kvcache=lambda *args, **kwargs: None,
                flash_attn_varlen_func=lambda *args, **kwargs: None,
            )
            if name == "flash_attn.cute" and "TensorwiseScales" in installed:
                result.TensorwiseScales = SimpleNamespace
            return result
        if name == "flashinfer.decode":
            if "flashinfer" not in installed:
                raise ImportError("flashinfer is not installed")
            return SimpleNamespace(trtllm_batch_decode_with_kv_cache=lambda *args, **kwargs: None)
        if name == "flashinfer.compilation_context":
            return SimpleNamespace(
                CompilationContext=lambda: SimpleNamespace(TARGET_CUDA_ARCHS={(12, "0f")})
            )
        if name == "flashinfer.jit.cpp_ext":
            return SimpleNamespace(is_cuda_version_at_least=lambda version: True)
        return original_import(name)

    monkeypatch.setattr(flash_attention.importlib, "import_module", import_module)
    monkeypatch.setattr(
        flash_attention, "version", lambda name: "4.0.0b39" if name == "flash-attn-4" else "test"
    )
    monkeypatch.setattr(flashinfer_attention, "version", lambda name: "test")
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
    "sm,dtype,head_dim,supported",
    [
        ((10, 3), torch.bfloat16, 128, True),
        ((9, 0), torch.bfloat16, 128, False),
        ((10, 3), torch.float16, 128, False),
        ((10, 3), torch.bfloat16, 256, True),
    ],
)
def test_fa4_static_fp8_selection(packages, monkeypatch, sm, dtype, head_dim, supported):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: sm)
    args = ("flash_attn_4", torch.device("cuda"), dtype, head_dim, 16)
    requirement = BackendCapabilities(fp8_kv_static=True)
    if not supported:
        with pytest.raises(ValueError, match="fp8_kv_static"):
            create_backend(*args, required_capabilities=requirement)
        return
    selected = create_backend(*args, required_capabilities=requirement)
    assert not selected.capabilities.packed_matches_decode
    with pytest.raises(ValueError, match="fp8_kv"):
        create_backend(*args, required_capabilities=BackendCapabilities(fp8_kv=True))
    selected = create_backend("auto", *args[1:], required_capabilities=requirement)
    assert selected.info["selected_backend"] == "flash_attn_4"


def test_xqa_static_fp8_selection(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = ("flashinfer", torch.device("cuda"), torch.bfloat16, 128, 16)
    required = BackendCapabilities(fp8_kv=True)
    selected = create_backend(*args, required_capabilities=required, allow_fp8_calibration=True)
    assert selected.capabilities.fp8_kv_static and not selected.capabilities.fp8_kv
    with pytest.raises(ValueError, match="fp8_kv"):
        create_backend(*args, required_capabilities=required)


def test_fa4_static_fp8_requires_public_scale_api(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    packages.remove("TensorwiseScales")
    args = ("flash_attn_4", torch.device("cuda"), torch.bfloat16, 128, 16)
    assert create_backend(*args).info["selected_backend"] == "flash_attn_4"
    with pytest.raises(ValueError, match="fp8_kv_static"):
        create_backend(*args, required_capabilities=BackendCapabilities(fp8_kv_static=True))


@pytest.mark.parametrize(
    "name,sm,block_size",
    [("flash_attn_2", (8, 9), 256), ("flash_attn_3", (9, 0), 16), ("flash_attn_4", (10, 3), 16)],
)
def test_flash_rejects_per_vector_fp8_without_staging(packages, monkeypatch, name, sm, block_size):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: sm)
    with pytest.raises(ValueError, match="fp8_kv"):
        create_backend(
            name,
            torch.device("cuda"),
            torch.bfloat16,
            128,
            block_size,
            required_capabilities=BackendCapabilities(fp8_kv=True),
        )


@pytest.mark.parametrize("calibrate", [False, True])
def test_auto_static_fp8_keeps_graph_support_and_falls_back_if_scale_api_missing(
    packages, monkeypatch, calibrate
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    args = ("auto", torch.device("cuda"), torch.bfloat16, 128, 16)
    required = BackendCapabilities(fp8_kv=calibrate, fp8_kv_static=not calibrate, cuda_graphs=True)
    options = dict(required_capabilities=required, allow_fp8_calibration=calibrate)
    selected = create_backend(*args, **options)
    assert selected.info["selected_backend"] == "flash_attn_4"
    assert selected.capabilities.fp8_kv_static and selected.capabilities.cuda_graphs
    assert not selected.capabilities.packed_matches_decode

    packages.remove("TensorwiseScales")
    selected = create_backend(*args, **options)
    assert selected.info["selected_backend"] == "triton"
    assert selected.capabilities.fp8_kv_static and selected.capabilities.cuda_graphs


def test_dory_requirements_select_native_fa4_and_reject_old_hd256(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    args = (torch.device("cuda"), torch.bfloat16, 256, 16)
    required = BackendCapabilities(
        sliding_window=True, fp8_kv_static=True, cuda_graphs=True, async_scheduling=True
    )
    selected = create_backend("auto", *args, required_capabilities=required)
    assert selected.info["selected_backend"] == "flash_attn_4"
    assert selected.capabilities.sliding_window
    packages.remove("TensorwiseScales")
    with pytest.raises(ValueError, match="sliding_window"):
        create_backend("flash_attn_4", *args, required_capabilities=required)
    assert (
        create_backend("auto", *args, required_capabilities=required).info["selected_backend"]
        == "triton"
    )


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
        (
            (9, 0),
            torch.bfloat16,
            128,
            16,
            ("flash_attn.cute", "flash_attn_3.flash_attn_interface"),
            "triton",
        ),
        ((10, 0), torch.bfloat16, 128, 16, ("flash_attn.cute",), "triton"),
        ((12, 0), torch.bfloat16, 128, 16, (), "flashinfer"),
        ((12, 1), torch.bfloat16, 128, 16, (), "triton"),
        ((12, 0), torch.bfloat16, 128, 16, ("flashinfer",), "triton"),
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


def test_flashinfer_wide_head_group_limit(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = (torch.device("cuda"), torch.bfloat16, 512, 16)
    assert create_backend("auto", *args).info["selected_backend"] == "torch"
    assert create_backend("flashinfer", *args).info["selected_backend"] == "flashinfer"
    assert (
        create_backend("auto", *args, head_group_ratio=16).info["selected_backend"] == "flashinfer"
    )
    assert (
        create_backend("auto", args[0], args[1], 256, 16).info["selected_backend"] == "flashinfer"
    )
    assert create_backend("flashinfer", *args, head_group_ratio=16).info["kernel"] == "xqa"
    with pytest.raises(ValueError, match="head_group_ratio <= 16"):
        create_backend("flashinfer", *args, head_group_ratio=32)
    assert create_backend("auto", *args, head_group_ratio=32).info["selected_backend"] == "torch"


@pytest.mark.parametrize("targets", [set(), {(9, "0a"), (12, "1a")}, {(12, "0a")}])
def test_flashinfer_selection_requires_a_jit_target_for_the_device(packages, monkeypatch, targets):
    from vllm_rlt.kernels import flashinfer_attention

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    original_import = flashinfer_attention.importlib.import_module

    def import_module(name):
        if name == "flashinfer.compilation_context":
            return SimpleNamespace(
                CompilationContext=lambda: SimpleNamespace(TARGET_CUDA_ARCHS=targets)
            )
        return original_import(name)

    monkeypatch.setattr(flashinfer_attention.importlib, "import_module", import_module)
    args = (torch.device("cuda"), torch.bfloat16, 128, 16)
    selected = create_backend("auto", *args)
    if (12, "0a") in targets:
        assert selected.info["selected_backend"] == "flashinfer"
    else:
        assert selected.info["selected_backend"] == "triton"
        assert "no XQA JIT target for SM120" in selected.info["selection_reason"]
        with pytest.raises(ValueError, match="no XQA JIT target for SM120"):
            create_backend("flashinfer", *args)


@pytest.mark.parametrize("reject_at_import", [True, False], ids=["import", "forced-120f"])
def test_flashinfer_jit_toolkit_failure_before_first_launch(
    packages, monkeypatch, reject_at_import
):
    from vllm_rlt.kernels import flashinfer_attention

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    original_import = flashinfer_attention.importlib.import_module

    def import_module(name):
        if name == "flashinfer.decode" and reject_at_import:
            raise RuntimeError("SM 12.x requires CUDA >= 12.9")
        if name == "flashinfer.jit.cpp_ext":
            return SimpleNamespace(is_cuda_version_at_least=lambda minimum: minimum == "12.8")
        return original_import(name)

    monkeypatch.setattr(flashinfer_attention.importlib, "import_module", import_module)
    args = (torch.device("cuda"), torch.bfloat16, 128, 16)
    selected = create_backend("auto", *args)
    assert selected.info["selected_backend"] == "triton"
    assert "CUDA >= 12.9" in selected.info["selection_reason"]
    with pytest.raises(ValueError, match="CUDA >= 12.9"):
        create_backend("flashinfer", *args)


@pytest.mark.parametrize("sm", [(9, 0), (12, 1)])
def test_explicit_flashinfer_opt_in_outside_sm120(packages, monkeypatch, caplog, sm):
    caplog.set_level(logging.WARNING, logger="vllm_rlt.kernels.flashinfer_attention")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: sm)
    selected = create_backend("flashinfer", torch.device("cuda"), torch.bfloat16, 128, 16)
    assert selected.info["selected_backend"] == "flashinfer"
    assert selected.capabilities.cuda_graphs and selected.capabilities.async_scheduling
    assert not selected.capabilities.packed_prefill
    assert "unvalidated" in caplog.text


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


@pytest.mark.parametrize(
    "backend,sm,block_size",
    [("flash_attn_2", (8, 9), 256), ("triton", (8, 9), 256), ("flashinfer", (12, 0), 16)],
)
def test_explicit_missing_package_never_falls_back(packages, monkeypatch, backend, sm, block_size):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: sm)
    packages.clear()
    with pytest.raises(ImportError):
        create_backend(backend, torch.device("cuda"), torch.bfloat16, 128, block_size)


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


@pytest.mark.parametrize("requested", ["triton", "auto"])
def test_triton_packed_prefill_is_per_selection_opt_in(packages, monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    packages.difference_update(
        {"flash_attn.cute", "flash_attn_3.flash_attn_interface", "flash_attn"}
    )
    args = (requested, torch.device("cuda"), torch.bfloat16, 128, 16)
    default = create_backend(*args)
    packed = create_backend(*args, required_capabilities=BackendCapabilities(packed_prefill=True))
    later = create_backend(*args)
    assert default.info["selected_backend"] == packed.info["selected_backend"] == "triton"
    assert not default.capabilities.packed_prefill
    assert packed.capabilities.packed_prefill
    assert not packed.capabilities.packed_matches_decode
    assert not later.capabilities.packed_prefill


@pytest.mark.parametrize(
    "missing,expected,packed",
    [
        ((), "flash_attn_4", True),
        (("flash_attn.cute",), "flash_attn_3", False),
        (
            ("flash_attn.cute", "flash_attn_3.flash_attn_interface", "flash_attn"),
            "triton",
            True,
        ),
    ],
)
def test_preferred_packed_prefill_preserves_backend_priority(
    packages, monkeypatch, missing, expected, packed
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    packages.difference_update(missing)
    args = ("auto", torch.device("cuda"), torch.bfloat16, 128, 256)
    baseline = create_backend(*args)
    preferred = create_backend(*args, prefer_packed_prefill=True)
    assert preferred.info["selected_backend"] == baseline.info["selected_backend"] == expected
    assert preferred.capabilities.packed_prefill is packed


def test_packed_prefill_requires_declared_support(packages, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    packages.remove("flash_attn.cute")
    # Without FA4, auto skips FA3/FA2 and selects Triton's packed paged prefill.
    backend = create_backend(
        "auto",
        torch.device("cuda"),
        torch.bfloat16,
        128,
        16,
        required_capabilities=BackendCapabilities(packed_prefill=True),
    )
    assert backend.info["selected_backend"] == "triton"
    assert backend.capabilities.packed_prefill
    with pytest.raises(ValueError, match="packed_prefill"):
        create_backend(
            "torch",
            torch.device("cpu"),
            torch.float32,
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

    def plan_cache(
        model,
        cache,
        scheduler,
        execution,
        selected,
        *,
        speculative=None,
        staged_prefill=None,
        fp8_scale=None,
        attention=None,
    ):
        assert selected == "triton"  # FA2 cannot use the configured 16-token pages.
        assert attention.info["selected_backend"] == selected
        assert staged_prefill is False
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


def test_triton_advertises_packed_window_only_after_selecting_a_tile(packages, monkeypatch):
    from vllm_rlt.attention import backend as selection
    from vllm_rlt.kernels.paged_attention import PackedWindowShape

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    base_import = selection.importlib.import_module
    shape = PackedWindowShape(16, 4, 36, 1024)
    calls = []

    def probe(device, dtype, head_dim, block_size, requested_shape):
        calls.append((device, dtype, head_dim, block_size, requested_shape))
        return (32, 32)

    def import_module(name):
        if name == "vllm_rlt.kernels.triton_attention":
            return SimpleNamespace(select_packed_window_tile=probe)
        return base_import(name)

    monkeypatch.setattr(selection.importlib, "import_module", import_module)
    args = (torch.device("cuda"), torch.bfloat16, 256, 16)
    backend = create_backend(
        "auto",
        *args,
        required_capabilities=BackendCapabilities(sliding_window=True),
        packed_window_shape=shape,
    )
    assert backend.info["selected_backend"] == "triton"
    assert backend.info["packed_prefill_tile"] == (32, 32)
    assert backend.capabilities.packed_prefill and backend.capabilities.packed_sliding_window
    assert not create_backend("triton", *args).capabilities.packed_sliding_window
    assert calls == [(*args, shape)]


@pytest.mark.parametrize("backend", ["triton", "auto"])
def test_packed_window_falls_back_to_per_query_when_probe_has_no_tile(
    packages, monkeypatch, backend
):
    from vllm_rlt.kernels.paged_attention import PackedWindowShape

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    shape = PackedWindowShape(16, 4, 36, 1024)
    selected = create_backend(
        backend,
        torch.device("cuda"),
        torch.bfloat16,
        256,
        16,
        required_capabilities=BackendCapabilities(sliding_window=True),
        packed_window_shape=shape,
    )
    assert selected.info["selected_backend"] == "triton"
    assert selected.info["packed_prefill_tile"] is None
    assert not selected.capabilities.packed_prefill
    assert not selected.capabilities.packed_sliding_window


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


@pytest.mark.parametrize("requested", [None, True, False])
@pytest.mark.parametrize("tile", [None, (32, 32)])
@pytest.mark.parametrize("backend", ["triton", "auto"])
def test_window_engine_resolves_tristate_before_cache_planning(
    packages, monkeypatch, requested, tile, backend
):
    from tests.test_dory import tiny_dory_config
    from vllm_rlt.attention import backend as selection
    from vllm_rlt.engine import stage_engine

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    original_import = selection.importlib.import_module
    probes = []

    def import_module(name):
        if name == "vllm_rlt.kernels.triton_attention":

            def probe(*args):
                probes.append(args[-1])
                return tile

            return SimpleNamespace(select_packed_window_tile=probe)
        return original_import(name)

    monkeypatch.setattr(selection.importlib, "import_module", import_module)
    config = tiny_dory_config()
    model = SimpleNamespace(
        config=config,
        requires_boundary_kv=True,
        requires_sliding_window=True,
        parameters=lambda: iter(
            [SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)]
        ),
    )

    class Planned(Exception):
        pass

    def plan_cache(model, cache, scheduler, execution, selected, **kwargs):
        assert execution.packed_prefill is (requested is not False and tile is not None)
        assert selected == "triton"
        raise Planned

    monkeypatch.setattr(stage_engine, "plan_cache", plan_cache)
    error = ValueError if requested is True and tile is None else Planned
    with pytest.raises(error):
        LLMEngine(
            model,
            attention_backend=backend,
            execution_config=ExecutionConfig(packed_prefill=requested),
        )
    assert len(probes) == int(requested is not False)
    if probes:
        assert probes[0].query_heads == config.num_attention_heads
        assert probes[0].kv_layers == config.num_kv_layers
        assert probes[0].window == config.sliding_window


@pytest.mark.parametrize("missing", ["packed_prefill", "packed_sliding_window"])
@pytest.mark.parametrize("requested", [None, True])
def test_window_engine_requires_both_packed_capabilities(monkeypatch, missing, requested):
    from dataclasses import replace

    from tests.test_dory import tiny_dory_config
    from vllm_rlt.attention import AttentionBackend
    from vllm_rlt.attention import backend as selection
    from vllm_rlt.engine import stage_engine

    capabilities = replace(
        BackendCapabilities(packed_prefill=True, sliding_window=True, packed_sliding_window=True),
        **{missing: False},
    )
    monkeypatch.setattr(
        selection,
        "_create_explicit",
        lambda *args, **kwargs: AttentionBackend(None, capabilities, {"backend": "triton"}),
    )
    model = SimpleNamespace(
        config=tiny_dory_config(),
        requires_boundary_kv=True,
        requires_sliding_window=True,
        parameters=lambda: iter(
            [SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)]
        ),
    )

    class Planned(Exception):
        pass

    def plan_cache(model, cache, scheduler, execution, selected, **kwargs):
        assert requested is None
        assert execution.packed_prefill is False
        raise Planned

    monkeypatch.setattr(stage_engine, "plan_cache", plan_cache)
    error = ValueError if requested is True else Planned
    with pytest.raises(error, match=missing if requested else None):
        LLMEngine(
            model,
            attention_backend="triton",
            execution_config=ExecutionConfig(packed_prefill=requested),
        )


@pytest.mark.parametrize("requested", [None, True])
@pytest.mark.parametrize(
    "asynchronous,graphs,preemption,role",
    [
        (True, False, False, "prefill_decode"),
        (False, True, False, "prefill_decode"),
        (False, False, True, "prefill_decode"),
        (False, False, False, "decode"),
    ],
)
def test_incompatible_window_engine_does_not_probe_packed_tile(
    packages, monkeypatch, requested, asynchronous, graphs, preemption, role
):
    from tests.test_dory import tiny_dory_config
    from vllm_rlt import SchedulerConfig
    from vllm_rlt.attention import backend as selection
    from vllm_rlt.engine import stage_engine
    from vllm_rlt.engine.stage_config import StageConfig

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    original_import = selection.importlib.import_module

    def import_module(name):
        if name == "vllm_rlt.kernels.triton_attention":
            return SimpleNamespace(
                select_packed_window_tile=lambda *args: pytest.fail(
                    "incompatible execution probed a packed tile"
                )
            )
        return original_import(name)

    monkeypatch.setattr(selection.importlib, "import_module", import_module)
    model = SimpleNamespace(
        config=tiny_dory_config(),
        requires_boundary_kv=True,
        requires_sliding_window=True,
        fixed_loop_count=True,
        supports_async_scheduling=True,
        parameters=lambda: iter(
            [SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)]
        ),
    )

    class Planned(Exception):
        pass

    def plan_cache(model, cache, scheduler, execution, selected, **kwargs):
        assert requested is None and execution.packed_prefill is False
        raise Planned

    monkeypatch.setattr(stage_engine, "plan_cache", plan_cache)
    with pytest.raises(ValueError if requested is True else Planned):
        LLMEngine(
            model,
            attention_backend="triton",
            stage_config=StageConfig(role=role),
            scheduler_config=SchedulerConfig(enable_preemption=preemption),
            execution_config=ExecutionConfig(
                packed_prefill=requested, async_scheduling=asynchronous, cuda_graphs=graphs
            ),
        )
