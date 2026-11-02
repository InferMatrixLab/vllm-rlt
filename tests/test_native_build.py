"""Build selection without requiring a local GPU or compiler in CPU CI."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def setup_calls(monkeypatch):
    calls = []
    setuptools = ModuleType("setuptools")
    setuptools.setup = lambda **options: calls.append(options)
    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    return calls


@pytest.mark.parametrize(
    "cuda_version,toolkit,nvcc,visible,architecture,expected",
    [
        (None, True, True, True, None, False),
        ("13.0", False, False, True, None, False),
        ("13.0", True, False, True, None, False),
        ("13.0", True, True, False, None, False),
        ("13.0", True, True, False, "native", False),
        ("13.0", True, True, True, None, True),
        ("13.0", True, True, False, "10.3", True),
    ],
)
def test_native_build_detects_toolchain_and_target(
    monkeypatch, tmp_path, setup_calls, cuda_version, toolkit, nvcc, visible, architecture, expected
):
    home = tmp_path / "cuda"
    if nvcc:
        (home / "bin").mkdir(parents=True)
        (home / "bin" / "nvcc").touch()
    torch = ModuleType("torch")
    torch.version = SimpleNamespace(cuda=cuda_version)
    torch.cuda = SimpleNamespace(is_available=lambda: visible)
    extension = ModuleType("torch.utils.cpp_extension")
    extension.CUDA_HOME = str(home) if toolkit else None
    extension.CUDAExtension = lambda name, **options: {"name": name, **options}
    extension.BuildExtension = object()
    for name, module in {
        "torch": torch,
        "torch.utils": ModuleType("torch.utils"),
        "torch.utils.cpp_extension": extension,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    if architecture is None:
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    else:
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", architecture)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
    assert len(setup_calls) == 1
    assert bool(setup_calls[0].get("ext_modules")) == expected
    if expected:
        assert setup_calls[0]["ext_modules"][0]["name"] == "vllm_rlt._C"
        # Architecture selection is delegated to PyTorch, not hardcoded here.
        assert "-gencode" not in repr(setup_calls[0])


def test_native_build_without_torch_produces_python_package(monkeypatch, setup_calls):
    monkeypatch.setitem(sys.modules, "torch", None)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
    assert setup_calls == [{}]
