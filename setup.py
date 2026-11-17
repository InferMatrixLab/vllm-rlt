"""Build native kernels automatically when a CUDA build environment is available."""

import os
from pathlib import Path

from setuptools import setup


def native_build_options():
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print("CUDA extension skipped: install PyTorch and use --no-build-isolation to build it.")
        return {}

    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    if torch.version.cuda is None or CUDA_HOME is None:
        print("CUDA extension skipped: CUDA-enabled PyTorch and the CUDA toolkit are required.")
        return {}
    if not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
        print("CUDA extension skipped: nvcc was not found in the CUDA toolkit.")
        return {}
    targets = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if targets in ("", "native") and not torch.cuda.is_available():
        print("CUDA extension skipped: no visible GPU to detect a target architecture.")
        return {}

    # CUDA_HOME and visible-device architecture detection are owned by PyTorch.
    # Its standard architecture override also supports cross-compiling without a GPU.
    return {
        "ext_modules": [
            CUDAExtension(
                "vllm_rlt._C",
                sources=[
                    "csrc/torch_bindings.cpp",
                    "csrc/activation/silu_and_mul.cu",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
            )
        ],
        "cmdclass": {"build_ext": BuildExtension},
    }


setup(**native_build_options())
