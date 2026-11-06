"""Accelerator platform facts used by backend selection and storage policies."""

import torch


def is_rocm() -> bool:
    """Whether PyTorch targets AMD ROCm (HIP) instead of NVIDIA CUDA."""
    return torch.version.hip is not None


def is_cuda() -> bool:
    """Whether PyTorch targets NVIDIA CUDA."""
    return torch.version.cuda is not None


def supports_flash_attention() -> bool:
    """Whether the FlashAttention packages can run on this build."""
    return is_cuda() and not is_rocm()
