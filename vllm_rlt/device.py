"""Register optional device types only when explicitly requested."""

import importlib

import torch


def resolve_device(device) -> torch.device:
    if isinstance(device, str) and device.split(":", 1)[0] == "npu":
        # torch.device cannot parse private-use names until torch-npu registers
        # them. Do this in each spawned worker, without importing it on CPU/CUDA.
        importlib.import_module("torch_npu")
    return torch.device(device)
