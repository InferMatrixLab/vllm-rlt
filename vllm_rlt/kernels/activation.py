"""MLP activation operators and their unfused correctness reference."""

import torch
import torch.nn.functional as F

from vllm_rlt.kernels._extension import try_load_extension


def _validate_inputs(gate: torch.Tensor, up: torch.Tensor) -> None:
    if gate.shape != up.shape:
        raise ValueError("silu_and_mul expects inputs with the same shape")
    if gate.device != up.device:
        raise ValueError("silu_and_mul expects inputs on the same device")
    if gate.dtype != up.dtype:
        raise TypeError("silu_and_mul expects inputs with the same dtype")
    if gate.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("silu_and_mul supports float32, float16 and bfloat16")


def silu_and_mul_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute ``F.silu(gate) * up`` without broadcasting or modifying either input.

    Inputs have equal shape, dtype and device. The result has the same shape,
    dtype and device and owns new storage. Noncontiguous inputs are supported
    by this reference; the native entry requires contiguous inputs.
    """
    _validate_inputs(gate, up)
    return F.silu(gate) * up


# Keep the library alive: Python registrations are removed when it is destroyed.
# Native CUDA takes precedence over the composite Torch implementation; CPU and
# installations without the optional extension use the same schema and fallback.
_library = torch.library.Library("vllm_rlt", "FRAGMENT")
if not try_load_extension():
    _library.define("silu_and_mul(Tensor gate, Tensor up) -> Tensor")
_library.impl("silu_and_mul", silu_and_mul_reference, "CompositeExplicitAutograd")

silu_and_mul = torch.ops.vllm_rlt.silu_and_mul
