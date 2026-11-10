"""Inference RMSNorm with an optional residual and a PyTorch reference."""

import torch

from vllm_rlt.kernels._extension import try_load_extension


def rms_norm_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    value = hidden.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    result = (value * weight).to(torch.promote_types(hidden.dtype, weight.dtype))
    return result if residual is None else residual + result


_library = torch.library.Library("vllm_rlt", "FRAGMENT")
if not try_load_extension():
    _library.define(
        "rms_norm(Tensor input, Tensor weight, float epsilon, Tensor? residual=None) -> Tensor"
    )


def _reference_dispatch(hidden, weight, epsilon, residual=None):
    return rms_norm_reference(hidden, weight, epsilon, residual)


_library.impl("rms_norm", _reference_dispatch, "CompositeExplicitAutograd")


def rms_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use the native CUDA operator where its inference layout contract applies."""
    # The module historically accepts autograd, mixed dtypes and arbitrary views.
    # Use the same arithmetic contract in Torch for unsupported inputs.
    supported = (
        hidden.is_cuda
        and 1 <= hidden.ndim <= 4
        and hidden.stride(-1) == 1
        and hidden.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and weight.ndim == 1
        and weight.is_contiguous()
        and weight.shape[0] == hidden.shape[-1]
        and weight.dtype == hidden.dtype
        and weight.device == hidden.device
        and (
            residual is None
            or (
                residual.shape == hidden.shape
                and residual.device == hidden.device
                and residual.dtype == hidden.dtype
                and residual.stride(-1) == 1
            )
        )
        and (
            not torch.is_grad_enabled()
            or not (
                hidden.requires_grad
                or weight.requires_grad
                or (residual is not None and residual.requires_grad)
            )
        )
    )
    if not supported:
        return rms_norm_reference(hidden, weight, epsilon, residual)
    return torch.ops.vllm_rlt.rms_norm(hidden, weight, epsilon, residual)
