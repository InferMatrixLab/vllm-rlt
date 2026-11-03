"""Inference RMSNorm with an optional residual and a PyTorch reference."""

from functools import cache

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
    result = weight * value.to(hidden.dtype)
    return result if residual is None else residual + result


_library = torch.library.Library("vllm_rlt", "FRAGMENT")
if not try_load_extension():
    _library.define(
        "rms_norm(Tensor input, Tensor weight, float epsilon, Tensor? residual=None, "
        "bool match_torch=False) -> Tensor"
    )


def _reference_dispatch(hidden, weight, epsilon, residual=None, match_torch=False):
    return rms_norm_reference(hidden, weight, epsilon, residual)


_library.impl("rms_norm", _reference_dispatch, "CompositeExplicitAutograd")


@cache
def _torch_order_validated(device_index: int) -> bool:
    # torch.mean's reduction tree is an implementation detail. Only use the
    # compatibility kernel for the framework and GPU on which it was checked.
    return torch.__version__.split("+", 1)[0] == "2.13.0" and torch.cuda.get_device_capability(
        device_index
    ) == (10, 3)


def rms_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
    *,
    match_torch: bool = False,
) -> torch.Tensor:
    """Use the native CUDA operator where its inference layout contract applies."""
    # The module historically accepts autograd, mixed dtypes and arbitrary views.
    # Keep those semantics in Torch rather than copying or narrowing inputs.
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
    if match_torch:
        # The model-compatible kernel reproduces torch.mean's CUDA reduction
        # order only for these shapes and alignments. Preserve the Torch path
        # elsewhere instead of silently changing early-exit behavior.
        compatible = (
            hidden.ndim == 2
            and hidden.shape[-1] == 2048
            and hidden.dtype in (torch.float16, torch.bfloat16)
            and _torch_order_validated(hidden.device.index)
            and hidden.data_ptr() % 8 == 0
            and weight.data_ptr() % 8 == 0
            and (hidden.shape[0] <= 1 or hidden.stride(0) % 4 == 0)
            and (
                residual is None
                or (
                    residual.data_ptr() % 8 == 0
                    and (residual.shape[0] <= 1 or residual.stride(0) % 4 == 0)
                )
            )
        )
        if not compatible:
            return rms_norm_reference(hidden, weight, epsilon, residual)
    return torch.ops.vllm_rlt.rms_norm(hidden, weight, epsilon, residual, match_torch)
