"""Select paged Attention implementations and declare their execution capabilities."""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_rlt.kernels.flash_attention import FLASH_BACKENDS, FlashPagedAttention
from vllm_rlt.kernels.flashinfer_attention import FlashInferPagedAttention
from vllm_rlt.kernels.paged_attention import torch_paged_attention, triton_paged_attention


@dataclass(frozen=True)
class BackendCapabilities:
    """Features relevant to execution routing, independent of kernel class."""

    packed_prefill: bool = False
    cuda_graphs: bool = False
    async_scheduling: bool = False


@dataclass
class AttentionBackend:
    """Thin kernel adapter with declared features and selection diagnostics."""

    kernel: Callable
    capabilities: BackendCapabilities
    info: dict

    def __call__(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)

    def prefill(self, *args, **kwargs):
        return self.kernel.prefill(*args, **kwargs)

    @property
    def generation(self):
        # Retain the existing Flash adapter's diagnostic attribute.
        return getattr(self.kernel, "generation", None)


_TORCH_CAPABILITIES = BackendCapabilities()
_TRITON_CAPABILITIES = BackendCapabilities(cuda_graphs=True, async_scheduling=True)
_FLASH_CAPABILITIES = {
    2: BackendCapabilities(cuda_graphs=True, async_scheduling=True),
    3: BackendCapabilities(cuda_graphs=True, async_scheduling=True),
    4: BackendCapabilities(packed_prefill=True, cuda_graphs=True, async_scheduling=True),
}
_FLASHINFER_CAPABILITIES = BackendCapabilities(cuda_graphs=True, async_scheduling=True)
_AUTO_PRIORITY = ("flash_attn_4", "flash_attn_3", "flash_attn_2", "flashinfer", "triton", "torch")


def backend_capabilities(implementation) -> BackendCapabilities:
    return implementation.capabilities


def _auto_flashinfer_supported(device) -> bool:
    """Gate automatic FlashInfer selection to the validated SM120 target.

    Other architectures the kernel accepts remain reachable through explicit
    `--attention-backend flashinfer` opt-in; auto keeps its pre-FlashInfer
    fallback behavior everywhere else.
    """
    if device.type != "cuda" or torch.version.hip:
        return False
    return torch.cuda.get_device_capability(device)[0] == 12


def validate_backend_name(backend: str) -> None:
    if backend not in {"auto", "torch", "triton", "flashinfer", *FLASH_BACKENDS}:
        raise ValueError("unknown attention backend")


def _create_explicit(backend, device, dtype, head_dim, block_size):
    if backend == "triton":
        if device.type != "cuda":
            raise ValueError("the Triton attention backend requires a CUDA or ROCm device")
        if head_dim > 256:
            raise ValueError("the Triton attention backend supports head_dim <= 256")
        if not torch.version.hip and torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("the Triton attention backend requires NVIDIA Compute Capability 8.0+")
        # Probe the actual module, including its dependency imports, before selection.
        importlib.import_module("vllm_rlt.kernels.triton_attention")
        return AttentionBackend(triton_paged_attention, _TRITON_CAPABILITIES, {"backend": backend})
    if backend == "flashinfer":
        kernel = FlashInferPagedAttention(device, dtype, head_dim, block_size)
        return AttentionBackend(kernel, _FLASHINFER_CAPABILITIES, dict(kernel.info))
    if backend in FLASH_BACKENDS:
        kernel = FlashPagedAttention(device, dtype, head_dim, block_size, backend)
        return AttentionBackend(kernel, _FLASH_CAPABILITIES[kernel.generation], dict(kernel.info))
    return AttentionBackend(torch_paged_attention, _TORCH_CAPABILITIES, {"backend": "torch"})


def create_backend(
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    block_size: int,
    *,
    required_capabilities: BackendCapabilities = BackendCapabilities(),
) -> AttentionBackend:
    """Explicit choices fail without fallback; auto tries FA4/3/2, FlashInfer
    (SM120 only), Triton, then Torch.

    Each candidate validates hardware/configuration and imports its implementation.
    Only compatibility and import failures permit auto to try the next candidate;
    kernel execution errors never trigger a runtime fallback. Execution requirements
    filter candidates before memory profiling or device-buffer allocation.
    """
    validate_backend_name(backend)
    candidates = _AUTO_PRIORITY if backend == "auto" else (backend,)
    skipped = []
    if backend == "auto" and device.type != "cuda":
        candidates = ("torch",)
        skipped.append(f"{device.type.upper()} uses Torch reference attention")
    for candidate in candidates:
        if (
            backend == "auto"
            and candidate == "flashinfer"
            and not _auto_flashinfer_supported(device)
        ):
            skipped.append("flashinfer: automatic selection is validated on SM120 only")
            continue
        try:
            implementation = _create_explicit(candidate, device, dtype, head_dim, block_size)
            missing = [
                name
                for name in ("packed_prefill", "cuda_graphs", "async_scheduling")
                if getattr(required_capabilities, name)
                and not getattr(implementation.capabilities, name)
            ]
            if missing:
                raise ValueError(f"{candidate} lacks required capabilities: {', '.join(missing)}")
        except (ValueError, ImportError, OSError) as error:
            if backend != "auto":
                raise
            skipped.append(f"{candidate}: {error}")
            continue
        selected = (
            f"flash_attn_{implementation.generation}" if candidate in FLASH_BACKENDS else candidate
        )
        reason = (
            "first compatible candidate in FA4 > FA3 > FA2 > FlashInfer > Triton > Torch priority"
            if backend == "auto"
            else f"explicitly requested {backend}"
        )
        if skipped:
            reason += "; " + "; ".join(skipped)
        implementation.info.update(
            requested_backend=backend, selected_backend=selected, selection_reason=reason
        )
        logging.getLogger(__name__).info("Attention selection: %s", implementation.info)
        return implementation
    raise ValueError("no compatible attention backend; " + "; ".join(skipped))
