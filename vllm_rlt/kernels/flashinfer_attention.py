"""FlashInfer XQA paged decode kernels; no packed prefill in this backend.

Decode uses independent single-query rows, exactly like the Triton path.
SM120 (RTX 50) is the validated target; other supported majors are accepted
but unvalidated. Requires a CUDA toolkit with nvcc >= 12.9 visible through
CUDA_HOME for the just-in-time compiled kernels, and that toolkit's lib
directory on LD_LIBRARY_PATH.
"""

import importlib
import logging
from importlib.metadata import PackageNotFoundError, version

import torch

FLASHINFER_PAGE_SIZES = (16, 32, 64, 128)
FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024


class FlashInferPagedAttention:
    def __init__(self, device, dtype, head_dim, block_size):
        if device.type != "cuda" or torch.version.hip:
            raise ValueError("flashinfer currently requires an NVIDIA CUDA device")
        # float16 kernels compute correctly, but flashinfer-python 0.7.0.post1
        # crashes the process (host SIGFPE) while building the f16 JIT module
        # on first in-process use; reject instead of shipping that.
        if dtype is torch.float16:
            raise ValueError(
                "the flashinfer backend is not validated with float16; use bfloat16"
            )
        if dtype is not torch.bfloat16:
            raise ValueError("flashinfer requires bfloat16")
        # The XQA JIT module generator rejects anything else at first call;
        # validate here so the failure is an explicit startup error instead.
        if head_dim % 16 or not 16 <= head_dim <= 512:
            raise ValueError(
                "flashinfer requires head_dim divisible by 16 and in [16, 512] (XQA kernels)"
            )
        # XQA kernels are compiled per page size and only these variants ship.
        if block_size not in FLASHINFER_PAGE_SIZES:
            raise ValueError(
                "flashinfer requires --block-size one of "
                f"{FLASHINFER_PAGE_SIZES} (XQA page-size variants)"
            )
        major, minor = torch.cuda.get_device_capability(device)
        if major not in (9, 10, 12):
            raise ValueError(f"flashinfer is not supported on SM{major}{minor}")
        try:
            package_version = version("flashinfer-python")
            self.kernel = getattr(
                importlib.import_module("flashinfer.decode"),
                "trtllm_batch_decode_with_kv_cache",
            )
        except (ImportError, AttributeError, PackageNotFoundError) as error:
            raise ImportError(
                "install flashinfer with: pip install 'flashinfer-python>=0.7.0' "
                "(a CUDA toolkit with nvcc >= 12.9 must be reachable via CUDA_HOME, "
                "and its lib directory on LD_LIBRARY_PATH for the JIT modules). "
                "No backend fallback was performed."
            ) from error
        if major != 12:
            logging.getLogger(__name__).warning(
                "flashinfer attention is validated on SM120 only; SM%d%d is unvalidated",
                major,
                minor,
            )
        # Persistent, pre-zeroed scratch shared by every call. Allocated on the
        # first execution rather than here so backend selection stays free of
        # device allocations: engines always profile before CUDA-graph capture,
        # so the workspace exists by the time any graph is recorded.
        self.workspace = None
        self.info = dict(
            backend="flashinfer",
            package="flashinfer-python",
            version=package_version,
            kernel="xqa",
            page_size=block_size,
        )
        logging.getLogger(__name__).warning("Attention implementation: %s", self.info)

    def __call__(self, q, key_cache, value_cache, block_tables, context_lengths):
        # An empty batch host-crashes (SIGFPE, not an exception) inside the
        # FlashInfer dispatch; return the same empty shape the other backends do.
        if q.shape[0] == 0:
            return torch.empty_like(q)
        if self.workspace is None:
            self.workspace = torch.zeros(
                FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device
            )
        out = torch.empty_like(q)
        self.kernel(
            q,
            (key_cache, value_cache),
            self.workspace,
            # Padding columns hold -1 (synchronous), 0 (async), or stale valid
            # ids (graph replay beyond the current width). The kernel bounds
            # reads by seq_lens, and clamped ids keep any full-width access on
            # allocated memory regardless.
            block_tables=block_tables.clamp_min(0),
            seq_lens=context_lengths.to(torch.uint32),
            max_seq_len=block_tables.shape[1] * key_cache.shape[1],
            bmm1_scale=q.shape[-1] ** -0.5,
            out=out,
            q_len_per_req=1,
            # The engine pool is [blocks, page, heads, dim]; the API default is
            # HND and silently misreads NHD data.
            kv_layout="NHD",
        )
        return out
