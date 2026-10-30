"""Where fused elementwise kernels run: inside CUDA graph capture only, or everywhere.

A fused Triton kernel replaces several PyTorch kernels, but each Triton launch from Python
costs more host time than a PyTorch op. Inside a captured CUDA graph that cost is paid
once at capture; in eager execution (prefill, Ouro prelude/coda, sync eager decode) it is
paid every call, and on GH200's Grace CPU it outweighs the saved launches: sync eager
Ouro-1.4B decode ran at 0.40x and prefill steps took 33-46% longer with the fused set on.

VLLM_RLT_FUSED_SCOPE=graph (default) uses the fused kernels only while a CUDA graph is
being captured; =always also uses them eagerly. The exact kernels are bitwise identical
to the eager reference, so the scope never changes numerics in exact mode.
"""

import os

import torch

FUSED_SCOPE = os.environ.get("VLLM_RLT_FUSED_SCOPE", "graph")
if FUSED_SCOPE not in ("graph", "always"):
    raise ValueError("VLLM_RLT_FUSED_SCOPE must be 'graph' or 'always'")


_CUDA = None


def fused_allowed() -> bool:
    """Whether a fused kernel may replace the eager reference at this call site."""
    global _CUDA
    if FUSED_SCOPE == "always":
        return True
    if _CUDA is None:
        # CPU-only builds have no capture query at all; nothing is ever captured there.
        _CUDA = torch.cuda.is_available()
    return _CUDA and torch.cuda.is_current_stream_capturing()
