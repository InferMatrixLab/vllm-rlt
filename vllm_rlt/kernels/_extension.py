"""Load the optional native library once, outside kernel implementation code."""

from functools import cache
from importlib import import_module


@cache
def try_load_extension() -> bool:
    """Load once; a missing optional extension is cached as unavailable."""
    try:
        import_module("vllm_rlt._C")
    except ModuleNotFoundError as exc:
        if exc.name != "vllm_rlt._C":
            raise
        return False
    return True


def load_extension() -> None:
    if not try_load_extension():
        raise RuntimeError(
            "Native CUDA kernels are not built. See docs/kernels.md for build instructions."
        )
