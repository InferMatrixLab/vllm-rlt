"""Register native and Torch kernels once without initializing a GPU."""

from .activation import silu_and_mul

__all__ = ["silu_and_mul"]
