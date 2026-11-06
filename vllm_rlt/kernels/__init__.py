"""Register native and Torch kernels once without initializing a GPU."""

from .activation import silu_and_mul
from .normalization import rms_norm

__all__ = ["rms_norm", "silu_and_mul"]
