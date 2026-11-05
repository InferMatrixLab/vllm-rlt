"""Module-level scratch caches of the fused sparse decode kernels.

The kernels reuse their scratch across layers of one stream, so the buffers live
outside any single call. They are keyed by ``(device, stream)`` so two engines on
different streams do not share them — but nothing owns an entry, which means a
closed engine's tensors stay reachable for the life of the process. The capture
score buffer alone is ``rows * heads * capacity`` floats, so a long-context run
leaves a large allocation behind.

This module is deliberately free of Triton imports: the engine that owns the
stream has to be able to release the scratch on a CPU-only install too.
"""

import torch

# Scratch of one capture launch, per (device, stream).
CAPTURE_SCRATCH: dict = {}

# Scratch of one consume launch, per (device, stream).
CONSUME_SCRATCH: dict = {}

# ``key_retention`` as a float64 device scalar, per (device, retention).
RETENTION_SCALARS: dict[tuple[torch.device, float], torch.Tensor] = {}


def release_workspace(device=None) -> None:
    """Drop the scratch cached for ``device``, or for every device.

    Call this only after the caller has drained the stream, or a kernel may
    still be reading the tensors. Releasing too much is never a correctness
    problem for the next user — an entry is rebuilt lazily on demand — only an
    extra allocation, so tearing down a whole device is safe when several
    engines share it.
    """
    for cache in (CAPTURE_SCRATCH, CONSUME_SCRATCH, RETENTION_SCALARS):
        for key in [key for key in cache if device is None or key[0] == device]:
            del cache[key]
