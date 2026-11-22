# SPDX-License-Identifier: Apache-2.0
"""Weight fusion for projections that share one input row.

Why this exists, stated as the measurement that motivates it. For a single decode row the
projections are pure GEMV and are bound by how fast the weight matrices can be streamed.
Cold-cache cuBLAS GEMV efficiency on this A100 (M=1, K=2048, bf16, weights not resident in
the 40 MB L2) rises steeply with matrix size:

| output width | bytes | GB/s | % of the 1770 GB/s measured peak |
| --- | --- | --- | --- |
| 2048 (q, k, v, o) | 8.4 MB | 597 | 34% |
| 5632 (gate, up) | 23.1 MB | 1118 | 63% |
| 6144 (fused qkv) | 25.2 MB | 1168 | 66% |
| 11264 (fused gate+up) | 46.1 MB | 1312 | 74% |

So the cost at this size is dominated by how few bytes are in flight, not by arithmetic.
Concatenating the matrices that share an input turns three 8.4 MB GEMVs into one 25.2 MB
GEMM and two 23.1 MB GEMVs into one 46.1 MB GEMM. On the real Ouro-1.4B weights one
24-layer pass drops from 1.920 ms to 1.355 ms (1.42x).

Three constraints shaped this file, and every one of them was found by a failing test
rather than by inspection. They are why the fusion is a plain cache object rather than an
``nn.Module`` wrapper:

1. Holding the owner module strongly makes the module graph cyclic, so ``state_dict`` and
   ``_apply`` recurse until the stack overflows.
2. Holding it through a ``weakref`` cannot be pickled, and the engine ships the model to a
   worker process.
3. Keeping the fused tensor as an ``nn.Parameter`` registers it, so it appears in
   ``state_dict`` and breaks checkpoint round-trips against the published key set.

What is left is a plain attribute holding a plain tensor: invisible to ``state_dict``, and
picklable.
"""

import torch

__all__ = ["FusedWeight"]


class FusedWeight:
    """A concatenated weight matrix, cached and rebuilt when its sources change.

    Not an ``nn.Module`` on purpose: the cache must stay out of ``state_dict`` (the
    checkpoint keeps the unfused keys) and out of ``.parameters()``.
    """

    def __init__(self, owner: torch.nn.Module, sources: tuple[str, ...]) -> None:
        self.owner = owner
        self.sources = tuple(sources)
        self._fused: torch.Tensor | None = None
        self._signature: tuple | None = None

    def _current(self) -> list[torch.Tensor]:
        return [getattr(self.owner, name).weight for name in self.sources]

    def weight(self) -> torch.Tensor:
        """Return the fused weight, rebuilding it if the sources were replaced.

        The signature check is what keeps this correct rather than fast-but-stale:
        checkpoint loading and dtype conversion both replace the source parameters, and a
        cached concatenation would otherwise keep computing with the previous values.
        """
        parts = self._current()
        signature = tuple((id(p), p.data_ptr(), p.dtype, tuple(p.shape)) for p in parts)
        if self._fused is None or self._signature != signature:
            self._fused = torch.cat(parts, dim=0)
            self._signature = signature
        return self._fused

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden, self.weight())

    def __repr__(self) -> str:
        return f"FusedWeight(sources={self.sources})"
