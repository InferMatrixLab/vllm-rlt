"""Data-parallel alignment of the loop step.

With DP + EP every rank must enter the same MoE collectives in the same order,
so all ranks must agree on which op the next step runs. Each rank reports what
it could run; ``choose_op`` is a deterministic function of everything that was
reported, so every rank computes the same answer. A rank with no rows of the
chosen op runs a dummy step of that op (``LoopStep.dummy``) instead of idling.

``Aligner`` is the seam: single-rank serving uses ``NoAlign``; a DP engine
supplies a gather over the DP CPU group. This is the interface reserved for
GLM-5.4 (DP + EP); the DP gather itself is unverified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

Candidates = Mapping[tuple[int, int], int]  # (op, depth) -> rows ready on this rank


def choose_op(per_rank: Sequence[Candidates]) -> tuple[int, int] | None:
    """Most total rows wins; ties go to the earlier op, then the shallower depth."""
    totals: dict[tuple[int, int], int] = {}
    for rank in per_rank:
        for key, rows in rank.items():
            totals[key] = totals.get(key, 0) + rows
    if not totals:
        return None
    return max(totals, key=lambda k: (totals[k], -k[0], -k[1]))


class Aligner(Protocol):
    def align(self, local: Candidates) -> tuple[int, int] | None: ...


class NoAlign:
    """Single rank: the local choice is the global one."""

    def align(self, local: Candidates) -> tuple[int, int] | None:
        return choose_op([local])


class DPAlign:
    """Gather every DP rank's candidates and choose once.

    ``gather`` takes this rank's candidates and returns every rank's, in rank
    order. vLLM's DP engine core owns a stateless process group whose gather is
    ``StatelessProcessGroup.all_gather_obj``; pass that in. Unverified: GLM-5.4
    (DP + EP) is the first model that needs it.
    """

    def __init__(self, gather) -> None:
        self.gather = gather

    def align(self, local: Candidates) -> tuple[int, int] | None:
        return choose_op(self.gather(dict(local)))
