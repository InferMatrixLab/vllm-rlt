"""Data-parallel alignment of the loop step.

With DP + EP every rank must enter the same MoE collectives in the same order,
so all ranks must run the same op in the same step. Each rank offers what it
could run; ``choose_op`` is a deterministic function of everything offered, so
every rank computes the same answer.

Alignment runs in the *workers*, not the scheduler. vLLM's DP engine only calls
``schedule()`` on a rank that has requests, but an idle rank still runs a dummy
batch every step, and that dummy batch is where vLLM already synchronizes its
CUDA-graph shape across ranks (``dispatch_cg_and_sync_dp``). Doing the same here
means an idle rank takes part without any change to the engine loop.

A rank whose own step differs from the aligned op runs a dummy of the aligned
op and reports its rows as skipped; the scheduler leaves those rows where they
were. Unverified on a real DP + EP deployment (GLM-5.4 is the first user).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

Candidates = Mapping[tuple[int, int], int]  # (op, depth) -> rows ready on this rank
NUM_OPS = 4


def choose_op(per_rank: Sequence[Candidates]) -> tuple[int, int] | None:
    """Most total rows wins; ties go to the earlier op, then the shallower depth."""
    totals: dict[tuple[int, int], int] = {}
    for rank in per_rank:
        for key, rows in rank.items():
            totals[key] = totals.get(key, 0) + rows
    if not totals:
        return None
    return max(totals, key=lambda k: (totals[k], -k[0], -k[1]))


def _slot(key: tuple[int, int], num_passes: int) -> int:
    op, depth = key  # depth -1 means "any", otherwise 0..num_passes-1
    if not (0 <= op < NUM_OPS and -1 <= depth < num_passes):
        raise ValueError(f"candidate {key} is outside the program of {num_passes} passes")
    return op * (num_passes + 1) + depth + 1


def encode(local: Candidates, rank: int, size: int, num_passes: int):
    """A [size, slots] table with only this rank's row filled. Summing the
    tables of all ranks (an all-reduce) gives every rank every row."""
    import torch

    table = torch.zeros(size, NUM_OPS * (num_passes + 1), dtype=torch.int32)
    for key, rows in local.items():
        table[rank, _slot(key, num_passes)] = rows
    return table


def decode(table, num_passes: int) -> list[dict[tuple[int, int], int]]:
    width = num_passes + 1
    ranks = []
    for row in table.tolist():
        ranks.append({(s // width, s % width - 1): n for s, n in enumerate(row) if n})
    return ranks


class Aligner(Protocol):
    def align(self, local: Candidates) -> tuple[int, int] | None: ...


class NoAlign:
    """Single rank: the local choice is the global one."""

    def align(self, local: Candidates) -> tuple[int, int] | None:
        return choose_op([local])


class DPOpAlign:
    """All-reduce the candidate tables over the DP CPU group.

    ``all_reduce`` sums a small CPU tensor in place across DP ranks, e.g.
    ``get_dp_group().cpu_all_reduce_`` inside a vLLM worker.
    """

    def __init__(self, rank: int, size: int, num_passes: int, all_reduce: Callable) -> None:
        self.rank, self.size, self.num_passes = rank, size, num_passes
        self.all_reduce = all_reduce

    def align(self, local: Candidates) -> tuple[int, int] | None:
        table = encode(local, self.rank, self.size, self.num_passes)
        self.all_reduce(table)
        return choose_op(decode(table, self.num_passes))
