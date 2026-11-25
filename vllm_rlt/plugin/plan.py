"""Choosing which ready rows form the next loop step. Pure Python, no vLLM.

Every step holds a single op kind, and (unless mixed depth is enabled) a single
core depth. The group chosen is the one with the most ready rows, which is what
fills a batch; ``choose_op`` in ``parallel.py`` makes that choice, so one
rule serves one rank and many. A row that has waited ``wait_limit`` steps makes
its group win, so a small group cannot starve behind one that keeps refilling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vllm_rlt.plugin.program import Op

STARVATION_WEIGHT = 10**6


@dataclass(frozen=True)
class Ready:
    req_id: str
    op: Op
    depth: int
    tokens: int
    waited: int = 0


def group_key(row: Ready, allow_mixed_depth: bool) -> tuple[int, int]:
    """(op, depth). Only CORE is split by depth; depth -1 means 'any depth'.

    A CODA row's depth is the depth its token left the loop at, which does not
    change what the CODA op computes, so those rows share one group.
    """
    depth = row.depth if (row.op == Op.CORE and not allow_mixed_depth) else -1
    return (int(row.op), depth)


def candidates(
    ready: Sequence[Ready], allow_mixed_depth: bool, wait_limit: int
) -> dict[tuple[int, int], int]:
    """Rows ready per (op, depth) on this rank, with a starving group boosted."""
    local: dict[tuple[int, int], int] = {}
    for row in ready:
        key = group_key(row, allow_mixed_depth)
        local[key] = local.get(key, 0) + 1
    starving = [r for r in ready if r.waited >= wait_limit]
    if starving:
        worst = max(starving, key=lambda r: r.waited)
        local[group_key(worst, allow_mixed_depth)] += STARVATION_WEIGHT
    return local


def take_rows(
    ready: Sequence[Ready],
    key: tuple[int, int],
    max_rows: int,
    token_budget: int,
    allow_mixed_depth: bool = False,
) -> list[Ready]:
    """Rows of ``key`` in arrival order, within the row and token limits.

    At least one row is always taken: a span was sized to the token budget
    when it was opened, so a single span always fits.
    """
    chosen: list[Ready] = []
    tokens = 0
    for row in ready:
        if group_key(row, allow_mixed_depth) != key:
            continue
        if chosen and (len(chosen) >= max_rows or tokens + row.tokens > token_budget):
            break
        chosen.append(row)
        tokens += row.tokens
    return chosen
