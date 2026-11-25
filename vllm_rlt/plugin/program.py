"""Loop programs: what one EngineCore step executes for a loop model.

A token goes through PRELUDE, CORE at depth 0..N-1, then CODA. Mapping A
(one EngineCore step = one loop step) emits a program of length one. A model
that vLLM already runs natively keeps its baseline path as one ``FULL`` op.

Every step holds a single op kind. CODA is the only op that samples, so a step
is either all-final (CODA, FULL) or all-non-final, and the runner never has to
sample a subset of a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Op(IntEnum):
    PRELUDE = 0
    CORE = 1
    CODA = 2
    FULL = 3


@dataclass(frozen=True)
class LoopLayout:
    """Which ops a token needs for one model.

    ``has_prelude=False``: the first CORE pass embeds the tokens itself
    (Ouro). ``full_only``: run the whole token as one FULL op, i.e. vLLM's
    native forward (the Inkling v0 baseline path).
    """

    num_passes: int
    has_prelude: bool = True
    full_only: bool = False
    has_gate: bool = False  # the model has an exit gate (Ouro); Inkling does not

    def first(self) -> tuple[Op, int]:
        if self.full_only:
            return (Op.FULL, 0)
        return (Op.PRELUDE, 0) if self.has_prelude else (Op.CORE, 0)

    def next_op(self, op: Op, depth: int, last_depth: int, exited: bool = False):
        """The op after (op, depth), or None when the token is complete.

        ``last_depth`` is the request's final core depth (its loop limit);
        ``exited`` means the exit gate fired at this core pass.
        """
        if op in (Op.FULL, Op.CODA):
            return None
        if op == Op.PRELUDE:
            return (Op.CORE, 0)
        if exited or depth >= last_depth:
            # CODA keeps the depth the token left at; the runner uses it to
            # fill the skipped passes' KV (LAST_EXITED).
            return (Op.CODA, depth)
        return (Op.CORE, depth + 1)


@dataclass(frozen=True)
class LoopStep:
    """One op over a set of rows (one row per request span).

    ``can_exit``: the runner evaluates the exit gate for this row after the
    pass. ``advanced``: the scheduler already moved the row to its next op
    when the step was emitted (async, and only for rows with no gate).
    """

    op: Op
    depths: tuple[int, ...]
    req_ids: tuple[str, ...]
    num_tokens: tuple[int, ...]
    can_exit: tuple[bool, ...]
    advanced: tuple[bool, ...]
    is_dummy: bool = False
    dummy_rows: int = 0

    @property
    def is_final(self) -> bool:
        return self.op in (Op.CODA, Op.FULL)

    @property
    def total_tokens(self) -> int:
        return sum(self.num_tokens)

    @property
    def uniform_depth(self) -> bool:
        return len(set(self.depths)) <= 1

    @classmethod
    def dummy(cls, op: Op, depth: int = 0, rows: int = 1) -> LoopStep:
        """A step with no real rows, run by a DP rank that has nothing to do
        so that collectives inside the model still line up."""
        return cls(op, (depth,), (), (), (), (), True, rows)
