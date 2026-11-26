"""Loop programs: what one EngineCore step executes for a loop model.

A token goes through PRELUDE, CORE at depth 0..N-1, then CODA. Mapping A
(one EngineCore step = one loop step) emits a program of length one.

Every step holds a single op kind. CODA is the only op that samples, so a step
is either all-final (CODA) or all-non-final, and the runner never has to
sample a subset of a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Op(IntEnum):
    PRELUDE = 0
    CORE = 1
    CODA = 2


@dataclass(frozen=True)
class LoopLayout:
    """Which ops a token needs for one model.

    ``has_prelude=False``: the first CORE pass embeds the tokens itself
    (Ouro).
    """

    num_passes: int  # passes one token runs by default (the configured count)
    has_prelude: bool = True
    has_gate: bool = False  # the model has an exit gate (Ouro); Inkling does not
    varloop: bool = False  # Inkling VarLoop: the pass count may be chosen
    share_cache: bool = False  # one KV cache for all passes (VarLoop option)
    mixed_depth: bool = False  # one CORE step may hold rows at different depths

    def limit(self, params) -> int:
        """Core passes one request runs.

        Ouro: up to its gate / ``max_loops`` / ``target_loops``. Inkling has no
        gate. Its pass count is the one the checkpoint was served with
        (``num_loop_passes``: Fixed always its own, VarLoop any 1..N chosen at
        serve time) and every request runs them all. A VarLoop checkpoint
        served with ``loop_share_cache`` can also stop earlier per request
        (``rlt_target_loops``): all passes read and write one KV cache, so the
        last executed pass already stands in for the skipped ones and no fill
        is needed. Without a shared cache a skipped pass would leave a hole in
        its own cache, so the request keeps all passes.
        """
        if self.has_gate:
            return params.limit(self.num_passes)
        if self.varloop and self.share_cache and params.target_loops:
            return max(1, min(params.target_loops, self.num_passes))
        return self.num_passes

    def rank(self, op: Op, depth: int) -> int:
        """Position of (op, depth) in the program; orders the chunks of one
        prompt (a chunk may run an op only after the chunk before it has)."""
        if op == Op.PRELUDE:
            return 0
        if op == Op.CORE:
            return depth + 1
        return self.num_passes + 1

    def first(self) -> tuple[Op, int]:
        return (Op.PRELUDE, 0) if self.has_prelude else (Op.CORE, 0)

    def next_op(self, op: Op, depth: int, last_depth: int, exited: bool = False):
        """The op after (op, depth), or None when the token is complete.

        ``last_depth`` is the request's final core depth (its loop limit);
        ``exited`` means the exit gate fired at this core pass.
        """
        if op == Op.CODA:
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
    spans: tuple[int, ...] = ()  # span id per row (a request can have several)

    @property
    def is_final(self) -> bool:
        return self.op == Op.CODA

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
