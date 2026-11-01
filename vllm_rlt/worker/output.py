"""Host results; execution snapshots are borrowed from their submission.

The batch retains Request objects, whose identity rejects cancelled/reused IDs
in this single-process runtime. Only delayed delivery needs depth/index snapshots.
"""

from dataclasses import dataclass

from vllm_rlt.request import Request


@dataclass(frozen=True)
class ExitSignal:
    request: Request
    position: int
    depth: int
    score: float


@dataclass(frozen=True)
class SpeculativeResult:
    token_ids: list[int]
    accepted_count: int
    # Exclusive KV extent still needed by retained pipeline tokens.
    kv_frontier: int | None = None


@dataclass(frozen=True, kw_only=True)
class ModelRunnerOutput:
    gate_scores: list[float] | None = None
    token_ids: list[int] | None = None
    completion: object = None
    depths: tuple[int, ...] = ()
    output_indices: tuple[int, ...] = ()
    speculative: list[SpeculativeResult] | None = None
    frontiers: list[int] | None = None
