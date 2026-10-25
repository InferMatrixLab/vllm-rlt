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
    """One request's committed tokens from one round.

    ``token_ids[:accepted_count]`` are accepted draft tokens; ``token_ids[accepted_count]``
    is the correction, or the bonus when every draft was accepted. Token ``j`` comes from
    verification row ``j``.
    """

    token_ids: list[int]
    accepted_count: int
    # Aligned with token_ids when the request asked for logprobs; else None. Each is the
    # target's logprob of the committed token on its verification row, never the draft's.
    logprobs: list[float] | None = None


@dataclass(frozen=True, kw_only=True)
class ModelRunnerOutput:
    gate_scores: list[float] | None = None
    token_ids: list[int] | None = None
    # CODA: FP32 logprob of each token_ids entry; None when no row asked. Rows that did
    # not ask hold arbitrary values.
    logprobs: list[float] | None = None
    completion: object = None
    depths: tuple[int, ...] = ()
    output_indices: tuple[int, ...] = ()
    speculative: list[SpeculativeResult] | None = None
    frontiers: list[int] | None = None
