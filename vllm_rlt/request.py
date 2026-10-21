"""Request state, immutable execution descriptions and public request outputs."""

from dataclasses import dataclass, field, replace
from enum import Enum

import torch

from vllm_rlt.sampling_params import SamplingParams


class RequestPhase(str, Enum):
    WAITING = "waiting"
    RECEIVING = "receiving"
    TRANSFERRING = "transferring"
    PREFILL = "prefill"
    PREFILLING_PRELUDE = "prefilling_prelude"
    PREFILLING_RECURRENT = "prefilling_recurrent"
    PRELUDE = "prelude"
    RECURRENT = "recurrent"
    SPECULATIVE = "speculative"
    CODA = "coda"
    FINISHED = "finished"


class Operation(Enum):
    PRELUDE = "prelude"
    RECURRENT = "recurrent"
    CODA = "coda"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    phase: RequestPhase = RequestPhase.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    exit_depths: list[int] = field(default_factory=list)
    # Scheduling advances before coda's CPU output delivery.
    num_output_placeholders: int = 0
    num_speculative_rounds: int = 0
    input_token_tensor: torch.Tensor | None = field(default=None, repr=False)
    num_prefilled_tokens: int = 0
    # Split prefill. [num_prefilled_tokens, num_preluded_tokens) is the staged
    # chunk: its prelude output is resident and it advances one depth per batch.
    num_preluded_tokens: int = 0
    # Start of the resident chunk. It stays fixed while the final depth advances
    # a partial chunk and is used for runner pool row addressing.
    prefill_chunk_start: int = 0
    # Split prefill only: per-depth logical prompt frontiers.
    prefill_depth_ranges: list[list[int]] = field(default_factory=list, repr=False)
    # Host-side decode progress; ModelRunner events track pending device work.
    loops_done: int = 0
    pending_exit_depth: int | None = None
    admission_bypasses: int = 0
    remaining_probability: float = 1.0
    hidden_state: torch.Tensor | None = field(default=None, repr=False)
    # An export can retain KV/transport ownership after releasing compute slots.
    compute_released: bool = False
    finish_reason: FinishReason | None = None
    exit_trace: tuple[int, ...] = field(default=(), repr=False)

    @staticmethod
    def prefill_boundaries(
        num_tokens: int, total_depths: int, *, start: int = 0
    ) -> list[list[int]]:
        """Per-depth logical frontiers of a freshly admitted split-prefill prompt.

        ``prefill_depth_ranges[d] == [X[d + 1], X[d])`` gives the prompt tokens
        at logical depth ``d``, where ``X[d]`` counts tokens logically advanced
        through at least ``d`` depths. A new request has ``X[0]`` at the full
        prompt and every deeper frontier at zero. Advancing a submitted run
        ``[left, right)`` from depth ``d`` to ``d + 1`` sets ``X[d + 1] = right``:
        the run leaves ``boundary[d]`` and extends the right edge of
        ``boundary[d + 1]``. A run may start only at ``X[d + 1]`` so no depth
        frontier ever has a hole. Every interval is empty once logical prefill
        completes, and ``num_prefilled_tokens`` then holds the finished prefix.
        """
        return [[start, num_tokens]] + [[start, start] for _ in range(total_depths - 1)]

    def prefill_depth(self, token_index: int) -> int:
        """Logical recurrence depth of the prompt token at ``token_index``."""
        for depth, (left, right) in enumerate(self.prefill_depth_ranges):
            if left <= token_index < right:
                return depth
        raise RuntimeError(
            f"request {self.request_id!r} has no prefill depth for token {token_index}"
        )

    @property
    def num_scheduled_outputs(self) -> int:
        return len(self.generated_token_ids) + self.num_output_placeholders

    @property
    def position(self) -> int:
        return len(self.prompt_token_ids) + self.num_scheduled_outputs - 1

    @property
    def input_token_id(self) -> int:
        return self.generated_token_ids[-1]


@dataclass(frozen=True)
class StateSpan:
    """Immutable execution range borrowing the original Request for state ownership."""

    request: Request
    token_start: int
    token_count: int
    # Zero-based starting recurrent depth, not the request's completed-loop count.
    # Scheduled PRELUDE starts at 0; CODA executes no recurrent loop and uses 0
    # for boundary-layer KV addressing when required. CODA consumes the final
    # recurrent hidden state; its actual exit depth remains in request.loops_done.
    depth: int
    # None addresses the request's single resident hidden row. Otherwise the
    # span addresses [token_start-chunk_start, ...] in its resident chunk.
    chunk_start: int | None = None
    retain_chunk: bool = False
    retain_last: bool = True
    emit_gate: bool = False
    # For chunk PRELUDE, None reads the immutable request prompt range directly.
    # Explicit tokens override that source (also used by direct execution callers).
    input_tokens: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ExecutionBatch:
    """Selected request ranges bound to one model operation and loop boundary."""

    operation: Operation
    spans: tuple[StateSpan, ...]
    # Number of recurrent operations following PRELUDE, or starting at RECURRENT.
    # PRELUDE + N loops retains activations locally until the submission boundary.
    loop_count: int = 0

    def __post_init__(self):
        if self.loop_count < 0:
            raise ValueError("loop_count must be nonnegative")
        if self.operation == Operation.RECURRENT and self.loop_count == 0:
            object.__setattr__(self, "loop_count", 1)
        if self.operation == Operation.CODA and self.loop_count:
            raise ValueError("CODA cannot include recurrent work")
        for span in self.spans:
            if span.token_start < 0 or span.token_count <= 0 or span.depth < 0:
                raise ValueError("state spans require nonnegative addresses and positive rows")

    def at_loop(self, offset: int):
        """Bind a loop view without changing request progress or storage ownership."""
        if not 0 <= offset < self.loop_count:
            raise ValueError("loop offset outside execution range")
        return replace(
            self,
            operation=Operation.RECURRENT,
            loop_count=1,
            spans=tuple(replace(span, depth=span.depth + offset) for span in self.spans),
        )

    def validate_depths(self, total_depths: int):
        if any(span.depth + self.loop_count > total_depths for span in self.spans):
            raise ValueError("execution range exceeds model loop depth")

    @property
    def num_rows(self):
        return sum(span.token_count for span in self.spans)

    @property
    def has_chunks(self):
        return any(span.chunk_start is not None for span in self.spans)

    @property
    def all_chunks(self):
        return bool(self.spans) and all(span.chunk_start is not None for span in self.spans)

    @property
    def emit_gate(self):
        return any(span.emit_gate for span in self.spans)

    def addresses(self):
        ids, depths, positions = [], [], []
        for span in self.spans:
            ids.extend([span.request.request_id] * span.token_count)
            depths.extend([span.depth] * span.token_count)
            positions.extend(range(span.token_start, span.token_start + span.token_count))
        return ids, depths, positions


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    exit_depths: list[int]
    finished: bool
    finish_reason: str | None = None
    text: str = ""

    @classmethod
    def from_request(cls, request: Request):
        return cls(
            request_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids),
            token_ids=list(request.generated_token_ids),
            exit_depths=list(request.exit_depths),
            finished=request.phase == RequestPhase.FINISHED,
            finish_reason=request.finish_reason.value
            if request.finish_reason is not None
            else None,
        )
