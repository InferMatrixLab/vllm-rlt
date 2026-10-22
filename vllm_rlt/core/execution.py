"""Model operations and immutable execution bindings for scheduled work."""

from dataclasses import dataclass, replace
from enum import Enum

from vllm_rlt.request import Request


class Operation(Enum):
    PRELUDE = "prelude"
    RECURRENT = "recurrent"
    CODA = "coda"


class WorkKey(Enum):
    PROMPT_ATOMIC = ("prompt_atomic", Operation.PRELUDE)
    PROMPT_PRELUDE = ("prompt_prelude", Operation.PRELUDE)
    PROMPT_RECURRENT = ("prompt_recurrent", Operation.RECURRENT)
    DECODE_PRELUDE = ("decode_prelude", Operation.PRELUDE)
    DECODE_RECURRENT = ("decode_recurrent", Operation.RECURRENT)
    DECODE_CODA = ("decode_coda", Operation.CODA)
    SPECULATIVE = ("speculative", None)

    @property
    def operation(self) -> Operation | None:
        return self.value[1]


@dataclass(frozen=True)
class StateSpan:
    """Immutable execution range borrowing the original Request for state ownership."""

    request: Request
    token_start: int
    token_count: int
    # Zero-based starting recurrent depth, not the request's completed-loop count.
    # CODA uses zero for boundary-layer KV; its exit depth remains on Request.
    depth: int
    # None addresses the request's single resident hidden row. Otherwise the
    # span addresses [token_start - chunk_start, ...] in its resident chunk.
    chunk_start: int | None = None
    retain_chunk: bool = False
    retain_last: bool = True
    emit_gate: bool = False
    # None reads the immutable prompt range for chunk PRELUDE; explicit tokens override it.
    input_tokens: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ExecutionBatch:
    """Selected request ranges bound to one model operation and loop boundary."""

    operation: Operation
    spans: tuple[StateSpan, ...]
    # Recurrent calls following PRELUDE, or starting at RECURRENT.
    loop_count: int = 0
    has_split_prefill_work: bool | None = None

    def __post_init__(self):
        if self.loop_count < 0:
            raise ValueError("loop_count must be nonnegative")
        if self.has_split_prefill_work is None:
            object.__setattr__(self, "has_split_prefill_work", self.has_chunks)
        elif self.has_chunks and not self.has_split_prefill_work:
            raise ValueError("chunk spans require split-prefill work")
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
