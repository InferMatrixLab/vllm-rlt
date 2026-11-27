"""Device state kept by the loop runner between operation boundaries."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_rlt.worker.loop_graph import LoopCudaGraphs

from vllm_rlt.core.loop import LoopOp, LoopSpan


class PackedLoopState(Protocol):
    def unpack(self) -> None: ...


@dataclass(slots=True)
class PreparedBatch:
    # All views borrow MRv2's most recently prepared input and backend buffers.
    rows: tuple[tuple[str, int, int, int, bool], ...]
    descriptor: BatchExecutionDescriptor
    inputs: InputBatch
    attention: tuple[tuple[torch.Tensor, ...], torch.Tensor] | None = None
    metadata: dict[str, object] | None = None
    metadata_layout: tuple | None = None
    packed_state: PackedLoopState | None = None
    decode_slots_prepared: bool = False

    def unpack_state(self) -> None:
        if self.packed_state is not None:
            self.packed_state.unpack()
            self.packed_state = None


@dataclass(slots=True)
class LoopExecution:
    op: LoopOp
    spans: tuple[LoopSpan, ...]
    state: dict[str, dict[int, object]]
    draft_state: dict[str, dict[int, object]]
    graphs: LoopCudaGraphs | None = None
    prepared_batch: PreparedBatch | None = field(init=False, default=None)
    release_uva_inputs: Callable[[], None] | None = field(init=False, default=None)
    padded_tokens: int | None = field(init=False, default=None)
    uniform_decode: bool | None = field(init=False, default=None)
    graph_stack: torch.Tensor | None = field(init=False, default=None)
    context_limit: int | None = field(init=False, default=None)
    coda_tail_moe: bool = field(init=False, default=False)
    merged_core: bool = field(init=False, default=False)
    compact_core: bool = field(init=False, default=False)
    core_group_metadata: tuple[dict[str, object], ...] = field(init=False, default=())
    decode_slot_capacity: int = field(init=False, default=0)
    ordered: tuple[LoopSpan, ...] = field(init=False, default=())
    exit_depths: dict[str, int] = field(init=False, default_factory=dict)
    draft_tokens: dict[str, int] = field(init=False, default_factory=dict)
    uncommitted_row_ids: torch.Tensor | None = field(init=False, default=None)


loop_execution: ContextVar[LoopExecution | None] = ContextVar("loop_execution", default=None)
