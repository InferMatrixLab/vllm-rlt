"""GPU decision policies, pinned readback, and state slot pooling for pipelined speculation."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from vllm_rlt.worker.speculative_pipeline import PipelineLane, PipelineSchedule


@dataclass
class DecisionBatch:
    """Describes a staged readback and optional proposal distributions."""

    staged: tuple[int, int]
    proposal_states: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """Policy-independent outcome consumed by the pipelined speculative driver."""

    prediction: int
    proposed: bool = False
    proposal_state: torch.Tensor | None = None
    accepted_stage: int | None = None
    proposal_action: str | None = None
    first_rejected: bool = False
    second_rejected: bool = False


class StateSlotPool:
    """Assign persistent state slot indices and recycle pruned lanes."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._free: list[int] = list(range(capacity))
        self._live: set[int] = set()

    def acquire(self) -> int:
        if not self._free:
            raise RuntimeError(
                f"pipeline requires more recurrent state slots (capacity={self.capacity})"
            )
        slot = heapq.heappop(self._free)
        self._live.add(slot)
        return slot

    def retain(self, slots: Iterable[int]) -> None:
        retained = set(slots)
        released = self._live - retained
        self._live.intersection_update(retained)
        for slot in released:
            heapq.heappush(self._free, slot)


class DecisionReadback:
    """Copy compact GPU decisions to CPU with one synchronization event.

    Compatible with both CUDA (using pinned host memory and non-blocking copy)
    and CPU fallback (for unit tests).
    """

    def __init__(self, max_rows: int, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        if self.is_cuda:
            self.host = torch.empty(max_rows * 5, dtype=torch.int64, pin_memory=True)
            self.ready = torch.cuda.Event()
        else:
            self.host = torch.empty(max_rows * 5, dtype=torch.int64)
            self.ready = None

    def stage(self, decisions: torch.Tensor) -> tuple[int, int]:
        rows, columns = decisions.shape
        target = self.host[: rows * columns].view(rows, columns)
        if self.is_cuda:
            target.copy_(decisions, non_blocking=True)
            self.ready.record()
        else:
            target.copy_(decisions)
        return rows, columns

    def read(self, staged: tuple[int, int]) -> list[list[int]]:
        rows, columns = staged
        if self.is_cuda and self.ready is not None:
            self.ready.synchronize()
        return self.host[: rows * columns].view(rows, columns).tolist()


class GreedyDecisionPolicy:
    """Select greedy argmax tokens directly on GPU and read back asynchronously."""

    def __init__(self, max_rows: int, device: torch.device | str) -> None:
        self.readback = DecisionReadback(max_rows, device)

    def queue(
        self,
        logits: torch.Tensor,
        endpoints: Mapping[int, str],
        lanes: Sequence[PipelineLane],
    ) -> DecisionBatch:
        del endpoints, lanes
        choices = torch.argmax(logits, dim=-1, keepdim=True)
        return DecisionBatch(self.readback.stage(choices))

    def read(
        self,
        lanes: Sequence[PipelineLane],
        endpoints: Mapping[int, str],
        batch: DecisionBatch,
    ) -> dict[int, list[int]]:
        del lanes
        decisions = self.readback.read(batch.staged)
        return dict(zip(endpoints, decisions, strict=True))

    def resolve(
        self,
        lane: PipelineLane,
        endpoint_kind: str,
        decision: list[int],
        proposal_state: torch.Tensor | None,
        schedule: PipelineSchedule,
    ) -> DecisionOutcome:
        prediction = decision[0]
        proposed = endpoint_kind == "q1" or (
            endpoint_kind == "q2" and prediction != lane.first_token
        )

        accepted_stage = None
        first_rejected = second_rejected = False

        if endpoint_kind == "final" and lane.first_token is not None:
            first_rejected = prediction != lane.first_token
            if not first_rejected:
                accepted_stage = schedule.proposal_stage
            elif lane.second_token is not None:
                second_rejected = prediction != lane.second_token
                if not second_rejected:
                    accepted_stage = schedule.second_stage

        return DecisionOutcome(
            prediction=prediction,
            proposed=proposed,
            proposal_state=proposal_state,
            accepted_stage=accepted_stage,
            proposal_action=("argmax" if endpoint_kind == "q2" and proposed else None),
            first_rejected=first_rejected,
            second_rejected=second_rejected,
        )
