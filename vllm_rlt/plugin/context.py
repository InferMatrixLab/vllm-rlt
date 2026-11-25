"""What the runner tells a loop model before each step, and helpers around it.

``state`` is the carry between ops, a tensor (Ouro: hidden states, token dim
0) or a ``TokenState`` (Inkling: several tensors whose token dims differ).
``out`` is filled by the model's forward with the state after the op.

Rows can sit at different CORE depths (``mixed``): ``depth`` is then -1,
``row_depth`` gives each token's depth and ``group_rows`` the token indices per
depth, both computed on the host by the runner so the model never syncs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_rlt.plugin.program import Op


@dataclass
class LoopContext:
    op: Op
    depth: int  # uniform depth, or -1 when rows are at different depths
    state: Any = None  # merged input state; None when the op builds it itself
    row_depth: torch.Tensor | None = None  # [tokens] int32, mixed only
    group_rows: dict[int, torch.Tensor] = field(default_factory=dict)
    out: Any = None  # state after the op, set by the model

    @property
    def mixed(self) -> bool:
        return self.depth < 0


@dataclass
class TokenState:
    """Named tensors that all carry a token dimension (possibly different axes).

    ``meta`` holds integers that are the same for every token of one op (for
    Inkling: how many AttnRes blocks are complete); merging states checks they
    agree, because rows that would disagree cannot share a step."""

    tensors: dict[str, torch.Tensor]
    dims: dict[str, int]
    meta: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def narrow(self, start: int, length: int) -> TokenState:
        return TokenState(
            {k: v.narrow(self.dims[k], start, length) for k, v in self.tensors.items()},
            self.dims,
            dict(self.meta),
        )


def state_slice(state: Any, start: int, length: int) -> Any:
    """Tokens ``[start, start + length)`` of a state (a copy, so later ops that
    reuse the source buffers cannot change it)."""
    if isinstance(state, TokenState):
        part = state.narrow(start, length)
        copied = {k: v.contiguous().clone() for k, v in part.tensors.items()}
        return TokenState(copied, part.dims, part.meta)
    return state.narrow(0, start, length).clone()


def state_cat(states: list[Any]) -> Any:
    if isinstance(states[0], TokenState):
        dims, meta = states[0].dims, states[0].meta
        for s in states[1:]:
            if s.dims != dims or s.meta != meta:
                raise ValueError(
                    f"rows in one step carry different loop state ({meta} vs {s.meta}); "
                    "a step must hold one op at one depth"
                )
        merged = {k: torch.cat([s.tensors[k] for s in states], dim=dims[k]) for k in dims}
        return TokenState(merged, dims, dict(meta))
    return torch.cat(states, dim=0)


def pad_dim(t: torch.Tensor, dim: int, rows: int, value: float = 0.0) -> torch.Tensor:
    """Pad ``t`` along ``dim`` up to ``rows`` (vLLM pads a step to a graph or
    data-parallel size and the model must work on the padded batch)."""
    short = rows - t.shape[dim]
    if short <= 0:
        return t
    shape = list(t.shape)
    shape[dim] = short
    return torch.cat([t, t.new_full(shape, value)], dim=dim)


def pad_rows(t: torch.Tensor, rows: int, value: float = 0.0) -> torch.Tensor:
    return pad_dim(t, 0, rows, value)


def pad_state(state: Any, rows: int) -> Any:
    """Pad every token axis of a state to ``rows`` tokens."""
    if isinstance(state, TokenState):
        padded = {k: pad_dim(v, state.dims[k], rows) for k, v in state.tensors.items()}
        return TokenState(padded, state.dims, dict(state.meta))
    return pad_rows(state, rows)


@contextmanager
def masked_slots(layer_name: str, row_depth: torch.Tensor | None, depth: int):
    """Let one pass's attention layer write KV only for the tokens at ``depth``.

    Every other token's slot becomes PAD (-1), which vLLM's cache-write kernels
    skip, so a step that mixes depths still writes each token's KV into its own
    pass's cache. Only this layer's entry is swapped, and restored afterwards;
    the runner's own slot mapping (used to fill skipped passes) is untouched.
    """
    from vllm.forward_context import get_forward_context

    slots = get_forward_context().slot_mapping
    if row_depth is None or not isinstance(slots, dict) or layer_name not in slots:
        yield
        return
    original = slots[layer_name]
    padded = pad_rows(row_depth, original.shape[0], -1)
    slots[layer_name] = torch.where(padded == depth, original, torch.full_like(original, -1))
    try:
        yield
    finally:
        slots[layer_name] = original
