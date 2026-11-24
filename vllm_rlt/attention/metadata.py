"""Build paged Attention tensors from M4-selected physical page tables."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionRows:
    """One traversal after M4 has selected each physical depth plane.

    The parallel arrays avoid per-query objects on the metadata hot path.
    ``sequence_keys`` distinguish allocation/depth chunks for packed prefill;
    they are opaque to M8 and unused for single-query rows.
    """

    block_tables: Sequence[tuple[int, ...]]
    positions: Sequence[int]
    sequence_keys: Sequence[tuple[object, int]] | None = None


@dataclass(frozen=True)
class AttentionMetadata:
    """Kernel-ready tensors borrowed for one execution traversal.

    M5 owns storage lifetime when these tensors live in reusable device buffers.
    """

    block_tables: torch.Tensor
    context_lengths: torch.Tensor
    cu_seqlens_q: torch.Tensor | None
    max_seqlen_q: int


@dataclass(frozen=True)
class AttentionMetadataValues:
    """Host values for M5 to combine and stage using its own buffer lifetime."""

    block_tables: list[int]
    num_rows: int
    table_width: int
    context_lengths: list[int]
    cu_seqlens_q: list[int] | None
    max_seqlen_q: int


def plan_attention_metadata(
    rows: AttentionRows,
    *,
    block_size: int,
    packed_prefill: bool = False,
) -> AttentionMetadataValues:
    """Pack selected tables and causal lengths without touching KV ownership.

    M4 validates positions and table coverage before calling this function.
    Packed rows must be consecutive positions in one chunk per sequence key.
    """
    if len(rows.block_tables) != len(rows.positions):
        raise ValueError("attention tables and positions must have equal lengths")
    if packed_prefill and rows.sequence_keys is not None:
        if len(rows.sequence_keys) != len(rows.positions):
            raise ValueError("packed attention keys and positions must have equal lengths")
    width = max((position // block_size + 1 for position in rows.positions), default=0)
    table_rows = range(len(rows.positions))
    cumulative = None
    max_query = 1
    if packed_prefill:
        if rows.sequence_keys is None:
            raise ValueError("packed prefill requires sequence keys")
        ends, cumulative, seen = [], [0], set()
        for index, key in enumerate(rows.sequence_keys):
            if index and key == rows.sequence_keys[index - 1]:
                if rows.positions[index] != rows.positions[index - 1] + 1:
                    raise ValueError("packed prefill positions must be contiguous")
                ends[-1] = index
                cumulative[-1] = index + 1
            else:
                if key in seen:
                    raise ValueError("packed prefill request/depth must form one sequence")
                seen.add(key)
                ends.append(index)
                cumulative.append(index + 1)
        table_rows = ends
        max_query = max((b - a for a, b in zip(cumulative, cumulative[1:])), default=1)
    tables = []
    for index in table_rows:
        table = rows.block_tables[index][:width]
        tables.extend(table)
        tables.extend([-1] * (width - len(table)))
    return AttentionMetadataValues(
        block_tables=tables,
        num_rows=len(table_rows),
        table_width=width,
        context_lengths=[rows.positions[index] + 1 for index in table_rows],
        cu_seqlens_q=cumulative,
        max_seqlen_q=max_query,
    )


def build_attention_metadata(
    rows: AttentionRows,
    *,
    block_size: int,
    device: torch.device,
    packed_prefill: bool = False,
) -> AttentionMetadata:
    """Materialize a plan for callers without a shared staging buffer."""
    plan = plan_attention_metadata(rows, block_size=block_size, packed_prefill=packed_prefill)
    return AttentionMetadata(
        block_tables=torch.tensor(plan.block_tables, device=device, dtype=torch.int32).reshape(
            plan.num_rows, plan.table_width
        ),
        context_lengths=torch.tensor(plan.context_lengths, device=device, dtype=torch.int32),
        cu_seqlens_q=(
            torch.tensor(plan.cu_seqlens_q, device=device, dtype=torch.int32)
            if plan.cu_seqlens_q is not None
            else None
        ),
        max_seqlen_q=plan.max_seqlen_q,
    )
