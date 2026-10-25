"""Phase-independent KV metadata contracts and host layout calculation.

The builder resolves addresses and attention layout without allocating tensors or
launching device work. Storage, transfers and completion belong to the callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm_rlt.attention import AttentionMetadataValues, AttentionRows, plan_attention_metadata

if TYPE_CHECKING:
    from vllm_rlt.core.kv_cache_manager import KVCacheManager, _Allocation


@dataclass(frozen=True, eq=False)
class KVMetadata:
    """Phase-independent KV inputs borrowed for one traversal.

    Token positions and write addresses describe actual query rows; padding has
    no write address. Packed query boundaries describe layout, not request stage.
    Host rows and allocation identities are captured independently of caller
    lists. Device tensors are private, read-only inputs to the cache operations.
    """

    owner: KVCacheManager
    rows: tuple[tuple[_Allocation, int, int], ...]
    allocations: tuple[tuple[str, _Allocation], ...]
    position_ids: torch.Tensor
    write_blocks: torch.Tensor
    write_offsets: torch.Tensor
    block_tables: torch.Tensor
    context_lengths: torch.Tensor
    writable: bool
    cu_seqlens_q: torch.Tensor | None = None
    max_seqlen_q: int = 1


@dataclass(frozen=True, eq=False)
class DeviceKVMetadata:
    """Stream-ordered speculative metadata with device-owned positions.

    Prepared by MetadataBuilder and checked against cache ownership; it contains borrowed allocation
    identities and read-only tensor metadata, never a manager reference. Tensor
    storage and completion events belong to the runner.

    The caller reserves all pages before submission and retains allocations until
    the final GPU reader completes. Unlike KVMetadata, this descriptor does
    not claim that CPU written-position bookkeeping describes in-flight GPU positions.
    """

    allocations: tuple[tuple[str, _Allocation], ...]
    position_ids: torch.Tensor
    write_blocks: torch.Tensor
    write_offsets: torch.Tensor
    block_tables: torch.Tensor
    context_lengths: torch.Tensor
    # The device-owned path currently uses independent single-query rows.
    # These layout fields also let recurrent graphs stage the same contract.
    cu_seqlens_q: torch.Tensor | None = None
    max_seqlen_q: int = 1


@dataclass(frozen=True)
class KVLayout:
    """Host values for one depth assignment, independent of destination storage.

    Lists shared between depth assignments are borrowed read-only. This is the
    calculation result, before a storage owner fills or uploads tensor buffers.
    """

    owner: KVCacheManager
    rows: tuple[tuple[_Allocation, int, int], ...]
    allocations: tuple[tuple[str, _Allocation], ...]
    positions: list[int]
    write_blocks: list[int]
    write_offsets: list[int]
    attention: AttentionMetadataValues
    writable: bool


class MetadataBuilder:
    def __init__(self, cache):
        self.cache = cache

    def prepare_device(
        self,
        *,
        allocations,
        position_ids,
        write_blocks,
        write_offsets,
        block_tables,
        context_lengths,
    ) -> DeviceKVMetadata:
        """Borrow stream-ordered metadata without reading device positions on CPU.

        The runner reserves the upper bound and retains these allocation objects
        until GPU completion. This does not advance host written-position state.
        Allocation identity also rejects descriptors from another cache manager.
        """
        batch = DeviceKVMetadata(
            allocations=tuple(allocations),
            position_ids=position_ids,
            write_blocks=write_blocks,
            write_offsets=write_offsets,
            block_tables=block_tables,
            context_lengths=context_lengths,
        )
        self.cache._require_live_batch(batch)
        rows = len(position_ids)
        for name, tensor, dtype in (
            ("position_ids", position_ids, torch.int64),
            ("write_blocks", write_blocks, torch.int64),
            ("write_offsets", write_offsets, torch.int64),
            ("context_lengths", context_lengths, torch.int32),
        ):
            if (
                tensor.shape != (rows,)
                or tensor.dtype != dtype
                or tensor.device != self.cache.device
            ):
                raise ValueError(
                    f"{name} must have shape [rows] and use {dtype} on {self.cache.device}"
                )
        if (
            not allocations
            or block_tables.ndim != 2
            or block_tables.shape[0] != rows
            or block_tables.shape[1] == 0
            or block_tables.dtype != torch.int32
            or block_tables.device != self.cache.device
            or not block_tables.is_contiguous()
        ):
            raise ValueError("device block tables must be contiguous int32 [rows, pages]")
        return batch

    def _write_addresses(self, rows, *, offsets=None, for_write=True):
        """Resolve physical addresses and reject aliased writes across query rows."""
        cache = self.cache
        blocks = [
            cache._entries(allocation, depth)[position // cache.block_size]
            for allocation, depth, position in rows
        ]
        if offsets is None:
            offsets = [position % cache.block_size for _, _, position in rows]
        if for_write and len(set(zip(blocks, offsets))) != len(rows):
            raise ValueError("duplicate KV write addresses for request/depth/position rows")
        return blocks, offsets

    def resolve_rows(self, ids, depths, positions):
        """Validate host-known rows for device routing without building host tables."""
        rows = tuple(self.cache._validate_rows(ids, depths, positions))
        self._write_addresses(rows)
        return rows

    def build(
        self,
        request_ids: Sequence[str],
        depth_sets: Sequence[Sequence[int]],
        positions: Sequence[int] | torch.Tensor,
        *,
        packed=False,
        for_write=True,
    ) -> tuple[KVLayout, ...]:
        """Calculate all requested depth layouts without selecting a storage path."""
        cache = self.cache
        if not depth_sets:
            raise ValueError("at least one depth assignment is required")
        if packed and (
            cache.layout != "last_exited" or not cache.attention_capabilities.packed_prefill
        ):
            raise ValueError("packed queries require a compatible attention backend and KV layout")
        first = tuple(cache._validate_rows(request_ids, depth_sets[0], positions))
        row_sets = [first]
        for depths in depth_sets[1:]:
            if len(depths) != len(first):
                raise ValueError("request_ids, depths, and positions must have equal lengths")
            for depth in depths:
                cache._validate_depth(depth)
            row_sets.append(
                tuple(
                    (allocation, int(depth), position)
                    for (allocation, _, position), depth in zip(first, depths)
                )
            )
        positions = [p for _, _, p in first]
        offsets = [p % cache.block_size for p in positions]
        allocations = tuple(dict(zip(request_ids, (a for a, _, _ in first))).items())
        layouts = []
        for rows in row_sets:
            blocks, _ = self._write_addresses(rows, offsets=offsets, for_write=for_write)
            attention = plan_attention_metadata(
                AttentionRows(
                    block_tables=[cache._entries(a, d) for a, d, _ in rows],
                    positions=positions,
                    sequence_keys=[(id(a), d) for a, d, _ in rows] if packed else None,
                ),
                block_size=cache.block_size,
                packed_prefill=packed,
            )
            layouts.append(
                KVLayout(
                    owner=cache,
                    rows=rows,
                    allocations=allocations,
                    positions=positions,
                    write_blocks=blocks,
                    write_offsets=offsets,
                    attention=attention,
                    writable=for_write,
                )
            )
        return tuple(layouts)

    @staticmethod
    def bind(metadata, storage):
        """Bind a host-planned layout to its uploaded typed storage."""

        def view(tensor):
            return storage[tensor.dtype].as_strided(
                tensor.shape, tensor.stride(), tensor.storage_offset()
            )

        return replace(
            metadata,
            position_ids=view(metadata.position_ids),
            write_blocks=view(metadata.write_blocks),
            write_offsets=view(metadata.write_offsets),
            block_tables=view(metadata.block_tables),
            context_lengths=view(metadata.context_lengths),
            cu_seqlens_q=(None if metadata.cu_seqlens_q is None else view(metadata.cu_seqlens_q)),
        )

    @staticmethod
    def slice_rows(metadata, start, end):
        """Select contiguous single-query rows, e.g. a dependent coda subset."""
        if metadata.cu_seqlens_q is not None:
            raise ValueError("packed query boundaries must be rebuilt when selecting rows")
        return replace(
            metadata,
            rows=metadata.rows[start:end],
            position_ids=metadata.position_ids[start:end],
            write_blocks=metadata.write_blocks[start:end],
            write_offsets=metadata.write_offsets[start:end],
            block_tables=metadata.block_tables[start:end],
            context_lengths=metadata.context_lengths[start:end],
        )

    @staticmethod
    def without_padding(metadata):
        if metadata.cu_seqlens_q is not None:
            return metadata
        count = len(metadata.rows)
        return replace(
            metadata,
            position_ids=metadata.position_ids[:count],
            block_tables=metadata.block_tables[:count],
            context_lengths=metadata.context_lengths[:count],
        )

    @staticmethod
    def prepare_shared_rows(
        base,
        spans,
        positions,
        *,
        position_ids,
        write_blocks,
        write_offsets,
        context_lengths,
        loop_count,
    ):
        """Reuse one shared-plane table across token groups and loop assignments."""
        count = len(spans)
        rows = tuple(
            (base.rows[row][0], span.depth, positions[row]) for row, span in enumerate(spans)
        )
        metadata = replace(
            base,
            rows=rows,
            allocations=base.allocations[:count],
            position_ids=position_ids,
            write_blocks=write_blocks,
            write_offsets=write_offsets,
            block_tables=base.block_tables[: len(position_ids)],
            context_lengths=context_lengths,
        )
        return tuple(
            replace(metadata, rows=tuple((a, depth + offset, pos) for a, depth, pos in rows))
            for offset in range(loop_count)
        )
