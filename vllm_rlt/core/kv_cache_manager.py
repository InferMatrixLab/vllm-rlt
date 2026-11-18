"""Paged KV with LAST-EXITED or SHARED physical storage planes.

LAST-EXITED copies the last computed per-layer KV into skipped deeper planes.
SHARED overwrites a single plane and needs no exit copies. Optional incremental
allocation separates logical capacity from physical pages. Immutable full-depth
prompt blocks can be shared through the prefix cache.
"""

import hashlib
import struct
import weakref
from collections import OrderedDict
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Integral

import torch

from vllm_rlt.attention import (
    AttentionBackend,
    BackendCapabilities,
    backend_capabilities,
    create_backend,
    validate_backend_name,
)
from vllm_rlt.device import resolve_device
from vllm_rlt.request import ExecutionBatch
from vllm_rlt.worker.buffers import stage_metadata
from vllm_rlt.worker.metadata import DeviceKVMetadata, KVMetadata, MetadataBuilder


@dataclass
class _WrittenPositions:
    # Common in-order writes only advance a scalar. Keep sparse writes until
    # preceding positions arrive, so packed batches may still be out of order.
    prefix: int = 0
    pending: set[int] = field(default_factory=set)

    def add(self, position: int) -> None:
        if position < self.prefix:
            return
        if position > self.prefix:
            self.pending.add(position)
            return
        self.prefix += 1
        while self.prefix in self.pending:
            self.pending.remove(self.prefix)
            self.prefix += 1

    def add_range(self, start: int, end: int) -> None:
        """Record [start, end), preserving sparse/out-of-order coverage."""
        if start >= end or end <= self.prefix:
            return
        if start > self.prefix:
            self.pending.update(range(start, end))
            return
        self.prefix = end
        if self.pending:
            self.pending = {position for position in self.pending if position >= end}
            while self.prefix in self.pending:
                self.pending.remove(self.prefix)
                self.prefix += 1

    def __contains__(self, position: int) -> bool:
        return position < self.prefix or position in self.pending


@dataclass
class _DenseFrontier:
    # Submission coverage only. Device completion still belongs to stream events.
    end: int
    all_layers_end: int = 0


class _CohortWrittenPositions(_WrittenPositions):
    """Share an append-only cohort frontier; detach on an individual mutation."""

    def __init__(self, source, frontier):
        self._prefix = source.prefix
        self.pending = source.pending.copy()
        self._frontier = frontier

    @property
    def prefix(self):
        return self._prefix if self._frontier is None else self._frontier.end

    @prefix.setter
    def prefix(self, value):
        self._prefix, self._frontier = value, None

    def _detach(self):
        self.prefix = self.prefix

    def add(self, position):
        self._detach()
        super().add(position)

    def add_range(self, start, end):
        self._detach()
        super().add_range(start, end)


@dataclass
class _Allocation:
    max_tokens: int
    block_tables: tuple[tuple[int, ...], ...]
    written: list[list[_WrittenPositions]]
    transfer_leases: set[str] = field(default_factory=set)
    release_requested: bool = False
    dense_submitted: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _SequenceCoverage:
    allocation: _Allocation
    plane: int
    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _PreparedCoverage:
    allocation_epoch: int
    sequences: Sequence[_SequenceCoverage]


class _LazyRangeCoverage(Sequence):
    """Materialize row coverage only when the generic paged path needs it."""

    def __init__(self, ranges, shared):
        self.ranges, self.shared, self.sequences = ranges, shared, None

    def __len__(self):
        return len(self.ranges)

    def __getitem__(self, index):
        return tuple(self)[index]

    def __iter__(self):
        if self.sequences is None:
            self.sequences = tuple(
                _SequenceCoverage(r.allocation, 0 if self.shared else r.depth, ((r.start, r.end),))
                for r in self.ranges
            )
        return iter(self.sequences)


def _sequence_coverage(rows, plane):
    """Collapse query rows once, leaving written state owned by the allocation."""
    grouped = {}
    for allocation, depth, position in rows:
        key = (id(allocation), plane(depth))
        if key not in grouped:
            grouped[key] = (allocation, [])
        grouped[key][1].append(position)
    sequences = []
    for (_, storage_plane), (allocation, positions) in grouped.items():
        ranges = []
        for position in sorted(positions):
            if ranges and position <= ranges[-1][1]:
                start, end = ranges[-1]
                ranges[-1] = (start, max(end, position + 1))
            else:
                ranges.append((position, position + 1))
        sequences.append(_SequenceCoverage(allocation, storage_plane, tuple(ranges)))
    return tuple(sequences)


@dataclass(frozen=True)
class KVSnapshot:
    """What preemption keeps for one request: its pages and its written state.

    ``keys`` and ``values`` hold every page of every plane, in block-table
    order, as ``[pages * planes, layer, token, kv_head, dim]`` host tensors.
    Leases are not captured because a leased allocation cannot be preempted.
    """

    max_tokens: int
    pages: int
    # Written-position state per plane and layer. Opaque: hand it back to
    # ``restore`` and do not read or modify it.
    written: list[list[_WrittenPositions]]
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class EagerLayerInputs:
    history: tuple | None
    mask: torch.Tensor
    batch_size: int
    token_start: int
    query_tokens: int
    head_dim: int
    kv_heads: int
    append_plan: object


class _KVTraversal:
    """One validated dense traversal inside an inference-only model call."""

    def __init__(self, manager, stream, batch, storage, context):
        self.manager, self.stream, self.batch = manager, stream, batch
        self.storage, self.context = storage, context
        self.epoch = manager._allocation_epoch
        self.pair = self.layer = None
        self.next_layer = 0
        self.ordered = True

    def _active(self, batch):
        return (
            batch is self.batch
            and self.manager._contiguous is self.storage
            and self.manager._allocation_epoch == self.epoch
        )

    def _write_prepared(self, layer, batch, k, v):
        manager = self.manager
        if not self._active(batch):
            return manager._write_prepared(layer, batch, k, v)
        manager._validate_layer(layer)
        manager._validate_tensor(k, len(batch.position_ids), "k")
        manager._validate_tensor(v, len(batch.position_ids), "v")
        if not self.storage._append(layer, batch, k, v, self.context):
            self.pair = self.layer = None
            return manager._write_prepared(layer, batch, k, v, _stream=self.stream)
        self.pair = self.storage.groups[self.context[-1], layer].pair
        self.ordered = self.ordered and layer == self.next_layer
        self.next_layer += 1
        self.layer = layer

    def prepare_native(self, layer, hidden):
        if not self._active(self.batch):
            return None
        self.manager._validate_layer(layer)
        size, start, count, _ = self.context
        if (
            hidden.ndim != 2
            or len(hidden) != size * count
            or hidden.device != self.manager.device
            or hidden.dtype != self.manager.dtype
        ):
            raise ValueError("native eager input must match the validated cache schema")
        plan = self.storage.prepare_append(layer, self.batch, self.context)
        if plan is None:
            return None
        _, mask = self.manager.attention.kernel.prepare_eager_attention(
            hidden, self.batch, stream=self.stream
        )
        return EagerLayerInputs(
            plan.history,
            mask,
            size,
            start,
            count,
            self.manager.head_dim,
            self.manager.num_kv_heads,
            plan,
        )

    def commit_native(self, layer, inputs, pair):
        if not self._active(self.batch):
            raise RuntimeError("stale native eager traversal")
        expected = (
            inputs.batch_size,
            inputs.kv_heads,
            inputs.token_start + inputs.query_tokens,
            inputs.head_dim,
        )
        if any(
            tuple(x.shape) != expected
            or x.device != self.manager.device
            or x.dtype != self.manager.dtype
            for x in pair
        ):
            raise ValueError("native eager KV output does not match the cache schema")
        self.storage.commit_append(layer, self.batch, self.context, inputs.append_plan, pair)
        self.pair = pair
        self.ordered = self.ordered and layer == self.next_layer
        self.next_layer += 1
        self.layer = layer

    def finish(self):
        if self._active(self.batch) and self.ordered and self.next_layer == self.manager.num_layers:
            group = self.storage.groups.get((self.context[-1], 0))
            if group is not None and group.allocations is self.batch.allocations:
                group.frontier.all_layers_end = self.context[1] + self.context[2]

    def _attend_prepared(self, layer, batch, q, *, window=None):
        manager = self.manager
        if window is not None:
            return manager._attend_prepared(layer, batch, q, window=window, _stream=self.stream)
        if not self._active(batch) or layer != self.layer or self.pair is None:
            return manager._attend_prepared(layer, batch, q)
        manager._validate_tensor(q, len(batch.position_ids), "q", query=True)
        return manager.attention.kernel.attend_contiguous(q, self.pair, batch, stream=self.stream)


class KVCacheManager:
    """Own a fixed physical page pool shared by request/depth allocations.

    ``num_blocks`` counts total physical blocks across all recurrence depths.
    Every physical block contains K and V for all decoder layers. Depth indices
    are zero-based throughout the interface.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        max_loops: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        backend: str = "torch",
        layout: str = "last_exited",
        enable_prefix_caching: bool = False,
        incremental_allocation: bool = False,
        watermark_ratio: float = 0.0,
        recurrent_layers: Sequence[int] | None = None,
        *,
        attention: AttentionBackend | None = None,
        contiguous_kv: bool = False,
    ):
        for name, value in (
            ("num_layers", num_layers),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("num_blocks", num_blocks),
            ("block_size", block_size),
            ("max_loops", max_loops),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        validate_backend_name(backend)
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("KV dtype must be float32, float16, or bfloat16")
        if layout not in {"last_exited", "shared"}:
            raise ValueError("unsupported KV layout")
        self.layout = layout
        self.recurrent_layers = (
            tuple(range(num_layers)) if recurrent_layers is None else tuple(recurrent_layers)
        )
        if not self.recurrent_layers or len(set(self.recurrent_layers)) != len(
            self.recurrent_layers
        ):
            raise ValueError("recurrent_layers must be nonempty and unique")
        if any(
            type(layer) is not int or not 0 <= layer < num_layers for layer in self.recurrent_layers
        ):
            raise ValueError("recurrent_layers must be valid cache layer indices")
        self.storage_depths = max_loops if layout == "last_exited" else 1
        self.device = resolve_device(device)
        self.dtype = dtype
        self.attention = attention or create_backend(
            backend, self.device, dtype, head_dim, block_size
        )
        self.attention_info = self.attention.info
        self.backend = self.attention_info["selected_backend"]
        shape = (num_blocks, num_layers, block_size, num_kv_heads, head_dim)
        if contiguous_kv and (
            not self.attention_capabilities.sequence_ranges
            or not self.attention_capabilities.owned_contiguous_kv
            or layout != "last_exited"
            or enable_prefix_caching
            or incremental_allocation
        ):
            raise ValueError(
                "contiguous KV requires private reserved sequence-range storage "
                "and owned contiguous attention"
            )
        self._paged_shape = shape
        self._contiguous = None
        self._key_cache = self._value_cache = None
        if contiguous_kv:
            # Only owned NPU KV needs device resolution without a paged allocation.
            self.device = torch.empty(0, device=self.device, dtype=dtype).device
            from vllm_rlt.kernels.contiguous_kv import ContiguousKVStorage

            self._contiguous = ContiguousKVStorage(
                num_kv_heads, head_dim, self.device, publish_coverage=self._publish_dense_coverage
            )
        else:
            self._key_cache = torch.empty(shape, device=self.device, dtype=dtype)
            self._value_cache = torch.empty_like(self._key_cache)
            self.device = self._key_cache.device
            self.key_cache, self.value_cache = self._key_cache, self._value_cache
        self._free_blocks = list(reversed(range(num_blocks)))
        self._allocations: dict[str, _Allocation] = {}
        self._allocation_epoch = 0
        # Only the Ascend host-metadata path changes. Weak keys must not keep
        # request descriptors or their staged tensor storage alive.
        self._prepared_coverage = (
            weakref.WeakKeyDictionary()
            if self.device.type == "npu" and self.backend == "ascend_eager"
            else None
        )
        self._validated_range_tables = weakref.WeakKeyDictionary()
        self.enable_prefix_caching = enable_prefix_caching
        self.incremental_allocation = incremental_allocation
        # Full allocation already reserves each request's maximum KV capacity.
        self.watermark_blocks = int(watermark_ratio * num_blocks) if incremental_allocation else 0
        self._refs = [0] * num_blocks
        self._prefixes = OrderedDict()
        self._pending_prefixes = []
        self.prefix_hits = self.prefix_queries = 0
        if enable_prefix_caching and layout != "last_exited":
            raise ValueError("prefix caching requires last_exited KV")

    @property
    def attention_capabilities(self) -> BackendCapabilities:
        return backend_capabilities(self.attention)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks) + sum(
            self._refs[b] == 1 for blocks in self._prefixes.values() for b in blocks
        )

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def bytes_per_block(self) -> int:
        return (
            2
            * self.num_layers
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.key_cache.element_size()
        )

    def required_blocks(self, max_tokens: int) -> int:
        if not isinstance(max_tokens, Integral) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        return ((max_tokens + self.block_size - 1) // self.block_size) * self.storage_depths

    def _drop_refs(self, blocks):
        for b in blocks:
            self._refs[b] -= 1
            assert self._refs[b] >= 0
            if not self._refs[b]:
                self._free_blocks.append(b)

    def _claim(self, count):
        """Take ``count`` free blocks, evicting prefix entries only if that is enough.

        ``num_free_blocks`` already includes every block eviction could
        release. If it is still short, return ``None`` without evicting, so a
        failed admission, growth, resumption, or PD reservation does not
        throw away prefixes it could not use anyway. ``num_free_blocks`` scans
        every prefix entry, so it is only consulted when eviction is needed.
        """
        if len(self._free_blocks) < count and self.num_free_blocks < count:
            return None
        while len(self._free_blocks) < count and self._prefixes:
            _, blocks = self._prefixes.popitem(last=False)
            self._drop_refs(blocks)
        if len(self._free_blocks) < count:
            return None
        blocks = [self._free_blocks.pop() for _ in range(count)]
        for b in blocks:
            assert self._refs[b] == 0
            self._refs[b] = 1
        return blocks

    def _prefix_keys(self, tokens):
        digest = b""
        for start in range(0, len(tokens) // self.block_size * self.block_size, self.block_size):
            values = tokens[start : start + self.block_size]
            digest = hashlib.sha256(digest + struct.pack(f"<{len(values)}q", *values)).digest()
            yield digest

    def poll_prefixes(self):
        pending, self._pending_prefixes = self._pending_prefixes, []
        for request_id, allocation, tokens, length, event in pending:
            if self._allocations.get(request_id) is not allocation:
                continue
            if event is not None and not event.query():
                self._pending_prefixes.append((request_id, allocation, tokens, length, event))
                continue
            self.publish_prefix(request_id, tokens, length)

    def publish_prefix(self, request_id, tokens, length, event=None):
        """Index complete prompt pages for reuse without copying their KV.

        length is the prefilled extent; len(tokens) is the full prompt length.
        A supplied event defers publication until the GPU writes complete.
        """
        if not self.enable_prefix_caching:
            return
        allocation = self._get_allocation(request_id)
        if event is not None:
            self._pending_prefixes.append((request_id, allocation, tuple(tokens), length, event))
            return
        # Prefix entries retain KV, not the final hidden state needed by CODA.
        # Leave at least the last prompt token to recompute on a cache hit;
        # full-page alignment may leave additional tokens to recompute.
        publishable_tokens = min(length, len(tokens) - 1)
        for index, key in enumerate(self._prefix_keys(tokens[:publishable_tokens])):
            if key in self._prefixes:
                continue
            end = (index + 1) * self.block_size
            # Every depth and layer must have a contiguous written prefix.
            # GPU completion alone does not establish this coverage.
            if any(w.prefix < end for plane in allocation.written for w in plane):
                break
            blocks = tuple(t[index] for t in allocation.block_tables)
            for b in blocks:
                self._refs[b] += 1
            self._prefixes[key] = blocks

    def lookup_prefix(self, tokens):
        if not self.enable_prefix_caching:
            return ()
        self.poll_prefixes()
        found = []
        for key in self._prefix_keys(tokens[:-1]):
            blocks = self._prefixes.get(key)
            if blocks is None:
                break
            self._prefixes.move_to_end(key)
            found.append(blocks)
        return tuple(found)

    def allocate(self, request_id: str, max_tokens: int, *, initial_tokens=None, prefix=()) -> bool:
        """Reserve a logical capacity; acquire physical pages for the current frontier."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if request_id in self._allocations:
            raise ValueError(f"request {request_id!r} already owns KV blocks")
        required = self.required_blocks(max_tokens)
        if required > self.num_blocks:
            raise ValueError(
                f"request needs {required} physical KV blocks, but the pool has {self.num_blocks}"
            )
        initial_tokens = max_tokens if initial_tokens is None else initial_tokens
        if not 0 < initial_tokens <= max_tokens:
            raise ValueError("invalid initial KV frontier")
        pages = (initial_tokens + self.block_size - 1) // self.block_size
        if len(prefix) > pages:
            raise ValueError("prefix exceeds initial KV frontier")
        claimed = [b for group in prefix for b in group]
        for b in claimed:
            self._refs[b] += 1
        fresh = self._claim((pages - len(prefix)) * self.storage_depths)
        if fresh is None:
            self._drop_refs(claimed)
            return False
        tail = pages - len(prefix)
        tables = tuple(
            tuple(g[d] for g in prefix) + tuple(fresh[d * tail : (d + 1) * tail])
            for d in range(self.storage_depths)
        )
        length = len(prefix) * self.block_size
        self._allocations[request_id] = _Allocation(
            int(max_tokens),
            tables,
            [
                [_WrittenPositions(prefix=length) for _ in range(self.num_layers)]
                for _ in range(self.storage_depths)
            ],
        )
        self._allocation_epoch += 1
        self._invalidate_contiguous(preserve_contiguous=True)
        self.prefix_queries += 1
        self.prefix_hits += length
        return True

    def ensure_capacity(self, request_id, tokens):
        allocation = self._get_allocation(request_id)
        if not 0 < tokens <= allocation.max_tokens:
            raise ValueError("KV growth exceeds logical capacity")
        pages = (tokens + self.block_size - 1) // self.block_size
        extra = pages - len(allocation.block_tables[0])
        if extra <= 0:
            return True
        blocks = self._claim(extra * self.storage_depths)
        if blocks is None:
            return False
        allocation.block_tables = tuple(
            t + tuple(blocks[d * extra : (d + 1) * extra])
            for d, t in enumerate(allocation.block_tables)
        )
        self._allocation_epoch += 1
        self._invalidate_contiguous(preserve_contiguous=True)
        return True

    def truncate_suffix(self, request_id: str, frontier: int) -> None:
        """Invalidate an uncommitted suffix, retaining reserved physical pages.

        Call only after all suffix users complete. Retaining pages preserves the
        admission reservation; stale bytes are hidden by written/context lengths.
        Shared prompt pages must never be truncated or subsequently overwritten.
        """
        allocation = self._get_allocation(request_id)
        if type(frontier) is not int or not 0 <= frontier <= allocation.max_tokens:
            raise ValueError("invalid KV truncation frontier")
        if allocation.transfer_leases:
            raise ValueError("cannot truncate KV during a transfer")
        for table in allocation.block_tables:
            if any(self._refs[b] > 1 for b in table[frontier // self.block_size :]):
                raise ValueError("cannot truncate shared prefix pages")
        self._invalidate_contiguous()
        allocation.dense_submitted.clear()
        for plane in allocation.written:
            for written in plane:
                written.prefix = min(written.prefix, frontier)
                written.pending = {p for p in written.pending if p < frontier}

    def free(self, request_id: str) -> None:
        allocation = self._allocations.get(request_id)
        self._invalidate_contiguous(preserve_contiguous=True)
        if allocation is not None and allocation.transfer_leases:
            allocation.release_requested = True
            return
        if self._contiguous is not None and allocation is not None:
            self._contiguous.release(allocation, self.num_layers, self.storage_depths)
        allocation = self._allocations.pop(request_id, None)
        if allocation is not None:
            self._allocation_epoch += 1
            self._drop_refs(b for table in allocation.block_tables for b in reversed(table))

    def pin_transfer(self, request_id: str, transfer_id: str):
        self._invalidate_contiguous()
        allocation = self._get_allocation(request_id)
        if allocation.release_requested:
            raise RuntimeError("cannot transfer KV scheduled for release")
        allocation.transfer_leases.add(transfer_id)

    def unpin_transfer(self, request_id: str, transfer_id: str):
        allocation = self._get_allocation(request_id)
        allocation.transfer_leases.discard(transfer_id)
        if allocation.release_requested and not allocation.transfer_leases:
            self.free(request_id)

    def mark_imported_prefix(self, request_id: str, length: int, *, start=0):
        """Publish received KV only after the connector confirms remote writes are complete."""
        allocation = self._get_allocation(request_id)
        if type(length) is not int or not 0 < length <= allocation.max_tokens:
            raise ValueError("imported prefix exceeds reserved capacity")
        if allocation.release_requested:
            raise RuntimeError("cannot activate cancelled KV")
        if any(w.prefix != start or w.pending for plane in allocation.written for w in plane):
            raise RuntimeError("KV prefix was already initialized")
        allocation.dense_submitted.clear()
        for plane in allocation.written:
            for written in plane:
                written.prefix = length

    # Other modules learn about allocations only through the methods below.
    # They return plain facts, never _Allocation objects or reference counts.

    def has_allocation(self, request_id: str) -> bool:
        return request_id in self._allocations

    def allocated_blocks(self, request_id: str) -> int:
        """How many physical blocks this request holds, counting every plane."""
        return len(self._get_allocation(request_id).block_tables[0]) * self.storage_depths

    def get_block_table(self, request_id: str, depth: int) -> tuple[int, ...]:
        self._validate_depth(depth)
        return self._get_allocation(request_id).block_tables[self._plane(depth)]

    def plane_block_tables(self, request_id: str) -> tuple[tuple[int, ...], ...]:
        """One block table per plane.

        This tuple is indexed by plane, not by loop depth. LAST_EXITED has
        ``max_loops`` planes and SHARED has one, so ``tables[2]`` fails under
        SHARED while ``get_block_table(rid, 2)`` works under both layouts.
        """
        return self._get_allocation(request_id).block_tables

    def has_transfer_lease(self, request_id: str) -> bool:
        return bool(self._get_allocation(request_id).transfer_leases)

    def exclusive_prefix_blocks(self, prefix: Sequence[Sequence[int]]) -> int:
        """How many blocks of a prefix hit are held only by the prefix cache.

        ``num_free_blocks`` counts those blocks as free because they could be
        evicted. Claiming them makes them non-evictable, so admission has to
        budget them on top of the fresh blocks it needs.
        """
        return sum(self._refs[b] == 1 for group in prefix for b in group)

    def token_written(self, request_id: str, position: int, depth: int) -> bool:
        """Whether every layer has written KV for ``position`` at ``depth``."""
        allocation = self._get_allocation(request_id)
        self._validate_depth(depth)
        self._validate_position(allocation, position)
        return self._token_written(allocation, position, depth)

    def _token_written(self, allocation: _Allocation, position: int, depth: int) -> bool:
        plane = allocation.written[self._plane(depth)]
        return all(position in plane[layer] for layer in range(self.num_layers))

    def exit_written(self, request_id: str, position: int, depth: int) -> bool:
        """Whether every recurrent layer has written KV for ``position`` at ``depth``.

        Boundary layers keep their history at depth 0 and the output stage runs
        after the exit, so they take no part in finalizing a token.
        """
        allocation = self._get_allocation(request_id)
        self._validate_depth(depth)
        self._validate_position(allocation, position)
        plane = allocation.written[self._plane(depth)]
        return all(position in plane[layer] for layer in self.recurrent_layers)

    def mark_finalized(self, request_id: str, position: int, exit_depth: int) -> None:
        """Mark the token written at every depth deeper than ``exit_depth``.

        This is the bookkeeping half of ``finalize_token``. The asynchronous
        finalize kernel copies the KV itself and then calls this. It must only
        be called after the copies are queued on the stream, or attention at
        those depths will read garbage. SHARED has nothing to record.
        """
        allocation = self._require_exit_written(request_id, position, exit_depth)
        self._record_finalized(allocation, position, exit_depth)

    def _require_exit_written(self, request_id, position, exit_depth) -> _Allocation:
        allocation = self._get_allocation(request_id)
        self._validate_depth(exit_depth)
        self._validate_position(allocation, position)
        proof = allocation.dense_submitted.get(self._plane(exit_depth))
        if proof is not None and position < proof.all_layers_end:
            return allocation
        if any(
            position not in allocation.written[self._plane(exit_depth)][layer]
            for layer in self.recurrent_layers
        ):
            raise RuntimeError(
                "cannot finalize a token before every layer has written its exit depth"
            )
        return allocation

    def _record_finalized(self, allocation, position, exit_depth) -> None:
        if self.layout == "shared":
            return
        for depth in range(exit_depth + 1, self.max_loops):
            for layer in self.recurrent_layers:
                allocation.written[self._plane(depth)][layer].add(position)

    def snapshot(self, request_id: str) -> KVSnapshot:
        """Copy a request's pages and written state to the host, page by page.

        The caller must have finished all device work on these pages first.
        Copying page by page avoids a request-sized temporary on a device
        that is already full, which is exactly when preemption happens.
        """
        allocation = self._get_allocation(request_id)
        if allocation.transfer_leases:
            raise RuntimeError("cannot snapshot KV during a transfer")
        blocks = [b for table in allocation.block_tables for b in table]
        keys = torch.empty((len(blocks), *self.key_cache.shape[1:]), dtype=self.dtype)
        values = torch.empty_like(keys)
        for row, block in enumerate(blocks):
            keys[row].copy_(self.key_cache[block])
            values[row].copy_(self.value_cache[block])
        return KVSnapshot(
            max_tokens=allocation.max_tokens,
            pages=len(allocation.block_tables[0]),
            written=deepcopy(allocation.written),
            keys=keys,
            values=values,
        )

    def restore(self, request_id: str, snapshot: KVSnapshot) -> None:
        """Load a snapshot back into a fresh allocation of the same size.

        ``allocate`` stays a separate step because it can fail and that
        decision belongs to the scheduler. The copies go on the current
        stream; the caller decides when other streams may read them.
        """
        allocation = self._get_allocation(request_id)
        if allocation.max_tokens != snapshot.max_tokens:
            raise ValueError("snapshot capacity does not match the allocation")
        if len(allocation.block_tables[0]) != snapshot.pages:
            raise ValueError("snapshot page count does not match the allocation")
        if any(w.prefix or w.pending for plane in allocation.written for w in plane):
            raise RuntimeError("cannot restore into an allocation that already holds KV")
        self._invalidate_contiguous()
        blocks = [b for table in allocation.block_tables for b in table]
        for row, block in enumerate(blocks):
            self.key_cache[block].copy_(snapshot.keys[row])
            self.value_cache[block].copy_(snapshot.values[row])
        allocation.dense_submitted.clear()
        allocation.written = deepcopy(snapshot.written)

    def _plane(self, depth: int) -> int:
        return depth if self.layout == "last_exited" else 0

    def _get_allocation(self, request_id: str) -> _Allocation:
        try:
            return self._allocations[request_id]
        except KeyError:
            raise KeyError(f"request {request_id!r} has no KV allocation") from None

    def _validate_layer(self, layer: int) -> None:
        if (
            not isinstance(layer, Integral)
            or isinstance(layer, bool)
            or not 0 <= layer < self.num_layers
        ):
            raise ValueError(f"layer must be in [0, {self.num_layers})")

    def _validate_depth(self, depth: int) -> None:
        if (
            not isinstance(depth, Integral)
            or isinstance(depth, bool)
            or not 0 <= depth < self.max_loops
        ):
            raise ValueError(f"depth must be in [0, {self.max_loops})")

    def _validate_position(self, allocation: _Allocation, position: int) -> None:
        if (
            not isinstance(position, Integral)
            or isinstance(position, bool)
            or not 0 <= position < allocation.max_tokens
        ):
            raise ValueError(f"position must be in [0, {allocation.max_tokens})")

    def _validate_rows(self, request_ids, depths, positions):
        if isinstance(positions, torch.Tensor):
            if positions.ndim != 1 or positions.dtype not in {torch.int32, torch.int64}:
                raise ValueError("positions must be a one-dimensional integer tensor")
            positions = positions.tolist()
        if len(request_ids) != len(depths) or len(request_ids) != len(positions):
            raise ValueError("request_ids, depths, and positions must have equal lengths")
        rows = []
        for request_id, depth, position in zip(request_ids, depths, positions):
            allocation = self._get_allocation(request_id)
            self._validate_depth(depth)
            self._validate_position(allocation, position)
            rows.append((allocation, int(depth), int(position)))
        return rows

    @property
    def metadata_builder(self):
        return MetadataBuilder(self)

    def prepare_recurrent(self, batch: ExecutionBatch, *, packed=False) -> KVMetadata:
        """Compatibility entry for standalone cache consumers."""
        ids, depths, positions = batch.addresses()
        return self._prepare_batch(
            ids,
            depths,
            positions,
            packed_prefill=packed and self.attention_capabilities.packed_prefill,
        )

    def _prepare_batch(
        self,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        *,
        for_write: bool = True,
        packed_prefill: bool = False,
        stage: Callable[[Sequence[int], torch.dtype], torch.Tensor] | None = None,
        padded_rows: int | None = None,
    ) -> KVMetadata:
        """Build layer-independent addresses once; do not initialize any KV slot."""
        return self._prepare_batches(
            request_ids,
            (depths,),
            positions,
            for_write=for_write,
            packed_prefill=packed_prefill,
            stage=stage,
            padded_rows=padded_rows,
        )[0]

    def _prepare_batches(
        self,
        request_ids: Sequence[str],
        depth_sets: Sequence[Sequence[int]],
        positions: Sequence[int] | torch.Tensor,
        *,
        for_write: bool = True,
        packed_prefill: bool = False,
        stage: Callable[[Sequence[int], torch.dtype], torch.Tensor] | None = None,
        padded_rows: int | None = None,
    ) -> tuple[KVMetadata, ...]:
        """Convenience API; layout construction belongs to MetadataBuilder."""
        layouts = self.metadata_builder.build(
            request_ids,
            depth_sets,
            positions,
            for_write=for_write,
            packed=packed_prefill,
        )
        return stage_metadata(
            layouts, self._stage if stage is None else stage, padded_rows=padded_rows
        )

    def _stage(self, values, dtype):
        host = torch.tensor(values, dtype=dtype, pin_memory=self.device.type == "cuda")
        return host.to(self.device, non_blocking=True)

    def _require_live_batch(self, batch: KVMetadata | DeviceKVMetadata):
        if isinstance(batch, KVMetadata) and batch.owner is not self:
            raise ValueError("prepared KV batch belongs to a different cache manager")
        if (
            isinstance(batch, KVMetadata)
            and batch.sequence_ranges is not None
            and not self.attention_capabilities.sequence_ranges
        ):
            raise ValueError("attention backend does not support compact sequence ranges")
        plan = None
        reusable = self._prepared_coverage is not None and isinstance(batch, KVMetadata)
        epoch = self._allocation_epoch
        if reusable:
            plan = self._prepared_coverage.get(batch)
            if plan is not None and plan.allocation_epoch == epoch:
                return plan
        table_plan = batch.table_plan if reusable else None
        validated = self._validated_range_tables.get(table_plan) if table_plan is not None else None
        if validated is None or validated[0] != epoch or validated[1] is not batch.allocations:
            for request_id, allocation in batch.allocations:
                if self._allocations.get(request_id) is not allocation:
                    raise RuntimeError(f"stale prepared KV batch for request {request_id!r}")
            if table_plan is not None:
                self._validated_range_tables[table_plan] = epoch, batch.allocations
        if reusable:
            plan = _PreparedCoverage(
                epoch,
                (
                    _LazyRangeCoverage(batch.sequence_ranges, self.layout == "shared")
                    if batch.sequence_ranges is not None
                    else _sequence_coverage(batch.rows, self._plane)
                )
                if plan is None
                else plan.sequences,
            )
            self._prepared_coverage[batch] = plan
        return plan

    def _validate_tensor(self, tensor, batch_size, name, *, query=False):
        if tensor.ndim != 3 or tensor.shape[0] != batch_size or tensor.shape[-1] != self.head_dim:
            raise ValueError(f"{name} must have shape [batch, heads, {self.head_dim}]")
        if query:
            if tensor.shape[1] == 0 or tensor.shape[1] % self.num_kv_heads:
                raise ValueError("query heads must be a positive multiple of KV heads")
        elif tensor.shape[1] != self.num_kv_heads:
            raise ValueError(f"{name} must have {self.num_kv_heads} KV heads")
        if tensor.device != self.device or tensor.dtype != self.dtype:
            raise ValueError(f"{name} must use device {self.device} and dtype {self.dtype}")

    @staticmethod
    def _publish_dense_coverage(spans, depth, layer, end, frontier):
        """Record submitted copies after storage has enqueued both K and V.

        Stable cohorts advance one shared proof. Individual mutation detaches a
        row, and an interrupted traversal never publishes all-layer coverage.
        """
        if frontier is not None:
            frontier.end = end
            return frontier
        frontier = _DenseFrontier(end)
        for span in spans:
            previous = span.allocation.written[depth][layer]
            span.allocation.written[depth][layer] = _CohortWrittenPositions(previous, frontier)
            if layer == 0:
                span.allocation.dense_submitted[depth] = frontier
        return frontier

    def _begin_traversal(self, batch):
        if (
            self._contiguous is None
            or not isinstance(batch, KVMetadata)
            or self.device.type != "npu"
            or not torch.is_inference_mode_enabled()
        ):
            return self
        if not batch.writable:
            return self
        coverage = self._require_live_batch(batch)
        if coverage is None:
            return self
        stream = torch.npu.current_stream(self.device)
        context = self._contiguous._context(batch, stream)
        if context is None:
            return self
        return _KVTraversal(self, stream, batch, self._contiguous, context)

    @torch.no_grad()
    def write(
        self,
        layer: int,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write a packed batch before calling attend; future tokens stay masked.

        This compatibility adapter builds the full descriptor, including the
        attention tensors. The model reuses one prepared descriptor per traversal;
        standalone writes should not be substituted into its per-layer hot path.
        Address/ownership validation intentionally precedes tensor validation.
        """
        self._validate_layer(layer)
        batch = self._prepare_batch(request_ids, depths, positions)
        self._write_prepared(layer, batch, k, v)

    @torch.no_grad()
    def _write_prepared(
        self,
        layer: int,
        batch: KVMetadata | DeviceKVMetadata,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        _stream=None,
    ) -> None:
        self._validate_layer(layer)
        coverage = self._require_live_batch(batch)
        if not isinstance(batch, KVMetadata) or batch.sequence_ranges is None:
            self._invalidate_contiguous()
        if isinstance(batch, DeviceKVMetadata):
            self._validate_tensor(k, len(batch.position_ids), "k")
            self._validate_tensor(v, len(batch.position_ids), "v")
            if batch.write_mask is not None:
                from vllm_rlt.kernels.routing import masked_kv_store

                masked_kv_store(self.key_cache[:, layer], self.value_cache[:, layer], k, v, batch)
            else:
                self.key_cache[batch.write_blocks, layer, batch.write_offsets] = k
                self.value_cache[batch.write_blocks, layer, batch.write_offsets] = v
            return
        if not batch.writable:
            raise ValueError("a read-only prepared KV batch cannot be written")
        self._validate_tensor(k, len(batch.position_ids), "k")
        self._validate_tensor(v, len(batch.position_ids), "v")
        if not batch.rows:
            return
        direct = self._contiguous is not None and self._contiguous.append(
            layer, batch, k, v, stream=_stream
        )
        if direct:
            # The store advances a shared layer frontier only after both KV
            # writes enqueue. Public coverage readers still observe each row.
            return
        elif self.device.type == "npu" and self.backend == "ascend_eager":
            if batch.deferred_addresses:
                batch = self.metadata_builder.materialize_paged(batch)
            if not hasattr(self, "_npu_writer"):
                from vllm_rlt.kernels.ascend_kv import AscendKVWriter

                self._npu_writer = AscendKVWriter()
            self._npu_writer.write(
                self.key_cache,
                self.value_cache,
                layer,
                batch,
                k if len(k) == len(batch.rows) else k[: len(batch.rows)],
                v if len(v) == len(batch.rows) else v[: len(batch.rows)],
            )
        else:
            self.key_cache[batch.write_blocks, layer, batch.write_offsets] = k[: len(batch.rows)]
            self.value_cache[batch.write_blocks, layer, batch.write_offsets] = v[: len(batch.rows)]
        if coverage is None:
            for allocation, depth, position in batch.rows:
                allocation.written[self._plane(depth)][layer].add(position)
        else:
            # Advance only this layer, only after BOTH device copies enqueue.
            # Keep holes and partial-layer writes observable to later attention.
            for sequence in coverage.sequences:
                written = sequence.allocation.written[sequence.plane][layer]
                for start, end in sequence.ranges:
                    written.add_range(start, end)

    def _require_prefix(self, allocation, layer, depth, length):
        written = allocation.written[self._plane(depth)][layer]
        if length > written.prefix:
            raise RuntimeError(
                f"uninitialized KV history at layer {layer}, depth {depth}, "
                f"context length {length}; "
                "write preceding tokens and finalize early exits before attention"
            )

    @torch.no_grad()
    def attend(
        self,
        layer: int,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        q: torch.Tensor,
        *,
        window: int | None = None,
    ) -> torch.Tensor:
        """Apply causal GQA at each row's own position and recurrence depth."""
        self._validate_layer(layer)
        batch = self._prepare_batch(request_ids, depths, positions, for_write=False)
        return self._attend_prepared(layer, batch, q, window=window)

    @torch.no_grad()
    def _attend_prepared(
        self,
        layer: int,
        batch: KVMetadata | DeviceKVMetadata,
        q: torch.Tensor,
        *,
        window: int | None = None,
        _stream=None,
    ) -> torch.Tensor:
        self._validate_layer(layer)
        coverage = self._require_live_batch(batch)
        self._validate_tensor(q, len(batch.position_ids), "q", query=True)
        if window is not None:
            if batch.cu_seqlens_q is None:
                return self._attend_windowed(layer, batch, q, window)
            if not self.attention_capabilities.packed_sliding_window:
                raise NotImplementedError("sliding-window attention needs one query row per token")
            if type(window) is not int or window <= 0:
                raise ValueError("window must be a positive integer")
        if isinstance(batch, DeviceKVMetadata):
            if window is not None:
                raise NotImplementedError(
                    "packed sliding-window attention requires host-planned KV"
                )
            return self.attention(
                q,
                self.key_cache[:, layer],
                self.value_cache[:, layer],
                batch.block_tables,
                batch.context_lengths,
            )
        if not batch.rows:
            return torch.empty_like(q)
        if self._contiguous is not None:
            pair = self._contiguous.get(layer, batch, stream=_stream)
            if pair is not None:
                # The authoritative append-only store proves an initialized
                # prefix for every live row. Holes, overwrite, truncate and
                # generic writes escape to the paged checks below.
                return self.attention.kernel.attend_contiguous(q, pair, batch, stream=_stream)
        if coverage is None:
            for allocation, depth, position in batch.rows:
                self._require_prefix(allocation, layer, depth, position + 1)
        else:
            for sequence in coverage.sequences:
                self._require_prefix(
                    sequence.allocation, layer, sequence.plane, sequence.ranges[-1][1]
                )
        if batch.deferred_addresses:
            batch = self.metadata_builder.materialize_paged(batch)
        if batch.cu_seqlens_q is not None:
            return self.attention.prefill(
                q,
                self.key_cache[:, layer],
                self.value_cache[:, layer],
                batch.block_tables,
                batch.context_lengths,
                batch.cu_seqlens_q,
                batch.max_seqlen_q,
                **({} if window is None else {"window": window}),
            )
        return self.attention.attend_metadata(
            q, self.key_cache[:, layer], self.value_cache[:, layer], batch
        )

    @torch.no_grad()
    def _attend_windowed(
        self, layer: int, batch: KVMetadata, q: torch.Tensor, window: int
    ) -> torch.Tensor:
        """Mask keys outside ``window`` without reclaiming their storage.

        Only kernels that advertise ``sliding_window`` are used; there is no
        silent fallback to full attention, which would change the model.
        """
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")
        if not self.attention_capabilities.sliding_window:
            raise NotImplementedError(
                f"the {self.backend} attention backend does not implement sliding-window masking"
            )
        if batch.cu_seqlens_q is not None:
            raise NotImplementedError("sliding-window attention needs one query row per token")
        # Device-owned positions are ordered by the runner's stream; the host
        # frontier lags them, as for unwindowed attention.
        if not isinstance(batch, DeviceKVMetadata):
            if not batch.rows:
                return torch.empty_like(q)
            for allocation, depth, position in batch.rows:
                self._require_prefix(allocation, layer, depth, position + 1)
        return self.attention.windowed(
            q,
            self.key_cache[:, layer],
            self.value_cache[:, layer],
            batch.block_tables,
            batch.context_lengths,
            window,
        )

    @torch.no_grad()
    def finalize_token(self, request_id: str, position: int, exit_depth: int) -> None:
        """Propagate the final executed depth's K/V into every unexecuted depth."""
        allocation = self._require_exit_written(request_id, position, exit_depth)
        if self.layout == "shared":
            return
        logical_page, offset = divmod(position, self.block_size)
        source = allocation.block_tables[self._plane(exit_depth)][logical_page]
        destinations = [
            allocation.block_tables[self._plane(depth)][logical_page]
            for depth in range(exit_depth + 1, self.max_loops)
        ]
        if destinations:
            self._invalidate_contiguous()
        # Basic-index copies avoid a blocking host->device index tensor on the
        # boundary stream. That transfer would serialize final core and routing.
        for destination in destinations:
            if len(self.recurrent_layers) == self.num_layers:
                self.key_cache[destination, :, offset].copy_(self.key_cache[source, :, offset])
                self.value_cache[destination, :, offset].copy_(self.value_cache[source, :, offset])
            else:
                for layer in self.recurrent_layers:
                    self.key_cache[destination, layer, offset].copy_(
                        self.key_cache[source, layer, offset]
                    )
                    self.value_cache[destination, layer, offset].copy_(
                        self.value_cache[source, layer, offset]
                    )
        self._record_finalized(allocation, position, exit_depth)

    @cached_property
    def key_cache(self):
        self._materialize_paged()
        return self._key_cache

    @cached_property
    def value_cache(self):
        self._materialize_paged()
        return self._value_cache

    def _materialize_paged(self):
        if self._key_cache is None:
            key = torch.empty(self._paged_shape, device=self.device, dtype=self.dtype)
            value = torch.empty_like(key)
            self._key_cache, self._value_cache = key, value
        if self._contiguous is not None:
            self._contiguous.export(self._key_cache, self._value_cache, self.block_size)
            self._contiguous = None
        # Preserve direct Tensor attribute access for the default paged backends.
        self.key_cache, self.value_cache = self._key_cache, self._value_cache

    def _invalidate_contiguous(self, *, preserve_contiguous=False):
        if not preserve_contiguous and self._contiguous is not None and self._contiguous.rows:
            self._materialize_paged()

    @torch.no_grad()
    def read(self, layer: int, request_id: str, depth: int, length: int | None = None):
        """Materialize a fully initialized prefix as [token, KV head, dimension]."""
        self._validate_layer(layer)
        self._validate_depth(depth)
        allocation = self._get_allocation(request_id)
        if length is None:
            length = allocation.written[self._plane(depth)][layer].prefix
        if (
            not isinstance(length, Integral)
            or isinstance(length, bool)
            or not 0 <= length <= allocation.max_tokens
        ):
            raise ValueError("read length exceeds the request's reserved token budget")
        self._require_prefix(allocation, layer, depth, length)
        positions = torch.arange(length, device=self.device)
        table = torch.tensor(
            allocation.block_tables[self._plane(depth)], device=self.device, dtype=torch.long
        )
        blocks = table[positions // self.block_size]
        offsets = positions % self.block_size
        return self.key_cache[blocks, layer, offsets], self.value_cache[blocks, layer, offsets]
