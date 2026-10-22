"""Paged KV with LAST-EXITED or SHARED physical storage planes.

LAST-EXITED copies the last computed per-layer KV into skipped deeper planes.
SHARED overwrites a single plane and needs no exit copies. Optional incremental
allocation separates logical capacity from physical pages. Immutable full-depth
prompt blocks can be shared through the prefix cache.
"""

import hashlib
import struct
from collections import OrderedDict
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral

import torch

from vllm_rlt.attention import (
    AttentionBackend,
    BackendCapabilities,
    backend_capabilities,
    create_backend,
    validate_backend_name,
)
from vllm_rlt.core.execution import ExecutionBatch
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

    def __contains__(self, position: int) -> bool:
        return position < self.prefix or position in self.pending


@dataclass
class _Allocation:
    max_tokens: int
    block_tables: tuple[tuple[int, ...], ...]
    written: list[list[_WrittenPositions]]
    transfer_leases: set[str] = field(default_factory=set)
    release_requested: bool = False


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
        self.device = torch.device(device)
        self.dtype = dtype
        self.attention = attention or create_backend(
            backend, self.device, dtype, head_dim, block_size
        )
        self.attention_info = self.attention.info
        self.backend = self.attention_info["selected_backend"]
        shape = (num_blocks, num_layers, block_size, num_kv_heads, head_dim)
        self.key_cache = torch.empty(shape, device=self.device, dtype=dtype)
        self.value_cache = torch.empty_like(self.key_cache)
        # Resolve an implicit CUDA index to the actual storage device once.
        self.device = self.key_cache.device
        self._free_blocks = list(reversed(range(num_blocks)))
        self._allocations: dict[str, _Allocation] = {}
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
        for plane in allocation.written:
            for written in plane:
                written.prefix = min(written.prefix, frontier)
                written.pending = {p for p in written.pending if p < frontier}

    def free(self, request_id: str) -> None:
        allocation = self._allocations.get(request_id)
        if allocation is not None and allocation.transfer_leases:
            allocation.release_requested = True
            return
        allocation = self._allocations.pop(request_id, None)
        if allocation is not None:
            self._drop_refs(b for table in allocation.block_tables for b in reversed(table))

    def pin_transfer(self, request_id: str, transfer_id: str):
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
        blocks = [b for table in allocation.block_tables for b in table]
        for row, block in enumerate(blocks):
            self.key_cache[block].copy_(snapshot.keys[row])
            self.value_cache[block].copy_(snapshot.values[row])
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

    def _require_live_batch(self, batch: KVMetadata | DeviceKVMetadata) -> None:
        if isinstance(batch, KVMetadata) and batch.owner is not self:
            raise ValueError("prepared KV batch belongs to a different cache manager")
        for request_id, allocation in batch.allocations:
            if self._allocations.get(request_id) is not allocation:
                raise RuntimeError(f"stale prepared KV batch for request {request_id!r}")

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
        self, layer: int, batch: KVMetadata | DeviceKVMetadata, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        self._validate_layer(layer)
        self._require_live_batch(batch)
        if isinstance(batch, DeviceKVMetadata):
            self._validate_tensor(k, len(batch.position_ids), "k")
            self._validate_tensor(v, len(batch.position_ids), "v")
            self.key_cache[batch.write_blocks, layer, batch.write_offsets] = k
            self.value_cache[batch.write_blocks, layer, batch.write_offsets] = v
            return
        if not batch.writable:
            raise ValueError("a read-only prepared KV batch cannot be written")
        self._validate_tensor(k, len(batch.position_ids), "k")
        self._validate_tensor(v, len(batch.position_ids), "v")
        if not batch.rows:
            return
        self.key_cache[batch.write_blocks, layer, batch.write_offsets] = k[: len(batch.rows)]
        self.value_cache[batch.write_blocks, layer, batch.write_offsets] = v[: len(batch.rows)]
        for allocation, depth, position in batch.rows:
            allocation.written[self._plane(depth)][layer].add(position)

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
    ) -> torch.Tensor:
        """Apply causal GQA at each row's own position and recurrence depth."""
        self._validate_layer(layer)
        batch = self._prepare_batch(request_ids, depths, positions, for_write=False)
        return self._attend_prepared(layer, batch, q)

    @torch.no_grad()
    def _attend_prepared(
        self, layer: int, batch: KVMetadata | DeviceKVMetadata, q: torch.Tensor
    ) -> torch.Tensor:
        self._validate_layer(layer)
        self._require_live_batch(batch)
        self._validate_tensor(q, len(batch.position_ids), "q", query=True)
        if isinstance(batch, DeviceKVMetadata):
            return self.attention(
                q,
                self.key_cache[:, layer],
                self.value_cache[:, layer],
                batch.block_tables,
                batch.context_lengths,
            )
        if not batch.rows:
            return torch.empty_like(q)
        for allocation, depth, position in batch.rows:
            self._require_prefix(allocation, layer, depth, position + 1)
        if batch.cu_seqlens_q is not None:
            return self.attention.prefill(
                q,
                self.key_cache[:, layer],
                self.value_cache[:, layer],
                batch.block_tables,
                batch.context_lengths,
                batch.cu_seqlens_q,
                batch.max_seqlen_q,
            )
        return self.attention(
            q,
            self.key_cache[:, layer],
            self.value_cache[:, layer],
            batch.block_tables,
            batch.context_lengths,
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
