"""Paged KV with LAST-EXITED or SHARED physical storage planes.

LAST-EXITED copies the last computed per-layer KV into skipped deeper planes.
SHARED overwrites a single plane and needs no exit copies. Optional incremental
allocation separates logical capacity from physical pages. Immutable full-depth
prompt blocks can be shared through the prefix cache.

With ``kv_cache_dtype="fp8_e4m3"`` the pool stores K/V as float8 e4m3 with one
float32 scale per (block, layer, token, KV head) vector: values are quantized on
write and dequantized by the attention backends on read.
"""

import hashlib
import struct
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral

import torch

from vllm_rlt.attention import (
    AttentionBackend,
    AttentionRows,
    BackendCapabilities,
    backend_capabilities,
    create_backend,
    plan_attention_metadata,
    validate_backend_name,
)
from vllm_rlt.device import resolve_device
from vllm_rlt.request import ExecutionBatch
from vllm_rlt.worker.buffers import stage_metadata
from vllm_rlt.worker.metadata import (
    DeviceKVMetadata,
    GroupedKVMetadata,
    KVLayout,
    KVMetadata,
    MetadataBuilder,
)

KV_CACHE_DTYPES = ("auto", "fp8_e4m3")
FP8_E4M3_MAX = 448.0


def kv_bytes_per_block(
    num_layers, block_size, num_kv_heads, head_dim, element_size, kv_cache_dtype
):
    """Physical bytes of one block (K and V for all layers), including FP8 scales."""
    if kv_cache_dtype not in KV_CACHE_DTYPES:
        raise ValueError(f"kv_cache_dtype must be one of {KV_CACHE_DTYPES}")
    vector = head_dim * element_size if kv_cache_dtype == "auto" else head_dim + 4
    return 2 * num_layers * block_size * num_kv_heads * vector


def quantize_fp8(x: torch.Tensor, fixed_scale=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric e4m3 with dynamic per-vector scales or an explicit fixed scale."""
    x = x.float()
    # A reciprocal multiply, as CUDA division by a scalar does anyway: CPU, CUDA and
    # the fused Triton writer then produce the same scale bits.
    scale = (
        (x.abs().amax(-1) * (1.0 / FP8_E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny)
        if fixed_scale is None
        else torch.full(x.shape[:-1], fixed_scale, dtype=torch.float32, device=x.device)
    )
    codes = (x / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return codes.to(torch.float8_e4m3fn), scale


def dequantize_fp8(codes: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (codes.float() * scale.unsqueeze(-1)).to(dtype)


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
class _Allocation:
    max_tokens: int
    block_tables: tuple[tuple[int, ...], ...]
    written: list[list[_WrittenPositions]]
    # Native group -> plane -> mutable logical-page list -> one page per group layer.
    group_tables: tuple = ()
    group_epochs: list[list[int]] = field(default_factory=list)
    retired_pages: list[list[int]] = field(default_factory=list)
    consumed: list[list[_WrittenPositions]] = field(default_factory=list)
    reclaim_pending: bool = False
    transfer_leases: set[str] = field(default_factory=set)
    release_requested: bool = False
    prefix_keys: list[bytes] = field(default_factory=list)
    prefix_epoch: int = 0


@dataclass(frozen=True)
class KVGroup:
    layers: tuple[int, ...]
    window: int | None
    recurrent: bool
    planes: int


@dataclass(frozen=True)
class _SequenceCoverage:
    allocation: _Allocation
    plane: int
    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _PreparedCoverage:
    allocation_epoch: int
    sequences: Sequence[_SequenceCoverage]


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
    # FP8 pools only: per-vector scales, laid out like keys/values minus head_dim.
    key_scales: torch.Tensor | None = None
    value_scales: torch.Tensor | None = None
    fp8_scale: float | None = None


class KVCacheManager:
    """Own a fixed physical page pool shared by request/depth allocations.

    ``num_blocks`` counts total physical blocks across all recurrence depths.
    Legacy blocks contain all decoder layers. With ``layer_windows``, a block
    is one physical attention-layer page; groups own independent page bundles.
    Depth indices are zero-based throughout the interface.
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
        kv_cache_dtype: str = "auto",
        fp8_scale: float | None = None,
        layer_windows: Mapping[int, int | None] | None = None,
        reclaim_swa: bool = False,
        max_submission_tokens: int = 1,
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
        if kv_cache_dtype not in KV_CACHE_DTYPES:
            raise ValueError(f"kv_cache_dtype must be one of {KV_CACHE_DTYPES}")
        self.kv_cache_dtype = kv_cache_dtype
        if fp8_scale is not None:
            if kv_cache_dtype != "fp8_e4m3":
                raise ValueError("fp8_scale requires kv_cache_dtype='fp8_e4m3'")
            # Keep the value finite and normal after conversion to device float32.
            if (
                type(fp8_scale) not in (int, float)
                or not 2**-126 <= fp8_scale <= (2 - 2**-23) * 2**127
            ):
                raise ValueError("fp8_scale must be a positive, finite, normal float32 value")
        self.fp8_scale = fp8_scale
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
        self.grouped = layer_windows is not None
        if type(reclaim_swa) is not bool:
            raise ValueError("reclaim_swa must be a boolean")
        if type(max_submission_tokens) is not int or max_submission_tokens <= 0:
            raise ValueError("max_submission_tokens must be a positive integer")
        if reclaim_swa and not self.grouped:
            raise ValueError("SWA reclamation requires grouped storage")
        self.reclaim_swa = reclaim_swa
        self.max_submission_tokens = max_submission_tokens
        self.groups = ()
        self.layer_groups = {}
        if self.grouped:
            policies = dict(layer_windows)
            if not policies or any(
                type(layer) is not int or not 0 <= layer < num_layers for layer in policies
            ):
                raise ValueError("layer_windows must name actual attention layer indices")
            if not set(self.recurrent_layers) <= policies.keys():
                raise ValueError("recurrent layers must be actual attention layers")
            groups = {}
            for layer, window in sorted(policies.items()):
                if window is not None and (type(window) is not int or window <= 0):
                    raise ValueError("layer windows must be positive integers or None")
                recurrent = layer in self.recurrent_layers
                groups.setdefault((window, recurrent), []).append(layer)
            self.groups = tuple(
                KVGroup(tuple(layers), window, recurrent, self.storage_depths if recurrent else 1)
                for (window, recurrent), layers in groups.items()
            )
            self.layer_groups = {
                layer: (group_id, local)
                for group_id, group in enumerate(self.groups)
                for local, layer in enumerate(group.layers)
            }
        # Physical pool pages needed for one token block across all layers and planes.
        self.pages_per_token_block = (
            sum(len(group.layers) * group.planes for group in self.groups)
            if self.grouped
            else self.storage_depths
        )
        self.device = resolve_device(device)
        self.dtype = dtype
        self.attention = attention or create_backend(
            backend,
            self.device,
            dtype,
            head_dim,
            block_size,
            required_capabilities=BackendCapabilities(
                fp8_kv=kv_cache_dtype == "fp8_e4m3" and fp8_scale is None,
                fp8_kv_static=fp8_scale is not None,
            ),
        )
        self.attention_info = self.attention.info
        self.backend = self.attention_info["selected_backend"]
        capability = "fp8_kv" if fp8_scale is None else "fp8_kv_static"
        if kv_cache_dtype != "auto" and not getattr(self.attention.capabilities, capability):
            raise ValueError(
                f"the {self.attention_info['backend']} backend does not read FP8 KV "
                f"with {'per-vector' if fp8_scale is None else 'fixed'} scales"
            )
        if self.grouped and kv_cache_dtype != "auto":
            raise NotImplementedError("grouped KV does not support FP8 storage yet")
        shape = (num_blocks, 1 if self.grouped else num_layers, block_size, num_kv_heads, head_dim)
        storage = torch.float8_e4m3fn if kv_cache_dtype == "fp8_e4m3" else dtype
        self.key_cache = torch.empty(shape, device=self.device, dtype=storage)
        self.value_cache = torch.empty_like(self.key_cache)
        # One scale per stored vector; None for an unquantized pool.
        self.key_scale = self.value_scale = None
        if kv_cache_dtype == "fp8_e4m3":
            # Imported fixed-scale pages use the same local scale as locally written pages.
            scale = 1.0 if fp8_scale is None else fp8_scale
            self.key_scale = torch.full(shape[:-1], scale, device=self.device, dtype=torch.float32)
            self.value_scale = torch.full_like(self.key_scale, scale)
        # Keep the existing per-vector storage/lifecycle for this prototype. FA
        # borrows the immutable scalar; Torch/Triton continue reading the scale pool.
        self._fixed_scale = (
            torch.tensor(fp8_scale, device=self.device, dtype=torch.float32)
            if fp8_scale is not None
            else None
        )
        # Resolve an implicit CUDA index to the actual storage device once.
        self.device = self.key_cache.device
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
        self.enable_prefix_caching = enable_prefix_caching
        self.incremental_allocation = incremental_allocation
        # Full allocation already reserves each request's maximum KV capacity.
        self.watermark_blocks = int(watermark_ratio * num_blocks) if incremental_allocation else 0
        self._refs = [0] * num_blocks
        self._prefixes = OrderedDict()
        self._prefix_epoch = 0
        self._pending_prefixes = []
        self.prefix_hits = self.prefix_queries = 0

    @property
    def attention_capabilities(self) -> BackendCapabilities:
        return backend_capabilities(self.attention)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks) + sum(
            self._refs[b] == 1 for blocks in self._prefixes.values() for b in blocks
        )

    @property
    def submission_bounded(self) -> bool:
        """Whether SWA pages stay within the window plus one submission."""
        return self.reclaim_swa and self.incremental_allocation

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def bytes_per_block(self) -> int:
        return kv_bytes_per_block(
            1 if self.grouped else self.num_layers,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            self.key_cache.element_size(),
            self.kv_cache_dtype,
        )

    def required_blocks(self, max_tokens: int) -> int:
        if not isinstance(max_tokens, Integral) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        pages = (max_tokens + self.block_size - 1) // self.block_size
        if self.submission_bounded:
            # W-1 retained keys plus one append S, including worst page alignment.
            return sum(
                min(
                    pages,
                    (group.window - 1 + self.max_submission_tokens + 2 * self.block_size - 2)
                    // self.block_size,
                )
                * len(group.layers)
                * group.planes
                if group.window is not None
                else pages * len(group.layers) * group.planes
                for group in self.groups
            )
        return pages * self.pages_per_token_block

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
            self._prefix_epoch += 1
            self._drop_refs(blocks)
        if len(self._free_blocks) < count:
            return None
        blocks = [self._free_blocks.pop() for _ in range(count)]
        for b in blocks:
            assert self._refs[b] == 0
            self._refs[b] = 1
        return blocks

    def _prefix_keys(self, tokens, digest=b""):
        for start in range(0, len(tokens) // self.block_size * self.block_size, self.block_size):
            values = tokens[start : start + self.block_size]
            digest = hashlib.sha256(digest + struct.pack(f"<{len(values)}q", *values)).digest()
            yield digest

    def poll_prefixes(self):
        pending, self._pending_prefixes = self._pending_prefixes, []
        for request_id, allocation, tokens, length, events in pending:
            if self._allocations.get(request_id) is not allocation:
                continue
            events[:] = [event for event in events if not event.query()]
            if events:
                self._pending_prefixes.append((request_id, allocation, tokens, length, events))
                continue
            self.publish_prefix(request_id, tokens, length)

    def publish_prefix(self, request_id, tokens, length, event=None):
        """Index complete prompt pages for reuse without copying their KV.

        length is the prefilled extent; len(tokens) is the full prompt length.
        A supplied event defers publication until the GPU writes complete.
        The full prompt stays immutable for the allocation's lifetime.
        """
        if not self.enable_prefix_caching:
            return
        allocation = self._get_allocation(request_id)
        if event is not None:
            ready = event.query()
            for index, (_, owner, snapshot, frontier, events) in enumerate(self._pending_prefixes):
                if owner is allocation:
                    if not ready:
                        events.append(event)
                    # Keep one prompt snapshot per allocation. Retain every
                    # unfinished event: later submissions may complete first.
                    self._pending_prefixes[index] = (
                        request_id,
                        allocation,
                        snapshot,
                        max(frontier, length),
                        events,
                    )
                    return
            if not ready:
                self._pending_prefixes.append(
                    (request_id, allocation, tuple(tokens), length, [event])
                )
                return
        # Prefix entries retain KV, not the final hidden state needed by CODA.
        # Leave at least the last prompt token to recompute on a cache hit;
        # full-page alignment may leave additional tokens to recompute.
        publishable_tokens = min(length, len(tokens) - 1)
        if allocation.prefix_epoch != self._prefix_epoch:
            allocation.prefix_keys.clear()
            allocation.prefix_epoch = self._prefix_epoch
        keys = allocation.prefix_keys
        start = len(keys)
        digest = keys[-1] if keys else b""
        # Eviction invalidates the saved chain's publication frontier. Without
        # eviction, hash only new token pages instead of revisiting every chunk.
        for index, key in enumerate(
            self._prefix_keys(tokens[start * self.block_size : publishable_tokens], digest),
            start=start,
        ):
            if key in self._prefixes:
                keys.append(key)
                continue
            end = (index + 1) * self.block_size
            # Every layer that attention reads must have a contiguous written
            # prefix. GPU completion alone does not establish this coverage.
            if any(
                written.prefix < end
                for plane, layers in enumerate(allocation.written)
                for layer, written in enumerate(layers)
                if self._plane_owns(plane, layer)
            ):
                break
            blocks = self._prefix_page(allocation, index)
            if blocks is None:
                break
            for b in blocks:
                self._refs[b] += 1
            self._prefixes[key] = blocks
            keys.append(key)

    def _prefix_page(self, allocation, index):
        """Every physical page holding token page ``index``, or None if any is gone.

        Grouped entries are flattened group -> plane -> layer for
        ``_group_prefix_tables`` to split back into bundles.
        """
        if not self.grouped:
            return tuple(t[index] for t in allocation.block_tables)
        bundles = [
            table[index] if index < len(table) else None
            for planes in allocation.group_tables
            for table in planes
        ]
        if any(bundle is None for bundle in bundles):
            return None
        return tuple(b for bundle in bundles for b in bundle)

    def _plane_owns(self, plane: int, layer: int) -> bool:
        """Whether ``layer`` keeps KV history in ``plane``.

        Boundary (non-recurrent) layers run once per token, at depth 0, so
        they only ever write plane 0. Recurrent layers own every plane.
        Grouped storage takes ownership from the layer's group policy.
        """
        if self.grouped:
            if layer not in self.layer_groups:
                return False
            group_id, _ = self.layer_groups[layer]
            return plane == 0 or self.groups[group_id].recurrent
        return plane == 0 or layer in self.recurrent_layers

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
        cached_tokens = len(prefix) * self.block_size
        if initial_tokens is None:
            initial_tokens = (
                min(max_tokens, cached_tokens + self.max_submission_tokens)
                if self.submission_bounded
                else max_tokens
            )
        if self.submission_bounded and initial_tokens - cached_tokens > self.max_submission_tokens:
            raise ValueError("uncached initial KV exceeds the configured submission bound")
        if not 0 < initial_tokens <= max_tokens:
            raise ValueError("invalid initial KV frontier")
        if cached_tokens > initial_tokens:
            raise ValueError("prefix exceeds initial KV frontier")
        pages = (initial_tokens + self.block_size - 1) // self.block_size
        retired_pages, cached_tables, claimed = self._prefix_claim(prefix)
        for b in claimed:
            self._refs[b] += 1
        tail = pages - len(prefix)
        if self.grouped:
            tables = self._claim_group_tables(tail, tables=cached_tables)
        else:
            blocks = self._claim(tail * self.storage_depths)
            tables = (
                None
                if blocks is None
                else tuple(
                    tuple(g[d] for g in prefix) + tuple(blocks[d * tail : (d + 1) * tail])
                    for d in range(self.storage_depths)
                )
            )
        if tables is None:
            self._drop_refs(claimed)
            return False
        # A hit restores exactly the coverage publish_prefix required.
        written = [
            [
                _WrittenPositions(prefix=cached_tokens if self._plane_owns(plane, layer) else 0)
                for layer in range(self.num_layers)
            ]
            for plane in range(self.storage_depths)
        ]
        self._allocations[request_id] = _Allocation(
            int(max_tokens),
            () if self.grouped else tables,
            written,
            group_tables=tables if self.grouped else (),
            group_epochs=[[0] * group.planes for group in self.groups],
            retired_pages=retired_pages,
            consumed=(
                [
                    [_WrittenPositions(prefix=positions.prefix) for positions in plane]
                    for plane in written
                ]
                if self.reclaim_swa
                else []
            ),
        )
        self._allocation_epoch += 1
        self.prefix_queries += 1
        self.prefix_hits += cached_tokens
        return True

    def _claim_group_tables(self, new_pages, *, tables=()):
        """Append ``new_pages`` fresh bundles to every plane, or None if the pool is short."""
        fresh = self._claim(new_pages * self.pages_per_token_block)
        if fresh is None:
            return None
        if not tables:
            tables = tuple(tuple([] for _ in range(group.planes)) for group in self.groups)
        offset = 0
        for group, planes in zip(self.groups, tables):
            width = len(group.layers)
            for table in planes:
                for _ in range(new_pages):
                    table.append(tuple(fresh[offset : offset + width]))
                    offset += width
        assert offset == len(fresh)
        return tables

    def ensure_capacity(self, request_id, tokens):
        allocation = self._get_allocation(request_id)
        if not 0 < tokens <= allocation.max_tokens:
            raise ValueError("KV growth exceeds logical capacity")
        pages = (tokens + self.block_size - 1) // self.block_size
        if self.grouped:
            current = len(allocation.group_tables[0][0])
            extra = pages - current
            if extra <= 0:
                return True
            added = self._claim_group_tables(extra)
            if added is None:
                return False
            for planes, new_planes in zip(allocation.group_tables, added):
                for table, tail in zip(planes, new_planes):
                    table.extend(tail)
            for epochs in allocation.group_epochs:
                for plane in range(len(epochs)):
                    epochs[plane] += 1
            self._allocation_epoch += 1
            return True
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
        return True

    def truncate_suffix(self, request_id: str, frontier: int) -> None:
        """Invalidate an uncommitted suffix, retaining reserved physical pages.

        Call only after all suffix users complete. Retaining pages preserves the
        admission reservation; stale bytes are hidden by written/context lengths.
        Shared prompt pages must never be truncated or subsequently overwritten.
        """
        self._require_legacy()
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
            self._allocation_epoch += 1
            if self.grouped:
                self._drop_refs(
                    b
                    for planes in allocation.group_tables
                    for table in planes
                    for bundle in reversed(table)
                    if bundle is not None
                    for b in reversed(bundle)
                )
            else:
                self._drop_refs(b for table in allocation.block_tables for b in reversed(table))
            self._pending_prefixes = [
                pending for pending in self._pending_prefixes if pending[1] is not allocation
            ]

    def pin_transfer(self, request_id: str, transfer_id: str):
        self._require_legacy()
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
        self._require_legacy()
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
        allocation = self._get_allocation(request_id)
        if self.grouped:
            return sum(
                (len(table) - allocation.retired_pages[group_id][plane]) * len(group.layers)
                for group_id, group in enumerate(self.groups)
                for plane, table in enumerate(allocation.group_tables[group_id])
            )
        return len(allocation.block_tables[0]) * self.storage_depths

    def charged_blocks(self, request_id: str) -> int:
        """Blocks already spent against this request's ``required_blocks`` peak.

        A window-bounded peak covers resident pages only. A full peak also
        covers retired SWA pages: they went back to the pool and are never
        needed again, so they are not future growth.
        """
        if not self.grouped or self.submission_bounded:
            return self.allocated_blocks(request_id)
        allocation = self._get_allocation(request_id)
        return sum(
            len(table) * len(group.layers)
            for group_id, group in enumerate(self.groups)
            for table in allocation.group_tables[group_id]
        )

    def get_block_table(self, request_id: str, depth: int) -> tuple[int, ...]:
        self._require_legacy()
        self._validate_depth(depth)
        return self._get_allocation(request_id).block_tables[self._plane(depth)]

    def plane_block_tables(self, request_id: str) -> tuple[tuple[int, ...], ...]:
        """One block table per plane.

        This tuple is indexed by plane, not by loop depth. LAST_EXITED has
        ``max_loops`` planes and SHARED has one, so ``tables[2]`` fails under
        SHARED while ``get_block_table(rid, 2)`` works under both layouts.
        """
        self._require_legacy()
        return self._get_allocation(request_id).block_tables

    def has_transfer_lease(self, request_id: str) -> bool:
        return bool(self._get_allocation(request_id).transfer_leases)

    def exclusive_prefix_blocks(self, prefix: Sequence[Sequence[int]]) -> int:
        """How many blocks a prefix hit claims that only the prefix cache holds.

        ``num_free_blocks`` counts those blocks as free because they could be
        evicted. Claiming them makes them non-evictable, so admission has to
        budget them on top of the fresh blocks it needs. Expired SWA pages the
        hit does not attach stay evictable.
        """
        _, _, claimed = self._prefix_claim(prefix)
        return sum(self._refs[b] == 1 for b in claimed)

    def token_written(self, request_id: str, position: int, depth: int) -> bool:
        """Whether every layer has written KV for ``position`` at ``depth``."""
        allocation = self._get_allocation(request_id)
        self._validate_depth(depth)
        self._validate_position(allocation, position)
        return self._token_written(allocation, position, depth)

    def _token_written(self, allocation: _Allocation, position: int, depth: int) -> bool:
        if self.grouped:
            return all(
                position in allocation.written[self._group_plane(group, depth)][layer]
                for group in self.groups
                for layer in group.layers
            )
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
        self._require_legacy()
        allocation = self._get_allocation(request_id)
        if allocation.transfer_leases:
            raise RuntimeError("cannot snapshot KV during a transfer")
        blocks = [b for table in allocation.block_tables for b in table]
        keys = torch.empty((len(blocks), *self.key_cache.shape[1:]), dtype=self.key_cache.dtype)
        values = torch.empty_like(keys)
        for row, block in enumerate(blocks):
            keys[row].copy_(self.key_cache[block])
            values[row].copy_(self.value_cache[block])
        key_scales = value_scales = None
        if self.key_scale is not None:
            key_scales = self.key_scale[blocks].cpu()
            value_scales = self.value_scale[blocks].cpu()
        return KVSnapshot(
            max_tokens=allocation.max_tokens,
            pages=len(allocation.block_tables[0]),
            written=deepcopy(allocation.written),
            keys=keys,
            values=values,
            key_scales=key_scales,
            value_scales=value_scales,
            fp8_scale=self.fp8_scale,
        )

    def restore(self, request_id: str, snapshot: KVSnapshot) -> None:
        """Load a snapshot back into a fresh allocation of the same size.

        ``allocate`` stays a separate step because it can fail and that
        decision belongs to the scheduler. The copies go on the current
        stream; the caller decides when other streams may read them.
        """
        self._require_legacy()
        allocation = self._get_allocation(request_id)
        if allocation.max_tokens != snapshot.max_tokens:
            raise ValueError("snapshot capacity does not match the allocation")
        if len(allocation.block_tables[0]) != snapshot.pages:
            raise ValueError("snapshot page count does not match the allocation")
        if any(w.prefix or w.pending for plane in allocation.written for w in plane):
            raise RuntimeError("cannot restore into an allocation that already holds KV")
        if (self.key_scale is None) != (snapshot.key_scales is None):
            raise ValueError("snapshot KV dtype does not match the pool")
        if self.fp8_scale != snapshot.fp8_scale:
            raise ValueError("snapshot FP8 scale policy does not match the pool")
        blocks = [b for table in allocation.block_tables for b in table]
        for row, block in enumerate(blocks):
            self.key_cache[block].copy_(snapshot.keys[row])
            self.value_cache[block].copy_(snapshot.values[row])
            if self.key_scale is not None:
                self.key_scale[block].copy_(snapshot.key_scales[row])
                self.value_scale[block].copy_(snapshot.value_scales[row])
        allocation.written = deepcopy(snapshot.written)

    def _prefix_claim(self, prefix):
        """Expired pages per group plane, the tables a hit attaches, and its blocks.

        A hit skips the leading SWA pages its next token cannot read, so it
        takes no references to them.
        """
        cached_tokens = len(prefix) * self.block_size
        retired_pages = [
            [self._expired_swa_pages(group, cached_tokens) if self.reclaim_swa else 0]
            * group.planes
            for group in self.groups
        ]
        if not (self.grouped and prefix):
            return retired_pages, (), [b for bundle in prefix for b in bundle]
        tables = self._group_prefix_tables(prefix, retired_pages)
        claimed = [
            b
            for planes in tables
            for table in planes
            for bundle in table
            if bundle is not None
            for b in bundle
        ]
        return retired_pages, tables, claimed

    def _group_prefix_tables(self, prefix, retired_pages):
        """Attach only the cached pages each group can still read."""
        offset, result = 0, []
        for group, retired_planes in zip(self.groups, retired_pages):
            planes = []
            width = len(group.layers)
            for retired in retired_planes:
                end = offset + width
                planes.append(
                    [None] * retired + [tuple(page[offset:end]) for page in prefix[retired:]]
                )
                offset = end
            result.append(tuple(planes))
        assert all(len(page) == offset for page in prefix)
        return tuple(result)

    def _group_plane(self, group: KVGroup, depth: int) -> int:
        """Storage plane owning ``depth`` for ``group``; boundary layers keep depth 0."""
        return self._plane(depth) if group.recurrent else 0

    def _group_table(self, allocation, layer, depth):
        group_id, local = self.layer_groups[layer]
        group = self.groups[group_id]
        if not group.recurrent and depth != 0:
            raise ValueError("boundary attention layers only own depth zero")
        return allocation.group_tables[group_id][self._group_plane(group, depth)], local

    def _layer_table(self, allocation, layer, depth):
        bundles, local = self._group_table(allocation, layer, depth)
        return tuple(-1 if bundle is None else bundle[local] for bundle in bundles)

    def layer_group(self, layer: int) -> tuple[int, KVGroup]:
        """Read-only group policy for a compact KV layer; this does not pin pages.

        Group IDs are local to this manager. The returned policy is immutable.
        """
        self._validate_layer(layer)
        if not self.grouped:
            raise NotImplementedError("layer groups require grouped KV storage")
        group_id, _ = self.layer_groups[layer]
        return group_id, self.groups[group_id]

    def layer_block_table(self, request_id, layer, depth):
        """Read-only absolute logical-page map for one actual attention layer."""
        self._validate_layer(layer)
        self._validate_depth(depth)
        allocation = self._get_allocation(request_id)
        if not self.grouped:
            return allocation.block_tables[self._plane(depth)]
        return self._layer_table(allocation, layer, depth)

    def layer_view(self, batch, layer):
        """Resolve native ownership; SWA read tables use a page-aligned local span.

        Rows, RoPE positions and write addresses retain their absolute meaning.
        Only the attention planner sees positions relative to each row's span.
        """
        self._validate_layer(layer)
        self._require_live_batch(batch)
        if not isinstance(batch, GroupedKVMetadata):
            raise TypeError("grouped KV requires a logical grouped descriptor")
        group_id, _ = self.layer_groups[layer]
        group = self.groups[group_id]
        epochs = tuple(
            a.group_epochs[group_id][self._group_plane(group, d)] for a, d, _ in batch.rows
        )
        cached = batch.layer_views.get(layer)
        if cached is not None and cached[0] == epochs:
            return cached[1]
        tables, blocks, read_positions = [], [], []
        positions = [p for _, _, p in batch.rows]
        for allocation, depth, position in batch.rows:
            bundles, local = self._group_table(allocation, layer, depth)
            page = position // self.block_size
            if page >= len(bundles):
                raise ValueError("KV row exceeds the acquired physical capacity")
            bundle = bundles[page]
            blocks.append(-1 if bundle is None else bundle[local])
            first = (
                0
                if group.window is None
                else max(0, position - group.window + 1) // self.block_size
            )
            tables.append(tuple(-1 if b is None else b[local] for b in bundles[first : page + 1]))
            read_positions.append(position - first * self.block_size)
        offsets = [p % self.block_size for p in positions]
        if any(block < 0 for block in blocks):
            raise RuntimeError("current KV write/query page is no longer resident")
        if self.reclaim_swa:
            self._require_resident_rows(batch, layer, group.window)
        if batch.writable and len(set(zip(blocks, offsets))) != len(blocks):
            raise ValueError("duplicate physical KV write addresses")
        attention = plan_attention_metadata(
            AttentionRows(block_tables=tables, positions=read_positions),
            block_size=self.block_size,
        )
        layout = KVLayout(
            self,
            batch.rows,
            batch.allocations,
            positions,
            blocks,
            offsets,
            attention,
            batch.writable,
        )
        view = stage_metadata((layout,), self._stage, padded_rows=len(batch.position_ids))[0]
        batch.layer_views[layer] = (epochs, view)
        return view

    def retained_range(self, request_id, layer, depth):
        """Absolute resident start and logical written end for a layer/plane."""
        self._validate_layer(layer)
        self._validate_depth(depth)
        allocation = self._get_allocation(request_id)
        start = 0
        if self.grouped:
            self._group_table(allocation, layer, depth)
            group_id, _ = self.layer_groups[layer]
            plane = self._group_plane(self.groups[group_id], depth)
            start = allocation.retired_pages[group_id][plane] * self.block_size
        return start, allocation.written[self._plane(depth)][layer].prefix

    def _require_resident_rows(self, batch, layer, window):
        for allocation, depth, position in batch.rows:
            bundles, _ = self._group_table(allocation, layer, depth)
            start = 0 if window is None else max(0, position - window + 1)
            if position // self.block_size >= len(bundles):
                raise ValueError("KV row exceeds the acquired physical capacity")
            if any(
                bundles[page] is None
                for page in range(start // self.block_size, position // self.block_size + 1)
            ):
                raise RuntimeError("required KV history is no longer resident")

    def _record_consumed(self, layer, batch):
        group_id, _ = self.layer_groups[layer]
        if self.groups[group_id].window is None:
            return
        group = self.groups[group_id]
        for allocation, depth, position in batch.rows:
            # Shared recurrent history is still read by later loops of this token.
            if self.layout == "shared" and group.recurrent and depth != self.max_loops - 1:
                continue
            allocation.consumed[self._plane(depth)][layer].add(position)
            allocation.reclaim_pending = True

    def _expired_swa_pages(self, group, frontier):
        """Count whole pages the next token at ``frontier`` cannot read."""
        if group.window is None:
            return 0
        return max(0, frontier - group.window + 1) // self.block_size

    def finish_submission(self, event=None):
        """Return whole expired bundles only after this submission's final reader.

        Consumption candidates are host progress, not device completion. Reuse
        the ordinary runner's event and wait only when physical pages retire.
        """
        if not self.reclaim_swa:
            return
        plans = []
        pending = [a for a in self._allocations.values() if a.reclaim_pending]
        for allocation in pending:
            for group_id, group in enumerate(self.groups):
                if group.window is None:
                    continue
                for plane, table in enumerate(allocation.group_tables[group_id]):
                    frontier = min(
                        allocation.consumed[plane][layer].prefix for layer in group.layers
                    )
                    stop = self._expired_swa_pages(group, frontier)
                    start = allocation.retired_pages[group_id][plane]
                    if stop > start:
                        plans.append((allocation, group_id, plane, start, stop))
        if plans:
            if event is None and self.device.type == "cuda":
                raise ValueError("CUDA reclamation requires the submission completion event")
            if event is not None:
                event.synchronize()
                # Deferred publications are now complete; index them before retiring.
                self.poll_prefixes()
            for allocation, group_id, plane, start, stop in plans:
                table = allocation.group_tables[group_id][plane]
                released = []
                for page in range(start, stop):
                    if table[page] is not None:
                        released.extend(table[page])
                        table[page] = None
                allocation.retired_pages[group_id][plane] = stop
                allocation.group_epochs[group_id][plane] += 1
                self._drop_refs(released)
        for allocation in pending:
            allocation.reclaim_pending = False

    def _plane(self, depth: int) -> int:
        return depth if self.layout == "last_exited" else 0

    def _storage_layer(self, layer: int) -> int:
        """Physical page slot holding ``layer``; grouped pages hold exactly one."""
        return 0 if self.grouped else layer

    def _require_legacy(self) -> None:
        """Require legacy all-layer KV storage."""
        if self.grouped:
            raise NotImplementedError("grouped KV supports ordinary layer views only")

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
        if self.grouped and layer not in self.layer_groups:
            raise ValueError("layer has no attention KV storage")

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

    def prepare_recurrent(
        self, batch: ExecutionBatch, *, packed=False
    ) -> KVMetadata | GroupedKVMetadata:
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
    ) -> KVMetadata | GroupedKVMetadata:
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
    ) -> tuple[KVMetadata | GroupedKVMetadata, ...]:
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

    def _require_live_batch(
        self, batch: KVMetadata | GroupedKVMetadata | DeviceKVMetadata
    ) -> _PreparedCoverage | None:
        if self.grouped and isinstance(batch, DeviceKVMetadata):
            raise NotImplementedError("grouped KV does not support device-owned metadata")
        if isinstance(batch, (KVMetadata, GroupedKVMetadata)) and batch.owner is not self:
            raise ValueError("prepared KV batch belongs to a different cache manager")
        plan = None
        reusable = self._prepared_coverage is not None and isinstance(batch, KVMetadata)
        epoch = self._allocation_epoch
        if reusable:
            plan = self._prepared_coverage.get(batch)
            if plan is not None and plan.allocation_epoch == epoch:
                return plan
        for request_id, allocation in batch.allocations:
            if self._allocations.get(request_id) is not allocation:
                raise RuntimeError(f"stale prepared KV batch for request {request_id!r}")
        if reusable:
            sequences = (
                _sequence_coverage(batch.rows, self._plane) if plan is None else plan.sequences
            )
            plan = _PreparedCoverage(epoch, sequences)
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
        batch: KVMetadata | GroupedKVMetadata | DeviceKVMetadata,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        self._validate_layer(layer)
        coverage = self._require_live_batch(batch)
        if isinstance(batch, DeviceKVMetadata):
            self._validate_tensor(k, len(batch.position_ids), "k")
            self._validate_tensor(v, len(batch.position_ids), "v")
            if self.key_scale is not None:
                self._store_layer(
                    layer, batch.write_blocks, batch.write_offsets, k, v, mask=batch.write_mask
                )
            elif batch.write_mask is not None:
                from vllm_rlt.kernels.routing import masked_kv_store

                masked_kv_store(self.key_cache[:, layer], self.value_cache[:, layer], k, v, batch)
            else:
                self.key_cache[batch.write_blocks, layer, batch.write_offsets] = k
                self.value_cache[batch.write_blocks, layer, batch.write_offsets] = v
            return
        if self.grouped:
            batch = self.layer_view(batch, layer)
        if not batch.writable:
            raise ValueError("a read-only prepared KV batch cannot be written")
        self._validate_tensor(k, len(batch.position_ids), "k")
        self._validate_tensor(v, len(batch.position_ids), "v")
        rows = len(batch.rows)
        if not rows:
            return
        self._store_layer(
            self._storage_layer(layer), batch.write_blocks, batch.write_offsets, k[:rows], v[:rows]
        )
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

    def _store_layer(self, layer, blocks, offsets, k, v, *, mask=None, active=None) -> None:
        """Write one layer's K/V rows, quantizing them for an FP8 pool.

        FP8 only: ``mask`` (bool [rows]) and ``active`` (int32 row count) skip
        rows on the device, like the unquantized masked and graph writers.
        """
        if self.key_scale is None:
            if mask is not None or active is not None:
                raise ValueError("row masks are handled by the unquantized device writers")
            self.key_cache[blocks, layer, offsets] = k
            self.value_cache[blocks, layer, offsets] = v
            return
        if (
            self.backend == "triton"
            or mask is not None
            or active is not None
            or (self.fp8_scale is not None and self.device.type == "cuda")
        ):
            # One fused launch instead of about 20 small kernels per layer write.
            # Masked writes always take it, as the unquantized masked writers are
            # Triton too and boolean indexing would read the mask on the host.
            from vllm_rlt.kernels.triton_attention import store_fp8

            store_fp8(
                k,
                v,
                self.key_cache[:, layer],
                self.value_cache[:, layer],
                self.key_scale[:, layer],
                self.value_scale[:, layer],
                blocks,
                offsets,
                FP8_E4M3_MAX,
                mask=mask,
                active=active,
                fixed_scale=self.fp8_scale,
            )
            return
        k_codes, k_scale = quantize_fp8(k, self.fp8_scale)
        v_codes, v_scale = quantize_fp8(v, self.fp8_scale)
        self.key_cache[blocks, layer, offsets] = k_codes
        self.value_cache[blocks, layer, offsets] = v_codes
        self.key_scale[blocks, layer, offsets] = k_scale
        self.value_scale[blocks, layer, offsets] = v_scale

    def _layer_scales(self, layer) -> dict:
        """Extra attention arguments for one layer: its scales, or nothing."""
        if self.key_scale is None:
            return {}
        if self._fixed_scale is not None and not self.attention.capabilities.fp8_kv:
            return dict(key_scale=self._fixed_scale, value_scale=self._fixed_scale)
        return dict(key_scale=self.key_scale[:, layer], value_scale=self.value_scale[:, layer])

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
        batch: KVMetadata | GroupedKVMetadata | DeviceKVMetadata,
        q: torch.Tensor,
        *,
        window: int | None = None,
    ) -> torch.Tensor:
        self._validate_layer(layer)
        coverage = self._require_live_batch(batch)
        self._validate_tensor(q, len(batch.position_ids), "q", query=True)
        logical = batch
        if self.grouped:
            group_id, _ = self.layer_groups[layer]
            if window != self.groups[group_id].window:
                raise ValueError("native attention window must match the layer policy")
            batch = self.layer_view(batch, layer)
        if window is not None:
            if batch.cu_seqlens_q is None:
                output = self._attend_windowed(layer, batch, q, window)
                if self.reclaim_swa:
                    self._record_consumed(layer, logical)
                return output
            if self.reclaim_swa:
                raise NotImplementedError("SWA reclamation requires unpacked prefill")
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
                **self._layer_scales(layer),
            )
        if not batch.rows:
            return torch.empty_like(q)
        if coverage is None:
            for allocation, depth, position in batch.rows:
                self._require_prefix(allocation, layer, depth, position + 1)
        else:
            for sequence in coverage.sequences:
                self._require_prefix(
                    sequence.allocation, layer, sequence.plane, sequence.ranges[-1][1]
                )
        return self.attention.attend_metadata(
            q,
            self.key_cache[:, self._storage_layer(layer)],
            self.value_cache[:, self._storage_layer(layer)],
            batch,
            **self._layer_scales(layer),
            **({} if window is None else {"window": window}),
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
        # Device-owned positions are ordered by the runner's stream; the host
        # frontier lags them, as for unwindowed attention.
        if not isinstance(batch, DeviceKVMetadata):
            if not batch.rows:
                return torch.empty_like(q)
            for allocation, depth, position in batch.rows:
                self._require_prefix(allocation, layer, depth, position + 1)
        return self.attention.windowed(
            q,
            self.key_cache[:, self._storage_layer(layer)],
            self.value_cache[:, self._storage_layer(layer)],
            batch.block_tables,
            batch.context_lengths,
            window,
            **self._layer_scales(layer),
        )

    @torch.no_grad()
    def finalize_token(self, request_id: str, position: int, exit_depth: int) -> None:
        """Propagate the final executed depth's K/V into every unexecuted depth."""
        if self.reclaim_swa and exit_depth != self.max_loops - 1:
            raise NotImplementedError(
                "SWA reclamation currently requires full fixed-loop execution"
            )
        allocation = self._require_exit_written(request_id, position, exit_depth)
        if self.layout == "shared" or exit_depth == self.max_loops - 1:
            return
        logical_page, offset = divmod(position, self.block_size)
        if self.grouped:
            for layer in self.recurrent_layers:
                source = self.layer_block_table(request_id, layer, exit_depth)[logical_page]
                if source < 0:
                    raise RuntimeError("cannot finalize reclaimed KV")
                for depth in range(exit_depth + 1, self.max_loops):
                    destination = self.layer_block_table(request_id, layer, depth)[logical_page]
                    if destination < 0:
                        raise RuntimeError("cannot finalize into reclaimed KV")
                    self.key_cache[destination, 0, offset].copy_(self.key_cache[source, 0, offset])
                    self.value_cache[destination, 0, offset].copy_(
                        self.value_cache[source, 0, offset]
                    )
            self._record_finalized(allocation, position, exit_depth)
            return
        source = allocation.block_tables[self._plane(exit_depth)][logical_page]
        destinations = [
            allocation.block_tables[self._plane(depth)][logical_page]
            for depth in range(exit_depth + 1, self.max_loops)
        ]
        # Basic-index copies avoid a blocking host->device index tensor on the
        # boundary stream. That transfer would serialize final core and routing.
        tensors = [self.key_cache, self.value_cache]
        if self.key_scale is not None:
            tensors += [self.key_scale, self.value_scale]
        for destination in destinations:
            for tensor in tensors:
                if len(self.recurrent_layers) == self.num_layers:
                    tensor[destination, :, offset].copy_(tensor[source, :, offset])
                else:
                    for layer in self.recurrent_layers:
                        tensor[destination, layer, offset].copy_(tensor[source, layer, offset])
        self._record_finalized(allocation, position, exit_depth)

    @torch.no_grad()
    def read(self, layer: int, request_id: str, depth: int, length: int | None = None):
        """Materialize a fully initialized prefix as [token, KV head, dimension]."""
        self._validate_layer(layer)
        self._validate_depth(depth)
        allocation = self._get_allocation(request_id)
        if self.grouped:
            self._group_table(allocation, layer, depth)
        if length is None:
            length = allocation.written[self._plane(depth)][layer].prefix
        if (
            not isinstance(length, Integral)
            or isinstance(length, bool)
            or not 0 <= length <= allocation.max_tokens
        ):
            raise ValueError("read length exceeds the request's reserved token budget")
        self._require_prefix(allocation, layer, depth, length)
        if self.grouped and length and self.retained_range(request_id, layer, depth)[0]:
            raise RuntimeError(
                "KV prefix has been reclaimed; use retained_range and layer_block_table"
            )
        positions = torch.arange(length, device=self.device)
        table = torch.tensor(
            (
                self._layer_table(allocation, layer, depth)
                if self.grouped
                else allocation.block_tables[self._plane(depth)]
            ),
            device=self.device,
            dtype=torch.long,
        )
        blocks = table[positions // self.block_size]
        offsets = positions % self.block_size
        storage_layer = self._storage_layer(layer)
        keys = self.key_cache[blocks, storage_layer, offsets]
        values = self.value_cache[blocks, storage_layer, offsets]
        if self.key_scale is None:
            return keys, values
        return (
            dequantize_fp8(keys, self.key_scale[blocks, storage_layer, offsets], self.dtype),
            dequantize_fp8(values, self.value_scale[blocks, storage_layer, offsets], self.dtype),
        )
