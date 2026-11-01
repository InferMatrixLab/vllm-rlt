"""Bounded recurrent-core graphs; request bookkeeping and sampling stay eager."""

import gc
import logging
import math
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from vllm_rlt.config import cuda_graph_rows
from vllm_rlt.worker.metadata import DeviceKVMetadata

logger = logging.getLogger(__name__)


def _query_count(batch):
    # Device-owned positions describe actual rows; ordinary metadata may include padding.
    return len(batch.position_ids) if isinstance(batch, DeviceKVMetadata) else len(batch.rows)


def graph_capture_sizes(execution, preferred=()):
    """Prefer full-batch shapes, then spend remaining entries on smaller rows.

    Under a tight budget retain the largest preferred shapes, including the
    maximum reachable row count. No preferred bucket exceeds the workload's
    range. Callers without preferred sizes use a ladder from the global limit.
    """
    limit = cuda_graph_rows(execution)
    budget = execution.cuda_graph_max_graphs
    preferred = sorted({min(size, limit) for size in preferred if size > 0})
    sizes = preferred[-budget:] if preferred else [limit]
    size = sizes[0]
    while size > 1 and len(sizes) < budget:
        size = (size + 1) // 2
        sizes.insert(0, size)
    return tuple(sizes)


def speculative_capture_sizes(execution, scheduler, speculative):
    """Buckets for async speculative rounds of max_num_seqs requests drafting 0..K."""
    rows = [
        min(scheduler.max_num_seqs * (drafts + 1), scheduler.max_num_batched_tokens)
        for drafts in range(speculative.num_speculative_tokens + 1)
    ]
    return graph_capture_sizes(execution, rows)


def _bucket_size(sizes, count):
    return next((size for size in sizes if count <= size), count)


def _cache_full(owner, key):
    full = key not in owner.entries and len(owner.entries) >= owner.execution.cuda_graph_max_graphs
    if full and not owner.warned_cache_full:
        logger.warning(
            "%s cache is full (%d shapes); unseen shapes will run eager. "
            "Existing shapes still replay; lazy capture synchronizes the device.",
            type(owner).__name__,
            len(owner.entries),
        )
        owner.warned_cache_full = True
    return full


def _capture(hidden, stream, pool, run, *, startup=False):
    """Warm and capture one stable-input graph on its own stream."""
    if startup:
        # Startup inputs, model/KV initialization and prior pool replays are
        # ordered on the producer stream. Each preceding capture is drained.
        stream.wait_stream(torch.cuda.current_stream(hidden.device))
    else:
        # Runtime lazy capture may encounter other in-flight users of the cache.
        torch.cuda.synchronize(hidden.device)
    gc_enabled = None
    try:
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        stream.synchronize()
        # Preserve main's cleanup of old engine cycles before entering capture.
        gc.collect()
        graph = torch.cuda.CUDAGraph()
        # Collecting an unreachable old CUDAGraph here calls cudaGraphExecDestroy,
        # which is forbidden during capture and invalidates the new graph.
        # Also protect allocations that create new cycles during capture.
        gc_enabled = gc.isenabled()
        gc.disable()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            output = run()
        return graph, output
    finally:
        # A failed warmup can leave KV writes on the capture stream without a
        # compute-stream event. Drain it before tensor/allocation owners unwind.
        try:
            stream.synchronize()
        finally:
            if gc_enabled:
                gc.enable()


class _DeviceCache:
    """Capture only GPU cache operations, never Python allocation/written state."""

    def __init__(self, cache):
        self.cache = cache

    def _write_prepared(self, layer, batch, k, v):
        cache = self.cache
        if batch.active_rows is not None:
            from vllm_rlt.kernels.graph_kv import write_graph_kv

            write_graph_kv(cache.key_cache[:, layer], cache.value_cache[:, layer], batch, k, v)
        elif batch.write_mask is not None:
            from vllm_rlt.kernels.routing import masked_kv_store

            masked_kv_store(cache.key_cache[:, layer], cache.value_cache[:, layer], k, v, batch)
        else:
            cache.key_cache[batch.write_blocks, layer, batch.write_offsets] = k
            cache.value_cache[batch.write_blocks, layer, batch.write_offsets] = v

    def _attend_prepared(self, layer, batch, q):
        cache = self.cache
        if batch.cu_seqlens_q is not None:
            return cache.attention.prefill(
                q,
                cache.key_cache[:, layer],
                cache.value_cache[:, layer],
                batch.block_tables,
                batch.context_lengths,
                batch.cu_seqlens_q,
                batch.max_seqlen_q,
            )
        return cache.attention(
            q,
            cache.key_cache[:, layer],
            cache.value_cache[:, layer],
            batch.block_tables,
            batch.context_lengths,
        )


def _graph_entry(
    hidden, batch, cache, max_position_embeddings, *, count=None, packed=None, bucketed=False
):
    count = _query_count(batch) if count is None else count
    packed = batch.cu_seqlens_q is not None if packed is None else packed
    tables = count if bucketed else len(batch.context_lengths)
    width = math.ceil(max_position_embeddings / cache.block_size)
    return SimpleNamespace(
        hidden=hidden.new_empty((count, *hidden.shape[1:])),
        metadata=SimpleNamespace(
            active_rows=torch.zeros((), device=cache.device, dtype=torch.int32)
            if bucketed
            else None,
            position_ids=torch.empty(count, device=cache.device, dtype=torch.long),
            write_blocks=torch.empty(count, device=cache.device, dtype=torch.long),
            write_offsets=torch.empty(count, device=cache.device, dtype=torch.long),
            block_tables=torch.zeros((tables, width), device=cache.device, dtype=torch.int32),
            context_lengths=torch.empty(tables, device=cache.device, dtype=torch.int32),
            cu_seqlens_q=(
                torch.empty(tables + 1, device=cache.device, dtype=torch.int32) if packed else None
            ),
            max_seqlen_q=batch.max_seqlen_q if packed else 1,
            write_mask=(
                torch.empty(count, device=cache.device, dtype=torch.bool)
                if bucketed or getattr(batch, "write_mask", None) is not None
                else None
            ),
        ),
    )


def _copy_graph_inputs(entry, hidden, batch, *, count=None):
    count = _query_count(batch) if count is None else count
    metadata = entry.metadata
    if metadata.active_rows is not None:
        metadata.active_rows.fill_(count)
        entry.hidden[count:].zero_()
        metadata.position_ids[count:].zero_()
        metadata.write_blocks[count:].zero_()
        metadata.write_offsets[count:].zero_()
        # Padding has zero context, so attention never reads its (stale) table
        # rows, and active rows read only pages they cover; no table reset.
        metadata.context_lengths[count:].zero_()
    entry.hidden[:count].copy_(hidden[:count])
    metadata.position_ids[:count].copy_(batch.position_ids[:count])
    metadata.write_blocks[:count].copy_(batch.write_blocks[:count])
    metadata.write_offsets[:count].copy_(batch.write_offsets[:count])
    if metadata.write_mask is not None:
        metadata.write_mask[count:].zero_()
        if getattr(batch, "write_mask", None) is None:
            metadata.write_mask[:count].fill_(True)
        else:
            metadata.write_mask[:count].copy_(batch.write_mask[:count])
    tables = len(batch.context_lengths)
    metadata.context_lengths[:tables].copy_(batch.context_lengths)
    if metadata.cu_seqlens_q is not None:
        metadata.cu_seqlens_q.copy_(batch.cu_seqlens_q)
    metadata.block_tables[:tables, : batch.block_tables.shape[1]].copy_(batch.block_tables)


@dataclass
class PreparedGraph:
    """A selected graph (or eager fallback), with per-call metadata ownership."""

    metadata: object
    entry: object = None
    inputs_ready: bool = False


class RecurrentGraphs:
    def __init__(self, model, cache, execution, compute_gate):
        self.model, self.cache = model, cache
        self.execution, self.compute_gate = execution, compute_gate
        self.entries = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_stream = torch.cuda.Stream(device=cache.device)
        self.last_event = None
        self.captures = self.replays = self.fallbacks = 0
        self.warned_cache_full = False

    def select(self, hidden, batch):
        count = _query_count(batch)
        packed = batch.cu_seqlens_q is not None
        if len(batch.position_ids) != count:
            batch = replace(
                batch,
                position_ids=batch.position_ids[:count],
                context_lengths=batch.context_lengths[:count],
                block_tables=batch.block_tables[:count],
            )
        key = (
            count,
            len(batch.context_lengths),
            batch.max_seqlen_q if packed else 1,
            packed,
            getattr(batch, "write_mask", None) is not None,
        )
        frontier = {}
        contiguous = True
        host_rows = () if isinstance(batch, DeviceKVMetadata) else batch.rows
        for allocation, depth, pos in host_rows:
            row_key = (id(allocation), depth)
            previous = frontier.get(row_key)
            if previous is not None and pos != previous + 1:
                contiguous = False
                break
            frontier[row_key] = pos
        if (
            not count
            or count > cuda_graph_rows(self.execution)
            or not contiguous
            or _cache_full(self, key)
        ):
            self.fallbacks += 1
            return PreparedGraph(batch)
        entry = self.entries.get(key)
        if entry is None:
            entry = _graph_entry(
                hidden, batch, self.cache, self.model.config.max_position_embeddings
            )
            entry.graph = None
            self.entries[key] = entry
        return PreparedGraph(batch, entry)

    @torch.inference_mode()
    def prepare_inputs(self, call, hidden):
        batch, entry = call.metadata, call.entry
        if entry is not None:
            cache = self.cache
            cache._require_live_batch(batch)
            # Prefix validity is checked at consumption, after earlier loops have
            # submitted their writes, not when all graph choices are selected.
            seen = set()
            # Device-owned progress is ordered by the runner's stream. The CPU
            # written frontier intentionally lags in-flight speculative rounds.
            host_rows = () if isinstance(batch, DeviceKVMetadata) else batch.rows
            for allocation, depth, pos in host_rows:
                key = (id(allocation), depth)
                if key not in seen:
                    for layer in cache.recurrent_layers:
                        cache._require_prefix(allocation, layer, depth, pos)
                    seen.add(key)
            stream = torch.cuda.current_stream(cache.device)
            if self.last_event is not None:
                stream.wait_event(self.last_event)
            _copy_graph_inputs(entry, hidden, batch)
            if entry.graph is None:
                proxy = _DeviceCache(cache)
                entry.graph, entry.output = _capture(
                    entry.hidden,
                    self.capture_stream,
                    self.pool,
                    lambda: self.model.recurrent_prepared(
                        entry.hidden, entry.metadata, proxy, compute_gate=self.compute_gate
                    ),
                )
                self.captures += 1
        call.inputs_ready = True

    @torch.inference_mode()
    def execute(self, call, hidden):
        if not call.inputs_ready:
            raise RuntimeError("graph inputs must be prepared before execution")
        batch, entry = call.metadata, call.entry
        call.inputs_ready = False
        if entry is None:
            return self.model.recurrent_prepared(
                hidden, batch, self.cache, compute_gate=self.compute_gate
            )
        entry.graph.replay()
        output = tuple(
            t[: _query_count(batch)].clone() if t is not None else None for t in entry.output
        )
        self.last_event = torch.cuda.Event()
        self.last_event.record(torch.cuda.current_stream(self.cache.device))
        host_rows = () if isinstance(batch, DeviceKVMetadata) else batch.rows
        for allocation, depth, pos in host_rows:
            for layer in self.cache.recurrent_layers:
                allocation.written[self.cache._plane(depth)][layer].add(pos)
        self.replays += 1
        return output


class DeviceRecurrentGraphs(RecurrentGraphs):
    """Core graphs whose actual positions and KV frontier remain on device.

    Allocation identity is checked on every call, outside capture. Entries retain
    only tensor metadata, never request allocations or CPU written-position state.
    """

    def __init__(self, model, cache, execution, compute_gate, capture_sizes=None):
        super().__init__(model, cache, execution, compute_gate)
        self.capture_sizes = capture_sizes or graph_capture_sizes(execution)

    @torch.inference_mode()
    def prepare(self):
        """Capture the finite bucket set before any serving requests exist."""
        parameter = next(self.model.parameters())
        hidden = parameter.new_empty((0, self.model.config.hidden_size))
        batch = SimpleNamespace(
            position_ids=torch.empty(0, device=self.cache.device, dtype=torch.long),
            write_blocks=torch.empty(0, device=self.cache.device, dtype=torch.long),
            write_offsets=torch.empty(0, device=self.cache.device, dtype=torch.long),
            context_lengths=torch.empty(0, device=self.cache.device, dtype=torch.int32),
            block_tables=torch.empty((0, 1), device=self.cache.device, dtype=torch.int32),
            cu_seqlens_q=None,
        )
        proxy = _DeviceCache(self.cache)
        for size in self.capture_sizes:
            entry = _graph_entry(
                hidden,
                batch,
                self.cache,
                self.model.config.max_position_embeddings,
                count=size,
                bucketed=True,
            )
            _copy_graph_inputs(entry, hidden, batch, count=0)
            entry.graph, entry.output = _capture(
                entry.hidden,
                self.capture_stream,
                self.pool,
                lambda: self.model.recurrent_prepared(
                    entry.hidden, entry.metadata, proxy, compute_gate=self.compute_gate
                ),
                startup=True,
            )
            self.entries[size] = entry
            self.captures += 1

    def select(self, hidden, batch):
        self.cache._require_live_batch(batch)
        count = _query_count(batch)
        if not count or count > self.capture_sizes[-1]:
            self.fallbacks += 1
            return PreparedGraph(batch)
        capacity = _bucket_size(self.capture_sizes, count)
        entry = self.entries.get(capacity)
        if entry is None:
            entry = _graph_entry(
                hidden,
                batch,
                self.cache,
                self.model.config.max_position_embeddings,
                count=capacity,
                bucketed=True,
            )
            entry.graph = None
            self.entries[capacity] = entry
        return PreparedGraph(batch, entry)


class CodaGraphs:
    """Select and stage coda inputs before replay; commit KV after submission."""

    def __init__(self, model, execution, cache=None, *, bucketed=False, capture_sizes=None):
        self.model, self.execution = model, execution
        self.cache = cache
        if bucketed and cache is not None:
            raise ValueError("bucketed coda graphs require a row-independent Ouro head")
        self.capture_sizes = (capture_sizes or graph_capture_sizes(execution)) if bucketed else None
        self.entries = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=next(model.parameters()).device)
        self.last_event = None
        self.captures = self.replays = self.fallbacks = 0
        self.warned_cache_full = False

    @torch.inference_mode()
    def prepare(self):
        if self.capture_sizes is None:
            return
        parameter = next(self.model.parameters())
        for size in self.capture_sizes:
            hidden = parameter.new_zeros((size, self.model.config.hidden_size))
            call = self.select(hidden)
            self.prepare_inputs(call, hidden, startup=True)
            # Warm replay without allocating a clone that startup discards.
            call.entry.graph.replay()
            self.last_event = torch.cuda.Event()
            self.last_event.record(torch.cuda.current_stream(hidden.device))
            call.inputs_ready = False
        self.replays = 0

    def select(self, hidden, batch=None):
        count = len(hidden)
        if self.cache is not None and (
            batch is None
            or not batch.writable
            or len(batch.rows) != count
            or batch.cu_seqlens_q is not None
        ):
            raise ValueError("coda graphs require matching decode rows and KV metadata")
        capacity = _bucket_size(self.capture_sizes, count) if self.capture_sizes else count
        limit = self.capture_sizes[-1] if self.capture_sizes else cuda_graph_rows(self.execution)
        if not count or count > limit or _cache_full(self, capacity):
            self.fallbacks += 1
            return PreparedGraph(batch)
        entry = self.entries.get(capacity)
        if entry is None:
            entry = (
                _graph_entry(hidden, batch, self.cache, self.model.config.max_position_embeddings)
                if self.cache is not None
                else SimpleNamespace(hidden=hidden.new_empty((capacity, *hidden.shape[1:])))
            )
            entry.graph = None
            self.entries[capacity] = entry
        return PreparedGraph(batch, entry)

    @torch.inference_mode()
    def prepare_inputs(self, call, hidden, *, startup=False):
        batch, entry = call.metadata, call.entry
        if self.cache is not None:
            self.cache._require_live_batch(batch)
            for allocation, depth, position in batch.rows:
                for layer in self.model.coda_kv_layers:
                    self.cache._require_prefix(allocation, layer, depth, position)
        if entry is not None:
            stream = torch.cuda.current_stream(hidden.device)
            if self.last_event is not None:
                stream.wait_event(self.last_event)
            if self.cache is not None:
                _copy_graph_inputs(entry, hidden, batch)
            else:
                entry.hidden[: len(hidden)].copy_(hidden)
                entry.hidden[len(hidden) :].zero_()
            if entry.graph is None:
                run = (
                    (
                        lambda: self.model.coda_prepared(
                            entry.hidden, entry.metadata, _DeviceCache(self.cache)
                        )
                    )
                    if self.cache is not None
                    else (lambda: self.model.coda(entry.hidden))
                )
                entry.graph, entry.output = _capture(
                    entry.hidden, self.stream, self.pool, run, startup=startup
                )
                self.captures += 1
        call.inputs_ready = True

    @torch.inference_mode()
    def execute(self, call, hidden):
        if not call.inputs_ready:
            raise RuntimeError("graph inputs must be prepared before execution")
        batch, entry = call.metadata, call.entry
        call.inputs_ready = False
        if entry is None:
            return (
                self.model.coda_prepared(hidden, batch, self.cache)
                if self.cache is not None
                else self.model.coda(hidden)
            )
        entry.graph.replay()
        output = entry.output[: len(hidden)].clone()
        self.last_event = torch.cuda.Event()
        self.last_event.record(torch.cuda.current_stream(hidden.device))
        if self.cache is not None:
            for allocation, depth, position in batch.rows:
                for layer in self.model.coda_kv_layers:
                    allocation.written[self.cache._plane(depth)][layer].add(position)
        self.replays += 1
        return output
