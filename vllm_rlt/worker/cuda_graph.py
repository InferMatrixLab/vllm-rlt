"""Bounded recurrent-core graphs; request bookkeeping and sampling stay eager."""

import gc
import logging
import math
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from vllm_rlt.attention.sparse import (
    CAPTURE,
    CONSUME,
    DENSE,
    DeviceSparseDecodePlan,
    SparseDecodePlan,
)
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

    def __init__(self, cache, workspace=None):
        self.cache = cache
        # Sparse capture scratch shared by every bucket; replays run in order.
        self.workspace = workspace

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
        plan = getattr(batch, "reference_plan", None)
        if plan is not None:
            from vllm_rlt.kernels.sparse_decode import fused_paged_attention

            return fused_paged_attention(
                q,
                cache.key_cache[:, layer],
                cache.value_cache[:, layer],
                batch.block_tables,
                batch.context_lengths,
                plan,
                layer=layer,
                dense_attention=cache.attention,
                workspace=self.workspace,
            )
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
        self.proxy = _DeviceCache(cache)

    def select(self, hidden, batch):
        count = _query_count(batch)
        packed = batch.cu_seqlens_q is not None
        plan = getattr(batch, "reference_plan", None)
        if plan is not None and not plan.dense():
            # These entries carry no reference plan; replaying one would
            # silently run dense attention.
            self.fallbacks += 1
            return PreparedGraph(batch)
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
                entry.graph, entry.output = _capture(
                    entry.hidden,
                    self.capture_stream,
                    self.pool,
                    lambda: self.model.recurrent_prepared(
                        entry.hidden, entry.metadata, self.proxy, compute_gate=self.compute_gate
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


# Graph key of entries that replay a device plan (token pipelines); not a row mode.
DEVICE_PLAN = "device"


class DeviceRecurrentGraphs(RecurrentGraphs):
    """Core graphs whose actual positions and KV frontier remain on device.

    Allocation identity is checked on every call, outside capture. Entries retain
    only tensor metadata, never request allocations or CPU written-position state.

    With a fused sparse reference store, every bucket is captured once per row
    mode. A sparse entry owns a static plan whose slot ids are refilled before
    each replay; the live plan's liveness is checked on the host at that point,
    since replay runs no Python. With ``device_plans`` (token pipelines) the
    sparse entries instead replay a device plan, whose per-row modes are
    refilled alongside the slot ids.
    """

    def __init__(
        self,
        model,
        cache,
        execution,
        compute_gate,
        capture_sizes=None,
        reference_store=None,
        key_retention=None,
        device_plans=False,
    ):
        super().__init__(model, cache, execution, compute_gate)
        self.capture_sizes = capture_sizes or graph_capture_sizes(execution)
        self.reference_store = reference_store
        self.key_retention = key_retention
        self.sparse_replays = 0
        self.modes = (DENSE,)
        if reference_store is not None:
            from vllm_rlt.kernels.sparse_decode import capture_workspace

            if reference_store.padding_slot is None:
                raise ValueError("sparse CUDA graphs need a reference store padding slot")
            if reference_store.selected.shape[-1] != (reference_store.key_bucket or 1):
                # A replay must never see a grown index pool at a new address.
                raise ValueError("sparse CUDA graphs need the reference index pool at full size")
            self.modes = (DENSE, DEVICE_PLAN) if device_plans else (DENSE, CAPTURE, CONSUME)
            config = model.config
            width = math.ceil(config.max_position_embeddings / cache.block_size)
            self.proxy = _DeviceCache(
                cache,
                capture_workspace(
                    cache.device,
                    self.capture_sizes[-1],
                    config.num_attention_heads,
                    width * cache.block_size,
                    config.head_dim,
                ),
            )

    @staticmethod
    def _key(size, mode):
        # Dense entries keep their bucket-size keys; sparse ones add the mode.
        return size if mode == DENSE else (size, mode)

    def _entry(self, hidden, batch, size, mode):
        entry = _graph_entry(
            hidden,
            batch,
            self.cache,
            self.model.config.max_position_embeddings,
            count=size,
            bucketed=True,
        )
        entry.mode = mode
        if mode == DEVICE_PLAN:
            store = self.reference_store
            entry.slot_ids = torch.full(
                (size,), store.padding_slot, device=self.cache.device, dtype=torch.int64
            )
            # Padded rows stay dense on the never-bound padding slot.
            entry.modes = torch.full((size,), DENSE, device=self.cache.device, dtype=torch.int8)
            entry.metadata.reference_plan = DeviceSparseDecodePlan(
                modes=entry.modes,
                slot_ids=entry.slot_ids,
                leases=(),
                key_retention=self.key_retention,
                key_bucket=store.key_bucket,
                store=store,
            )
        elif mode != DENSE:
            store = self.reference_store
            entry.slot_ids = torch.full(
                (size,), store.padding_slot, device=self.cache.device, dtype=torch.int64
            )
            # Host liveness is checked against the live plan before replay,
            # so the captured plan names no host slots.
            entry.metadata.reference_plan = SparseDecodePlan(
                modes=(mode,) * size,
                slots=(),
                incarnations=(),
                key_retention=self.key_retention,
                key_bucket=store.key_bucket,
                slot_ids=entry.slot_ids,
                store=store,
            )
        return entry

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
        for mode in self.modes:
            for size in self.capture_sizes:
                entry = self._entry(hidden, batch, size, mode)
                _copy_graph_inputs(entry, hidden, batch, count=0)
                entry.graph, entry.output = _capture(
                    entry.hidden,
                    self.capture_stream,
                    self.pool,
                    lambda entry=entry: self.model.recurrent_prepared(
                        entry.hidden, entry.metadata, self.proxy, compute_gate=self.compute_gate
                    ),
                    startup=True,
                )
                self.entries[self._key(size, mode)] = entry
                self.captures += 1

    def _mode(self, batch):
        plan = getattr(batch, "reference_plan", None)
        if plan is None or plan.dense():
            return DENSE
        if self.reference_store is None or plan.store is not self.reference_store:
            return None
        if isinstance(plan, DeviceSparseDecodePlan):
            if DEVICE_PLAN not in self.modes or plan.key_retention != self.key_retention:
                return None
            return DEVICE_PLAN
        modes = set(plan.modes)
        if len(modes) != 1 or plan.key_retention != self.key_retention:
            return None
        return plan.modes[0]

    def select(self, hidden, batch):
        self.cache._require_live_batch(batch)
        count = _query_count(batch)
        mode = self._mode(batch)
        if not count or count > self.capture_sizes[-1] or mode is None:
            self.fallbacks += 1
            return PreparedGraph(batch)
        capacity = _bucket_size(self.capture_sizes, count)
        key = self._key(capacity, mode)
        entry = self.entries.get(key)
        if entry is None:
            entry = self._entry(hidden, batch, capacity, mode)
            entry.graph = None
            self.entries[key] = entry
        return PreparedGraph(batch, entry)

    @torch.inference_mode()
    def prepare_inputs(self, call, hidden):
        entry = call.entry
        if entry is not None and entry.mode != DENSE:
            plan = call.metadata.reference_plan
            self.reference_store._require_live(plan)
            count = _query_count(call.metadata)
            stream = torch.cuda.current_stream(self.cache.device)
            if self.last_event is not None:
                stream.wait_event(self.last_event)
            entry.slot_ids[count:].fill_(self.reference_store.padding_slot)
            entry.slot_ids[:count].copy_(plan.slot_ids[:count])
            if entry.mode == DEVICE_PLAN:
                entry.modes[count:].fill_(DENSE)
                entry.modes[:count].copy_(plan.modes[:count])
        super().prepare_inputs(call, hidden)

    @torch.inference_mode()
    def execute(self, call, hidden):
        sparse = call.entry is not None and call.entry.mode != DENSE
        output = super().execute(call, hidden)
        self.sparse_replays += sparse
        return output


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
