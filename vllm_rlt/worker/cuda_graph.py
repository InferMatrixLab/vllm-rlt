"""Bounded recurrent-core graphs; request bookkeeping and sampling stay eager."""

import gc
import math
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from vllm_rlt.worker.metadata import DeviceKVMetadata


def _query_count(batch):
    # Device-owned positions describe actual rows; ordinary metadata may include padding.
    return len(batch.position_ids) if isinstance(batch, DeviceKVMetadata) else len(batch.rows)


def _capture(hidden, stream, pool, run):
    """Warm and capture one stable-input graph on its own stream."""
    torch.cuda.synchronize(hidden.device)
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    stream.synchronize()
    # torch.cuda.graph no longer collects garbage before capture. Do it here, so
    # an allocation inside the capture cannot run the collector and free an earlier
    # engine's cyclic CUDA state (graphs, streams, pinned buffers) mid-capture.
    gc.collect()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool, stream=stream):
        output = run()
    return graph, output


class _DeviceCache:
    """Capture only GPU cache operations, never Python allocation/written state."""

    def __init__(self, cache):
        self.cache = cache

    def _write_prepared(self, layer, batch, k, v):
        self.cache._store_layer(layer, batch.write_blocks, batch.write_offsets, k, v)

    def _attend_prepared(self, layer, batch, q):
        cache = self.cache
        keys, values, tables = cache._layer_kv(layer, batch.block_tables)
        if batch.cu_seqlens_q is not None:
            return cache.attention.prefill(
                q,
                keys,
                values,
                tables,
                batch.context_lengths,
                batch.cu_seqlens_q,
                batch.max_seqlen_q,
            )
        return cache.attention(q, keys, values, tables, batch.context_lengths)


def _graph_entry(hidden, batch, cache, max_position_embeddings):
    count, tables = _query_count(batch), len(batch.context_lengths)
    width = math.ceil(max_position_embeddings / cache.block_size)
    return SimpleNamespace(
        hidden=torch.empty_like(hidden[:count]),
        metadata=SimpleNamespace(
            position_ids=torch.empty(count, device=cache.device, dtype=torch.long),
            write_blocks=torch.empty(count, device=cache.device, dtype=torch.long),
            write_offsets=torch.empty(count, device=cache.device, dtype=torch.long),
            block_tables=torch.zeros((tables, width), device=cache.device, dtype=torch.int32),
            context_lengths=torch.empty(tables, device=cache.device, dtype=torch.int32),
            cu_seqlens_q=(
                torch.empty(tables + 1, device=cache.device, dtype=torch.int32)
                if batch.cu_seqlens_q is not None
                else None
            ),
            max_seqlen_q=batch.max_seqlen_q,
        ),
    )


def _copy_graph_inputs(entry, hidden, batch):
    count = _query_count(batch)
    entry.hidden.copy_(hidden[:count])
    entry.metadata.position_ids.copy_(batch.position_ids[:count])
    entry.metadata.write_blocks.copy_(batch.write_blocks[:count])
    entry.metadata.write_offsets.copy_(batch.write_offsets[:count])
    entry.metadata.context_lengths.copy_(batch.context_lengths)
    if batch.cu_seqlens_q is not None:
        entry.metadata.cu_seqlens_q.copy_(batch.cu_seqlens_q)
    entry.metadata.block_tables[:, : batch.block_tables.shape[1]].copy_(batch.block_tables)


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
        key = (count, len(batch.context_lengths), batch.max_seqlen_q if packed else 1, packed)
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
            or count > self.execution.cuda_graph_max_batch_size
            or not contiguous
            or (
                key not in self.entries
                and len(self.entries) >= self.execution.cuda_graph_max_graphs
            )
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
        output = tuple(t.clone() if t is not None else None for t in entry.output)
        self.last_event = torch.cuda.Event()
        self.last_event.record(torch.cuda.current_stream(self.cache.device))
        host_rows = () if isinstance(batch, DeviceKVMetadata) else batch.rows
        for allocation, depth, pos in host_rows:
            for layer in self.cache.recurrent_layers:
                allocation.written[self.cache._plane(depth)][layer].add(pos)
        self.replays += 1
        return output


class CodaGraphs:
    """Select and stage coda inputs before replay; commit KV after submission."""

    def __init__(self, model, execution, cache=None):
        self.model, self.execution = model, execution
        self.cache = cache
        self.entries = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=next(model.parameters()).device)
        self.last_event = None
        self.captures = self.replays = self.fallbacks = 0

    def select(self, hidden, batch=None):
        count = len(hidden)
        if self.cache is not None and (
            batch is None
            or not batch.writable
            or len(batch.rows) != count
            or batch.cu_seqlens_q is not None
        ):
            raise ValueError("coda graphs require matching decode rows and KV metadata")
        if (
            not count
            or count > self.execution.cuda_graph_max_batch_size
            or (
                count not in self.entries
                and len(self.entries) >= self.execution.cuda_graph_max_graphs
            )
        ):
            self.fallbacks += 1
            return PreparedGraph(batch)
        entry = self.entries.get(count)
        if entry is None:
            entry = (
                _graph_entry(hidden, batch, self.cache, self.model.config.max_position_embeddings)
                if self.cache is not None
                else SimpleNamespace(hidden=torch.empty_like(hidden))
            )
            entry.graph = None
            self.entries[count] = entry
        return PreparedGraph(batch, entry)

    @torch.inference_mode()
    def prepare_inputs(self, call, hidden):
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
                entry.hidden.copy_(hidden)
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
                entry.graph, entry.output = _capture(entry.hidden, self.stream, self.pool, run)
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
        output = entry.output.clone()
        self.last_event = torch.cuda.Event()
        self.last_event.record(torch.cuda.current_stream(hidden.device))
        if self.cache is not None:
            for allocation, depth, position in batch.rows:
                for layer in self.model.coda_kv_layers:
                    allocation.written[self.cache._plane(depth)][layer].add(position)
        self.replays += 1
        return output
