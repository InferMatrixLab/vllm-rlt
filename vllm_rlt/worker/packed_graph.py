"""Packed Inkling graphs with eager, batch-dependent native attention."""

import gc
import math
from contextlib import contextmanager

import torch

from vllm_rlt.config import cuda_graph_rows
from vllm_rlt.request import Operation
from vllm_rlt.worker.cuda_graph import (
    _bucket_size,
    _cache_full,
    _copy_graph_inputs,
    _DeviceCache,
    _graph_entry,
    _query_count,
)
from vllm_rlt.worker.metadata import MixedRecurrentKVMetadata


def packed_capture_sizes(limit):
    """Use the vLLM throughput grid for prefill and mixed token counts."""
    sizes = []
    size = 1
    while size < limit:
        sizes.append(size)
        step = min(size, max(8, size / 4), 0.07 * max(size, 4096))
        stride = 1 << math.floor(math.log2(step))
        size = (size // stride + 1) * stride
    return (*sizes, limit)


class _AttentionSegments:
    """Capture model segments around in-place attention calls."""

    def __init__(self, pool=None):
        self.segments = []
        self.shared_pool = pool is not None
        self.pool = torch.cuda.graph_pool_handle() if pool is None else pool
        self.current = None
        self.batch = None

    def _begin(self):
        self.current = torch.cuda.CUDAGraph()
        self.current.capture_begin(pool=self.pool)

    def _finish(self):
        graph, self.current = self.current, None
        graph.capture_end()
        return graph

    @contextmanager
    def capture(self, batch):
        self.batch = batch
        try:
            self._begin()
            yield
        finally:
            try:
                if self.current is not None:
                    self.segments.append((self._finish(), None))
            finally:
                self.batch = None

    def attention(self, run, *tensors):
        graph = self._finish()
        # The eager break reads real queries and writes live KV during capture.
        graph.replay()
        run(self.batch, *tensors)
        # Graphs own the device addresses. Non-owning aliases let the allocator
        # reuse transient allocations when capturing the next bucket.
        if self.shared_pool:
            tensors = tuple(_borrow_graph_tensor(tensor) for tensor in tensors)
        self.segments.append((graph, (run, tensors)))
        self._begin()

    def replay(self, batch):
        for graph, attention in self.segments:
            graph.replay()
            if attention is not None:
                run, tensors = attention
                run(batch, *tensors)


def _borrow_graph_tensor(tensor):
    return None if tensor is None else torch.ops._C.weak_ref_tensor(tensor)


class PackedGraphs:
    """Own packed prelude, core and KV-only coda graphs apart from decode."""

    def __init__(self, model, cache, execution, scheduler):
        self.model, self.cache = model, cache
        self.execution = execution
        self.max_rows = scheduler.max_num_batched_tokens
        self.max_sequences = scheduler.max_num_seqs
        self.speculative_core = False
        self.bucketed = execution.packed_graph_buckets
        self.min_rows = (
            1
            if self.bucketed
            else min(scheduler.prefill_chunk_size, 1 + cuda_graph_rows(execution))
        )
        self.capture_sizes = packed_capture_sizes(self.max_rows) if self.bucketed else ()
        self.pool = torch.cuda.graph_pool_handle() if self.bucketed else None
        self.entries = {}
        self.stream = torch.cuda.Stream(device=cache.device)
        self.proxy = _DeviceCache(cache)
        self.last_event = None
        self.captures = self.replays = self.fallbacks = 0
        self.warned_cache_full = False
        self.key_layers = {
            Operation.PRELUDE: model.prelude_kv_layers,
            Operation.RECURRENT: model.recurrent_kv_layers,
            Operation.CODA: model.coda_kv_layers,
        }

    def enable_speculative_core(self):
        self.speculative_core = True

    def _forward(self, operation, inputs, metadata, cache, *, generators=None, compute_gate=False):
        if operation == Operation.PRELUDE:
            return self.model.prelude_prepared(inputs, metadata, cache, generators=generators)
        if operation == Operation.RECURRENT:
            return self.model.recurrent_prepared(inputs, metadata, cache, compute_gate=compute_gate)
        return self.model.coda_prepared(inputs, metadata, cache, compute_logits=False)

    def _capture_pieces(self, entry, live, warm, run, *, startup=False):
        # As with a lazy full capture, drain existing consumers of the live KV pages.
        if startup:
            self.stream.wait_stream(torch.cuda.current_stream(self.cache.device))
        else:
            torch.cuda.synchronize(self.cache.device)
        try:
            with torch.cuda.stream(self.stream):
                warm()
                self.stream.synchronize()
                if not startup or gc.isenabled():
                    gc.collect()
                enabled = gc.isenabled()
                gc.disable()
                try:
                    entry.metadata.attention_segments = entry.segments
                    with entry.segments.capture(live):
                        return run()
                finally:
                    if enabled:
                        gc.enable()
        finally:
            self.stream.synchronize()

    @torch.inference_mode()
    def forward(
        self, operation, inputs, batch, *, generators=None, compute_gate=False, startup=False
    ):
        # Inkling uses the same recurrent forward with either caller gate flag.
        compute_gate = compute_gate and getattr(self.model, "supports_recurrent_gate", True)
        count = _query_count(batch)
        sequences = len(batch.context_lengths)
        host_batch = batch.prompt if isinstance(batch, MixedRecurrentKVMetadata) else batch
        masked = self.speculative_core and operation == Operation.RECURRENT
        max_sequences = (
            min(self.max_rows, self.max_sequences * self.model.config.total_ut_steps)
            if masked
            else self.max_sequences
        )
        layers = self.key_layers[operation]
        variant = (
            self.model.recurrent_graph_variant(batch) if operation == Operation.RECURRENT else None
        )
        size = _bucket_size(self.capture_sizes, count)
        key = (operation, size, variant, compute_gate)
        if (
            not self.min_rows <= count <= self.max_rows
            or sequences > max_sequences
            or host_batch.packed_ranges is None
            or not host_batch.writable
            or _cache_full(self, key)
        ):
            self.fallbacks += 1
            return self._forward(
                operation,
                inputs,
                batch,
                self.cache,
                generators=generators,
                compute_gate=compute_gate,
            )

        self.cache._require_live_batch(batch)
        for seq in host_batch.packed_ranges:
            for layer in layers:
                self.cache._require_prefix(seq.allocation, layer, seq.depth, seq.start)
        entry = self.entries.get(key)
        if entry is None:
            entry = _graph_entry(
                inputs,
                batch,
                self.cache,
                self.model.config.max_position_embeddings,
                count=size,
                bucketed=self.bucketed,
                table_count=min(size, max_sequences),
                masked=masked,
            )
            entry.metadata.graph_variant = variant
            entry.graph = None
            entry.segments = _AttentionSegments(self.pool)
        stream = torch.cuda.current_stream(inputs.device)
        if self.last_event is not None:
            stream.wait_event(self.last_event)
        _copy_graph_inputs(entry, inputs, batch)
        # Live attention between existing graph segments uses the fused
        # row mapping. Capture uses the staged query table and pad mask.
        live = self.model.batch_type(
            self.proxy,
            batch,
            self.model.config,
            fused_sconv_metadata=entry.graph is not None and not startup,
        )
        if entry.graph is None:

            def run():
                return self._forward(
                    operation, entry.hidden, entry.metadata, self.proxy, compute_gate=compute_gate
                )

            def warm():
                return self._forward(
                    operation, inputs, batch, self.proxy, compute_gate=compute_gate
                )

            entry.output = self._capture_pieces(entry, live, warm, run, startup=startup)
            entry.graph = entry.segments
            if self.bucketed:
                if isinstance(entry.output, tuple):
                    entry.output = tuple(_borrow_graph_tensor(value) for value in entry.output)
                else:
                    entry.output = _borrow_graph_tensor(entry.output)
            self.entries[key] = entry
            self.captures += 1
        entry.graph.replay(live)
        output = entry.output
        if isinstance(output, tuple):
            output = tuple(value[:count].clone() if value is not None else None for value in output)
        elif output is not None:
            output = output[:count].clone()
        self.last_event = torch.cuda.Event()
        self.last_event.record(stream)
        for layer in layers:
            self.cache._mark_prepared_write(layer, host_batch, None)
        self.replays += 1
        return output
