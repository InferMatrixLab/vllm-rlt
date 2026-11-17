"""Owned eager KV in attention layout, with cohort reuse and lossless paged escape.

Only append-only, uniform private traversals enter this storage. Logical pages
and written coverage remain with KVCacheManager. A generic storage access exports
all live rows before the manager permanently switches back to paged storage.
"""

import weakref
from dataclasses import dataclass

import torch

from vllm_rlt.worker.metadata import UniformSequenceRanges


@dataclass(eq=False)
class _Group:
    allocations: tuple
    pair: tuple
    live: int
    frontier: object = None


@dataclass(frozen=True)
class _AppendPlan:
    history: tuple | None
    refs: list | None
    group: _Group | None
    previous_pair: tuple | None
    same: bool
    epoch: int


class ContiguousKVStorage:
    def __init__(self, heads, dim, device, *, publish_coverage=None):
        self.heads, self.dim, self.device = heads, dim, device
        self.publish_coverage = publish_coverage
        self.rows = {}
        self.groups = {}
        self.stream = None
        self._metadata = None
        self._epoch = None
        self._geometry = None
        self.writes = self.regroups = self.compactions = 0

    def _on_owner_stream(self, stream=None):
        if self.device.type != "npu":
            return True
        stream = stream if stream is not None else torch.npu.current_stream(self.device)
        if self.stream is None:
            self.stream = stream
        return stream.stream_id == self.stream.stream_id

    def _context(self, batch, stream=None):
        if not self._on_owner_stream(stream):
            return None
        epoch = batch.owner._allocation_epoch
        if self._metadata is not None and self._metadata() is batch and self._epoch == epoch:
            return self._geometry
        self._metadata = weakref.ref(batch)
        self._epoch = epoch
        spans = batch.sequence_ranges
        self._geometry = None
        if isinstance(spans, UniformSequenceRanges):
            self._geometry = len(spans), spans.start, spans.query_count, spans.depth
        elif spans:
            start, count, depth = spans[0].start, spans[0].count, spans[0].depth
            if all((s.start, s.count, s.depth) == (start, count, depth) for s in spans):
                self._geometry = (len(spans), start, count, depth)
        return self._geometry

    def append(self, layer, batch, k, v, *, stream=None):
        context = self._context(batch, stream)
        if context is None:
            return False
        return self._append(layer, batch, k, v, context)

    def prepare_append(self, layer, batch, context):
        size, start, count, depth = context
        spans = batch.sequence_ranges
        slot = depth, layer
        group = self.groups.get(slot)
        same = group is not None and group.allocations is batch.allocations
        history = None
        refs = None
        if same:
            if group.pair[0].shape[2] != start:
                return None  # Overwrite/hole semantics use paged storage.
            history = group.pair
        else:
            refs = [self.rows.get((id(s.allocation), depth, layer)) for s in spans]
            if start == 0:
                if any(r is not None for r in refs):
                    return None
                if any(s.allocation.written[depth][layer].prefix for s in spans):
                    return None
            else:
                if any(r is None or r[0].pair[0].shape[2] != start for r in refs):
                    return None
                # Rebatch only the requested live rows. The usual stable cohort
                # takes the identity fast path above and needs no row lookups.
                history = tuple(
                    torch.cat([r[0].pair[i][r[1] : r[1] + 1] for r in refs], dim=0)
                    for i in range(2)
                )
        if self.publish_coverage is not None and not same:
            if any(
                s.allocation.written[depth][layer].prefix != start
                or s.allocation.written[depth][layer].pending
                for s in spans
            ):
                return None
        return _AppendPlan(
            history, refs, group, group.pair if group else None, same, batch.owner._allocation_epoch
        )

    def _append(self, layer, batch, k, v, context):
        plan = self.prepare_append(layer, batch, context)
        if plan is None:
            return False
        size, _, count, _ = context
        incoming = tuple(
            x.reshape(size, count, self.heads, self.dim).transpose(1, 2) for x in (k, v)
        )
        pair = (
            tuple(x.clone() for x in incoming)
            if plan.history is None
            else tuple(torch.cat((old, new), dim=2) for old, new in zip(plan.history, incoming))
        )
        self.commit_append(layer, batch, context, plan, pair)
        return True

    def commit_append(self, layer, batch, context, plan, pair):
        size, start, count, depth = context
        slot = depth, layer
        group = self.groups.get(slot)
        if (
            batch.owner._allocation_epoch != plan.epoch
            or group is not plan.group
            or (group is not None and group.pair is not plan.previous_pair)
        ):
            raise RuntimeError("stale prepared contiguous KV append")
        same, refs = plan.same, plan.refs
        spans = batch.sequence_ranges
        # Publish after BOTH copies enqueue. Public write inputs may be mutated,
        # and returned attention outputs never alias these storage tensors.
        if same:
            group.pair = pair
        else:
            group = _Group(batch.allocations, pair, size)
            displaced = set()
            for index, (span, previous) in enumerate(zip(spans, refs)):
                if previous is not None:
                    previous[0].live -= 1
                    displaced.add(previous[0])
                self.rows[id(span.allocation), depth, layer] = group, index
            self.groups[slot] = group
            for old in displaced:
                self._compact(old, depth, layer)
            self.regroups += 1
        if self.publish_coverage is not None:
            group.frontier = self.publish_coverage(
                spans, depth, layer, start + count, group.frontier if same else None
            )
        self.writes += 1

    def _compact(self, group, depth, layer):
        if not group.live or group.live == group.pair[0].shape[0]:
            return
        refs = []
        for _, allocation in group.allocations:
            ref = self.rows.get((id(allocation), depth, layer))
            if ref is not None and ref[0] is group:
                refs.append((allocation, ref[1]))
        indices = torch.tensor([i for _, i in refs], device=self.device, dtype=torch.long)
        group.pair = tuple(x.index_select(0, indices) for x in group.pair)
        for i, (allocation, _) in enumerate(refs):
            self.rows[id(allocation), depth, layer] = group, i
        self.compactions += 1

    def get(self, layer, batch, *, stream=None):
        context = self._context(batch, stream)
        if context is None:
            return None
        _, start, count, depth = context
        group = self.groups.get((depth, layer))
        if group is None or group.allocations is not batch.allocations:
            return None
        if group.pair[0].shape[2] != start + count:
            return None
        return group.pair

    def release(self, allocation, layers, planes):
        for depth in range(planes):
            for layer in range(layers):
                ref = self.rows.pop((id(allocation), depth, layer), None)
                if ref is not None:
                    group = ref[0]
                    group.live -= 1
                    if not group.live and self.groups.get((depth, layer)) is group:
                        del self.groups[depth, layer]

    def export(self, keys, values, block_size):
        # Materialization may be requested on a different stream by a generic
        # caller. Order after the producer and protect allocator reuse there.
        stream = None
        if self.device.type == "npu" and self.stream is not None:
            stream = torch.npu.current_stream(self.device)
            if stream.stream_id != self.stream.stream_id:
                stream.wait_stream(self.stream)
        allocations = {
            id(a): a
            for group in {ref[0] for ref in self.rows.values()}
            for _, a in group.allocations
        }
        addresses = {}
        for (identity, depth, layer), (group, row) in self.rows.items():
            length = group.pair[0].shape[2]
            address_key = identity, depth, length
            if address_key not in addresses:
                positions = torch.arange(length, device=self.device)
                table = torch.tensor(allocations[identity].block_tables[depth], device=self.device)
                addresses[address_key] = table[positions // block_size], positions % block_size
            blocks, offsets = addresses[address_key]
            for destination, source in zip((keys, values), group.pair):
                if stream is not None and stream.stream_id != self.stream.stream_id:
                    source.record_stream(stream)
                destination[blocks, layer, offsets] = source[row].transpose(0, 1)
        self.rows.clear()
        self.groups.clear()
