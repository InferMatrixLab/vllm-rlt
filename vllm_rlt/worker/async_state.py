"""Stable request slots and event-owned routing buffers for asynchronous CUDA work.

Hot state and page tables remain on device. Small CPU routing descriptions live
in mapped pinned memory (UVA); the H2D fallback uses the same buffer ownership.
No buffer set or request slot may be reused until its last GPU reader completes.
"""

import math
from dataclasses import replace

import torch

from vllm_rlt.worker.metadata import KVMetadata
from vllm_rlt.worker.submission_queue import SubmissionQueue
from vllm_rlt.worker.tensors import StagedWriteTensor, UvaBuffer, use_uva_on


class RoutingBuffers:
    def __init__(self, owner, rows, *, control_only=False):
        self.owner = owner
        self.submissions = owner.finalize_submissions if control_only else owner.submissions
        self.routing = UvaBuffer((rows, 3), torch.int64, owner.cache.device, use_uva=owner.use_uva)
        self.host = self.routing.cpu
        self.done = self.ready_event = None
        self.table_updates = StagedWriteTensor(
            owner.tables,
            min(rows, owner.hidden.shape[0]) * owner.planes,
            owner.width,
            use_uva=owner.use_uva,
        )
        self.imports = []
        self.descriptor = None
        if control_only:
            return
        device = owner.cache.device
        self.slots = torch.empty(rows, dtype=torch.int64, device=device)
        self.positions = torch.empty_like(self.slots)
        self.blocks = torch.empty_like(self.slots)
        self.offsets = torch.empty_like(self.slots)
        self.lengths = torch.empty(rows, dtype=torch.int32, device=device)
        self.tables = torch.empty((rows, owner.width), dtype=torch.int32, device=device)
        self.hidden = torch.empty(
            (rows, owner.hidden.shape[1]), dtype=owner.hidden.dtype, device=device
        )
        self.tokens = torch.empty((rows, 1), dtype=torch.int64, device=device)
        self.descriptor = None

    def acquire(self):
        self.done = None
        self.table_updates.reset()
        self.imports.clear()
        self.descriptor = None

    def transfer(self, batch=None):
        self.table_updates.submit()
        for pool, slot, value in self.imports:
            value.record_stream(torch.cuda.current_stream(self.owner.cache.device))
            pool[slot].copy_(value)
        self.descriptor = self.routing.stage(self.count)
        result = self.materialize_metadata(batch)
        self.ready_event = torch.cuda.Event()
        self.ready_event.record(torch.cuda.current_stream(self.owner.cache.device))
        return result

    def materialize_metadata(self, metadata):
        from vllm_rlt.kernels.routing import metadata_kernel, slots_kernel

        if metadata is None:
            slots_kernel[(math.ceil(self.size / 256),)](
                self.descriptor, self.slots, self.count, self.size, 256
            )
            return None
        metadata_kernel[(self.size,)](
            self.descriptor,
            self.owner.tables,
            self.slots,
            self.positions,
            self.lengths,
            self.blocks,
            self.offsets,
            self.tables,
            self.count,
            self.size,
            self.owner.width,
            self.owner.cache.block_size,
            self.owner.planes,
            self.width,
            256,
        )
        return replace(
            metadata,
            position_ids=self.positions[: self.size],
            context_lengths=self.lengths[: self.size],
            write_blocks=self.blocks[: self.count],
            write_offsets=self.offsets[: self.count],
            block_tables=self.tables[: self.size, : self.width],
        )

    def record_done(self):
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(self.owner.cache.device))
        self.submissions.complete(self.done)

    def gather(self, tokens=False):
        from vllm_rlt.kernels.routing import gather_kernel

        pool = self.owner.tokens if tokens else self.owner.hidden
        output = self.tokens if tokens else self.hidden
        h = pool.shape[1]
        gather_kernel[(self.size, math.ceil(h / 256))](
            pool, self.slots, output, self.count, h, pool.stride(0), output.stride(0), 256
        )
        return output[: self.size, 0] if tokens else output[: self.size]

    def scatter(self, values, tokens=False):
        from vllm_rlt.kernels.routing import scatter_kernel

        pool = self.owner.tokens if tokens else self.owner.hidden
        values = values.reshape(-1, 1) if tokens else values
        h = pool.shape[1]
        scatter_kernel[(self.count, math.ceil(h / 256))](
            values, self.slots, pool, h, values.stride(0), pool.stride(0), 256
        )


class AsyncState:
    def __init__(self, cache, config, scheduler, rows, *, use_uva=True):
        self.cache = cache
        self.metadata = cache.metadata_builder
        self.use_uva = use_uva_on(cache.device, use_uva)
        self.width = math.ceil(config.max_position_embeddings / cache.block_size)
        self.planes = config.total_ut_steps if cache.layout == "last_exited" else 1
        self.hidden = torch.empty(
            (scheduler.max_num_seqs, config.hidden_size),
            dtype=cache.key_cache.dtype,
            device=cache.device,
        )
        self.tokens = torch.empty(
            (scheduler.max_num_seqs, 1), dtype=torch.int64, device=cache.device
        )
        self.tables = torch.zeros(
            (scheduler.max_num_seqs, self.planes, self.width),
            dtype=torch.int32,
            device=cache.device,
        )
        self.slots = {}
        self.table_versions = {}
        self.owners = {}
        self.free = list(reversed(range(scheduler.max_num_seqs)))
        self.submissions = SubmissionQueue(max_pending=4)
        self.finalize_submissions = SubmissionQueue(max_pending=2)
        self.routing_buffers = [RoutingBuffers(self, rows) for _ in range(4)]
        self.finalize_buffers = [
            RoutingBuffers(self, scheduler.max_num_seqs, control_only=True) for _ in range(2)
        ]
        self.index = self.finalize_index = 0

    def reserve_slot(self, request):
        rid = request.request_id
        allocation = self.cache._get_allocation(rid)
        if rid in self.slots:
            if self.owners[rid] != (id(request), id(allocation)):
                raise RuntimeError("request slot reused before retiring its previous owner")
            return self.slots[rid]
        if not self.free:
            raise RuntimeError("no retired request state slots")
        slot = self.free.pop()
        self.slots[rid] = slot
        self.owners[rid] = (id(request), id(allocation))
        return slot

    def ensure_slot(self, request, routing_buffers):
        rid = request.request_id
        allocation = self.cache._get_allocation(rid)
        was_reserved = rid in self.slots
        slot = self.reserve_slot(request)
        previous = self.table_versions.get(rid)
        # Page entries per logical plane: physical blocks, or flattened pages when packed.
        current = self.cache.entry_tables(allocation)
        if previous != current:
            for plane, pages in enumerate(current):
                old = previous[plane] if previous is not None else ()
                start = 0
                while start < min(len(old), len(pages)) and old[start] == pages[start]:
                    start += 1
                end = len(pages)
                while end > start and end <= len(old) and old[end - 1] == pages[end - 1]:
                    end -= 1
                if start < end:
                    routing_buffers.table_updates.stage_write(
                        (slot, plane), start, pages[start:end]
                    )
            self.table_versions[rid] = current
        if was_reserved:
            return slot
        if request.hidden_state is not None:
            routing_buffers.imports.append((self.hidden, slot, request.hidden_state))
            request.hidden_state = self.hidden[slot]
        # A resumed PRELUDE reads its input from the slot, not the request.
        if request.input_token_tensor is not None:
            routing_buffers.imports.append((self.tokens, slot, request.input_token_tensor))
        return slot

    def prepare(self, requests, depths, positions, size, *, recurrent=False, finalize=False):
        buffer_pool = self.finalize_buffers if finalize else self.routing_buffers
        submissions = self.finalize_submissions if finalize else self.submissions
        submissions.begin()
        index = self.finalize_index if finalize else self.index
        routing_buffers = buffer_pool[index]
        if finalize:
            self.finalize_index = (index + 1) % len(buffer_pool)
        else:
            self.index = (index + 1) % len(buffer_pool)
        routing_buffers.acquire()
        routing_buffers.count, routing_buffers.size = len(requests), size
        routing_buffers.width = max(p // self.cache.block_size + 1 for p in positions)
        descriptors = [
            (self.ensure_slot(r, routing_buffers), d if recurrent or finalize else 0, p)
            for r, d, p in zip(requests, depths, positions)
        ]
        # A normal list is CPU-owned; only the contiguous pinned snapshot is GPU-readable.
        routing_buffers.host.numpy()[: len(descriptors)] = descriptors
        batch = None
        if recurrent:
            ids = [r.request_id for r in requests]
            rows = self.metadata.resolve_rows(ids, depths, positions)
            batch = KVMetadata(
                owner=self.cache,
                rows=rows,
                allocations=tuple(dict(zip(ids, (a for a, _, _ in rows))).items()),
                position_ids=routing_buffers.host[: len(rows), 2],
                write_blocks=routing_buffers.blocks[: len(rows)],
                write_offsets=routing_buffers.offsets[: len(rows)],
                block_tables=routing_buffers.tables[:size, : routing_buffers.width],
                context_lengths=routing_buffers.lengths[:size],
                writable=True,
            )
        return routing_buffers, batch

    def release(self, rid):
        self.table_versions.pop(rid, None)
        self.owners.pop(rid, None)
        slot = self.slots.pop(rid, None)
        if slot is not None:
            self.free.append(slot)
