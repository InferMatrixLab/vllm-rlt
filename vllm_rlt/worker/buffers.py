"""Reusable stage inputs and metadata; padding never acquires a KV address."""

import math
from dataclasses import dataclass, replace

import torch

from vllm_rlt.worker.metadata import KVLayout, KVMetadata
from vllm_rlt.worker.submission_queue import MAX_CONCURRENT_SUBMISSIONS, SubmissionQueue
from vllm_rlt.worker.tensors import UvaBackedTensor, copy_tensor, use_uva_on


def prefill_stage_rows(config, scheduler, execution) -> int:
    """Hidden rows per request that staged prompt prefill keeps resident.

    Split prefill stages one chunk at a time, so a chunk is the unit that must
    survive between two scheduling boundaries. Wavefront chunked prefill advances
    several chunks of one prompt at different depths concurrently; it then keeps a
    chunk per recurrence depth in flight, and this becomes
    ``scheduler.prefill_chunk_size * config.total_ut_steps``.
    """
    if execution.atomic_prefill:
        return 0
    return scheduler.prefill_chunk_size


def execution_buffer_bytes(config, scheduler, cache, execution, element_size):
    rows = scheduler.max_num_batched_tokens
    if execution.pad_to_power_of_two:
        rows = 1 << (rows - 1).bit_length()
    width = math.ceil(config.max_position_embeddings / cache.block_size)
    hidden = config.hidden_size * element_size
    total = 0
    if execution.static_buffers:
        # Existing prefill/eager workspaces remain separate from async buffer sets.
        total += 4 * rows * (hidden + 4 * width + 36) + scheduler.max_num_seqs * hidden
    if execution.async_scheduling:
        planes = config.total_ut_steps if cache.layout == "last_exited" else 1
        # Four routing buffer sets, including capacity for an H2D descriptor fallback.
        total += 4 * rows * (hidden + 4 * width + 68)
        total += scheduler.max_num_seqs * (hidden + 8 + planes * width * 4)
        total += 2 * scheduler.max_num_seqs * 24  # fallback exit descriptors
        # Typed sparse-write descriptors for routing and finalization buffer sets.
        total += (
            (4 * min(rows, scheduler.max_num_seqs) + 2 * scheduler.max_num_seqs) * planes * 8 * 8
        )
    # Split prefill holds the staged chunk's hidden rows across batch boundaries.
    total += scheduler.max_num_seqs * prefill_stage_rows(config, scheduler, execution) * hidden
    # Split recurrent batches gather staged rows into a contiguous model input.
    if not execution.atomic_prefill:
        total += rows * hidden
        total += MAX_CONCURRENT_SUBMISSIONS * (
            scheduler.max_num_seqs * (5 + 4 * 8) * 8 + max(rows, scheduler.prefill_chunk_size) * 8
        )
    return total


class Workspace:
    def __init__(self, rows, width, hidden_size, device, dtype, *, use_uva=True):
        self.hidden = torch.empty((rows, hidden_size), device=device, dtype=dtype)
        self.device = device
        self.use_uva = use_uva
        self.submissions = SubmissionQueue()
        specs = {
            "tokens": ((rows,), torch.int64),
            "positions": ((rows,), torch.int64),
            "blocks": ((rows,), torch.int64),
            "offsets": ((rows,), torch.int64),
            "tables": ((rows, width), torch.int32),
            "lengths": ((rows,), torch.int32),
        }
        self.tensors = {}
        self.gpu = {}
        for name, (shape, field_dtype) in specs.items():
            self.tensors[name] = UvaBackedTensor(shape, field_dtype, device)
            self.gpu[name] = torch.empty(shape, dtype=field_dtype, device=device)

    def acquire(self):
        # Protect pinned snapshots until the previous workspace consumer retires.
        self.submissions.begin()

    def record_done(self):
        """Record the last GPU consumer without waiting or freeing storage."""
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.submissions.complete(event)

    def store_metadata(self, layout: KVLayout, size):
        """Fill fixed host buffers from a storage-independent layout."""
        n = len(layout.rows)
        attention = layout.attention
        if attention.cu_seqlens_q is not None:
            raise ValueError("workspace metadata requires single-query rows")
        if not n <= size <= len(self.tensors["positions"].cpu):
            raise ValueError("metadata size must cover query rows and fit the workspace")
        for name in ("positions", "lengths", "tables"):
            self.tensors[name].cpu[:size].zero_()
        for name, values in (
            ("positions", layout.positions),
            ("lengths", attention.context_lengths),
            ("blocks", layout.write_blocks),
            ("offsets", layout.write_offsets),
        ):
            self.tensors[name].np[:n] = values
        self.tensors["tables"].np[:n, : attention.table_width] = (
            torch.tensor(attention.block_tables, dtype=torch.int32)
            .view(n, attention.table_width)
            .numpy()
        )
        return KVMetadata(
            owner=layout.owner,
            rows=layout.rows,
            allocations=layout.allocations,
            position_ids=self.tensors["positions"].cpu[:size],
            write_blocks=self.tensors["blocks"].cpu[:n],
            write_offsets=self.tensors["offsets"].cpu[:n],
            block_tables=self.tensors["tables"].cpu[:size],
            context_lengths=self.tensors["lengths"].cpu[:size],
            writable=layout.writable,
        )

    def upload_metadata(self, metadata):
        """Upload prepared host inputs into the workspace's fixed device addresses."""
        n, size = len(metadata.rows), len(metadata.position_ids)
        for name in ("positions", "lengths", "tables", "blocks", "offsets"):
            count = n if name in ("blocks", "offsets") else size
            snapshot = self.tensors[name].publish(count)
            copy_tensor(snapshot, self.gpu[name][:count], use_uva=self.use_uva)
        return replace(
            metadata,
            position_ids=self.gpu["positions"][:size],
            write_blocks=self.gpu["blocks"][:n],
            write_offsets=self.gpu["offsets"][:n],
            block_tables=self.gpu["tables"][:size],
            context_lengths=self.gpu["lengths"][:size],
        )

    def tokens(self, ids, size):
        host = self.tensors["tokens"].cpu[:size]
        host.zero_()
        host[: len(ids)] = torch.tensor(ids, dtype=host.dtype)
        snapshot = self.tensors["tokens"].publish(size)
        copy_tensor(snapshot, self.gpu["tokens"][:size], use_uva=self.use_uva)
        return self.gpu["tokens"][:size]


def stage_metadata(layouts, stage, *, padded_rows=None):
    """Store computed layouts in two typed buffers using caller-owned staging."""
    first = layouts[0]
    n = len(first.rows)
    size = n if padded_rows is None else padded_rows
    if size < n or (size != n and first.attention.cu_seqlens_q is not None):
        raise ValueError("padding requires non-packed metadata and at least the real row count")
    wide = first.positions + [0] * (size - n) + first.write_offsets
    narrow, shared, plans = [], {}, []
    for layout in layouts:
        attention = layout.attention
        block_start = len(wide)
        wide += layout.write_blocks
        table_start = len(narrow)
        narrow += attention.block_tables + [0] * ((size - n) * attention.table_width)
        lengths = attention.context_lengths + [0] * (size - n)
        cumulative = attention.cu_seqlens_q
        key = (tuple(lengths), None if cumulative is None else tuple(cumulative))
        if key not in shared:
            shared[key] = len(narrow)
            narrow += lengths + (cumulative or [])
        plans.append((layout, block_start, table_start, shared[key]))
    wide, narrow = stage(wide, torch.int64), stage(narrow, torch.int32)
    batches = []
    for layout, block_start, table_start, lengths_start in plans:
        attention = layout.attention
        t, width = attention.num_rows + size - n, attention.table_width
        batches.append(
            KVMetadata(
                owner=layout.owner,
                rows=layout.rows,
                allocations=layout.allocations,
                position_ids=wide[:size],
                write_blocks=wide[block_start : block_start + n],
                write_offsets=wide[size : size + n],
                block_tables=narrow[table_start : table_start + t * width].reshape(t, width),
                context_lengths=narrow[lengths_start : lengths_start + t],
                writable=layout.writable,
                cu_seqlens_q=(
                    narrow[lengths_start + t : lengths_start + 2 * t + 1]
                    if attention.cu_seqlens_q is not None
                    else None
                ),
                max_seqlen_q=attention.max_seqlen_q,
            )
        )
    return tuple(batches)


@dataclass(frozen=True)
class PackedLayout:
    """Offsets and shapes for one depth view in the compact transfer buffer.

    CPU preparation records these offsets; GPU preparation uses them to slice
    the uploaded storage into KVMetadata tensors without rebuilding the layout.
    """

    rows: tuple
    writable: bool
    sequence_start: int
    lengths_start: int
    cumulative_start: int
    table_start: int
    num_sequences: int
    table_width: int
    max_query: int


class PackedMetadataBuffer:
    """Compact transfer storage owned by one runner submission slot."""

    def __init__(self, device, *, use_uva=True):
        self.device = device
        self.use_uva = use_uva_on(device, use_uva)
        self.capacity = self.row_capacity = self.view_capacity = 0
        self.targets: dict[torch.dtype, torch.Tensor | None] = {
            dtype: None for dtype in (torch.int64, torch.int32)
        }

    def store(self, metadata, tokens=()):
        """Serialize already computed layouts; do not resolve cache addresses."""
        first = metadata[0]
        n = len(first.rows)
        if not n or len(tokens) not in (0, n):
            raise ValueError("compact inputs require nonempty rows and matching token counts")
        values = list(tokens) + first.positions
        layouts, shared = [], {}
        for item in metadata:
            plan = item.attention
            if plan.cu_seqlens_q is None:
                raise ValueError("compact metadata currently requires packed query layouts")
            key = (tuple(plan.context_lengths), tuple(plan.cu_seqlens_q))
            if key not in shared:
                sequence_ids = [
                    sequence
                    for sequence, (start, end) in enumerate(
                        zip(plan.cu_seqlens_q, plan.cu_seqlens_q[1:])
                    )
                    for _ in range(start, end)
                ]
                sequence_start = len(values)
                values += sequence_ids
                lengths_start = len(values)
                values += plan.context_lengths
                cumulative_start = len(values)
                values += plan.cu_seqlens_q
                shared[key] = (sequence_start, lengths_start, cumulative_start)
            table_start = len(values)
            values += plan.block_tables
            sequence_start, lengths_start, cumulative_start = shared[key]
            layouts.append(
                PackedLayout(
                    rows=item.rows,
                    writable=item.writable,
                    sequence_start=sequence_start,
                    lengths_start=lengths_start,
                    cumulative_start=cumulative_start,
                    table_start=table_start,
                    num_sequences=plan.num_rows,
                    table_width=plan.table_width,
                    max_query=plan.max_seqlen_q,
                )
            )
        self.layouts = tuple(layouts)
        self.owner, self.allocations = first.owner, first.allocations
        self.token_count, self.num_rows = len(tokens), n
        self.value_count = len(values)
        if len(values) > self.capacity:
            self.capacity = 1 << (len(values) - 1).bit_length()
            self.storage = UvaBackedTensor(
                self.capacity, torch.int64, self.device, max_concurrency=1
            )
        self.storage.np[: len(values)] = values

    def upload(self):
        from vllm_rlt.kernels.metadata import expand_kv_addresses

        for dtype, target in self.targets.items():
            if target is None or len(target) < self.capacity:
                self.targets[dtype] = torch.empty(self.capacity, dtype=dtype, device=self.device)
        count, n = self.value_count, self.num_rows
        snapshot = self.storage.publish(count)
        copy_tensor(
            snapshot,
            self.targets[torch.int64][:count],
            secondary=self.targets[torch.int32][:count],
            use_uva=self.use_uva,
        )
        rows, views = n, len(self.layouts)
        if rows > self.row_capacity or views > self.view_capacity:
            self.row_capacity = max(self.row_capacity, 1 << (rows - 1).bit_length())
            self.view_capacity = max(self.view_capacity, 1 << (views - 1).bit_length())
            self.blocks = torch.empty(
                (self.view_capacity, self.row_capacity),
                dtype=torch.int64,
                device=self.device,
            )
            self.offsets = torch.empty_like(self.blocks)
        positions = self.targets[torch.int64][self.token_count : self.token_count + n]
        metadata = []
        for index, layout in enumerate(self.layouts):
            q, width = layout.num_sequences, layout.table_width
            sequences = self.targets[torch.int32][layout.sequence_start : layout.sequence_start + n]
            tables = self.targets[torch.int32][
                layout.table_start : layout.table_start + q * width
            ].view(q, width)
            blocks, offsets = self.blocks[index, :n], self.offsets[index, :n]
            expand_kv_addresses[(math.ceil(n / 256),)](
                positions,
                sequences,
                tables,
                blocks,
                offsets,
                n,
                width,
                self.owner.block_size,
                256,
            )
            metadata.append(
                KVMetadata(
                    owner=self.owner,
                    rows=layout.rows,
                    allocations=self.allocations,
                    position_ids=positions,
                    write_blocks=blocks,
                    write_offsets=offsets,
                    block_tables=tables,
                    context_lengths=self.targets[torch.int32][
                        layout.lengths_start : layout.lengths_start + q
                    ],
                    writable=layout.writable,
                    cu_seqlens_q=self.targets[torch.int32][
                        layout.cumulative_start : layout.cumulative_start + q + 1
                    ],
                    max_seqlen_q=layout.max_query,
                )
            )
        return self.targets[torch.int64][: self.token_count], tuple(metadata)
