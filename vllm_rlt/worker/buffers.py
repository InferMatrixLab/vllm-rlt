"""Reusable stage inputs and metadata; padding never acquires a KV address."""

import math

import torch

from vllm_rlt.core.kv_cache_manager import _PreparedKVBatch
from vllm_rlt.worker.loopcd import reserved_bytes


def execution_buffer_bytes(config, scheduler, cache, execution, element_size):
    rows = scheduler.max_num_batched_tokens
    if execution.pad_to_power_of_two:
        rows = 1 << (rows - 1).bit_length()
    width = math.ceil(config.max_position_embeddings / cache.block_size)
    hidden = config.hidden_size * element_size
    total = reserved_bytes(config, scheduler, execution, element_size)
    if execution.static_buffers:
        # Existing prefill/eager workspaces remain separate from async banks.
        total += 4 * rows * (hidden + 4 * width + 36) + scheduler.max_num_seqs * hidden
    if execution.async_scheduling:
        planes = config.total_ut_steps if cache.layout == "last_exited" else 1
        # Four routing banks, including capacity for an H2D descriptor fallback.
        total += 4 * rows * (hidden + 4 * width + 68)
        total += scheduler.max_num_seqs * (hidden + 8 + planes * width * 4)
        total += 2 * scheduler.max_num_seqs * 24  # fallback exit descriptors
    return total


class Workspace:
    def __init__(self, rows, width, hidden_size, device, dtype):
        self.hidden = torch.empty((rows, hidden_size), device=device, dtype=dtype)
        self.device = device
        self.event = None
        shapes = dict(
            tokens=(rows,),
            positions=(rows,),
            blocks=(rows,),
            offsets=(rows,),
            tables=(rows, width),
            lengths=(rows,),
        )
        self.host = {}
        self.gpu = {}
        for name, shape in shapes.items():
            kind = torch.int32 if name in ("tables", "lengths") else torch.int64
            self.host[name] = torch.empty(shape, dtype=kind, pin_memory=device.type == "cuda")
            self.gpu[name] = torch.empty(shape, dtype=kind, device=device)

    def acquire(self):
        if self.event is not None:
            self.event.synchronize()  # Protect host DMA inputs as well as device scratch.

    def release(self):
        if self.device.type == "cuda":
            self.event = torch.cuda.Event()
            self.event.record(torch.cuda.current_stream(self.device))

    def tokens(self, ids, size):
        host = self.host["tokens"][:size]
        host.zero_()
        host[: len(ids)] = torch.tensor(ids, dtype=host.dtype)
        self.gpu["tokens"][:size].copy_(host, non_blocking=True)
        return self.gpu["tokens"][:size]

    def prepare(self, cache, ids, depths, positions, size):
        rows = tuple(cache._validate_rows(ids, depths, positions))
        addresses = [
            (a.block_tables[cache._plane(d)][p // cache.block_size], p % cache.block_size)
            for a, d, p in rows
        ]
        if len(set(addresses)) != len(addresses):
            raise ValueError("duplicate KV write addresses")
        n = len(rows)
        for name in ("positions", "lengths", "tables"):
            self.host[name][:size].zero_()
        for index, ((allocation, depth, pos), (block, offset)) in enumerate(zip(rows, addresses)):
            self.host["positions"][index] = pos
            self.host["lengths"][index] = pos + 1
            self.host["blocks"][index] = block
            self.host["offsets"][index] = offset
            table = allocation.block_tables[cache._plane(depth)]
            self.host["tables"][index, : len(table)] = torch.tensor(table, dtype=torch.int32)
        for name in ("positions", "lengths", "tables", "blocks", "offsets"):
            count = n if name in ("blocks", "offsets") else size
            self.gpu[name][:count].copy_(self.host[name][:count], non_blocking=True)
        return _PreparedKVBatch(
            owner=cache,
            rows=rows,
            allocations=tuple(dict(zip(ids, (a for a, _, _ in rows))).items()),
            position_ids=self.gpu["positions"][:size],
            write_blocks=self.gpu["blocks"][:n],
            write_offsets=self.gpu["offsets"][:n],
            block_tables=self.gpu["tables"][:size],
            context_lengths=self.gpu["lengths"][:size],
            writable=True,
        )
