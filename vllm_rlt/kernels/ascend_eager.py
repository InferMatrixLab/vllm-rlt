# SPDX-License-Identifier: Apache-2.0
"""Explicit BF16 eager attention on Ascend, matching released Ouro arithmetic.

Two batching paths reduce host submissions without changing the arithmetic:
equal-length whole-prompt prefill becomes one attention call, and independent
single-row decode groups become one ragged batch. Everything else keeps the
original per-group path, including the layout-sensitive initial prefill.
"""

import math
import weakref

import torch


def _flat_kv_rows(keys, values):
    blocks, block_size, heads, head_dim = keys.shape
    block_stride, *row_strides = keys.stride()
    row_width = heads * head_dim
    if (
        not blocks
        or not row_width
        or values.shape != keys.shape
        or values.stride() != keys.stride()
        or tuple(row_strides) != (row_width, head_dim, 1)
        or block_stride < block_size * row_width
        or block_stride % row_width
    ):
        return None
    rows_per_block = block_stride // row_width
    # The view starts at this layer's storage offset. Include the gaps between
    # its pages, then gather only rows belonging to the selected layer.
    shape = ((blocks - 1) * rows_per_block + block_size, heads, head_dim)
    return (
        keys.as_strided(shape, row_strides),
        values.as_strided(shape, row_strides),
        rows_per_block,
    )


def _prefill_geometry(rows):
    """Detect a regular grid of equal-length, complete prompt traversals."""
    if not rows or rows[0][2] != 0:
        return None
    width = rows[-1][2] + 1
    size, remainder = divmod(len(rows), width)
    if size < 1 or width < 2 or remainder:
        return None
    groups = [(id(alloc), depth) for alloc, depth, _ in rows[::width]]
    if len(set(groups)) != size or any(
        ((id(alloc), depth), pos) != (groups[i // width], i % width)
        for i, (alloc, depth, pos) in enumerate(rows)
    ):
        return None
    return size, width


class AscendEagerAttention:
    """Batch compatible queries; borrow only validated host metadata."""

    def __init__(self):
        self._metadata = None
        self._groups = None
        self._geometry = None
        self._prefill_mask = None
        self._write_slots = None
        self._ragged_plan = None
        self.batched_prefill_calls = 0
        self.ragged_calls = 0
        self.fallback_calls = 0

    def _plan(self, keys, lengths, device):
        grouped = {}
        for row, (key, length) in enumerate(zip(keys, lengths, strict=True)):
            if length:
                grouped.setdefault(key, []).append(row)
        planned = []
        for rows in grouped.values():
            maximum = max(lengths[row] for row in rows)
            indices = torch.tensor(rows, dtype=torch.long, device=device)
            positions = torch.arange(maximum, device=device)
            valid = positions.unsqueeze(0) < torch.tensor(
                [lengths[row] for row in rows], device=device
            ).unsqueeze(1)
            mask = torch.zeros((len(rows), maximum), dtype=torch.bfloat16, device=device)
            mask.masked_fill_(~valid, torch.finfo(torch.bfloat16).min)
            initial = len(rows) == maximum and [lengths[row] for row in rows] == list(
                range(1, maximum + 1)
            )
            row_lengths = torch.tensor([lengths[row] for row in rows], device=device)
            planned.append((rows[0], indices, positions, mask, initial, row_lengths))
        return planned

    def _forward(self, q, keys, values, tables, groups, *, flat_rows=None, result=None):
        if result is None:
            result = torch.zeros_like(q)
        block_size, kv_heads, head_dim = keys.shape[-3:]
        head_groups = q.shape[1] // kv_heads
        for first, indices, positions, mask, initial, _lengths in groups:
            blocks = tables[first, positions // block_size]
            offsets = positions % block_size
            if flat_rows is None:
                k, v = keys[blocks, offsets], values[blocks, offsets]
            else:
                flat_keys, flat_values, rows_per_block = flat_rows
                slots = blocks.to(torch.int64) * rows_per_block + offsets
                # Padding table entries are -1. CANN gather rejects negative
                # indices, and the additive mask already zeroes these columns,
                # so clamping only changes data that cannot affect the output.
                slots = slots.clamp(0, flat_keys.shape[0] - 1)
                k, v = flat_keys.index_select(0, slots), flat_values.index_select(0, slots)
            if head_groups != 1:
                k = k.repeat_interleave(head_groups, dim=1)
                v = v.repeat_interleave(head_groups, dim=1)
            query = q.index_select(0, indices).transpose(0, 1).unsqueeze(0).contiguous()
            key = k.transpose(0, 1).unsqueeze(0)
            value = v.transpose(0, 1).unsqueeze(0)
            # HF's initial V is a projection view; subsequent cache concatenation
            # makes K/V contiguous. Preserve that operation layout on this path.
            if not initial:
                key, value = key.contiguous(), value.contiguous()
            scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
            # Keep the BF16 score/scaling and probability rounding boundaries.
            scores = scores + mask[None, None, :, :]
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            output = torch.matmul(probabilities, value).squeeze(0).transpose(0, 1)
            result.index_copy_(0, indices, output)
        return result

    def _batched_prefill(self, q, keys, values, batch, flat_rows, geometry):
        size, width = geometry
        flat_keys, flat_values, rows_per_block = flat_rows
        kv_heads, head_dim = keys.shape[-2:]
        head_groups = q.shape[1] // kv_heads
        if self._write_slots is None:
            self._write_slots = (
                batch.write_blocks.to(torch.int64) * rows_per_block
                + batch.write_offsets.to(torch.int64)
            ).clamp(0, flat_keys.shape[0] - 1)
        slots = self._write_slots
        key = (
            flat_keys.index_select(0, slots)
            .reshape(size, width, kv_heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            flat_values.index_select(0, slots)
            .reshape(size, width, kv_heads, head_dim)
            .transpose(1, 2)
        )
        if head_groups != 1:
            key = key.repeat_interleave(head_groups, dim=1)
            value = value.repeat_interleave(head_groups, dim=1)
        query = q.reshape(size, width, q.shape[1], head_dim).transpose(1, 2).contiguous()
        if self._prefill_mask is None:
            positions = torch.arange(width, device=q.device)
            mask = torch.zeros((width, width), dtype=q.dtype, device=q.device)
            mask.masked_fill_(
                positions[None, :] > positions[:, None], torch.finfo(q.dtype).min
            )
            self._prefill_mask = mask
        scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        scores = scores + self._prefill_mask[None, None, :, :]
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probabilities, value).transpose(1, 2).reshape_as(q)

    def _ragged_decode(self, q, keys, values, tables, groups, flat_rows, result):
        """One attention call for independent single-row decode groups."""
        flat_keys, flat_values, rows_per_block = flat_rows
        block_size, kv_heads, head_dim = keys.shape[-3:]
        head_groups = q.shape[1] // kv_heads
        if self._ragged_plan is None:
            indices = torch.cat([group[1] for group in groups])
            lengths = torch.cat([group[5] for group in groups])
            maximum = max(group[2].numel() for group in groups)
            positions = torch.arange(maximum, device=q.device)
            pages = positions // block_size
            offsets = positions % block_size
            row_tables = tables.index_select(0, indices)
            blocks = row_tables[:, pages].to(torch.int64)
            slots = (blocks * rows_per_block + offsets.to(torch.int64)).clamp(
                0, flat_keys.shape[0] - 1
            )
            rows = indices.numel()
            valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
            mask = torch.zeros((rows, maximum), dtype=torch.bfloat16, device=q.device)
            mask.masked_fill_(~valid, torch.finfo(torch.bfloat16).min)
            self._ragged_plan = (indices, lengths, maximum, positions, mask, slots, rows)
        indices, _lengths, maximum, _positions, mask, slots, rows = self._ragged_plan
        key = (
            flat_keys.index_select(0, slots.reshape(-1))
            .reshape(rows, maximum, kv_heads, head_dim)
        )
        value = (
            flat_values.index_select(0, slots.reshape(-1))
            .reshape(rows, maximum, kv_heads, head_dim)
        )
        if head_groups != 1:
            key = key.repeat_interleave(head_groups, dim=2)
            value = value.repeat_interleave(head_groups, dim=2)
        query = q.index_select(0, indices).reshape(rows, q.shape[1], 1, head_dim)
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()
        scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        scores = scores + mask[:, None, None, :]
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probabilities, value).squeeze(2)
        result.index_copy_(0, indices, output)
        return result

    def __call__(self, q, keys, values, tables, lengths):
        groups = self._plan([tuple(row) for row in tables.tolist()], lengths.tolist(), q.device)
        return self._forward(q, keys, values, tables, groups)

    def attend_metadata(self, q, keys, values, batch):
        flat_rows = _flat_kv_rows(keys, values)
        if self._metadata is None or self._metadata() is not batch:
            self._groups = self._plan(
                [(id(allocation), depth) for allocation, depth, _ in batch.rows],
                [position + 1 for _, _, position in batch.rows],
                q.device,
            )
            self._geometry = _prefill_geometry(batch.rows)
            self._prefill_mask = None
            self._write_slots = None
            self._ragged_plan = None
            self._metadata = weakref.ref(batch)
        if flat_rows is None:
            self.fallback_calls += 1
            return self._forward(q, keys, values, batch.block_tables, self._groups)
        if self._geometry is not None:
            self.batched_prefill_calls += 1
            return self._batched_prefill(q, keys, values, batch, flat_rows, self._geometry)
        ragged, remaining = [], []
        for group in self._groups:
            if not group[4] and group[1].numel() == 1:
                ragged.append(group)
            else:
                remaining.append(group)
        # A single row gains nothing from batching but pays extra plan construction.
        if len(ragged) < 2:
            remaining.extend(ragged)
            ragged = []
        result = self._forward(
            q, keys, values, batch.block_tables, remaining, flat_rows=flat_rows
        )
        if ragged:
            self.ragged_calls += 1
            result = self._ragged_decode(
                q, keys, values, batch.block_tables, ragged, flat_rows, result
            )
        return result

