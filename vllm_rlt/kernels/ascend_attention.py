# SPDX-License-Identifier: Apache-2.0
"""Use Torch SDPA on Ascend with the runtime's paged KV cache."""

import weakref
from collections import OrderedDict

import torch
from torch.nn import functional as F


def _attend(query, key, value, mask=None, *, is_causal=False):
    # Match HF's BNSD inputs. A bool mask lets Torch-NPU dispatch fused SDPA
    # for cached chunks and variable-length decode on supported hardware.
    groups = query.shape[1] // key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=mask,
        is_causal=is_causal,
    )


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


class AscendAttention:
    """Batch compatible queries; borrow only validated host metadata."""

    def __init__(self):
        self._metadata = None
        self._groups = None
        self._geometry = None
        self._write_slots = None
        self._ragged_plan = None
        self._signature = None
        self._direct_decode = False
        self._decode_plan = None
        # Read-only layouts may be shared across recurrent depths. Addresses
        # stay local to each descriptor, and tensors never cross stream keys.
        self._layouts = OrderedDict()
        self._layout_bytes = 0
        self._layout_limit = 64 * 1024 * 1024
        self.layout_reuses = 0
        self.batched_prefill_calls = 0
        self.ragged_calls = 0
        self.fallback_calls = 0
        self.continuation_calls = 0
        self._range_metadata = None
        self._range_signature = None
        self._range_mask = self._range_slots = None

    def _prepare_ranges(self, q, batch):
        ranges = batch.sequence_ranges
        rows = len(ranges)
        width, start = ranges[0].count, ranges[0].start
        signature = (q.device, q.dtype, q.shape, self._stream_key(q.device))
        if (
            self._range_metadata is None
            or self._range_metadata() is not batch
            or self._range_signature != signature
        ):
            self._range_metadata = self._range_signature = None
            self._range_slots = self._range_mask = None

            def build():
                positions = torch.arange(start + width, device=q.device)
                return (positions[None, :] <= positions[start:, None],)

            if start and width > 1:
                self._range_mask = self._layout(
                    ("range", q.device, signature[-1], start, width), build
                )[0]
            self._range_metadata = weakref.ref(batch)
            self._range_signature = signature
        return rows, width, start

    def _range_attention(self, q, pair, geometry):
        rows, width, start = geometry
        key, value = pair
        head_dim = q.shape[-1]
        query = (
            q.reshape(rows, q.shape[1], 1, head_dim)
            if width == 1
            else q.reshape(rows, width, q.shape[1], head_dim).transpose(1, 2)
        )
        output = _attend(query, key, value, self._range_mask, is_causal=start == 0 and width > 1)
        if width == 1:
            self.ragged_calls += 1
            return output.squeeze(2)
        self.batched_prefill_calls += 1
        if start:
            self.continuation_calls += 1
        return output.transpose(1, 2).reshape_as(q)

    def _attend_ranges(self, q, keys, values, batch):
        geometry = self._prepare_ranges(q, batch)
        rows, width, start = geometry
        flat_rows = _flat_kv_rows(keys, values)
        if flat_rows is None:
            raise ValueError("compact Ascend ranges require regular paged KV storage")
        flat_keys, flat_values, rows_per_block = flat_rows
        if self._range_slots is None:
            positions = torch.arange(start + width, device=q.device)
            pages = positions // keys.shape[-3]
            blocks = batch.block_tables[:, pages].to(torch.int64)
            self._range_slots = (blocks * rows_per_block + positions % keys.shape[-3]).reshape(-1)
        shape = (rows, start + width, keys.shape[-2], keys.shape[-1])
        key = flat_keys.index_select(0, self._range_slots).reshape(shape).transpose(1, 2)
        value = flat_values.index_select(0, self._range_slots).reshape(shape).transpose(1, 2)
        return self._range_attention(q, (key, value), geometry)

    @staticmethod
    def _stream_key(device):
        if device.type == "npu":
            # Raw npu_stream resolves an ACL handle and drains the task queue.
            # Cache identity only needs the logical, device-scoped stream ID.
            return torch.npu.current_stream(device).stream_id
        if device.type == "cuda":
            return torch.cuda.current_stream(device).cuda_stream
        return None

    def _layout(self, key, build):
        cached = self._layouts.get(key)
        if cached is not None:
            self._layouts.move_to_end(key)
            self.layout_reuses += 1
            return cached[0]
        tensors = build()
        size = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        if size <= self._layout_limit:
            while self._layouts and self._layout_bytes + size > self._layout_limit:
                _, (_, old_size) = self._layouts.popitem(last=False)
                self._layout_bytes -= old_size
            # Bound Python entries too: many tiny short-context plans are common.
            if len(self._layouts) == 8:
                _, (_, old_size) = self._layouts.popitem(last=False)
                self._layout_bytes -= old_size
            self._layouts[key] = (tensors, size)
            self._layout_bytes += size
        return tensors

    def _prepare_decode(self, q, keys, batch, flat_rows, stream):
        lengths = tuple(position + 1 for _, _, position in batch.rows)
        maximum = max(lengths)
        block_size = keys.shape[-3]

        def build():
            positions = torch.arange(maximum, device=q.device)
            valid = None
            if min(lengths) != maximum:
                valid = positions[None, :] < torch.tensor(lengths, device=q.device)[:, None]
            return positions // block_size, positions % block_size, valid

        pages, offsets, valid = self._layout(
            ("decode", q.device, stream, block_size, lengths), build
        )
        rows = len(lengths)
        tables = batch.block_tables
        if len(tables) != rows:
            tables = tables[:rows]
        blocks = tables[:, pages].to(torch.int64)
        flat_keys, _, rows_per_block = flat_rows
        slots = (blocks * rows_per_block + offsets).clamp(0, flat_keys.shape[0] - 1)
        if valid is not None:
            # Avoid reading unwritten bytes in masked columns. Position zero is
            # initialized whenever the cache's prefix check admits this row.
            slots = torch.where(valid, slots, slots[:, :1])
        mask = valid[:, None, None, :] if valid is not None else None
        self._decode_plan = (slots.reshape(-1), mask, rows, maximum)

    def _decode(self, q, keys, values, flat_rows):
        slots, mask, rows, maximum = self._decode_plan
        flat_keys, flat_values, _ = flat_rows
        kv_heads, head_dim = keys.shape[-2:]
        key = flat_keys.index_select(0, slots).reshape(rows, maximum, kv_heads, head_dim)
        value = flat_values.index_select(0, slots).reshape(rows, maximum, kv_heads, head_dim)
        # Real rows form an identity prefix. Avoid gathering Q and scattering
        # the result on every layer.
        query = q if len(q) == rows else q[:rows]
        query = query.reshape(rows, q.shape[1], 1, head_dim)
        output = _attend(query, key.transpose(1, 2), value.transpose(1, 2), mask).squeeze(2)
        if len(q) == rows:
            return output
        result = torch.zeros_like(q)
        result[:rows].copy_(output)
        return result

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
            group_lengths = [lengths[row] for row in rows]
            initial = group_lengths == list(range(1, maximum + 1))
            row_lengths = torch.tensor(group_lengths, device=device)
            mask = (
                positions[None, :] < row_lengths[:, None] if not initial and len(rows) > 1 else None
            )
            planned.append((rows[0], indices, positions, mask, initial, row_lengths))
        return planned

    def _forward(self, q, keys, values, tables, groups, *, flat_rows=None, result=None):
        if result is None:
            result = torch.zeros_like(q)
        block_size = keys.shape[-3]
        for first, indices, positions, mask, initial, _lengths in groups:
            blocks = tables[first, positions // block_size]
            offsets = positions % block_size
            if flat_rows is None:
                k, v = keys[blocks, offsets], values[blocks, offsets]
            else:
                flat_keys, flat_values, rows_per_block = flat_rows
                slots = blocks.to(torch.int64) * rows_per_block + offsets
                slots = slots.clamp(0, flat_keys.shape[0] - 1)
                k, v = flat_keys.index_select(0, slots), flat_values.index_select(0, slots)
            query = q.index_select(0, indices).transpose(0, 1).unsqueeze(0)
            key = k.transpose(0, 1).unsqueeze(0)
            value = v.transpose(0, 1).unsqueeze(0)
            output = (
                _attend(query, key, value, mask, is_causal=initial and indices.numel() > 1)
                .squeeze(0)
                .transpose(0, 1)
            )
            result.index_copy_(0, indices, output)
        return result

    def _batched_prefill(self, q, keys, values, batch, flat_rows, geometry):
        size, width = geometry
        flat_keys, flat_values, rows_per_block = flat_rows
        kv_heads, head_dim = keys.shape[-2:]
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
        query = q.reshape(size, width, q.shape[1], head_dim).transpose(1, 2)
        return _attend(query, key, value, is_causal=True).transpose(1, 2).reshape_as(q)

    def _ragged_decode(self, q, keys, values, tables, groups, flat_rows, result):
        """One attention call for independent single-row decode groups."""
        flat_keys, flat_values, rows_per_block = flat_rows
        block_size, kv_heads, head_dim = keys.shape[-3:]
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
            slots = torch.where(valid, slots, slots[:, :1])
            self._ragged_plan = (indices, maximum, valid[:, None, None, :], slots, rows)
        indices, maximum, mask, slots, rows = self._ragged_plan
        key = flat_keys.index_select(0, slots.reshape(-1)).reshape(
            rows, maximum, kv_heads, head_dim
        )
        value = flat_values.index_select(0, slots.reshape(-1)).reshape(
            rows, maximum, kv_heads, head_dim
        )
        query = q.index_select(0, indices).reshape(rows, q.shape[1], 1, head_dim)
        output = _attend(query, key.transpose(1, 2), value.transpose(1, 2), mask).squeeze(2)
        result.index_copy_(0, indices, output)
        return result

    def __call__(self, q, keys, values, tables, lengths):
        groups = self._plan([tuple(row) for row in tables.tolist()], lengths.tolist(), q.device)
        return self._forward(q, keys, values, tables, groups)

    def attend_metadata(self, q, keys, values, batch):
        if batch.sequence_ranges is not None:
            return self._attend_ranges(q, keys, values, batch)
        flat_rows = _flat_kv_rows(keys, values)
        stream = self._stream_key(q.device)
        signature = (q.device, q.dtype, q.shape, keys.shape, keys.stride(), values.stride(), stream)
        if self._metadata is None or self._metadata() is not batch or self._signature != signature:
            # Failed preparation must not leave the previous descriptor key
            # pointing at partially replaced groups/addresses.
            self._metadata = None
            self._signature = None
            self._geometry = _prefill_geometry(batch.rows)
            self._write_slots = None
            self._ragged_plan = None
            self._decode_plan = None
            self._direct_decode = (
                flat_rows is not None
                and len(batch.rows) >= 2
                and all(position > 0 for _, _, position in batch.rows)
                and len({(id(a), depth) for a, depth, _ in batch.rows}) == len(batch.rows)
            )
            self._groups = None
            if self._direct_decode:
                self._prepare_decode(q, keys, batch, flat_rows, stream)
            elif flat_rows is None or self._geometry is None:
                self._groups = self._plan(
                    [(id(allocation), depth) for allocation, depth, _ in batch.rows],
                    [position + 1 for _, _, position in batch.rows],
                    q.device,
                )
            self._metadata = weakref.ref(batch)
            self._signature = signature
        if self._direct_decode:
            self.ragged_calls += 1
            return self._decode(q, keys, values, flat_rows)
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
        result = self._forward(q, keys, values, batch.block_tables, remaining, flat_rows=flat_rows)
        if ragged:
            self.ragged_calls += 1
            result = self._ragged_decode(
                q, keys, values, batch.block_tables, ragged, flat_rows, result
            )
        return result
