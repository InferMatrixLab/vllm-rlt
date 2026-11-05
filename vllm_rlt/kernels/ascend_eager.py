# SPDX-License-Identifier: Apache-2.0
"""Explicit BF16 eager attention on Ascend, matching released Ouro arithmetic."""

import math
import weakref

import torch


class AscendEagerAttention:
    """Batch each request/depth's queries; borrow only validated host metadata."""

    def __init__(self):
        self._metadata = None
        self._groups = None

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
            planned.append((rows[0], indices, positions, mask, initial))
        return planned

    def _forward(self, q, keys, values, tables, groups):
        result = torch.zeros_like(q)
        block_size = keys.shape[1]
        head_groups = q.shape[1] // keys.shape[2]
        for first, indices, positions, mask, initial in groups:
            blocks = tables[first, positions // block_size]
            offsets = positions % block_size
            k, v = keys[blocks, offsets], values[blocks, offsets]
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

    def __call__(self, q, keys, values, tables, lengths):
        groups = self._plan([tuple(row) for row in tables.tolist()], lengths.tolist(), q.device)
        return self._forward(q, keys, values, tables, groups)

    def attend_metadata(self, q, keys, values, batch):
        if self._metadata is None or self._metadata() is not batch:
            self._groups = self._plan(
                [(id(allocation), depth) for allocation, depth, _ in batch.rows],
                [position + 1 for _, _, position in batch.rows],
                q.device,
            )
            self._metadata = weakref.ref(batch)
        return self._forward(q, keys, values, batch.block_tables, self._groups)
