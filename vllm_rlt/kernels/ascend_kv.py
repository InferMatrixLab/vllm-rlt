# SPDX-License-Identifier: Apache-2.0
"""Exact KV copies through native Scatter ND on Ascend."""

from __future__ import annotations

import weakref

import torch


class AscendKVWriter:
    """Reuse immutable traversal indices across layers without retaining the descriptor."""

    def __init__(self):
        self._metadata = None
        self._indices = None

    def write(self, keys, values, layer, batch, k, v):
        import torch_npu

        if self._metadata is None or self._metadata() is not batch:
            owner = batch.owner
            addresses = [
                (
                    allocation.block_tables[owner._plane(depth)][position // owner.block_size],
                    position % owner.block_size,
                )
                for allocation, depth, position in batch.rows
            ]
            self._indices = torch.tensor(
                [
                    [[block, index, offset] for block, offset in addresses]
                    for index in range(keys.shape[1])
                ],
                device=keys.device,
                dtype=torch.int32,
            )
            self._metadata = weakref.ref(batch)
        indices = self._indices[layer]
        torch_npu.npu_scatter_nd_update_(keys, indices, k)
        torch_npu.npu_scatter_nd_update_(values, indices, v)
