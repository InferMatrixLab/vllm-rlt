# SPDX-License-Identifier: Apache-2.0
"""Exact KV copies through native Scatter ND on Ascend."""

from __future__ import annotations

import weakref

import torch


class AscendKVWriter:
    """Reuse immutable traversal indices across layers without retaining the descriptor.

    The batch already carries its write addresses as staged device tensors
    (``write_blocks`` / ``write_offsets``). Building the Scatter ND index from
    those tensors keeps the host out of the per-traversal hot path.
    """

    def __init__(self):
        self._metadata = None
        self._indices = None
        self._layer_ids = None

    def write(self, keys, values, layer, batch, k, v):
        import torch_npu

        if self._metadata is None or self._metadata() is not batch:
            blocks = torch.as_tensor(batch.write_blocks, dtype=torch.int32, device=keys.device)
            offsets = torch.as_tensor(batch.write_offsets, dtype=torch.int32, device=keys.device)
            rows = len(batch.rows)
            if blocks.numel() != rows or offsets.numel() != rows:
                raise ValueError("KV write address count does not match the prepared rows")
            layers = keys.shape[1]
            if self._layer_ids is None or self._layer_ids.shape[0] != layers:
                self._layer_ids = torch.arange(
                    layers, device=keys.device, dtype=torch.int32
                )[:, None]
            layer_ids = self._layer_ids.expand(layers, rows)
            self._indices = torch.stack(
                (blocks.expand(layers, -1), layer_ids, offsets.expand(layers, -1)), dim=-1
            )
            self._metadata = weakref.ref(batch)
        indices = self._indices[layer]
        torch_npu.npu_scatter_nd_update_(keys, indices, k)
        torch_npu.npu_scatter_nd_update_(values, indices, v)
