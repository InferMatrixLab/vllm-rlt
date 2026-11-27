"""Address each loop pass through the pages allocated by vLLM."""

from copy import copy
from dataclasses import dataclass, replace

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d


@dataclass(frozen=True)
class _PagedView:
    cache: torch.Tensor
    first_pages: tuple[int, ...]

    @classmethod
    def from_passes(cls, caches):
        first = caches[0]
        if first.ndim != 4 or not first.numel():
            return None
        storage = first.untyped_storage().data_ptr()
        if any(
            cache.shape != first.shape
            or cache.stride() != first.stride()
            or cache.dtype != first.dtype
            or cache.untyped_storage().data_ptr() != storage
            for cache in caches[1:]
        ):
            return None
        offsets = tuple(cache.storage_offset() for cache in caches)
        start = min(offsets)
        stride = first.stride(0)
        if any((offset - start) % stride for offset in offsets):
            return None
        first_pages = tuple((offset - start) // stride for offset in offsets)
        pages = max(first_pages) + first.shape[0]
        if pages >= 2**31:
            return None
        # Each pass keeps its allocated blocks. The view only re-expresses
        # the addresses of those blocks relative to its first page.
        view = first.as_strided((pages, *first.shape[1:]), first.stride(), start)
        return cls(view, first_pages)


@triton.jit
def _route_pages(
    Depths,
    QueryStarts,
    Table,
    Slots,
    tables,
    mappings,
    source_strides: tl.constexpr,
    source_widths: tl.constexpr,
    bases: tl.constexpr,
    block_size: tl.constexpr,
    columns: tl.constexpr,
    tile: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.program_id(1) * tile + tl.arange(0, tile)
    depth = tl.load(Depths + row)
    start = tl.load(QueryStarts + row)
    end = tl.load(QueryStarts + row + 1)
    for i in tl.static_range(len(bases)):
        if depth == i:
            page = tl.load(
                tables[i] + row * source_strides[i] + x,
                x < source_widths[i],
                other=0,
            )
            page = tl.where(page >= 0, page + bases[i], page)
            tl.store(Table + row * columns + x, page, x < columns)
            slot = tl.load(mappings[i] + start + x, start + x < end, other=-1)
            slot = tl.where(slot >= 0, slot + bases[i] * block_size, slot)
            tl.store(Slots + start + x, slot, start + x < end)


@dataclass(frozen=True)
class _PackedLayerCache:
    attention: object
    attention_view: _PagedView
    attention_names: tuple[str, ...]
    conv_view: _PagedView
    conv_names: tuple[str, ...]

    @classmethod
    def from_layer(cls, layer, num_loops):
        attn = layer.attn
        names, caches = zip(*(attn._pass_cache(depth) for depth in range(num_loops)))
        owners = layer.conv_states[:num_loops]
        attention_view = _PagedView.from_passes(caches)
        conv_view = _PagedView.from_passes(tuple(owner.kv_cache for owner in owners))
        if attention_view is None or conv_view is None:
            return None

        # Bind lightweight module views for the native methods. The ordinary
        # path still sees the original bindings and pass selection.
        attention = copy(attn)
        attention.kv_cache = attention_view.cache
        owner = copy(owners[0])
        owner.kv_cache = conv_view.cache
        attention.conv_states = (owner,)
        return cls(attention, attention_view, names, conv_view, tuple(o.prefix for o in owners))


class PackedInklingCache:
    def __init__(self, model):
        layers = [model.layers[index] for index in model.loop_core]
        self.layers = tuple(
            _PackedLayerCache.from_layer(layer, len(layer.conv_states)) for layer in layers
        )

    @property
    def available(self):
        return all(layer is not None for layer in self.layers)

    def prepare(self, spans, metadata, inputs):
        device = inputs.positions.device
        depths = async_tensor_h2d([span.depth for span in spans], dtype=torch.int32, device=device)
        remapped = {}
        result = []
        for layer in self.layers:
            attention = self._route(
                metadata, layer.attention_names, layer.attention_view, depths, inputs, remapped
            )
            conv = self._route(
                metadata, layer.conv_names, layer.conv_view, depths, inputs, remapped
            )
            result.append(
                (
                    layer.attention,
                    {
                        layer.attention_names[0]: attention,
                        layer.conv_names[0]: conv,
                    },
                )
            )
        return tuple(result)

    @staticmethod
    def _route(metadata, names, view, depths, inputs, remapped):
        entries = tuple(metadata[name] for name in names)
        tables = tuple(entry.block_table for entry in entries)
        slots = tuple(entry.slot_mapping for entry in entries)
        key = (
            tuple(map(id, entries)),
            view.first_pages,
            view.cache.shape[2],
        )
        cached = remapped.get(key)
        if cached is not None:
            return cached
        if all(entry is entries[0] for entry in entries) and not any(view.first_pages):
            routed = replace(entries[0], slot_mapping=slots[0][: inputs.num_tokens])
            remapped[key] = routed
            return routed

        requests = depths.numel()
        # Slots for the real query rows only. CUDA graph padding is carried
        # by the dense operations and never reaches these cache addresses.
        tokens = inputs.num_tokens
        columns = max(table.shape[1] for table in tables)
        table = tables[0].new_empty((requests, columns))
        mapping = slots[0].new_empty(tokens)
        tile = 128
        max_query = int(inputs.num_scheduled_tokens.max())
        _route_pages[(requests, triton.cdiv(max(columns, max_query), tile))](
            depths,
            inputs.query_start_loc,
            table,
            mapping,
            tables,
            slots,
            tuple(table.stride(0) for table in tables),
            tuple(table.shape[1] for table in tables),
            view.first_pages,
            view.cache.shape[2],
            columns,
            tile,
        )
        routed = replace(entries[0], block_table=table, slot_mapping=mapping)
        remapped[key] = routed
        return routed
