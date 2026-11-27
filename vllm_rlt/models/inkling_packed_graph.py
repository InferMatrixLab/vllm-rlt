"""Fixed graph inputs for Inkling's cross-pass prefill CORE."""

import torch
from vllm.models.inkling.nvidia.sconv_swa_attn import InklingSconvMetadata
from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
from vllm.models.inkling2.nvidia.ops.linear import linear

from vllm_rlt.models.inkling_core_routes import pack_core_inputs
from vllm_rlt.models.inkling_mixed_core import _attend, _attention_metadata


class PackedCoreGraphInputs:
    def __init__(self, runner, model, cache, width):
        self.model = model
        self.width = width
        core = model.loop_core
        block_size = model.block_size
        self.prelude_blocks = sum(i % block_size == 0 for i in range(core.start))
        self.core_blocks = sum(i % block_size == 0 for i in core)
        self.completed = self.prelude_blocks + self.core_blocks
        prototype = model.layers[core.start].attn_norm.weight
        hidden = prototype.shape[-1]
        dtype = runner.model_config.dtype
        self.stack = prototype.new_zeros((model._max_loop_blocks() + 1, width, hidden), dtype=dtype)
        self.bank = prototype.new_zeros((self.core_blocks, width, hidden), dtype=dtype)
        self.positions = prototype.new_zeros(width, dtype=torch.int64)
        self.routes = prototype.new_full((width,), -1, dtype=torch.int64)
        self.rows = torch.arange(width, device=prototype.device)
        # The native controller mixes FP32 inputs. First-pass BF16 values can
        # be stored here exactly and are selected without another norm.
        self.initial = prototype.new_zeros(
            (model.layers[core.start].num_attn_inputs, width, hidden), dtype=torch.float32
        )
        self.qkv = prototype.new_empty((3, width, hidden), dtype=dtype)
        self.mlp = prototype.new_empty((1, width, hidden), dtype=dtype)
        self.coda = prototype.new_empty(
            (model.layers[core.stop].num_attn_inputs, width, hidden), dtype=dtype
        )
        self.seq_idx = prototype.new_zeros(width, dtype=torch.int32)
        self.query_start = prototype.new_zeros(width, dtype=torch.int32)

        group_for = {
            name: index
            for index, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }
        by_route = {}
        self.conv_metadata = {}
        self.conv_sources = []
        self.conv_owners = []
        for i, layer in enumerate(cache.layers):
            groups = tuple(group_for[name] for name in layer.conv_names)
            key = groups, layer.conv_view.first_pages
            owner = layer.attention.conv_states[0]
            metadata = by_route.get(key)
            if metadata is None:
                tables = runner.block_tables.input_block_tables
                columns = max(tables[group].shape[1] for group in groups)
                metadata = InklingSconvMetadata(
                    prototype.new_zeros((runner.max_num_reqs, columns), dtype=torch.int32),
                    prototype.new_full((width,), -1, dtype=torch.int64),
                    self.seq_idx,
                    self.query_start,
                )
                by_route[key] = metadata
                self.conv_sources.append((i, owner.prefix, metadata))
            self.conv_metadata[owner.prefix] = metadata
            self.conv_owners.append(owner)

    def warm_initial_inputs(self):
        # Staging is outside the graph and can receive BF16 first-pass rows
        # alongside FP32 continuation rows, or FP32 continuation rows only.
        norm = self.model.layers[self.model.loop_core.start].attn_norm
        first = self.initial[:, :1].to(self.qkv.dtype)
        for source in (first, self.initial):
            pack_core_inputs(
                self.initial,
                source,
                self.initial,
                self.routes,
                0,
                norm,
                normalize_first=False,
                first_is_full=False,
            )

    def stage(self, positions, execution):
        live = execution.positions.numel()
        if execution.blocks != self.prelude_blocks or positions.numel() != self.width:
            return False
        sources = [execution.metadata[i][1][name] for i, name, _ in self.conv_sources]
        if any(
            source.block_table.shape[1] != target.block_table.shape[1]
            or source.block_table.shape[0] > target.block_table.shape[0]
            for source, (_, _, target) in zip(sources, self.conv_sources, strict=True)
        ):
            return False
        blocks = execution.blocks + 1
        self.stack[:blocks, :live].copy_(execution.stack[:blocks])
        self.stack[:blocks, live:].zero_()
        self.positions.copy_(positions)
        self.bank.zero_()
        if execution.later_rows is not None:
            self.bank.index_copy_(1, execution.later_rows, execution.bank)
        self.routes[:live].copy_(torch.where(execution.row_routes >= 0, self.rows[:live], -1))
        self.routes[live:].fill_(-1)
        pack_core_inputs(
            self.initial,
            execution.first_inputs,
            execution.later_inputs,
            execution.row_routes,
            live,
            self.model.layers[self.model.loop_core.start].attn_norm,
            normalize_first=False,
            first_is_full=False,
        )
        self.seq_idx[:live].copy_(sources[0].seq_idx[:live])
        self.seq_idx[live:].zero_()
        self.query_start[:live].copy_(sources[0].query_start[:live])
        self.query_start[live:].zero_()
        for source, (_, _, target) in zip(sources, self.conv_sources, strict=True):
            target.block_table[: source.block_table.shape[0]].copy_(source.block_table)
            target.slot_mapping[:live].copy_(source.slot_mapping[:live])
            target.slot_mapping[live:].fill_(-1)
        return True

    def finish(self, execution):
        live = execution.positions.numel()
        execution.blocks = self.completed
        execution.stack[: self.completed + 1].copy_(self.stack[: self.completed + 1, :live])
        # Decode rows may survive another replay before their CODA consumes them.
        execution.coda_inputs = self.coda[:, :live].clone(memory_format=torch.contiguous_format)

    def forward(self):
        model = self.model
        core = model.loop_core
        stack, blocks = self.stack, self.prelude_blocks
        inputs = self.initial
        with _attention_metadata(self.conv_metadata):
            for i in core:
                layer = model.layers[i]
                norm = layer.attn_norm
                weights = norm.weight.expand(3, -1) if layer.is_local else norm.weight
                messages = attn_res(
                    self.bank[:1],
                    self.bank[1],
                    None,
                    model.message_queries[i - core.start],
                    None,
                    0.0,
                    True,
                )
                mixed = layer.blend(
                    slice(0, 3), inputs.expand(3, -1, -1), messages[:3], weights, norm.eps
                )
                pack_core_inputs(
                    self.qkv,
                    inputs,
                    mixed,
                    self.routes,
                    self.width,
                    norm,
                    normalize_first=i != core.start,
                    first_is_full=True,
                )
                xq, xk, xv = self.qkv.unbind()
                attn = layer.attn
                q, r, gate, sinks, k, v = attn._project(xq, xk, xv)
                k_dynamic = attn.k_sconv.dynamic_taps(xk)
                v_dynamic = attn.v_sconv.dynamic_taps(xv)
                out = torch.empty_like(gate, dtype=q.dtype)
                _attend(model, i, out, q, r, gate, sinks, k, v, k_dynamic, v_dynamic)
                projected = linear(out, attn.wo_ud.weight)

                first = i % model.block_size == 0
                partial = stack[blocks]
                attn_out = torch.empty_like(partial)
                owner = self.conv_owners[i - core.start]
                mlp_inputs = layer.attn_sconv.conv_mix(
                    projected,
                    self.positions,
                    attn_out,
                    stack[: blocks + first],
                    None if first else partial,
                    model.attn_res.mlp_queries[i],
                    None,
                    owner=owner,
                )
                norm = layer.mlp_norm
                mixed = layer.blend(slice(3, 4), mlp_inputs, messages[3:], norm.weight, norm.eps)
                pack_core_inputs(
                    self.mlp,
                    mlp_inputs,
                    mixed,
                    self.routes,
                    self.width,
                    norm,
                    normalize_first=True,
                    first_is_full=True,
                )
                blocks += first
                inputs = layer.mlp_sconv.conv_mix(
                    layer.mlp(self.mlp[0]),
                    self.positions,
                    stack[blocks],
                    stack[:blocks],
                    None,
                    model.attn_res.attn_queries[i],
                    model.layers[core.stop].attn_norm if i + 1 == core.stop else None,
                    addend=attn_out,
                    residual=None if first else partial,
                    owner=owner,
                )
        self.coda.copy_(inputs)
        return self.coda
