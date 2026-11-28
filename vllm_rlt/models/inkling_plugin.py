"""Resume Inkling's upstream layer loop at Prelude, CORE, and Coda boundaries."""

from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, replace
from itertools import groupby
from operator import attrgetter
from types import MethodType

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.inkling.nvidia.sconv_swa_attn import (
    InklingSconvMetadata,
    InklingSconvMetadataBuilder,
)
from vllm.models.inkling2.nvidia.loop import Inkling2LoopForConditionalGeneration
from vllm.models.inkling2.nvidia.ops import use_sm100_kernels
from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
from vllm.models.inkling2.nvidia.sconv import conv_metadata
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchSlice, _slice_input_batch
from vllm.v1.worker.utils import groups_with_metadata_builders

from vllm_rlt.core.loop import LoopLayout, LoopOp, LoopSpan
from vllm_rlt.models.inkling_core_routes import prepare_routes, route_live_rows
from vllm_rlt.models.inkling_mixed_core import execute_compact_core, execute_mixed_core
from vllm_rlt.models.inkling_packed_cache import PackedInklingCache
from vllm_rlt.worker.loop_state import LoopExecution, loop_execution

_coda_tail_padding = ContextVar("rlt_coda_tail_padding", default=None)
_sconv_sequence_metadata = ContextVar("rlt_sconv_sequence_metadata", default=None)
logger = init_logger(__name__)


def _share_sconv_sequence_metadata(builder):
    if type(builder) is not InklingSconvMetadataBuilder or getattr(
        builder, "_rlt_shared_sequence", False
    ):
        return
    native = builder.build

    def build(common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        cache = _sconv_sequence_metadata.get()
        if cache is None:
            return native(common_prefix_len, common, fast_build)
        key = (id(common.query_start_loc), common.num_reqs, common.slot_mapping.shape[0])
        source = cache.get(key)
        if source is not None:
            return InklingSconvMetadata(
                block_table=common.block_table_tensor,
                slot_mapping=common.slot_mapping,
                seq_idx=source.seq_idx,
                query_start=source.query_start,
            )
        result = native(common_prefix_len, common, fast_build)
        cache[key] = result
        return result

    builder.build = build
    builder._rlt_shared_sequence = True


def _bind_uniform_decode_plan(layer):
    native = layer._plan_decode

    def plan(self, metadata, tokens):
        requests, decode_tokens, splits = native(metadata, tokens)
        step = loop_execution.get()
        warming_full = (
            step is not None
            and step.graphs is not None
            and step.graphs.warming_up
            and not any(span.prefill for span in step.ordered)
        )
        if (
            step is not None
            and splits
            and requests
            and decode_tokens == tokens
            and metadata.max_query_len == 1
            and (warming_full or get_forward_context().cudagraph_runtime_mode is CUDAGraphMode.FULL)
        ):
            work_per_split = requests * self.num_kv_heads
            splits = min(splits, cdiv(2 * num_compute_units(), work_per_split))
        return requests, decode_tokens, splits

    if not layer.is_local:
        layer._plan_decode = MethodType(plan, layer)


class LoopInklingModelState(DefaultModelState):
    def can_pack_core(self):
        cache = getattr(self, "_packed_loop_cache", None)
        if cache is None:
            cache = self._packed_loop_cache = PackedInklingCache(self.model.model)
            if not cache.available:
                logger.warning("WCPB uses grouped CORE for this KV cache layout")
        return cache.available

    def prepare_inputs(self, input_batch, req_states):
        model_inputs = super().prepare_inputs(input_batch, req_states)
        step = loop_execution.get()
        if step is not None and step.release_uva_inputs is not None:
            step.release_uva_inputs()
            step.release_uva_inputs = None
        return model_inputs

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ):
        step = loop_execution.get()
        if step is not None and step.compact_core and not step.packed_core:
            return self._prepare_compact_core(
                step, input_batch, block_tables, slot_mappings, attn_groups, kv_cache_config
            )
        prepared = step.prepared_batch if step is not None else None
        if step is not None and step.padded_graph_mode is not None:
            # MRv2 passes its original descriptor's mode. FULL metadata must
            # use the padding chosen for this loop op, even with fresh inputs.
            cudagraph_mode = step.padded_graph_mode
        eligible = (
            prepared is not None
            and prepared.inputs is input_batch
            and prepared.descriptor.cg_mode is cudagraph_mode
            and prepared.attention is not None
            and prepared.attention[0] is block_tables
            and prepared.attention[1] is slot_mappings
            and not for_capture
            and not ubatch_idx
            and model_specific_attn_metadata is None
        )
        if eligible and prepared.metadata is not None:
            return prepared.metadata
        if prepared is not None:
            prepared.metadata = None
            prepared.metadata_layout = None

        if getattr(self, "_bound_sconv_groups", None) is not attn_groups:
            for groups in attn_groups:
                for group in groups_with_metadata_builders(groups):
                    _share_sconv_sequence_metadata(group.get_metadata_builder(0))
            self._bound_sconv_groups = attn_groups

        # MRv2 already shares query geometry across KV groups. Compute the
        # short-conv row map once, keeping each group's own KV slots and table.
        sequence_cache = _sconv_sequence_metadata.set({} if ubatch_idx == 0 else None)
        try:
            metadata = super().prepare_attn(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture,
                ubatch_idx,
                model_specific_attn_metadata,
            )
        finally:
            _sconv_sequence_metadata.reset(sequence_cache)
        if step is not None and step.packed_core:
            step.packed_core_metadata = self._packed_loop_cache.prepare(
                step.ordered, metadata, input_batch
            )
        if eligible:
            prepared.metadata = metadata
        return metadata

    def _prepare_compact_core(
        self, step, input_batch, block_tables, slot_mappings, attn_groups, kv_cache_config
    ):
        # MRv2's metadata builders can own persistent scratch tensors. Each
        # sub-batch needs its own slot because the layer loop interleaves them.
        groups_by_depth = getattr(self, "_compact_core_attn_groups", None)
        if groups_by_depth is None:
            layout = LoopLayout.from_model_config(self.model_config)
            model = self.model.model
            groups_by_depth = []
            for depth in range(layout.num_loops):
                layer_names = set()
                for index in model.loop_core:
                    layer = model.layers[index]
                    layer_names.add(layer.attn._pass_cache(depth)[0])
                    layer_names.add(layer.conv_states[depth].prefix)
                groups = []
                for kv_group, original in zip(
                    kv_cache_config.kv_cache_groups, attn_groups, strict=True
                ):
                    copies = []
                    for group in original:
                        names = [name for name in group.layer_names if name in layer_names]
                        if not names or not group.metadata_builders:
                            continue
                        copied = replace(group, layer_names=names)
                        copied.create_metadata_builders(
                            self.vllm_config,
                            self.device,
                            kernel_block_size=group.kernel_block_size,
                            num_metadata_builders=2,
                            kv_cache_group=kv_group,
                        )
                        copies.append(copied)
                    groups.append(copies)
                groups_by_depth.append(groups)
            self._compact_core_attn_groups = groups_by_depth

        req_start = token_start = 0
        metadata = []
        for (depth, terminal), spans in groupby(step.ordered, attrgetter("depth", "last_core")):
            spans = tuple(spans)
            req_end = req_start + len(spans)
            token_end = token_start + sum(span.token_count for span in spans)
            requests = slice(req_start, req_end)
            tokens = slice(token_start, token_end)
            batch = _slice_input_batch(
                input_batch,
                UBatchSlice(requests, tokens),
                input_batch.query_start_loc.new_empty(len(spans) + 1),
                input_batch.seq_lens.new_empty(len(spans)),
            )
            batch.max_query_len = max(span.token_count for span in spans)
            metadata.append(
                super().prepare_attn(
                    batch,
                    CUDAGraphMode.NONE,
                    tuple(table[requests] for table in block_tables),
                    slot_mappings[:, tokens],
                    groups_by_depth[depth],
                    kv_cache_config,
                    ubatch_idx=int(terminal),
                )
            )
            req_start, token_start = req_end, token_end
        step.core_group_metadata = tuple(metadata)
        # Only the per-group metadata is consumed by this CORE execution.
        return {}


@dataclass(slots=True)
class _InklingState:
    stack: torch.Tensor
    prelude: torch.Tensor
    inputs: torch.Tensor
    bank: torch.Tensor | None
    prelude_blocks: int
    num_blocks: int
    uses_graph_stack: bool = False
    frame: "_InklingState | None" = None
    frame_start: int = 0
    live_tokens: int | None = None

    def pad_rows(self, count):
        if count == 0:
            return
        self.stack = torch.nn.functional.pad(self.stack, (0, 0, 0, count))
        self.prelude = torch.nn.functional.pad(self.prelude, (0, 0, 0, count))
        self.inputs = torch.nn.functional.pad(self.inputs, (0, 0, 0, count))
        if self.bank is not None:
            self.bank = torch.nn.functional.pad(self.bank, (0, 0, 0, count))

    def rows(self, start, end, *, copy=True):
        stack = self.stack[:, start:end]
        prelude = self.prelude[start:end]
        inputs = self.inputs[:, start:end]
        bank = None if self.bank is None else self.bank[:, start:end]
        if copy:
            stack, prelude, inputs = stack.clone(), prelude.clone(), inputs.clone()
            bank = None if bank is None else bank.clone()
        return _InklingState(
            stack,
            prelude,
            inputs,
            bank,
            self.prelude_blocks,
            self.num_blocks,
        )

    def store_rows(self, spans, store, *, copy=True, keep_frame=False):
        start = 0
        for span in spans:
            end = start + span.token_count
            row = self.rows(start, end, copy=copy)
            if keep_frame:
                row.frame = self
                row.frame_start = start
            store.setdefault(span.request_id, {})[span.token_start] = row
            start = end
        if keep_frame:
            self.live_tokens = start

    def select_rows(self, indices):
        return _InklingState(
            self.stack.index_select(1, indices),
            self.prelude.index_select(0, indices),
            self.inputs.index_select(1, indices),
            None if self.bank is None else self.bank.index_select(1, indices),
            self.prelude_blocks,
            self.num_blocks,
        )

    @classmethod
    def concatenate(cls, states, padded_tokens=None, zero_storage=None):
        first = states[0]
        actual = sum(state.stack.shape[1] for state in states)
        tokens = actual if padded_tokens is None else padded_tokens
        padding = tokens - actual
        if len(states) == 1 and padding == 0:
            first.stack = first.stack.contiguous()
            first.inputs = first.inputs.contiguous()
            if first.bank is not None:
                first.bank = first.bank.contiguous()
            return first

        def join(tensors, dim, out=None):
            if padding:
                prototype = tensors[0]
                shape = list(prototype.shape)
                shape[dim] = padding
                size = prototype.numel() // prototype.shape[dim] * padding
                zeros = zero_storage.get(prototype.dtype)
                if zeros is None or zeros.numel() < size:
                    capacity = 1 << (size - 1).bit_length()
                    zeros = zero_storage[prototype.dtype] = prototype.new_zeros(capacity)
                tensors.append(zeros[:size].view(shape))
            return torch.cat(tensors, dim=dim, out=out)

        # The graph expects room for later blocks. Only the blocks produced by
        # the previous operation will be read by the next one.
        stack = first.stack.new_empty((first.stack.shape[0], tokens, first.stack.shape[2]))
        live = first.num_blocks + 1
        join([state.stack[:live] for state in states], 1, out=stack[:live])
        return cls(
            stack,
            join([state.prelude for state in states], 0),
            join([state.inputs for state in states], 1),
            None if first.bank is None else join([state.bank for state in states], 1),
            first.prelude_blocks,
            first.num_blocks,
        )


@dataclass(slots=True)
class _InklingStateSlot:
    state: _InklingState
    index: int

    def rows(self, *, copy=True):
        return self.state.rows(self.index, self.index + 1, copy=copy)


class _InklingDecodeStatePool:
    """Own decode rows at MRv2 request slots for each intermediate state layout."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.layouts = {}

    def store(self, source, spans, store, slots, indices):
        layout = (
            source.prelude_blocks,
            source.num_blocks,
            source.inputs.shape[0],
            source.inputs.dtype,
            None if source.bank is None else (source.bank.shape[0], source.bank.dtype),
        )
        entry = self.layouts.get(layout)
        if entry is None:
            target = _InklingState(
                source.stack.new_empty(
                    (source.stack.shape[0], self.capacity, source.stack.shape[2])
                ),
                source.prelude.new_empty((self.capacity, source.prelude.shape[1])),
                source.inputs.new_empty(
                    (source.inputs.shape[0], self.capacity, source.inputs.shape[2])
                ),
                None
                if source.bank is None
                else source.bank.new_empty(
                    (source.bank.shape[0], self.capacity, source.bank.shape[2])
                ),
                source.prelude_blocks,
                source.num_blocks,
            )
            entry = target, {}
            self.layouts[layout] = entry
        target, owners = entry
        for span, slot in zip(spans, slots, strict=True):
            key = (span.request_id, span.token_start)
            old = owners.get(slot)
            if old is not None and old != key:
                positions = store.get(old[0], {})
                existing = positions.get(old[1])
                if isinstance(existing, _InklingStateSlot) and existing.state is target:
                    # An older position can still be live with an async lookahead.
                    positions[old[1]] = target.rows(slot, slot + 1)
            owners[slot] = key
            store.setdefault(span.request_id, {})[span.token_start] = _InklingStateSlot(
                target, slot
            )
        count = len(spans)
        live = source.num_blocks + 1
        target.stack[:live].index_copy_(1, indices, source.stack[:live, :count])
        target.prelude.index_copy_(0, indices, source.prelude[:count])
        target.inputs.index_copy_(1, indices, source.inputs[:, :count])
        if target.bank is not None:
            target.bank.index_copy_(1, indices, source.bank[:, :count])


@dataclass(slots=True)
class _InklingBatchState:
    state: _InklingState
    spans: tuple[LoopSpan, ...]
    store: dict[str, dict[int, object]]
    pool: _InklingDecodeStatePool | None = None
    slots: tuple[int, ...] = ()
    indices: torch.Tensor | None = None
    move_rows: bool = False

    @torch.inference_mode()
    def unpack(self):
        # Use the physical order of the preceding op and exclude padding rows.
        if self.pool is None:
            if self.move_rows and self.state.uses_graph_stack:
                stack = torch.empty_like(self.state.stack)
                live = self.state.num_blocks + 1
                stack[:live].copy_(self.state.stack[:live])
                self.state.stack = stack
                self.state.uses_graph_stack = False
            self.state.store_rows(self.spans, self.store, copy=not self.move_rows)
        else:
            assert self.indices is not None
            self.pool.store(self.state, self.spans, self.store, self.slots, self.indices)


class LoopInklingForConditionalGeneration(Inkling2LoopForConditionalGeneration):
    rlt_loop_ops = True
    rlt_graph_warmup = True
    rlt_partition_core_variants = True

    @staticmethod
    def get_model_state_cls():
        return LoopInklingModelState

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs
    ):
        step = loop_execution.get()
        if step is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        if use_sm100_kernels() and not getattr(self, "_rlt_uniform_decode_plan", False):
            for layer in self.model.layers:
                _bind_uniform_decode_plan(layer.attn)
            self._rlt_uniform_decode_plan = True
        if step.op is LoopOp.PRELUDE and inputs_embeds is None:
            inputs_embeds = self.model.embed_input_ids(input_ids)
        return self._forward_step(step, positions, inputs_embeds)

    def _forward_step(self, step: LoopExecution, positions, inputs_embeds):
        model = self.model
        core = model.loop_core
        # Within the scaling floor the native formula yields exactly 1.
        # The prologue already has an unscaled branch with the same rounding.
        floor = model.config.log_scaling_n_floor
        identity_scale = (
            step.context_limit is not None
            and floor is not None
            and floor > 0
            and floor & (floor - 1) == 0
            and step.context_limit <= floor
        )
        tau = None if identity_scale else model._log_scaling_tau(positions)
        if step.compact_core:
            return execute_compact_core(self, step, positions, tau)
        if (
            step.op is LoopOp.CORE
            and not step.merged_core
            and len({(span.depth, span.last_core, span.propose_draft) for span in step.ordered}) > 1
        ):
            return execute_mixed_core(self, step, positions, tau)
        plan = [(False, False)] * 2
        workspace = model.allreduce_workspace
        if workspace is not None and conv_metadata(model.layers[0].conv_state) is not None:
            plan = workspace.plan(positions.numel())

        prepared = step.prepared_batch
        if prepared is not None and not prepared.keep_model_state:
            prepared = None
        if step.op is LoopOp.PRELUDE:
            if prepared is not None:
                prepared.unpack_state()
            stack = inputs_embeds.new_empty((model._max_loop_blocks() + 1, *inputs_embeds.shape))
            stack[0].copy_(inputs_embeds)
            blocks, inputs = self._execute_layers(
                step,
                range(core.start),
                0,
                stack,
                0,
                None,
                None,
                core.start,
                0,
                positions,
                tau,
                plan,
            )
            if step.graph_stack is not None:
                stack = step.graph_stack
            state = _InklingState(
                stack,
                stack[blocks].clone(),
                inputs,
                None,
                blocks,
                blocks,
                uses_graph_stack=step.graph_stack is not None,
            )
        else:
            pending = prepared.packed_state if prepared is not None else None
            if pending is not None:
                assert isinstance(pending, _InklingBatchState)
                assert prepared is not None
                prepared.packed_state = None
                state = pending.state
            else:
                state = self._take_state(step)
            if step.op is LoopOp.CODA:
                _, inputs = self._execute_layers(
                    step,
                    range(core.stop, len(model.layers)),
                    0,
                    state.stack,
                    state.num_blocks,
                    state.inputs,
                    None,
                    len(model.layers),
                    0,
                    positions,
                    tau,
                    plan,
                )
                return inputs[0]

            depth = step.ordered[0].depth
            final = step.ordered[0].last_core
            proposal = step.ordered[0].propose_draft
            conv = (
                conv_metadata(model.layers[core.start].conv_states[depth])
                if step.merged_core and not step.decode_slot_capacity
                else None
            )
            if conv is not None:
                routes = prepare_routes(step.ordered, positions.device)
            else:
                routes = (
                    self._prepare_core_routes(step, positions.device) if step.merged_core else None
                )
            blocks, inputs = self._execute_layers(
                step,
                core,
                depth,
                state.stack,
                state.num_blocks,
                state.inputs,
                state.bank,
                core.stop if final or proposal or step.merged_core else core.start,
                0 if final or proposal or step.merged_core else depth + 1,
                positions,
                tau,
                plan,
            )
            if step.graph_stack is not None:
                state.stack = step.graph_stack
                state.uses_graph_stack = True
            if conv is not None:
                self._store_fused_core_routes(step, state, blocks, inputs, routes, conv.seq_idx)
                return state.stack[blocks]
            if routes is not None:
                self._store_merged_core(step, state, blocks, inputs, routes)
                return state.stack[blocks]
            prelude_dests = state.inputs if state.bank is not None else None
            state.num_blocks = blocks
            state.inputs = inputs
            if final:
                state.bank = None
            else:
                if proposal:
                    state.store_rows(step.ordered, step.draft_state)
                self._continue_core(state, prelude_dests)

        if prepared is None:
            self._store_state_rows(step, state, step.ordered)
        else:
            if step.graphs is not None and (step.op is LoopOp.PRELUDE or step.ordered[0].last_core):
                # These outputs can alias a replay buffer that the next use of the graph overwrites.
                state.inputs = state.inputs.clone()
            pool = self._decode_pool_for(step)
            if pool is None:
                prepared.packed_state = _InklingBatchState(
                    state, step.ordered, step.state, move_rows=True
                )
            else:
                prepared.packed_state = _InklingBatchState(
                    state,
                    step.ordered,
                    step.state,
                    pool,
                    tuple(row[1] for row in prepared.rows),
                    prepared.inputs.idx_mapping,
                )
        return state.stack[state.num_blocks]

    @staticmethod
    def _store_span_state(step, span, state):
        step.state.setdefault(span.request_id, {})[span.token_start] = state

    def _store_state_rows(self, step, state, spans, *, copy=True, keep_frame=False):
        state.store_rows(spans, step.state, copy=copy, keep_frame=keep_frame)

    @staticmethod
    def _pop_core_states(step, spans, *, copy_slots=True):
        states = [step.state[span.request_id].pop(span.token_start) for span in spans]
        # Mixed CORE can retain an input row after the decode slot is reused.
        return [
            state.rows(copy=copy_slots) if isinstance(state, _InklingStateSlot) else state
            for state in states
        ]

    def _take_state(self, step):
        states = []
        for span in step.ordered:
            positions = (step.draft_state if span.draft_coda else step.state)[span.request_id]
            state = positions.pop(span.token_start)
            if span.coda_remainder:
                assert step.op is LoopOp.CODA and isinstance(state, _InklingState)
                end = span.token_count + span.coda_remainder
                assert state.prelude.shape[0] == end
                positions[span.token_end] = state.rows(span.token_count, end, copy=False)
                state = state.rows(0, span.token_count, copy=False)
            states.append(state)
        prepared = step.prepared_batch
        first = states[0]
        if isinstance(first, _InklingState) and first.frame is not None:
            frame = first.frame
            expected = step.padded_tokens or sum(span.token_count for span in step.ordered)
            if frame.prelude.shape[0] == expected:
                offset = 0
                for row in states:
                    if (
                        not isinstance(row, _InklingState)
                        or row.frame is not frame
                        or row.frame_start != offset
                    ):
                        break
                    offset += row.prelude.shape[0]
                else:
                    if offset == frame.live_tokens:
                        return frame
        if (
            isinstance(first, _InklingStateSlot)
            and prepared is not None
            and all(
                isinstance(item, _InklingStateSlot)
                and item.state is first.state
                and item.index == row[1]
                for item, row in zip(states, prepared.rows, strict=True)
            )
        ):
            state = first.state.select_rows(prepared.inputs.idx_mapping)
            if step.padded_tokens is not None:
                state.pad_rows(step.padded_tokens - state.prelude.shape[0])
            return state
        zero_storage = getattr(self, "_state_padding", None)
        if step.padded_tokens is not None and zero_storage is None:
            zero_storage = self._state_padding = {}
        return _InklingState.concatenate(
            [
                item.rows(copy=len(states) == 1)
                if isinstance(item, _InklingStateSlot)
                else item
                for item in states
            ],
            step.padded_tokens,
            zero_storage,
        )

    def _decode_pool_for(self, step):
        if not step.decode_slot_capacity or step.prepared_batch is None:
            return None
        pool = getattr(self, "_decode_state_pool", None)
        if pool is None:
            pool = self._decode_state_pool = _InklingDecodeStatePool(step.decode_slot_capacity)
        return pool

    def _store_packed_decode(self, step, pool, states, execution):
        prepared = step.prepared_batch
        assert prepared is not None
        routed = defaultdict(list)
        for index, (span, state) in enumerate(zip(step.ordered, states, strict=True)):
            routed[span.last_core, not span.last_core and state.bank is not None].append(index)
        order = async_tensor_h2d(
            [index for indices in routed.values() for index in indices],
            device=execution.stack.device,
            dtype=torch.long,
        )
        offset = 0
        for (terminal, reuse_inputs), indices in routed.items():
            end = offset + len(indices)
            rows = order[offset:end]
            offset = end
            source = _InklingState(
                execution.stack.index_select(1, rows),
                torch.cat([states[index].prelude for index in indices], dim=0),
                execution.coda_inputs.index_select(1, rows) if terminal else None,
                None,
                states[0].prelude_blocks,
                execution.blocks,
            )
            if not terminal:
                previous = (
                    torch.cat([states[index].inputs for index in indices], dim=1)
                    if reuse_inputs else None
                )
                self._continue_core(source, previous)
            pool.store(
                source,
                tuple(step.ordered[index] for index in indices),
                step.state,
                tuple(prepared.rows[index][1] for index in indices),
                prepared.inputs.idx_mapping.index_select(0, rows),
            )

    def _continue_core(self, state, prelude_dests=None):
        model = self.model
        blocks = state.prelude_blocks
        state.bank = state.stack[blocks + 1 : state.num_blocks + 1].clone()
        state.num_blocks = blocks
        state.stack[blocks].copy_(state.prelude)
        if prelude_dests is None:
            prelude_dests = attn_res(
                state.stack[:blocks],
                state.prelude,
                None,
                model.attn_res.attn_queries[model.loop_core.start - 1],
                None,
                0.0,
                True,
            )
        state.inputs = prelude_dests

    @staticmethod
    def _prepare_core_routes(step, device):
        continuing, terminal = [], []
        continuing_rows, terminal_rows = [], []
        offset = 0
        for index, span in enumerate(step.ordered):
            spans, rows = (
                (terminal, terminal_rows) if span.last_core else (continuing, continuing_rows)
            )
            spans.append(index)
            end = offset + span.token_count
            rows.extend(range(offset, end))
            offset = end
        # MRv2's pinned, nonblocking input transfer keeps the host ahead of replay.
        indices = async_tensor_h2d(continuing_rows + terminal_rows, device=device, dtype=torch.long)
        cutoff = len(continuing_rows)
        return ((continuing, indices[:cutoff]), (terminal, indices[cutoff:]))

    def _store_fused_core_routes(self, step, state, blocks, inputs, routes, seq_idx):
        continuing_rows = routes.lengths[0]
        continue_width = continuing_rows
        if step.graphs is not None:
            prefill = any(span.prefill for span in routes.spans[0])
            choose = (
                step.graphs.padded_piecewise_size if prefill else step.graphs.padded_decode_size
            )
            continue_width = choose(continuing_rows) or continuing_rows
        cont_stack, cont_prelude, cont_inputs, bank, term_stack, term_prelude, term_inputs = (
            route_live_rows(state, blocks, inputs, routes, seq_idx, continue_width)
        )
        prelude_blocks = state.prelude_blocks
        continuing = _InklingState(
            cont_stack,
            cont_prelude,
            cont_inputs,
            bank,
            prelude_blocks,
            prelude_blocks,
        )
        if continuing.inputs is None:
            model = self.model
            continuing.inputs = attn_res(
                cont_stack[:prelude_blocks, :continuing_rows],
                cont_prelude[:continuing_rows],
                None,
                model.attn_res.attn_queries[model.loop_core.start - 1],
                None,
                0.0,
                True,
            )
        terminal = _InklingState(
            term_stack,
            term_prelude,
            term_inputs,
            None,
            prelude_blocks,
            blocks,
        )
        if continuing.inputs.shape[1] < continue_width:
            continuing.inputs = torch.nn.functional.pad(
                continuing.inputs, (0, 0, 0, continue_width - continuing_rows)
            )
        self._store_state_rows(step, continuing, routes.spans[0], copy=False, keep_frame=True)
        self._store_state_rows(step, terminal, routes.spans[1], copy=False)

    def _store_merged_core(self, step, state, blocks, inputs, routes):
        # The last layer writes the same stack for both routes. Its Coda mix is
        # consumed by terminal rows; continuing rows re-mix the Prelude stack.
        pool = self._decode_pool_for(step)
        for final, (indices, rows) in zip((False, True), routes, strict=True):
            spans = tuple(step.ordered[i] for i in indices)
            child = _InklingState(
                state.stack.index_select(1, rows),
                state.prelude.index_select(0, rows),
                inputs.index_select(1, rows) if final else None,
                None,
                state.prelude_blocks,
                blocks,
            )
            if not final:
                prelude_dests = (
                    state.inputs.index_select(1, rows) if state.bank is not None else None
                )
                self._continue_core(child, prelude_dests)
            if pool is None:
                self._store_state_rows(step, child, spans, copy=False)
            else:
                prepared = step.prepared_batch
                assert prepared is not None
                pool.store(
                    child,
                    spans,
                    step.state,
                    (prepared.rows[i][1] for i in indices),
                    prepared.inputs.idx_mapping.index_select(0, rows),
                )

    def _coda_mlp_padding(self, step, positions):
        if (
            not step.coda_tail_moe
            or step.op is not LoopOp.CODA
            or positions.numel() < 256
            or not any(span.prefill for span in step.ordered)
        ):
            return None
        moe = self.model.layers[-1].mlp
        if not getattr(moe, "_rlt_tail_padding", False):
            forward = moe.forward

            def masked_mlp(*args, **kwargs):
                padding = _coda_tail_padding.get()
                if padding is None:
                    return forward(*args, **kwargs)
                context = get_forward_context()
                original = context.is_padding
                context.is_padding = padding
                try:
                    return forward(*args, **kwargs)
                finally:
                    context.is_padding = original

            moe.forward = masked_mlp
            moe._rlt_tail_padding = True

        # The last layer has already produced attention KV for every row.
        # Retain the short-conv window at each request's tail for sampling
        # and for the next chunk's causal convolution.
        width = self.model.layers[-1].mlp_sconv.conv_weight.shape[1]
        padding = np.ones(positions.numel(), dtype=np.bool_)
        start = 0
        for span in step.ordered:
            end = start + span.token_count
            padding[max(start, end - width) : end] = False
            start = end
        return async_tensor_h2d(padding, device=positions.device, dtype=torch.bool)

    def _execute_layers(
        self,
        step,
        layers,
        depth,
        stack,
        blocks,
        inputs,
        bank,
        next_layer,
        next_depth,
        positions,
        tau,
        plan,
    ):
        coda_padding = self._coda_mlp_padding(step, positions)

        def compute(stack, inputs, bank, positions, tau, coda_padding):
            _, output = self._run_layers(
                layers,
                depth,
                stack,
                blocks,
                inputs,
                bank,
                next_layer,
                next_depth,
                positions,
                tau,
                plan,
                coda_padding,
            )
            return output

        if step.graphs is None:
            return self._run_layers(
                layers,
                depth,
                stack,
                blocks,
                inputs,
                bank,
                next_layer,
                next_depth,
                positions,
                tau,
                plan,
                coda_padding,
            )
        variant = (
            step.op,
            layers.start,
            layers.stop,
            depth,
            blocks,
            next_layer,
            next_depth,
            tuple(plan),
        )
        completed = blocks + sum(i % self.model.block_size == 0 for i in layers)
        output = step.graphs.run(
            variant,
            step,
            compute,
            stack,
            inputs,
            bank,
            positions,
            tau,
            coda_padding,
            stack_input_blocks=blocks + 1,
            stack_output_blocks=0 if step.op is LoopOp.CODA else completed + 1,
        )
        return completed, output

    def _run_layers(
        self,
        layers,
        depth,
        stack,
        blocks,
        inputs,
        bank,
        next_layer,
        next_depth,
        positions,
        tau,
        plan,
        coda_padding=None,
    ):
        model = self.model
        for layer in layers:
            following = layer + 1
            following_depth = depth
            if following == layers.stop:
                following, following_depth = next_layer, next_depth
            if following == len(model.layers):
                query, norm = model.attn_res.output_query, model.norm
            else:
                query = model.attn_res.attn_queries[following - 1]
                norm = None if following_depth else model.layers[following].attn_norm
            padding = (
                coda_padding if following == len(model.layers) and layer + 1 == following else None
            )
            token = _coda_tail_padding.set(padding) if padding is not None else None
            try:
                blocks, inputs = model._run_layer(
                    layer, depth, stack, blocks, inputs, bank, query, norm, positions, tau, plan
                )
            finally:
                if token is not None:
                    _coda_tail_padding.reset(token)
        return blocks, inputs
