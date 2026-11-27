"""Execute shared Inkling CORE weights for spans at different depths."""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from itertools import groupby
from operator import attrgetter

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.models.inkling2.nvidia.lora import add_linear_lora
from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
from vllm.models.inkling2.nvidia.ops.linear import linear
from vllm.utils.torch_utils import async_tensor_h2d

from vllm_rlt.models.inkling_core_routes import pack_core_inputs


@dataclass
class _CoreGroup:
    depth: int
    final: bool
    proposal: bool
    rows: torch.Tensor
    metadata: dict
    inputs: torch.Tensor
    bank: torch.Tensor | None


def _group_metadata(metadata, rows, size, ranges):
    active = torch.zeros(size, device=rows.device, dtype=torch.bool)
    active.index_fill_(0, rows, True)
    by_metadata = {}
    grouped = {}
    for name, value in metadata.items():
        if id(value) not in by_metadata:
            slots = value.slot_mapping
            updates = {"slot_mapping": slots.masked_fill(~active, -1)}
            if hasattr(value, "query_start"):
                conv_starts = value.query_start.clone()
                for start, end in ranges:
                    conv_starts[start:end].fill_(start)
                updates["query_start"] = conv_starts
            by_metadata[id(value)] = replace(value, **updates)
        grouped[name] = by_metadata[id(value)]
    return grouped


@contextmanager
def _attention_metadata(metadata):
    context = get_forward_context()
    previous = context.attn_metadata
    context.attn_metadata = metadata
    try:
        yield
    finally:
        context.attn_metadata = previous


def _gather_rows(values, groups, size):
    first = values[0]
    result = first.new_empty((*first.shape[:-2], size, first.shape[-1]))
    for group, value in zip(groups, values, strict=True):
        result.index_copy_(-2, group.rows, value)
    return result


def _run_mixed_layer(model, i, positions, tau, stack, blocks, groups):
    layer = model.layers[i]
    size = positions.numel()
    partial = stack[blocks]
    messages = [
        attn_res(
            group.bank[:1],
            group.bank[1],
            None,
            model.message_queries[i - model.loop_core.start],
            None,
            0.0,
            True,
        )
        if group.bank is not None
        else None
        for group in groups
    ]
    norm = layer.attn_norm
    weights = norm.weight.expand(3, -1) if layer.is_local else norm.weight
    attn_inputs = [
        group.inputs.expand(3, -1, -1)
        if message is None
        else layer.blend(
            slice(0, 3), group.inputs.expand(3, -1, -1), message[:3], weights, norm.eps
        )
        for group, message in zip(groups, messages, strict=True)
    ]
    xq, xk, xv = _gather_rows(attn_inputs, groups, size)
    attn = layer.attn
    q, r, gate, sinks, k, v = attn._project(xq, xk, xv)
    k_dynamic = attn.k_sconv.dynamic_taps(xk)
    v_dynamic = attn.v_sconv.dynamic_taps(xv)
    attn_result = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
    scratch = torch.empty_like(attn_result)
    for group in groups:
        with _attention_metadata(group.metadata):
            attn._attention(
                positions,
                q,
                r,
                gate,
                sinks,
                k,
                v,
                k_dynamic,
                v_dynamic,
                None if layer.is_local else tau,
                scratch,
                group.depth,
            )
        attn_result.index_copy_(0, group.rows, scratch.index_select(0, group.rows))
    projected = linear(attn_result, attn.wo_ud.weight)
    add_linear_lora(attn.wo_ud, projected, attn_result)
    projected = tensor_model_parallel_all_reduce(projected)

    first = i % model.block_size == 0
    attn_out = torch.empty_like(partial)
    scratch = torch.empty_like(partial)
    mlp_inputs = []
    for group, message in zip(groups, messages, strict=True):
        with _attention_metadata(group.metadata):
            mixed = layer.attn_sconv.conv_mix(
                projected,
                positions,
                scratch,
                stack[: blocks + first],
                None if first else partial,
                model.attn_res.mlp_queries[i],
                layer.mlp_norm if message is None else None,
                owner=layer.conv_states[group.depth],
            )
        attn_out.index_copy_(0, group.rows, scratch.index_select(0, group.rows))
        mixed = mixed.index_select(1, group.rows)
        if message is not None:
            norm = layer.mlp_norm
            mixed = layer.blend(slice(3, 4), mixed, message[3:], norm.weight, norm.eps)
        mlp_inputs.append(mixed)
    mlp_out = layer.mlp(_gather_rows(mlp_inputs, groups, size)[0])

    next_blocks = blocks + first
    next_partial = torch.empty_like(partial)
    for group in groups:
        following = i + 1
        following_depth = group.depth
        if following == model.loop_core.stop:
            if group.final or group.proposal:
                following_depth = 0
            else:
                following = model.loop_core.start
                following_depth += 1
        query = model.attn_res.attn_queries[following - 1]
        next_norm = None if following_depth else model.layers[following].attn_norm
        with _attention_metadata(group.metadata):
            mixed = layer.mlp_sconv.conv_mix(
                mlp_out,
                positions,
                scratch,
                stack[:next_blocks],
                None,
                query,
                next_norm,
                addend=attn_out,
                residual=None if first else partial,
                owner=layer.conv_states[group.depth],
            )
        next_partial.index_copy_(0, group.rows, scratch.index_select(0, group.rows))
        group.inputs = mixed.index_select(1, group.rows)
    stack[next_blocks].copy_(next_partial)
    return next_blocks


def execute_mixed_core(adapter, step, positions, tau):
    model = adapter.model
    metadata = get_forward_context().attn_metadata
    assert isinstance(metadata, dict)
    groups_by_key = defaultdict(list)
    states = []
    offset = 0
    for span in step.ordered:
        state = step.state[span.request_id].pop(span.token_start)
        end = offset + span.token_count
        key = (span.depth, span.last_core, span.propose_draft)
        groups_by_key[key].append((span, state, offset, end))
        states.append(state)
        offset = end
    assert offset == positions.numel()
    stack = torch.cat([state.stack for state in states], dim=1)
    blocks = states[0].num_blocks
    groups = []
    for key, spans in groups_by_key.items():
        ranges = [(start, end) for _, _, start, end in spans]
        row_ids = [row for start, end in ranges for row in range(start, end)]
        rows = torch.as_tensor(row_ids, device=positions.device)
        state = type(states[0]).concatenate([state for _, state, _, _ in spans])
        groups.append(
            _CoreGroup(
                *key,
                rows,
                _group_metadata(metadata, rows, offset, ranges),
                state.inputs,
                state.bank,
            )
        )

    for i in model.loop_core:
        blocks = _run_mixed_layer(model, i, positions, tau, stack, blocks, groups)

    for spans, group in zip(groups_by_key.values(), groups, strict=True):
        group_offset = 0
        for span, state, start, end in spans:
            group_end = group_offset + span.token_count
            state.stack = stack[:, start:end].clone()
            state.num_blocks = blocks
            state.inputs = group.inputs[:, group_offset:group_end].contiguous()
            if group.final:
                state.bank = None
            else:
                if group.proposal:
                    step.draft_state.setdefault(span.request_id, {})[span.token_start] = state.rows(
                        0, span.token_count
                    )
                state.bank = state.stack[state.prelude_blocks + 1 : blocks + 1].clone()
                state.num_blocks = state.prelude_blocks
                state.stack[state.num_blocks].copy_(state.prelude)
                state.inputs = attn_res(
                    state.stack[: state.num_blocks],
                    state.prelude,
                    None,
                    model.attn_res.attn_queries[model.loop_core.start - 1],
                    None,
                    0.0,
                    True,
                )
            step.state[span.request_id][span.token_start] = state
            group_offset = group_end
    return stack[blocks]


@dataclass
class _CompactCoreGroup:
    depth: int
    final: bool
    spans: tuple
    rows: slice
    metadata: dict
    state: object


@dataclass
class _CompactCoreExecution:
    groups: list[_CompactCoreGroup]
    positions: torch.Tensor
    tau: torch.Tensor | None
    stack: torch.Tensor
    blocks: int
    messages: list[torch.Tensor | None] | None = None
    attention: torch.Tensor | None = None


@dataclass
class _PackedCoreExecution:
    metadata: tuple
    positions: torch.Tensor
    tau: torch.Tensor | None
    stack: torch.Tensor
    blocks: int
    later_rows: torch.Tensor | None
    first_inputs: torch.Tensor | None
    later_inputs: torch.Tensor | None
    bank: torch.Tensor | None
    row_routes: torch.Tensor
    messages: torch.Tensor | None = None
    attention: torch.Tensor | None = None
    coda_inputs: torch.Tensor | None = None

    def rows(self, tensor, indices):
        if indices.shape[0] == self.positions.numel():
            return tensor
        return tensor.index_select(-2, indices)

_compact_core_execution = ContextVar("compact_core_execution", default=None)


def _prepare_packed_qkv(model, i, execution, qkv, inputs=None):
    layer = model.layers[i]
    norm = layer.attn_norm
    live = execution.positions.numel()
    first = execution.first_inputs if inputs is None else inputs
    mixed = None
    if execution.later_rows is not None:
        bank = execution.bank
        execution.messages = attn_res(
            bank[:1], bank[1], None,
            model.message_queries[i - model.loop_core.start], None, 0.0, True,
        )
        later = execution.later_inputs if inputs is None else execution.rows(
            inputs, execution.later_rows
        )
        weights = norm.weight.expand(3, -1) if layer.is_local else norm.weight
        mixed = layer.blend(
            slice(0, 3), later.expand(3, -1, -1), execution.messages[:3], weights, norm.eps
        )
    pack_core_inputs(
        qkv, first, mixed, execution.row_routes, live, norm,
        normalize_first=inputs is not None, first_is_full=inputs is not None,
    )


def _prepare_qkv(model, i, qkv):
    execution = _compact_core_execution.get()
    if execution is None:
        qkv.zero_()
        return
    if isinstance(execution, _PackedCoreExecution):
        _prepare_packed_qkv(model, i, execution, qkv)
        return
    layer = model.layers[i]
    execution.messages = []
    norm = layer.attn_norm
    weights = norm.weight.expand(3, -1) if layer.is_local else norm.weight
    for group in execution.groups:
        inputs = group.state.inputs.expand(3, -1, -1)
        bank = group.state.bank
        message = None
        if bank is not None:
            message = attn_res(
                bank[:1],
                bank[1],
                None,
                model.message_queries[i - model.loop_core.start],
                None,
                0.0,
                True,
            )
            inputs = layer.blend(slice(0, 3), inputs, message[:3], weights, norm.eps)
        qkv[:, group.rows].copy_(inputs)
        execution.messages.append(message)
    qkv[:, execution.stack.shape[1] :].zero_()


@eager_break_during_capture
def _start_compact_layers(model, qkv):
    _prepare_qkv(model, model.loop_core.start, qkv)


@eager_break_during_capture
def _attend(model, i, out, q, r, gate, sinks, k, v, k_dynamic, v_dynamic):
    execution = _compact_core_execution.get()
    if execution is None:
        out.zero_()
        return
    layer = model.layers[i]
    tau = execution.tau
    if isinstance(execution, _PackedCoreExecution):
        live = execution.positions.numel()
        attn, metadata = execution.metadata[i - model.loop_core.start]
        with _attention_metadata(metadata):
            attn._attention(
                execution.positions, q[:live], r[:live], gate[:live], sinks[:live],
                k[:live], v[:live], None if k_dynamic is None else k_dynamic[:live],
                None if v_dynamic is None else v_dynamic[:live],
                None if layer.is_local else tau, out[:live],
            )
        out[live:].zero_()
        return
    for group in execution.groups:
        rows = group.rows
        with _attention_metadata(group.metadata):
            layer.attn._attention(
                execution.positions[rows],
                q[rows],
                r[rows],
                gate[rows],
                sinks[rows],
                k[rows],
                v[rows],
                None if k_dynamic is None else k_dynamic[rows],
                None if v_dynamic is None else v_dynamic[rows],
                None if layer.is_local or tau is None else tau[rows],
                out[rows],
                group.depth,
            )
    out[execution.stack.shape[1] :].zero_()


@eager_break_during_capture
def _prepare_mlp(model, i, projected, mlp_inputs):
    execution = _compact_core_execution.get()
    if execution is None:
        mlp_inputs.zero_()
        return
    layer = model.layers[i]
    first = i % model.block_size == 0
    stack, blocks = execution.stack, execution.blocks
    partial = stack[blocks]
    execution.attention = torch.empty_like(partial)
    if isinstance(execution, _PackedCoreExecution):
        live = execution.positions.numel()
        attn, metadata = execution.metadata[i - model.loop_core.start]
        norm = layer.mlp_norm
        with _attention_metadata(metadata):
            mixed = layer.attn_sconv.conv_mix(
                projected[:live], execution.positions, execution.attention,
                stack[: blocks + first], None if first else partial,
                model.attn_res.mlp_queries[i], None if execution.later_rows is not None else norm,
                owner=attn.conv_states[0],
            )
        later_inputs = (
            layer.blend(
                slice(3, 4), execution.rows(mixed, execution.later_rows), execution.messages[3:],
                norm.weight, norm.eps,
            )
            if execution.later_rows is not None else None
        )
        pack_core_inputs(
            mlp_inputs.unsqueeze(0), mixed, later_inputs, execution.row_routes, live, norm,
            normalize_first=later_inputs is not None, first_is_full=True,
        )
        return
    assert execution.messages is not None
    for group, message in zip(execution.groups, execution.messages, strict=True):
        rows = group.rows
        with _attention_metadata(group.metadata):
            mixed = layer.attn_sconv.conv_mix(
                projected[rows],
                execution.positions[rows],
                execution.attention[rows],
                stack[: blocks + first, rows],
                None if first else partial[rows],
                model.attn_res.mlp_queries[i],
                layer.mlp_norm if message is None else None,
                owner=layer.conv_states[group.depth],
            )
        if message is not None:
            norm = layer.mlp_norm
            mixed = layer.blend(slice(3, 4), mixed, message[3:], norm.weight, norm.eps)
        mlp_inputs[rows].copy_(mixed[0])
    mlp_inputs[stack.shape[1] :].zero_()


@eager_break_during_capture
def _finish_compact_layer(model, i, mlp_out, qkv):
    execution = _compact_core_execution.get()
    if execution is None:
        return
    layer = model.layers[i]
    first = i % model.block_size == 0
    stack, blocks = execution.stack, execution.blocks
    partial = stack[blocks]
    next_blocks = blocks + first
    assert execution.attention is not None
    if isinstance(execution, _PackedCoreExecution):
        following = i + 1
        last_layer = following == model.loop_core.stop
        attn, metadata = execution.metadata[i - model.loop_core.start]
        with _attention_metadata(metadata):
            inputs = layer.mlp_sconv.conv_mix(
                mlp_out[: execution.positions.numel()], execution.positions, stack[next_blocks],
                stack[:next_blocks], None, model.attn_res.attn_queries[i],
                model.layers[following].attn_norm if last_layer else None,
                addend=execution.attention, residual=None if first else partial,
                owner=attn.conv_states[0],
            )
        execution.blocks = next_blocks
        if last_layer:
            execution.coda_inputs = inputs
        else:
            _prepare_packed_qkv(model, following, execution, qkv, inputs)
        return
    for group in execution.groups:
        rows = group.rows
        following = i + 1
        following_depth = group.depth
        if following == model.loop_core.stop:
            following_depth = 0 if group.final else group.depth + 1
            if not group.final:
                following = model.loop_core.start
        query = model.attn_res.attn_queries[following - 1]
        next_norm = None if following_depth else model.layers[following].attn_norm
        with _attention_metadata(group.metadata):
            group.state.inputs = layer.mlp_sconv.conv_mix(
                mlp_out[rows],
                execution.positions[rows],
                stack[next_blocks, rows],
                stack[:next_blocks, rows],
                None,
                query,
                next_norm,
                addend=execution.attention[rows],
                residual=None if first else partial[rows],
                owner=layer.conv_states[group.depth],
            )
    execution.blocks = next_blocks
    if i + 1 < model.loop_core.stop:
        _prepare_qkv(model, i + 1, qkv)


def run_compact_layers(model, qkv_inputs, mlp_inputs):
    # The graph is keyed only by row width. Depth-specific KV and continuation
    # routing read the current execution inside the breakable graph callbacks.
    _start_compact_layers(model, qkv_inputs)
    for i in model.loop_core:
        layer = model.layers[i]
        attn = layer.attn
        xq, xk, xv = qkv_inputs.unbind()
        q, r, gate, sinks, k, v = attn._project(xq, xk, xv)
        k_dynamic = attn.k_sconv.dynamic_taps(xk)
        v_dynamic = attn.v_sconv.dynamic_taps(xv)
        out = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        _attend(model, i, out, q, r, gate, sinks, k, v, k_dynamic, v_dynamic)
        projected = linear(out, attn.wo_ud.weight)
        _prepare_mlp(model, i, projected, mlp_inputs)
        _finish_compact_layer(model, i, layer.mlp(mlp_inputs), qkv_inputs)
    return qkv_inputs


def _run_compact_core(model, step, positions, execution):
    token = _compact_core_execution.set(execution)
    try:
        if step.packed_core and step.graphs is not None:
            if step.graphs.replay_packed_core(positions, execution):
                return
        graph = step.graphs is not None and step.graphs.replay_compact_core(positions.numel())
        if not graph:
            width, hidden = positions.numel(), execution.stack.shape[-1]
            run_compact_layers(
                model, execution.stack.new_empty((3, width, hidden)),
                execution.stack.new_empty((width, hidden)),
            )
    finally:
        _compact_core_execution.reset(token)


def _execute_packed_core(adapter, step, positions, tau):
    states = adapter._pop_core_states(step, step.ordered)
    first = states[0]
    live = sum(span.token_count for span in step.ordered)
    stack = first.stack.new_empty((first.stack.shape[0], live, first.stack.shape[2]))
    blocks = first.num_blocks + 1
    torch.cat([state.stack[:blocks] for state in states], dim=1, out=stack[:blocks])
    later_rows, row_routes = [], []
    first_inputs, later_inputs, banks = [], [], []
    offset = first_count = 0
    for span, state in zip(step.ordered, states, strict=True):
        end = offset + span.token_count
        if state.bank is None:
            row_routes.extend(range(-first_count - 1, -first_count - span.token_count - 1, -1))
            first_count += span.token_count
            first_inputs.append(state.inputs)
        else:
            row_routes.extend(range(len(later_rows), len(later_rows) + span.token_count))
            later_rows.extend(range(offset, end))
            later_inputs.append(state.inputs)
            banks.append(state.bank)
        offset = end
    assert offset == live
    rows = async_tensor_h2d(later_rows + row_routes, dtype=torch.long, device=positions.device)
    execution = _PackedCoreExecution(
        step.packed_core_metadata,
        positions[:live],
        None if tau is None else tau[:live],
        stack,
        first.num_blocks,
        rows[:len(later_rows)] if later_rows else None,
        torch.cat(first_inputs, dim=1) if first_inputs else None,
        torch.cat(later_inputs, dim=1) if later_inputs else None,
        torch.cat(banks, dim=1) if banks else None,
        rows[len(later_rows):],
    )
    _run_compact_core(adapter.model, step, positions, execution)

    offset = 0
    for span, state in zip(step.ordered, states, strict=True):
        end = offset + span.token_count
        state.stack = stack[:, offset:end]
        state.num_blocks = execution.blocks
        if span.last_core:
            state.bank = None
            state.inputs = execution.coda_inputs[:, offset:end]
        else:
            previous_inputs = state.inputs if state.bank is not None else None
            adapter._continue_core(state, previous_inputs)
        adapter._store_span_state(step, span, state)
        offset = end
    return stack[execution.blocks]


def execute_compact_core(adapter, step, positions, tau):
    """Run shared CORE weights, using one packed attention and convolution when supported."""
    if step.packed_core:
        return _execute_packed_core(adapter, step, positions, tau)
    groups = []
    offset = 0
    variants = groupby(step.ordered, attrgetter("depth", "last_core"))
    for ((depth, final), spans), metadata in zip(variants, step.core_group_metadata, strict=True):
        spans = tuple(spans)
        states = adapter._pop_core_states(step, spans)
        state = type(states[0]).concatenate(states)
        end = offset + sum(span.token_count for span in spans)
        groups.append(_CompactCoreGroup(depth, final, spans, slice(offset, end), metadata, state))
        offset = end
    assert offset == sum(span.token_count for span in step.ordered)
    stack = torch.cat([group.state.stack for group in groups], dim=1)
    execution = _CompactCoreExecution(groups, positions, tau, stack, groups[0].state.num_blocks)
    _run_compact_core(adapter.model, step, positions, execution)
    blocks = execution.blocks

    for group in groups:
        state = group.state
        state.stack = stack[:, group.rows]
        state.num_blocks = blocks
        state.bank = None
        if not group.final:
            adapter._continue_core(state)
        adapter._store_state_rows(step, state, group.spans, copy=False)
    return stack[blocks]
