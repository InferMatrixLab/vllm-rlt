"""Execute shared Inkling CORE weights for spans at different depths."""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import groupby
from operator import attrgetter

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.models.inkling2.nvidia.lora import add_linear_lora
from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
from vllm.models.inkling2.nvidia.ops.linear import linear


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


def _compact_core_layer(model, i, positions, tau, stack, blocks, groups):
    layer = model.layers[i]
    messages = []
    attn_inputs = []
    norm = layer.attn_norm
    weights = norm.weight.expand(3, -1) if layer.is_local else norm.weight
    for group in groups:
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
        attn_inputs.append(inputs)
        messages.append(message)

    xq, xk, xv = torch.cat(attn_inputs, dim=1).unbind()
    attn = layer.attn
    q, r, gate, sinks, k, v = attn._project(xq, xk, xv)
    k_dynamic = attn.k_sconv.dynamic_taps(xk)
    v_dynamic = attn.v_sconv.dynamic_taps(xv)
    out = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
    for group in groups:
        rows = group.rows
        with _attention_metadata(group.metadata):
            attn._attention(
                positions[rows],
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
    projected = linear(out, attn.wo_ud.weight)

    first = i % model.block_size == 0
    partial = stack[blocks]
    attn_out = torch.empty_like(partial)
    mlp_inputs = []
    for group, message in zip(groups, messages, strict=True):
        rows = group.rows
        with _attention_metadata(group.metadata):
            mixed = layer.attn_sconv.conv_mix(
                projected[rows],
                positions[rows],
                attn_out[rows],
                stack[: blocks + first, rows],
                None if first else partial[rows],
                model.attn_res.mlp_queries[i],
                layer.mlp_norm if message is None else None,
                owner=layer.conv_states[group.depth],
            )
        if message is not None:
            norm = layer.mlp_norm
            mixed = layer.blend(slice(3, 4), mixed, message[3:], norm.weight, norm.eps)
        mlp_inputs.append(mixed[0])
    mlp_out = layer.mlp(torch.cat(mlp_inputs))

    next_blocks = blocks + first
    for group in groups:
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
                positions[rows],
                stack[next_blocks, rows],
                stack[:next_blocks, rows],
                None,
                query,
                next_norm,
                addend=attn_out[rows],
                residual=None if first else partial[rows],
                owner=layer.conv_states[group.depth],
            )
    return next_blocks


def execute_compact_core(adapter, step, positions, tau):
    """Use each depth's real KV rows and run the shared weights before/after them."""
    groups = []
    offset = 0
    variants = groupby(step.ordered, attrgetter("depth", "last_core"))
    for ((depth, final), spans), metadata in zip(variants, step.core_group_metadata, strict=True):
        spans = tuple(spans)
        states = [step.state[span.request_id].pop(span.token_start) for span in spans]
        state = type(states[0]).concatenate(states)
        end = offset + sum(span.token_count for span in spans)
        groups.append(_CompactCoreGroup(depth, final, spans, slice(offset, end), metadata, state))
        offset = end
    assert offset == positions.numel()
    stack = torch.cat([group.state.stack for group in groups], dim=1)
    blocks = groups[0].state.num_blocks
    for i in adapter.model.loop_core:
        blocks = _compact_core_layer(adapter.model, i, positions, tau, stack, blocks, groups)

    for group in groups:
        state = group.state
        state.stack = stack[:, group.rows]
        state.num_blocks = blocks
        state.bank = None
        if not group.final:
            adapter._continue_core(state)
        state.store_rows(group.spans, step.state, copy=False)
    return stack[blocks]
