"""Build one-logit MRv2 inputs for ordinary loop batches."""

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch


@triton.jit
def _prepare_rows(
    Idx,
    Offsets,
    Computed,
    Prefill,
    TokenIds,
    LastSampled,
    NextPrefill,
    InputIds,
    Positions,
    SeqLens,
    QueryOffsets,
    LogitRows,
    Padding,
    BlockTablePtrs,
    BlockTableStrides,
    BlockSizes,
    SlotEnabled,
    SlotMappings,
    actual: tl.int32,
    padded: tl.int32,
    requests: tl.int32,
    max_requests: tl.constexpr,
    history_stride: tl.constexpr,
    next_stride: tl.constexpr,
    next_count: tl.constexpr,
    next_block: tl.constexpr,
    request_block: tl.constexpr,
    tile: tl.constexpr,
    group_count: tl.constexpr,
    group_tile: tl.constexpr,
    slot_stride: tl.constexpr,
    slot_capacity: tl.constexpr,
    pad_slot: tl.constexpr,
    span_positions: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    local = chunk * tile + tl.arange(0, tile)
    if row == requests:
        if group_count and chunk == 0:
            groups = tl.arange(0, group_tile)
            for begin in tl.range(actual, slot_capacity, tile):
                column = begin + tl.arange(0, tile)
                tl.store(
                    SlotMappings + groups[:, None] * slot_stride + column[None, :],
                    pad_slot,
                    (groups[:, None] < group_count) & (column[None, :] < slot_capacity),
                )
        trailing = actual + local
        tl.store(InputIds + trailing, 0, trailing < padded)
        tl.store(Positions + trailing, 0, trailing < padded)
        tl.store(Padding + trailing, True, trailing < padded)
        if chunk == 0:
            slots = requests + tl.arange(0, request_block)
            tl.store(SeqLens + slots, 0, slots < max_requests)
            tl.store(QueryOffsets + slots, actual, slots <= max_requests)
        return

    start = tl.load(Offsets + row).to(tl.int32)
    end = tl.load(Offsets + row + 1).to(tl.int32)
    if start + chunk * tile >= end:
        return
    slot = tl.load(Idx + row).to(tl.int32)
    position = tl.load(Computed + (row if span_positions else slot)).to(tl.int32)
    prompt_end = tl.load(Prefill + slot)
    if chunk == 0:
        tl.store(SeqLens + row, position + end - start)
        tl.store(QueryOffsets + row, start)
        tl.store(LogitRows + row, end - 1)
        if group_count:
            groups = tl.arange(0, group_tile)
            valid_group = groups < group_count
            table = tl.load(BlockTablePtrs + groups, valid_group, other=0).to(
                tl.pointer_type(tl.int32)
            )
            stride = tl.load(BlockTableStrides + groups, valid_group, other=0)
            size = tl.load(BlockSizes + groups, valid_group, other=1)
            enabled = tl.load(SlotEnabled + groups, valid_group, other=False)
            block = tl.load(
                table + slot * stride + position // size, valid_group & enabled, other=0
            )
            current_slot = tl.where(enabled, block.to(tl.int64) * size + position % size, pad_slot)
            tl.store(SlotMappings + groups * slot_stride + start, current_slot, valid_group)

    qpos = position + local
    valid = start + local < end
    if position < prompt_end:
        history = TokenIds + slot.to(tl.int64) * history_stride
        tokens = tl.load(history + qpos, valid, other=0)
        if not span_positions and chunk == 0:
            next_rows = tl.arange(0, next_block)
            next_pos = position + end - start + next_rows
            next_token = tl.load(
                history + next_pos, (next_rows < next_count) & (next_pos < prompt_end), other=0
            )
            tl.store(
                NextPrefill + next_rows * next_stride + slot, next_token, next_rows < next_count
            )
    else:
        tokens = tl.full((tile,), 0, tl.int32) + tl.load(LastSampled + slot).to(tl.int32)
    tl.store(InputIds + start + local, tokens, valid)
    tl.store(Positions + start + local, qpos.to(tl.int64), valid)
    tl.store(Padding + start + local, False, valid)


class OrdinaryInputBuilder:
    def __init__(self, input_buffers, req_states, block_tables):
        self.buffers = input_buffers
        self.requests = req_states
        self.block_tables = block_tables
        size, device = input_buffers.max_num_reqs, input_buffers.device
        self.logit_rows = torch.empty(size, device=device, dtype=torch.int64)
        self.logit_counts = torch.arange(size + 1, device=device, dtype=torch.int32)
        self.local_pos = torch.zeros(size, device=device, dtype=torch.int32)
        self.cpu_counts = np.arange(size + 1, dtype=np.int32)

    def prepare(self, output, state, descriptor, *, prepare_decode_slots=False, positions=None):
        num_reqs = len(state.req_ids)
        num_tokens = state.num_tokens
        padded = max(num_tokens, descriptor.num_tokens)
        num_reqs_padded = descriptor.num_reqs or num_reqs
        mapping = state.idx_mapping_np
        counts = state.num_scheduled_tokens
        buf = self.buffers
        req = self.requests

        cpu_offsets = np.empty(buf.max_num_reqs + 1, dtype=np.int32)
        cpu_offsets[0] = 0
        np.cumsum(counts, out=cpu_offsets[1 : num_reqs + 1])
        cpu_offsets[num_reqs + 1 :] = num_tokens
        pack = np.empty((2 if positions is None else 3, buf.max_num_reqs + 1), dtype=np.int64)
        pack[0, :num_reqs] = mapping
        pack[0, num_reqs:] = 0
        pack[1] = cpu_offsets
        if positions is not None:
            pack[2, :num_reqs] = positions
        staged = async_tensor_h2d(pack, device=buf.device)
        indices, offsets = staged[0], staged[1]
        token_tile = 256
        max_length = max(int(counts.max()), padded - num_tokens)
        tables = self.block_tables
        table_args = (
            (
                tables.block_table_ptrs,
                tables.block_table_strides,
                tables.block_sizes_tensor,
                tables.slot_mapping_enabled,
                tables.slot_mappings,
            )
            if prepare_decode_slots
            else (buf.positions,) * 5
        )
        groups = tables.num_kv_cache_groups if prepare_decode_slots else 0
        _prepare_rows[(num_reqs + 1, triton.cdiv(max_length, token_tile))](
            indices,
            offsets,
            req.num_computed_tokens.gpu if positions is None else staged[2],
            req.prefill_len.gpu,
            req.all_token_ids.gpu,
            req.last_sampled_tokens,
            req.next_prefill_tokens,
            buf.input_ids,
            buf.positions,
            buf.seq_lens,
            buf.query_start_loc,
            self.logit_rows,
            buf.is_padding,
            *table_args,
            num_tokens,
            padded,
            num_reqs,
            buf.max_num_reqs,
            req.all_token_ids.gpu.stride(0),
            req.next_prefill_tokens.stride(0),
            req.next_prefill_tokens.shape[0],
            triton.next_power_of_2(req.next_prefill_tokens.shape[0]),
            triton.next_power_of_2(buf.max_num_reqs + 1),
            token_tile,
            groups,
            triton.next_power_of_2(max(groups, 1)),
            tables.slot_mappings.stride(0) if groups else 0,
            tables.slot_mappings.shape[1] if groups else 0,
            PAD_SLOT_ID,
            positions is not None,
        )

        computed = req.num_computed_tokens_np[mapping] if positions is None else positions
        seq_lens_cpu = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(computed, counts, out=seq_lens_cpu[:num_reqs])
        indices = indices[:num_reqs]
        return InputBatch(
            req_ids=state.req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=indices,
            idx_mapping_np=mapping,
            expanded_idx_mapping=indices,
            expanded_local_pos=self.local_pos[:num_reqs],
            num_scheduled_tokens=counts,
            num_tokens=num_tokens,
            num_tokens_after_padding=padded,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=buf.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_np=cpu_offsets[: num_reqs_padded + 1],
            seq_lens=buf.seq_lens[:num_reqs_padded],
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=computed,
            prefill_len_np=state.prefill_len_np,
            num_computed_prefill_tokens_np=state.num_computed_prefill_tokens_np,
            is_prefilling_np=state.is_prefilling_np,
            has_prefill=state.has_prefill,
            decode_graph_eligible=state.decode_graph_eligible,
            input_ids=buf.input_ids[:padded],
            positions=buf.positions[:padded],
            is_padding=buf.is_padding[:padded],
            logits_indices=self.logit_rows[:num_reqs],
            cu_num_logits=self.logit_counts[: num_reqs + 1],
            cu_num_logits_np=self.cpu_counts[: num_reqs + 1],
            has_structured_output_reqs=output.has_structured_output_requests,
            prompt_lens=None,
            prefill_runs_as_decode_np=state.prefill_runs_as_decode_np,
            max_seq_len_np=state.max_seq_len_np,
        )
