"""Execute loop steps through the existing MRv2 worker and input builders."""

from collections import defaultdict, deque
from dataclasses import replace

import numpy as np
import torch
from vllm import envs
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.utils.gc_utils import freeze_gc_for_cudagraph_capture
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsLists, ModelRunnerOutput
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.model_runner import BatchReqState, GPUModelRunner
from vllm.v1.worker.workspace import lock_workspace

from vllm_rlt.config import LoopRuntimeConfig, supports_prefill_pipeline
from vllm_rlt.core.loop import LoopLayout, LoopOp, LoopSpan
from vllm_rlt.core.output import LoopModelRunnerOutput, LoopSchedulerOutput
from vllm_rlt.worker.loop_graph import LoopCudaGraphs
from vllm_rlt.worker.loop_state import LoopExecution, PreparedBatch, loop_execution
from vllm_rlt.worker.ordinary_inputs import OrdinaryInputBuilder

logger = init_logger("vllm.rlt.loop_model_runner")


class LoopModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device, graph_mode):
        super().__init__(vllm_config, device)
        self._loop_graphs = (
            LoopCudaGraphs(vllm_config, graph_mode)
            if graph_mode not in (None, CUDAGraphMode.NONE)
            else None
        )
        self._loop_states: dict[str, dict[int, object]] = {}
        self._draft_states: dict[str, dict[int, object]] = {}
        self._loop_completion: ModelRunnerOutput | None = None
        self._prepared_batch: PreparedBatch | None = None
        self._uva_capacity = max(2, vllm_config.max_concurrent_batches)
        self._uva_pending: deque[torch.cuda.Event] = deque()
        self._uva_current: torch.cuda.Event | None = None
        self._uva_current_recorded = False
        options = LoopRuntimeConfig.from_additional_config(vllm_config.additional_config)
        self._early_uva_release = (
            self.parallel_config.world_size == 1
            and vllm_config.kv_transfer_config is None
            and vllm_config.speculative_config is None
            and vllm_config.lora_config is None
            and not options.self_speculation
        )
        self._coda_tail_moe = (
            envs.VLLM_MOE_SKIP_PADDING
            and self._early_uva_release
            and not vllm_config.cache_config.enable_prefix_caching
            and not getattr(self.model_config, "enable_return_routed_experts", False)
            and not options.decode_pipeline
        )
        self._trace_steps = options.trace_steps
        self._mixed_core = options.mixed_core
        self._compact_mixed_core = options.compact_mixed_core
        self._packed_prefill = options.packed_prefill
        self._decode_pipeline = options.decode_pipeline
        self._prefill_pipeline = supports_prefill_pipeline(vllm_config, options)
        if self._prefill_pipeline or self._decode_pipeline:
            # MRv2 already uses its own write pool when admitting requests.
            # These pools rotate at most once per physical forward, matching
            # the lifetime tracked by _submit_forward.
            self._loop_position_slots = UvaBufferPool(
                self.max_num_reqs, torch.int64, self._uva_capacity
            )
            self._loop_positions = UvaBufferPool(
                self.max_num_reqs, torch.int32, self._uva_capacity
            )
        self._merge_decode_core = options.merge_decode_core
        self._merge_prefill_core = options.merge_prefill_core or self._prefill_pipeline
        self._decode_state_slots = options.decode_state_slots
        if (self._mixed_core or self._packed_prefill) and (
            self.parallel_config.tensor_parallel_size != 1
            or getattr(self.model_config, "enable_return_routed_experts", False)
        ):
            raise ValueError(
                "RLT CORE packing requires TP=1 with routed-experts reporting disabled"
            )

    def _needs_loop_graph_capture(self):
        graphs = self._loop_graphs
        return (
            graphs is not None
            and bool(graphs.sizes)
            and not graphs.warmed_up
            and getattr(self.get_model(), "rlt_graph_warmup", False)
        )

    def needs_cudagraph_capture(self):
        return self._needs_loop_graph_capture() or super().needs_cudagraph_capture()

    @torch.inference_mode()
    def capture_model(self, *, profile_only=False):
        self._clear_prepared_batch()
        if not self._needs_loop_graph_capture():
            return super().capture_model(profile_only=profile_only)
        memory = 0
        if super().needs_cudagraph_capture():
            memory += super().capture_model(profile_only=profile_only)
        assert self._loop_graphs is not None
        layout = LoopLayout.from_model_config(self.model_config)
        with freeze_gc_for_cudagraph_capture():
            torch.accelerator.empty_cache()
            free_before = torch.accelerator.get_memory_info()[0]
            try:
                self._loop_graphs.warmup(self, layout, profile_only=profile_only)
                torch.accelerator.empty_cache()
                memory += max(0, free_before - torch.accelerator.get_memory_info()[0])
            finally:
                if profile_only:
                    # vLLM releases the temporary KV cache and graph pool next.
                    self._loop_graphs.entries.clear()
                    self._loop_graphs.compact_core_graphs.clear()
                    self._loop_graphs.packed_core_graphs.clear()
        if not profile_only:
            lock_workspace()
        return memory

    def update_requests(self, scheduler_output):
        super().update_requests(scheduler_output)
        step = loop_execution.get()
        if step is None or not (step.explicit_positions or self._decode_pipeline):
            return
        req_states = self.req_states
        starts = {}
        for span in step.spans:
            # Gather shared block tables with the earliest live position for
            # each request. A packed CORE uses a separate start for every row.
            starts[span.request_id] = min(starts.get(span.request_id, span.token_start),
                                          span.token_start)
        slots = []
        positions = []
        for request_id, start in starts.items():
            slot = req_states.req_id_to_index[request_id]
            req_states.num_computed_tokens_np[slot] = start
            req_states.num_computed_prefill_tokens[slot] = min(
                start, req_states.prefill_len.np[slot]
            )
            slots.append(slot)
            positions.append(start)
        req_states.num_computed_tokens.gpu.index_copy_(
            0,
            self._loop_position_slots.copy_to_uva(slots),
            self._loop_positions.copy_to_uva(positions),
        )

    def gather_batch_req_state(self, scheduler_output, dummy_run):
        step = loop_execution.get()
        if (
            dummy_run or step is None or not step.compact_core
            or not step.explicit_positions
        ):
            return super().gather_batch_req_state(scheduler_output, dummy_run)

        model_state = getattr(self, "model_state", None)
        can_pack = getattr(model_state, "can_pack_core", None)
        step.packed_core = can_pack is not None and can_pack()
        if step.packed_core:
            # Inkling's native metadata expects one-token and prefill queries
            # in this order. Pass selection is independent of query ordering.
            step.ordered = tuple(sorted(step.spans, key=lambda span: span.token_count))
        else:
            step.ordered = tuple(sorted(step.spans, key=lambda span: (
                span.depth, span.last_core, span.token_count != self.decode_query_len,
                span.token_count,
            )))
        req_ids = [span.request_id for span in step.ordered]
        states = self.req_states
        mapping = np.asarray([states.req_id_to_index[req_id] for req_id in req_ids], dtype=np.intp)
        counts = np.asarray([span.token_count for span in step.ordered], dtype=np.int32)
        positions = np.asarray([span.token_start for span in step.ordered], dtype=np.int32)
        prefill_lens = states.prefill_len.np[mapping]
        prefilling = positions < prefill_lens
        return BatchReqState(
            req_ids=req_ids,
            num_scheduled_tokens=counts,
            num_tokens=scheduler_output.total_num_scheduled_tokens,
            num_draft_tokens_np=None,
            idx_mapping_np=mapping,
            prefill_len_np=prefill_lens,
            num_computed_prefill_tokens_np=np.minimum(positions, prefill_lens),
            is_prefilling_np=prefilling,
            max_seq_len_np=None,
            has_prefill=bool(prefilling.any()),
            prefill_runs_as_decode_np=prefilling & (counts == 1) & (positions > 0),
            decode_graph_eligible=False,
        ), None

    def _input_builder(self):
        builder = getattr(self, "_ordinary_inputs", None)
        if builder is None:
            builder = self._ordinary_inputs = OrdinaryInputBuilder(
                self.input_buffers, self.req_states, self.block_tables
            )
        return builder

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc, num_active_loras):
        step = loop_execution.get()
        explicit_positions = step is not None and step.explicit_positions
        if step is not None and step.compact_core and explicit_positions:
            self._clear_prepared_batch()
            positions = np.asarray([span.token_start for span in step.ordered], dtype=np.int32)
            real_tokens = batch_req_state.num_tokens
            graphs = self._loop_graphs
            width = graphs.padded_piecewise_size(real_tokens) if graphs is not None else None
            if width is not None and graphs.has_compact_core_graph(width, step.packed_core):
                if width > real_tokens:
                    step.padded_tokens = width
            else:
                width = real_tokens
            # The upstream descriptor counts unique request IDs. A packed
            # batch owns one query row per span, including repeated IDs.
            batch_desc = replace(batch_desc, num_reqs=None, num_tokens=width)
            return self._input_builder().prepare(
                scheduler_output, batch_req_state, batch_desc, positions=positions
            )
        if step is not None and step.compact_core:
            # Preserve MRv2's decode/prefill ordering within each variant. Its
            # native slot and block tables then have contiguous sub-batch views.
            variants = {span.request_id: (span.depth, span.last_core) for span in step.spans}
            order = sorted(
                range(len(batch_req_state.req_ids)),
                key=lambda index: variants[batch_req_state.req_ids[index]],
            )
            batch_req_state = batch_req_state._replace(
                req_ids=[batch_req_state.req_ids[index] for index in order],
                num_scheduled_tokens=batch_req_state.num_scheduled_tokens[order],
                idx_mapping_np=batch_req_state.idx_mapping_np[order],
                prefill_len_np=batch_req_state.prefill_len_np[order],
                num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np[
                    order
                ],
                is_prefilling_np=batch_req_state.is_prefilling_np[order],
                prefill_runs_as_decode_np=(
                    None
                    if batch_req_state.prefill_runs_as_decode_np is None
                    else batch_req_state.prefill_runs_as_decode_np[order]
                ),
            )
        rows = self._input_rows(step, batch_req_state, num_active_loras)
        if (
            rows is not None
            and self._loop_graphs is not None
            and self.parallel_config.tensor_parallel_size == 1
        ):
            real_tokens = batch_req_state.num_tokens
            uniform_decode = real_tokens == len(rows) and not any(
                span.prefill for span in step.spans
            )
            step.uniform_decode = uniform_decode
            graphs = self._loop_graphs
            mode = graphs.mode.decode_mode() if uniform_decode else graphs.mode.mixed_mode()
            if mode is CUDAGraphMode.FULL and uniform_decode and not batch_req_state.has_prefill:
                tokens = graphs.padded_decode_size(real_tokens)
            elif mode is CUDAGraphMode.PIECEWISE:
                tokens = graphs.padded_piecewise_size(real_tokens)
            else:
                tokens = None
            if tokens is not None and tokens > real_tokens:
                step.padded_tokens = tokens
                step.padded_graph_mode = mode
                batch_desc = replace(
                    batch_desc,
                    cg_mode=mode,
                    num_reqs=tokens if mode is CUDAGraphMode.FULL else None,
                    num_tokens=tokens,
                    uniform_token_count=1 if mode is CUDAGraphMode.FULL else None,
                    max_query_len=1 if mode is CUDAGraphMode.FULL else None,
                )
        prepared = self._prepared_batch
        if (
            rows is not None
            and prepared is not None
            and (prepared.rows == rows and prepared.descriptor == batch_desc)
            and prepared.keep_model_state == (not explicit_positions)
        ):
            order = {req_id: index for index, req_id in enumerate(prepared.inputs.req_ids)}
            step.ordered = tuple(sorted(step.spans, key=lambda span: order[span.request_id]))
            step.prepared_batch = prepared
            return prepared.inputs

        self._clear_prepared_batch()
        prepare_decode_slots = False
        if (
            rows is not None
            and not explicit_positions
            and getattr(self, "_early_uva_release", False)
            and self.parallel_config.world_size == 1
            and self.fast_prefill is None
            and self.model_config.rswa_window is None
            and self.model_state.num_new_sampled_tokens_per_step == 1
            and self._ordinary_input_spans(rows, batch_req_state)
        ):
            prepare_decode_slots = (
                step.uniform_decode is True
                and not batch_req_state.has_prefill
                and self.pcp_manager is None
                and self.block_tables.cp_size == 1
                and self.block_tables.num_kv_cache_groups > 0
            )
            batch = self._input_builder().prepare(
                scheduler_output,
                batch_req_state,
                batch_desc,
                prepare_decode_slots=prepare_decode_slots,
            )
        else:
            batch = super().prepare_inputs(
                scheduler_output, batch_req_state, batch_desc, num_active_loras
            )
            if step is not None and step.padded_tokens is not None:
                batch.input_ids[batch.num_tokens :].zero_()
                batch.positions[batch.num_tokens :].zero_()
        self._prepared_batch = (
            PreparedBatch(
                rows,
                batch_desc,
                batch,
                decode_slots_prepared=prepare_decode_slots,
                # Overlapping chunks keep activations by request and token position while
                # borrowing inputs only for an identical physical batch.
                keep_model_state=not explicit_positions,
            )
            if rows is not None
            else None
        )
        if step is not None:
            step.prepared_batch = self._prepared_batch
            by_id = defaultdict(list)
            for span in step.spans:
                by_id[span.request_id].append(span)
            step.ordered = tuple(span for request_id in batch.req_ids for span in by_id[request_id])
            offset = 0
            for span in step.ordered:
                if span.input_token is not None:
                    batch.input_ids[offset] = span.input_token
                offset += span.token_count
        return batch

    @staticmethod
    def _ordinary_input_spans(rows, state):
        return all(
            start + count <= prompt_end if prefilling else count == 1
            for (_, _, start, count, prefilling), prompt_end in zip(
                rows, state.prefill_len_np, strict=True
            )
        )

    def _supports_input_layout(self, spans):
        return (
            not self._decode_pipeline
            and not getattr(self.model, "rlt_core_samples", False)
            and self.pcp_manager is None
            and self.dcp_size == 1
            and self.ubatch_runner is None
            and self.speculator is None
            and not any(
                span.input_token is not None or span.propose_draft or span.draft_coda
                for span in spans
            )
        )

    def _input_rows(self, step, batch_req_state, num_active_loras):
        if (
            step is None
            or step.compact_core
            or num_active_loras
            or batch_req_state.num_draft_tokens_np is not None
            or (batch_req_state.has_prefill and self.fast_prefill is not None)
            or not self._supports_input_layout(step.spans)
            or (
                step.op is LoopOp.CORE
                and not step.merged_core
                and len({(span.depth, span.last_core) for span in step.spans}) > 1
            )
        ):
            return None
        # MRv2 uses one input row for a contiguous prelude, when the
        # scheduler prepares several independently scheduled chunks.
        prelude = step.explicit_positions and step.op is LoopOp.PRELUDE
        by_id = {}
        for span in step.spans:
            previous = by_id.get(span.request_id)
            if previous is not None:
                if not (prelude and previous.prefill and span.prefill):
                    return None
                if previous.token_end != span.token_start:
                    return None
                span = replace(previous, token_count=previous.token_count + span.token_count)
            by_id[span.request_id] = span
        if len(by_id) != len(batch_req_state.req_ids):
            return None
        computed_tokens = self.req_states.num_computed_tokens_np
        rows = []
        for req_id, slot, count, prefilling in zip(
            batch_req_state.req_ids,
            batch_req_state.idx_mapping_np,
            batch_req_state.num_scheduled_tokens,
            batch_req_state.is_prefilling_np,
            strict=True,
        ):
            span = by_id.get(req_id)
            if (
                span is None
                or span.token_start != computed_tokens[slot]
                or span.token_count != count
            ):
                return None
            rows.append((req_id, int(slot), span.token_start, span.token_count, bool(prefilling)))
        return tuple(rows)

    def prepare_attn(self, input_batch):
        prepared = self._prepared_batch
        if prepared is None or prepared.inputs is not input_batch:
            return super().prepare_attn(input_batch)
        if prepared.attention is None:
            if prepared.decode_slots_prepared:
                tables = self.block_tables
                prepared.attention = (
                    tables.gather_block_tables(
                        input_batch.idx_mapping,
                        num_reqs_padded=input_batch.num_reqs_after_padding,
                        num_computed_tokens=self.req_states.num_computed_tokens.gpu,
                    ),
                    tables.slot_mappings[:, : input_batch.num_tokens_after_padding],
                )
            else:
                prepared.attention = super().prepare_attn(input_batch)
        return prepared.attention

    def _clear_prepared_batch(self):
        if self._prepared_batch is not None:
            self._prepared_batch.unpack_state()
            self._prepared_batch = None

    def _record_uva_completion(self):
        if not self._uva_current_recorded:
            assert self._uva_current is not None
            self._uva_current.record(self.main_stream)
            self._uva_current_recorded = True

    def _submit_forward(self, scheduler_output, *args, **kwargs):
        if not kwargs.get("dummy_run"):
            # An intermediate RLT receipt can reach EngineCore before device work
            # finishes, and one plan can issue several MRv2 forwards. Count the
            # actual consumers before rotating MRv2's host-visible staging slots.
            if self._uva_current is not None:
                self._record_uva_completion()
                self._uva_pending.append(self._uva_current)
            if len(self._uva_pending) == self._uva_capacity:
                self._uva_current = self._uva_pending.popleft()
                self._uva_current.synchronize()
            else:
                self._uva_current = torch.cuda.Event(blocking=True)
            self._uva_current_recorded = False
            step = loop_execution.get()
            if (
                self._early_uva_release
                and step is not None
                and step.op in (LoopOp.PRELUDE, LoopOp.CORE)
                and self.ubatch_runner is None
                and not getattr(self.model, "rlt_core_samples", False)
            ):
                # The Inkling model state signals after enqueuing the input readers.
                step.release_uva_inputs = self._record_uva_completion
        return super().execute_model(scheduler_output, *args, **kwargs)

    def execute_model(self, scheduler_output, *args, **kwargs):
        if not isinstance(scheduler_output, LoopSchedulerOutput) or kwargs.get("dummy_run"):
            self._clear_prepared_batch()
            return self._submit_forward(scheduler_output, *args, **kwargs)

        if (
            scheduler_output.finished_req_ids
            or scheduler_output.preempted_req_ids
            or scheduler_output.scheduled_new_reqs
            or scheduler_output.discarded_states
            or scheduler_output.new_block_ids_to_zero
            or scheduler_output.kv_cache_block_copies
            or any(
                block_ids is not None
                for block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids
            )
        ):
            self._clear_prepared_batch()
        for request_id in scheduler_output.finished_req_ids:
            self._loop_states.pop(request_id, None)
            self._draft_states.pop(request_id, None)
        for request_id in scheduler_output.preempted_req_ids or ():
            self._loop_states.pop(request_id, None)
            self._draft_states.pop(request_id, None)
        for request_id, positions in scheduler_output.discarded_states.items():
            for states in (self._loop_states, self._draft_states):
                if request_id in states:
                    for position in positions:
                        states[request_id].pop(position, None)
        op = scheduler_output.op
        if op is None:
            return self._submit_forward(scheduler_output, *args, **kwargs)

        if not getattr(self.model, "rlt_loop_ops", False):
            raise RuntimeError("the rlt worker loaded a model without the loop plugin adapter")
        if self._trace_steps:
            logger.info("rlt step %s %s", op.value, scheduler_output.spans)
        if self._loop_completion is not None:
            raise RuntimeError("a loop completion is still waiting to be collected")

        core_samples = op is LoopOp.CORE and getattr(self.model, "rlt_core_samples", False)
        manual_coda = op is LoopOp.CODA and any(span.draft_coda for span in scheduler_output.spans)
        groups: dict[tuple[int, bool, bool] | int | str, list[LoopSpan]] = defaultdict(list)
        request_lanes: dict[str, int] = defaultdict(int)
        contains_prefill = any(span.prefill for span in scheduler_output.spans)
        mixed_prefill = (
            op is LoopOp.CORE and contains_prefill and self._prefill_pipeline
            and len({span.depth for span in scheduler_output.spans}) > 1
        )
        variants = {
            (span.depth, mixed_prefill and span.last_core) for span in scheduler_output.spans
        }
        compact_core = (
            op is LoopOp.CORE
            and (mixed_prefill or (
                self._mixed_core and self._compact_mixed_core
            ))
            and getattr(self.model, "rlt_partition_core_variants", False)
            and self.parallel_config.tensor_parallel_size == 1
            and self.vllm_config.kv_transfer_config is None
            and self.vllm_config.lora_config is None
            and (mixed_prefill or not self._decode_state_slots)
            and self.fast_prefill is None
            and not scheduler_output.scheduled_spec_decode_tokens
            and self._supports_input_layout(scheduler_output.spans)
            and (mixed_prefill or len(scheduler_output.num_scheduled_tokens)
                 == len(scheduler_output.spans))
            and len(variants) > 1
        )
        pack_core = self._mixed_core or (self._packed_prefill and contains_prefill)
        if (
            op is LoopOp.CORE
            and getattr(self.model, "rlt_partition_core_variants", False)
            and self.vllm_config.kv_transfer_config is None
        ):
            # Build real request and attention metadata per variant through MRv2.
            pack_core = False
        ordinary_core = (
            (self._merge_decode_core or self._merge_prefill_core or self._decode_state_slots)
            and getattr(self.model, "rlt_partition_core_variants", False)
            and self.parallel_config.tensor_parallel_size == 1
            and self.vllm_config.kv_transfer_config is None
            and self.vllm_config.lora_config is None
            and self._supports_input_layout(scheduler_output.spans)
        )
        ordinary_decode = (
            ordinary_core
            and not contains_prefill
            and all(span.token_count == 1 for span in scheduler_output.spans)
        )
        merge_core = op is LoopOp.CORE and (
            (self._merge_decode_core and ordinary_decode)
            or (self._merge_prefill_core and ordinary_core and contains_prefill)
        )
        if merge_core and mixed_prefill:
            # The same request can appear at several depths with different
            # chunks. Each depth still needs one unique MRv2 row per request.
            requests_by_depth = {(span.depth, span.request_id) for span in scheduler_output.spans}
            merge_core = len(requests_by_depth) == len(scheduler_output.spans)
        for span in scheduler_output.spans:
            if compact_core:
                key = 0
            elif op is LoopOp.PRELUDE:
                key = 0
            elif core_samples and not self._mixed_core:
                key = span.request_id
            elif op is LoopOp.CORE and not pack_core:
                key = span.depth, span.last_core and not merge_core, span.propose_draft
            elif op is LoopOp.CORE and span.prefill and self._packed_prefill:
                key = 0
            else:
                key = request_lanes[span.request_id]
                request_lanes[span.request_id] += 1
            groups[key].append(span)

        if self._trace_steps and pack_core and op is LoopOp.CORE:
            for spans in groups.values():
                depths = sorted({span.depth for span in spans})
                if len(depths) > 1:
                    logger.info(
                        "rlt packed core depths=%s spans=%d tokens=%d",
                        depths,
                        len(spans),
                        sum(span.token_count for span in spans),
                    )
        # Worker request/block deltas apply once even when several groups execute.
        exits: dict[str, int] = {}
        sampled: dict[str, list[int]] = {}
        drafts: dict[str, int] = {}
        logprobs: dict[str, LogprobsLists] = {}
        for index, spans in enumerate(groups.values()):
            group_output = scheduler_output
            if len(groups) > 1:
                counts: dict[str, int] = defaultdict(int)
                for span in spans:
                    counts[span.request_id] += span.token_count
                group_output = replace(
                    scheduler_output,
                    spans=tuple(spans),
                    num_scheduled_tokens=dict(counts),
                    total_num_scheduled_tokens=sum(span.token_count for span in spans),
                    scheduled_new_reqs=scheduler_output.scheduled_new_reqs if index == 0 else [],
                    scheduled_cached_reqs=(
                        scheduler_output.scheduled_cached_reqs
                        if index == 0
                        else CachedRequestData.make_empty()
                    ),
                    finished_req_ids=scheduler_output.finished_req_ids if index == 0 else set(),
                    new_block_ids_to_zero=(
                        scheduler_output.new_block_ids_to_zero if index == 0 else None
                    ),
                    kv_cache_block_copies=(
                        scheduler_output.kv_cache_block_copies if index == 0 else None
                    ),
                )
            step = LoopExecution(
                op, tuple(spans), self._loop_states, self._draft_states, self._loop_graphs
            )
            step.explicit_positions = mixed_prefill or any(span.explicit_position for span in spans)
            step.context_limit = getattr(self, "max_model_len", None)
            step.coda_tail_moe = op is LoopOp.CODA and self._coda_tail_moe
            step.compact_core = compact_core
            step.merged_core = (
                not compact_core
                and merge_core
                and any(span.last_core != spans[0].last_core for span in spans[1:])
            )
            if ordinary_decode and self._decode_state_slots:
                step.decode_slot_capacity = self.max_num_reqs
            context = loop_execution.set(step)
            try:
                result = self._submit_forward(group_output, *args, **kwargs)
            finally:
                loop_execution.reset(context)
            if result is not None:
                return result
            if manual_coda or (
                core_samples and (any(span.last_core for span in spans) or step.exit_depths)
            ):
                context = loop_execution.set(step)
                try:
                    result = super().sample_tokens(None)
                finally:
                    loop_execution.reset(context)
                if isinstance(result, AsyncModelRunnerOutput):
                    result = result.get_output()
                if manual_coda:
                    drafts.update(step.draft_tokens)
                if result is not None:
                    sampled.update(
                        (request_id, ids)
                        for request_id, ids in zip(result.req_ids, result.sampled_token_ids)
                        if request_id not in step.draft_tokens
                    )
                    if result.logprobs is not None:
                        for row, (request_id, ids) in enumerate(
                            zip(result.req_ids, result.sampled_token_ids)
                        ):
                            if ids:
                                logprobs[request_id] = result.logprobs.slice_request(row, len(ids))
                exits.update(step.exit_depths)
                if self._trace_steps:
                    for span in spans:
                        if span.request_id in step.exit_depths:
                            logger.info(
                                "rlt exited depth=%d span=%s",
                                step.exit_depths[span.request_id],
                                span,
                            )
            elif op is LoopOp.CORE or op is LoopOp.PRELUDE:
                self.execute_model_state = None

        if op is not LoopOp.CODA or manual_coda:
            req_ids = list(scheduler_output.num_scheduled_tokens)
            self._loop_completion = LoopModelRunnerOutput(
                req_ids=req_ids,
                sampled_token_ids=[sampled.get(request_id, []) for request_id in req_ids],
                exit_depths=tuple(exits.get(span.request_id) for span in scheduler_output.spans),
                draft_tokens=drafts or None,
                logprobs_by_request=logprobs,
            )
        return None

    def sample_tokens(self, grammar_output):
        if self._loop_completion is None:
            return super().sample_tokens(grammar_output)
        result, self._loop_completion = self._loop_completion, None
        return result

    def sample(self, hidden_states, input_batch, grammar_output):
        output, num_sampled, num_rejected = super().sample(
            hidden_states, input_batch, grammar_output
        )
        step = loop_execution.get()
        if (
            step is not None
            and step.op is LoopOp.CODA
            and any(span.draft_coda for span in step.ordered)
        ):
            rows = [i for i, span in enumerate(step.ordered) if span.draft_coda]
            row_ids = torch.as_tensor(rows, device=num_sampled.device)
            step.draft_tokens.update(
                zip(
                    (step.ordered[i].request_id for i in rows),
                    output.sampled_token_ids[row_ids, 0].tolist(),
                )
            )
            step.uncommitted_row_ids = row_ids
            num_sampled = num_sampled.clone()
            num_sampled.index_fill_(0, row_ids, 0)
        return output, num_sampled, num_rejected

    def postprocess_sampled(
        self,
        idx_mapping,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc=None,
        draft_tokens=None,
    ):
        self._clear_prepared_batch()
        step = loop_execution.get()
        if step is not None and step.uncommitted_row_ids is not None:
            idx_mapping = idx_mapping.clone()
            idx_mapping.index_fill_(0, step.uncommitted_row_ids, -1)
        return super().postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            draft_tokens,
        )
