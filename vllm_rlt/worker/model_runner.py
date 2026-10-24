"""Request-phase execution, random lookahead signals and event-owned CUDA submissions."""

import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch

from vllm_rlt.config import ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import ExecutionBatch, Operation, Request
from vllm_rlt.worker.buffers import (
    PackedMetadataBuffer,
    Workspace,
    prefill_stage_rows,
    stage_metadata,
)
from vllm_rlt.worker.cuda_graph import CodaGraphs, RecurrentGraphs
from vllm_rlt.worker.executor import ForwardBatch, ModelRuntime
from vllm_rlt.worker.output import ModelRunnerOutput
from vllm_rlt.worker.sampler import RngRegistry, Sampler
from vllm_rlt.worker.submission_queue import MAX_CONCURRENT_SUBMISSIONS, SubmissionQueue
from vllm_rlt.worker.tensors import StagedInputTensor, TensorTransfer, UvaBuffer, pinned_tensor


@dataclass
class ReadbackSlot:
    storage: torch.Tensor
    leased: bool = False
    event: object = None


class BatchInputBuffers:
    """One event-owned snapshot; source tensors survive until the final reader."""

    def __init__(self, max_seqs, max_tokens, device, hidden_dtype, *, use_uva=True):
        self.segments = UvaBuffer((max_seqs, 5), torch.int64, device, use_uva=use_uva)
        self.segment_view = None
        self.token_input = pinned_tensor(max_tokens, torch.int64, device)
        self.token_output = torch.empty(max_tokens, dtype=torch.int64, device=device)
        self.token_transfers = TensorTransfer(max_seqs, torch.int64, device, use_uva=use_uva)
        self.hidden_gather = TensorTransfer(max_seqs, hidden_dtype, device, use_uva=use_uva)
        self.hidden_scatter = TensorTransfer(2 * max_seqs, hidden_dtype, device, use_uva=use_uva)
        self.hidden_outputs = []

    def reset(self):
        self.token_transfers.reset()
        self.hidden_gather.reset()
        self.hidden_scatter.reset()
        self.hidden_outputs.clear()


class InputSegment(NamedTuple):
    state_slot: int
    chunk_offset: int
    token_count: int
    batch_row_start: int
    keep_intermediate: bool


@dataclass(frozen=True)
class BatchInputs:
    buffers: BatchInputBuffers
    rows: int
    max_rows: int
    token_ids: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    mixed: bool = False


@dataclass(frozen=True)
class PreparedForward:
    """One model call and its output bindings within a submission.

    outputs=None uses the batch's routing/chunk storage; explicit rows save only
    the listed request states. hidden_rows selects the preceding call's output
    on device, for operations whose inputs cannot exist before execution.
    """

    forward: ForwardBatch
    outputs: tuple[tuple[int, Request], ...] | None = None
    hidden_rows: torch.Tensor | None = None
    requests: tuple[Request, ...] = ()


@dataclass(frozen=True)
class PreparedExecution:
    batch: ExecutionBatch
    depths: tuple[int, ...]
    positions: tuple[int, ...]
    output_indices: tuple[int, ...]
    token_sources: tuple[torch.Tensor | tuple[int, ...], ...] = ()
    token_ids: torch.Tensor | None = None
    gate_indices: torch.Tensor | None = None
    workspace: Workspace | None = None
    forwards: tuple[PreparedForward, ...] = ()
    kv: object = None
    routing: object = None
    batch_inputs: BatchInputs | None = None
    packed_metadata: PackedMetadataBuffer | None = None
    sampling_params: tuple = ()
    sampling_generators: tuple = ()


@dataclass
class Submission:
    batch: SchedulerOutput | None
    values: torch.Tensor | None
    event: object = None
    cached: ModelRunnerOutput | None = None
    slot: ReadbackSlot | None = None
    device_values: torch.Tensor | None = None
    depths: tuple[int, ...] = ()
    positions: tuple[int, ...] = ()
    output_indices: tuple[int, ...] = ()
    operation: Operation | None = None

    def __del__(self):
        if self.slot is not None:
            self.slot.leased = False

    @property
    def has_result(self):
        return self.values is not None or self.cached is not None

    def ready(self):
        return self.event is None or self.event.query()

    def collect(self) -> ModelRunnerOutput:
        if self.cached is not None:
            return self.cached
        if self.event is not None:
            self.event.synchronize()
        values = self.values.tolist() if self.values is not None else None
        is_coda = self.operation == Operation.CODA
        is_recurrent = self.operation == Operation.RECURRENT
        self.cached = ModelRunnerOutput(
            gate_scores=values if is_recurrent else None,
            token_ids=values if is_coda else None,
            completion=self.event,
            depths=self.depths,
            output_indices=self.output_indices,
        )
        self.values = None
        if self.slot is not None:
            self.slot.leased = False
            self.slot = None
        return self.cached


class ModelRunner:
    def __init__(
        self,
        model,
        cache_manager,
        *,
        exit_config=None,
        execution_config=None,
        scheduler_config=None,
        rng: RngRegistry | None = None,
    ):
        self.model = model.eval()
        self.cache_manager = cache_manager
        self.metadata = cache_manager.metadata_builder
        self.runtime = ModelRuntime(self.model, cache_manager)
        parameter = next(model.parameters())
        self.device = parameter.device
        self.sampler = Sampler()
        # Sampling RNG state lives with the engine, which shares this registry
        # between the CODA runner and the speculative runner.
        self.rng = rng if rng is not None else RngRegistry()
        self.exit_config = exit_config or ExitConfig()
        self.execution_config = execution_config or ExecutionConfig()
        scheduler = scheduler_config or SchedulerConfig()
        self.graphs = (
            RecurrentGraphs(
                model,
                cache_manager,
                self.execution_config,
                self.exit_config.mode in ("ouro", "ouro_delayed"),
            )
            if self.execution_config.cuda_graphs
            else None
        )
        self.coda_graphs = (
            CodaGraphs(model, self.execution_config, cache_manager)
            if self.graphs is not None and getattr(model, "requires_boundary_kv", False)
            else None
        )
        self.input_buffers: dict[torch.dtype, StagedInputTensor] = {
            dtype: StagedInputTensor(dtype, self.device, use_uva=self.execution_config.use_uva)
            for dtype in (torch.int64, torch.int32)
        }
        self.loop_hidden = [None] * MAX_CONCURRENT_SUBMISSIONS
        self.packed_metadata_buffers = [
            PackedMetadataBuffer(self.device, use_uva=self.execution_config.use_uva)
            for _ in range(MAX_CONCURRENT_SUBMISSIONS)
        ]
        self.lookahead_head = None
        if self.exit_config.mode == "random_lookahead":
            # Keep auxiliary random weights separate from the strict base checkpoint.
            # CPU RNG isolation also makes initialization independent of device count.
            with torch.random.fork_rng(devices=[]):
                generator = torch.Generator(device="cpu").manual_seed(self.exit_config.seed)
                head = torch.nn.Linear(model.config.hidden_size, 1, device="cpu")
                torch.nn.init.normal_(
                    head.weight, std=model.config.initializer_range, generator=generator
                )
                torch.nn.init.zeros_(head.bias)
            self.lookahead_head = head.to(device=self.device, dtype=parameter.dtype).eval()
            self.lookahead_head.requires_grad_(False)
        self.split_prefill = not self.execution_config.atomic_prefill
        self.core_stream = self.boundary_stream = self.copy_stream = self.readback_stream = None
        if self.device.type == "cuda":
            self.core_stream = torch.cuda.Stream(device=self.device)
            self.boundary_stream = (
                torch.cuda.Stream(device=self.device)
                if self.execution_config.multi_stream
                else self.core_stream
            )
            self.copy_stream = (
                torch.cuda.Stream(device=self.device)
                if self.execution_config.multi_stream
                else self.core_stream
            )
            self.readback_stream = (
                torch.cuda.Stream(device=self.device)
                if self.execution_config.multi_stream
                else self.core_stream
            )
            for stream in (
                self.core_stream,
                self.boundary_stream,
                self.copy_stream,
                self.readback_stream,
            ):
                stream.wait_stream(torch.cuda.current_stream(self.device))
        self.readback_slots = []
        if self.device.type == "cuda":
            # cudaHostAlloc during submission can serialize streams. Allocate all
            # readback storage before work starts, retaining leases until consumed.
            for dtype in (torch.float32, torch.int64):
                for _ in range(scheduler.max_num_seqs + 4):
                    self.readback_slots.append(
                        ReadbackSlot(
                            pinned_tensor(scheduler.max_num_batched_tokens, dtype, self.device)
                        )
                    )
        # Retire the oldest consumer BEFORE preparing another batch's inputs.
        # All rotating prefill pools share this maximum in-flight depth.
        self.submissions = SubmissionQueue()
        self.split_prefill_submissions = 0
        self.split_prefill_submission_waits = 0
        self.events = {}
        self.requestspaces = {}
        self.requestspace_index = {"core": 0, "boundary": 0}
        self.state_slots = {}
        self.free_state_slots = list(reversed(range(scheduler.max_num_seqs)))
        self.states = None
        # Prompt hidden states staged between split-prefill scheduling boundaries.
        # Sized by the same helper the memory plan reserves with.
        stage_rows = prefill_stage_rows(model.config, scheduler, self.execution_config)
        self.prefill_states = (
            torch.empty(
                (scheduler.max_num_seqs, stage_rows, model.config.hidden_size),
                device=self.device,
                dtype=parameter.dtype,
            )
            if self.split_prefill
            else None
        )
        prefill_rows = scheduler.max_num_batched_tokens
        if self.execution_config.pad_to_power_of_two:
            prefill_rows = 1 << (prefill_rows - 1).bit_length()
        self.prefill_hidden = (
            torch.empty(
                (prefill_rows, model.config.hidden_size),
                device=self.device,
                dtype=parameter.dtype,
            )
            if self.split_prefill
            else None
        )
        self.batch_input_buffers = (
            [
                BatchInputBuffers(
                    scheduler.max_num_seqs,
                    max(scheduler.max_num_batched_tokens, scheduler.prefill_chunk_size),
                    self.device,
                    parameter.dtype,
                    use_uva=self.execution_config.use_uva,
                )
                for _ in range(MAX_CONCURRENT_SUBMISSIONS)
            ]
            if self.split_prefill
            else []
        )
        # Immutable prompt rows are filled once per request/runner-slot lifetime.
        # Preallocation avoids cudaHostAlloc and prompt packing on every chunk.
        self.prompt_token_buffers = (
            pinned_tensor(
                (scheduler.max_num_seqs, model.config.max_position_embeddings),
                torch.int64,
                self.device,
            )
            if self.split_prefill
            else None
        )
        self.prompt_token_owners = {}
        self.last_effective_size = self.last_submitted_size = 0
        if self.execution_config.static_buffers:
            rows = scheduler.max_num_batched_tokens
            if self.execution_config.pad_to_power_of_two:
                rows = 1 << (rows - 1).bit_length()
            width = math.ceil(model.config.max_position_embeddings / cache_manager.block_size)
            for group in ("core", "boundary"):
                self.requestspaces[group] = [
                    Workspace(
                        rows,
                        width,
                        model.config.hidden_size,
                        self.device,
                        parameter.dtype,
                        use_uva=self.execution_config.use_uva,
                    )
                    for _ in range(2)
                ]
            self.states = torch.empty(
                (scheduler.max_num_seqs, model.config.hidden_size),
                device=self.device,
                dtype=parameter.dtype,
            )
            # Buffers allocated after stream creation must be visible there too.
            for stream in (self.core_stream, self.boundary_stream):
                if stream is not None:
                    stream.wait_stream(torch.cuda.current_stream(self.device))

        self.async_state = None
        if self.execution_config.async_scheduling and self.device.type == "cuda":
            from vllm_rlt.worker.async_state import AsyncState

            rows = scheduler.max_num_batched_tokens
            if self.execution_config.pad_to_power_of_two:
                rows = 1 << (rows - 1).bit_length()
            self.async_state = AsyncState(
                cache_manager,
                model.config,
                scheduler,
                rows,
                use_uva=self.execution_config.use_uva,
            )
            self.states = self.async_state.hidden
            self.state_slots = self.async_state.slots
            self.free_state_slots = self.async_state.free
            for stream in (self.core_stream, self.boundary_stream, self.copy_stream):
                stream.wait_stream(torch.cuda.current_stream(self.device))

    def _size(self, count):
        size = count
        if self.execution_config.pad_to_power_of_two:
            size = 1 << (count - 1).bit_length()
        self.last_effective_size, self.last_submitted_size = count, size
        return size

    def _workspace(self, group):
        if not self.requestspaces:
            return None
        index = self.requestspace_index[group]
        self.requestspace_index[group] = 1 - index
        workspace = self.requestspaces[group][index]
        workspace.acquire()
        return workspace

    def _save(self, request, state):
        if self.states is None or (
            self.async_state is not None and request.request_id not in self.async_state.owners
        ):
            request.hidden_state = state.clone()
        else:
            slot = self._ensure_request_slot(request)
            request.hidden_state = self.states[slot]
            request.hidden_state.copy_(state)

    def _ensure_request_slot(self, request):
        if self.async_state is not None:
            return self.async_state.reserve_slot(request)
        rid = request.request_id
        if rid not in self.state_slots:
            self.state_slots[rid] = self.free_state_slots.pop()
        return self.state_slots[rid]

    def _prefill_segments(self, batch, buffers):
        row_start = 0
        max_rows = 0
        for index, span in enumerate(batch.spans):
            state_slot = self._ensure_request_slot(span.request)
            chunk_offset = span.token_start - span.chunk_start
            keep_intermediate = span.retain_chunk
            segment = InputSegment(
                state_slot,
                chunk_offset,
                span.token_count,
                row_start,
                keep_intermediate,
            )
            buffers.segments.np[index] = (
                segment.state_slot,
                segment.chunk_offset,
                segment.token_count,
                segment.batch_row_start,
                int(segment.keep_intermediate),
            )
            row_start += span.token_count
            max_rows = max(max_rows, span.token_count)
        return row_start, max_rows

    def _prepare_prefill_tokens(self, batch, buffers, rows, max_rows):
        row = 0
        for index, span in enumerate(batch.spans):
            request = span.request
            end = row + span.token_count
            if span.input_tokens is not None:
                buffers.token_input.numpy()[row:end] = span.input_tokens
                source, start = buffers.token_input, row
            elif span.chunk_start is not None:
                slot = self._ensure_request_slot(request)
                if self.prompt_token_owners.get(request.request_id) is not request:
                    count = len(request.prompt_token_ids)
                    self.prompt_token_buffers[slot, :count].numpy()[:] = request.prompt_token_ids
                    self.prompt_token_owners[request.request_id] = request
                source, start = self.prompt_token_buffers[slot], span.token_start
            elif request.input_token_tensor is not None:
                source, start = request.input_token_tensor.reshape(-1), 0
                request.input_token_tensor = None
            else:
                buffers.token_input.numpy()[row] = request.input_token_id
                source, start = buffers.token_input, row
            buffers.token_transfers.append(
                source[start : start + span.token_count], buffers.token_output[row:end]
            )
            row = end
        return buffers.token_output[:rows]

    def _mixed_hidden_sources(self, batch, buffers):
        # Borrow inputs before updating request.hidden_state to its output storage.
        # Transfer descriptors retain their pointer targets until safe slot reuse.
        row = 0
        for index, span in enumerate(batch.spans):
            request = span.request
            slot = self._ensure_request_slot(request)
            chunk = None
            if span.chunk_start is not None:
                start = span.token_start - span.chunk_start
                chunk = self.prefill_states[slot, start : start + span.token_count]
                source = chunk
            else:
                source = request.hidden_state
                source = None if source is None else source.reshape(1, -1)
            last = None
            if span.retain_last:
                if self.states is None:
                    last = torch.empty(
                        self.model.config.hidden_size,
                        device=self.device,
                        dtype=self.prefill_states.dtype,
                    )
                else:
                    last = self.states[slot]
                request.hidden_state = last
            if source is not None:
                buffers.hidden_gather.append(
                    source, self.prefill_hidden[row : row + span.token_count]
                )
            buffers.hidden_outputs.append(
                (
                    row,
                    span.token_count,
                    chunk if chunk is not None and span.retain_chunk else None,
                    last,
                )
            )
            row += span.token_count

    def _mixed_hidden(self, hidden, prepared, *, scatter):
        buffers = prepared.batch_inputs.buffers
        if not scatter:
            buffers.hidden_gather.submit()
            return
        for row, count, chunk, last in buffers.hidden_outputs:
            if chunk is not None:
                buffers.hidden_scatter.append(hidden[row : row + count], chunk)
            if last is not None:
                buffers.hidden_scatter.append(hidden[row + count - 1], last)
        buffers.hidden_scatter.submit()

    def _gather_prefill(self, prepared):
        metadata = prepared.batch_inputs
        count = metadata.rows
        hidden = self.prefill_hidden[:count]
        if metadata.mixed:
            self._mixed_hidden(hidden, prepared, scatter=False)
            return hidden
        if self.device.type == "cuda":
            from vllm_rlt.kernels.routing import gather_segments_kernel

            pool = self.prefill_states.view(-1, self.model.config.hidden_size)
            gather_segments_kernel[
                (
                    len(prepared.batch.spans),
                    math.ceil(metadata.max_rows / 4),
                    math.ceil(pool.shape[1] / 256),
                )
            ](
                pool,
                metadata.buffers.segment_view[: len(prepared.batch.spans)],
                hidden,
                self.prefill_states.shape[1],
                pool.shape[1],
                pool.stride(0),
                hidden.stride(0),
                4,
                256,
            )
            return hidden
        for values in metadata.buffers.segments.np[: len(prepared.batch.spans)]:
            segment = InputSegment(*values)
            start = segment.batch_row_start
            end = start + segment.token_count
            hidden[start:end].copy_(
                self.prefill_states[
                    segment.state_slot,
                    segment.chunk_offset : segment.chunk_offset + segment.token_count,
                ]
            )
        return hidden

    def _scatter_prefill(self, hidden, prepared):
        metadata = prepared.batch_inputs
        if metadata.mixed:
            self._mixed_hidden(hidden, prepared, scatter=True)
            return
        if self.device.type == "cuda":
            from vllm_rlt.kernels.routing import scatter_segments_kernel

            pool = self.prefill_states.view(-1, self.model.config.hidden_size)
            scatter_segments_kernel[
                (
                    len(prepared.batch.spans),
                    math.ceil(metadata.max_rows / 4),
                    math.ceil(pool.shape[1] / 256),
                )
            ](
                hidden,
                metadata.buffers.segment_view[: len(prepared.batch.spans)],
                pool,
                self.prefill_states.shape[1],
                pool.shape[1],
                hidden.stride(0),
                pool.stride(0),
                4,
                256,
            )
        else:
            for values in metadata.buffers.segments.np[: len(prepared.batch.spans)]:
                segment = InputSegment(*values)
                if not segment.keep_intermediate:
                    continue
                self.prefill_states[
                    segment.state_slot,
                    segment.chunk_offset : segment.chunk_offset + segment.token_count,
                ].copy_(
                    hidden[segment.batch_row_start : segment.batch_row_start + segment.token_count]
                )

    def _store_split_prefill_outputs(self, hidden, prepared):
        row = 0
        batch = prepared.batch
        for span in batch.spans:
            if span.retain_last:
                self._save(span.request, hidden[row + span.token_count - 1])
            row += span.token_count

    def _save_batch(self, requests, hidden, routing):
        if routing is None:
            for request, state in zip(requests, hidden):
                self._save(request, state)
            return
        routing.scatter(hidden)
        for request in requests:
            request.hidden_state = self.states[self.state_slots[request.request_id]]

    def _gather(self, requests, workspace, size):
        if workspace is None:
            return torch.stack([r.hidden_state for r in requests])
        hidden = workspace.hidden[:size]
        hidden.zero_()
        for row, request in enumerate(requests):
            hidden[row].copy_(request.hidden_state)
        return hidden

    def _prepare_prefill_coda(self, ids, positions, requests, kv=None):
        if not getattr(self.model, "requires_boundary_kv", False):
            return None, None
        rows = [
            row
            for row, (position, request) in enumerate(zip(positions, requests))
            if position < len(request.prompt_token_ids) - 1
        ]
        if not rows:
            return None, None
        index = self.input_buffers[torch.int64].append(rows)
        if kv is not None and rows == list(range(rows[0], rows[-1] + 1)):
            kv_metadata = self.metadata.slice_rows(kv, rows[0], rows[-1] + 1)
        else:
            kv_metadata = self._stage_metadata(
                [ids[row] for row in rows],
                [[0] * len(rows)],
                [positions[row] for row in rows],
                packed=self.cache_manager.attention_capabilities.packed_prefill,
            )[0]
        return ForwardBatch(
            Operation.CODA, prelude_coda_kv_metadata=kv_metadata, compute_logits=False
        ), index

    def _prepare_shared_loop_inputs(self, batch):
        cache = self.cache_manager
        spans = sorted(batch.spans, key=lambda span: span.token_count, reverse=True)
        ids = [span.request.request_id for span in spans]
        base = self._stage_metadata(
            ids,
            [[span.depth for span in spans]],
            [span.token_start + span.token_count - 1 for span in spans],
            padded_rows=self._size(len(spans)),
        )[0]
        groups, positions, blocks, tokens, contexts = [], [], [], [], []
        for offset in range(spans[0].token_count):
            active = [span for span in spans if offset < span.token_count]
            start = len(positions)
            for row, span in enumerate(active):
                position = span.token_start + offset
                positions.append(position)
                blocks.append(base.rows[row][0].block_tables[0][position // cache.block_size])
                tokens.append(span.input_tokens[offset])
                contexts.append(position + 1)
            size = self._size(len(active))
            padding = size - len(active)
            positions.extend([0] * padding)
            blocks.extend([0] * padding)
            tokens.extend([0] * padding)
            contexts.extend([0] * padding)
            outputs = tuple(
                (row, span.request)
                for row, span in enumerate(active)
                if offset == span.token_count - 1
            )
            groups.append((start, size, active, outputs))
        n = len(positions)
        wide = self.input_buffers[torch.int64].append(
            positions + blocks + [p % cache.block_size for p in positions]
        )
        lengths = self.input_buffers[torch.int32].append(contexts)
        token_ids = self.input_buffers[torch.int64].append(tokens)
        inputs = []
        for start, size, active, outputs in groups:
            count, end = len(active), start + size
            metadata = self.metadata.prepare_shared_rows(
                base,
                active,
                positions[start : start + count],
                position_ids=wide[start:end],
                write_blocks=wide[n + start : n + start + count],
                write_offsets=wide[2 * n + start : 2 * n + start + count],
                context_lengths=lengths[start:end],
                loop_count=batch.loop_count,
            )
            kv = metadata[0]
            requests = tuple(span.request for span in active)
            coda, coda_rows = self._prepare_prefill_coda(
                ids[:count],
                positions[start : start + count],
                requests,
                kv=kv,
            )
            inputs.append(
                PreparedForward(
                    ForwardBatch(Operation.PRELUDE, tokens=token_ids[start:end], metadata=metadata),
                    outputs=outputs,
                    requests=requests,
                )
            )
            if coda is not None:
                inputs.append(PreparedForward(coda, outputs=(), hidden_rows=coda_rows))
        return tuple(inputs)

    def _uses_packed_loop_inputs(self, batch):
        return (
            batch.operation == Operation.PRELUDE
            and batch.loop_count
            and self.cache_manager.layout != "shared"
            and self.cache_manager.attention_capabilities.packed_prefill
            and self.execution_config.prefill_uva
        )

    def _prepare_loop_inputs(self, batch, packed_metadata=None):
        cache = self.cache_manager
        if cache.layout == "shared":
            return self._prepare_shared_loop_inputs(batch)
        ids, positions, tokens, requests, outputs, depths = [], [], [], [], [], []
        for span in batch.spans:
            count, request = span.token_count, span.request
            ids.extend([request.request_id] * count)
            depths.extend([span.depth] * count)
            positions.extend(range(span.token_start, span.token_start + count))
            tokens.extend(span.input_tokens)
            requests.extend([request] * count)
            outputs.append((len(tokens) - 1, request))
        packed = cache.attention_capabilities.packed_prefill
        size = len(tokens) if packed else self._size(len(tokens))
        depth_sets = [[depth + offset for depth in depths] for offset in range(batch.loop_count)]
        if packed_metadata is not None:
            layouts = self.metadata.build(ids, depth_sets, positions, packed=True)
            packed_metadata.store(layouts, tokens)
            tensor, metadata = None, ()
        else:
            tensor = self.input_buffers[torch.int64].append(tokens + [0] * (size - len(tokens)))
            metadata = self._stage_metadata(
                ids,
                depth_sets,
                positions,
                packed=packed,
                padded_rows=size,
            )
        coda, coda_rows = self._prepare_prefill_coda(ids, positions, requests)
        forwards = [
            PreparedForward(
                ForwardBatch(Operation.PRELUDE, tokens=tensor, metadata=metadata),
                outputs=tuple(outputs),
                requests=tuple(requests),
            )
        ]
        if coda is not None:
            forwards.append(PreparedForward(coda, outputs=(), hidden_rows=coda_rows))
        return tuple(forwards)

    def _stage_metadata(self, ids, depth_sets, positions, *, packed=False, padded_rows=None):
        layouts = self.metadata.build(ids, depth_sets, positions, packed=packed)
        return stage_metadata(layouts, self._append_input, padded_rows=padded_rows)

    def _append_input(self, values, dtype):
        return self.input_buffers[dtype].append(values)

    def _prepare_generators(self, requests):
        return [self.rng.acquire(r.request_id, r.sampling_params, self.device) for r in requests]

    def _prepare_loop_forwards(self, prepared):
        slot = self.input_buffers[torch.int64].pool.index
        uploaded = {dtype: buffer.upload() for dtype, buffer in self.input_buffers.items()}

        def input_view(tensor):
            return uploaded[tensor.dtype].as_strided(
                tensor.shape, tensor.stride(), tensor.storage_offset()
            )

        forwards = []
        for call in prepared.forwards:
            forward = call.forward
            if forward.operation == Operation.CODA:
                forward = replace(
                    forward,
                    prelude_coda_kv_metadata=self.metadata.bind(
                        forward.prelude_coda_kv_metadata, uploaded
                    ),
                )
                forwards.append(
                    replace(call, forward=forward, hidden_rows=input_view(call.hidden_rows))
                )
                continue
            if prepared.packed_metadata is not None:
                tokens, metadata = prepared.packed_metadata.upload()
            else:
                tokens = input_view(forward.tokens)
                metadata = tuple(self.metadata.bind(kv, uploaded) for kv in forward.metadata)
            kv_metadata = None
            hidden_buffer = None
            if getattr(self.model, "requires_boundary_kv", False):
                kv_metadata = metadata[0]
                count, size = len(call.requests), len(tokens)
                if size > count:
                    kv_metadata = self.metadata.without_padding(kv_metadata)
                    storage = self.loop_hidden[slot]
                    if storage is None or len(storage) < size:
                        capacity = 1 << (size - 1).bit_length()
                        storage = torch.empty(
                            (capacity, self.model.config.hidden_size),
                            device=self.device,
                            dtype=next(self.model.parameters()).dtype,
                        )
                        self.loop_hidden[slot] = storage
                    hidden_buffer = storage[:size]
            forward = replace(
                forward,
                tokens=tokens,
                metadata=metadata,
                prelude_coda_kv_metadata=kv_metadata,
                hidden_buffer=hidden_buffer,
                generators=self._prepare_generators(call.requests) if kv_metadata else None,
            )
            forwards.append(replace(call, forward=forward))
        return tuple(forwards)

    def prepare_on_cpu(self, batch: ExecutionBatch):
        """Snapshot inputs and plan all host-known metadata before GPU preparation."""
        for buffer in self.input_buffers.values():
            buffer.reset()
        prepared = self._prepare_cpu_inputs(batch)
        if batch.operation == Operation.PRELUDE and batch.loop_count:
            return prepared
        metadata, boundary_metadata = (), None
        if batch.operation == Operation.RECURRENT:
            start = 1 if prepared.kv is not None else 0
            if start < batch.loop_count:
                ids, depths, positions = batch.addresses()
                metadata = self._stage_metadata(
                    ids,
                    [
                        [depth + offset for depth in depths]
                        for offset in range(start, batch.loop_count)
                    ],
                    positions,
                    packed=batch.has_chunks
                    and self.cache_manager.attention_capabilities.packed_prefill,
                )
        elif getattr(self.model, "requires_boundary_kv", False):
            ids, depths, positions = batch.addresses()
            boundary_metadata = self._stage_metadata(ids, (depths,), positions)[0]
        forward = ForwardBatch(
            batch.operation, metadata=metadata, prelude_coda_kv_metadata=boundary_metadata
        )
        return replace(prepared, forwards=(PreparedForward(forward),))

    def _prepare_cpu_inputs(self, batch: ExecutionBatch):
        """Snapshot request inputs and acquire the storage used by this submission."""
        batch.validate_depths(self.model.config.total_ut_steps)
        requests = [span.request for span in batch.spans]
        # Result bookkeeping needs CODA's actual completed-loop count, not
        # the zero depth used in its execution descriptor for boundary KV.
        depths = tuple(
            span.request.loops_done if batch.operation == Operation.CODA else span.depth
            for span in batch.spans
        )
        positions = tuple(span.token_start for span in batch.spans)
        indices = tuple(r.num_scheduled_outputs for r in requests)
        gate_rows = []
        if (
            batch.operation == Operation.RECURRENT
            and batch.emit_gate
            and batch.num_rows != len(batch.spans)
        ):
            end = 0
            for span in batch.spans:
                end += span.token_count
                gate_rows.append(end - 1)
        prepared = PreparedExecution(batch, depths, positions, indices)
        if gate_rows:
            prepared = replace(
                prepared, gate_indices=self.input_buffers[torch.int64].append(gate_rows)
            )
        if batch.operation == Operation.CODA:
            prepared = replace(prepared, sampling_params=tuple(r.sampling_params for r in requests))
        if batch.operation == Operation.PRELUDE and batch.loop_count:
            packed_metadata = (
                self.packed_metadata_buffers[self.submissions.slot]
                if self._uses_packed_loop_inputs(batch)
                else None
            )
            forwards = self._prepare_loop_inputs(batch, packed_metadata)
            return replace(prepared, packed_metadata=packed_metadata, forwards=forwards)
        if batch.has_chunks:
            buffers = self.batch_input_buffers[self.submissions.slot]
            buffers.reset()
            if batch.all_chunks:
                rows, max_rows = self._prefill_segments(batch, buffers)
            else:
                rows = batch.num_rows
                max_rows = max(span.token_count for span in batch.spans)
            metadata = BatchInputs(buffers, rows, max_rows, mixed=not batch.all_chunks)
            prepared = replace(prepared, batch_inputs=metadata)
            if batch.operation == Operation.PRELUDE:
                tokens = self._prepare_prefill_tokens(batch, buffers, rows, max_rows)
                return replace(prepared, batch_inputs=replace(metadata, token_ids=tokens))
            return prepared
        tokens = []
        if batch.operation == Operation.PRELUDE:
            explicit = any(span.input_tokens is not None for span in batch.spans)
            for span in batch.spans:
                request = span.request
                if span.input_tokens is not None:
                    tokens.append(tuple(span.input_tokens))
                elif request.input_token_tensor is not None:
                    tokens.append(request.input_token_tensor)
                elif self.async_state is not None and not explicit:
                    raise RuntimeError("async prelude requires the preceding device sample")
                else:
                    tokens.append((request.input_token_id,))
            prepared = replace(prepared, token_sources=tuple(tokens))
            if explicit:
                return replace(prepared, workspace=self._workspace("boundary"))
        if self.async_state is not None:
            routing, kv = self.async_state.prepare(
                requests,
                depths,
                positions,
                self._size(len(requests)),
                recurrent=batch.operation == Operation.RECURRENT,
            )
            return replace(prepared, kv=kv, routing=routing)
        group = "core" if batch.operation == Operation.RECURRENT else "boundary"
        workspace = self._workspace(group)
        prepared = replace(prepared, workspace=workspace)
        if (
            workspace is not None
            and batch.operation == Operation.RECURRENT
            and not (batch.emit_gate and self.graphs is not None)
            and batch.loop_count == 1
        ):
            ids, row_depths, row_positions = batch.addresses()
            layout = self.metadata.build(ids, (row_depths,), row_positions)[0]
            kv = workspace.store_metadata(layout, self._size(batch.num_rows))
            prepared = replace(prepared, kv=kv)
        return prepared

    def prepare_on_gpu(self, prepared):
        """Prepare all model inputs before forward on the selected execution streams."""
        batch = prepared.batch
        if prepared.sampling_params:
            prepared = replace(
                prepared,
                sampling_generators=tuple(
                    None
                    if params.temperature == 0
                    else self.rng.acquire(span.request.request_id, params, self.device)
                    for span, params in zip(batch.spans, prepared.sampling_params)
                ),
            )
        if prepared.routing is not None:
            with torch.cuda.stream(self.copy_stream):
                kv = prepared.routing.transfer(prepared.kv)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).wait_event(prepared.routing.ready_event)
            prepared = replace(prepared, kv=kv)
        elif prepared.kv is not None and prepared.workspace is not None:
            prepared = replace(prepared, kv=prepared.workspace.upload_metadata(prepared.kv))
        if prepared.batch_inputs is not None:
            metadata = prepared.batch_inputs
            buffers = metadata.buffers
            if metadata.mixed:
                self._mixed_hidden_sources(batch, buffers)
            else:
                buffers.segment_view = buffers.segments.stage(len(batch.spans))
            if batch.operation == Operation.PRELUDE:
                buffers.token_transfers.submit()
                prepared = replace(prepared, token_ids=metadata.token_ids)
            else:
                hidden = self._gather_prefill(prepared)
                prepared = replace(prepared, batch_inputs=replace(metadata, hidden=hidden))
        elif batch.operation == Operation.PRELUDE and not batch.loop_count:
            tokens = self._prepare_tokens(prepared, self._size(batch.num_rows))
            prepared = replace(prepared, token_ids=tokens)
        return self._prepare_forward(prepared)

    def _prepare_forward(self, prepared):
        batch, workspace = prepared.batch, prepared.workspace
        if batch.operation == Operation.PRELUDE and batch.loop_count:
            return replace(prepared, forwards=self._prepare_loop_forwards(prepared))
        requests = [span.request for span in batch.spans]
        size = batch.num_rows if batch.has_chunks else self._size(batch.num_rows)
        inputs = prepared.forwards[0].forward
        metadata = inputs.metadata
        kv_metadata = inputs.prelude_coda_kv_metadata
        if metadata or kv_metadata is not None or prepared.gate_indices is not None:
            uploaded = {dtype: buffer.upload() for dtype, buffer in self.input_buffers.items()}
            if prepared.gate_indices is not None:
                indices = prepared.gate_indices
                prepared = replace(
                    prepared,
                    gate_indices=uploaded[indices.dtype].as_strided(
                        indices.shape, indices.stride(), indices.storage_offset()
                    ),
                )
            metadata = tuple(self.metadata.bind(kv, uploaded) for kv in metadata)
            if kv_metadata is not None:
                kv_metadata = self.metadata.bind(kv_metadata, uploaded)
        if batch.operation == Operation.PRELUDE:
            forward = ForwardBatch(
                Operation.PRELUDE,
                tokens=prepared.token_ids,
                prelude_coda_kv_metadata=kv_metadata,
                generators=self._prepare_generators(requests) if kv_metadata else None,
            )
            return replace(prepared, forwards=(PreparedForward(self.runtime.prepare(forward)),))
        hidden = self._load_hidden(batch, prepared, workspace, size)
        if batch.operation == Operation.CODA:
            forward = ForwardBatch(
                Operation.CODA,
                hidden=hidden[: batch.num_rows] if kv_metadata else hidden,
                prelude_coda_kv_metadata=kv_metadata,
                graphs=self.coda_graphs if kv_metadata else None,
            )
            return replace(prepared, forwards=(PreparedForward(self.runtime.prepare(forward)),))
        graphs = self.graphs if batch.emit_gate else None
        kv = prepared.kv
        if kv is not None:
            metadata = (kv,) + metadata
        forward = ForwardBatch(
            Operation.RECURRENT,
            hidden=hidden[: batch.num_rows]
            if graphs is not None or batch.loop_count > 1
            else hidden,
            metadata=metadata,
            graphs=graphs,
            compute_gate=batch.emit_gate and self.exit_config.mode in ("ouro", "ouro_delayed"),
        )
        return replace(
            prepared, kv=metadata[0], forwards=(PreparedForward(self.runtime.prepare(forward)),)
        )

    def _load_hidden(self, batch, prepared, workspace, size):
        if prepared is not None and prepared.batch_inputs is not None:
            return prepared.batch_inputs.hidden
        routing = prepared.routing if prepared is not None else None
        return (
            routing.gather()
            if routing is not None
            else self._gather([span.request for span in batch.spans], workspace, size)
        )

    def _store_hidden(self, batch, hidden, prepared):
        if prepared is not None and prepared.batch_inputs is not None:
            self._scatter_prefill(hidden, prepared)
            if not prepared.batch_inputs.mixed:
                self._store_split_prefill_outputs(hidden, prepared)
            return
        routing = prepared.routing if prepared is not None else None
        self._save_batch([span.request for span in batch.spans], hidden, routing)

    def _prepare_tokens(self, prepared, size):
        """Materialize snapshotted token sources on the current compute stream."""
        sources = prepared.token_sources
        workspace = prepared.workspace
        if prepared.routing is not None:
            tokens = prepared.routing.gather(tokens=True)
        elif all(isinstance(source, torch.Tensor) for source in sources):
            tokens = torch.stack(sources)
        elif all(isinstance(source, tuple) and len(source) == 1 for source in sources):
            ids = [source[0] for source in sources]
            tokens = (
                workspace.tokens(ids, size)
                if workspace is not None
                else torch.tensor(ids, device=self.device, dtype=torch.long)
            )
        else:
            tokens = torch.cat(
                [
                    source.reshape(-1)
                    if isinstance(source, torch.Tensor)
                    else torch.tensor(source, device=self.device, dtype=torch.long)
                    for source in sources
                ]
            )
        if prepared.routing is None and size > len(tokens):
            tokens = torch.cat((tokens, tokens.new_zeros(size - len(tokens))))
        for span, source in zip(prepared.batch.spans, sources):
            if isinstance(source, torch.Tensor) and source.is_cuda:
                source.record_stream(torch.cuda.current_stream(self.device))
            if span.request.input_token_tensor is source:
                span.request.input_token_tensor = None
        return tokens

    @torch.inference_mode()
    def _execute(self, scheduled_batch, prepared):
        return self._execute_batch(prepared.batch, prepared)

    def _execute_batch(self, batch: ExecutionBatch, prepared):
        hidden = None
        values = None
        for entry in prepared.forwards:
            forward = entry.forward
            if entry.hidden_rows is not None:
                forward = replace(forward, hidden=hidden[entry.hidden_rows])
            result = self.runtime.forward(forward)
            hidden = result.hidden
            if forward.operation in (Operation.PRELUDE, Operation.RECURRENT):
                if entry.outputs is None:
                    self._store_hidden(batch, hidden, prepared)
                else:
                    for row, request in entry.outputs:
                        self._save(request, hidden[row])
                if forward.operation == Operation.RECURRENT and batch.emit_gate:
                    logits = result.gate_logits
                    if self.lookahead_head is not None:
                        logits = self.lookahead_head(hidden).squeeze(-1)
                    if logits is not None:
                        selected = (
                            logits.index_select(0, prepared.gate_indices)
                            if prepared.gate_indices is not None
                            else logits[: len(batch.spans)]
                        )
                        values = selected.float().sigmoid()
            elif forward.compute_logits:
                values = torch.stack(
                    [
                        self.sampler.sample(row, params, generator)
                        for row, params, generator in zip(
                            result.logits, prepared.sampling_params, prepared.sampling_generators
                        )
                    ]
                )
                if prepared.routing is not None:
                    prepared.routing.scatter(values, tokens=True)
        return values

    def _readback_slot(self, result):
        pending = None
        for slot in self.readback_slots:
            if slot.storage.dtype != result.dtype or slot.leased:
                continue
            if slot.event is None or slot.event.query():
                slot.leased = True
                return slot
            if pending is None:
                pending = slot
        if pending is not None:
            pending.event.synchronize()
            pending.leased = True
            return pending
        raise RuntimeError("readback buffers exhausted; retire completed submissions")

    def _synchronize_submission(self, stream, prepared):
        if self.device.type != "cuda":
            return
        (stream or torch.cuda.current_stream(self.device)).synchronize()
        if self.readback_stream is not None and self.readback_stream is not stream:
            self.readback_stream.synchronize()
        if prepared is not None and prepared.routing is not None:
            if self.copy_stream is not None and self.copy_stream is not stream:
                self.copy_stream.synchronize()

    def _wait_for_requests(self, batch, stream):
        # A previous batch often gives many requests the same completion event.
        dependencies = set()
        for span in batch.spans:
            event = self.events.get(span.request.request_id)
            if event is not None and event not in dependencies:
                stream.wait_event(event)
                dependencies.add(event)
            if span.request.hidden_state is not None:
                span.request.hidden_state.record_stream(stream)

    @torch.inference_mode()
    def submit(self, scheduled_batch: SchedulerOutput | None = None, *, batch: ExecutionBatch):
        """Submit model computation and return its result/completion ownership.

        A scheduled batch supplies lifecycle snapshots to the engine. Direct
        execution needs only ExecutionBatch; neither path waits for readback here
        unless bounded buffers must be recycled.
        """
        split_prefill = batch.has_chunks
        # A fused prelude + loops stays resident on core; chunking does not
        # change the stream of an otherwise standalone prelude or coda.
        stream = self.core_stream if batch.loop_count else self.boundary_stream
        prepared = None
        routing_finish_attempted = False
        slot = None
        try:
            if self.submissions.begin() and split_prefill:
                self.split_prefill_submission_waits += 1
            prepared = self.prepare_on_cpu(batch)
            source_stream = (
                torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
            )
            if prepared.routing is not None and prepared.routing.imports:
                self.copy_stream.wait_stream(source_stream)
                # Imported hidden/token tensors may have been produced by core,
                # not by the caller's stream. The copy itself must wait for them.
                self._wait_for_requests(batch, self.copy_stream)
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                if stream is not None:
                    if self.async_state is None and source_stream != stream:
                        stream.wait_stream(source_stream)
                    self._wait_for_requests(batch, stream)
                prepared = self.prepare_on_gpu(prepared)
                try:
                    result = self._execute(scheduled_batch, prepared)
                finally:
                    if prepared.routing is not None:
                        routing_finish_attempted = True
                        prepared.routing.record_done()
                # Fixed-depth/trace execution never consumes a gate. In the
                # delayed policy the final loop's signal also has no consumer.
                # Keep graph selection and its captured compute unchanged.
                if (
                    scheduled_batch is not None
                    and self.execution_config.async_scheduling
                    and batch.operation == Operation.RECURRENT
                    and not any(
                        span.emit_gate
                        and self.exit_config.mode in ("ouro_delayed", "random_lookahead")
                        and span.request.sampling_params.exit_threshold < 1
                        and span.depth + batch.loop_count
                        < (
                            span.request.sampling_params.max_loops
                            or self.model.config.total_ut_steps
                        )
                        for span in batch.spans
                    )
                ):
                    result = None
                device_values = result
                event = compute_done = None
                if self.device.type == "cuda":
                    compute_stream = torch.cuda.current_stream(self.device)
                    compute_done = torch.cuda.Event()
                    compute_done.record(compute_stream)
                    event = compute_done
                    if result is not None:
                        slot = self._readback_slot(result)
                        readback_stream = self.readback_stream or compute_stream
                        with torch.cuda.stream(readback_stream):
                            if readback_stream != compute_stream:
                                readback_stream.wait_event(compute_done)
                            result.record_stream(readback_stream)
                            host = slot.storage[: result.numel()].view(result.shape)
                            host.copy_(result, non_blocking=True)
                            event = torch.cuda.Event()
                            event.record(readback_stream)
                        slot.event = event
                        result = host
                    for span in batch.spans:
                        self.events[span.request.request_id] = compute_done
                    self.submissions.complete(compute_done)
            if split_prefill:
                self.split_prefill_submissions += 1
            return Submission(
                scheduled_batch,
                result,
                event,
                slot=slot,
                device_values=device_values,
                depths=prepared.depths,
                positions=prepared.positions,
                output_indices=prepared.output_indices,
                operation=batch.operation,
            )
        except Exception:
            self._synchronize_submission(stream, prepared)
            # A failed preparation may advance a pool without enqueueing a batch.
            # Retire all prior readers before any pool can wrap after that gap.
            self.submissions.drain()
            try:
                if (
                    prepared is not None
                    and prepared.routing is not None
                    and not routing_finish_attempted
                ):
                    with torch.cuda.stream(stream) if stream is not None else nullcontext():
                        routing_finish_attempted = True
                        prepared.routing.record_done()
            finally:
                if slot is not None:
                    slot.leased = False
            raise
        finally:
            if prepared is not None and prepared.workspace is not None:
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    prepared.workspace.record_done()

    def finalize_many(self, requests):
        if not requests:
            return
        if self.async_state is None or self.cache_manager.layout == "shared":
            for request in requests:
                self.finalize(request)
            return
        from vllm_rlt.kernels.routing import finalize_kernel

        cache = self.cache_manager
        copies = []
        for request in requests:
            depth = request.loops_done - 1
            if not cache.token_written(request.request_id, request.position, depth):
                raise RuntimeError("cannot finalize before every layer has written KV")
            if depth + 1 < cache.max_loops:
                copies.append(request)
        if not copies:
            return
        routing_buffers, _ = self.async_state.prepare(
            copies,
            [r.loops_done - 1 for r in copies],
            [r.position for r in copies],
            len(copies),
            finalize=True,
        )
        with torch.cuda.stream(self.boundary_stream):
            for request in copies:
                event = self.events.get(request.request_id)
                if event is not None:
                    self.boundary_stream.wait_event(event)
            routing_buffers.descriptor = routing_buffers.routing.stage(routing_buffers.count)
            channels = cache.num_kv_heads * cache.head_dim
            finalize_kernel[
                (len(copies), cache.max_loops, math.ceil(cache.num_layers * channels / 256))
            ](
                routing_buffers.descriptor,
                self.async_state.tables,
                cache.key_cache,
                cache.value_cache,
                self.async_state.width,
                cache.max_loops,
                cache.block_size,
                cache.num_layers,
                channels,
                *cache.key_cache.stride()[:3],
                256,
            )
            if cache.key_scale is not None:
                # FP8 pools: the same copy for the per-vector scales.
                finalize_kernel[
                    (
                        len(copies),
                        cache.max_loops,
                        math.ceil(cache.num_layers * cache.num_kv_heads / 256),
                    )
                ](
                    routing_buffers.descriptor,
                    self.async_state.tables,
                    cache.key_scale,
                    cache.value_scale,
                    self.async_state.width,
                    cache.max_loops,
                    cache.block_size,
                    cache.num_layers,
                    cache.num_kv_heads,
                    *cache.key_scale.stride()[:3],
                    256,
                )
            routing_buffers.record_done()
            for request in copies:
                self.events[request.request_id] = routing_buffers.done
                cache.mark_finalized(request.request_id, request.position, request.loops_done - 1)

    def finalize(self, request):
        # The final core event must precede copies and coda on the boundary stream.
        stream = self.boundary_stream
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            event = self.events.get(request.request_id)
            if stream is not None and event is not None:
                stream.wait_event(event)
            self.cache_manager.finalize_token(
                request.request_id, request.position, request.loops_done - 1
            )
            if stream is not None:
                event = torch.cuda.Event()
                event.record(stream)
                self.events[request.request_id] = event

    def _free_runner_slot(self, request_id):
        # The latest submission or KV-finalization event follows earlier work
        # through stream dependencies. Wait before recycling this request's slot.
        event = self.events.pop(request_id, None)
        if event is not None:
            event.synchronize()
        self.prompt_token_owners.pop(request_id, None)
        if self.async_state is not None:
            self.async_state.release(request_id)
            return
        slot = self.state_slots.pop(request_id, None)
        if slot is not None:
            self.free_state_slots.append(slot)

    def suspend(self, request_id):
        """Free the request's runner resources but keep its sampling RNG.

        Preemption suspends a request without terminating it, so its generator
        must survive for the resumed token sequence to match an uninterrupted run.
        """
        self._free_runner_slot(request_id)

    def release(self, request_id):
        """Terminate the request: free its runner resources and drop its RNG.

        A new request reusing the same ID must start from a fresh generator, so
        termination drops the RNG slot while suspension keeps it.
        """
        self._free_runner_slot(request_id)
        self.rng.drop(request_id)

    def synchronize(self):
        for stream in (
            self.core_stream,
            self.boundary_stream,
            self.copy_stream,
            self.readback_stream,
        ):
            if stream is not None:
                stream.synchronize()
