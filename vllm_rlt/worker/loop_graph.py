"""Capture one Inkling loop operation with vLLM's CUDA graph wrappers."""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import fields, replace

import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config.compilation import (
    CUDAGraphMode,
    decode_cudagraph_capture_max_tokens,
    decode_cudagraph_capture_shapes,
)
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.models.inkling.nvidia.sconv_swa_attn import (
    InklingSconvMetadata,
    InklingSconvMetadataBuilder,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.cudagraph_utils import prepare_inputs_to_capture

from vllm_rlt.core.loop import LoopLayout, LoopOp, LoopSpan
from vllm_rlt.worker.loop_state import LoopExecution, loop_execution

logger = init_logger("vllm.rlt.loop_graph")
_SCONV_TOKEN_FIELDS = ("slot_mapping", "seq_idx", "query_start")
_POSITIONS_INPUT = 3


@triton.jit
def _copy_field(dst, src, index, n, tile: tl.constexpr):
    offset = index * tile + tl.arange(0, tile)
    data = tl.load(src + offset, offset < n, other=0)
    tl.store(dst + offset, data, offset < n)


@triton.jit
def _copy_graph_inputs(DstA, SrcA, DstB, SrcB, DstC, SrcC, a, b, c, tile: tl.constexpr):
    block = tl.program_id(0)
    end_a = tl.cdiv(a, tile)
    end_b = end_a + tl.cdiv(b, tile)
    if block < end_a:
        _copy_field(DstA, SrcA, block, a, tile)
    elif block < end_b:
        _copy_field(DstB, SrcB, block - end_a, b, tile)
    else:
        _copy_field(DstC, SrcC, block - end_b, c, tile)


def _tensor_layout(tensor):
    return tensor.shape, tensor.stride(), tensor.dtype


def _graph_input_layout(tensor):
    # Packing a single request out of a batch can leave arbitrary strides on
    # singleton axes. Staging copies logical elements into the captured inputs.
    shape = tensor.shape
    dtype = tensor.dtype
    if tensor.is_contiguous():
        return shape, None, dtype
    return shape, tuple(stride for size, stride in zip(shape, tensor.stride()) if size != 1), dtype


def _same_view(left, right):
    return (
        isinstance(right, torch.Tensor)
        and left.data_ptr() == right.data_ptr()
        and left.device == right.device
        and _tensor_layout(left) == _tensor_layout(right)
    )


def _persistent_metadata_storage(runner):
    # These buffers are written by MRv2 on the replay stream and live for the
    # lifetime of the worker. Leave any temporary backend tensors staged.
    inputs = runner.input_buffers
    tables = runner.block_tables
    owners = [inputs.query_start_loc, inputs.seq_lens, tables.slot_mappings]
    owners.extend(tables.input_block_tables)
    for groups in runner.attn_groups:
        for group in groups:
            builder = group.get_metadata_builder()
            if isinstance(builder, InklingSconvMetadataBuilder):
                owners.extend((builder.seq_idx_buffer, builder.query_start_buffer))
    return {tensor.untyped_storage().data_ptr() for tensor in owners}


def _metadata_fields(metadata):
    return {
        field.name: tensor
        for field in fields(metadata)
        if isinstance(tensor := getattr(metadata, field.name), torch.Tensor) and tensor.is_cuda
    }


def _metadata_layout(metadata):
    layout = {}
    for value in metadata.values():
        if id(value) not in layout:
            layout[id(value)] = (
                type(value),
                tuple(
                    (name, _tensor_layout(tensor))
                    for name, tensor in _metadata_fields(value).items()
                ),
            )
    return tuple(layout.values())


class _FullGraphMetadata:
    """Keep the same device metadata addresses in all op graphs of one shape."""

    def __init__(self, metadata, persistent_storage=frozenset()):
        self.metadata = {}
        self.bindings = []
        self.aliases = []
        by_source = {}
        for layer, value in metadata.items():
            if id(value) not in by_source:
                buffers = {}
                borrowed = {}
                for name, tensor in _metadata_fields(value).items():
                    if tensor.untyped_storage().data_ptr() in persistent_storage:
                        borrowed[name] = tensor
                    else:
                        buffers[name] = tensor.clone()
                by_source[id(value)] = layer, replace(value, **buffers) if buffers else value
                self.bindings.append((layer, tuple(buffers.items()), tuple(borrowed.items())))
            original_layer, self.metadata[layer] = by_source[id(value)]
            if layer != original_layer:
                self.aliases.append((layer, original_layer))
        self.source = metadata
        self.valid = True

    def stage(self, metadata):
        if metadata is not self.source:
            self.source = metadata
            self.valid = all(
                metadata.get(layer) is metadata.get(original) for layer, original in self.aliases
            ) and all(
                type(metadata.get(layer)) is type(self.metadata[layer])
                and all(
                    _same_view(target, getattr(metadata[layer], name)) for name, target in borrowed
                )
                for layer, _, borrowed in self.bindings
            )
            if self.valid:
                for layer, fields, _ in self.bindings:
                    value = metadata[layer]
                    for name, target in fields:
                        target.copy_(getattr(value, name))
        return self.metadata if self.valid else None


class _PiecewiseGraphMetadata:
    """Stage the SConv inputs read by captured segments, keeping attention live."""

    def __init__(self, metadata, max_reqs, persistent_storage=frozenset()):
        aliases = defaultdict(list)
        for name, value in metadata.items():
            if isinstance(value, InklingSconvMetadata):
                aliases[id(value)].append(name)
        self.bindings = []
        self.borrowed = []
        self.max_reqs = max_reqs
        self.source = None
        self.staged = None
        for names in aliases.values():
            value = metadata[names[0]]
            if all(
                getattr(value, field).untyped_storage().data_ptr() in persistent_storage
                for field in ("block_table", *_SCONV_TOKEN_FIELDS)
            ):
                self.borrowed.append((tuple(names), value))
                continue
            target = InklingSconvMetadata(
                block_table=value.block_table.new_zeros((max_reqs, value.block_table.shape[1])),
                **{name: getattr(value, name).clone() for name in _SCONV_TOKEN_FIELDS},
            )
            self.bindings.append((tuple(names), target))

    def stage(self, metadata):
        if metadata is self.source:
            return self.staged
        for names, target in self.borrowed:
            source = metadata.get(names[0])
            if not isinstance(source, InklingSconvMetadata) or any(
                metadata.get(name) is not source for name in names[1:]
            ):
                return None
            table = source.block_table
            # SConv specializes on the column count and stride. Its runtime
            # token-to-request map chooses which rows of the MRv2 table to read.
            if (
                table.shape[0] > self.max_reqs
                or not _same_view(target.block_table[:1], table[:1])
                or any(
                    not _same_view(getattr(target, field), getattr(source, field))
                    for field in _SCONV_TOKEN_FIELDS
                )
            ):
                return None
        sources = []
        for names, target in self.bindings:
            source = metadata.get(names[0])
            if not isinstance(source, InklingSconvMetadata) or any(
                metadata.get(name) is not source for name in names[1:]
            ):
                return None
            rows, columns = source.block_table.shape
            if (
                rows > target.block_table.shape[0]
                or columns != target.block_table.shape[1]
                or source.block_table.dtype != target.block_table.dtype
                or any(
                    getattr(source, field).shape != getattr(target, field).shape
                    or getattr(source, field).dtype != getattr(target, field).dtype
                    for field in _SCONV_TOKEN_FIELDS
                )
            ):
                return None
            sources.append(source)

        # Eager attention reads the runtime batch offsets and decode/prefill split.
        staged = dict(metadata) if self.bindings else metadata
        for (names, target), source in zip(self.bindings, sources, strict=True):
            target.block_table[: source.block_table.shape[0]].copy_(source.block_table)
            for field in _SCONV_TOKEN_FIELDS:
                getattr(target, field).copy_(getattr(source, field))
            for name in names:
                staged[name] = target
        self.source = metadata
        self.staged = staged
        return staged


class _GraphEntry:
    def __init__(
        self,
        config,
        mode,
        function,
        inputs,
        full_metadata,
        piecewise_metadata,
        *,
        stack_input_blocks,
        padding,
        persistent_inputs,
        shared_stack=None,
    ):
        positions = inputs[_POSITIONS_INPUT]
        self.borrowed_positions = persistent_inputs is not None and _same_view(
            positions, persistent_inputs.positions[: positions.numel()]
        )
        self.borrowed_padding = (
            persistent_inputs is not None
            and padding is not None
            and _same_view(padding, persistent_inputs.is_padding[: padding.numel()])
        )
        self.inputs = (
            shared_stack if shared_stack is not None else inputs[0].clone(),
            *(
                tensor
                if tensor is None or (i == _POSITIONS_INPUT and self.borrowed_positions)
                else tensor.clone()
                for i, tensor in enumerate(inputs[1:], 1)
            ),
        )
        self.full_metadata = full_metadata
        self.piecewise_metadata = piecewise_metadata
        self.stack_input_blocks = stack_input_blocks
        self.padding = padding if self.borrowed_padding else None
        if mode is CUDAGraphMode.FULL:
            self.forward = CUDAGraphWrapper(
                function, config, mode, CUDAGraphOptions(gc_disable=True)
            )
        else:
            self.forward = BreakableCUDAGraphWrapper(function, config, mode)

    def stage_metadata(self, metadata):
        if self.full_metadata is not None:
            return self.full_metadata.stage(metadata)
        assert self.piecewise_metadata is not None
        return self.piecewise_metadata.stage(metadata)

    def stage_inputs(self, inputs):
        copies = []
        if self.inputs[0] is not inputs[0]:
            blocks = self.stack_input_blocks
            copies.append((self.inputs[0][:blocks], inputs[0][:blocks]))
        for index, (buffer, source) in enumerate(zip(self.inputs[1:], inputs[1:], strict=True), 1):
            if (
                buffer is not None
                and buffer is not source
                and (index != _POSITIONS_INPUT or not self.borrowed_positions)
            ):
                copies.append((buffer, source))
        if 2 <= len(copies) <= 3 and all(
            dst.is_cuda and dst.is_contiguous() and src.is_contiguous() for dst, src in copies
        ):
            tile = 1024
            lengths = [dst.numel() for dst, _ in copies]
            grid = sum(cdiv(n, tile) for n in lengths)
            if len(copies) == 2:
                copies.append(copies[0])
                lengths.append(0)
            tensors = [tensor for pair in copies for tensor in pair]
            _copy_graph_inputs[(grid,)](*tensors, *lengths, tile)
        else:
            for target, source in copies:
                target.copy_(source)

    def matches_borrowed_inputs(self, inputs, padding):
        return (
            not self.borrowed_positions
            or _same_view(self.inputs[_POSITIONS_INPUT], inputs[_POSITIONS_INPUT])
        ) and (not self.borrowed_padding or _same_view(self.padding, padding))


class LoopCudaGraphs:
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        compilation = config.compilation_config
        scheduler = config.scheduler_config
        prefill_limit = min(
            scheduler.max_num_batched_tokens,
            scheduler.max_num_seqs * config.model_config.max_model_len,
        )
        prefill_sizes = sorted(set(compilation.prefill_cudagraph_capture_token_sizes or ()))
        self.prefill_sizes = tuple(
            size
            for size in prefill_sizes
            if mode.mixed_mode() is CUDAGraphMode.PIECEWISE and 0 < size <= prefill_limit
        )
        decode_limit = min(scheduler.max_num_seqs, scheduler.max_num_batched_tokens)
        if mode.decode_mode() is CUDAGraphMode.FULL:
            self.decode_sizes = tuple(
                requests
                for requests, _ in decode_cudagraph_capture_shapes(
                    compilation.decode_cudagraph_capture_request_sizes or (),
                    {1: decode_limit},
                    decode_cudagraph_capture_max_tokens(compilation),
                )
            )
        elif mode.decode_mode() is CUDAGraphMode.PIECEWISE:
            self.decode_sizes = tuple(size for size in prefill_sizes if 0 < size <= decode_limit)
        else:
            self.decode_sizes = ()
        self.sizes = frozenset(self.prefill_sizes + self.decode_sizes)
        self.entries = {}
        self.capture_stream = torch.cuda.Stream()
        self._capture_metadata: _FullGraphMetadata | None = None
        self._capture_piecewise: _PiecewiseGraphMetadata | None = None
        self._capture_inputs = None
        self._capture_stacks = {}
        self.full_decode_sizes: tuple[int, ...] = ()
        self.piecewise_sizes: tuple[int, ...] = ()
        self.warming_up = False
        self.warmed_up = False

    def warmup(self, runner, layout: LoopLayout, *, profile_only=False):
        depth = layout.num_loops
        # Every legal target runs the same prefix. Only its terminal CORE differs.
        decode_targets = (depth, *range(1, depth)) if layout.variable_depth else (depth,)
        steps_per_decode = depth + 2 + len(decode_targets) - 1
        self.warming_up = True
        full_decode = self.mode.decode_mode() is CUDAGraphMode.FULL
        captured_decode_sizes = []
        captured_piecewise_sizes = []
        piecewise_targets = (
            (depth,)
            if full_decode and not getattr(runner, "_merge_prefill_core", False)
            else decode_targets
        )
        try:
            for tokens in reversed(self.decode_sizes):
                before = len(self.entries)
                self._warm_shape(runner, tokens, tokens, False, decode_targets)
                if full_decode:
                    captured = len(self.entries) - before
                    if captured == steps_per_decode:
                        captured_decode_sizes.append(tokens)
                    else:
                        logger.warning(
                            "RLT FULL batch %d captured %d/%d per-op graphs",
                            tokens,
                            captured,
                            steps_per_decode,
                        )
            for tokens in reversed(self.prefill_sizes):
                requests = cdiv(tokens, self.config.model_config.max_model_len)
                self._warm_shape(runner, requests, tokens, True, piecewise_targets)
                captured = sum(
                    key[0] is CUDAGraphMode.PIECEWISE and key[2] == tokens for key in self.entries
                )
                if captured >= depth + 1 + len(piecewise_targets):
                    captured_piecewise_sizes.append(tokens)
            logger.info(
                "RLT CUDA Graph %s complete: %d per-op graphs, "
                "FULL decode batches=%s with %d graphs per batch, "
                "PIECEWISE prefill capture sizes=%s",
                "memory estimate capture" if profile_only else "startup warmup",
                len(self.entries),
                sorted(captured_decode_sizes),
                steps_per_decode,
                sorted(captured_piecewise_sizes),
            )
            self.warmed_up = not profile_only
            if self.warmed_up:
                self.full_decode_sizes = tuple(sorted(captured_decode_sizes))
                self.piecewise_sizes = tuple(sorted(captured_piecewise_sizes))
        finally:
            self.warming_up = False
            self._capture_stacks.clear()

    def padded_decode_size(self, tokens):
        index = bisect_left(self.full_decode_sizes, tokens)
        if index >= len(self.full_decode_sizes):
            return None
        return self.full_decode_sizes[index]

    def padded_piecewise_size(self, tokens):
        index = bisect_left(self.piecewise_sizes, tokens)
        if index >= len(self.piecewise_sizes):
            return None
        return self.piecewise_sizes[index]

    def _warm_shape(self, runner, requests, tokens, prefill, targets):
        model = runner.get_model()
        mode = self.mode.mixed_mode() if prefill else self.mode.decode_mode()
        query_len, extra = divmod(tokens, requests)
        attn = prepare_inputs_to_capture(
            requests,
            tokens,
            runner.model_state,
            runner.input_buffers,
            runner.block_tables,
            runner.attn_groups,
            runner.kv_cache_config,
            full_cudagraph=mode is CUDAGraphMode.FULL,
            max_query_len=query_len + bool(extra),
            pcp_manager=runner.pcp_manager,
        )
        inputs = runner.input_buffers
        # Match MRv2 InputBatch.make_dummy's distribution, including the remainder.
        rows = [
            LoopSpan(str(index), 0, query_len + (index >= requests - extra), 0, prefill=prefill)
            for index in range(requests)
        ]
        if isinstance(attn.attn_metadata, dict):
            if mode is CUDAGraphMode.FULL:
                storage = frozenset()
                if (
                    runner.pcp_manager is None
                    and runner.dcp_size == 1
                    and runner.ubatch_runner is None
                ):
                    self._capture_inputs = inputs
                    storage = _persistent_metadata_storage(runner)
                self._capture_metadata = _FullGraphMetadata(attn.attn_metadata, storage)
            elif mode is CUDAGraphMode.PIECEWISE:
                # Every op of this shape reads the same runtime SConv layout.
                storage = frozenset()
                if (
                    runner.pcp_manager is None
                    and runner.dcp_size == 1
                    and runner.ubatch_runner is None
                ):
                    self._capture_inputs = inputs
                    storage = _persistent_metadata_storage(runner)
                self._capture_piecewise = _PiecewiseGraphMetadata(
                    attn.attn_metadata,
                    min(tokens, self.config.scheduler_config.max_num_seqs),
                    storage,
                )
        try:
            with set_forward_context(
                attn.attn_metadata,
                runner.vllm_config,
                num_tokens=tokens,
                slot_mapping=attn.slot_mappings,
                is_padding=inputs.is_padding[:tokens],
            ):
                for num_loops in targets:
                    states = {}
                    plan = (
                        [(LoopOp.PRELUDE, 0)]
                        + [(LoopOp.CORE, depth) for depth in range(num_loops)]
                        + [(LoopOp.CODA, 0)]
                    )
                    for op, depth in plan:
                        spans = tuple(
                            replace(
                                row,
                                depth=depth,
                                last_core=op is LoopOp.CORE and depth == num_loops - 1,
                            )
                            for row in rows
                        )
                        step = LoopExecution(op, spans, states, {}, self)
                        step.context_limit = self.config.model_config.max_model_len
                        step.coda_tail_moe = getattr(runner, "_coda_tail_moe", False)
                        step.ordered = spans
                        context = loop_execution.set(step)
                        try:
                            model(
                                input_ids=inputs.input_ids[:tokens],
                                positions=inputs.positions[:tokens],
                            )
                        finally:
                            loop_execution.reset(context)
        finally:
            self._capture_metadata = None
            self._capture_piecewise = None
            self._capture_inputs = None

    def run(self, variant, step, function, *inputs, stack_input_blocks, stack_output_blocks):
        context = get_forward_context()
        metadata = context.attn_metadata
        prepared = step.prepared_batch
        uniform_decode = step.uniform_decode
        if uniform_decode is None:
            uniform_decode = all(
                not span.prefill and span.token_count == 1 for span in step.ordered
            )
        tokens = step.padded_tokens or (
            prepared.inputs.num_tokens
            if prepared is not None
            else sum(span.token_count for span in step.ordered)
        )
        mode = self.mode.decode_mode() if uniform_decode else self.mode.mixed_mode()
        if mode is CUDAGraphMode.FULL and not uniform_decode:
            # Inkling's full prefill attention chooses its plan with host sequence lengths.
            mode = CUDAGraphMode.NONE
        if tokens not in self.sizes or mode is CUDAGraphMode.NONE or not isinstance(metadata, dict):
            return function(*inputs)

        layout = None
        if mode is CUDAGraphMode.FULL:
            if prepared is not None and prepared.metadata is metadata:
                if prepared.metadata_layout is None:
                    prepared.metadata_layout = _metadata_layout(metadata)
                layout = prepared.metadata_layout
            else:
                layout = _metadata_layout(metadata)
        key = (
            mode,
            variant,
            tokens,
            tuple(_graph_input_layout(tensor) if tensor is not None else None for tensor in inputs),
            layout,
            _tensor_layout(context.is_padding) if context.is_padding is not None else None,
        )
        entry = self.entries.get(key)
        if entry is None and not self.warming_up:
            return function(*inputs)
        padding = context.is_padding
        is_new = entry is None
        if is_new:
            stack_layout = _tensor_layout(inputs[0])
            stack = self._capture_stacks.get(stack_layout)
            if stack is None:
                stack = self._capture_stacks[stack_layout] = torch.empty_like(inputs[0])
            entry = _GraphEntry(
                self.config,
                mode,
                function,
                inputs,
                self._capture_metadata,
                self._capture_piecewise,
                stack_input_blocks=stack_input_blocks,
                padding=padding,
                persistent_inputs=self._capture_inputs,
                shared_stack=stack,
            )
        if not entry.matches_borrowed_inputs(inputs, padding):
            return function(*inputs)
        staged_metadata = entry.stage_metadata(metadata)
        if staged_metadata is None:
            return function(*inputs)
        entry.stage_inputs(inputs)
        previous = context.cudagraph_runtime_mode, context.batch_descriptor
        if padding is not None and not entry.borrowed_padding:
            if entry.padding is None:
                entry.padding = padding.clone()
            else:
                entry.padding.copy_(padding)
        context.attn_metadata = staged_metadata
        context.is_padding = entry.padding
        context.batch_descriptor = BatchDescriptor(num_tokens=tokens)
        try:
            if is_new:
                context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                entry.forward(*entry.inputs)
                entry.stage_inputs(inputs)
                context.cudagraph_runtime_mode = mode
                current = torch.cuda.current_stream()
                self.capture_stream.wait_stream(current)
                with torch.cuda.stream(self.capture_stream):
                    entry.forward(*entry.inputs)
                current.wait_stream(self.capture_stream)
                self.entries[key] = entry
                span = step.ordered[0]
                logger.debug(
                    "rlt captured %s %s for %d tokens (depth=%d, terminal=%s)",
                    mode.name,
                    step.op.value,
                    tokens,
                    span.depth,
                    span.last_core,
                )
            context.cudagraph_runtime_mode = mode
            output = entry.forward(*entry.inputs)
            if stack_output_blocks:
                if step.prepared_batch is not None:
                    # A batch change unpacks live state before another replay
                    # can overwrite this shared stack on the same stream.
                    step.graph_stack = entry.inputs[0]
                elif inputs[0] is not entry.inputs[0]:
                    inputs[0][:stack_output_blocks].copy_(entry.inputs[0][:stack_output_blocks])
            return output
        finally:
            context.attn_metadata = metadata
            context.is_padding = padding
            context.cudagraph_runtime_mode, context.batch_descriptor = previous
