from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.core.loop import LoopLayout, LoopOp, LoopSpan


def test_prefill_coda_keeps_the_exact_suffix_of_the_worker_state():
    pytest.importorskip("vllm")
    from vllm_rlt.models.inkling_plugin import LoopInklingForConditionalGeneration, _InklingState
    from vllm_rlt.worker.loop_state import LoopExecution

    stack = torch.arange(16).reshape(2, 8, 1)
    original = _InklingState(stack, stack[0], stack[:1], None, 0, 1)
    prefix = LoopSpan("prompt", 10, 7, 2, prefill=True, coda_remainder=1)
    states = {"prompt": {10: original}}
    step = LoopExecution(LoopOp.CODA, (prefix,), states, {})
    step.ordered = step.spans
    actual = LoopInklingForConditionalGeneration._take_state(SimpleNamespace(), step)
    assert torch.equal(actual.stack, stack[:, :7])
    assert tuple(states["prompt"]) == (17,)
    assert torch.equal(states["prompt"][17].stack, stack[:, 7:])

    suffix = LoopSpan("prompt", 17, 1, 2, prefill=True, sample=True)
    step = LoopExecution(LoopOp.CODA, (suffix,), states, {})
    step.ordered = step.spans
    actual = LoopInklingForConditionalGeneration._take_state(SimpleNamespace(), step)
    assert torch.equal(actual.inputs, stack[:1, 7:])
    assert not states["prompt"]


def test_full_op_graphs_refresh_shared_metadata_once(monkeypatch):
    pytest.importorskip("vllm")
    from vllm_rlt.worker import loop_graph

    @dataclass
    class Metadata:
        seq_lens: torch.Tensor

    monkeypatch.setattr(loop_graph, "_metadata_fields", lambda md: {"seq_lens": md.seq_lens})
    dummy = Metadata(torch.zeros(2, dtype=torch.int32))
    buffers = loop_graph._FullGraphMetadata({"a": dummy, "b": dummy})
    live = Metadata(torch.tensor([11, 22], dtype=torch.int32))
    source = {"a": live, "b": live}

    first = buffers.stage(source)
    assert first["a"] is first["b"]
    assert first["a"].seq_lens.tolist() == [11, 22]
    target = first["a"].seq_lens

    live.seq_lens.add_(1)
    # The same prepared batch does not enqueue another refresh for the next op.
    assert buffers.stage(source) is first
    assert target.tolist() == [11, 22]

    assert buffers.stage(dict(source))["a"].seq_lens is target
    assert target.tolist() == [12, 23]


def test_full_graph_checks_borrowed_metadata_before_staging_temporaries(monkeypatch):
    pytest.importorskip("vllm")
    from vllm_rlt.worker import loop_graph

    @dataclass
    class Metadata:
        seq_lens: torch.Tensor
        temporary: torch.Tensor

    monkeypatch.setattr(
        loop_graph,
        "_metadata_fields",
        lambda md: {"seq_lens": md.seq_lens, "temporary": md.temporary},
    )
    owner = torch.zeros(4, dtype=torch.int32)
    dummy = Metadata(owner[:2], torch.zeros(2))
    buffers = loop_graph._FullGraphMetadata(
        {"a": dummy, "b": dummy}, {owner.untyped_storage().data_ptr()}
    )
    live = Metadata(owner[:2], torch.ones(2))
    staged = buffers.stage({"a": live, "b": live})["a"]
    assert staged.seq_lens is dummy.seq_lens
    assert staged.temporary is not live.temporary
    assert staged.temporary.tolist() == [1.0, 1.0]

    wrong = Metadata(owner[1:3], torch.full((2,), 9.0))
    assert buffers.stage({"a": wrong, "b": wrong}) is None
    assert buffers.stage({"a": live, "b": wrong}) is None
    assert staged.temporary.tolist() == [1.0, 1.0]
    assert buffers.stage({"a": live, "b": live})["a"] is staged


def test_piecewise_graph_keeps_fixed_conv_addresses_and_live_attention():
    pytest.importorskip("vllm")
    from vllm.models.inkling.nvidia.sconv_swa_attn import InklingSconvMetadata

    from vllm_rlt.worker.loop_graph import _PiecewiseGraphMetadata

    def conv(blocks):
        return InklingSconvMetadata(
            block_table=torch.tensor(blocks),
            slot_mapping=torch.tensor([1, 2, 3, 4]),
            seq_idx=torch.tensor([0, 0, 0, len(blocks) - 1]),
            query_start=torch.tensor([0, 0, 0, 3]),
        )

    single = conv([[1, 2]])
    bindings = _PiecewiseGraphMetadata({"conv": single, "alias": single}, max_reqs=3)
    fixed = bindings.stage({"conv": single, "alias": single})["conv"]

    mixed = conv([[5, 6], [7, 8]])
    attention = object()
    metadata = {"conv": mixed, "alias": mixed, "attention": attention}
    result = bindings.stage(metadata)
    assert result["conv"] is fixed and result["alias"] is fixed
    assert result["attention"] is attention
    assert bindings.stage(metadata) is result
    assert fixed.block_table.tolist() == [[5, 6], [7, 8], [0, 0]]
    assert fixed.seq_idx.tolist() == [0, 0, 0, 1]
    invalid = conv([[1], [2]])
    assert bindings.stage({"conv": invalid, "alias": invalid}) is None


def test_piecewise_graph_borrows_mrv2_table_rows_with_stable_addresses():
    pytest.importorskip("vllm")
    from dataclasses import replace

    from vllm.models.inkling.nvidia.sconv_swa_attn import InklingSconvMetadata

    from vllm_rlt.worker.loop_graph import _PiecewiseGraphMetadata

    table = torch.arange(6).reshape(3, 2)
    offsets = torch.tensor([0, 0, 0, 3])
    slots = torch.arange(4)
    sequence = torch.zeros(4, dtype=torch.int32)
    captured = InklingSconvMetadata(table[:1], slots, sequence, offsets)
    storage = frozenset(
        tensor.untyped_storage().data_ptr() for tensor in (table, slots, sequence, offsets)
    )
    bindings = _PiecewiseGraphMetadata({"conv": captured}, max_reqs=3, persistent_storage=storage)

    live = replace(captured, block_table=table[:2])
    runtime = {"conv": live, "attn": object()}
    assert bindings.stage(runtime) is runtime
    assert bindings.stage({"conv": replace(live, block_table=table[1:])}) is None
    assert bindings.stage({"conv": replace(live, slot_mapping=slots.clone())}) is None


def test_graph_input_staging_copies_only_live_stack_blocks():
    pytest.importorskip("vllm")
    from vllm_rlt.worker.loop_graph import _GraphEntry

    entry = object.__new__(_GraphEntry)
    entry.stack_input_blocks = 2
    entry.borrowed_positions = False
    entry.inputs = torch.full((4, 2), -1), torch.zeros(1), None

    entry.stage_inputs((torch.arange(8).reshape(4, 2), torch.ones(1), None))

    assert entry.inputs[0].tolist() == [[0, 1], [2, 3], [-1, -1], [-1, -1]]
    assert entry.inputs[1].tolist() == [1.0]


def test_full_graph_rejects_changed_positions_or_padding_address(monkeypatch):
    pytest.importorskip("vllm")
    from vllm_rlt.worker import loop_graph

    monkeypatch.setattr(loop_graph, "CUDAGraphWrapper", lambda function, *args: function)
    owner = SimpleNamespace(positions=torch.arange(4), is_padding=torch.zeros(4, dtype=torch.bool))
    positions = owner.positions[:2]
    padding = owner.is_padding[:2]
    inputs = (torch.zeros(2, 2), None, None, positions, torch.zeros(2))
    entry = loop_graph._GraphEntry(
        None,
        loop_graph.CUDAGraphMode.FULL,
        lambda *args: None,
        inputs,
        None,
        None,
        stack_input_blocks=1,
        padding=padding,
        persistent_inputs=owner,
    )
    assert entry.inputs[3] is positions and entry.padding is padding
    assert entry.matches_borrowed_inputs(inputs, owner.is_padding[:2])
    assert not entry.matches_borrowed_inputs(inputs, padding.clone())
    changed = (*inputs[:3], owner.positions[1:3], inputs[4])
    assert not entry.matches_borrowed_inputs(changed, padding)


@pytest.mark.parametrize("explicit_positions", (False, True))
def test_mixed_batch_reuses_mrv2_inputs_and_metadata_until_batch_changes(
    monkeypatch, explicit_positions,
):
    pytest.importorskip("vllm")
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    from vllm_rlt.models.inkling_plugin import (
        LoopInklingModelState,
        _InklingBatchState,
        _InklingState,
    )
    from vllm_rlt.worker.loop_model_runner import LoopModelRunner
    from vllm_rlt.worker.loop_state import LoopExecution, loop_execution

    calls = []

    def make_inputs(self, output, state, descriptor, num_loras):
        calls.append("inputs")
        return SimpleNamespace(req_ids=state.req_ids)

    def prepare_attn(self, batch):
        calls.append("block tables")
        return object(), object()

    def build_metadata(self, *args):
        calls.append("builder")
        return {}

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", make_inputs)
    monkeypatch.setattr(GPUModelRunner, "prepare_attn", prepare_attn)
    monkeypatch.setattr(DefaultModelState, "prepare_attn", build_metadata)
    runner = object.__new__(LoopModelRunner)
    runner._prepared_batch = None
    runner._loop_graphs = None
    runner._decode_pipeline = False
    runner.model = object()
    runner.pcp_manager = runner.ubatch_runner = runner.speculator = runner.fast_prefill = None
    runner.dcp_size = 1
    runner.req_states = SimpleNamespace(num_computed_tokens_np=[8, 0])
    batch = SimpleNamespace(
        req_ids=["decode", "prefill"],
        idx_mapping_np=[0, 1],
        num_scheduled_tokens=[1, 7],
        is_prefilling_np=[False, True],
        has_prefill=True,
        num_draft_tokens_np=None,
    )
    model_state = object.__new__(LoopInklingModelState)
    descriptor = SimpleNamespace(cg_mode=CUDAGraphMode.NONE)
    state_store = {}
    old_spans = (LoopSpan("decode", 8, 1, 0), LoopSpan("prefill", 0, 7, 0, prefill=True))

    def prepare(op, prefill_start, prefill_count):
        runner.req_states.num_computed_tokens_np[1] = prefill_start
        batch.num_scheduled_tokens[1] = prefill_count
        step = LoopExecution(
            op,
            (
                LoopSpan("decode", 8, 1, 0),
                LoopSpan("prefill", prefill_start, prefill_count, 0, prefill=True),
            ),
            state_store,
            {},
        )
        step.explicit_positions = explicit_positions
        token = loop_execution.set(step)
        try:
            inputs = runner.prepare_inputs(None, batch, descriptor, 0)
            blocks, slots = runner.prepare_attn(inputs)
            return inputs, model_state.prepare_attn(
                inputs, descriptor.cg_mode, blocks, slots, [], None
            )
        finally:
            loop_execution.reset(token)

    first_inputs, first_metadata = prepare(LoopOp.PRELUDE, 0, 7)
    assert runner._prepared_batch.keep_model_state is not explicit_positions
    stack = torch.arange(10).view(1, 10, 1)
    packed = _InklingBatchState(
        _InklingState(stack, stack[0], stack, None, 0, 0), old_spans, state_store
    )
    if not explicit_positions:
        runner._prepared_batch.packed_state = packed
    next_inputs, next_metadata = prepare(LoopOp.CORE, 0, 7)
    assert next_inputs is first_inputs and next_metadata is first_metadata
    assert runner._prepared_batch.packed_state is (None if explicit_positions else packed)
    assert not state_store
    assert calls == ["inputs", "block tables", "builder"]

    changed_inputs, changed_metadata = prepare(LoopOp.CORE, 0, 6)
    assert changed_inputs is not first_inputs and changed_metadata is not first_metadata
    assert runner._prepared_batch.packed_state is None
    if not explicit_positions:
        stack.fill_(-1)
        assert state_store["decode"][8].stack.flatten().tolist() == [0]
        assert state_store["prefill"][0].stack.flatten().tolist() == list(range(1, 8))
    assert calls == ["inputs", "block tables", "builder"] * 2

    spans = (old_spans[0], LoopSpan("prefill", 0, 6, 2, prefill=True))
    assert runner._input_rows(LoopExecution(LoopOp.CORE, spans, {}, {}), batch, 0) is None
    assert runner._input_rows(LoopExecution(LoopOp.CODA, spans, {}, {}), batch, 0) is not None

    # The same slot and graph width can belong to the next WCPB chunk.
    moved_inputs, moved_metadata = prepare(LoopOp.PRELUDE, 6, 6)
    assert moved_inputs is not changed_inputs and moved_metadata is not changed_metadata
    assert calls == ["inputs", "block tables", "builder"] * 3


@pytest.mark.parametrize(("rows", "expected"), ((1, 1), (3, 4), (15, 16), (16, 16), (17, None)))
def test_decode_padding_chooses_smallest_captured_bucket(rows, expected):
    pytest.importorskip("vllm")
    from vllm_rlt.worker.loop_graph import LoopCudaGraphs

    graphs = object.__new__(LoopCudaGraphs)
    graphs.full_decode_sizes = (1, 2, 4, 8, 16)

    assert graphs.padded_decode_size(rows) == expected


def test_worker_preserves_engine_graph_mode_for_scheduler(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.config import get_current_vllm_config
    from vllm.config.compilation import CUDAGraphMode

    from vllm_rlt.worker import loop_model_runner
    from vllm_rlt.worker.loop_worker import LoopWorker

    selected_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=selected_mode),
        parallel_config=SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1),
        cache_config=object(),
    )

    def runner(worker_config, device, mode):
        assert get_current_vllm_config() is worker_config
        assert worker_config.compilation_config.cudagraph_mode is CUDAGraphMode.NONE
        assert worker_config.cache_config is config.cache_config
        assert mode is selected_mode
        return worker_config

    monkeypatch.setattr(loop_model_runner, "LoopModelRunner", runner)
    worker = object.__new__(LoopWorker)
    worker.vllm_config = config
    worker.compilation_config = config.compilation_config
    worker.parallel_config = config.parallel_config
    worker.device = None

    assert worker._make_model_runner() is worker.vllm_config
    assert worker.compilation_config is worker.vllm_config.compilation_config
    assert config.compilation_config.cudagraph_mode is selected_mode


def test_graph_input_layout_ignores_only_singleton_strides():
    pytest.importorskip("vllm")
    from vllm_rlt.worker.loop_graph import _graph_input_layout

    captured = torch.empty(1, 1, 16)
    row = torch.empty(1, 4, 16)[:, 1:2].clone()
    assert row.stride() != captured.stride()
    assert _graph_input_layout(row) == _graph_input_layout(captured)
    assert _graph_input_layout(torch.empty(1, 1, 32)[:, :, ::2]) != _graph_input_layout(captured)


@pytest.mark.parametrize(("variable", "targets"), ((True, (3, 1, 2)), (False, (3,))))
def test_warmup_uses_separate_native_buckets_and_all_legal_decode_targets(
    monkeypatch, variable, targets
):
    pytest.importorskip("vllm")
    from vllm.config.compilation import CUDAGraphMode

    from vllm_rlt.worker.loop_graph import LoopCudaGraphs

    monkeypatch.setattr(torch.cuda, "Stream", object)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            decode_cudagraph_capture_request_sizes=[1, 2, 4, 8, 16],
            prefill_cudagraph_capture_token_sizes=[1, 64, 128, 256],
            max_cudagraph_capture_size=128,
            max_decode_cudagraph_capture_tokens=6,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=128),
        model_config=SimpleNamespace(max_model_len=100),
    )
    graphs = LoopCudaGraphs(config, CUDAGraphMode.FULL_AND_PIECEWISE)
    shapes = []

    def warm_shape(runner, requests, tokens, prefill, loop_targets):
        shapes.append((requests, tokens, prefill, loop_targets))
        entries = max(loop_targets) + 1 + len(loop_targets)
        if not prefill and tokens == 4:
            entries -= 1  # An incomplete bucket must not be used for FULL padding.
        mode = CUDAGraphMode.PIECEWISE if prefill else CUDAGraphMode.FULL
        for index in range(entries):
            graphs.entries[mode, index, tokens] = object()

    monkeypatch.setattr(graphs, "_warm_shape", warm_shape)
    graphs.warmup(object(), LoopLayout(3, True, True, variable))

    assert shapes == [(n, n, False, targets) for n in (6, 4, 2, 1)] + [
        (2, 128, True, (3,)),
        (1, 64, True, (3,)),
        (1, 1, True, (3,)),
    ]
    assert graphs.full_decode_sizes == (1, 2, 6)
    assert graphs.padded_decode_size(3) == 6
    assert graphs.piecewise_sizes == (1, 64, 128)
    assert graphs.padded_piecewise_size(63) == 64
    assert graphs.padded_piecewise_size(65) == 128
    assert graphs.padded_piecewise_size(129) is None


def test_piecewise_capture_matches_native_dummy_rows(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.config.compilation import CUDAGraphMode

    from vllm_rlt.worker import loop_graph
    from vllm_rlt.worker.loop_state import loop_execution

    def capture_inputs(requests, tokens, *args, max_query_len, **kwargs):
        assert (requests, tokens, max_query_len) == (2, 11, 6)
        return SimpleNamespace(attn_metadata=None, slot_mappings=None)

    def model(**kwargs):
        step = loop_execution.get()
        assert [span.token_count for span in step.ordered] == [5, 6]
        assert step.context_limit == 32
        calls.append(step.op)

    monkeypatch.setattr(loop_graph, "prepare_inputs_to_capture", capture_inputs)
    monkeypatch.setattr(loop_graph, "set_forward_context", lambda *args, **kwargs: nullcontext())
    inputs = SimpleNamespace(
        input_ids=torch.zeros(11),
        positions=torch.zeros(11),
        is_padding=torch.zeros(11, dtype=torch.bool),
    )
    runner = SimpleNamespace(
        get_model=lambda: model,
        input_buffers=inputs,
        vllm_config=None,
        model_state=None,
        block_tables=None,
        attn_groups=None,
        kv_cache_config=None,
        pcp_manager=None,
    )
    graphs = object.__new__(loop_graph.LoopCudaGraphs)
    graphs.mode = CUDAGraphMode.FULL_AND_PIECEWISE
    graphs.config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=32))
    calls = []

    graphs._warm_shape(runner, 2, 11, True, (2,))

    assert calls == [LoopOp.PRELUDE, LoopOp.CORE, LoopOp.CORE, LoopOp.CODA]


def test_explicit_position_update_preserves_mrv2_admission_staging(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_rlt.worker.loop_model_runner import LoopModelRunner
    from vllm_rlt.worker.loop_state import LoopExecution, loop_execution

    def keep_native_request_updates(self, output):
        self.req_states.num_computed_tokens_np[0] = 7

    def position_pool(dtype):
        return SimpleNamespace(copy_to_uva=lambda values: torch.tensor(values, dtype=dtype))

    monkeypatch.setattr(GPUModelRunner, "update_requests", keep_native_request_updates)
    runner = object.__new__(LoopModelRunner)
    runner._loop_position_slots = position_pool(torch.int64)
    runner._loop_positions = position_pool(torch.int32)
    # The native add_requests path owns the staging methods on this writer.
    gpu = torch.tensor([7, 0, 123], dtype=torch.int32)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"existing": 0, "admitted": 1},
        num_computed_tokens=SimpleNamespace(gpu=gpu),
        num_computed_tokens_np=[0, 0, 123],
        num_computed_prefill_tokens=[0, 0, 0],
        prefill_len=SimpleNamespace(np=[32, 64, 256]),
    )
    spans = (LoopSpan("existing", 39, 1, 0), LoopSpan("admitted", 0, 7, 0, prefill=True))
    step = LoopExecution(LoopOp.PRELUDE, spans, {}, {})
    step.explicit_positions = True
    token = loop_execution.set(step)
    try:
        runner.update_requests(None)
    finally:
        loop_execution.reset(token)

    assert runner.req_states.num_computed_tokens_np == [39, 0, 123]
    assert runner.req_states.num_computed_prefill_tokens == [32, 0, 0]
    assert gpu.tolist() == [39, 0, 123]


@pytest.mark.parametrize("packed", (False, True))
def test_cross_round_core_uses_distinct_positions_for_a_repeated_request_slot(monkeypatch, packed):
    pytest.importorskip("vllm")
    import numpy as np
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_rlt.core.output import LoopSchedulerOutput
    from vllm_rlt.worker.loop_model_runner import LoopModelRunner
    from vllm_rlt.worker.loop_state import LoopExecution, loop_execution

    spans = (LoopSpan("prompt", 0, 4, 1, prefill=True),
             LoopSpan("prompt", 4, 4, 0, prefill=True))
    runner = object.__new__(LoopModelRunner)
    runner._loop_graphs = runner._prepared_batch = None
    runner.model_state = SimpleNamespace(can_pack_core=lambda: packed)
    runner.decode_query_len = 1
    runner.req_states = SimpleNamespace(
        req_id_to_index={"prompt": 3}, prefill_len=SimpleNamespace(np=np.array([0, 0, 0, 8]))
    )
    step = LoopExecution(LoopOp.CORE, spans, {}, {})
    step.compact_core = True
    output = LoopSchedulerOutput.for_step(step.op, spans)
    native_descriptor = BatchExecutionDescriptor(CUDAGraphMode.NONE, 8, 1)

    def prepare(plan, batch, descriptor, *, positions):
        assert plan is output
        assert batch.idx_mapping_np.tolist() == [3, 3]
        assert batch.num_scheduled_tokens.tolist() == [4, 4]
        assert positions.tolist() == ([0, 4] if packed else [4, 0])
        assert descriptor.num_reqs is None and descriptor.num_tokens == 8
        return batch

    runner._ordinary_inputs = SimpleNamespace(prepare=prepare)
    step.explicit_positions = True
    token = loop_execution.set(step)
    try:
        batch, uniform = runner.gather_batch_req_state(output, False)
        assert uniform is None and step.ordered == (spans if packed else spans[::-1])
        assert step.packed_core is packed
        assert runner.prepare_inputs(output, batch, native_descriptor, 0) is batch
    finally:
        loop_execution.reset(token)


@pytest.mark.parametrize(
    ("explicit_positions", "prefill"), ((False, True), (True, True), (True, False)),
)
def test_graph_padding_preserves_actual_token_positions(monkeypatch, explicit_positions, prefill):
    pytest.importorskip("vllm")
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    from vllm_rlt.models.inkling_plugin import LoopInklingModelState
    from vllm_rlt.worker.loop_graph import LoopCudaGraphs
    from vllm_rlt.worker.loop_model_runner import LoopModelRunner
    from vllm_rlt.worker.loop_state import LoopExecution, loop_execution

    runner = object.__new__(LoopModelRunner)
    runner._prepared_batch = None
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    runner._decode_pipeline = False
    runner.model = object()
    runner.fast_prefill = runner.pcp_manager = runner.ubatch_runner = runner.speculator = None
    runner.dcp_size = 1
    runner._loop_graphs = object.__new__(LoopCudaGraphs)
    runner._loop_graphs.mode = CUDAGraphMode.FULL_AND_PIECEWISE
    runner._loop_graphs.piecewise_sizes = (8, 16, 32)
    runner._loop_graphs.full_decode_sizes = (1, 2, 4)
    spans = (
        (LoopSpan("a", 8, 1, 0), LoopSpan("b", 32, 14, 0, prefill=True))
        if prefill
        else (LoopSpan("a", 8, 1, 0), LoopSpan("b", 9, 1, 0), LoopSpan("c", 10, 1, 0))
    )
    real_positions = [
        position for span in spans for position in range(span.token_start, span.token_end)
    ]
    runner.req_states = SimpleNamespace(num_computed_tokens_np=[span.token_start for span in spans])
    state = SimpleNamespace(
        req_ids=[span.request_id for span in spans],
        idx_mapping_np=list(range(len(spans))),
        num_scheduled_tokens=[span.token_count for span in spans],
        is_prefilling_np=[span.prefill for span in spans],
        has_prefill=prefill,
        num_draft_tokens_np=None,
        num_tokens=len(real_positions),
    )
    expected = (
        BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 16, None)
        if prefill
        else BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 4, 1, 1)
    )

    def prepare_upstream(self, output, batch, descriptor, loras):
        assert descriptor == expected
        return SimpleNamespace(
            req_ids=state.req_ids,
            num_tokens=batch.num_tokens,
            input_ids=torch.ones(expected.num_tokens),
            positions=torch.tensor([*real_positions, -1]),
        )

    def prepare_metadata(self, input_batch, cudagraph_mode, *args):
        assert cudagraph_mode is expected.cg_mode
        return {}

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prepare_upstream)
    monkeypatch.setattr(DefaultModelState, "prepare_attn", prepare_metadata)
    model_state = object.__new__(LoopInklingModelState)
    step = LoopExecution(LoopOp.CORE, spans, {}, {})
    step.explicit_positions = explicit_positions
    token = loop_execution.set(step)
    try:
        desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, state.num_tokens, len(spans))
        inputs = runner.prepare_inputs(None, state, desc, 0)
        model_state.prepare_attn(inputs, desc.cg_mode, (), None, [], None)
    finally:
        loop_execution.reset(token)

    assert step.ordered == spans and step.padded_tokens == expected.num_tokens
    assert step.prepared_batch is not None
    assert step.prepared_batch.keep_model_state is not explicit_positions
    assert inputs.num_tokens == state.num_tokens
    assert inputs.positions.tolist() == [*real_positions, 0]
    assert inputs.input_ids.tolist() == [1] * state.num_tokens + [0]
    runner.req_states.num_computed_tokens_np[-1] -= 1
    assert runner._input_rows(step, state, 0) is None


@pytest.mark.parametrize(
    ("has_prefill", "merge", "compact", "merge_prefill", "prefill_pipeline"),
    (
        (False, False, False, False, False),
        (True, True, False, False, False),
        (False, True, False, False, False),
        (False, True, True, False, False),
        (True, True, True, False, False),
        (True, True, False, True, False),
        (False, True, False, False, True),
        (True, True, False, True, True),
    ),
)
def test_mixed_core_uses_variant_groups_with_one_logical_receipt(
    monkeypatch, has_prefill, merge, compact, merge_prefill, prefill_pipeline
):
    pytest.importorskip("vllm")
    from vllm_rlt.core.output import LoopSchedulerOutput
    from vllm_rlt.worker.loop_model_runner import LoopModelRunner
    from vllm_rlt.worker.loop_state import loop_execution

    runner = object.__new__(LoopModelRunner)
    runner.model = SimpleNamespace(rlt_loop_ops=True, rlt_partition_core_variants=True)
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    runner.vllm_config = SimpleNamespace(lora_config=None, kv_transfer_config=None)
    runner._prepared_batch = runner._loop_completion = None
    runner._loop_states = {}
    runner._draft_states = {}
    runner._prefill_pipeline = prefill_pipeline
    runner._decode_pipeline = runner._trace_steps = False
    runner._mixed_core = runner._packed_prefill = True
    runner._compact_mixed_core = compact
    runner._merge_decode_core = merge
    runner._merge_prefill_core = merge_prefill
    runner._decode_state_slots = False
    runner.pcp_manager = runner.ubatch_runner = runner.speculator = None
    runner.fast_prefill = None
    runner.dcp_size = 1
    runner._loop_graphs = None  # Partition even in eager mode.
    submissions = []
    merged = []
    packed = compact or (prefill_pipeline and has_prefill)

    def submit(output):
        assert loop_execution.get().spans == output.spans
        assert loop_execution.get().compact_core is packed
        merged.append(loop_execution.get().merged_core)
        submissions.append(output)

    monkeypatch.setattr(runner, "_submit_forward", submit)
    spans = (
        LoopSpan("a", 25, 1, 0, last_core=True),
        LoopSpan("b", 17, 1, 0),
        LoopSpan("c", 19, 1, 1, last_core=True),
        LoopSpan("d", 41, 1, 0, last_core=True),
    )
    if has_prefill:
        spans += (LoopSpan("e", 14 if prefill_pipeline else 0, 14, 0, prefill=True),)
    if prefill_pipeline and has_prefill:
        spans += (LoopSpan("e", 0, 14, 1, prefill=True),)
    output = LoopSchedulerOutput.for_step(LoopOp.CORE, spans)
    output.new_block_ids_to_zero = [11]
    runner.execute_model(output)

    groups = [tuple(step.num_scheduled_tokens) for step in submissions]
    if packed:
        assert groups == [tuple(dict.fromkeys(span.request_id for span in spans))]
        assert not any(merged)
    elif merge_prefill or (merge and not has_prefill):
        merged_ids = ("a", "b", "d", "e") if has_prefill else ("a", "b", "d")
        other_chunk = prefill_pipeline and has_prefill
        assert groups == [merged_ids, ("c", "e") if other_chunk else ("c",)]
        assert merged == [True, other_chunk]
    else:
        assert groups == [("a", "d"), ("b", "e") if has_prefill else ("b",), ("c",)]
        assert not any(merged)
    assert [step.new_block_ids_to_zero for step in submissions] == (
        [[11]] + [None] * (len(groups) - 1)
    )
    expected_req_ids = list(dict.fromkeys(span.request_id for span in spans))
    assert runner.sample_tokens(None).req_ids == expected_req_ids


def test_packed_inkling_cache_pages_alias_original_pass_storage():
    pytest.importorskip("vllm")
    from vllm_rlt.models.inkling_packed_cache import _PagedView

    storage = torch.zeros((6, 2, 2, 4))
    passes = (storage[4:], storage[:2], storage[2:4])
    view = _PagedView.from_passes(passes)
    assert view.first_pages == (4, 0, 2)
    view.cache[view.first_pages[0] + 1].fill_(7)
    assert passes[0][1].eq(7).all()
    assert passes[1].eq(0).all() and passes[2].eq(0).all()
    misaligned = storage.as_strided(passes[1].shape, passes[1].stride(), 1)
    assert _PagedView.from_passes((passes[1], misaligned)) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
def test_packed_core_inputs_blend_dtypes_and_zero_capture_padding():
    pytest.importorskip("vllm")
    from vllm.models.inkling2.nvidia.ops.norm import rms_norm

    from vllm_rlt.models.inkling_core_routes import pack_core_inputs

    width, hidden = 5, 1024
    first = torch.randn((1, 2, hidden), device="cuda")
    later = torch.randn((3, 1, hidden), device="cuda", dtype=torch.bfloat16)
    routes = torch.tensor([-1, 0, -2], dtype=torch.int64, device="cuda")
    weight = torch.ones((1, hidden), device="cuda")
    norm = SimpleNamespace(weight=weight, eps=1e-6)
    out = torch.empty((3, width, hidden), device="cuda", dtype=torch.bfloat16)

    pack_core_inputs(
        out, first, later, routes, 3, norm, normalize_first=True, first_is_full=False
    )

    first_expected = rms_norm(first[0], weight[0], norm.eps, torch.bfloat16)
    torch.testing.assert_close(out[:, 0], first_expected[0].expand(3, -1), rtol=1 / 128, atol=0)
    assert torch.equal(out[:, 1], later[:, 0])
    torch.testing.assert_close(out[:, 2], first_expected[1].expand(3, -1), rtol=1 / 128, atol=0)
    assert torch.count_nonzero(out[:, 3:]).item() == 0


def test_mixed_core_retains_owned_rows_from_a_decode_slot():
    pytest.importorskip("vllm")
    from vllm_rlt.models.inkling_plugin import (
        LoopInklingForConditionalGeneration,
        _InklingState,
        _InklingStateSlot,
    )

    rows = torch.arange(12).reshape(3, 4, 1)
    backing = _InklingState(rows, rows[0], rows[:1], rows[1:2], 0, 2)
    prefill = backing.rows(0, 2)
    spans = (LoopSpan("decode", 8, 1, 1), LoopSpan("prompt", 0, 2, 0, prefill=True))
    step = SimpleNamespace(state={
        "decode": {8: _InklingStateSlot(backing, 2)}, "prompt": {0: prefill},
    })

    decode, prompt = LoopInklingForConditionalGeneration._pop_core_states(step, spans)
    rows.zero_()

    assert decode.stack[:, 0, 0].tolist() == [2, 6, 10]
    assert decode.inputs.flatten().tolist() == [2]
    assert decode.bank.flatten().tolist() == [6]
    assert prompt is prefill
    assert not any(step.state.values())


def test_decode_state_slots_gather_by_mrv2_order_and_preserve_live_positions():
    pytest.importorskip("vllm")
    from vllm_rlt.models.inkling_plugin import _InklingDecodeStatePool, _InklingState

    source = torch.tensor([10, 20]).reshape(1, 2, 1).expand(3, 2, 1)
    first = _InklingState(source, source[0], source[:1], None, 1, 2)
    pool = _InklingDecodeStatePool(4)
    store = {}
    spans = (LoopSpan("a", 8, 1, 0), LoopSpan("b", 11, 1, 0))
    pool.store(first, spans, store, (2, 0), torch.tensor([2, 0]))

    slab = store["a"][8].state
    assert slab.select_rows(torch.tensor([0, 2])).stack[0, :, 0].tolist() == [20, 10]

    next_state = _InklingState(
        first.stack[:, :1] + 5, first.prelude[:1], first.inputs[:, :1], None, 1, 2
    )
    pool.store(next_state, (LoopSpan("a", 9, 1, 0),), store, (2,), torch.tensor([2]))
    assert store["a"][8].stack[:, 0, 0].tolist() == [10, 10, 10]
    assert slab.stack[:, 2, 0].tolist() == [15, 15, 15]


def test_packed_decode_writes_terminal_and_continuing_slots_in_request_order(monkeypatch):
    pytest.importorskip("vllm")
    from vllm_rlt.models import inkling_plugin

    state_cls = inkling_plugin._InklingState
    scheduler_state = {}
    slots = (2, 0, 3, 1)
    spans = (
        LoopSpan("a", 8, 1, 1, last_core=True), LoopSpan("b", 11, 1, 0),
        LoopSpan("c", 7, 1, 1), LoopSpan("d", 13, 1, 2, last_core=True),
    )
    step = SimpleNamespace(
        ordered=spans, state=scheduler_state,
        prepared_batch=SimpleNamespace(
            rows=tuple((span.request_id, slot) for span, slot in zip(spans, slots, strict=True)),
            inputs=SimpleNamespace(idx_mapping=torch.tensor(slots)),
        ),
    )
    stack = torch.arange(12).reshape(3, 4, 1)
    execution = SimpleNamespace(stack=stack, blocks=2, coda_inputs=stack[:1] + 100)
    states = []
    for index, span in enumerate(spans):
        row = stack[:, index:index + 1]
        states.append(state_cls(row, row[0], row[:1] + 20, row[1:2] if span.depth else None, 0, 0))
    monkeypatch.setattr(
        inkling_plugin, "async_tensor_h2d",
        lambda values, *, device, dtype: torch.tensor(values, device=device, dtype=dtype),
    )

    def continue_core(state, previous):
        state.inputs = previous if previous is not None else state.prelude.unsqueeze(0)
        state.bank = state.stack[1:2].clone()
        state.num_blocks = 0

    pool = inkling_plugin._InklingDecodeStatePool(4)
    adapter = SimpleNamespace(_continue_core=continue_core)
    inkling_plugin.LoopInklingForConditionalGeneration._store_packed_decode(
        adapter, step, pool, states, execution,
    )
    for span, slot, expected in zip(spans, slots, (100, 1, 22, 103), strict=True):
        stored = scheduler_state[span.request_id][span.token_start]
        assert stored.index == slot
        assert stored.rows().inputs.item() == expected


@pytest.mark.parametrize("prefill_count", (1, 4))
def test_merged_core_stages_both_routes_in_one_transfer(monkeypatch, prefill_count):
    pytest.importorskip("vllm")
    from vllm_rlt.models import inkling_plugin

    transfers = []

    def transfer(indices, *, device, dtype):
        transfers.append((indices, device, dtype))
        return torch.tensor(indices, dtype=dtype)

    monkeypatch.setattr(inkling_plugin, "async_tensor_h2d", transfer)
    step = SimpleNamespace(
        ordered=(
            LoopSpan("a", 5, 1, 0, last_core=True),
            LoopSpan("b", 7, prefill_count, 0, prefill=prefill_count > 1),
            LoopSpan("c", 9, 1, 0, last_core=True),
        )
    )
    continuing, terminal = inkling_plugin.LoopInklingForConditionalGeneration._prepare_core_routes(
        step, "cuda"
    )
    continuing_rows = list(range(1, prefill_count + 1))
    terminal_rows = [0, prefill_count + 1]
    assert transfers == [(continuing_rows + terminal_rows, "cuda", torch.long)]
    assert continuing[0] == [1] and continuing[1].tolist() == continuing_rows
    assert terminal[0] == [0, 2] and terminal[1].tolist() == terminal_rows


@pytest.mark.parametrize(
    ("first_op", "early"),
    [(None, False), (LoopOp.PRELUDE, True), (LoopOp.CORE, True), (LoopOp.CODA, False)],
)
def test_forward_retires_the_oldest_uva_consumer_before_slot_reuse(monkeypatch, first_op, early):
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    from vllm_rlt.models.inkling_plugin import LoopInklingModelState
    from vllm_rlt.worker.loop_model_runner import LoopModelRunner
    from vllm_rlt.worker.loop_state import LoopExecution, loop_execution

    calls = []
    event_ids = iter(range(2))
    stream = object()

    class Event:
        def __init__(self, *, blocking):
            assert blocking
            self.event_id = next(event_ids)

        def record(self, actual_stream):
            assert actual_stream is stream
            calls.append(("record", self.event_id))

        def synchronize(self):
            calls.append(("wait", self.event_id))

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(GPUModelRunner, "main_stream", property(lambda self: stream))
    model_state = object.__new__(LoopInklingModelState)

    def prepare_inputs(self, *args):
        calls.append(("prepare", 0))
        return {}

    def execute_model(self, plan, **kwargs):
        if plan == 0:
            model_state.prepare_inputs(None, None)
        calls.append(("submit", plan))

    monkeypatch.setattr(DefaultModelState, "prepare_inputs", prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", execute_model)
    runner = object.__new__(LoopModelRunner)
    runner.model = object()
    runner.ubatch_runner = None
    runner._early_uva_release = True
    runner._uva_capacity = 2
    runner._uva_pending = deque()
    runner._uva_current = None

    step = LoopExecution(first_op, (), {}, {}) if first_op is not None else None
    context = loop_execution.set(step)
    try:
        runner._submit_forward(0)
    finally:
        loop_execution.reset(context)
    prefix = (
        [("prepare", 0), ("record", 0), ("submit", 0)] if early else [("prepare", 0), ("submit", 0)]
    )
    assert calls == prefix

    runner._submit_forward(1)
    runner._submit_forward(2)

    assert calls == prefix + ([] if early else [("record", 0)]) + [
        ("submit", 1),
        ("record", 1),
        ("wait", 0),
        ("submit", 2),
    ]


@pytest.mark.parametrize(
    ("rows", "physical_tokens", "reuse"),
    (((0, 1), 8, True), ((1, 0), 8, False), ((0,), 8, False), ((0, 1), 5, False)),
)
def test_routed_state_reuse_requires_the_complete_frame_in_order(rows, physical_tokens, reuse):
    pytest.importorskip("vllm")
    from vllm_rlt.models.inkling_plugin import LoopInklingForConditionalGeneration, _InklingState

    stack = torch.arange(16).reshape(2, 8, 1)
    frame = _InklingState(stack, stack[0], stack[:1], None, 0, 0)
    spans = (LoopSpan("prefill", 32, 3, 1, prefill=True), LoopSpan("decode", 8, 2, 1))
    states = {}
    frame.store_rows(spans, states, copy=False, keep_frame=True)
    step = SimpleNamespace(
        ordered=tuple(spans[index] for index in rows),
        state=states,
        draft_state={},
        prepared_batch=None,
        padded_tokens=physical_tokens,
    )

    model = object.__new__(LoopInklingForConditionalGeneration)
    actual = model._take_state(step)
    assert (actual is frame) is reuse
    if not reuse:
        row_tokens = ([0, 1, 2], [3, 4])
        expected = [token for row in rows for token in row_tokens[row]]
        expected += [0] * (physical_tokens - len(expected))
        assert actual.prelude.flatten().tolist() == expected
        assert actual.stack[0].flatten().tolist() == expected


def test_prepared_decode_slots_reuse_native_block_tables(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_rlt.worker.loop_model_runner import LoopModelRunner
    from vllm_rlt.worker.loop_state import PreparedBatch

    def unexpected_slot_computation(*args):
        pytest.fail("the fused input path already prepared the slot mappings")

    monkeypatch.setattr(GPUModelRunner, "prepare_attn", unexpected_slot_computation)
    tables = (torch.zeros((4, 2), dtype=torch.int32),)
    slots = torch.tensor([[5, 8, 9, -1, -1, -1]])
    computed = torch.tensor([2, 8, 4])
    batch = SimpleNamespace(
        idx_mapping=torch.tensor([0, 2, 1]),
        num_reqs_after_padding=4,
        num_tokens_after_padding=4,
    )
    calls = []

    def gather(indices, *, num_reqs_padded, num_computed_tokens):
        assert indices is batch.idx_mapping
        assert num_reqs_padded == 4
        assert num_computed_tokens is computed
        calls.append(indices)
        return tables

    runner = object.__new__(LoopModelRunner)
    runner.block_tables = SimpleNamespace(gather_block_tables=gather, slot_mappings=slots)
    runner.req_states = SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=computed))
    runner._prepared_batch = PreparedBatch((), None, batch, decode_slots_prepared=True)

    attention = runner.prepare_attn(batch)
    assert attention[0] is tables
    assert attention[1].data_ptr() == slots.data_ptr()
    assert attention[1].tolist() == [[5, 8, 9, -1]]
    assert runner.prepare_attn(batch) is attention
    assert len(calls) == 1


def test_sconv_shares_query_layout_and_preserves_group_kv_metadata():
    pytest.importorskip("vllm")
    from vllm.models.inkling.nvidia.sconv_swa_attn import (
        InklingSconvMetadata,
        InklingSconvMetadataBuilder,
    )

    from vllm_rlt.models.inkling_plugin import (
        _sconv_sequence_metadata,
        _share_sconv_sequence_metadata,
    )

    calls = []

    def native_build(common_prefix_len, common, fast_build):
        calls.append(common)
        return InklingSconvMetadata(
            block_table=common.block_table_tensor,
            slot_mapping=common.slot_mapping,
            seq_idx=torch.zeros_like(common.slot_mapping),
            query_start=torch.ones_like(common.slot_mapping),
        )

    def make_builder():
        builder = object.__new__(InklingSconvMetadataBuilder)
        builder.build = native_build
        _share_sconv_sequence_metadata(builder)
        return builder

    def make_batch(query_offsets, block):
        return SimpleNamespace(
            query_start_loc=query_offsets,
            num_reqs=2,
            slot_mapping=torch.arange(6) + block,
            block_table_tensor=torch.tensor([[block]]),
        )

    first, second = make_builder(), make_builder()
    offsets = torch.tensor([0, 2, 6])
    first_group, next_group = make_batch(offsets, 2), make_batch(offsets, 3)
    other_layout = make_batch(torch.tensor([0, 3, 6]), 4)

    handle = _sconv_sequence_metadata.set({})
    try:
        native = first.build(0, first_group)
        shared = second.build(0, next_group)
        separate = second.build(0, other_layout)
    finally:
        _sconv_sequence_metadata.reset(handle)

    assert calls == [first_group, other_layout]
    assert shared.seq_idx is native.seq_idx
    assert shared.query_start is native.query_start
    assert shared.slot_mapping is next_group.slot_mapping
    assert shared.block_table is next_group.block_table_tensor
    assert separate.seq_idx is not shared.seq_idx
    assert second.build(0, next_group).seq_idx is not shared.seq_idx
