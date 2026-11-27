from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.core.loop import LoopLayout, LoopOp, LoopSpan


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


def test_mixed_batch_reuses_mrv2_inputs_and_metadata_until_batch_changes(monkeypatch):
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
    runner._wavefront_prefill = runner._decode_pipeline = False
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

    def prepare(op, prefill_count):
        step = LoopExecution(
            op,
            (LoopSpan("decode", 8, 1, 0), LoopSpan("prefill", 0, prefill_count, 0, prefill=True)),
            state_store,
            {},
        )
        token = loop_execution.set(step)
        try:
            inputs = runner.prepare_inputs(None, batch, descriptor, 0)
            blocks, slots = runner.prepare_attn(inputs)
            return inputs, model_state.prepare_attn(
                inputs, descriptor.cg_mode, blocks, slots, [], None
            )
        finally:
            loop_execution.reset(token)

    first_inputs, first_metadata = prepare(LoopOp.PRELUDE, 7)
    stack = torch.arange(10).view(1, 10, 1)
    packed = _InklingBatchState(
        _InklingState(stack, stack[0], stack, None, 0, 0), old_spans, state_store
    )
    runner._prepared_batch.packed_state = packed
    next_inputs, next_metadata = prepare(LoopOp.CORE, 7)
    assert next_inputs is first_inputs and next_metadata is first_metadata
    assert runner._prepared_batch.packed_state is packed and not state_store
    assert calls == ["inputs", "block tables", "builder"]

    batch.num_scheduled_tokens[1] = 6
    changed_inputs, changed_metadata = prepare(LoopOp.CORE, 6)
    assert changed_inputs is not first_inputs and changed_metadata is not first_metadata
    assert runner._prepared_batch.packed_state is None
    stack.fill_(-1)
    assert state_store["decode"][8].stack.flatten().tolist() == [0]
    assert state_store["prefill"][0].stack.flatten().tolist() == list(range(1, 8))
    assert calls == ["inputs", "block tables", "builder"] * 2

    spans = (old_spans[0], LoopSpan("prefill", 0, 6, 2, prefill=True))
    assert runner._input_rows(LoopExecution(LoopOp.CORE, spans, {}, {}), batch, 0) is None
    assert runner._input_rows(LoopExecution(LoopOp.CODA, spans, {}, {}), batch, 0) is not None


@pytest.mark.parametrize(("rows", "expected"), ((1, 1), (3, 4), (15, 16), (16, 16), (17, None)))
def test_decode_padding_chooses_smallest_captured_bucket(rows, expected):
    pytest.importorskip("vllm")
    from vllm_rlt.worker.loop_graph import LoopCudaGraphs

    graphs = object.__new__(LoopCudaGraphs)
    graphs.full_decode_sizes = (1, 2, 4, 8, 16)

    assert graphs.padded_decode_size(rows) == expected


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


def test_piecewise_pads_only_the_physical_token_count(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_rlt.worker.loop_graph import LoopCudaGraphs
    from vllm_rlt.worker.loop_model_runner import LoopModelRunner
    from vllm_rlt.worker.loop_state import LoopExecution, loop_execution

    runner = object.__new__(LoopModelRunner)
    runner._prepared_batch = None
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    runner._loop_graphs = object.__new__(LoopCudaGraphs)
    runner._loop_graphs.mode = CUDAGraphMode.FULL_AND_PIECEWISE
    runner._loop_graphs.piecewise_sizes = (8, 16, 32)
    monkeypatch.setattr(
        runner, "_input_rows", lambda *args: (("a", 0, 8, 1, False), ("b", 1, 0, 14, True))
    )

    def prepare_upstream(self, output, batch, descriptor, loras):
        assert descriptor == BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 16, None)
        return SimpleNamespace(
            req_ids=["a", "b"],
            num_tokens=batch.num_tokens,
            input_ids=torch.ones(16),
            positions=torch.ones(16),
        )

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prepare_upstream)
    spans = (LoopSpan("a", 8, 1, 0), LoopSpan("b", 0, 14, 0, prefill=True))
    step = LoopExecution(LoopOp.CORE, spans, {}, {})
    token = loop_execution.set(step)
    try:
        batch = SimpleNamespace(num_tokens=15, has_prefill=True)
        desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, 15, 2)
        inputs = runner.prepare_inputs(None, batch, desc, 0)
    finally:
        loop_execution.reset(token)

    assert step.ordered == spans and step.padded_tokens == 16
    assert inputs.num_tokens == 15
    assert inputs.input_ids.tolist() == inputs.positions.tolist() == [1] * 15 + [0]


@pytest.mark.parametrize(
    ("has_prefill", "merge", "compact", "merge_prefill"),
    (
        (False, False, False, False),
        (True, True, False, False),
        (False, True, False, False),
        (False, True, True, False),
        (True, True, True, False),
        (True, True, False, True),
    ),
)
def test_mixed_core_uses_variant_groups_with_one_logical_receipt(
    monkeypatch, has_prefill, merge, compact, merge_prefill
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
    runner._wavefront_prefill = runner._decode_pipeline = runner._trace_steps = False
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

    def submit(output):
        assert loop_execution.get().spans == output.spans
        assert loop_execution.get().compact_core is compact
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
        spans += (LoopSpan("e", 0, 14, 0, prefill=True),)
    output = LoopSchedulerOutput.for_step(LoopOp.CORE, spans)
    output.new_block_ids_to_zero = [11]
    runner.execute_model(output)

    groups = [tuple(step.num_scheduled_tokens) for step in submissions]
    if compact:
        assert groups == [tuple(span.request_id for span in spans)]
        assert not any(merged)
    elif merge_prefill or (merge and not has_prefill):
        merged_ids = ("a", "b", "d", "e") if has_prefill else ("a", "b", "d")
        assert groups == [merged_ids, ("c",)]
        assert merged == [True, False]
    else:
        assert groups == [("a", "d"), ("b", "e") if has_prefill else ("b",), ("c",)]
        assert not any(merged)
    assert [step.new_block_ids_to_zero for step in submissions] == (
        [[11]] + [None] * (len(groups) - 1)
    )
    expected_req_ids = [span.request_id for span in spans]
    assert runner.sample_tokens(None).req_ids == expected_req_ids


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
