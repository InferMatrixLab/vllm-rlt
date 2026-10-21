"""CPU/GPU input preparation, reusable metadata and forward boundaries."""

import pytest
import torch

from tests.helpers import make_prefill_engine, tiny_ouro_config, tiny_ouro_model
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import ExecutionBatch, Operation, Request, StateSpan
from vllm_rlt.worker.buffers import PackedMetadataBuffer, Workspace, stage_metadata
from vllm_rlt.worker.metadata import KVMetadata


@pytest.mark.parametrize(
    "device,layout,backend,uva",
    [
        ("cpu", "shared", "torch", False),
        ("cpu", "last_exited", "torch", False),
        pytest.param("cuda", "shared", "triton", False, marks=pytest.mark.gpu),
        pytest.param("cuda", "last_exited", "triton", False, marks=pytest.mark.gpu),
        pytest.param("cuda", "last_exited", "flash_attn_4", False, marks=pytest.mark.gpu),
        pytest.param("cuda", "last_exited", "flash_attn_4", True, marks=pytest.mark.gpu),
    ],
)
def test_loop_preparation_reuses_growing_storage_and_only_saves_final_hidden(
    monkeypatch, device, layout, backend, uva
):
    torch.manual_seed(123)
    config = (
        tiny_ouro_config(hidden_size=256, intermediate_size=512, head_dim=64)
        if backend == "flash_attn_4"
        else tiny_ouro_config()
    )
    model = OuroForCausalLM(config).to(
        device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(128, 16 if backend == "flash_attn_4" else 2, layout=layout),
        execution_config=ExecutionConfig(
            atomic_prefill=True, static_buffers=True, pad_to_power_of_two=True, prefill_uva=uva
        ),
        exit_config=ExitConfig("ouro"),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
        attention_backend=backend,
    )
    runner = engine.model_runner
    cpu_prepare, gpu_prepare = runner.prepare_on_cpu, runner.prepare_on_gpu
    save, recurrent = runner._save, model.recurrent_prepared
    snapshots, saved, gates = [], [], []
    from vllm_rlt.worker import buffers, tensors

    def reject(*args, **kwargs):
        pytest.fail("operation crossed its input preparation boundary")

    def prepare_cpu(batch, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(engine.cache_manager, "_stage", reject)
            patch.setattr(tensors, "copy_tensor", reject)
            patch.setattr(buffers, "copy_tensor", reject)
            patch.setattr(PackedMetadataBuffer, "upload", reject)
            prepared = cpu_prepare(batch, **kwargs)
        for call in prepared.forwards:
            if call.forward.tokens is not None:
                assert call.forward.tokens.device.type == "cpu"
            for kv in call.forward.metadata:
                assert kv.position_ids.device.type == "cpu"
        return prepared

    def prepare_gpu(prepared):
        if prepared.batch.loop_count and prepared.forwards:
            with monkeypatch.context() as patch:
                patch.setattr(runner.metadata, "build", reject)
                patch.setattr(runner, "_prepare_loop_inputs", reject)
                patch.setattr(PackedMetadataBuffer, "store", reject)
                result = gpu_prepare(prepared)
            slot = (runner.input_buffers[torch.int64].pool.index - 1) % 2
            snapshots.append(
                (
                    runner.input_buffers[torch.int64].targets[slot],
                    result.packed_metadata.targets[torch.int64] if result.packed_metadata else None,
                    runner.submissions.slot,
                    result.packed_metadata.value_count if result.packed_metadata else 0,
                )
            )
            return result
        return gpu_prepare(prepared)

    def save_final(request, state):
        saved.append(request.request_id)
        save(request, state)

    def core(*args, **kwargs):
        gates.append(kwargs.get("compute_gate", True))
        return recurrent(*args, **kwargs)

    monkeypatch.setattr(runner, "prepare_on_cpu", prepare_cpu)
    monkeypatch.setattr(runner, "prepare_on_gpu", prepare_gpu)
    monkeypatch.setattr(runner, "_save", save_final)
    monkeypatch.setattr(model, "recurrent_prepared", core)
    outputs = []
    try:
        # Each pool slot sees reuse; slot 1 also sees growth followed by reuse.
        for i, length in enumerate((2, 2, 2, 5, 2, 5)):
            engine.add_request(
                str(i), list(range(1, length + 1)), SamplingParams(max_tokens=1, ignore_eos=True)
            )
            for _ in range(30):
                finished = [o for o in engine.step() if o.finished]
                if finished:
                    outputs.append(finished[0].token_ids)
                    break
            else:
                pytest.fail("request did not finish")
        assert saved == [str(i) for i in range(6)]
        assert gates and not any(gates)
        assert outputs[0] == outputs[1] == outputs[2] == outputs[4]
        assert outputs[3] == outputs[5]
        if uva:
            previous = {}
            reused = grew = False
            for _, tensor, slot, count in snapshots:
                if slot in previous:
                    if previous[slot].numel() >= count:
                        assert tensor is previous[slot]
                        reused = True
                    else:
                        assert tensor is not previous[slot]
                        grew = True
                previous[slot] = tensor
            assert reused and grew
        else:
            assert snapshots[0][0] is snapshots[2][0] is snapshots[4][0]
            assert snapshots[1][0] is not snapshots[3][0]
            assert snapshots[3][0] is snapshots[5][0]
    finally:
        engine.close()


@pytest.mark.gpu
def test_model_execution_consumes_selected_and_staged_graphs(monkeypatch):
    from vllm_rlt.worker import cuda_graph

    torch.manual_seed(123)
    model = OuroForCausalLM(tiny_ouro_config()).to(device="cuda", dtype=torch.bfloat16)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
        execution_config=ExecutionConfig(
            static_buffers=True, cuda_graphs=True, cuda_graph_max_batch_size=1
        ),
        attention_backend="triton",
    )
    runner = engine.model_runner
    execute = runner._execute

    def reject(*args, **kwargs):
        pytest.fail("graph selection, capture and input staging belong to preparation")

    def checked_execute(scheduled, prepared):
        with monkeypatch.context() as patch:
            patch.setattr(runner.graphs, "select", reject)
            patch.setattr(runner.graphs, "prepare_inputs", reject)
            patch.setattr(cuda_graph, "_copy_graph_inputs", reject)
            patch.setattr(cuda_graph, "_capture", reject)
            return execute(scheduled, prepared)

    monkeypatch.setattr(runner, "_execute", checked_execute)
    try:
        engine.add_request("a", [1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
        engine.add_request("b", [3, 4], SamplingParams(max_tokens=3, ignore_eos=True))
        finished = {}
        for _ in range(100):
            finished.update({o.request_id: o for o in engine.step() if o.finished})
            if len(finished) == 2:
                break
        assert len(finished["a"].token_ids) == 2
        assert len(finished["b"].token_ids) == 3
        assert runner.graphs.captures > 0 and runner.graphs.replays > 0
        assert runner.graphs.fallbacks > 0
    finally:
        engine.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("phase", ["prepare_on_cpu", "prepare_on_gpu", "_execute"])
def test_loop_input_failure_releases_ownership_before_pool_wrap(monkeypatch, device, phase):
    from vllm_rlt.request import ExecutionBatch, Operation, Request, StateSpan

    model = OuroForCausalLM(tiny_ouro_config()).to(device)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(64, 2),
        attention_backend="triton" if device == "cuda" else "torch",
    )
    runner = engine.model_runner
    request = Request("a", [1, 2], SamplingParams(max_tokens=1))
    assert engine.cache_manager.allocate("a", 3)
    batch = ExecutionBatch(
        Operation.PRELUDE,
        (StateSpan(request, 0, 2, 0, input_tokens=(1, 2)),),
        loop_count=model.config.total_ut_steps,
    )
    original = getattr(runner, phase)

    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        if device == "cuda" and phase == "prepare_on_gpu":
            torch.cuda._sleep(5_000_000)
            result.forwards[0].forward.tokens.fill_(42)
        raise RuntimeError("injected loop input failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(runner, phase, fail)
            with pytest.raises(RuntimeError, match="injected loop input failure"):
                runner.submit(batch=batch)
        assert all(event.query() for event in runner.submissions.events)
        for _ in range(2):
            runner.submit(batch=batch).collect()
        assert torch.isfinite(request.hidden_state).all()
        assert all(event.query() for event in runner.submissions.events)
    finally:
        engine.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_query_spans_and_static_padding_share_metadata_contract(monkeypatch, device):
    cache = KVCacheManager(1, 1, 8, 64, 4, 4, device=device)
    requests = [Request(name, list(range(12)), SamplingParams()) for name in ("a", "b", "c")]
    for request in requests:
        assert cache.allocate(request.request_id, 12)
    batch = ExecutionBatch(
        Operation.RECURRENT,
        (
            StateSpan(requests[0], 3, 2, 1),
            StateSpan(requests[1], 7, 1, 2),
            StateSpan(requests[2], 1, 3, 0),
        ),
    )
    ordinary = cache.prepare_recurrent(batch)
    ids, depths, positions = batch.addresses()
    workspace = Workspace(8, 3, 8, torch.device(device), torch.float32)

    def reject(*args, **kwargs):
        pytest.fail("metadata calculation must not allocate tensors or transfer data")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", reject)
        patch.setattr(torch, "empty", reject)
        patch.setattr(cache, "_stage", reject)
        layout = cache.metadata_builder.build(ids, (depths,), positions)[0]
    static = workspace.store_metadata(layout, 8)
    static = workspace.upload_metadata(static)
    addresses = {name: tensor.data_ptr() for name, tensor in workspace.gpu.items()}
    workspace.record_done()
    workspace.acquire()
    static = workspace.upload_metadata(workspace.store_metadata(layout, 8))
    assert addresses == {name: tensor.data_ptr() for name, tensor in workspace.gpu.items()}
    assert type(ordinary) is type(static) is KVMetadata
    assert ordinary.position_ids.tolist() == [3, 4, 7, 1, 2, 3]
    assert ordinary.context_lengths.tolist() == [4, 5, 8, 2, 3, 4]
    assert [depth for _, depth, _ in ordinary.rows] == [1, 1, 2, 0, 0, 0]
    for name in ("position_ids", "write_blocks", "write_offsets", "context_lengths"):
        assert torch.equal(getattr(ordinary, name), getattr(static, name)[:6])
    assert static.position_ids.shape == (8,)
    assert static.write_blocks.shape == static.write_offsets.shape == (6,)
    assert static.context_lengths[6:].tolist() == [0, 0]
    # Padding has no KV address, even though the model produced eight rows.
    cache.key_cache.fill_(-1)
    cache.value_cache.fill_(-1)
    values = torch.arange(8 * 8, dtype=torch.float32, device=device).view(8, 1, 8)
    cache._write_prepared(0, static, values, values)
    assert torch.equal(cache.key_cache[static.write_blocks, 0, static.write_offsets], values[:6])
    assert (cache.key_cache != -1).sum().item() == 6 * 8


@pytest.mark.gpu
@pytest.mark.parametrize("use_uva", [False, True])
@pytest.mark.parametrize("single_query", [False, True])
def test_packed_metadata_supports_query_lengths_and_per_request_depths(use_uva, single_query):
    cache = KVCacheManager(
        1, 1, 64, 64, 16, 4, device="cuda", dtype=torch.bfloat16, backend="flash_attn_4"
    )
    for name in ("a", "b", "c"):
        assert cache.allocate(name, 16)
    if single_query:
        ids, positions = ["a", "b", "c"], [4, 7, 2]
        depth_sets = [[1, 2, 0], [2, 3, 1]]
        cumulative = [[0, 1, 2, 3], [0, 1, 2, 3]]
        lengths = [[5, 8, 3], [5, 8, 3]]
    else:
        ids, positions = ["a", "a", "b", "c", "c"], [3, 4, 7, 1, 2]
        # The second view splits A into two distinct request/depth sequences.
        depth_sets = [[1, 1, 2, 0, 0], [2, 3, 3, 1, 1]]
        cumulative = [[0, 2, 3, 5], [0, 1, 2, 3, 5]]
        lengths = [[5, 8, 3], [4, 5, 8, 3]]
    buffer = PackedMetadataBuffer(cache.device, use_uva=use_uva)
    builder = cache.metadata_builder
    # No token payload is necessary to build KV metadata.
    buffer.store(builder.build(ids, depth_sets, positions, packed=True))
    tokens, actual = buffer.upload()
    assert tokens.numel() == 0
    expected = stage_metadata(builder.build(ids, depth_sets, positions, packed=True), cache._stage)
    assert len(actual) == 2  # Requested views, not cache.max_loops.
    for index, (metadata, reference) in enumerate(zip(actual, expected)):
        assert type(metadata) is KVMetadata
        assert metadata.cu_seqlens_q.cpu().tolist() == cumulative[index]
        assert metadata.context_lengths.cpu().tolist() == lengths[index]
        assert [depth for _, depth, _ in metadata.rows] == depth_sets[index]
        for name in (
            "position_ids",
            "write_blocks",
            "write_offsets",
            "block_tables",
            "context_lengths",
            "cu_seqlens_q",
        ):
            assert torch.equal(getattr(metadata, name), getattr(reference, name)), name
    assert actual[0].position_ids.data_ptr() == actual[1].position_ids.data_ptr()
    if single_query:
        assert actual[0].cu_seqlens_q.data_ptr() == actual[1].cu_seqlens_q.data_ptr()
    # Grow the number of requested views independently of token count.
    torch.cuda.synchronize()
    tokens = [2**54 + index for index in range(len(ids))]
    buffer.store(builder.build(ids, depth_sets + depth_sets[:1], positions, packed=True), tokens)
    output, metadata = buffer.upload()
    assert len(metadata) == 3
    assert output.cpu().tolist() == tokens


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("operation", [Operation.PRELUDE, Operation.RECURRENT])
def test_cpu_preparation_defers_transfers_and_hidden_gather(monkeypatch, device, operation):
    from vllm_rlt.request import ExecutionBatch, StateSpan
    from vllm_rlt.worker.tensors import TensorTransfer, UvaBuffer

    e = LLMEngine(
        tiny_ouro_model().to(device),
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(atomic_prefill=False),
        attention_backend="triton" if device == "cuda" else "torch",
    )
    runner = e.model_runner
    prepared = None
    try:
        e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
        request = e.scheduler.requests["a"]
        assert e.cache_manager.allocate("a", 4)
        slot = runner._ensure_request_slot(request)
        runner.prefill_states[slot, :2].fill_(7)
        batch = ExecutionBatch(
            operation, (StateSpan(request, 0, 2, 0, chunk_start=0, retain_chunk=True),)
        )

        def unexpected(*args, **kwargs):
            pytest.fail("CPU preparation submitted device work")

        with monkeypatch.context() as patch:
            patch.setattr(TensorTransfer, "submit", unexpected)
            patch.setattr(UvaBuffer, "stage", unexpected)
            patch.setattr(e.cache_manager, "prepare_recurrent", unexpected)
            patch.setattr(runner, "_gather_prefill", unexpected)
            prepared = runner.prepare_on_cpu(batch)
        assert prepared.kv is None
        assert prepared.batch_inputs.hidden is None
        prepared = runner.prepare_on_gpu(prepared)
        if operation == Operation.PRELUDE:
            assert prepared.batch_inputs.token_ids.cpu().tolist() == [1, 2]
        else:
            assert prepared.kv.position_ids.cpu().tolist() == [0, 1]
            assert torch.all(prepared.batch_inputs.hidden.cpu() == 7)
    finally:
        if device == "cuda":
            torch.cuda.synchronize()
        e.close()


def test_cpu_preparation_failure_returns_acquired_buffer(monkeypatch):
    from vllm_rlt.request import ExecutionBatch, StateSpan

    e = make_prefill_engine()
    runner = e.model_runner
    e.add_request("a", [1, 2], SamplingParams(max_tokens=1))
    request = e.scheduler.requests["a"]
    batch = ExecutionBatch(
        Operation.PRELUDE, (StateSpan(request, 0, 2, 0, chunk_start=0, retain_chunk=True),)
    )
    original = runner.prepare_on_cpu

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected CPU preparation failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(runner, "prepare_on_cpu", fail)
            with pytest.raises(RuntimeError, match="CPU preparation failure"):
                runner.submit(batch=batch)
        assert not runner.submissions.events
        runner.submit(batch=batch).collect()
    finally:
        e.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_prelude_gpu_preparation_uses_cpu_snapshot_and_preserves_new_token_reference(device):
    from vllm_rlt.request import ExecutionBatch, StateSpan

    e = LLMEngine(
        tiny_ouro_model().to(device),
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(static_buffers=True, pad_to_power_of_two=True),
        attention_backend="triton" if device == "cuda" else "torch",
    )
    try:
        requests = [Request(str(i), [1], SamplingParams()) for i in range(3)]
        requests[0].generated_token_ids.append(7)
        requests[1].input_token_tensor = torch.tensor(8, device=device)
        replacement = torch.tensor(18, device=device)
        batch = ExecutionBatch(
            Operation.PRELUDE,
            (
                StateSpan(requests[0], 1, 1, 0),
                StateSpan(requests[1], 1, 1, 0),
                StateSpan(requests[2], 1, 1, 0, input_tokens=(9,)),
            ),
        )
        runner = e.model_runner
        prepared = runner.prepare_on_cpu(batch)
        assert prepared.token_ids is None
        requests[0].generated_token_ids[-1] = 17
        requests[1].input_token_tensor = replacement
        prepared = runner.prepare_on_gpu(prepared)
        assert prepared.token_ids.cpu().tolist() == [7, 8, 9, 0]
        assert requests[1].input_token_tensor is replacement
    finally:
        e.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("fail_preparation", [False, True])
def test_prelude_workspace_owned_once_across_gpu_prepare_and_forward(
    monkeypatch, device, fail_preparation
):
    from vllm_rlt.request import ExecutionBatch, StateSpan

    e = LLMEngine(
        tiny_ouro_model().to(device),
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(static_buffers=True),
        attention_backend="triton" if device == "cuda" else "torch",
    )
    runner = e.model_runner
    acquired, recorded, forwarded = [], [], []
    for workspace in runner.requestspaces["boundary"]:
        acquire, record_done = workspace.acquire, workspace.record_done

        def track_acquire(workspace=workspace, acquire=acquire):
            acquired.append(workspace)
            return acquire()

        def track_done(workspace=workspace, record_done=record_done):
            recorded.append(workspace)
            return record_done()

        monkeypatch.setattr(workspace, "acquire", track_acquire)
        monkeypatch.setattr(workspace, "record_done", track_done)
    e.add_request("a", [1], SamplingParams())
    request = e.scheduler.requests["a"]
    request.generated_token_ids.append(7)
    batch = ExecutionBatch(Operation.PRELUDE, (StateSpan(request, 1, 1, 0),))
    original_prepare, original_execute = runner.prepare_on_gpu, runner._execute

    def prepare(value):
        result = original_prepare(value)
        assert result.token_ids.data_ptr() == result.workspace.gpu["tokens"].data_ptr()
        if fail_preparation:
            raise RuntimeError("injected token preparation failure")
        return result

    def execute(scheduled, prepared):
        assert prepared.workspace is acquired[-1]
        forwarded.append(prepared.token_ids)
        # No input organization is allowed to move back into forward.
        with monkeypatch.context() as patch:
            patch.setattr(runner, "_prepare_tokens", lambda *args: pytest.fail("late tokens"))
            return original_execute(scheduled, prepared)

    monkeypatch.setattr(runner, "prepare_on_gpu", prepare)
    monkeypatch.setattr(runner, "_execute", execute)
    try:
        if fail_preparation:
            with pytest.raises(RuntimeError, match="token preparation failure"):
                runner.submit(batch=batch)
            assert not forwarded
        else:
            runner.submit(batch=batch).collect()
            assert forwarded[0].cpu().tolist() == [7]
        assert len(acquired) == 1
        assert recorded == acquired
        monkeypatch.setattr(runner, "prepare_on_gpu", original_prepare)
        runner.submit(batch=batch).collect()
        assert len(acquired) == 2
        assert recorded == acquired
    finally:
        e.close()


@pytest.mark.parametrize(
    "device,static_buffers,async_scheduling",
    [
        ("cpu", False, False),
        ("cpu", True, False),
        pytest.param("cuda", False, False, marks=pytest.mark.gpu),
        pytest.param("cuda", True, False, marks=pytest.mark.gpu),
        pytest.param("cuda", True, True, marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    "layout,atomic", [("last_exited", False), ("last_exited", True), ("shared", True)]
)
def test_forward_consumes_prepared_inputs(
    monkeypatch, device, layout, atomic, static_buffers, async_scheduling
):
    e = LLMEngine(
        tiny_ouro_model().to(device),
        cache_config=CacheConfig(128, 2, layout=layout),
        exit_config=ExitConfig("random_lookahead"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=3
        ),
        execution_config=ExecutionConfig(
            atomic_prefill=atomic,
            static_buffers=static_buffers,
            pad_to_power_of_two=static_buffers,
            async_scheduling=async_scheduling,
        ),
        attention_backend="triton" if device == "cuda" else "torch",
    )
    runner = e.model_runner
    cpu_prepare, gpu_prepare = runner.prepare_on_cpu, runner.prepare_on_gpu
    execute = runner._execute
    operations = set()

    def reject(*args, **kwargs):
        pytest.fail("model execution must consume already prepared inputs")

    def checked_cpu(batch):
        from vllm_rlt.worker import buffers
        from vllm_rlt.worker.async_state import RoutingBuffers

        with monkeypatch.context() as patch:
            patch.setattr(buffers, "copy_tensor", reject)
            patch.setattr(e.cache_manager, "_stage", reject)
            patch.setattr(buffers.Workspace, "upload_metadata", reject)
            patch.setattr(buffers.PackedMetadataBuffer, "upload", reject)
            patch.setattr(RoutingBuffers, "materialize_metadata", reject)
            return cpu_prepare(batch)

    def checked_gpu(prepared):
        from vllm_rlt.worker import buffers

        with monkeypatch.context() as patch:
            patch.setattr(e.cache_manager, "_validate_rows", reject)
            for name in ("build", "resolve_rows", "_write_addresses"):
                patch.setattr(runner.metadata, name, reject)
            patch.setattr(buffers.Workspace, "store_metadata", reject)
            patch.setattr(buffers.PackedMetadataBuffer, "store", reject)
            return gpu_prepare(prepared)

    def checked_execute(scheduled, prepared):
        operations.add(prepared.batch.operation)
        forwards = [entry.forward for entry in prepared.forwards]
        assert all(isinstance(forward.metadata, tuple) for forward in forwards)
        with monkeypatch.context() as patch:
            for name in (
                "_prepare_tokens",
                "_load_hidden",
                "_gather_prefill",
                "_prepare_loop_inputs",
            ):
                patch.setattr(runner, name, reject)
            for name in ("_prepare_batch", "_prepare_batches", "prepare_recurrent"):
                patch.setattr(e.cache_manager, name, reject)
            for name in ("build", "resolve_rows", "prepare_device"):
                patch.setattr(runner.metadata, name, reject)
            from vllm_rlt.worker import buffers
            from vllm_rlt.worker.async_state import RoutingBuffers

            for name in ("store_metadata", "upload_metadata"):
                patch.setattr(buffers.Workspace, name, reject)
            for name in ("store", "upload"):
                patch.setattr(buffers.PackedMetadataBuffer, name, reject)
            patch.setattr(RoutingBuffers, "materialize_metadata", reject)
            return execute(scheduled, prepared)

    monkeypatch.setattr(runner, "prepare_on_cpu", checked_cpu)
    monkeypatch.setattr(runner, "prepare_on_gpu", checked_gpu)
    monkeypatch.setattr(runner, "_execute", checked_execute)
    try:
        params = SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True)
        e.add_request("a", [1, 2, 3], params)
        e.add_request("b", [4, 5], params)
        e.add_request("c", [6], params)
        finished = set()
        for _ in range(100):
            finished.update(output.request_id for output in e.step() if output.finished)
            if len(finished) == 3:
                break
        assert finished == {"a", "b", "c"}
        assert operations == {Operation.PRELUDE, Operation.RECURRENT, Operation.CODA}
    finally:
        e.close()
