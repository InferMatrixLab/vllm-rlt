"""FlashInfer FA2 adapter: SM89 selection, CSR metadata and eager decode."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
)
from vllm_rlt.attention import BackendCapabilities, create_backend, validate_backend_name
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.kernels import flashinfer_fa2_attention as fa2
from vllm_rlt.kernels.flashinfer_fa2_attention import FlashInferFA2PagedAttention
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.models import OuroConfig, OuroForCausalLM


@pytest.fixture
def fake_flashinfer(monkeypatch):
    package = types.ModuleType("flashinfer")
    package.__path__ = []
    decode = types.ModuleType("flashinfer.decode")

    class Wrapper:
        def __init__(self, *args, **kwargs):
            pass

    decode.BatchDecodeWithPagedKVCacheWrapper = Wrapper
    decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper = Wrapper
    package.decode = decode
    monkeypatch.setitem(sys.modules, "flashinfer", package)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode)
    monkeypatch.setattr(fa2, "version", lambda name: "0.7.2.post1")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))


def test_backend_name_and_explicit_selection(fake_flashinfer):
    validate_backend_name("flashinfer_fa2")
    selected = create_backend("flashinfer_fa2", torch.device("cuda"), torch.bfloat16, 128, 16)
    assert selected.info["selected_backend"] == "flashinfer_fa2"
    capabilities = selected.capabilities
    assert capabilities.cuda_graphs
    assert capabilities.async_scheduling
    assert not capabilities.packed_prefill
    assert not capabilities.device_context_lengths
    assert selected.info["cuda_graphs"] and selected.info["async_scheduling"]
    create_backend(
        "flashinfer_fa2",
        torch.device("cuda"),
        torch.bfloat16,
        128,
        16,
        required_capabilities=BackendCapabilities(cuda_graphs=True, async_scheduling=True),
    )
    with pytest.raises(ValueError, match="device_context_lengths"):
        create_backend(
            "flashinfer_fa2",
            torch.device("cuda"),
            torch.bfloat16,
            128,
            16,
            required_capabilities=BackendCapabilities(device_context_lengths=True),
        )


def test_constructor_rejects_unvalidated_shapes(fake_flashinfer, monkeypatch):
    with pytest.raises(ValueError, match="SM89"):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
        FlashInferFA2PagedAttention(torch.device("cuda"), torch.bfloat16, 128, 16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    with pytest.raises(ValueError, match="bfloat16"):
        FlashInferFA2PagedAttention(torch.device("cuda"), torch.float16, 128, 16)
    with pytest.raises(ValueError, match="block-size"):
        FlashInferFA2PagedAttention(torch.device("cuda"), torch.bfloat16, 128, 32)
    with pytest.raises(ValueError, match="head_dim"):
        FlashInferFA2PagedAttention(torch.device("cuda"), torch.bfloat16, 64, 16)


def test_missing_package_never_falls_back(fake_flashinfer, monkeypatch):
    def missing(name):
        raise fa2.PackageNotFoundError(name)

    monkeypatch.setattr(fa2, "version", missing)
    with pytest.raises(ImportError, match="flashinfer-python"):
        create_backend("flashinfer_fa2", torch.device("cuda"), torch.bfloat16, 128, 16)


def test_call_rejects_row_mismatch(fake_flashinfer):
    kernel = create_backend("flashinfer_fa2", torch.device("cuda"), torch.bfloat16, 128, 16).kernel
    q = torch.empty(3, 4, 128, dtype=torch.bfloat16)
    tables = torch.zeros(2, 4, dtype=torch.int32)
    lengths = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(ValueError, match="one metadata row per query row"):
        kernel(q, None, None, tables, lengths)


def test_wrapper_rejects_unsupported_head_ratio(fake_flashinfer):
    kernel = create_backend("flashinfer_fa2", torch.device("cuda"), torch.bfloat16, 128, 16).kernel
    with pytest.raises(ValueError, match="head ratio"):
        kernel._wrapper(2, 3, 1, 4)


@pytest.mark.gpu
def test_strided_gqa_padding_and_zero_rows():
    torch.manual_seed(7)
    page, head_dim, q_heads, kv_heads = 16, 128, 4, 2
    # Exactly KVCacheManager.key_cache[:, layer]: non-contiguous block stride.
    keys = torch.randn(12, 3, page, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)[:, 1]
    values = torch.randn(12, 3, page, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)[:, 1]
    q = torch.randn(4, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor(
        [[7, 2, 9], [3, 0, 0], [1, 8, 6], [0, 0, 0]], dtype=torch.int32, device="cuda"
    )
    lengths = torch.tensor([35, 1, 47, 0], dtype=torch.int32, device="cuda")
    attention = FlashInferFA2PagedAttention(torch.device("cuda"), torch.bfloat16, head_dim, page)
    attention.prepare_step((SimpleNamespace(block_tables=tables, context_lengths=lengths),))
    actual = attention(q, keys, values, tables, lengths)
    expected = torch_paged_attention(q, keys, values, tables, lengths)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=0)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[-1]) == 0
    assert attention.plan_misses == 0
    assert attention.plan_entries == 1

    for padding in (-1, 0, 9):
        padded = tables.clone()
        padded[1, 1:] = padding
        padded[3, :] = padding
        attention.prepare_step((SimpleNamespace(block_tables=padded, context_lengths=lengths),))
        out = attention(q, keys, values, padded, lengths)
        assert torch.equal(out[:3], actual[:3])
        assert torch.count_nonzero(out[3]) == 0

    subset_tables, subset_lengths = tables[:3], lengths[:3]
    attention.prepare_step(
        (SimpleNamespace(block_tables=subset_tables, context_lengths=subset_lengths),)
    )
    smaller = attention(q[:3], keys, values, subset_tables, subset_lengths)
    assert torch.equal(smaller, actual[:3])
    assert attention.plan_misses == 0


@pytest.mark.gpu
def test_empty_batch_and_unprimed_fallback():
    attention = FlashInferFA2PagedAttention(torch.device("cuda"), torch.bfloat16, 128, 16)
    keys = torch.randn(4, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    empty = attention(
        torch.empty(0, 4, 128, device="cuda", dtype=torch.bfloat16),
        keys,
        keys,
        torch.empty(0, 1, dtype=torch.int32, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
    )
    assert empty.shape == (0, 4, 128)

    q = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([32], dtype=torch.int32, device="cuda")
    expected = torch_paged_attention(q, keys, keys, tables, lengths)
    actual = attention(q, keys, keys, tables, lengths)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=0)
    assert attention.plan_misses == 1


def _tiny_config():
    return OuroConfig(
        vocab_size=64,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=128,
        total_ut_steps=4,
        bos_token_id=0,
        eos_token_id=0,
    )


@pytest.mark.gpu
def test_vectorized_csr_matches_reference():
    torch.manual_seed(11)
    attention = FlashInferFA2PagedAttention(torch.device("cuda"), torch.bfloat16, 128, 16)
    tables = torch.randint(0, 40, (6, 10), device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1, 15, 16, 17, 129, 0], device="cuda", dtype=torch.int32)
    attention.prepare_step((SimpleNamespace(block_tables=tables, context_lengths=lengths),))
    csr = attention._csr[(id(tables), id(lengths))]
    pages = ((lengths + 15) // 16).tolist()
    rows = [tables[row, :count].tolist() for row, count in enumerate(pages)]
    assert csr.indptr.tolist() == [0] + torch.cumsum(torch.tensor(pages, device="cuda"), 0).tolist()
    assert csr.indices[: csr.indptr[-1]].tolist() == [page for row in rows for page in row]
    # Empty rows keep the kernel's required last-page length of one.
    assert csr.last_page_len.tolist() == [
        1 if count == 0 else int(length) - (count - 1) * 16
        for length, count in zip(lengths.tolist(), pages)
    ]


@pytest.mark.gpu
def test_prepare_step_does_not_synchronize():
    page, head_dim, rows, width = 16, 128, 2, 64
    torch.manual_seed(17)
    keys = torch.randn(32, 3, page, 16, head_dim, device="cuda", dtype=torch.bfloat16)[:, 1]
    values = torch.randn(32, 3, page, 16, head_dim, device="cuda", dtype=torch.bfloat16)[:, 1]
    q = torch.randn(rows, 16, head_dim, device="cuda", dtype=torch.bfloat16)
    tables = torch.randint(0, 32, (rows, width), device="cuda", dtype=torch.int32)
    lengths = torch.tensor([33, 47], device="cuda", dtype=torch.int32)
    attention = FlashInferFA2PagedAttention(torch.device("cuda"), torch.bfloat16, head_dim, page)
    batch = SimpleNamespace(block_tables=tables, context_lengths=lengths)
    # Warm module load and allocator caches outside the measured region.
    attention.prepare_step((batch,))
    attention(q, keys, values, tables, lengths)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        attention.prepare_step((batch,))
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.gpu
def test_graph_replay_reads_refreshed_metadata():
    torch.manual_seed(13)
    page, head_dim, q_heads, kv_heads = 16, 128, 4, 2
    keys = torch.randn(8, 3, page, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)[:, 1]
    values = torch.randn(8, 3, page, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)[:, 1]
    q = torch.randn(2, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    tables = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
    lengths = torch.tensor([16, 32], dtype=torch.int32, device="cuda")
    attention = FlashInferFA2PagedAttention(torch.device("cuda"), torch.bfloat16, head_dim, page)
    attention.prepare_step((SimpleNamespace(block_tables=tables, context_lengths=lengths),))
    attention(q, keys, values, tables, lengths)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = attention(q, keys, values, tables, lengths)
    tables.copy_(torch.tensor([[7, 2, 3, 0], [3, 0, 0, 0]], dtype=torch.int32, device="cuda"))
    lengths.copy_(torch.tensor([35, 1], dtype=torch.int32, device="cuda"))
    attention.prepare_step((SimpleNamespace(block_tables=tables, context_lengths=lengths),))
    graph.replay()
    torch.cuda.synchronize()
    expected = torch_paged_attention(q, keys, values, tables, lengths)
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=0)
    assert attention.plan_misses == 0


@pytest.mark.gpu
@pytest.mark.parametrize(("q_heads", "kv_heads"), [(4, 2), (16, 16)])
def test_paged_attention_matches_triton(q_heads, kv_heads):
    from vllm_rlt.kernels.paged_attention import TritonPagedAttention

    torch.manual_seed(7)
    page, head_dim, rows = 16, 128, 5
    # Strided views exactly like KVCacheManager.key_cache[:, layer].
    keys = torch.randn(12, 3, page, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)[:, 1]
    values = torch.randn(12, 3, page, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)[:, 1]
    q = torch.randn(rows, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor(
        [[7, 2, 9, 3], [3, 0, 0, 0], [1, 8, 6, 4], [0, 0, 0, 0], [5, 5, 5, 5]],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([35, 1, 47, 0, 64], dtype=torch.int32, device="cuda")
    attention = FlashInferFA2PagedAttention(torch.device("cuda"), torch.bfloat16, head_dim, page)
    attention.prepare_step((SimpleNamespace(block_tables=tables, context_lengths=lengths),))
    actual = attention(q, keys, values, tables, lengths)
    expected = TritonPagedAttention()(q, keys, values, tables, lengths)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=0)
    torch.testing.assert_close(
        actual, torch_paged_attention(q, keys, values, tables, lengths), atol=1e-2, rtol=0
    )
    assert torch.count_nonzero(actual[3]) == 0
    assert attention.plan_misses == 0


def _tiny_ouro_outputs(model, layout, execution):
    traces = {str(i): [4, 2 + i % 3, 4, 2, 3, 4] for i in range(3)}
    engine = StageEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=16, layout=layout),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
        execution_config=execution,
        exit_config=ExitConfig("trace", depths_by_request=traces),
        attention_backend="flashinfer_fa2",
    )
    for i in range(3):
        engine.add_request(
            str(i),
            [2, 3, 4, 5] * (i + 1),
            SamplingParams(max_tokens=6, min_loops=1, ignore_eos=True),
        )
    finished = {}
    for _ in range(300):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            if out.finished:
                finished[out.request_id] = (out.token_ids, out.exit_depths)
    assert len(finished) == 3
    assert engine.cache_manager.num_used_blocks == 0
    assert engine.cache_manager.attention.kernel.plan_misses == 0
    return engine, finished


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize(
    "execution",
    [
        ExecutionConfig(cuda_graphs=True),
        ExecutionConfig(async_scheduling=True),
        ExecutionConfig(async_scheduling=True, cuda_graphs=True),
    ],
    ids=["graphs", "async", "async+graphs"],
)
def test_tiny_ouro_execution_variants_match_eager(layout, execution):
    """The TF2 tensor-core kernel is the shared arithmetic; only routing changes."""
    torch.manual_seed(123)
    model = OuroForCausalLM(_tiny_config()).to(device="cuda", dtype=torch.bfloat16)
    reference, expected = _tiny_ouro_outputs(model, layout, ExecutionConfig())
    engine, actual = _tiny_ouro_outputs(model, layout, execution)
    assert actual == expected
    if execution.cuda_graphs:
        assert engine.model_runner.graphs.captures > 0
        assert engine.model_runner.graphs.replays > engine.model_runner.graphs.captures
