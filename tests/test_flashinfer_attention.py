"""FlashInfer paged attention: gates, padding states, graphs and mixed depths."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.attention import BackendCapabilities, create_backend
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.kernels.flashinfer_attention import FlashInferPagedAttention
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.models import OuroForCausalLM


def _cuda_like(monkeypatch, sm):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: sm)


def test_unsupported_architecture_rejected_before_import(monkeypatch):
    from vllm_rlt.kernels import flashinfer_attention

    _cuda_like(monkeypatch, (7, 5))

    def unexpected_import(name):
        pytest.fail(f"unsupported architecture reached dependency import: {name}")

    monkeypatch.setattr(flashinfer_attention.importlib, "import_module", unexpected_import)
    with pytest.raises(ValueError, match="not supported on SM75"):
        FlashInferPagedAttention(torch.device("cuda"), torch.bfloat16, 128, 16)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.complex64], ids=["fp16", "fp32", "complex"]
)
def test_unvalidated_dtypes_rejected(monkeypatch, dtype):
    _cuda_like(monkeypatch, (12, 0))
    with pytest.raises(ValueError):
        FlashInferPagedAttention(torch.device("cuda"), dtype, 128, 16)


def test_geometry_and_page_size_rejections(monkeypatch):
    _cuda_like(monkeypatch, (12, 0))
    with pytest.raises(ValueError, match="divisible by 16"):
        FlashInferPagedAttention(torch.device("cuda"), torch.bfloat16, 100, 16)
    with pytest.raises(ValueError, match="divisible by 16"):
        FlashInferPagedAttention(torch.device("cuda"), torch.bfloat16, 8, 16)
    with pytest.raises(ValueError, match="page-size variants"):
        FlashInferPagedAttention(torch.device("cuda"), torch.bfloat16, 128, 17)
    with pytest.raises(ValueError, match="NVIDIA"):
        FlashInferPagedAttention(torch.device("cpu"), torch.bfloat16, 128, 16)


def test_missing_package_raises_import_error_without_fallback(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    from vllm_rlt.kernels import flashinfer_attention

    _cuda_like(monkeypatch, (12, 0))

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(flashinfer_attention, "version", missing)
    with pytest.raises(ImportError, match="install flashinfer"):
        FlashInferPagedAttention(torch.device("cuda"), torch.bfloat16, 128, 16)


@pytest.mark.gpu
def test_backend_contract_anchors():
    attention = FlashInferPagedAttention(torch.device("cuda"), torch.bfloat16, 128, 16)
    assert attention.info["backend"] == "flashinfer"
    assert attention.info["package"] == "flashinfer-python"
    assert attention.info["page_size"] == 16
    assert attention.info["kernel"] == "xqa"
    # Absence is the contract: capability declarations live only in the
    # attention registration layer, and the FA4 packed-prefill gates key off
    # a missing generation attribute staying missing.
    assert not hasattr(attention, "capabilities")
    assert not hasattr(attention, "generation")
    assert not hasattr(attention, "prefill")
    wrapper = create_backend("flashinfer", torch.device("cuda"), torch.bfloat16, 128, 16)
    assert wrapper.info["selected_backend"] == "flashinfer"
    assert wrapper.capabilities == BackendCapabilities(cuda_graphs=True, async_scheduling=True)


@pytest.mark.gpu
@pytest.mark.parametrize("heads", [2, 4])
def test_ragged_strided_paged_attention_with_zero_length_row(heads):
    torch.manual_seed(9)
    # Layer slicing leaves a non-contiguous block stride, as in the real cache.
    keys = torch.randn(12, 3, 16, 2, 128, device="cuda", dtype=torch.bfloat16)[:, 1]
    values = torch.randn(12, 3, 16, 2, 128, device="cuda", dtype=torch.bfloat16)[:, 1]
    q = torch.randn(4, heads, 128, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor(
        [[7, 2, 9], [3, 0, 0], [1, 8, 6], [0, 0, 0]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([35, 1, 47, 0], device="cuda", dtype=torch.int32)
    attention = FlashInferPagedAttention(q.device, torch.bfloat16, 128, 16)
    expected = torch_paged_attention(q, keys, values, tables, lengths)
    actual = attention(q, keys, values, tables, lengths)
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.02)
    assert torch.isfinite(actual).all()
    # Zero-length rows are async padding rows; they must read as exact zeros,
    # not small garbage, so downstream masking stays optional.
    assert torch.count_nonzero(actual[-1]) == 0


@pytest.mark.gpu
def test_padding_states_are_invisible():
    torch.manual_seed(11)
    keys = torch.randn(24, 3, 16, 2, 64, device="cuda", dtype=torch.bfloat16)[:, 1]
    values = torch.randn(24, 3, 16, 2, 64, device="cuda", dtype=torch.bfloat16)[:, 1]
    q = torch.randn(3, 2, 64, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([16, 31, 20], device="cuda", dtype=torch.int32)
    attention = FlashInferPagedAttention(q.device, torch.bfloat16, 64, 16)
    outputs = []
    for fill in (-1, 0, 19):  # sync pad, async pad, stale-but-valid replay id
        tables = torch.full((3, 4), fill, device="cuda", dtype=torch.int32)
        for row in range(3):
            tables[row, :2] = torch.arange(row * 2, row * 2 + 2, device="cuda")
        outputs.append(attention(q, keys, values, tables, lengths))
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


@pytest.mark.gpu
def test_flashinfer_decode_same_queries_match_across_batch_sizes():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    dtype = torch.bfloat16
    # Ouro geometry and long enough prefixes to exercise any KV split selection.
    keys = torch.randn(260, 2, 16, 4, 128, generator=generator, device="cuda", dtype=dtype)[:, 1]
    values = torch.randn(260, 2, 16, 4, 128, generator=generator, device="cuda", dtype=dtype)[:, 1]
    q = torch.randn(4, 16, 128, generator=generator, device="cuda", dtype=dtype)
    tables = torch.arange(260, device="cuda", dtype=torch.int32).view(4, 65)
    lengths = torch.tensor([1025, 1029, 1025, 1031], device="cuda", dtype=torch.int32)
    attention = FlashInferPagedAttention(q.device, dtype, 128, 16)
    together = attention(q, keys, values, tables, lengths)
    for rows in ([2, 3], [3, 0], [1]):
        separate = attention(q[rows], keys, values, tables[rows], lengths[rows])
        torch.testing.assert_close(separate, together[rows], rtol=0, atol=0)


@pytest.mark.gpu
def test_graph_replay_updates_buffers():
    torch.manual_seed(5)
    keys = torch.randn(24, 3, 16, 2, 64, device="cuda", dtype=torch.bfloat16)[:, 1]
    values = torch.randn(24, 3, 16, 2, 64, device="cuda", dtype=torch.bfloat16)[:, 1]
    rows, width = 4, 4
    q = torch.randn(rows, 2, 64, device="cuda", dtype=torch.bfloat16)
    static_tables = torch.zeros(rows, width, device="cuda", dtype=torch.int32)
    static_lengths = torch.full((rows,), 16, device="cuda", dtype=torch.int32)
    attention = FlashInferPagedAttention(q.device, torch.bfloat16, 64, 16)
    # Warm JIT modules and workspace outside the capture region.
    attention(q, keys, values, static_tables, static_lengths)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = attention(q, keys, values, static_tables, static_lengths)
    generator = torch.Generator().manual_seed(21)
    for trial in range(3):
        for row in range(rows):
            start = (row * 3 + trial * 5) % 16
            static_tables[row] = torch.arange(start, start + width, dtype=torch.int32)
        static_lengths.copy_(
            torch.randint(1, width * 16, (rows,), generator=generator, dtype=torch.int32)
        )
        graph.replay()
        expected = torch_paged_attention(q, keys, values, static_tables, static_lengths)
        torch.testing.assert_close(out, expected, atol=0.015, rtol=0.02)


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("static", [False, True])
def test_flashinfer_sync_async_mixed_depths(layout, static):
    torch.manual_seed(123)
    config = tiny_ouro_config(head_dim=64)
    model = OuroForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    traces = {str(i): [4, 2 + i % 3, 4, 2, 3, 4] for i in range(3)}
    outputs = []
    for backend, asynchronous, multi in [
        ("triton", False, False),
        ("flashinfer", False, False),
        ("flashinfer", True, False),
        ("flashinfer", True, True),
    ]:
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=48, block_size=16, layout=layout),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
            ),
            execution_config=ExecutionConfig(
                async_scheduling=asynchronous,
                multi_stream=multi,
                static_buffers=static,
                pad_to_power_of_two=static,
            ),
            exit_config=ExitConfig("trace", depths_by_request=traces),
            attention_backend=backend,
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
        outputs.append(finished)
    assert all(value == outputs[0] for value in outputs)
