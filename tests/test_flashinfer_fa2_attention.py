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
from vllm_rlt.attention import create_backend, validate_backend_name
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
    assert not any(
        (
            capabilities.packed_prefill,
            capabilities.cuda_graphs,
            capabilities.async_scheduling,
            capabilities.device_context_lengths,
        )
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

    attention.prepare_step((SimpleNamespace(block_tables=tables[:3], context_lengths=lengths[:3]),))
    smaller = attention(q[:3], keys, values, tables[:3], lengths[:3])
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
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
def test_tiny_ouro_matches_triton(layout):
    torch.manual_seed(123)
    model = OuroForCausalLM(_tiny_config()).to(device="cuda", dtype=torch.bfloat16)
    traces = {str(i): [4, 2 + i % 3, 4, 2, 3, 4] for i in range(3)}
    outputs = []
    for backend in ("triton", "flashinfer_fa2"):
        engine = StageEngine(
            model,
            cache_config=CacheConfig(num_blocks=64, block_size=16, layout=layout),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
            ),
            execution_config=ExecutionConfig(),
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
    assert outputs[0] == outputs[1]
    kernel = engine.cache_manager.attention.kernel
    assert kernel.plan_misses == 0
