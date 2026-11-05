"""Official paged attention: ragged prefixes, physical strides and async exits."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.kernels.flash_attention import FlashPagedAttention, select_version
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.models import OuroForCausalLM

pytestmark = pytest.mark.skipif(
    torch.version.hip is not None, reason="FlashAttention is NVIDIA-only"
)


@pytest.mark.parametrize(
    "sm,expected", [((8, 0), 2), ((8, 9), 2), ((9, 0), 3), ((10, 0), 4), ((10, 3), 4)]
)
def test_architecture_selection(sm, expected):
    assert select_version(sm, "flash_attn") == expected


@pytest.mark.parametrize("sm", [(12, 0), (12, 1)])
@pytest.mark.parametrize("backend", ["flash_attn", "flash_attn_4"])
def test_sm12_paged_backend_rejected_before_import(monkeypatch, sm, backend):
    from vllm_rlt.kernels import flash_attention

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: sm)

    def unexpected_import(name):
        pytest.fail(f"unsupported architecture reached dependency import: {name}")

    monkeypatch.setattr(flash_attention.importlib, "import_module", unexpected_import)
    message = r"Paged FlashAttention-4.*SM12.*use --attention-backend triton"
    with pytest.raises(ValueError, match=message):
        select_version(sm, backend)
    with pytest.raises(ValueError, match=message):
        FlashPagedAttention(torch.device("cuda"), torch.bfloat16, 128, 16, backend)


def test_unsupported_architecture_and_dtype():
    with pytest.raises(ValueError, match="not supported"):
        select_version((10, 3), "flash_attn_3")
    with pytest.raises(ValueError, match="not supported"):
        select_version((7, 5), "flash_attn")
    with pytest.raises(ValueError, match="NVIDIA"):
        FlashPagedAttention(torch.device("cpu"), torch.bfloat16, 128, 16)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [2, 4])
def test_ragged_strided_paged_attention(dtype, heads):
    torch.manual_seed(9)
    # Layer slicing leaves a non-contiguous block stride, as in the real cache.
    keys = torch.randn(12, 3, 16, 2, 128, device="cuda", dtype=dtype)[:, 1]
    values = torch.randn(12, 3, 16, 2, 128, device="cuda", dtype=dtype)[:, 1]
    q = torch.randn(4, heads, 128, device="cuda", dtype=dtype)
    tables = torch.tensor(
        [[7, 2, 9], [3, 0, 0], [1, 8, 6], [0, 0, 0]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([35, 1, 47, 0], device="cuda", dtype=torch.int32)
    attention = FlashPagedAttention(q.device, dtype, 128, 16)
    expected = torch_paged_attention(q, keys, values, tables, lengths)
    actual = attention(q, keys, values, tables, lengths)
    torch.testing.assert_close(
        actual, expected, atol=0.015 if dtype == torch.bfloat16 else 0.002, rtol=0.02
    )
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[-1]) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("static", [False, True])
def test_flash_sync_async_mixed_depths(layout, static):
    torch.manual_seed(123)
    config = tiny_ouro_config(head_dim=64)
    model = OuroForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    traces = {str(i): [4, 2 + i % 3, 4, 2, 3, 4] for i in range(3)}
    outputs = []
    for backend, asynchronous, multi in [
        ("triton", False, False),
        ("flash_attn", False, False),
        ("flash_attn", True, False),
        ("flash_attn", True, True),
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


def test_packed_prefill_metadata_groups_queries_and_rejects_gaps():
    from types import SimpleNamespace

    from vllm_rlt.attention import BackendCapabilities
    from vllm_rlt.core.kv_cache_manager import KVCacheManager

    cache = KVCacheManager(1, 1, 8, 32, 4, 2)
    cache.attention = SimpleNamespace(capabilities=BackendCapabilities(packed_prefill=True))
    cache.allocate("a", 20)
    cache.allocate("b", 20)
    batch = cache._prepare_batch(
        ["a"] * 3 + ["b"] * 2, [0] * 3 + [1] * 2, [7, 8, 9, 3, 4], packed_prefill=True
    )
    assert batch.block_tables.shape == (2, 3)
    assert batch.context_lengths.tolist() == [10, 5]
    assert batch.cu_seqlens_q.tolist() == [0, 3, 5]
    assert batch.max_seqlen_q == 3
    assert batch.position_ids.tolist() == [7, 8, 9, 3, 4]
    assert len(batch.write_blocks) == 5
    with pytest.raises(ValueError, match="contiguous"):
        cache._prepare_batch(["a", "a"], [0, 0], [1, 3], packed_prefill=True)
    with pytest.raises(ValueError, match="one sequence"):
        cache._prepare_batch(["a", "b", "a"], [0, 0, 0], [1, 1, 2], packed_prefill=True)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_prefill_ragged_prefixes_depths_and_causal_mask(dtype):
    from vllm_rlt.core.kv_cache_manager import KVCacheManager

    torch.manual_seed(712)
    cache = KVCacheManager(2, 2, 64, 80, 16, 4, "cuda", dtype, "flash_attn")
    if cache.attention.generation != 4:
        pytest.skip("packed paged prefill is FA4-specific")
    # Fragment physical allocation and cross page boundaries at unequal depths.
    for rid in ["hole", "a", "b", "c"]:
        cache.allocate(rid, 65)
    cache.free("hole")
    ids, depths, positions = [], [], []
    for rid, depth, prefix, count in [("a", 0, 0, 19), ("b", 2, 17, 7), ("c", 3, 31, 18)]:
        for layer in range(2):
            if prefix:
                kv = torch.randn(prefix, 2, 64, device="cuda", dtype=dtype)
                cache.write(layer, [rid] * prefix, [depth] * prefix, list(range(prefix)), kv, -kv)
        ids.extend([rid] * count)
        depths.extend([depth] * count)
        positions.extend(range(prefix, prefix + count))
    batch = cache._prepare_batch(ids, depths, positions, packed_prefill=True)
    flat = cache._prepare_batch(ids, depths, positions)
    assert batch.block_tables.shape[0] == 3
    assert flat.block_tables.shape[0] == len(ids)
    for layer in range(2):
        k = torch.randn(len(ids), 2, 64, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        q = torch.randn(len(ids), 4, 64, device="cuda", dtype=dtype)
        cache._write_prepared(layer, batch, k, v)
        # All chunk keys (including future positions) exist before attention.
        # A per-token prefix oracle detects wrong causal/prefix alignment.
        expected = torch_paged_attention(
            q,
            cache.key_cache[:, layer],
            cache.value_cache[:, layer],
            flat.block_tables,
            flat.context_lengths,
        )
        actual = cache._attend_prepared(layer, batch, q)
        torch.testing.assert_close(
            actual, expected, atol=0.015 if dtype == torch.bfloat16 else 0.002, rtol=0.02
        )


@pytest.mark.gpu
def test_fa4_decode_same_queries_match_across_batch_sizes():
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("Pinned paged FA4 requires SM9 or SM10")
    generator = torch.Generator(device="cuda").manual_seed(1234)
    dtype = torch.bfloat16
    # Ouro geometry and long enough prefixes to exercise auto-SplitKV selection.
    keys = torch.randn(260, 2, 16, 4, 128, generator=generator, device="cuda", dtype=dtype)[:, 1]
    values = torch.randn(260, 2, 16, 4, 128, generator=generator, device="cuda", dtype=dtype)[:, 1]
    q = torch.randn(4, 16, 128, generator=generator, device="cuda", dtype=dtype)
    tables = torch.arange(260, device="cuda", dtype=torch.int32).view(4, 65)
    lengths = torch.tensor([1025, 1029, 1025, 1031], device="cuda", dtype=torch.int32)
    attention = FlashPagedAttention(q.device, dtype, 128, 16, "flash_attn_4")
    together = attention(q, keys, values, tables, lengths)
    for rows in ([2, 3], [3, 0], [1]):
        separate = attention(q[rows], keys, values, tables[rows], lengths[rows])
        torch.testing.assert_close(separate, together[rows], rtol=0, atol=0)
