"""Static-scale FA4 prototype: CPU interface checks and GPU numerics."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from vllm_rlt import CacheConfig
from vllm_rlt.attention import create_backend
from vllm_rlt.core.kv_cache_manager import KVCacheManager, quantize_fp8
from vllm_rlt.entrypoints.runtime_args import add_runtime_args, runtime_configs
from vllm_rlt.kernels.paged_attention import torch_paged_attention, torch_paged_attention_windowed


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), 1e-50, 1e40, True])
def test_invalid_static_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="fp8_scale"):
        KVCacheManager(1, 1, 64, 8, 16, 1, kv_cache_dtype="fp8_e4m3", fp8_scale=scale)


def test_static_scale_requires_fp8():
    with pytest.raises(ValueError, match="requires"):
        KVCacheManager(1, 1, 64, 8, 16, 1, fp8_scale=0.02)


def test_fp8_cli_leaves_scale_selection_to_engine():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--block-size", type=int, default=16)
    add_runtime_args(parser)
    args = parser.parse_args(["--kv-cache-dtype", "fp8_e4m3"])
    configs = runtime_configs(args)
    assert configs["cache_config"].kv_cache_dtype == "fp8_e4m3"
    with pytest.raises(SystemExit):
        parser.parse_args(["--kv-cache-dtype", "fp8_e4m3", "--fp8-scale", "0.02"])


def test_static_quantization_saturates_and_does_not_rescale_old_tokens():
    cache = KVCacheManager(1, 1, 64, 8, 16, 1, kv_cache_dtype="fp8_e4m3", fp8_scale=0.02)
    cache.allocate("a", 2)
    for pos, value in enumerate([0.25, 1000.0]):
        x = torch.full((1, 1, 64), value)
        cache.write(0, ["a"], [0], [pos], x, -x)
    k, v = cache.read(0, "a", 0)
    torch.testing.assert_close(k[0], torch.full_like(k[0], 0.24))
    torch.testing.assert_close(k[1], torch.full_like(k[1], 448 * 0.02))
    torch.testing.assert_close(v, -k)


@pytest.mark.parametrize("packed", [False, True])
def test_cache_to_fa4_fp8_contract_on_cpu(monkeypatch, packed):
    """Execute the real adapter/cache; substitute only the CUDA FA implementation."""
    from vllm_rlt.kernels import flash_attention

    calls = []

    def fake_fa(q, k, v, **kwargs):
        assert q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
        batch = kwargs["page_table"].shape[0]
        for name in ("q", "k", "v"):
            scale = getattr(kwargs["scales"], name)
            assert scale.shape == (batch, 2) and scale.dtype == torch.float32
            torch.testing.assert_close(scale, torch.full_like(scale, 0.02))
        assert kwargs["causal"] and kwargs["num_splits"] == 1
        calls.append((q.clone(), kwargs))
        return torch.zeros(q.shape, dtype=torch.bfloat16), None

    original_import = flash_attention.importlib.import_module

    def public_fa(q, k, v, **kwargs):
        if q.dtype == torch.float8_e4m3fn:
            return fake_fa(q, k, v, **kwargs)
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        assert "scales" not in kwargs
        return torch.zeros_like(q), None

    def import_module(name):
        if name == "flash_attn.cute":
            return SimpleNamespace(
                flash_attn_varlen_func=public_fa, TensorwiseScales=SimpleNamespace
            )
        return original_import(name)

    monkeypatch.setattr(flash_attention.importlib, "import_module", import_module)
    monkeypatch.setattr(flash_attention, "version", lambda name: "4.0.0b39")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    from vllm_rlt.attention import BackendCapabilities

    attention = create_backend(
        "flash_attn_4",
        torch.device("cuda"),
        torch.bfloat16,
        64,
        16,
        required_capabilities=BackendCapabilities(fp8_kv_static=True),
    )
    cache = KVCacheManager(
        2,
        2,
        64,
        16,
        16,
        2,
        dtype=torch.bfloat16,
        attention=attention,
        kv_cache_dtype="fp8_e4m3",
        fp8_scale=0.02,
    )
    cache.allocate("a", 3)
    cache.allocate("b", 3)
    batch = cache._prepare_batch(["a", "a", "b"], [0, 0, 1], [0, 1, 0], packed_prefill=packed)
    q = torch.linspace(-10, 10, 3 * 4 * 64).reshape(3, 4, 64).bfloat16()
    kv = q[:, :2]
    cache._write_prepared(1, batch, kv, -kv)
    output = cache._attend_prepared(1, batch, q)
    assert output.shape == q.shape and output.dtype == q.dtype
    q_codes, kwargs = calls.pop()
    expected, _ = quantize_fp8(q, 0.02)
    torch.testing.assert_close(q_codes.reshape_as(q).float(), expected.float(), rtol=0, atol=0)
    if packed:
        assert kwargs["cu_seqlens_q"].tolist() == [0, 2, 3]
        assert kwargs["seqused_k"].tolist() == [2, 1]
    else:
        assert kwargs["seqused_k"].tolist() == [1, 2, 1]
    with pytest.raises(ValueError, match="fixed scalar"):
        attention(
            q,
            cache.key_cache[:, 1],
            cache.value_cache[:, 1],
            batch.block_tables,
            batch.context_lengths,
            key_scale=cache.key_scale[:, 1],
            value_scale=cache.value_scale[:, 1],
        )
    # Both BF16 and FP8 use the public API; only FP8 supplies scales.
    flat = cache._prepare_batch(["a", "a", "b"], [0, 0, 1], [0, 1, 0])
    attention(
        q,
        cache.key_cache[:, 1].bfloat16(),
        cache.value_cache[:, 1].bfloat16(),
        flat.block_tables,
        flat.context_lengths,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_fa4_fp8_matches_dequantized_paged_reference(packed, head_dim, record_property):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FA4 FP8 requires SM10x")
    torch.manual_seed(91)
    cache = KVCacheManager(
        2,
        2,
        head_dim,
        32,
        16,
        2,
        "cuda",
        torch.bfloat16,
        "flash_attn_4",
        kv_cache_dtype="fp8_e4m3",
        fp8_scale=0.02,
    )
    cache.allocate("a", 32)
    cache.allocate("b", 32)
    ids = ["a"] * 19 + ["b"] * 7
    depths = [0] * 19 + [1] * 7
    positions = list(range(19)) + list(range(7))
    batch = cache._prepare_batch(ids, depths, positions, packed_prefill=packed)
    flat = cache._prepare_batch(ids, depths, positions)
    q = torch.randn(26, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(26, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    cache._write_prepared(1, batch, k, -k)
    actual = cache._attend_prepared(1, batch, q)
    q_codes, q_scale = quantize_fp8(q, 0.02)
    expected = torch_paged_attention(
        q_codes.float() * q_scale.unsqueeze(-1),
        cache.key_cache[:, 1],
        cache.value_cache[:, 1],
        flat.block_tables,
        flat.context_lengths,
        cache.key_scale[:, 1],
        cache.value_scale[:, 1],
    )
    # Compare to quantized inputs; this is a kernel check, not model quality.
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.04, rtol=0.05)
    error = actual.float() - expected.float()
    record_property("max_abs_error", error.abs().max().item())
    record_property("rms_error", error.square().mean().sqrt().item())


@pytest.mark.gpu
def test_fa4_static_fp8_prefix_snapshot_and_mixed_depth_finalize():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FA4 FP8 requires SM10x")
    torch.manual_seed(163)
    cache = KVCacheManager(
        2,
        2,
        128,
        32,
        16,
        3,
        "cuda",
        torch.bfloat16,
        "flash_attn_4",
        enable_prefix_caching=True,
        kv_cache_dtype="fp8_e4m3",
        fp8_scale=0.02,
    )
    tokens = list(range(18))
    assert cache.allocate("source", 32)
    for depth in range(3):
        for layer in range(2):
            values = torch.randn(17, 2, 128, device="cuda", dtype=torch.bfloat16)
            cache.write(layer, ["source"] * 17, [depth] * 17, list(range(17)), values, -values)
    for depth in (0, 1):
        for layer in range(2):
            values = torch.randn(1, 2, 128, device="cuda", dtype=torch.bfloat16)
            cache.write(layer, ["source"], [depth], [17], values, -values)
    cache.finalize_token("source", 17, 1)
    for layer in range(2):
        for source, copied in zip(cache.read(layer, "source", 1), cache.read(layer, "source", 2)):
            torch.testing.assert_close(source[-1], copied[-1], rtol=0, atol=0)
    cache.publish_prefix("source", tokens, len(tokens))
    prefix = cache.lookup_prefix(tokens)
    assert len(prefix) == 1  # Only the full 16-token block can be shared.
    assert cache.allocate("shared", 32, prefix=prefix)
    for depth in range(3):
        for layer in range(2):
            for original, shared in zip(
                cache.read(layer, "source", depth, 16), cache.read(layer, "shared", depth)
            ):
                torch.testing.assert_close(shared, original, rtol=0, atol=0)
    q = torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16)
    before = cache.attend(1, ["source"] * 3, [0, 1, 2], [17] * 3, q)
    snapshot = cache.snapshot("source")
    cache.free("source")
    assert cache.allocate("restored", 32)
    cache.restore("restored", snapshot)
    after = cache.attend(1, ["restored"] * 3, [0, 1, 2], [17] * 3, q)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    cache.free("shared")
    cache.free("restored")


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True])
def test_fa4_static_fp8_async_graphs_match_eager(multi_stream):
    from tests.helpers import tiny_ouro_config
    from vllm_rlt import ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
    from vllm_rlt.engine.llm_engine import LLMEngine
    from vllm_rlt.models import OuroForCausalLM

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FA4 FP8 requires SM10x")
    torch.manual_seed(321)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    traces = {str(i): [4, 1, 3, 2, 4, 2, 1][: 5 + i] for i in range(3)}

    def run(execution):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(
                num_blocks=None if execution.cuda_graphs else 64, kv_cache_dtype="fp8_e4m3"
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
            ),
            execution_config=execution,
            exit_config=ExitConfig("trace", depths_by_request=traces),
            attention_backend="flash_attn_4",
        )
        try:
            scale = engine.cache_manager.fp8_scale
            assert scale is not None and scale > 0
            for i in range(3):
                engine.add_request(
                    str(i),
                    [2 + i, 3, 4] * (i + 1),
                    SamplingParams(max_tokens=5 + i, min_loops=1, ignore_eos=True),
                )
            done = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        done[output.request_id] = (output.token_ids, list(output.exit_depths))
            assert len(done) == 3
            assert engine.cache_manager.num_used_blocks == 0
            if execution.cuda_graphs:
                graphs = engine.model_runner.graphs
                assert graphs.captures > 0
                assert graphs.replays > graphs.captures
                assert graphs.fallbacks == 0
            return scale, done
        finally:
            engine.close()

    reference = run(ExecutionConfig())
    candidate = run(
        ExecutionConfig(async_scheduling=True, multi_stream=multi_stream, cuda_graphs=True)
    )
    assert candidate == reference


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("fixed_scale", [None, 0.02])
def test_fa4_windowed_attention_matches_quantized_reference(head_dim, fixed_scale):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FA4 FP8 requires SM10x")
    torch.manual_seed(419)
    attention = create_backend("flash_attn_4", torch.device("cuda"), torch.bfloat16, head_dim, 16)
    q = torch.randn(4, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    # Layer slices preserve the real pool strides, including for head_dim=256.
    k, v = [
        torch.randn(12, 3, 16, 2, head_dim, device="cuda", dtype=q.dtype)[:, 1] for _ in range(2)
    ]
    tables = torch.tensor(
        [[7, 2, 9], [3, 0, 0], [1, 8, 6], [0, 0, 0]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([35, 1, 47, 0], device="cuda", dtype=torch.int32)
    scales = {}
    reference_q = q
    key_scale = value_scale = None
    if fixed_scale is not None:
        q_codes, q_scale = quantize_fp8(q, fixed_scale)
        reference_q = q_codes.float() * q_scale.unsqueeze(-1)
        # Keep non-contiguous layer strides for the FP8 pool too.
        k, v = [
            x.to(torch.float8_e4m3fn)
            for x in (
                torch.randn(12, 3, 16, 2, head_dim, device="cuda") / fixed_scale,
                torch.randn(12, 3, 16, 2, head_dim, device="cuda") / fixed_scale,
            )
        ]
        k, v = k[:, 1], v[:, 1]
        key_scale = value_scale = torch.full(k.shape[:-1], fixed_scale, device="cuda")
        scalar = torch.tensor(fixed_scale, device="cuda")
        scales = dict(key_scale=scalar, value_scale=scalar)
    for window in (1, 4, 16, 17, 1024):
        actual = attention.windowed(q, k, v, tables, lengths, window, **scales)
        expected = torch_paged_attention_windowed(
            reference_q, k, v, tables, lengths, window, key_scale, value_scale
        )
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.04, rtol=0.05)
        assert torch.count_nonzero(actual[-1]) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("kv_mode,loops", [("per_loop", 2), ("last_loop", 4)])
@pytest.mark.parametrize("mode", ["async", "multi", "graph", "async_graph", "multi_graph"])
def test_dory_fa4_fp8_execution_modes_match_eager(kv_mode, loops, mode):
    from tests.test_dory import tiny_dory_config
    from vllm_rlt import ExecutionConfig, SamplingParams, SchedulerConfig
    from vllm_rlt.engine.llm_engine import LLMEngine
    from vllm_rlt.models import DoryForCausalLM

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FA4 FP8 requires SM10x")
    torch.manual_seed(42)
    model = DoryForCausalLM(
        tiny_dory_config(head_dim=256, n_recurrent_loops=loops, recurrent_kv_cache_mode=kv_mode)
    ).to("cuda", torch.bfloat16)
    prompts = {"a": [4, 7, 3], "b": [9, 2, 6, 5], "c": [8, 3, 3, 11, 5, 2, 7, 13, 6]}

    def run(execution):
        engine = LLMEngine(
            model,
            attention_backend="auto",
            cache_config=CacheConfig(
                num_blocks=128,
                block_size=16,
                layout=model.kv_cache_layout,
                kv_cache_dtype="fp8_e4m3",
            ),
            scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=16),
            execution_config=execution,
        )
        try:
            assert engine.cache_manager.backend == "flash_attn_4"
            assert engine.execution_config.packed_prefill == (
                kv_mode == "per_loop"
                and not execution.async_scheduling
                and not execution.cuda_graphs
            )
            params = SamplingParams(max_tokens=8, min_loops=loops, max_loops=loops, ignore_eos=True)
            finished = {}
            for rid, prompt in prompts.items():
                engine.add_request(rid, prompt, params)
            for _ in range(300):
                if not engine.has_unfinished_requests():
                    break
                for out in engine.step():
                    if out.finished:
                        finished[out.request_id] = (out.token_ids, list(out.exit_depths))
            assert len(finished) == len(prompts)
            assert engine.cache_manager.num_used_blocks == 0
            if execution.cuda_graphs:
                runner = engine.model_runner
                for graphs in (runner.prelude_graphs, runner.graphs, runner.coda_graphs):
                    assert graphs.captures > 0 and graphs.replays > 0
                    assert graphs.fallbacks == 0
            return finished
        finally:
            engine.close()

    expected = run(ExecutionConfig())
    actual = run(
        ExecutionConfig(
            async_scheduling=mode != "graph",
            multi_stream=mode.startswith("multi"),
            cuda_graphs="graph" in mode,
        )
    )
    assert actual == expected


def test_dory_fp8_staged_prefill_rejects_graphs():
    from tests.test_dory import tiny_dory_config
    from vllm_rlt import ExecutionConfig, SchedulerConfig
    from vllm_rlt.engine.llm_engine import LLMEngine
    from vllm_rlt.models import DoryForCausalLM

    model = DoryForCausalLM(tiny_dory_config(head_dim=256))
    with pytest.raises(ValueError, match="staged prefill requires synchronous eager"):
        LLMEngine(
            model,
            attention_backend="flash_attn_4",
            cache_config=CacheConfig(kv_cache_dtype="fp8_e4m3"),
            scheduler_config=SchedulerConfig(mode="mixed"),
            execution_config=ExecutionConfig(cuda_graphs=True),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_fa4_fp8_packed_window_with_existing_prefix_matches_per_row_reference(head_dim):
    from vllm_rlt.worker.cuda_graph import _DeviceCache

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FA4 FP8 requires SM10x")
    torch.manual_seed(819)
    cache = KVCacheManager(
        2,
        2,
        head_dim,
        32,
        16,
        2,
        "cuda",
        torch.bfloat16,
        "flash_attn_4",
        kv_cache_dtype="fp8_e4m3",
        fp8_scale=0.02,
    )
    for rid in ("a", "b"):
        assert cache.allocate(rid, 32)
    ids, depths, positions = (
        ["a"] * 19 + ["b"] * 7,
        [0] * 19 + [1] * 7,
        list(range(19)) + list(range(7)),
    )
    k, v = [torch.randn(26, 2, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    cache.write(1, ids, depths, positions, k, v)
    ids, depths, positions = (
        ["a"] * 4 + ["b"] * 3,
        [0] * 4 + [1] * 3,
        list(range(15, 19)) + list(range(4, 7)),
    )
    packed = cache._prepare_batch(ids, depths, positions, packed_prefill=True)
    flat = cache._prepare_batch(ids, depths, positions)
    q = torch.randn(7, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    codes, scales = quantize_fp8(q, 0.02)
    expected = torch_paged_attention_windowed(
        codes.float() * scales.unsqueeze(-1),
        cache.key_cache[:, 1],
        cache.value_cache[:, 1],
        flat.block_tables,
        flat.context_lengths,
        4,
        cache.key_scale[:, 1],
        cache.value_scale[:, 1],
    )
    # FA4 quantizes the softmax numerator to E4M3 before P@V. Its rounding
    # error is bounded by 1/16 of each weighted |V| contribution, including
    # outputs near zero where an output-relative tolerance is inappropriate.
    weighted_abs_v = torch_paged_attention_windowed(
        codes.float() * scales.unsqueeze(-1),
        cache.key_cache[:, 1],
        cache.value_cache[:, 1].float().abs(),
        flat.block_tables,
        flat.context_lengths,
        4,
        cache.key_scale[:, 1],
        cache.value_scale[:, 1],
    )
    for target in (cache, _DeviceCache(cache)):
        actual = target._attend_prepared(1, packed, q, window=4)
        # Include BF16 output rounding and small softmax reduction error.
        assert torch.all((actual.float() - expected.float()).abs() <= 0.07 * weighted_abs_v + 0.005)
