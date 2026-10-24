"""FP8 e4m3 KV pool: per-vector quantization, attention, LAST_EXITED copies and preemption."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.attention import AttentionBackend, BackendCapabilities
from vllm_rlt.core.kv_cache_manager import (
    FP8_E4M3_MAX,
    KVCacheManager,
    dequantize_fp8,
    kv_bytes_per_block,
    quantize_fp8,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.models import OuroForCausalLM


def make_cache(kv_cache_dtype="fp8_e4m3", **overrides):
    values = dict(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=64,
        block_size=4,
        max_loops=4,
        kv_cache_dtype=kv_cache_dtype,
    )
    values.update(overrides)
    return KVCacheManager(**values)


def fill(cache, rid, length, depths=range(4), seed=0):
    """Write random K/V for positions [0, length) at every depth and layer."""
    generator = torch.Generator().manual_seed(seed)
    for depth in depths:
        for layer in range(cache.num_layers):
            shape = (length, cache.num_kv_heads, cache.head_dim)
            k = torch.randn(shape, generator=generator) * (1 + layer)
            v = torch.randn(shape, generator=generator)
            cache.write(layer, [rid] * length, [depth] * length, list(range(length)), k, v)


def test_quantize_round_trip_is_bounded_per_vector():
    x = torch.randn(64, 4, 32) * torch.logspace(-3, 3, 64).view(64, 1, 1)
    codes, scale = quantize_fp8(x)
    assert codes.dtype == torch.float8_e4m3fn and scale.shape == x.shape[:-1]
    error = (dequantize_fp8(codes, scale, torch.float32) - x).abs()
    # e4m3 keeps 3 mantissa bits: rounding error is at most 1/16 of the vector's amax.
    assert (error <= x.abs().amax(-1, keepdim=True) / 16 + 1e-30).all()
    zeros = quantize_fp8(torch.zeros(2, 8))
    assert torch.isfinite(zeros[1]).all() and not dequantize_fp8(*zeros, torch.float32).any()


def test_storage_layout_and_byte_accounting():
    cache = make_cache()
    assert cache.key_cache.dtype == torch.float8_e4m3fn
    assert cache.key_scale.shape == cache.key_cache.shape[:-1]
    assert cache.bytes_per_block == kv_bytes_per_block(2, 4, 2, 8, 1, "fp8_e4m3")
    assert cache.bytes_per_block == 2 * 2 * 4 * 2 * (8 + 4)
    plain = make_cache("auto", dtype=torch.bfloat16)
    assert plain.key_scale is None and plain.bytes_per_block == 2 * 2 * 4 * 2 * 8 * 2
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        make_cache("int4")


def test_attention_equals_attention_over_dequantized_values():
    cache = make_cache()
    reference = make_cache("auto")
    assert cache.allocate("r", 11) and reference.allocate("r", 11)
    fill(cache, "r", 11)
    for layer in range(cache.num_layers):
        for depth in range(4):
            k, v = cache.read(layer, "r", depth)
            reference.write(layer, ["r"] * 11, [depth] * 11, list(range(11)), k, v)
    q = torch.randn(3, 2, 8)
    rows = (["r"] * 3, [0, 2, 3], [10, 4, 7])
    for layer in range(cache.num_layers):
        got = cache.attend(layer, *rows, q)
        want = reference.attend(layer, *rows, q)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_torch_kernel_scales_multiply_each_token_vector():
    codes, scale = quantize_fp8(torch.randn(5, 4, 2, 8))  # [block, token, head, dim]
    tables = torch.tensor([[3, 1, 4]], dtype=torch.int32)
    lengths = torch.tensor([10], dtype=torch.int32)
    q = torch.randn(1, 2, 8)
    got = torch_paged_attention(q, codes, codes, tables, lengths, scale, scale)
    dense = dequantize_fp8(codes, scale, torch.float32)
    want = torch_paged_attention(q, dense, dense, tables, lengths)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_finalize_copies_codes_and_scales_to_deeper_planes():
    cache = make_cache()
    assert cache.allocate("r", 6)
    fill(cache, "r", 5)
    fill(cache, "r", 6, depths=[0], seed=1)  # position 5 exists only at depth 0
    cache.finalize_token("r", 5, 0)
    for layer in range(cache.num_layers):
        exited = [t[5] for t in cache.read(layer, "r", 0)]
        for depth in (1, 2, 3):
            deeper = [t[5] for t in cache.read(layer, "r", depth)]
            assert all(torch.equal(a, b) for a, b in zip(exited, deeper))


def test_snapshot_restore_is_bitwise_and_checks_dtype():
    cache = make_cache()
    assert cache.allocate("r", 7)
    fill(cache, "r", 7)
    before = [cache.read(layer, "r", depth) for layer in range(2) for depth in range(4)]
    snapshot = cache.snapshot("r")
    assert snapshot.keys.dtype == torch.float8_e4m3fn and snapshot.key_scales is not None
    cache.free("r")
    assert cache.allocate("r", 7)
    cache.restore("r", snapshot)
    after = [cache.read(layer, "r", depth) for layer in range(2) for depth in range(4)]
    assert all(torch.equal(a, b) for x, y in zip(before, after) for a, b in zip(x, y))
    plain = make_cache("auto")
    assert plain.allocate("r", 7)
    with pytest.raises(ValueError, match="dtype"):
        plain.restore("r", snapshot)


def test_backends_without_fp8_support_are_rejected():
    info = {"backend": "flash", "selected_backend": "flash"}
    backend = AttentionBackend(torch_paged_attention, BackendCapabilities(), info)
    with pytest.raises(ValueError, match="FP8"):
        make_cache(attention=backend)
    from vllm_rlt.pd.engine import PDEngine

    torch.manual_seed(0)
    with pytest.raises(ValueError, match="FP8"):
        PDEngine(
            OuroForCausalLM(tiny_ouro_config()),
            prefill_cache_config=CacheConfig(kv_cache_dtype="fp8_e4m3"),
            attention_backend="triton",
        )


def run_engine(model, kv_cache_dtype, prompts, params, exit_config):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(128, 4, kv_cache_dtype=kv_cache_dtype),
        exit_config=exit_config,
    )
    assert (engine.cache_manager.key_scale is not None) == (kv_cache_dtype == "fp8_e4m3")
    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, params)
    done = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                done[output.request_id] = output
    assert engine.cache_manager.num_used_blocks == 0
    return done


def tiny_model():
    torch.manual_seed(5)
    model = OuroForCausalLM(tiny_ouro_config())
    with torch.no_grad():
        model.lm_head.weight.mul_(8)  # decisive logits: FP8 KV error stays below the margins
    return model


def test_fixed_depth_engine_tokens_match_unquantized_kv():
    model = tiny_model()
    prompts = [[2, 3, 4, 5], [7, 8], [9, 10, 11]]
    params = SamplingParams(max_tokens=12, ignore_eos=True)
    plain = run_engine(model, "auto", prompts, params, ExitConfig("ouro"))
    fp8 = run_engine(model, "fp8_e4m3", prompts, params, ExitConfig("ouro"))
    assert {k: o.token_ids for k, o in fp8.items()} == {k: o.token_ids for k, o in plain.items()}


def test_early_exit_trace_replays_through_fp8_finalize_copies():
    # A fixed exit trace isolates the LAST_EXITED copy path from gate sensitivity.
    traces = {"0": [4, 1, 3, 2, 4, 1, 2, 3], "1": [4, 2, 2, 1, 1, 4, 3, 2]}
    params = SamplingParams(max_tokens=8, ignore_eos=True, min_loops=1)
    done = run_engine(
        tiny_model(),
        "fp8_e4m3",
        [[2, 3, 4, 5], [7, 8]],
        params,
        ExitConfig("trace", depths_by_request=traces),
    )
    assert {k: list(o.exit_depths) for k, o in done.items()} == traces


@pytest.mark.gpu
def test_triton_fp8_attention_matches_torch_reference():
    from vllm_rlt.kernels.triton_attention import paged_attention

    torch.manual_seed(3)
    codes, scale = quantize_fp8(torch.randn(40, 16, 4, 128, device="cuda"))
    tables = torch.randint(0, 40, (6, 9), device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1, 17, 50, 100, 144, 3], device="cuda", dtype=torch.int32)
    q = torch.randn(6, 8, 128, device="cuda", dtype=torch.bfloat16)
    got = paged_attention(q, codes, codes, tables, lengths, scale, scale)
    want = torch_paged_attention(q, codes, codes, tables, lengths, scale, scale)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["async", "multi"])
@pytest.mark.parametrize("graphs", [False, True])
def test_fp8_async_and_graph_paths_match_sync_eager(mode, graphs):
    # Covers the async finalize kernel (scale copies) and quantized writes inside graphs.
    torch.manual_seed(321)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    traces = {str(i): [4, 1, 3, 2, 4, 2, 1, 3, 2][: 5 + i] for i in range(3)}

    def run(execution):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=64, block_size=16, kv_cache_dtype="fp8_e4m3"),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
            ),
            execution_config=execution,
            exit_config=ExitConfig("trace", depths_by_request=traces),
            attention_backend="triton",
        )
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
        assert engine.cache_manager.num_used_blocks == 0
        return done

    reference = run(ExecutionConfig())
    candidate = run(
        ExecutionConfig(async_scheduling=True, multi_stream=mode == "multi", cuda_graphs=graphs)
    )
    assert candidate == reference


@pytest.mark.gpu
def test_triton_fp8_store_matches_quantize_fp8_bitwise():
    from vllm_rlt.kernels.triton_attention import store_fp8

    torch.manual_seed(2)
    pages, layers, block, kv_heads, head_dim, rows, layer = 12, 3, 8, 4, 128, 40, 1
    shape = (pages, layers, block, kv_heads, head_dim)
    codes = [torch.zeros(shape, device="cuda", dtype=torch.float8_e4m3fn) for _ in range(2)]
    scales = [torch.ones(shape[:-1], device="cuda") for _ in range(2)]
    slots = torch.randperm(pages * block, device="cuda")[:rows]
    blocks, offsets = slots // block, slots % block
    magnitudes = torch.logspace(-30, 4, rows, device="cuda").view(rows, 1, 1)
    k = (torch.randn(rows, kv_heads, head_dim, device="cuda") * magnitudes).to(torch.bfloat16)
    k[3] = 0  # A zero vector keeps a finite scale and zero codes.
    # V rows strided across heads, as a slice of a wider projection would be.
    v = torch.randn(rows, kv_heads, 2 * head_dim, device="cuda", dtype=torch.bfloat16)
    v = v[..., :head_dim]
    store_fp8(
        k,
        v,
        *(t[:, layer] for t in codes),
        *(t[:, layer] for t in scales),
        blocks,
        offsets,
        FP8_E4M3_MAX,
    )
    untouched = torch.ones(shape[:-1], dtype=torch.bool, device="cuda")
    untouched[blocks, layer, offsets] = False
    for x, code, scale in zip((k, v), codes, scales):
        want_codes, want_scale = quantize_fp8(x)
        assert torch.equal(code[blocks, layer, offsets].float(), want_codes.float())
        assert torch.equal(scale[blocks, layer, offsets], want_scale)
        assert not code.float()[untouched].any() and bool((scale[untouched] == 1).all())


def run_cuda_engine(model, backend, execution, traces, prompts, kv_cache_dtype="fp8_e4m3"):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=16, kv_cache_dtype=kv_cache_dtype),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=64, prefill_chunk_size=32
        ),
        execution_config=execution,
        exit_config=ExitConfig("trace", depths_by_request=traces),
        attention_backend=backend,
    )
    for i, prompt in enumerate(prompts):
        params = SamplingParams(max_tokens=len(traces[str(i)]), min_loops=1, ignore_eos=True)
        engine.add_request(str(i), prompt, params)
    done = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                done[output.request_id] = (output.token_ids, list(output.exit_depths))
    assert engine.cache_manager.num_used_blocks == 0
    return done


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_triton_fp8_engine_matches_torch_fp8_engine(graphs):
    # Fused quantized writes and Triton reads, through chunked prefill and decode, against
    # the Torch backend's reference writes and reads; FP32 isolates the KV path.
    torch.manual_seed(5)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda")
    with torch.no_grad():
        model.lm_head.weight.mul_(8)
    prompts = [[2 + (7 * i + n) % 60 for i in range(n)] for n in (5, 41, 100)]
    traces = {str(i): [4, 1, 3, 2, 4, 2, 1, 3, 2, 4, 3, 1][: 8 + 2 * i] for i in range(3)}
    reference = run_cuda_engine(model, "torch", ExecutionConfig(), traces, prompts)
    execution = ExecutionConfig(async_scheduling=graphs, cuda_graphs=graphs)
    assert run_cuda_engine(model, "triton", execution, traces, prompts) == reference
