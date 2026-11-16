# SPDX-License-Identifier: Apache-2.0
"""CUDA checks for Dory: windowed Triton parity, engine parity and KV profiling.

Every case needs an explicit ``CUDA_VISIBLE_DEVICES`` and ``--run-gpu``. The CPU
suite covers the same contracts against the independent dense oracle.
"""

import pytest
import torch

from tests.reference.dory import dense_dory_reference
from tests.test_dory import tiny_dory_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.attention import create_backend
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import plan_cache
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import DoryForCausalLM

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


def make_gpu_model(**overrides):
    torch.manual_seed(42)
    return (
        DoryForCausalLM(tiny_dory_config(**overrides)).to(device=DEVICE, dtype=torch.float32).eval()
    )


def make_gpu_cache(model, backend="triton", block_size=2, num_blocks=256):
    config = model.config
    return KVCacheManager(
        num_layers=config.num_kv_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        max_loops=config.total_ut_steps,
        device=DEVICE,
        dtype=torch.float32,
        backend=backend,
        layout=config.kv_cache_layout,
        recurrent_layers=model.recurrent_kv_layers,
    )


def staged_gpu_logits(model, tokens):
    """Reference schedule from the CPU tests, on CUDA with Triton attention."""
    config = model.config
    cache = make_gpu_cache(model)
    assert cache.allocate("request", len(tokens))
    logits = []
    for position in range(len(tokens)):
        boundary = cache._prepare_batch(["request"], [0], [position])
        hidden = model.prelude_prepared(tokens[position : position + 1].to(DEVICE), boundary, cache)
        for depth in range(config.total_ut_steps):
            core = cache._prepare_batch(["request"], [depth], [position])
            hidden, _ = model.recurrent_prepared(hidden, core, cache)
        logits.append(model.coda_prepared(hidden, boundary, cache)[0].float())
    cache.free("request")
    assert cache.num_used_blocks == 0
    return torch.stack(logits).cpu()


def greedy_reference(model, prompt, count):
    tokens = list(prompt)
    for _ in range(count):
        logits = dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(logits[-1].argmax()))
    return tokens[len(prompt) :]


@pytest.mark.parametrize("head_dim", [8, 128, 256])
@pytest.mark.parametrize("window", [1, 3, 16, 64])
@pytest.mark.parametrize("length", [1, 7, 63, 64, 65])
def test_windowed_triton_decode_matches_torch_reference(head_dim, window, length):
    device = torch.device(DEVICE)
    reference = create_backend("torch", device, torch.float32, head_dim, 16)
    kernel = create_backend("triton", device, torch.float32, head_dim, 16)
    assert kernel.capabilities.sliding_window
    torch.manual_seed(head_dim + window + length)
    blocks, block_size, kv_heads, heads = 8, 16, 2, 4
    keys = torch.randn(blocks, block_size, kv_heads, head_dim, device=device)
    values = torch.randn(blocks, block_size, kv_heads, head_dim, device=device)
    tables = torch.arange(blocks, device=device, dtype=torch.int32)[None, :]
    query = torch.randn(1, heads, head_dim, device=device)
    lengths = torch.tensor([length], device=device, dtype=torch.int32)
    expected = reference.windowed(query, keys, values, tables, lengths, window)
    actual = kernel.windowed(query, keys, values, tables, lengths, window)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("head_dim", [8, 256])
def test_unwindowed_triton_decode_is_unchanged(head_dim):
    device = torch.device(DEVICE)
    reference = create_backend("torch", device, torch.float32, head_dim, 16)
    kernel = create_backend("triton", device, torch.float32, head_dim, 16)
    torch.manual_seed(head_dim)
    blocks, block_size, kv_heads, heads = 4, 16, 2, 4
    keys = torch.randn(blocks, block_size, kv_heads, head_dim, device=device)
    values = torch.randn(blocks, block_size, kv_heads, head_dim, device=device)
    tables = torch.arange(blocks, device=device, dtype=torch.int32)[None, :]
    query = torch.randn(1, heads, head_dim, device=device)
    lengths = torch.tensor([37], device=device, dtype=torch.int32)
    torch.testing.assert_close(
        kernel(query, keys, values, tables, lengths),
        reference(query, keys, values, tables, lengths),
        atol=2e-6,
        rtol=2e-5,
    )


@pytest.mark.parametrize("kv_mode", ["per_loop", "last_loop"])
def test_dory_gpu_triton_matches_cpu_dense_oracle(kv_mode):
    tokens = torch.tensor([4, 7, 9, 3, 12, 5])
    reference = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=kv_mode)).eval()
    torch.manual_seed(42)
    reference.load_state_dict(
        {
            name: value.cpu()
            for name, value in make_gpu_model(recurrent_kv_cache_mode=kv_mode).state_dict().items()
        }
    )
    expected = dense_dory_reference(reference, tokens)["logits"]
    model = make_gpu_model(recurrent_kv_cache_mode=kv_mode)
    model.load_state_dict(reference.state_dict())
    torch.testing.assert_close(staged_gpu_logits(model, tokens), expected, atol=2e-5, rtol=2e-4)


def test_dory_engine_greedy_matches_cpu_oracle_on_gpu():
    reference = DoryForCausalLM(tiny_dory_config()).eval()
    torch.manual_seed(42)
    reference.load_state_dict(
        {name: value.cpu() for name, value in make_gpu_model().state_dict().items()}
    )
    model = make_gpu_model()
    model.load_state_dict(reference.state_dict())
    engine = LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(num_blocks=256, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
    )
    prompt = [4, 7, 3]
    engine.add_request(
        "request",
        prompt,
        SamplingParams(max_tokens=3, min_loops=2, max_loops=2, ignore_eos=True),
    )
    finished = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output
    result = finished["request"]
    assert result.exit_depths == [2, 2, 2]
    assert engine.cache_manager.num_used_blocks == 0
    assert result.token_ids == greedy_reference(reference, prompt, 3)


def test_plan_cache_profiles_dory_on_cuda():
    model = make_gpu_model()
    cache = CacheConfig(block_size=2, memory_reserve_bytes=0)
    scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8)
    blocks, plan = plan_cache(model, cache, scheduler, ExecutionConfig(), "triton")
    assert blocks > 0
    # The probe ran the real staged path, including windowed boundary attention.
    assert plan["source"] == "cuda_profile"
    config = model.config
    assert plan["bytes_per_block"] == (
        2
        * config.num_kv_layers
        * cache.block_size
        * config.num_key_value_heads
        * config.head_dim
        * 4
    )


EXECUTION_MODES = {
    "sync_eager": dict(),
    "sync_graph": dict(cuda_graphs=True),
    "async_eager": dict(async_scheduling=True),
    "async_graph": dict(async_scheduling=True, cuda_graphs=True),
}
# Prompts shorter than, equal to and longer than the tiny window of four.
PROMPTS = {"a": [4, 7, 3], "b": [9, 2, 6, 5], "c": [8, 3, 3, 11, 5, 2, 7, 13, 6]}


def reference_pair():
    reference = DoryForCausalLM(tiny_dory_config()).eval()
    torch.manual_seed(42)
    model = make_gpu_model()
    reference.load_state_dict({name: value.cpu() for name, value in model.state_dict().items()})
    return reference, model


def make_engine(model, mode, *, max_num_seqs=3):
    return LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(num_blocks=512, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=16),
        execution_config=ExecutionConfig(**EXECUTION_MODES[mode]),
    )


def drain(engine, *, abort=None, after=0):
    finished, steps = {}, 0
    while engine.has_unfinished_requests():
        if abort is not None and steps == after:
            engine.abort_request(abort)
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output
        steps += 1
    return finished


@pytest.mark.parametrize("mode", EXECUTION_MODES)
@pytest.mark.parametrize("max_num_seqs", [1, 2, 3])
def test_dory_execution_modes_match_cpu_oracle(mode, max_num_seqs):
    """Graph replay and asynchronous submission are separate switches; each
    combination serves the same greedy tokens as the dense oracle."""
    reference, model = reference_pair()
    engine = make_engine(model, mode, max_num_seqs=max_num_seqs)
    params = SamplingParams(max_tokens=6, min_loops=2, max_loops=2, ignore_eos=True)
    for request_id, prompt in PROMPTS.items():
        engine.add_request(request_id, prompt, params)
    finished = drain(engine)
    assert set(finished) == set(PROMPTS)
    for request_id, prompt in PROMPTS.items():
        assert finished[request_id].token_ids == greedy_reference(reference, prompt, 6)
        assert finished[request_id].exit_depths == [2] * 6
    assert engine.cache_manager.num_used_blocks == 0
    runner = engine.model_runner
    if "graph" in mode:
        # Every decode stage replays a graph; prompt prefill stays eager.
        for graphs in (runner.prelude_graphs, runner.graphs, runner.coda_graphs):
            assert graphs.replays > 0
    else:
        assert runner.graphs is None and runner.coda_graphs is None
        assert runner.prelude_graphs is None


@pytest.mark.parametrize("mode", EXECUTION_MODES)
@pytest.mark.parametrize("after", [0, 1, 3, 6, 9])
def test_dory_cancellation_releases_kv_in_every_execution_mode(mode, after):
    reference, model = reference_pair()
    engine = make_engine(model, mode)
    params = SamplingParams(max_tokens=6, min_loops=2, max_loops=2, ignore_eos=True)
    for request_id, prompt in PROMPTS.items():
        engine.add_request(request_id, prompt, params)
    finished = drain(engine, abort="b", after=after)
    assert set(finished) >= {"a", "c"}
    for request_id in ("a", "c"):
        assert finished[request_id].token_ids == greedy_reference(reference, PROMPTS[request_id], 6)
    assert engine.cache_manager.num_used_blocks == 0
    # The cancelled ID is reusable and serves the full answer afterwards.
    engine.add_request("b", PROMPTS["b"], params)
    assert drain(engine)["b"].token_ids == greedy_reference(reference, PROMPTS["b"], 6)
    assert engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_dory_failed_submission_releases_kv_in_every_execution_mode(mode, monkeypatch):
    reference, model = reference_pair()
    engine = make_engine(model, mode)
    params = SamplingParams(max_tokens=4, min_loops=2, max_loops=2, ignore_eos=True)
    for request_id, prompt in PROMPTS.items():
        engine.add_request(request_id, prompt, params)
    for _ in range(4):
        engine.step()
    submit = engine.executor.submit

    def failing(batch):
        raise RuntimeError("injected submission failure")

    monkeypatch.setattr(engine.executor, "submit", failing)
    with pytest.raises(RuntimeError, match="injected"):
        engine.step()
    monkeypatch.setattr(engine.executor, "submit", submit)
    for request_id in list(engine.scheduler.requests):
        engine.abort_request(request_id)
    assert engine.cache_manager.num_used_blocks == 0
    engine.add_request("a", PROMPTS["a"], params)
    assert drain(engine)["a"].token_ids == greedy_reference(reference, PROMPTS["a"], 4)
    assert engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("window", [0, 32])
def test_dory_packed_attention_matches_per_row_reference(window):
    if torch.cuda.get_device_capability() != (12, 0) or torch.version.hip:
        pytest.skip("Dory packed prefill requires NVIDIA SM 12.0")
    from vllm_rlt.kernels.triton_attention import paged_attention, paged_prefill_attention

    # The first chunk is far enough into its history to skip a full SWA tile.
    # The two sequences have distinct pages and different causal offsets.
    torch.manual_seed(123)
    q = torch.randn((42, 16, 256), device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn((10, 16, 4, 256), device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    tables = torch.tensor(
        [[8, 2, 3, 0, 1, 4, 5, 6, 7], [9, -1, -1, -1, -1, -1, -1, -1, -1]],
        device=DEVICE,
        dtype=torch.int32,
    )
    lengths = torch.tensor([132, 15], device=DEVICE, dtype=torch.int32)
    offsets = torch.tensor([0, 35, 42], device=DEVICE, dtype=torch.int32)
    row_tables = torch.cat((tables[:1].expand(35, -1), tables[1:].expand(7, -1)))
    row_lengths = torch.tensor([*range(98, 133), *range(9, 16)], device=DEVICE, dtype=torch.int32)
    expected = paged_attention(q, k, v, row_tables, row_lengths, window=window)
    actual = paged_prefill_attention(
        q, k, v, tables, lengths, offsets, 35, window=window, compensate=True
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("multi_stream", [True, False])
@pytest.mark.parametrize(("long_tokens", "max_num_seqs"), [(65, 1), (129, 2)])
def test_dory_packed_prefill_preserves_short_prompt_logits(
    monkeypatch, multi_stream, long_tokens, max_num_seqs
):
    if torch.cuda.get_device_capability() != (12, 0) or torch.version.hip:
        pytest.skip("Dory packed prefill requires NVIDIA SM 12.0")
    model = make_gpu_model(head_dim=256, max_position_embeddings=256, sliding_window=32).to(
        dtype=torch.bfloat16
    )
    prompt = [4, 7, 9, 3, 12, 5, 23]
    scheduler = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=128,
        prefill_chunk_size=64,
    )

    def run(*, enable):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=64),
            scheduler_config=scheduler,
            attention_backend="triton",
            execution_config=ExecutionConfig(dory_packed_prefill=enable, multi_stream=multi_stream),
        )
        packed_layers = set()
        short_logits = []
        sampler = engine.model_runner.sampler
        original_sample = sampler.sample
        original_attention = engine.cache_manager._attend_prepared

        def sample(logits, params, generator):
            if params is short_options:
                short_logits.append(logits.detach().cpu())
            return original_sample(logits, params, generator)

        def attention(layer, metadata, q, *, window=None):
            if metadata.cu_seqlens_q is not None:
                assert all(rid == "long" for rid, _ in metadata.allocations)
                packed_layers.add((layer, window is not None))
            return original_attention(layer, metadata, q, window=window)

        short_options = SamplingParams(max_tokens=3, ignore_eos=True)
        with monkeypatch.context() as mp:
            mp.setattr(sampler, "sample", sample)
            mp.setattr(engine.cache_manager, "_attend_prepared", attention)
            try:
                assert engine.execution_config.packed_prefill is False
                engine.add_request(
                    "long",
                    (prompt * 19)[:long_tokens],
                    SamplingParams(max_tokens=3, ignore_eos=True),
                )
                engine.add_request("short", (prompt * 10)[:64], short_options)
                while engine.has_unfinished_requests():
                    engine.step()
                assert engine.cache_manager.num_used_blocks == 0
            finally:
                engine.close()
        return torch.stack(short_logits), packed_layers

    baseline, baseline_packed = run(enable=False)
    actual, packed = run(enable=True)
    assert not baseline_packed
    assert {layer for layer, _ in packed} == set(range(model.config.num_kv_layers))
    assert {window for _, window in packed} == {False, True}
    assert torch.equal(baseline, actual)


def test_dory_packed_prefill_keeps_small_value_residual():
    if torch.cuda.get_device_capability() != (12, 0) or torch.version.hip:
        pytest.skip("Dory packed prefill requires NVIDIA SM 12.0")
    from vllm_rlt.kernels.triton_attention import paged_attention, paged_prefill_attention

    q = torch.zeros((1, 1, 256), device=DEVICE, dtype=torch.bfloat16)
    kv = torch.zeros((1, 16, 1, 256), device=DEVICE, dtype=torch.bfloat16)
    k, v = kv.clone(), kv.clone()
    q[0, 0, 0], k[0, 1, 0, 0] = 16, -0.5
    # Rounding exp(-0.5) to BF16 before multiplying would zero the numerator.
    v[0, 0, 0, 0], v[0, 1, 0, 0] = -0.60546875, 1
    table = torch.tensor([[0]], device=DEVICE, dtype=torch.int32)
    length = torch.tensor([2], device=DEVICE, dtype=torch.int32)
    offset = torch.tensor([0, 1], device=DEVICE, dtype=torch.int32)
    expected = paged_attention(q, k, v, table, length)
    actual = paged_prefill_attention(q, k, v, table, length, offset, 1, compensate=True)
    assert expected[0, 0, 0] > 0.0006
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=0)
