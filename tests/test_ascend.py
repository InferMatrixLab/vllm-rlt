"""Opt-in synchronous BF16 Ouro checks using explicit Torch attention on Ascend."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.models import OuroForCausalLM

pytestmark = pytest.mark.npu


@pytest.fixture
def npu_device():
    resolve_device("npu")
    torch.npu.set_device(0)
    yield torch.device("npu:0")
    torch.npu.synchronize()
    torch.npu.empty_cache()


def assert_bf16_close(actual, expected):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert (a - b).norm() / b.norm().clamp_min(1e-8) <= 0.02
    torch.testing.assert_close(a, b, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_torch_attention_on_npu_matches_cpu_prefix_oracle(npu_device, kv_heads):
    torch.manual_seed(83)
    keys = torch.randn(8, 2, 16, kv_heads, 64, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    q = torch.randn(64, 4, 64, dtype=torch.bfloat16)
    lengths = torch.tensor([0, 1, 15, 16, 17, 31, 32, 3] * 8, dtype=torch.int32)
    tables = torch.tensor([[5, 2], [1, 7]] * 32, dtype=torch.int32)
    expected = torch_paged_attention(q, keys[:, 1], values[:, 1], tables, lengths)
    actual = torch_paged_attention(
        q.to(npu_device),
        keys.to(npu_device)[:, 1],
        values.to(npu_device)[:, 1],
        tables.to(npu_device),
        lengths.to(npu_device),
    )
    assert_bf16_close(actual, expected)
    assert torch.count_nonzero(actual[0]).item() == 0


def make_npu_model(device):
    torch.manual_seed(41)
    config = tiny_ouro_config(hidden_size=256, intermediate_size=512, head_dim=64)
    return OuroForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()


@torch.inference_mode()
def recurrent_states(model, chunks, *, tokens=None, incremental=False):
    c = model.config
    cache = KVCacheManager(
        c.num_hidden_layers,
        c.num_key_value_heads,
        c.head_dim,
        32,
        16,
        4,
        device=next(model.parameters()).device,
        dtype=torch.bfloat16,
        backend="torch",
        incremental_allocation=incremental,
    )
    tokens = tokens if tokens is not None else ([3, 7, 2, 11, 5, 17, 8] * 3)[:21]
    assert cache.allocate("a", len(tokens), initial_tokens=16 if incremental else None)
    states, kv, start = {}, {}, 0
    try:
        for size in chunks:
            assert cache.ensure_capacity("a", start + size)
            positions = list(range(start, start + size))
            hidden = model.prelude(torch.tensor(tokens[start : start + size], device=cache.device))
            for depth in range(4):
                hidden, _ = model.recurrent(hidden, ["a"] * size, [depth] * size, positions, cache)
                states[(start, depth)] = (hidden.clone(), model.coda(hidden).clone())
            start += size
        assert start == len(tokens)
        for depth in range(4):
            for layer in range(c.num_hidden_layers):
                assert all(cache.token_written("a", p, depth) for p in range(len(tokens)))
                kv[(layer, depth)] = tuple(
                    t.clone() for t in cache.read(layer, "a", depth, len(tokens))
                )
        assert len({tuple(cache.get_block_table("a", d)) for d in range(4)}) == 4
        return states, kv
    finally:
        cache.free("a")
        assert cache.num_used_blocks == 0


def test_npu_chunked_prefill_and_incremental_kv(npu_device):
    model = make_npu_model(npu_device)
    expected, expected_kv = recurrent_states(model, [19, 1, 1])
    incremental, incremental_kv = recurrent_states(model, [19, 1, 1], incremental=True)
    for key in expected:
        for a, b in zip(incremental[key], expected[key], strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key in expected_kv:
        for a, b in zip(incremental_kv[key], expected_kv[key], strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    chunked, chunked_kv = recurrent_states(model, [5, 14, 1, 1], incremental=True)
    for key in ((19, d) for d in range(4)):
        for a, b in zip(chunked[key], expected[key], strict=True):
            assert_bf16_close(a, b)
    for key in expected_kv:
        for a, b in zip(chunked_kv[key], expected_kv[key], strict=True):
            assert_bf16_close(a, b)


def test_npu_cancel_reuse_and_multiple_requests(npu_device):
    engine = StageEngine(
        make_npu_model(npu_device),
        attention_backend="torch",
        cache_config=CacheConfig(num_blocks=32, block_size=16, incremental_allocation=True),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=32),
    )
    try:
        params = SamplingParams(max_tokens=3, temperature=0, ignore_eos=True)
        engine.add_request("cancel", [3, 7, 2] * 6, params)
        engine.add_request("keep", [5, 9, 11], params)
        engine.step()
        engine.abort_request("cancel")
        engine.add_request("cancel", [13, 14, 15], params)
        finished = {}
        for _ in range(200):
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = output
            if not engine.has_unfinished_requests():
                break
        assert set(finished) == {"keep", "cancel"}
        assert all(
            o.exit_depths == [4] * 3 and o.finish_reason == "length" for o in finished.values()
        )
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_npu_eos_and_length_handling(npu_device, ignore_eos):
    model = make_npu_model(npu_device)
    model.lm_head.weight.zero_()  # Argmax 0 is the tiny model's actual EOS token.
    engine = StageEngine(model, attention_backend="torch", cache_config=CacheConfig(num_blocks=16))
    try:
        engine.add_request(
            "eos", [3, 7, 2], SamplingParams(max_tokens=3, temperature=0, ignore_eos=ignore_eos)
        )
        output = None
        for _ in range(100):
            for item in engine.step():
                if item.finished:
                    output = item
            if not engine.has_unfinished_requests():
                break
        assert output is not None
        assert output.token_ids == [0] * (3 if ignore_eos else 1)
        assert output.finish_reason == ("length" if ignore_eos else "stop")
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()
