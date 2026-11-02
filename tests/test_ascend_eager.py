"""Causal/GQA semantics for the separately selected BF16 eager contract."""

import pytest
import torch

from vllm_rlt.kernels.ascend_eager import AscendEagerAttention


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_eager_groups_match_independent_causal_queries(kv_heads):
    torch.manual_seed(730)
    keys = torch.randn(8, 16, kv_heads, 64, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    q = torch.randn(7, 4, 64, dtype=torch.bfloat16)
    lengths = torch.tensor([0, 1, 15, 16, 17, 3, 31], dtype=torch.int32)
    tables = torch.tensor([[4, 1], [4, 1], [4, 1], [4, 1], [4, 1], [7, 2], [7, 2]])
    actual = AscendEagerAttention()(q, keys, values, tables, lengths)
    expected = torch.zeros_like(q)
    for row, length in enumerate(lengths.tolist()):
        if not length:
            continue
        pos = torch.arange(length)
        k = keys[tables[row, pos // 16], pos % 16].repeat_interleave(4 // kv_heads, dim=1)
        v = values[tables[row, pos // 16], pos % 16].repeat_interleave(4 // kv_heads, dim=1)
        scores = torch.einsum("hd,thd->ht", q[row], k) * (64**-0.5)
        probs = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        expected[row] = torch.einsum("ht,thd->hd", probs, v)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0]) == 0


@pytest.mark.npu
@pytest.mark.parametrize("chunk_size", [4096, 128])
@torch.inference_mode()
def test_engine_matches_the_same_manual_prefill_chunks(chunk_size):
    from tests.helpers import tiny_ouro_config
    from vllm_rlt.models import OuroForCausalLM
    from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.device import resolve_device
    from vllm_rlt.engine.stage_engine import StageEngine

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(41)
    model = OuroForCausalLM(
        tiny_ouro_config(
            hidden_size=256, intermediate_size=512, head_dim=64, max_position_embeddings=256
        )
    ).to(device=device, dtype=torch.bfloat16)
    c = model.config
    tokens = ([3, 7, 2, 11, 5, 17, 8] * 22)[:149]
    cache = KVCacheManager(
        c.num_hidden_layers,
        c.num_key_value_heads,
        c.head_dim,
        128,
        16,
        4,
        device=device,
        dtype=torch.bfloat16,
        backend="ascend_eager",
    )
    assert cache.allocate("manual", len(tokens) + 3)
    expected, manual_states = [], []
    handle = model.model.norm.register_forward_hook(
        lambda _m, _i, output: manual_states.append(output.detach().cpu().clone())
    )
    try:
        for start in range(0, len(tokens), chunk_size):
            end = min(start + chunk_size, len(tokens))
            hidden = model.prelude(torch.tensor(tokens[start:end], device=device))
            for depth in range(4):
                hidden, _ = model.recurrent(
                    hidden,
                    ["manual"] * (end - start),
                    [depth] * (end - start),
                    list(range(start, end)),
                    cache,
                    compute_gate=False,
                )
        for step in range(3):
            expected.append(int(model.coda(hidden[-1:])[0].argmax()))
            if step < 2:
                pos = len(tokens) + step
                hidden = model.prelude(torch.tensor([expected[-1]], device=device))
                for depth in range(4):
                    hidden, _ = model.recurrent(
                        hidden, ["manual"], [depth], [pos], cache, compute_gate=False
                    )
    finally:
        handle.remove()
        cache.free("manual")
    actual_states = []
    handle = model.model.norm.register_forward_hook(
        lambda _m, _i, output: actual_states.append(output.detach().cpu().clone())
    )
    engine = StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(num_blocks=128),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=4096, prefill_chunk_size=chunk_size
        ),
    )
    try:
        engine.add_request(
            "engine",
            tokens,
            SamplingParams(max_tokens=3, temperature=0, min_loops=4, max_loops=4, ignore_eos=True),
        )
        result = None
        while engine.has_unfinished_requests():
            for out in engine.step():
                if out.finished:
                    result = out
        assert result is not None
        assert result.token_ids == expected
        assert result.exit_depths == [4] * 3
        assert len(actual_states) == len(manual_states)
        for actual, reference in zip(actual_states, manual_states, strict=True):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        handle.remove()
        engine.close()
