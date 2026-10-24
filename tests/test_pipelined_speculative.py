"""Integration tests for PipelinedSpeculativeRunner matching native autoregressive decode."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.worker.pipelined_speculative import PipelinedSpeculativeRunner
from vllm_rlt.worker.speculative_pipeline import PipelineSchedule


def make_cache(cfg, blocks=128):
    return KVCacheManager(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        num_blocks=blocks,
        block_size=4,
        max_loops=cfg.total_ut_steps,
        device="cpu",
        dtype=torch.float32,
    )


def native_autoregressive_decode(model, cache, prompt, max_new_tokens):
    """Ground truth sequential step-by-step decode for comparison."""
    p_len = len(prompt)
    cache.allocate("ref_seq", p_len + max_new_tokens + 16)

    # Prefill
    p_t = torch.tensor(prompt, dtype=torch.long)
    h = model.prelude(p_t)
    for d in range(model.config.total_ut_steps):
        b = cache._prepare_batch(["ref_seq"] * p_len, [d] * p_len, list(range(p_len)))
        h, _ = model.recurrent_prepared(h, b, cache, compute_gate=False)

    tokens = [int(model.coda(h[-1:]).argmax(-1).item())]

    # Decode remaining tokens sequentially
    for i in range(max_new_tokens - 1):
        curr_t = torch.tensor([tokens[-1]], dtype=torch.long)
        pos = p_len + i
        h_curr = model.prelude(curr_t)
        for d in range(model.config.total_ut_steps):
            b = cache._prepare_batch(["ref_seq"], [d], [pos])
            h_curr, _ = model.recurrent_prepared(h_curr, b, cache, compute_gate=False)
        next_t = int(model.coda(h_curr).argmax(-1).item())
        tokens.append(next_t)

    return tokens


@pytest.mark.parametrize(
    "seed,prompt,length",
    [
        (42, [1, 2, 3], 6),
        (999, [4, 12, 18, 22], 8),
        (71, [5], 5),
    ],
)
def test_pipelined_speculation_matches_native_losslessly(seed, prompt, length):
    torch.manual_seed(seed)
    cfg = tiny_ouro_config(total_ut_steps=4)
    model = OuroForCausalLM(cfg)

    # 1. Native Autoregressive Reference Decode
    ref_cache = make_cache(cfg)
    ref_tokens = native_autoregressive_decode(model, ref_cache, prompt, length)

    # 2. Pipelined Speculative Runner Decode
    pipe_cache = make_cache(cfg)
    sched = PipelineSchedule(stages=4, step_size=1, second_step=None)
    runner = PipelinedSpeculativeRunner(model, pipe_cache, sched)
    pipe_tokens = runner.generate("pipe_req", prompt, max_new_tokens=length)

    # 3. Assert 100% Token-for-Token Lossless Match
    assert ref_tokens == pipe_tokens
    assert runner.stats.committed_tokens == length
    assert runner.stats.cycles > 0


@pytest.mark.parametrize("second_step", [2, 3])
def test_second_step_rejected_at_construction(second_step):
    """Verify that unsupported q2 schedules raise cleanly at construction."""
    cfg = tiny_ouro_config(total_ut_steps=4)
    model = OuroForCausalLM(cfg)
    cache = make_cache(cfg)
    sched = PipelineSchedule(stages=4, step_size=1, second_step=second_step)
    with pytest.raises(NotImplementedError, match="second_step"):
        PipelinedSpeculativeRunner(model, cache, sched)


@pytest.mark.parametrize("mismatched_stages", [2, 8])
def test_stages_mismatch_rejected_at_construction(mismatched_stages):
    """Verify that schedule.stages not matching model total_ut_steps raises ValueError."""
    cfg = tiny_ouro_config(total_ut_steps=4)
    model = OuroForCausalLM(cfg)
    cache = make_cache(cfg)
    sched = PipelineSchedule(stages=mismatched_stages, step_size=1)
    with pytest.raises(
        ValueError,
        match=r"schedule\.stages .* must match .*model\.config\.total_ut_steps",
    ):
        PipelinedSpeculativeRunner(model, cache, sched)


def test_huginn_rejected_at_construction():
    """Verify that Huginn models lacking prelude/coda are rejected at construction."""
    from tests.test_huginn import tiny_huginn_config
    from vllm_rlt.models import HuginnForCausalLM

    cfg = tiny_huginn_config()
    model = HuginnForCausalLM(cfg)
    loops = cfg.total_ut_steps() if callable(cfg.total_ut_steps) else cfg.total_ut_steps
    cache = KVCacheManager(
        num_layers=cfg.n_layers,
        num_kv_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        num_blocks=128,
        block_size=4,
        max_loops=loops,
        device="cpu",
        dtype=torch.float32,
    )
    sched = PipelineSchedule(stages=loops, step_size=1)
    with pytest.raises(NotImplementedError, match="prelude"):
        PipelinedSpeculativeRunner(model, cache, sched)


@pytest.mark.parametrize(
    "prompt,max_new_tokens",
    [
        ([1, 2, 3], 8),
        ([4, 12, 18, 22], 12),
        ([5], 4),
    ],
)
def test_speculative_stats_invariants(prompt, max_new_tokens):
    """Verify acceptance counter invariants required by maintainer review."""
    torch.manual_seed(42)
    cfg = tiny_ouro_config(total_ut_steps=4)
    model = OuroForCausalLM(cfg)
    cache = make_cache(cfg)
    sched = PipelineSchedule(stages=4, step_size=1, second_step=None)
    runner = PipelinedSpeculativeRunner(model, cache, sched)

    tokens = runner.generate("req_stats", prompt, max_new_tokens=max_new_tokens)
    stats = runner.stats

    assert stats.committed_tokens == len(tokens)
    assert stats.accepted_proposals + stats.rejected_proposals == len(tokens) - 1
    assert 0 <= stats.accepted_proposals <= len(tokens) - 1
    assert stats.cycles >= 1
