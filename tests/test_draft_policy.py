"""CPU checks of the adaptive draft-count policy."""

import random

import pytest

from vllm_rlt import SpeculativeConfig
from vllm_rlt.worker.draft_policy import AdaptiveDraftPolicy, calibrated_prior, round_features


def policy(theta, **kwargs):
    kwargs.setdefault("explore_every", 0)
    kwargs.setdefault("token_budget", 4096)
    return AdaptiveDraftPolicy(4, 2, 4, prior=theta, **kwargs)


def test_uniform_features_match_general_round_features():
    rng = random.Random(0)
    p = policy([0] * 5, bucket=lambda rows: 1 << (rows - 1).bit_length())
    for _ in range(20):
        n, k = rng.randint(1, 40), rng.randint(0, 4)
        starts = [rng.randint(0, 500) for _ in range(n)]
        expected = round_features(starts, [k + 1] * n, 2, 4, p.bucket)
        assert p.uniform_features(n, float(sum(starts)), k) == pytest.approx(expected)


def test_latency_bound_cost_drafts_and_row_bound_cost_does_not():
    starts = [100] * 64
    # Pass-bound: rate(K') = sum(0.8**i) / (2K' + 4) peaks at K'=2 (drafts are sequential).
    assert policy([1.0, 0, 0, 0, 0]).choose(starts) == 2
    assert policy([0, 1.0, 0, 0, 0]).choose(starts) == 0
    # A pass cost plus a small row cost: drafting pays only for small batches.
    mixed = [1.0, 0.01, 0, 0, 0]
    assert policy(mixed).choose([100] * 4) > policy(mixed).choose([100] * 512)


def test_budget_caps_drafts_and_exploration_keeps_estimates_live():
    assert policy([1.0, 0, 0, 0, 0], token_budget=128).choose([0] * 64) == 1
    explorer = policy([0, 1.0, 0, 0, 0], explore_every=4)
    picks = [explorer.choose([0] * 8) for _ in range(16)]
    assert picks.count(0) == 12 and all(p >= 1 for p in picks[3::4])


def test_acceptance_and_cost_estimates_converge():
    p = policy(None)
    rng = random.Random(1)
    for _ in range(4000):
        accepted = 0
        while accepted < 4 and rng.random() < 0.7:
            accepted += 1
        p.observe_acceptance(4, accepted)
    assert p.alpha == pytest.approx(0.7, abs=0.02)
    truth = [0.3, 0.004, 0.001, 0.02, 0.5]
    for _ in range(400):
        n, k = rng.randint(1, 256), rng.randint(0, 4)
        x = round_features([rng.randint(0, 900) for _ in range(n)], [k + 1] * n, 2, 4)
        p.observe_time(x, sum(t * v for t, v in zip(truth, x)))
    assert p.theta == pytest.approx(truth, rel=0.05, abs=1e-4)


def test_calibrated_prior_reproduces_bucket_round_cost():
    core, coda = {1: 1.1, 64: 1.7, 128: 2.38}, {1: 0.2, 128: 1.47}
    prior = calibrated_prior(core, coda, 2, 4)
    assert prior[1] == pytest.approx(0.01, rel=0.1) and prior[2] == pytest.approx(0.01, rel=0.1)
    # One request drafting K'=2: six draft passes, two verify passes, three coda calls.
    x = round_features([0], [3], 2, 4)
    expected = 8 * 1.1 + 3 * 0.19 + 0.01 * x[1] + 0.01 * x[2]
    assert sum(t * v for t, v in zip(prior, x)) == pytest.approx(expected, rel=0.05)


def test_speculative_config_validates_draft_knobs():
    with pytest.raises(ValueError, match="draft_tokens_override"):
        SpeculativeConfig(2, draft_tokens_override=3)
    with pytest.raises(ValueError, match="draft_policy"):
        SpeculativeConfig(2, draft_policy="greedy")
    assert SpeculativeConfig(2, draft_tokens_override=0).draft_tokens_override == 0


def test_per_position_acceptance_shapes_expected_tokens():
    p = policy([1.0, 0, 0, 0, 0])
    # Position 0 is always accepted, position 1 never: deeper drafts stop paying.
    for _ in range(500):
        p.observe_acceptance(4, 1)
    assert p.acceptance(0) > 0.99 and p.acceptance(1) < 0.01
    assert p.expected_tokens(1) == pytest.approx(2.0, abs=0.02)
    assert p.expected_tokens(4) == pytest.approx(p.expected_tokens(1), abs=0.03)
    assert p.choose([0] * 64) == 1


def test_async_speculation_adapts_by_default_and_cli_can_pin_k():
    import argparse

    from vllm_rlt.entrypoints.runtime_args import add_runtime_args

    assert SpeculativeConfig(2).draft_policy == "adaptive"
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    assert parser.parse_args([]).draft_policy == "adaptive"
    assert parser.parse_args(["--draft-policy", "fixed"]).draft_policy == "fixed"
