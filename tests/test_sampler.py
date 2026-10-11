"""Contract tests for the extracted Sampler: greedy, top-k and top-p.

These pin the current arithmetic exactly as it behaved inside
``ModelRunner._sample_tensor``; they are not a specification of what sampling
should do in the future.
"""

import pytest
import torch

from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.sampler import Sampler, apply_repetition_penalty, probabilities


def draw(sampler, logits, params, generator=None, count=1):
    """Sample ``count`` tokens, threading the generator like the runner does."""
    tokens = []
    for _ in range(count):
        token, generator = sampler.sample(logits, params, generator)
        tokens.append(token)
    return tokens, generator


def test_greedy_returns_argmax_and_never_creates_a_generator():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.tensor([0.1, 2.0, 1.0, -1.0])
    # top_k/top_p are deliberately set: the temperature == 0 short circuit runs
    # first, so greedy ignores them and stays RNG-free.
    params = SamplingParams(temperature=0.0, top_k=1, top_p=0.5)
    tokens, generator = draw(sampler, logits, params)
    token = tokens[0]
    assert token.item() == int(logits.argmax())
    assert token.dim() == 0
    assert token.dtype == torch.long
    assert generator is None


GREEDY_CASES = [
    torch.tensor([0.1, 2.0, 1.0, -1.0]),
    torch.tensor([-3.0, -3.5, -2.0, -9.0]),
    torch.tensor([5.0, 4.999, 4.998]),
]


@pytest.mark.parametrize("logits", GREEDY_CASES)
def test_top_k_one_is_equivalent_to_greedy(logits):
    sampler = Sampler(torch.device("cpu"))
    greedy_tokens, _ = draw(sampler, logits, SamplingParams(temperature=0.0))
    # top_k=1 makes the threshold the maximum, so masked_fill leaves exactly one
    # candidate and the result cannot depend on the RNG stream. Checking several
    # logits rows and seeds pins that mechanism, not one coincidental argmax.
    for seed in (0, 7, 12345):
        params = SamplingParams(temperature=1.0, top_k=1, seed=seed)
        top_k_tokens, generator = draw(sampler, logits, params)
        assert generator is not None
        assert top_k_tokens[0].item() == greedy_tokens[0].item()


def test_top_k_keeps_every_token_tied_at_the_threshold():
    sampler = Sampler(torch.device("cpu"))
    # top_k=2 selects a threshold of 3.0, which three tokens share. The mask is
    # ``logits < threshold``, so the candidate set is larger than k.
    logits = torch.tensor([3.0, 3.0, 3.0, 0.0])
    params = SamplingParams(temperature=1.0, top_k=2, seed=11)
    tokens, _ = draw(sampler, logits, params, count=200)
    observed = {token.item() for token in tokens}
    assert observed <= {0, 1, 2}
    assert len(observed) > 1


def test_top_p_keeps_at_least_one_token():
    sampler = Sampler(torch.device("cpu"))
    # A top_p far below the head probability would empty the candidate set; the
    # mask shift keeps the single most likely token instead.
    logits = torch.linspace(1.0, 0.0, 100)
    params = SamplingParams(temperature=1.0, top_p=0.005, seed=3)
    tokens, _ = draw(sampler, logits, params, count=20)
    assert {token.item() for token in tokens} == {0}


def test_top_p_one_does_not_truncate_the_support():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.tensor([1.0, 0.9, 0.8, 0.7])
    truncated = SamplingParams(temperature=1.0, top_p=0.5, seed=13)
    untruncated = SamplingParams(temperature=1.0, top_p=1.0, seed=13)
    truncated_tokens, _ = draw(sampler, logits, truncated, count=200)
    untruncated_tokens, _ = draw(sampler, logits, untruncated, count=200)
    assert {token.item() for token in truncated_tokens} <= {0, 1}
    # Index 3 is the least likely token, so it is reachable only while the
    # support is intact. This asserts reachability over a fixed 200-draw stream
    # (p(miss) ~ 0.79**200), not the exact draw sequence: the exact sequence
    # would couple the test to torch's multinomial implementation.
    assert 3 in {token.item() for token in untruncated_tokens}


def sequence(sampler, logits, params, count=32):
    tokens, _ = draw(sampler, logits, params, count=count)
    return [token.item() for token in tokens]


def test_same_seed_reproduces_the_same_sequence():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, top_p=0.9, seed=42)
    assert sequence(sampler, logits, params) == sequence(sampler, logits, params)


def test_different_seeds_produce_different_sequences():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.linspace(1.0, -1.0, 64)
    base = dict(temperature=0.7, top_k=8, top_p=0.9)
    assert sequence(sampler, logits, SamplingParams(seed=1, **base)) != sequence(
        sampler, logits, SamplingParams(seed=2, **base)
    )


def test_generator_is_reused_and_advanced_across_calls():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, seed=42)
    _, generator = draw(sampler, logits, params)
    assert generator is not None
    state = generator.get_state().clone()
    # Later calls must advance the same generator instead of reseeding it, which
    # is what makes a multi-token request reproducible.
    _, same_generator = draw(sampler, logits, params, generator=generator)
    assert same_generator is generator
    assert not torch.equal(generator.get_state(), state)


def test_repetition_penalty_scales_positive_and_negative_logits():
    logits = torch.tensor([2.0, -2.0, 1.0])
    penalized = apply_repetition_penalty(logits.clone(), 2.0, [0, 1])
    assert torch.allclose(penalized, torch.tensor([1.0, -4.0, 1.0]))


def test_repetition_penalty_with_empty_or_neutral():
    logits = torch.tensor([2.0, -2.0, 1.0])
    assert torch.equal(apply_repetition_penalty(logits.clone(), 1.0, [0, 1]), logits)
    assert torch.equal(apply_repetition_penalty(logits.clone(), 2.0, []), logits)


def test_repetition_penalty_greedy_in_sampler():
    sampler = Sampler(torch.device("cpu"))
    # Token 1 has highest logit initially (3.0 > 2.0 > 0.0)
    logits = torch.tensor([2.0, 3.0, 0.0])
    params = SamplingParams(temperature=0.0, repetition_penalty=2.0)
    tok_clean, gen = sampler.sample(logits, params, None)
    assert tok_clean.item() == 1
    assert gen is None
    # With token 1 penalized: token 1 logit becomes 3.0 / 2.0 = 1.5 < 2.0 (token 0)
    tok_penalized, gen = sampler.sample(logits, params, None, token_ids=[1])
    assert tok_penalized.item() == 0
    assert gen is None


def test_dynamic_depth_temperature_distribution():
    # At temperature 1.0, depth 1 (loops_done=1) cools down (0.7), sharpening the mode.
    # Depth 4 (loops_done=4) heats up (1.2), flattening the distribution.
    logits = torch.tensor([2.0, 0.0])
    params = SamplingParams(temperature=1.0, dynamic_depth_temp=True)
    p_base = probabilities(logits, params, temperature=1.0)
    p_cold = probabilities(logits, params, temperature=max(params.temperature * 0.7, 0.1))
    p_hot = probabilities(logits, params, temperature=params.temperature * 1.2)
    assert p_cold[0] > p_base[0] > p_hot[0]


def test_dynamic_depth_temp_in_sampler():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.tensor([2.0, 0.0])
    params = SamplingParams(temperature=1.0, dynamic_depth_temp=True, seed=42)

    # Verify that dynamic depth temperature preserves generator threading across steps
    _, gen = sampler.sample(logits, params, None, loops_done=1)
    assert gen is not None
    state = gen.get_state().clone()
    _, gen_next = sampler.sample(logits, params, gen, loops_done=2)
    assert gen_next is gen
    assert not torch.equal(gen.get_state(), state)

    # In greedy mode, dynamic_depth_temp is ignored and creates no generator
    greedy_params = SamplingParams(temperature=0.0, dynamic_depth_temp=True)
    tok, gen_greedy = sampler.sample(logits, greedy_params, None, loops_done=1)
    assert tok.item() == 0
    assert gen_greedy is None
