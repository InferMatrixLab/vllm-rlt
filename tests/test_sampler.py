"""Contract tests for sampling: the extracted Sampler, per-token logprobs and seeds.

``Sampler`` only consumes a generator the caller resolved, so these tests create
one themselves (mirroring what ``RngRegistry.acquire`` does in production) and
keep ``RngRegistry``'s own creation and lifetime contracts separate. The
distribution tests state the sampling contract itself, including strict top-k:
exactly ``k`` candidates survive by index, so a tie at the k-th logit does not
widen the support. The rest cover ``token_logprob`` (``worker/logprobs.py``) in raw
and processed modes, the batched ``processed_logprobs`` and
``Sampler.sample_with_logprob`` against it, and the registry's rejection of an
unresolved ``seed=None``.
"""

import math

import pytest
import torch

import vllm_rlt.worker.sampler as sampler_module
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.logprobs import processed_logprobs, token_logprob, truncates
from vllm_rlt.worker.sampler import RngRegistry, Sampler


def seeded_generator(params, device=torch.device("cpu")):
    """Caller-side generator creation, mirroring ``RngRegistry.acquire``."""
    return torch.Generator(device=device).manual_seed(params.seed)


def sample_many(sampler, logits, params, count=1):
    """Sample ``count`` tokens with a generator this test owns (greedy gets None)."""
    generator = None if params.temperature == 0 else seeded_generator(params)
    return [int(sampler.sample(logits, params, generator)) for _ in range(count)]


def test_greedy_returns_argmax_without_a_generator():
    logits = torch.tensor([0.1, 2.0, 1.0, -1.0])
    # top_k/top_p are deliberately set: the temperature == 0 short circuit runs
    # first, so greedy ignores them and needs no RNG.
    params = SamplingParams(temperature=0.0, top_k=1, top_p=0.5)
    token = Sampler().sample(logits, params, None)
    assert token.item() == int(logits.argmax())
    assert token.dim() == 0
    assert token.dtype == torch.long
    assert token.device == logits.device


def test_greedy_leaves_a_supplied_generator_untouched():
    logits = torch.tensor([0.1, 3.0, -1.0])
    generator = torch.Generator().manual_seed(9)
    before = generator.get_state().clone()
    token = Sampler().sample(logits, SamplingParams(temperature=0.0), generator)
    assert token.item() == 1
    assert torch.equal(generator.get_state(), before)


def test_non_greedy_requires_a_generator():
    # None would silently draw from the process-wide default generator, so the
    # contract is an explicit error rather than an implicit fallback.
    logits = torch.tensor([0.1, 2.0, 1.0])
    with pytest.raises(ValueError, match="requires a generator"):
        Sampler().sample(logits, SamplingParams(temperature=1.0), None)


def test_non_greedy_advances_the_supplied_generator():
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, seed=42)
    generator = seeded_generator(params)
    before = generator.get_state().clone()
    Sampler().sample(logits, params, generator)
    assert not torch.equal(generator.get_state(), before)


GREEDY_CASES = [
    torch.tensor([0.1, 2.0, 1.0, -1.0]),
    torch.tensor([-3.0, -3.5, -2.0, -9.0]),
    torch.tensor([5.0, 4.999, 4.998]),
]


@pytest.mark.parametrize("logits", GREEDY_CASES)
def test_top_k_one_is_equivalent_to_greedy(logits):
    sampler = Sampler()
    greedy = sample_many(sampler, logits, SamplingParams(temperature=0.0))[0]
    # top_k=1 keeps the single highest logit by index, so the draw cannot depend
    # on the RNG stream. Checking several logits rows and seeds pins that
    # mechanism, not one coincidental argmax.
    for seed in (0, 7, 12345):
        params = SamplingParams(temperature=1.0, top_k=1, seed=seed)
        assert sample_many(sampler, logits, params)[0] == greedy


def test_top_k_keeps_exactly_k_candidates_when_the_threshold_is_tied():
    # [5, 4, 4, 1] with k=2 has three tokens at or above the k-th logit 4.0. A
    # threshold mask kept all three; selecting by index keeps exactly two.
    logits = torch.tensor([5.0, 4.0, 4.0, 1.0])
    params = SamplingParams(temperature=1.0, top_k=2, top_p=1.0)
    probs = Sampler.probabilities(logits, params)

    assert probs.shape == logits.shape
    # top_p=1 on finite logits that cannot underflow: the support is exactly the
    # candidate set, so counting non-zero entries is meaningful.
    candidates = probs.nonzero().flatten().tolist()
    assert len(candidates) == 2
    assert 0 in candidates  # the maximum always survives
    assert 3 not in candidates  # the minimum never does
    # Which of the two tied tokens survives is not guaranteed across devices, so
    # assert "exactly one of them" instead of a specific index.
    assert len({1, 2} & set(candidates)) == 1
    # The survivors carry their own logits, normalized into a distribution.
    assert probs.sum().item() == pytest.approx(1.0)
    tied = next(index for index in candidates if index != 0)
    retained = torch.softmax(torch.tensor([5.0, 4.0]), dim=-1)
    assert probs[0].item() == pytest.approx(retained[0].item())
    assert probs[tied].item() == pytest.approx(retained[1].item())


def test_top_k_one_on_all_tied_logits_keeps_one_candidate():
    # Every token shares the maximum, so the threshold mask kept all four at a
    # uniform 0.25 each. Strict top-k keeps a single one at weight 1.
    probs = Sampler.probabilities(
        torch.zeros(4), SamplingParams(temperature=1.0, top_k=1, top_p=1.0)
    )
    candidates = probs.nonzero().flatten()
    assert len(candidates) == 1
    assert probs[candidates[0]].item() == pytest.approx(1.0)


def test_probabilities_leaves_the_caller_row_unmodified():
    # top-k rebuilds the row with an in-place scatter, so pin that the buffer it
    # writes to is the one created inside probabilities(), not the caller's row
    # (a row view of the batch tensor).
    logits = torch.tensor([5.0, 4.0, 4.0, 1.0])
    before = logits.clone()
    Sampler.probabilities(logits, SamplingParams(temperature=1.0, top_k=2, top_p=1.0))
    torch.testing.assert_close(logits, before)


def test_top_k_at_or_above_the_vocabulary_keeps_every_token():
    logits = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.softmax(logits, dim=-1)
    for top_k in (3, 5):
        params = SamplingParams(temperature=1.0, top_k=top_k)
        torch.testing.assert_close(Sampler.probabilities(logits, params), expected)


def test_disabled_top_k_keeps_every_token():
    logits = torch.tensor([1.0, 2.0, 3.0])
    probs = Sampler.probabilities(logits, SamplingParams(temperature=1.0, top_k=-1))
    torch.testing.assert_close(probs, torch.softmax(logits, dim=-1))


def test_top_p_trims_the_strict_top_k_candidate_set():
    # top-k keeps 3 candidates and top-p then keeps the head of their sorted
    # distribution, so the final support is smaller than k.
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
    params = SamplingParams(temperature=1.0, top_k=3, top_p=0.8)
    probs = Sampler.probabilities(logits, params)
    assert probs.nonzero().flatten().tolist() == [0, 1]
    torch.testing.assert_close(probs[:2], torch.softmax(logits[:2], dim=-1))


def test_top_p_keeps_at_least_one_token():
    sampler = Sampler()
    # A top_p far below the head probability would empty the candidate set; the
    # mask shift keeps the single most likely token instead.
    logits = torch.linspace(1.0, 0.0, 100)
    params = SamplingParams(temperature=1.0, top_p=0.005, seed=3)
    assert set(sample_many(sampler, logits, params, count=20)) == {0}


def test_top_p_one_does_not_truncate_the_support():
    sampler = Sampler()
    logits = torch.tensor([1.0, 0.9, 0.8, 0.7])
    truncated = SamplingParams(temperature=1.0, top_p=0.5, seed=13)
    untruncated = SamplingParams(temperature=1.0, top_p=1.0, seed=13)
    assert set(sample_many(sampler, logits, truncated, count=200)) <= {0, 1}
    # Index 3 is the least likely token, so it is reachable only while the
    # support is intact. This asserts reachability over a fixed 200-draw stream
    # (p(miss) ~ 0.79**200), not the exact draw sequence: the exact sequence
    # would couple the test to torch's multinomial implementation.
    assert 3 in set(sample_many(sampler, logits, untruncated, count=200))


def sequence(sampler, logits, params, count=32):
    return sample_many(sampler, logits, params, count=count)


def test_same_seed_reproduces_the_same_sequence():
    sampler = Sampler()
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, top_p=0.9, seed=42)
    assert sequence(sampler, logits, params) == sequence(sampler, logits, params)


def test_different_seeds_produce_different_sequences():
    sampler = Sampler()
    logits = torch.linspace(1.0, -1.0, 64)
    base = dict(temperature=0.7, top_k=8, top_p=0.9)
    assert sequence(sampler, logits, SamplingParams(seed=1, **base)) != sequence(
        sampler, logits, SamplingParams(seed=2, **base)
    )


def test_generator_is_reused_and_advanced_across_calls():
    sampler = Sampler()
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, seed=42)
    generator = seeded_generator(params)
    state = generator.get_state().clone()
    sampler.sample(logits, params, generator)
    # Later calls must advance the same generator instead of reseeding it, which
    # is what makes a multi-token request reproducible.
    sampler.sample(logits, params, generator)
    assert not torch.equal(generator.get_state(), state)


def test_rng_registry_creates_lazily_and_reuses_the_slot():
    registry = RngRegistry()
    params = SamplingParams(temperature=1.0, seed=42)
    assert registry.peek("r") is None

    first = registry.acquire("r", params, torch.device("cpu"))
    assert registry.peek("r") is first
    # A later draw for the same request must advance the same generator.
    assert registry.acquire("r", params, torch.device("cpu")) is first


def test_rng_registry_creates_the_generator_on_the_requested_device(monkeypatch):
    # The registry is the single creation owner, so pin that it forwards the device
    # instead of falling back to the process default. The fake device never reaches
    # torch: the recorded constructor returns a real CPU generator for the caller.
    recorded = []
    real_generator = torch.Generator

    def recording_generator(device=None):
        recorded.append(device)
        return real_generator(device="cpu")

    monkeypatch.setattr(sampler_module.torch, "Generator", recording_generator)
    registry = RngRegistry()
    registry.acquire("r", SamplingParams(temperature=1.0, seed=1), torch.device("meta"))
    assert recorded == [torch.device("meta")]


def test_rng_registry_drop_removes_only_that_requests_slot():
    registry = RngRegistry()
    params = SamplingParams(temperature=1.0, seed=5)
    dropped = registry.acquire("dropped", params, torch.device("cpu"))
    kept = registry.acquire("kept", params, torch.device("cpu"))

    registry.drop("dropped")
    assert registry.peek("dropped") is None
    assert registry.peek("kept") is kept
    # Termination can run twice for the same request; the second drop is a no-op.
    registry.drop("dropped")
    assert registry.peek("kept") is kept
    assert dropped is not kept


def test_unresolved_seed_is_rejected_before_any_sampling():
    # The registry is the only generator creator, so it rejects an unresolved seed,
    # and without a generator no non-greedy token can be drawn. Engines resolve
    # seed=None at admission, greedy requests included (tests/test_rollout_outputs.py).
    params = SamplingParams(seed=None, temperature=1.0)
    registry = RngRegistry()
    with pytest.raises(ValueError, match="seed=None"):
        registry.acquire("r", params, torch.device("cpu"))
    assert registry.peek("r") is None
    with pytest.raises(ValueError, match="requires a generator"):
        Sampler().sample(torch.zeros(4), params, None)


LOGPROB_LOGITS = torch.tensor([1.5, 0.2, -0.7, 2.1, 0.0, -3.0])


@pytest.mark.parametrize("token", range(len(LOGPROB_LOGITS)), ids=lambda t: f"token={t}")
def test_raw_token_logprob_is_the_model_log_softmax_for_any_token(token):
    for dtype in (torch.float32, torch.bfloat16):
        logits = LOGPROB_LOGITS.to(dtype)
        params = SamplingParams(temperature=0.6, top_k=2, top_p=0.5, seed=1)
        value = token_logprob(logits, params, torch.tensor(token), "raw_logprobs")
        assert value.dim() == 0 and value.dtype == torch.float32
        expected = torch.log_softmax(logits.float(), -1)[token]
        assert torch.equal(value, expected)


PROCESSED_PARAMS = [
    SamplingParams(temperature=0.6, seed=1),
    SamplingParams(temperature=1.3, top_k=3, seed=1),
    SamplingParams(temperature=0.9, top_p=0.7, seed=1),
    SamplingParams(temperature=0.8, top_k=4, top_p=0.9, seed=1),
]
PROCESSED_IDS = ["temperature", "temperature-top_k", "temperature-top_p", "temperature-top_k-top_p"]


@pytest.mark.parametrize("params", PROCESSED_PARAMS, ids=PROCESSED_IDS)
def test_processed_token_logprob_scores_the_sampling_distribution(params):
    probs = Sampler.probabilities(LOGPROB_LOGITS, params)
    for token in range(len(LOGPROB_LOGITS)):
        value = token_logprob(LOGPROB_LOGITS, params, torch.tensor(token), "processed_logprobs")
        if probs[token] == 0:
            assert value.item() == -math.inf  # Filtered tokens cannot be sampled.
        else:
            assert value.item() == pytest.approx(probs[token].log().item(), abs=1e-6)


def test_processed_token_logprob_edge_cases():
    argmax = torch.tensor(int(LOGPROB_LOGITS.argmax()))
    greedy = SamplingParams(temperature=0.0, top_k=1)
    # Greedy requests report the unscaled model distribution, as vLLM does.
    assert torch.equal(
        token_logprob(LOGPROB_LOGITS, greedy, argmax, "processed_logprobs"),
        token_logprob(LOGPROB_LOGITS, greedy, argmax, "raw_logprobs"),
    )
    # Greedy has no temperature-scaled logits to score, so building them is an error.
    for build in (Sampler.processed_logits, Sampler.probabilities):
        with pytest.raises(ValueError, match="greedy"):
            build(LOGPROB_LOGITS, greedy)
    # One token survives top_k=1, or top_p=0.6 after top_k=2: top-p sees the renormalized
    # top-2 mass (argmax 0.65), not the full distribution (0.53, which would keep two).
    for single in (dict(temperature=0.7, top_k=1), dict(temperature=1.0, top_k=2, top_p=0.6)):
        params = SamplingParams(seed=1, **single)
        assert token_logprob(LOGPROB_LOGITS, params, argmax, "processed_logprobs").item() == 0.0


def test_processed_token_logprob_scores_the_strict_top_k_support():
    # A tie at the k-th logit: strict top-k keeps exactly k tokens, so the processed
    # logprob of the surviving tied token is log(e^4 / (e^5 + e^4)), and the other
    # tied token cannot be sampled.
    logits = torch.tensor([5.0, 4.0, 4.0, 1.0])
    params = SamplingParams(temperature=1.0, top_k=2, seed=1)
    probs = Sampler.probabilities(logits, params)
    kept = next(index for index in (1, 2) if probs[index] > 0)
    dropped = 3 - kept
    value = token_logprob(logits, params, torch.tensor(kept), "processed_logprobs")
    assert value.item() == pytest.approx(torch.log_softmax(torch.tensor([5.0, 4.0]), -1)[1].item())
    assert token_logprob(logits, params, torch.tensor(dropped), "processed_logprobs") == -math.inf


def test_token_logprob_consumes_no_rng():
    params = SamplingParams(temperature=0.8, top_k=4, top_p=0.9, seed=42)
    generator = seeded_generator(params)
    token = Sampler().sample(LOGPROB_LOGITS, params, generator)
    state, global_state = generator.get_state().clone(), torch.get_rng_state()
    for logprobs_mode in ("raw_logprobs", "processed_logprobs"):
        token_logprob(LOGPROB_LOGITS, params, token, logprobs_mode)
    assert torch.equal(generator.get_state(), state)
    assert torch.equal(torch.get_rng_state(), global_state)


@pytest.mark.parametrize(
    "params", [SamplingParams(top_k=1), *PROCESSED_PARAMS], ids=["greedy", *PROCESSED_IDS]
)
def test_sample_with_logprob_draws_like_sample_and_scores_the_drawn_logits(params):
    sampler, plain, scored = Sampler(), seeded_generator(params), seeded_generator(params)
    initial = plain.get_state().clone()
    for _ in range(4):
        token = sampler.sample(LOGPROB_LOGITS, params, plain)
        same, logprob = sampler.sample_with_logprob(LOGPROB_LOGITS, params, scored)
        assert torch.equal(same, token)
        expected = token_logprob(LOGPROB_LOGITS, params, token, "processed_logprobs")
        assert torch.equal(logprob, expected)
    # Same RNG consumption as sample(): none for greedy rows, which ignore the generator.
    assert torch.equal(plain.get_state(), scored.get_state())
    assert torch.equal(scored.get_state(), initial) == (params.temperature == 0)
    if params.temperature != 0:  # Like sample(), never a silent global-RNG fallback.
        with pytest.raises(ValueError, match="requires a generator"):
            sampler.sample_with_logprob(LOGPROB_LOGITS, params, None)


# One row per processed-logprob path: batched raw (greedy, even with top-k, and T == 1), batched
# per temperature, the sampler's top-k/top-p logits, and (last) a row without logprobs.
BATCH_PARAMS = [
    SamplingParams(temperature=0.0, top_k=3, logprobs=0),
    SamplingParams(temperature=1.0, logprobs=0, seed=1),
    SamplingParams(temperature=0.7, logprobs=0, seed=2),
    SamplingParams(temperature=1.3, logprobs=0, seed=3),
    SamplingParams(temperature=0.9, top_k=8, logprobs=0, seed=4),
    SamplingParams(temperature=1.0, top_p=0.8, logprobs=0, seed=5),
    SamplingParams(temperature=0.7, top_k=16, top_p=0.9, logprobs=0, seed=6),
    SamplingParams(temperature=0.8, seed=7),
]


ROWS = 48  # Rows laid out by a BATCH_LAYOUTS entry; a row without logprobs follows them.


def at(temperature):
    return SamplingParams(temperature=temperature, logprobs=0, seed=1)


# Layout -> (row -> params, log_softmax passes of processed_logprobs or None: only bounded).
BATCH_LAYOUTS = {
    "mixed": (lambda row: BATCH_PARAMS[row % 8], None),
    "t1": (lambda row: BATCH_PARAMS[1], 1),  # The only group: one pass over the whole batch.
    "t0.7": (lambda row: BATCH_PARAMS[2], 1),
    "top_k-top_p": (lambda row: BATCH_PARAMS[4 + row % 3], 0),
    # Many temperatures: one per row; rows r and ROWS - 1 - r sharing one (nested spans, the
    # outer ones scored row by row); two alternating (one pass per overlapping span).
    "distinct_t": (lambda row: at(0.5 + 0.01 * row), ROWS),
    "nested_t": (lambda row: at(0.5 + 0.01 * min(row, ROWS - 1 - row)), None),
    "alternating_t": (lambda row: at((0.7, 1.3)[row % 2]), 2),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", BATCH_LAYOUTS.values(), ids=BATCH_LAYOUTS.keys())
def test_batched_processed_logprobs_equal_per_row_token_logprob(monkeypatch, layout, dtype):
    params = [layout[0](row) for row in range(ROWS)] + [BATCH_PARAMS[-1]]
    logits = 4 * torch.randn(len(params), 256, generator=torch.Generator().manual_seed(0))
    logits, sampler, tokens, truncated = logits.to(dtype), Sampler(), [], {}
    for row, row_params in enumerate(params):  # As ModelRunner._sample_with_logprobs.
        # A fresh generator per row, as RngRegistry.acquire creates on a first draw.
        generator = None if row_params.temperature == 0 else seeded_generator(row_params)
        if row_params.logprobs is not None and truncates(row_params):
            token, truncated[row] = sampler.sample_with_logprob(logits[row], row_params, generator)
        else:
            token = sampler.sample(logits[row], row_params, generator)
        tokens.append(token)
    token_ids, passes = torch.stack(tokens), []  # passes: rows scored by each log_softmax.

    raw_logprobs = Sampler.raw_logprobs

    def counted(logits, token_ids):
        passes.append(len(logits))
        return raw_logprobs(logits, token_ids)

    monkeypatch.setattr(Sampler, "raw_logprobs", staticmethod(counted))
    values = processed_logprobs(logits, params, token_ids, truncated)
    monkeypatch.undo()
    # However many temperatures mix, at most twice the batched rows are scored (57d2308 scored
    # each requesting row once, in a pass of its own).
    assert sum(passes) <= 2 * (sum(p.logprobs is not None for p in params) - len(truncated))
    assert layout[1] is None or len(passes) == layout[1]
    assert values.shape == (len(params),) and values.dtype == torch.float32
    for row, row_params in enumerate(params):
        if row_params.logprobs is not None:
            expected = token_logprob(logits[row], row_params, token_ids[row], "processed_logprobs")
            assert torch.equal(values[row], expected)
