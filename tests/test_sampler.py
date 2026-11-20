"""Contract tests for the extracted Sampler: greedy, top-k and top-p.

``Sampler`` only consumes a generator the caller resolved, so these tests create
one themselves (mirroring what ``RngRegistry.acquire`` does in production) and
keep ``RngRegistry``'s own creation and lifetime contracts separate. The
distribution tests state the sampling contract itself, including strict top-k:
exactly ``k`` candidates survive by index, so a tie at the k-th logit does not
widen the support.

Every distributional test is parameterized by device. The CUDA variants run the
same arithmetic with a generator owned by that device, which is the contract the
CODA and speculative runners rely on: ``multinomial`` and its generator must
agree on the device. The CPU-only ``meta`` spy below keeps its minimal-boundary
meaning (the fake device never reaches torch); the real CUDA registry checks are
separate so a device spy cannot stand in for an actual device.
"""

import pytest
import torch

import vllm_rlt.worker.sampler as sampler_module
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.sampler import RngRegistry, Sampler

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def generator_for(params, device):
    """Caller-side generator creation, mirroring ``RngRegistry.acquire``."""
    return torch.Generator(device=device).manual_seed(params.seed)


def synchronize(device):
    """Wait for device work before comparing a CUDA generator's state."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def assert_generator_on(generator, device):
    """Assert the generator's device, tolerating an unspecified default index."""
    device = torch.device(device)
    assert generator.device.type == device.type
    if device.index is not None:
        assert generator.device.index == device.index


def sample_many(sampler, logits, params, count=1):
    """Sample ``count`` tokens with a generator on the logits device (greedy gets None)."""
    generator = None if params.temperature == 0 else generator_for(params, logits.device)
    return [int(sampler.sample(logits, params, generator)) for _ in range(count)]


@pytest.mark.parametrize("device", DEVICES)
def test_greedy_returns_argmax_without_a_generator(device):
    logits = torch.tensor([0.1, 2.0, 1.0, -1.0], device=device)
    # top_k/top_p are deliberately set: the temperature == 0 short circuit runs
    # first, so greedy ignores them and needs no RNG.
    params = SamplingParams(temperature=0.0, top_k=1, top_p=0.5)
    token = Sampler().sample(logits, params, None)
    assert token.item() == int(logits.argmax())
    assert token.dim() == 0
    assert token.dtype == torch.long
    assert token.device == logits.device


@pytest.mark.parametrize("device", DEVICES)
def test_greedy_leaves_a_supplied_generator_untouched(device):
    logits = torch.tensor([0.1, 3.0, -1.0], device=device)
    generator = torch.Generator(device=device).manual_seed(9)
    before = generator.get_state().clone()
    token = Sampler().sample(logits, SamplingParams(temperature=0.0), generator)
    assert token.item() == 1
    assert torch.equal(generator.get_state(), before)


@pytest.mark.parametrize("device", DEVICES)
def test_non_greedy_requires_a_generator(device):
    # None would silently draw from the process-wide default generator, so the
    # contract is an explicit error rather than an implicit fallback.
    logits = torch.tensor([0.1, 2.0, 1.0], device=device)
    with pytest.raises(ValueError, match="requires a generator"):
        Sampler().sample(logits, SamplingParams(temperature=1.0), None)


@pytest.mark.parametrize("device", DEVICES)
def test_non_greedy_advances_the_supplied_generator(device):
    logits = torch.linspace(1.0, -1.0, 64, device=device)
    params = SamplingParams(temperature=0.7, top_k=8, seed=42)
    generator = generator_for(params, device)
    before = generator.get_state().clone()
    Sampler().sample(logits, params, generator)
    synchronize(device)
    assert not torch.equal(generator.get_state(), before)


GREEDY_CASES = [
    torch.tensor([0.1, 2.0, 1.0, -1.0]),
    torch.tensor([-3.0, -3.5, -2.0, -9.0]),
    torch.tensor([5.0, 4.999, 4.998]),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("logits", GREEDY_CASES)
def test_top_k_one_is_equivalent_to_greedy(logits, device):
    logits = logits.to(device)
    sampler = Sampler()
    greedy = sample_many(sampler, logits, SamplingParams(temperature=0.0))[0]
    # top_k=1 keeps the single highest logit by index, so the draw cannot depend
    # on the RNG stream. Checking several logits rows and seeds pins that
    # mechanism, not one coincidental argmax.
    for seed in (0, 7, 12345):
        params = SamplingParams(temperature=1.0, top_k=1, seed=seed)
        assert sample_many(sampler, logits, params)[0] == greedy


@pytest.mark.parametrize("device", DEVICES)
def test_top_k_keeps_exactly_k_candidates_when_the_threshold_is_tied(device):
    # [5, 4, 4, 1] with k=2 has three tokens at or above the k-th logit 4.0. A
    # threshold mask kept all three; selecting by index keeps exactly two.
    logits = torch.tensor([5.0, 4.0, 4.0, 1.0], device=device)
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
    retained = torch.softmax(torch.tensor([5.0, 4.0], device=device), dim=-1)
    assert probs[0].item() == pytest.approx(retained[0].item())
    assert probs[tied].item() == pytest.approx(retained[1].item())


@pytest.mark.parametrize("device", DEVICES)
def test_top_k_one_on_all_tied_logits_keeps_one_candidate(device):
    # Every token shares the maximum, so the threshold mask kept all four at a
    # uniform 0.25 each. Strict top-k keeps a single one at weight 1.
    probs = Sampler.probabilities(
        torch.zeros(4, device=device), SamplingParams(temperature=1.0, top_k=1, top_p=1.0)
    )
    candidates = probs.nonzero().flatten()
    assert len(candidates) == 1
    assert probs[candidates[0]].item() == pytest.approx(1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_probabilities_leaves_the_caller_row_unmodified(device):
    # top-k rebuilds the row with an in-place scatter, so pin that the buffer it
    # writes to is the one created inside probabilities(), not the caller's row
    # (a row view of the batch tensor).
    rows = torch.tensor([[1.5, -1.0, 0.0, 2.0], [5.0, 4.0, 4.0, 1.0]], device=device)
    row = rows[1]
    before = rows.clone()
    probs = Sampler.probabilities(row, SamplingParams(temperature=1.0, top_k=2, top_p=1.0))
    assert probs.shape == row.shape
    torch.testing.assert_close(rows, before)


@pytest.mark.parametrize("device", DEVICES)
def test_top_k_at_or_above_the_vocabulary_keeps_every_token(device):
    logits = torch.tensor([1.0, 2.0, 3.0], device=device)
    expected = torch.softmax(logits, dim=-1)
    for top_k in (3, 5):
        params = SamplingParams(temperature=1.0, top_k=top_k)
        torch.testing.assert_close(Sampler.probabilities(logits, params), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_disabled_top_k_keeps_every_token(device):
    logits = torch.tensor([1.0, 2.0, 3.0], device=device)
    probs = Sampler.probabilities(logits, SamplingParams(temperature=1.0, top_k=-1))
    torch.testing.assert_close(probs, torch.softmax(logits, dim=-1))


@pytest.mark.parametrize("device", DEVICES)
def test_top_p_trims_the_strict_top_k_candidate_set(device):
    # top-k keeps 3 candidates and top-p then keeps the head of their sorted
    # distribution, so the final support is smaller than k.
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0], device=device)
    params = SamplingParams(temperature=1.0, top_k=3, top_p=0.8)
    probs = Sampler.probabilities(logits, params)
    assert probs.nonzero().flatten().tolist() == [0, 1]
    torch.testing.assert_close(probs[:2], torch.softmax(logits[:2], dim=-1))


@pytest.mark.parametrize("device", DEVICES)
def test_top_p_keeps_at_least_one_token(device):
    sampler = Sampler()
    # A top_p far below the head probability would empty the candidate set; the
    # mask shift keeps the single most likely token instead.
    logits = torch.linspace(1.0, 0.0, 100, device=device)
    params = SamplingParams(temperature=1.0, top_p=0.005, seed=3)
    assert set(sample_many(sampler, logits, params, count=20)) == {0}


@pytest.mark.parametrize("device", DEVICES)
def test_top_p_one_does_not_truncate_the_support(device):
    sampler = Sampler()
    logits = torch.tensor([1.0, 0.9, 0.8, 0.7], device=device)
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


@pytest.mark.parametrize("device", DEVICES)
def test_same_seed_reproduces_the_same_sequence(device):
    sampler = Sampler()
    logits = torch.linspace(1.0, -1.0, 64, device=device)
    params = SamplingParams(temperature=0.7, top_k=8, top_p=0.9, seed=42)
    assert sequence(sampler, logits, params) == sequence(sampler, logits, params)


@pytest.mark.parametrize("device", DEVICES)
def test_different_seeds_produce_different_sequences(device):
    sampler = Sampler()
    logits = torch.linspace(1.0, -1.0, 64, device=device)
    base = dict(temperature=0.7, top_k=8, top_p=0.9)
    assert sequence(sampler, logits, SamplingParams(seed=1, **base)) != sequence(
        sampler, logits, SamplingParams(seed=2, **base)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_generator_is_reused_and_advanced_across_calls(device):
    sampler = Sampler()
    logits = torch.linspace(1.0, -1.0, 64, device=device)
    params = SamplingParams(temperature=0.7, top_k=8, seed=42)
    generator = generator_for(params, device)
    state = generator.get_state().clone()
    sampler.sample(logits, params, generator)
    # Later calls must advance the same generator instead of reseeding it, which
    # is what makes a multi-token request reproducible.
    sampler.sample(logits, params, generator)
    synchronize(device)
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


@pytest.mark.gpu
def test_cuda_registry_creates_reuses_and_drops_real_device_generators():
    # The meta-device spy above proves the argument is forwarded; only a real CUDA
    # generator proves the registry can create the object production samples from.
    registry = RngRegistry()
    params = SamplingParams(temperature=0.8, top_k=8, seed=23)
    device = torch.device("cuda")

    generator = registry.acquire("r", params, device)
    assert_generator_on(generator, device)
    assert generator.initial_seed() == params.seed
    assert registry.peek("r") is generator
    # Reuse returns the same slot instead of minting a second stream.
    assert registry.acquire("r", params, device) is generator

    other = registry.acquire("other", params, device)
    assert other is not generator
    assert registry.peek("other") is other
    registry.drop("r")
    assert registry.peek("r") is None
    assert registry.peek("other") is other


@pytest.mark.gpu
def test_cuda_registry_generator_advances_and_samples_on_device():
    registry = RngRegistry()
    params = SamplingParams(temperature=0.8, top_k=8, top_p=0.9, seed=7)
    logits = torch.linspace(1.0, -1.0, 64, device="cuda")
    generator = registry.acquire("r", params, logits.device)
    assert_generator_on(generator, logits.device)
    before = generator.get_state().clone()

    token = Sampler().sample(logits, params, generator)
    synchronize(logits.device)
    # The token must stay a device scalar: the async CODA path feeds it to the
    # next prelude without a host readback in the submission region.
    assert token.device == logits.device
    assert token.dim() == 0
    assert token.dtype == torch.long
    assert 0 <= token.item() < logits.numel()
    first_state = generator.get_state().clone()
    assert not torch.equal(first_state, before)

    # A second draw on the same registry entry advances the same stream instead
    # of recreating or reseeding it.
    second = Sampler().sample(logits, params, registry.acquire("r", params, logits.device))
    synchronize(logits.device)
    assert registry.peek("r") is generator
    assert second.device == logits.device
    assert not torch.equal(generator.get_state(), first_state)
