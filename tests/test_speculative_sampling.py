import pytest
import torch

from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.sampler import Sampler
from vllm_rlt.worker.speculative_sampling import batched_probabilities

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def host_args(top_k, top_p, vocab):
    """Derive the host-side arguments the way a caller would from per-row params."""
    effective = [k for k in top_k if 0 < k < vocab]
    return {
        "max_top_k": max(effective, default=0),
        "apply_top_p": any(p < 1 for p in top_p),
    }


def run(logits, temperature, top_k, top_p):
    device = logits.device
    vocab = logits.shape[-1]
    return batched_probabilities(
        logits,
        torch.tensor(temperature, dtype=torch.float32, device=device),
        torch.tensor(top_k, dtype=torch.int64, device=device),
        torch.tensor(top_p, dtype=torch.float32, device=device),
        **host_args(top_k, top_p, vocab),
    )


def reference(logits, temperature, top_k, top_p):
    rows = [
        Sampler.probabilities(logits[i], SamplingParams(temperature=t, top_k=k, top_p=p))
        for i, (t, k, p) in enumerate(zip(temperature, top_k, top_p, strict=True))
    ]
    return torch.stack(rows)


@pytest.mark.parametrize("device", DEVICES)
def test_matches_row_by_row_sampler(device):
    vocab = 17
    generator = torch.Generator().manual_seed(0)
    top_ks = [-1, 1, 3, vocab, vocab + 5]
    top_ps = [1.0, 0.9, 0.5]
    temperatures = [0.5, 0.8, 1.0, 1.5, 2.0]
    top_k = [top_ks[i % len(top_ks)] for i in range(30)]
    top_p = [top_ps[(i // len(top_ks)) % len(top_ps)] for i in range(30)]
    temperature = [temperatures[(i * 3) % len(temperatures)] for i in range(30)]
    logits = (torch.randn(30, vocab, generator=generator) * 3).to(device)

    actual = run(logits, temperature, top_k, top_p)

    assert actual.dtype == torch.float32
    expected = reference(logits, temperature, top_k, top_p)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.sum(-1), torch.ones(30, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_top_k_is_strict_with_ties(device):
    logits = torch.tensor([[5.0, 4.0, 4.0, 1.0]], device=device)

    probs = run(logits, [1.0], [2], [1.0])[0].cpu()

    assert (probs > 0).sum().item() == 2
    assert probs[0] > 0
    assert probs[3] == 0
    assert (probs[1] > 0) != (probs[2] > 0)
    kept = 1 if probs[1] > 0 else 2
    expected = torch.tensor([5.0, 4.0]).softmax(-1)
    torch.testing.assert_close(probs[[0, kept]], expected)
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


@pytest.mark.parametrize("device", DEVICES)
def test_top_k_all_tied_row_keeps_exactly_k(device):
    logits = torch.zeros(1, 6, device=device)

    probs = run(logits, [1.0], [2], [1.0])[0]

    assert (probs > 0).sum().item() == 2


@pytest.mark.parametrize("device", DEVICES)
def test_top_p_one_row_keeps_full_support_next_to_top_p_row(device):
    vocab = 64
    generator = torch.Generator().manual_seed(1)
    logits = (torch.randn(2, vocab, generator=generator) * 2).to(device)

    probs = run(logits, [1.0, 1.0], [-1, -1], [1.0, 0.5])

    assert (probs[0] > 0).all()
    assert (probs[1] > 0).sum() < vocab
    torch.testing.assert_close(probs[0], logits[0].softmax(-1), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_non_truncating_rows_keep_full_support_next_to_truncating_rows(device):
    vocab = 32
    generator = torch.Generator().manual_seed(2)
    logits = (torch.randn(3, vocab, generator=generator) * 2).to(device)

    probs = run(logits, [1.0, 1.0, 1.0], [-1, 3, vocab], [1.0, 1.0, 1.0])

    assert (probs[0] > 0).all()
    assert (probs[1] > 0).sum().item() == 3
    assert (probs[2] > 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_inputs_are_not_modified(device):
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(4, 11, generator=generator).to(device)
    temperature = torch.tensor([0.7, 1.0, 1.3, 2.0], device=device)
    top_k = torch.tensor([-1, 2, 5, 20], device=device)
    top_p = torch.tensor([1.0, 0.8, 0.6, 1.0], device=device)
    originals = [t.clone() for t in (logits, temperature, top_k, top_p)]

    batched_probabilities(logits, temperature, top_k, top_p, max_top_k=5, apply_top_p=True)

    for tensor, original in zip((logits, temperature, top_k, top_p), originals, strict=True):
        assert torch.equal(tensor, original)


@pytest.mark.parametrize("device", DEVICES)
def test_bf16_logits_return_float32_matching_float32_reference(device):
    generator = torch.Generator().manual_seed(4)
    logits = (torch.randn(5, 23, generator=generator) * 2).to(device=device, dtype=torch.bfloat16)
    temperature = [0.6, 1.0, 1.2, 0.9, 2.0]
    top_k = [-1, 4, 7, 2, -1]
    top_p = [1.0, 1.0, 0.7, 0.9, 0.5]

    actual = run(logits, temperature, top_k, top_p)

    assert actual.dtype == torch.float32
    expected = reference(logits.float(), temperature, top_k, top_p)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_host_argument_validation():
    logits = torch.zeros(2, 4)
    ones = torch.ones(2)
    top_k = torch.full((2,), -1, dtype=torch.int64)
    kwargs = {"max_top_k": 0, "apply_top_p": False}

    with pytest.raises(ValueError, match="max_top_k"):
        batched_probabilities(logits, ones, top_k, ones, max_top_k=-1, apply_top_p=False)
    with pytest.raises(ValueError, match="2-D"):
        batched_probabilities(logits[0], ones, top_k, ones, **kwargs)
    with pytest.raises(ValueError, match="temperature"):
        batched_probabilities(logits, ones[:1], top_k, ones, **kwargs)
    with pytest.raises(ValueError, match="top_k"):
        batched_probabilities(logits, ones, top_k[:1], ones, **kwargs)
    with pytest.raises(ValueError, match="top_p"):
        batched_probabilities(logits, ones, top_k, ones[:1], **kwargs)
