import pytest
import torch

from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.sampler import Sampler
from vllm_rlt.worker.speculative_sampling import (
    batched_probabilities,
    categorical,
    speculative_accept,
)

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


# ---------------------------------------------------------------------------
# categorical
# ---------------------------------------------------------------------------

BELOW_ONE = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_midpoint_quadrature_matches_weights(device):
    m = 4096
    weights = torch.tensor([3.0, 0.0, 1.0, 4.0, 0.5, 0.0, 2.5], device=device)
    u = ((torch.arange(m, dtype=torch.float64) + 0.5) / m).float().to(device)

    index, valid = categorical(weights.expand(m, -1), u)

    assert valid.all()
    freq = torch.bincount(index, minlength=weights.numel()).double().cpu() / m
    expected = (weights / weights.sum()).double().cpu()
    assert (freq - expected).abs().max() <= 1 / m + 1e-9
    assert freq[weights.cpu() == 0].sum() == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weight", [1.0, 1 / 41])
def test_categorical_top_of_range_never_picks_trailing_zeros(device, weight):
    row = torch.tensor([weight] * 41 + [0.0, 0.0], dtype=torch.float32, device=device)
    mass32 = row.cumsum(-1)[-1].item()
    probes = [BELOW_ONE]
    for _ in range(3):
        probes.append(torch.nextafter(torch.tensor(probes[-1]), torch.tensor(0.0)).item())
    if mass32 < 1:  # the FP32 cumulative value is a legal uniform only when below 1
        probes.append(mass32)
    u = torch.tensor(probes, dtype=torch.float32, device=device)

    index, valid = categorical(row.expand(len(probes), -1), u)

    assert valid.all()
    assert (index < 41).all()
    assert index[0] == 40  # the largest u picks the last positive-weight token


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_out_of_contract_uniform_still_lands_on_positive_weight(device):
    # u * mass < mass for every u < 1, so the threshold clamp is only a safety net: a
    # uniform that reaches 1.0 must not index past the last positive-weight token.
    row = torch.tensor([1.0] * 41 + [0.0, 0.0], device=device)
    u = torch.tensor([1.0, 1.5], device=device)

    index, valid = categorical(row.expand(2, -1), u)

    assert valid.all()
    assert index.tolist() == [40, 40]


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_small_token_after_large_mass_has_its_exact_interval(device):
    row = torch.tensor([1.0, 1e-6], dtype=torch.float32, device=device)
    upper = torch.tensor(1 - 0.25e-6, dtype=torch.float32)  # inside (1 / (1 + 1e-6), 1)
    assert upper.item() > 1 / (1 + 1e-6)
    u = torch.tensor([0.5, upper.item(), 0.999998], dtype=torch.float32, device=device)

    index, valid = categorical(row.expand(3, -1), u)

    assert valid.all()
    assert index.tolist() == [0, 1, 0]  # 0.999998 < 1 / (1 + 1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_cdf_is_not_quantized_to_float32(device):
    # True boundary 1 / (1 + 1e-5) = 0.99999000010 > u, so the exact answer is token 0; an
    # FP32 cumulative sum rounds 1 + 1e-5 up and moves this u into token 1.
    row = torch.tensor([1.0, 1e-5], dtype=torch.float32, device=device)
    u = torch.tensor([0.9999899864196777, 0.99999499], dtype=torch.float32, device=device)

    index, _ = categorical(row.expand(2, -1), u)

    assert index.tolist() == [0, 1]


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_exact_cdf_boundaries_skip_zero_weight_tokens(device):
    row = torch.tensor([0.0, 1.0, 0.0, 3.0], device=device)  # cumulative [0, 1, 1, 4]
    u = torch.tensor([0.0, 0.25, 0.5], device=device)  # thresholds 0, 1, 2 (exact)

    index, valid = categorical(row.expand(3, -1), u)

    assert valid.all()
    assert index.tolist() == [1, 3, 3]


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_plateau_tail_is_never_chosen(device):
    row = torch.tensor([1.0, 1e-20], dtype=torch.float32, device=device)
    assert row.double().cumsum(-1)[-1] == 1.0  # the tail increment is rounded away in float64
    u = torch.tensor([0.0, 0.5, 0.999, BELOW_ONE], dtype=torch.float32, device=device)

    index, valid = categorical(row.expand(4, -1), u)

    assert valid.all()
    assert (index == 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_invalid_rows_return_index_zero_and_leave_valid_rows_alone(device):
    nan, inf = float("nan"), float("inf")
    weights = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, nan, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, inf, 1.0],
        ],
        device=device,
    )
    u = torch.full((5,), 0.7, device=device)

    index, valid = categorical(weights, u)

    assert valid.tolist() == [False, True, False, True, False]
    assert index.tolist() == [0, 1, 0, 2, 0]
    assert index.dtype == torch.int64


# ---------------------------------------------------------------------------
# speculative_accept
# ---------------------------------------------------------------------------


def midpoints(m, device):
    return ((torch.arange(m, dtype=torch.float64) + 0.5) / m).float().to(device)


def exact_first_position(p0, p1, q, m, device):
    """Enumerate every draft x, u_accept on an m-grid and u_final on an m-grid (d=1).

    Returns (first-token marginal, P(accepted == 1), bonus distribution given accepted == 1),
    each a float64 vector/scalar weighted by the proposal probability of the draft.
    """
    vocab = len(q)
    qn = [w / sum(q) for w in q]
    xs = [x for x in range(vocab) if qn[x] > 0]
    grid = midpoints(m, device)
    x_idx = torch.tensor(xs, device=device)
    n = len(xs) * m * m
    draft = x_idx.repeat_interleave(m * m)[:, None]
    ua = grid.repeat_interleave(m).repeat(len(xs))[:, None]
    uf = grid.repeat(m * len(xs))
    weight = torch.tensor(qn, dtype=torch.float64, device=device)[draft[:, 0]]
    p = torch.tensor([p0, p1], dtype=torch.float32, device=device).expand(n, 2, vocab)
    qq = torch.tensor(q, dtype=torch.float32, device=device).expand(n, 1, vocab)

    out = speculative_accept(draft, torch.ones(n, dtype=torch.int64, device=device), qq, p, ua, uf)

    assert not out.error.any()
    total = float(m * m)
    first = torch.bincount(out.tokens[:, 0], weights=weight, minlength=vocab) / total
    accepted = (out.accepted == 1).double()
    p_acc = (weight * accepted).sum() / total
    bonus = torch.bincount(out.tokens[:, 1], weights=weight * accepted, minlength=vocab)
    bonus = bonus / (weight * accepted).sum().clamp_min(1e-30)
    return first.cpu(), p_acc.item(), bonus.cpu(), (weight * accepted).sum().item() > 0


def normalized(w):
    return torch.tensor([x / sum(w) for x in w], dtype=torch.float64)


EXACT_CASES = {
    "generic": ([0.5, 4.0, 1.0, 0.0, 2.0], [3.0, 1.0, 0.0, 2.0, 6.0], [2.0, 0.0, 1.0, 3.0, 0.5]),
    "disjoint": ([0.0, 0.0, 1.0, 3.0, 2.0], [1.0, 1.0, 2.0, 0.0, 5.0], [1.0, 2.0, 0.0, 0.0, 0.0]),
    "equal": ([2.0, 0.0, 1.0, 3.0, 0.5], [4.0, 1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 1.0, 3.0, 0.5]),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(EXACT_CASES))
def test_single_draft_matches_target_by_exact_quadrature(device, case):
    m = 256
    p0, p1, q = EXACT_CASES[case]
    pn0, pn1, qn = normalized(p0), normalized(p1), normalized(q)

    first, p_acc, bonus, has_accepts = exact_first_position(p0, p1, q, m, device)

    assert (first - pn0).abs().max() <= 3 / m
    assert abs(p_acc - torch.minimum(pn0, qn).sum().item()) <= 3 / m
    if case == "disjoint":
        assert p_acc == 0
    if case == "equal":
        assert p_acc == pytest.approx(1.0, abs=1e-9)
    if has_accepts:
        assert (bonus - pn1).abs().max() <= 3 / m


@pytest.mark.parametrize("device", DEVICES)
def test_raw_weights_are_normalized_before_comparison(device):
    m = 256
    first, *_ = exact_first_position([3.0, 2.0], [1.0, 1.0], [1.0, 3.0], m, device)

    # Normalized: [0.6, 0.4]. Comparing raw weights would give [0.5, 0.5].
    assert (first - torch.tensor([0.6, 0.4], dtype=torch.float64)).abs().max() <= 3 / m


@pytest.mark.parametrize("device", DEVICES)
def test_ragged_multi_draft_monte_carlo_matches_target(device):
    vocab, w1, reps = 6, 3, 40_000
    counts = [0, 1, 2, 3, 3]
    generator = torch.Generator().manual_seed(7)

    def rand_weights(*shape):
        w = torch.rand(*shape, generator=generator) + 0.05
        w[torch.rand(*shape, generator=generator) < 0.2] = 0.0
        w[..., 0] += 0.5  # keep every row's mass positive
        return w

    q = rand_weights(len(counts), w1, vocab)
    p = rand_weights(len(counts), w1 + 1, vocab)
    drafts = torch.zeros(len(counts), reps, w1, dtype=torch.int64)
    for i, d in enumerate(counts):
        for j in range(d):
            drafts[i, :, j] = torch.multinomial(
                q[i, j], reps, replacement=True, generator=generator
            )
    q_in, p_in = q.clone(), p.clone()
    for i, d in enumerate(counts):  # poison everything past the real drafts
        q_in[i, d:] = float("nan")
        p_in[i, d + 1 :] = float("nan")
    n = len(counts) * reps

    def rep(t):
        return t.repeat_interleave(reps, dim=0).to(device)

    u_accept = torch.rand(n, w1, generator=generator).to(device)
    u_final = torch.rand(n, generator=generator).to(device)

    out = speculative_accept(
        drafts.reshape(n, w1).to(device),
        rep(torch.tensor(counts)),
        rep(q_in),
        rep(p_in),
        u_accept,
        u_final,
    )

    assert not out.error.any()
    tokens, accepted = out.tokens.cpu().view(len(counts), reps, w1 + 1), out.accepted.cpu()
    accepted = accepted.view(len(counts), reps)
    pn, qn = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)

    def assert_distribution(samples, target):
        freq = torch.bincount(samples, minlength=vocab).double() / samples.numel()
        se = (target.double() * (1 - target.double()) / samples.numel()).sqrt()
        assert ((freq - target.double()).abs() <= 5 * se + 1e-6).all()

    for i, d in enumerate(counts):
        assert (accepted[i] <= d).all()
        assert_distribution(tokens[i, :, 0], pn[i, 0])  # K = 0 rows: a plain draw from p[0]
        reach = 1.0
        for k in range(1, d + 1):
            reach *= torch.minimum(pn[i, k - 1], qn[i, k - 1]).sum().item()
            hit = (accepted[i] >= k).double().mean().item()
            assert abs(hit - reach) <= 5 * (reach * (1 - reach) / reps) ** 0.5 + 1e-6
        if d >= 1:  # the token after an accepted first draft follows the target at position 1
            assert_distribution(tokens[i, accepted[i] >= 1, 1], pn[i, 1])


def three_state(draft, d, q, p, device, *, u_accept=0.5, u_final=0.5, vocab=4):
    w1 = len(draft)
    result = speculative_accept(
        torch.tensor([draft], dtype=torch.int64, device=device),
        torch.tensor([d], dtype=torch.int64, device=device),
        torch.tensor([q], dtype=torch.float32, device=device).view(1, w1, vocab),
        torch.tensor([p], dtype=torch.float32, device=device).view(1, w1 + 1, vocab),
        torch.full((1, w1), u_accept, device=device),
        torch.full((1,), u_final, device=device),
    )
    return (
        result.tokens[0].tolist(),
        result.accepted.item(),
        result.final.item(),
        result.error.item(),
    )


NAN = float("nan")
E0, E1, E2, E3 = ([1.0 if i == j else 0.0 for i in range(4)] for j in range(4))
BAD = [NAN] * 4


@pytest.mark.parametrize("device", DEVICES)
def test_valid_reject_before_invalid_q_row_delivers_correction(device):
    # Position 0: q = e0, p = e2 -> certain reject, residual = e2. Position 2 q is NaN.
    tokens, accepted, final, error = three_state(
        [0, 1, 0], 3, [E0, E1, BAD], [E2, E1, E0, E0], device
    )

    assert (tokens, accepted, final, error) == ([2, 0, 0, 0], 0, 2, False)


@pytest.mark.parametrize("device", DEVICES)
def test_invalid_q_row_after_accepts_is_an_error(device):
    tokens, accepted, final, error = three_state(
        [0, 1, 0], 3, [E0, E1, BAD], [E0, E1, E0, E0], device
    )

    assert error and accepted == 2 and final == 0
    assert all(0 <= t < 4 for t in tokens)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bonus", [[0.0] * 4, BAD, [float("inf"), 0.0, 0.0, 0.0]])
def test_invalid_bonus_row_after_full_accept_is_an_error(device, bonus):
    tokens, accepted, final, error = three_state([0, 1], 2, [E0, E1], [E0, E1, bonus], device)

    assert error and accepted == 2 and final == 0
    assert tokens == [0, 1, 0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pad_token", [0, 3, 99, -1])
def test_nan_in_padding_positions_is_ignored(device, pad_token):
    tokens, accepted, final, error = three_state(
        [1, pad_token, pad_token], 1, [E1, BAD, BAD], [E1, E2, BAD, BAD], device
    )

    assert (tokens, accepted, final, error) == ([1, 2, 0, 0], 1, 2, False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tiny", [1e-38, 1e-30])
def test_equal_tiny_probabilities_are_accepted_for_any_uniform(device, tiny):
    row = [tiny, 1.0, 0.0, 0.0]  # u * tiny rounds back to tiny in FP32 when subnormal

    tokens, accepted, final, error = three_state(
        [0], 1, [row], [row, E3], device, u_accept=BELOW_ONE
    )

    assert (tokens, accepted, final, error) == ([0, 3], 1, 3, False)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_drafts_draw_from_first_target_row(device):
    n, vocab = 5, 4
    p = torch.zeros(n, 1, vocab, device=device)
    p[torch.arange(n), 0, torch.arange(n) % vocab] = 2.0
    result = speculative_accept(
        torch.zeros(n, 0, dtype=torch.int64, device=device),
        torch.zeros(n, dtype=torch.int64, device=device),
        torch.zeros(n, 0, vocab, device=device),
        p,
        torch.zeros(n, 0, device=device),
        torch.full((n,), 0.3, device=device),
    )

    assert result.accepted.tolist() == [0] * n
    assert result.final.tolist() == [i % vocab for i in range(n)]
    assert result.tokens.tolist() == [[i % vocab] for i in range(n)]
    assert not result.error.any()


@pytest.mark.parametrize("device", DEVICES)
def test_output_layout_error_rows_and_inputs_untouched(device):
    vocab, w1 = 4, 3
    # row 0: 2 accepts then a reject; row 1: all accepted (bonus e3); row 2: no drafts;
    # row 3: invalid q at the first position; row 4: invalid bonus row.
    q = torch.tensor(
        [
            [E0, E1, E0],
            [E2, E2, E2],
            [BAD, BAD, BAD],
            [BAD, E1, E1],
            [E0, E0, E0],
        ],
        device=device,
    )
    p = torch.tensor(
        [
            [E0, E1, E3, E3],
            [E2, E2, E2, E3],
            [E1, BAD, BAD, BAD],
            [E0, E1, E1, E1],
            [E0, E0, E0, [0.0] * 4],
        ],
        device=device,
    )
    draft_tokens = torch.tensor(
        [[0, 1, 0], [2, 2, 2], [0, 0, 0], [0, 1, 1], [0, 0, 0]], device=device
    )
    drafts = torch.tensor([3, 3, 0, 3, 3], device=device)
    u_accept = torch.full((5, w1), 0.5, device=device)
    u_final = torch.full((5,), 0.5, device=device)
    originals = [t.clone() for t in (draft_tokens, drafts, q, p, u_accept, u_final)]

    result = speculative_accept(draft_tokens, drafts, q, p, u_accept, u_final)

    assert result.error.tolist() == [False, False, False, True, True]
    assert result.accepted.tolist() == [2, 3, 0, 0, 3]
    assert result.final.tolist() == [3, 3, 1, 0, 0]
    assert result.tokens.tolist() == [
        [0, 1, 3, 0],  # drafts[:2], residual at position 2 (q = e0, p = e3), zeros after
        [2, 2, 2, 3],  # all drafts, then the bonus
        [1, 0, 0, 0],  # K = 0
        [0, 0, 0, 0],  # error row: legal placeholders only
        [0, 0, 0, 0],
    ]
    for out in (result.tokens, result.accepted, result.final):
        assert out.dtype == torch.int64
    for out in (result.tokens, result.final):
        assert ((out >= 0) & (out < vocab)).all()
    assert result.error.dtype == torch.bool
    for tensor, original in zip(
        (draft_tokens, drafts, q, p, u_accept, u_final), originals, strict=True
    ):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0, equal_nan=True)
