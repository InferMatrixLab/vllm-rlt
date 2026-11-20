"""CPU contracts for random sampling in async speculative rounds.

Covers the pure host plan, the pure tensor merge, and (with stubbed kernels, model and
coda) the runner's draft/verify/accept wiring that otherwise needs CUDA streams.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_rlt import SamplingParams, SpeculativeConfig
from vllm_rlt.core.scheduler import ScheduledRequest
from vllm_rlt.request import Request
from vllm_rlt.worker.async_speculative import (
    AsyncSpeculativeRunner,
    merge_random_round,
    plan_random_sampling,
)
from vllm_rlt.worker.sampler import RngRegistry
from vllm_rlt.worker.speculative_sampling import (
    AcceptResult,
    batched_probabilities,
    categorical,
    speculative_accept,
)

VOCAB = 100


def params(temperature=0.0, top_k=-1, top_p=1.0, seed=0):
    return SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)


def ints(array):
    return np.asarray(array).tolist()


def test_mixed_ragged_plan_has_exact_indices_and_parameters():
    plan = plan_random_sampling(
        [
            params(0.5, top_k=5, top_p=0.9),
            params(),
            params(1.0, top_k=VOCAB),
            params(2.0, top_k=7),
        ],
        [3, 1, 2, 3],
        VOCAB,
        4,
    )
    assert ints(plan.requests) == [0, 2, 3]
    assert ints(plan.temperature) == [0.5, 1.0, 2.0]
    assert ints(plan.top_k) == [5, VOCAB, 7]
    assert ints(plan.top_p) == [0.9, 1.0, 1.0]
    assert plan.max_top_k == 7 and plan.apply_top_p
    assert plan.width == 3
    # Offset 0 drafts items 0, 2, 3; offset 1 drafts items 0 and 3 (rows 0 and 1 of its block).
    assert [(ints(r), ints(k)) for r, k in plan.draft_rows] == [
        ([0, 1, 2], [0, 1, 2]),
        ([0, 1], [0, 2]),
    ]
    # Request starts are [0, 3, 4, 6]; greedy item 1 owns verify row 3.
    assert ints(plan.verify_rows) == [0, 1, 2, 4, 5, 6, 7, 8]
    assert ints(plan.verify_ranks) == [0, 0, 0, 1, 1, 2, 2, 2]
    assert ints(plan.verify_offsets) == [0, 1, 2, 0, 1, 0, 1, 2]
    assert ints(plan.verify_temperature) == [0.5] * 3 + [1.0] * 2 + [2.0] * 3
    assert ints(plan.verify_top_k) == [5] * 3 + [VOCAB] * 2 + [7] * 3
    assert ints(plan.verify_top_p) == [0.9] * 3 + [1.0] * 5


def test_greedy_rows_between_random_drafts_keep_block_positions():
    # Item 1 (greedy) drafts at offset 0 too, so random item 2 is row 1 of that block.
    plan = plan_random_sampling([params(), params(), params(1.0)], [2, 2, 2], VOCAB, 3)
    assert ints(plan.requests) == [2]
    assert [(ints(r), ints(k)) for r, k in plan.draft_rows] == [([2], [0])]
    assert ints(plan.verify_rows) == [4, 5]


def test_all_greedy_has_no_random_plan():
    assert plan_random_sampling([params(), params()], [3, 2], VOCAB, 2) is None


def test_random_requests_without_drafts():
    plan = plan_random_sampling([params(1.0), params(), params(0.7)], [1, 3, 1], VOCAB, 2)
    assert ints(plan.requests) == [0, 2] and plan.width == 1
    assert plan.draft_rows == ()
    assert ints(plan.verify_rows) == [0, 4]
    assert ints(plan.verify_ranks) == [0, 1] and ints(plan.verify_offsets) == [0, 0]
    names = [name for name, _ in plan.uploads()]
    assert not any(name.startswith(("r_rows", "r_ranks")) for name in names)


def test_truncation_flags_come_from_effective_parameters():
    plan = plan_random_sampling(
        [params(1.0), params(1.0, top_k=VOCAB), params(1.0, top_k=VOCAB + 50, top_p=1.0)],
        [1, 1, 1],
        VOCAB,
        1,
    )
    assert plan.max_top_k == 0 and not plan.apply_top_p
    plan = plan_random_sampling(
        [params(1.0, top_k=VOCAB - 1), params(1.0, top_p=0.5)], [1, 1], 100, 1
    )
    assert plan.max_top_k == VOCAB - 1 and plan.apply_top_p


def test_random_request_exceeding_configured_drafts_is_rejected():
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        plan_random_sampling([params(1.0)], [4], VOCAB, 2)
    # A greedy request is not bound to the uniform budget.
    assert plan_random_sampling([params(), params(1.0)], [4, 2], VOCAB, 2) is not None


def test_uploads_are_int64_and_floats_round_trip_through_bit_casts():
    plan = plan_random_sampling([params(0.3, top_k=4, top_p=0.7)], [2], VOCAB, 2)
    uploads = dict(plan.uploads())
    assert all(v.dtype == np.int64 for v in uploads.values())
    device = torch.from_numpy(np.concatenate(list(uploads.values())))
    names = list(uploads)
    start = sum(len(uploads[k]) for k in names[: names.index("r_temperature")])
    decoded = device[start : start + 1].view(torch.float64).float()
    assert decoded.tolist() == pytest.approx([0.3])
    assert decoded.dtype == torch.float32


def test_merge_replaces_random_rows_and_keeps_greedy_and_failed_slots():
    # Items: 0 random (3 rows), 1 greedy (2 rows), 2 random (2 rows), failed.
    target = torch.tensor([10, 11, 12, 20, 21, 30, 31])
    accepted = torch.tensor([0, 1, 0])  # greedy-computed values
    next_tokens = torch.tensor([10, 21, 30])
    starts = torch.tensor([5, 8, 12])
    inputs = torch.tensor([1, 2, 3])
    result = AcceptResult(
        # Request 0 accepts drafts 77 (not the argmax 11 of its row) and 66, then final 55.
        tokens=torch.tensor([[77, 66, 55], [9, 0, 0]]),
        accepted=torch.tensor([2, 0]),
        final=torch.tensor([55, 0]),
        error=torch.tensor([False, True]),
    )
    merged = merge_random_round(
        target,
        accepted,
        next_tokens,
        starts,
        inputs,
        result,
        torch.tensor([0, 2]),
        torch.tensor([0, 1, 2, 5, 6]),
        torch.tensor([0, 0, 0, 1, 1]),
        torch.tensor([0, 1, 2, 0, 1]),
    )
    assert merged.rows.tolist() == [77, 66, 55, 20, 21, 9, 0]
    assert target.tolist() == [10, 11, 12, 20, 21, 30, 31]  # inputs are not modified
    assert merged.accepted.tolist() == [2, 1, 0]
    assert merged.next_tokens.tolist() == [55, 21, 0]
    assert merged.next_positions.tolist() == [8, 10, 13]
    # The failed request keeps its old (input token, start); the others advance.
    assert merged.next_values.tolist() == [[55, 8], [21, 10], [3, 12]]
    assert merged.error.tolist() == [0, 0, 1]
    assert merged.error.dtype == torch.int64


# --- Runner wiring on CPU: stubbed kernels, model and CODA --------------------------------


class FakeKernel:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, grid):
        return self.function


def fake_gather(source, index, out, n, **_):
    out.copy_(source[index])


def fake_scatter(values, index, out, **_):
    out[index] = values


class Harness:
    """Drive one round of the real draft/verify/accept methods with a toy Markov model.

    Draft logits are ``E[token] @ W`` and verification logits ``(E[token] @ M) @ W``, so q and
    p differ and both depend on the drafted tokens.
    """

    K = 3
    HIDDEN = 8

    def __init__(self, monkeypatch, seed):
        from vllm_rlt.kernels import routing

        monkeypatch.setattr(routing, "gather_kernel", FakeKernel(fake_gather))
        monkeypatch.setattr(routing, "scatter_kernel", FakeKernel(fake_scatter))
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
        g = torch.Generator().manual_seed(seed)
        self.E = torch.randn(VOCAB, self.HIDDEN, generator=g)
        eye = torch.eye(self.HIDDEN)
        self.M = eye + 0.3 * torch.randn(self.HIDDEN, self.HIDDEN, generator=g)
        self.W = 0.5 * torch.randn(self.HIDDEN, VOCAB, generator=g)
        runner = object.__new__(AsyncSpeculativeRunner)
        runner.model = SimpleNamespace(config=SimpleNamespace(vocab_size=VOCAB))
        runner.config = SpeculativeConfig(
            num_speculative_tokens=self.K, draft_loops=1, target_loops=2
        )
        runner.device = torch.device("cpu")
        runner.boundary_kv = False
        runner.rng = RngRegistry()
        runner.slot_values = torch.zeros((8, 2), dtype=torch.long)
        runner.runtime = SimpleNamespace(
            prepare=lambda batch: batch,
            forward=lambda batch: SimpleNamespace(hidden=self.E[batch.tokens]),
        )
        runner.prelude_graphs = None
        runner._execute_recurrent_step = lambda hidden, positions, tables, depth, allocations: (
            hidden if depth < 1 else hidden @ self.M
        )
        runner._coda = lambda hidden, boundary=None: hidden @ self.W
        self.runner = runner

    def run(self, specs):
        """specs: (SamplingParams, token_count, input_token, position) per item."""
        runner = self.runner
        items = []
        for i, (sampling, count, token, position) in enumerate(specs):
            request = Request(f"r{i}", [1], sampling)
            items.append(ScheduledRequest(request, token_start=position, token_count=count))
            runner.slot_values[i] = torch.tensor([token, position])
        states = [SimpleNamespace(slot=i, submitted=0) for i in range(len(specs))]
        buffer = SimpleNamespace(keepalive=[], result=torch.full((64,), -1, dtype=torch.long))
        plan = runner._round_plan(items, states, buffer)
        u = None if plan.random is None else runner._uniforms(items, plan.random)
        tables = torch.zeros((len(specs), 1, 1))
        starts, tokens, hidden_blocks, q = runner._draft(plan, tables, None, u)
        target, p = runner._verify(plan, starts, hidden_blocks, tables, None)
        count = runner._accept_and_update_state(
            plan, states, starts, tokens, target, buffer, u=u, q=q, p=p
        )
        return SimpleNamespace(
            plan=plan, u=u, tokens=tokens, target=target, count=count, buffer=buffer, q=q, p=p
        )

    def q_row(self, token, sampling):
        return self.row(self.E[token] @ self.W, sampling)

    def p_row(self, token, sampling):
        return self.row(self.E[token] @ self.M @ self.W, sampling)

    @staticmethod
    def row(logits, sampling):
        k = sampling.top_k
        return batched_probabilities(
            logits[None],
            torch.tensor([sampling.temperature]),
            torch.tensor([k]),
            torch.tensor([sampling.top_p]),
            max_top_k=k if 0 < k < VOCAB else 0,
            apply_top_p=sampling.top_p < 1,
        )[0]


SPECS = [
    # (sampling, token_count, input token, position)
    (params(0.8, top_k=20, top_p=0.95, seed=1), 4, 3, 10),
    (params(), 3, 4, 20),
    (params(1.3, seed=2), 2, 5, 30),
    (params(0.6, top_p=0.9, seed=3), 1, 6, 40),
    (params(1.0, top_k=5, seed=4), 4, 7, 50),
]


@pytest.mark.parametrize("seed", range(8))
def test_runner_round_matches_per_request_reference(monkeypatch, seed):
    h = Harness(monkeypatch, seed)
    out = h.run(SPECS)
    n, K = len(SPECS), h.K
    result = out.buffer.result
    accepted_all = result[out.count : out.count + n].tolist()
    positions = result[out.count + n : out.count + 2 * n].tolist()
    flags = result[out.count + 2 * n : out.count + 3 * n].tolist()
    assert flags == [0] * n
    slot = h.runner.slot_values
    ranks = {i: r for r, i in enumerate(i for i, spec in enumerate(SPECS) if spec[0].temperature)}
    start = 0
    for i, (sampling, count, token, position) in enumerate(SPECS):
        drafts = count - 1
        row_tokens = out.tokens[i]
        assert row_tokens[0].item() == token
        if sampling.temperature == 0:
            # Greedy: target argmax rows, accepted while the draft equals the argmax.
            argmax = [(h.E[row_tokens[j]] @ h.M @ h.W).argmax().item() for j in range(count)]
            assert out.target[start : start + count].tolist() == argmax
            expected = 0
            while expected < drafts and row_tokens[expected + 1].item() == argmax[expected]:
                expected += 1
            committed = argmax[: expected + 1]
        else:
            gen = torch.Generator().manual_seed(sampling.seed)
            u = torch.rand((2 * K + 1,), generator=gen)
            assert torch.equal(out.u[ranks[i]], u)
            q = (
                torch.stack([h.q_row(row_tokens[j], sampling) for j in range(drafts)])
                if drafts
                else None
            )
            for j in range(drafts):
                drawn, _ = categorical(q[j][None], u[j : j + 1])
                assert drawn.item() == row_tokens[j + 1].item()
            p = torch.stack([h.p_row(row_tokens[j], sampling) for j in range(count)])
            res = speculative_accept(
                row_tokens[None, 1:count],
                torch.tensor([drafts]),
                q[None] if drafts else p.new_empty((1, 0, VOCAB)),
                p[None],
                u[None, K : K + drafts],
                u[2 * K : 2 * K + 1],
            )
            assert not res.error.item()
            expected = res.accepted.item()
            committed = res.tokens[0, : expected + 1].tolist()
        assert accepted_all[i] == expected
        assert result[start : start + expected + 1].tolist() == committed
        assert positions[i] == position + expected + 1
        assert slot[i].tolist() == [committed[-1], position + expected + 1]
        start += count
    assert out.q.shape == (4, 3, VOCAB) and out.p.shape == (4, 4, VOCAB)


def test_wiring_exercises_non_argmax_acceptance_and_rejection(monkeypatch):
    seen = {"accepted_non_argmax": 0, "rejected": 0, "accepted": 0}
    for seed in range(8):
        h = Harness(monkeypatch, seed)
        out = h.run(SPECS)
        n = len(SPECS)
        accepted = out.buffer.result[out.count : out.count + n].tolist()
        start = 0
        for i, (sampling, count, _, _) in enumerate(SPECS):
            if sampling.temperature:
                committed = out.buffer.result[start : start + accepted[i]]
                argmax = out.target[start : start + accepted[i]]
                seen["accepted"] += accepted[i]
                seen["rejected"] += accepted[i] < count - 1
                seen["accepted_non_argmax"] += int((committed != argmax).sum())
            start += count
    assert all(seen.values()), seen


def test_all_greedy_round_builds_no_random_state_and_keeps_greedy_layout(monkeypatch):
    h = Harness(monkeypatch, 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("greedy rounds must not touch the RNG registry")

    h.runner.rng.acquire = forbidden
    out = h.run([(params(), 3, 4, 20), (params(), 2, 5, 30), (params(), 1, 6, 40)])
    assert out.plan.random is None and out.u is None and out.q is None and out.p is None
    assert not any(name.startswith(("r_", "v_")) for name in vars(out.plan))
    n, count = 3, out.count
    assert count == 6
    result = out.buffer.result
    assert result[:count].tolist() == out.target.tolist()
    assert result[count + 2 * n].item() == -1  # no flag region is written
    assert not h.runner.rng.generators


def test_random_draw_consumes_one_fixed_block_per_round_and_only_for_random(monkeypatch):
    h = Harness(monkeypatch, 0)
    acquired = []
    original = h.runner.rng.acquire
    h.runner.rng.acquire = lambda rid, sp, device: (
        acquired.append(rid),
        original(rid, sp, device),
    )[1]
    h.run(SPECS)
    assert acquired == ["r0", "r2", "r3", "r4"]
    # The generators advanced by exactly one (2K + 1)-draw regardless of draft counts.
    for rid, seed in (("r0", 1), ("r3", 3)):
        reference = torch.Generator().manual_seed(seed)
        torch.rand((2 * h.K + 1,), generator=reference)
        expected = torch.rand((1,), generator=reference)
        assert torch.rand((1,), generator=h.runner.rng.generators[rid]).equal(expected)


def test_numerical_failure_keeps_old_slot_and_flags_only_that_request(monkeypatch):
    h = Harness(monkeypatch, 3)
    h.E[7] = float("nan")  # any row fed token 7 yields NaN logits
    specs = [
        (params(1.0, seed=1), 3, 7, 10),  # invalid q at offset 0 -> failed
        (params(1.0, seed=2), 3, 4, 20),
        (params(), 2, 5, 30),
    ]
    out = h.run(specs)
    n, count = 3, out.count
    result = out.buffer.result
    assert result[count + 2 * n : count + 3 * n].tolist() == [1, 0, 0]
    assert h.runner.slot_values[0].tolist() == [7, 10]  # not advanced
    assert h.runner.slot_values[1, 1].item() > 20 and h.runner.slot_values[2, 1].item() > 30
