"""CUDA integration contracts for random sampling in async block speculation (M7 T3').

Every test needs a reserved device (``--run-gpu``): the async speculative runner allocates CUDA
streams and events. The tests drive the engine/runner directly (not the public entrypoint) and
observe the runner through thin pass-through wrappers (``RoundRecorder``, ``NanInjector``,
``spy_tickets``) that never change behaviour, except ``NanInjector`` which deliberately corrupts
one draft distribution.

Reproducibility scope (plan section 3.2): outputs are only compared on the *same* execution path
(same engine configuration, same dtype, same graph mode) and, where noted, with identical request
shapes. Graph and eager outputs are never compared token by token.
"""

from contextlib import closing
from dataclasses import dataclass

import pytest
import torch

import vllm_rlt.worker.async_speculative as async_speculative
from tests.helpers import make_speculative_engine, tiny_ouro_model
from tests.test_async_speculative import drain, make_engine
from vllm_rlt import SamplingParams

PROMPT = (2, 3, 4)
K = 3  # make_engine's default num_speculative_tokens
TEMPERATURE = 1.5
DTYPES = [torch.float32, torch.bfloat16]


# --------------------------------------------------------------------------- helpers


def random_params(seed, max_tokens, temperature=TEMPERATURE, **kwargs):
    return SamplingParams(
        max_tokens=max_tokens, temperature=temperature, seed=seed, ignore_eos=True, **kwargs
    )


def greedy_params(max_tokens):
    return SamplingParams(max_tokens=max_tokens, ignore_eos=True)


def bootstrap_random(engine, rid, params, prompt=PROMPT):
    """Run steps until the bootstrap (ordinary CODA) token of ``rid`` has been delivered.

    Same loop as ``tests.helpers.bootstrap_speculation``, which only builds greedy requests.
    """
    engine.add_request(rid, list(prompt), params)
    for _ in range(30):
        if engine.step():
            request = engine.scheduler.requests[rid]
            assert len(request.generated_token_ids) == 1
            return request
    pytest.fail("bootstrap did not produce a token")


def assert_reclaimed(engine):
    """No request, state, leased buffer, slot, block or generator survives a drained engine."""
    runner = engine.speculative_runner
    assert not engine.scheduler.requests
    assert not engine._pending_speculative and not engine._retiring_speculative
    assert engine.cache_manager.num_used_blocks == 0
    assert not runner.states
    assert all(not buffer.leased for buffer in runner.buffers)
    assert len(runner.free_slots) == len(runner.slot_values)
    assert not engine.rng.generators


@dataclass
class RoundRecord:
    """One submitted block round: host item list plus device tensors cloned at accept time."""

    items: list  # [(request_id, token_count)] in item order
    tokens: torch.Tensor  # [n, width] input token then drafts (padding zeros)
    target: torch.Tensor  # [count] target argmax per request-major verify row
    random: bool  # the round had a random plan

    def view(self, request_id):
        """Return (input token + drafts, target argmax rows) of one request in this round."""
        tokens = self.tokens.cpu()
        target = self.target.cpu().tolist()
        start = 0
        for row, (rid, count) in enumerate(self.items):
            if rid == request_id:
                return tokens[row, :count].tolist(), target[start : start + count]
            start += count
        raise KeyError(request_id)


class RoundRecorder:
    """Record every submitted block round without altering it.

    Wraps ``_round_plan`` (host only) and ``_accept_and_update_state`` (clones its device inputs
    on the compute stream; no host read happens during submission). Read ``rounds`` only after
    ``runner.synchronize()``.
    """

    def __init__(self, monkeypatch, runner):
        self.rounds = []
        self._items = None
        round_plan, accept = runner._round_plan, runner._accept_and_update_state

        def recording_round_plan(items, states, buffer):
            self._items = [(item.request.request_id, item.token_count) for item in items]
            return round_plan(items, states, buffer)

        def recording_accept(plan, states, starts, tokens, target, buffer, **kwargs):
            self.rounds.append(
                RoundRecord(
                    self._items,
                    tokens.detach().clone(),
                    target.detach().clone(),
                    plan.random is not None,
                )
            )
            return accept(plan, states, starts, tokens, target, buffer, **kwargs)

        monkeypatch.setattr(runner, "_round_plan", recording_round_plan)
        monkeypatch.setattr(runner, "_accept_and_update_state", recording_accept)

    def counts(self):
        """Per round: {request_id: token_count}."""
        return [dict(record.items) for record in self.rounds]


class NanInjector:
    """Make the first draft distribution of one speculative round NaN (offset 0 of the draft).

    ``_draft`` computes the draft distribution of every offset with the module-level
    ``batched_probabilities``; the verify distribution uses it too, so calls are attributed to
    the draft only while a (wrapped) ``_draft`` is executing. Offset 0 always drafts when a round
    has more than one row, and an invalid ``q`` at the first draft position is a deterministic
    stop-point error (plan 3.3/3.4). ``round_index`` counts ``_draft`` calls from 1.
    """

    def __init__(self, monkeypatch, runner, round_index):
        self.round_index = round_index
        self.rounds = 0
        self.fired = 0
        self._in_draft = False
        self._draft_calls = 0
        draft, probabilities = runner._draft, async_speculative.batched_probabilities

        def wrapped_draft(*args, **kwargs):
            self.rounds += 1
            self._in_draft, self._draft_calls = True, 0
            try:
                return draft(*args, **kwargs)
            finally:
                self._in_draft = False

        def wrapped_probabilities(*args, **kwargs):
            result = probabilities(*args, **kwargs)
            if self._in_draft:
                self._draft_calls += 1
                if self.rounds == self.round_index and self._draft_calls == 1:
                    self.fired += 1
                    return torch.full_like(result, float("nan"))
            return result

        monkeypatch.setattr(runner, "_draft", wrapped_draft)
        monkeypatch.setattr(async_speculative, "batched_probabilities", wrapped_probabilities)


def spy_tickets(monkeypatch, runner):
    """Collect every speculative ticket the runner returns."""
    tickets = []
    submit = runner.submit

    def spying_submit(batch):
        ticket = submit(batch)
        tickets.append(ticket)
        return ticket

    monkeypatch.setattr(runner, "submit", spying_submit)
    return tickets


def spy_acquire(monkeypatch, rng):
    """Record (request_id, generator) for every ``RngRegistry.acquire`` and pass it through."""
    calls = []
    acquire = rng.acquire

    def spying_acquire(request_id, params, device):
        generator = acquire(request_id, params, device)
        calls.append((request_id, generator))
        return generator

    monkeypatch.setattr(rng, "acquire", spying_acquire)
    return calls


def accepted_non_argmax(record, request_id, result):
    """Check the CPU result against the recorded GPU round; count accepted non-argmax drafts."""
    row, target = record.view(request_id)
    drafts = row[1:]
    accepted = result.accepted_count
    assert not result.numerical_error
    assert 0 <= accepted <= len(drafts)
    # Section 3.1: the committed prefix is the accepted drafts, then one correction/bonus token.
    assert len(result.token_ids) == accepted + 1
    assert result.token_ids[:accepted] == drafts[:accepted]
    return sum(1 for j in range(accepted) if drafts[j] != target[j])


def assert_kv_matches_serial_replay(engine, model, oracle, rid, tokens, dtype):
    """Per-depth KV of every position in ``tokens`` equals a serial replay (copied oracle)."""
    assert oracle.allocate("serial", len(tokens))
    try:
        with torch.inference_mode():
            for position, token in enumerate(tokens):
                hidden = model.prelude(torch.tensor([token], device="cuda"))
                for depth in range(model.config.total_ut_steps):
                    hidden, _ = model.recurrent(hidden, ["serial"], [depth], [position], oracle)
        allocation = engine.cache_manager._get_allocation(rid)
        pos = torch.arange(len(tokens), device="cuda")
        atol, rtol = (4e-5, 4e-5) if dtype == torch.float32 else (0.08, 0.04)
        for depth in range(model.config.total_ut_steps):
            pages = torch.tensor(allocation.block_tables[depth], device="cuda")
            blocks, offsets = pages[pos // 2], pos % 2
            for layer in range(model.config.num_hidden_layers):
                expected_k, expected_v = oracle.read(layer, "serial", depth, len(tokens))
                torch.testing.assert_close(
                    engine.cache_manager.key_cache[blocks, layer, offsets],
                    expected_k,
                    atol=atol,
                    rtol=rtol,
                )
                torch.testing.assert_close(
                    engine.cache_manager.value_cache[blocks, layer, offsets],
                    expected_v,
                    atol=atol,
                    rtol=rtol,
                )
    finally:
        oracle.free("serial")


# ----------------------------------------------------------------- 1. key regression (3.1)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("graphs", [False, True])
def test_accepted_non_argmax_drafts_keep_cpu_sequence_gpu_input_and_kv_consistent(
    monkeypatch, dtype, graphs
):
    """Plan 3.1: a random round commits accepted drafts + correction/bonus, not target argmax.

    Two rounds are in flight. Per round the recorded GPU draft tokens must equal the CPU-committed
    prefix, the second round's GPU input token must be the first round's last committed token, the
    slot table must hold the final (token, position), and per-depth KV of every committed
    position must match a serial replay of the CPU-committed tokens. Over all seeds at least one
    accepted draft must differ from the target argmax, so the test cannot pass vacuously.
    Several seeds share one engine (abort + drain between them) to make that overwhelmingly likely.
    """
    model = tiny_ouro_model("cuda", dtype)
    oracle = make_speculative_engine(model, asynchronous=False).cache_manager
    non_argmax = 0
    with closing(make_engine(model, graphs=graphs)) as e:
        runner = e.speculative_runner
        recorder = RoundRecorder(monkeypatch, runner)
        for seed in (31, 32, 33):
            rid = f"r{seed}"
            request = bootstrap_random(e, rid, random_params(seed, 24, temperature=2.0))
            start = request.position
            first = len(recorder.rounds)
            e.step()
            e.step()
            assert len(e._pending_speculative) == 2
            assert runner.submitted_before_collect >= 1
            runner.synchronize()
            records = recorder.rounds[first:]
            assert len(records) == 2 and all(r.random for r in records)
            assert [r.items for r in records] == [[(rid, count)] for count in (K + 1, K + 1)]
            results = [ticket.collect().speculative[0] for ticket in e._pending_speculative]
            for record, result in zip(records, results):
                non_argmax += accepted_non_argmax(record, rid, result)
            committed = [token for result in results for token in result.token_ids]
            # The next round's GPU input token is the previous round's last committed token.
            assert records[1].view(rid)[0][0] == results[0].token_ids[-1]
            tokens = request.prompt_token_ids + request.generated_token_ids + committed
            assert len(request.generated_token_ids) == 1
            # Slot table: (last committed token, its position); that token has no KV yet.
            state = runner.states[rid]
            assert int(state.token) == tokens[-1]
            assert int(state.position) == start + sum(len(r.token_ids) for r in results)
            assert int(state.position) == len(tokens) - 1
            assert_kv_matches_serial_replay(e, model, oracle, rid, tokens[:-1], dtype)
            e.abort_request(rid)
            drain(e)
            assert e.cache_manager.num_used_blocks == 0
        assert_reclaimed(e)
    assert non_argmax > 0, "no accepted draft differed from the target argmax: vacuous test"


# --------------------------------------------------------------- 2. two rounds in flight


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("graphs", [False, True])
def test_two_inflight_random_rounds_finish_with_exact_lengths_and_reclaim(dtype, graphs):
    with closing(make_engine(tiny_ouro_model("cuda", dtype), graphs=graphs)) as e:
        runner = e.speculative_runner
        bootstrap_random(e, "r", random_params(1, 20))
        e.step()
        e.step()
        assert len(e._pending_speculative) == 2
        assert all(ticket.error_flags for ticket in e._pending_speculative)
        assert runner.submitted_before_collect == 1
        # A second random request (with top-k/top-p truncation) joins while two rounds fly.
        e.add_request("s", list(PROMPT), random_params(2, 11, top_k=16, top_p=0.9))
        outputs = drain(e)
        assert runner.submitted_before_collect > 0
        assert {rid: len(o.token_ids) for rid, o in outputs.items()} == {"r": 20, "s": 11}
        assert all(o.finished and o.finish_reason == "length" for o in outputs.values())
        vocab = e.model.config.vocab_size
        assert all(0 <= t < vocab for o in outputs.values() for t in o.token_ids)
        assert_reclaimed(e)


# ------------------------------------------------------------------------ 3. mixed batch


@pytest.mark.gpu
def test_greedy_request_in_a_mixed_batch_is_unaffected_and_never_acquires_a_generator(
    monkeypatch,
):
    """A greedy request next to a random one matches its solo run and never touches the RNG.

    FP32 only. The solo run and the mixed runs have different batch shapes, so equality relies on
    FP32 row-wise numerics being far from any argmax tie on this model; the second mixed run
    (different seed for the random neighbour, same shapes up to acceptance) is the weaker check.
    """
    model = tiny_ouro_model("cuda", torch.float32)

    def run(neighbour_seed):
        with closing(make_engine(model)) as e:
            recorder = RoundRecorder(monkeypatch, e.speculative_runner)
            acquired = spy_acquire(monkeypatch, e.rng)
            e.add_request("g", list(PROMPT), greedy_params(16))
            if neighbour_seed is not None:
                e.add_request("x", list(PROMPT), random_params(neighbour_seed, 16))
            outputs = drain(e)
            assert outputs["g"].finish_reason == "length" and len(outputs["g"].token_ids) == 16
            ids = [rid for rid, _ in acquired]
            assert "g" not in ids
            if neighbour_seed is None:
                assert not ids and not any(r.random for r in recorder.rounds)
            else:
                assert "x" in ids and len(outputs["x"].token_ids) == 16
                # Greedy and random requests really shared speculative batches.
                assert any(
                    r.random and {rid for rid, _ in r.items} == {"g", "x"} for r in recorder.rounds
                )
            assert_reclaimed(e)
            return {rid: o.token_ids for rid, o in outputs.items()}

    alone = run(None)
    mixed = run(3)
    other = run(4)
    assert mixed["g"] == alone["g"]
    assert other["g"] == alone["g"]
    assert mixed["x"] != other["x"]  # the neighbour really changed


# ------------------------------------------------------------------- 4. RNG lifecycle


@pytest.mark.gpu
def test_speculative_rounds_consume_the_bootstrap_generator_and_reused_ids_start_fresh(
    monkeypatch,
):
    with closing(make_engine(tiny_ouro_model("cuda"), k=K)) as e:
        bootstrap_random(e, "r", random_params(21, 24))
        generator = e.rng.peek("r")
        assert generator is not None  # the bootstrap CODA created it
        initial = generator.get_state()
        acquired = spy_acquire(monkeypatch, e.rng)
        # A probe replays exactly one rand((2K + 1,)) per round from the same state.
        probe = torch.Generator(device=generator.device)
        probe.set_state(initial)
        previous = initial
        for submitted in (1, 2):
            e.step()
            assert len(acquired) == submitted
            assert acquired[-1][0] == "r" and acquired[-1][1] is generator
            torch.rand((2 * K + 1,), generator=probe, device=probe.device, dtype=torch.float32)
            state = generator.get_state()
            assert not torch.equal(state, previous), "generator did not advance"
            assert torch.equal(state, probe.get_state()), "not exactly one draw per round"
            previous = state
        assert e.rng.peek("r") is generator
        torch.cuda.synchronize()

        # Abort drops the entry once the request retires.
        assert e.abort_request("r").finish_reason == "abort"
        drain(e)
        assert e.rng.peek("r") is None and not e.rng.generators
        assert_reclaimed(e)

        # A reused ID gets a new generator that starts from the same seed and bootstrap draws.
        bootstrap_random(e, "r", random_params(21, 24))
        fresh = e.rng.peek("r")
        assert fresh is not None and fresh is not generator
        assert torch.equal(fresh.get_state(), initial)
        outputs = drain(e)
        assert len(outputs["r"].token_ids) == 24
        # Natural completion drops it as well.
        assert e.rng.peek("r") is None and not e.rng.generators
        assert_reclaimed(e)


# ------------------------------------------------------------ 5. same-path fixed-seed replay


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_fixed_seed_single_request_replays_and_a_new_seed_changes_the_output(graphs):
    """Single request only, so the batch plan is fixed (reproducibility scope, plan 3.2)."""
    model = tiny_ouro_model("cuda", torch.float32)

    def generate(engine, seed):
        engine.add_request("r", list(PROMPT), random_params(seed, 32))
        output = drain(engine)["r"]
        assert len(output.token_ids) == 32 and output.finish_reason == "length"
        return output.token_ids

    with closing(make_engine(model, graphs=graphs)) as first:
        expected = generate(first, 7)
        assert_reclaimed(first)
    with closing(make_engine(model, graphs=graphs)) as second:
        assert generate(second, 7) == expected
        assert generate(second, 8) != expected
        assert_reclaimed(second)


# ----------------------------------------------------------------- 6. request isolation


@pytest.mark.gpu
def test_changing_one_requests_seed_leaves_its_neighbour_unchanged():
    """Request A is unchanged when only request B's seed changes.

    Assumptions: FP32; ``ignore_eos=True``; two requests of four rows each fit the default
    16-token budget, so both get full rounds in lockstep (the same plan except for B's tail
    rounds, whose length depends on B's acceptances). Row numerics are batch-shape independent
    to FP32 accuracy, far below any sampling threshold.
    """
    model = tiny_ouro_model("cuda", torch.float32)

    def run(seed_b):
        with closing(make_engine(model)) as e:
            e.add_request("a", list(PROMPT), random_params(11, 21))
            e.add_request("b", list(PROMPT), random_params(seed_b, 17))
            outputs = drain(e)
            assert_reclaimed(e)
            return {rid: o.token_ids for rid, o in outputs.items()}

    first, second = run(22), run(33)
    assert len(first["a"]) == len(second["a"]) == 21
    assert len(first["b"]) == len(second["b"]) == 17
    assert first["a"] == second["a"]
    assert first["b"] != second["b"]


# --------------------------------------------------------------------- 7. ragged / K = 0


@pytest.mark.gpu
@pytest.mark.parametrize("dtype,graphs", [(torch.float32, False), (torch.bfloat16, True)])
def test_ragged_and_zero_draft_rounds_complete_with_exact_lengths(monkeypatch, dtype, graphs):
    """Request lengths 2, 6 and 22 force K = 0 rounds next to full rounds.

    ``a`` (max_tokens=2) has exactly one round with one row (K=0). ``b`` (6) has a full round of
    four rows and, scheduled while that is in flight, a one-row round (remaining budget 1).
    ``c`` (22) keeps full rounds going, so rounds mixing K=0 and K>0 requests occur. All three
    use identical prompts and are admitted together, so they advance in lockstep.
    """
    with closing(make_engine(tiny_ouro_model("cuda", dtype), graphs=graphs)) as e:
        recorder = RoundRecorder(monkeypatch, e.speculative_runner)
        lengths = {"a": 2, "b": 6, "c": 22}
        for i, (rid, length) in enumerate(lengths.items()):
            e.add_request(rid, list(PROMPT), random_params(10 + i, length))
        outputs = drain(e)
        assert {rid: len(o.token_ids) for rid, o in outputs.items()} == lengths
        assert all(o.finish_reason == "length" for o in outputs.values())
        counts = recorder.counts()
        assert all(record.random for record in recorder.rounds)
        assert [c["a"] for c in counts if "a" in c] == [1]
        b_counts = [c["b"] for c in counts if "b" in c]
        assert b_counts[0] == K + 1 and 1 in b_counts
        assert max(max(c.values()) for c in counts) == K + 1
        assert any(min(c.values()) == 1 and max(c.values()) > 1 for c in counts)
        assert_reclaimed(e)

        # Every random request of a round has K = 0 (no q buffer): one-row requests only.
        seen = len(recorder.rounds)
        for rid in ("d", "e"):
            e.add_request(rid, list(PROMPT), random_params(40 + ord(rid), 2))
        outputs = drain(e)
        assert {rid: len(o.token_ids) for rid, o in outputs.items()} == {"d": 2, "e": 2}
        assert recorder.rounds[seen:]
        assert all(count == 1 for record in recorder.rounds[seen:] for _, count in record.items)
        assert_reclaimed(e)


# --------------------------------------------------------------------------- 8. graphs


@pytest.mark.gpu
def test_random_and_greedy_requests_replay_graphs_and_produce_valid_tokens(monkeypatch):
    """Graph path: replays really happen; outputs are only checked for shape and validity."""
    model = tiny_ouro_model("cuda", torch.bfloat16)
    with closing(make_engine(model, graphs=True)) as e:
        runner = e.speculative_runner
        recorder = RoundRecorder(monkeypatch, runner)
        core_before = runner.graphs.replays
        coda_before = runner.coda_graphs.replays
        e.add_request("x", list(PROMPT), random_params(51, 19))
        e.add_request("y", list(PROMPT), random_params(52, 13, top_k=16, top_p=0.9))
        e.add_request("g", list(PROMPT), greedy_params(10))
        outputs = drain(e)
        assert runner.graphs.replays > core_before
        assert runner.coda_graphs.replays > coda_before
        assert any(record.random for record in recorder.rounds)
        assert {rid: len(o.token_ids) for rid, o in outputs.items()} == {"x": 19, "y": 13, "g": 10}
        vocab = e.model.config.vocab_size
        assert all(0 <= t < vocab for o in outputs.values() for t in o.token_ids)
        assert all(o.finish_reason == "length" for o in outputs.values())
        assert_reclaimed(e)


# ------------------------------------------------------------------ 9. error handling (3.4)


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_invalid_distribution_in_an_inflight_round_is_fatal_and_the_engine_recovers(
    monkeypatch, graphs
):
    with closing(make_engine(tiny_ouro_model("cuda"), graphs=graphs)) as e:
        runner = e.speculative_runner
        injector = NanInjector(monkeypatch, runner, round_index=1)
        bootstrap_random(e, "r", random_params(41, 24))
        e.step()
        e.step()
        # Round 2 was submitted behind the corrupted round 1 (failed rows keep their slot).
        assert len(e._pending_speculative) == 2
        assert runner.submitted_before_collect == 1
        with pytest.raises(RuntimeError, match="invalid speculative sampling distribution"):
            e.step()
        assert injector.fired == 1
        assert not e.has_unfinished_requests()
        assert_reclaimed(e)

        e.add_request("r", list(PROMPT), random_params(42, 12))
        output = drain(e)["r"]
        assert len(output.token_ids) == 12 and output.finish_reason == "length"
        assert injector.fired == 1
        assert_reclaimed(e)


@pytest.mark.gpu
def test_stale_error_flag_of_an_aborted_request_is_ignored_when_the_id_is_reused(monkeypatch):
    with closing(make_engine(tiny_ouro_model("cuda"))) as e:
        runner = e.speculative_runner
        injector = NanInjector(monkeypatch, runner, round_index=1)
        bootstrap_random(e, "r", random_params(51, 24))
        e.step()
        e.step()
        assert len(e._pending_speculative) == 2
        runner.synchronize()
        assert injector.fired == 1
        # The older ticket does carry the error; it must be dropped with the request.
        assert e._pending_speculative[0].collect().speculative[0].numerical_error
        e.abort_request("r")
        e.add_request("r", list(PROMPT), random_params(52, 14))
        output = drain(e)["r"]  # must not raise
        assert len(output.token_ids) == 14 and output.finish_reason == "length"
        assert runner.discarded_rounds >= 1
        assert_reclaimed(e)


@pytest.mark.gpu
def test_greedy_rounds_reusing_random_round_buffers_never_report_errors(monkeypatch):
    with closing(make_engine(tiny_ouro_model("cuda"))) as e:
        runner = e.speculative_runner
        tickets = spy_tickets(monkeypatch, runner)
        e.add_request("x", list(PROMPT), random_params(61, 16))
        assert len(drain(e)["x"].token_ids) == 16
        assert tickets and all(ticket.error_flags for ticket in tickets)
        assert_reclaimed(e)

        # Fill the tail of both result buffers with "error" flags a buggy reader would see.
        runner.synchronize()
        torch.cuda.synchronize()
        for buffer in runner.buffers:
            buffer.host_result.fill_(1)
            buffer.result.fill_(1)
        torch.cuda.synchronize()

        seen = len(tickets)
        e.add_request("g", list(PROMPT), greedy_params(16))
        output = drain(e)["g"]
        assert len(output.token_ids) == 16 and output.finish_reason == "length"
        greedy = tickets[seen:]
        assert greedy and all(not ticket.error_flags for ticket in greedy)
        assert all(
            ticket.cached is not None
            and not any(result.numerical_error for result in ticket.cached.speculative)
            for ticket in greedy
        )
        assert_reclaimed(e)


# ------------------------------------------------------------- 10. greedy path untouched


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_all_greedy_async_batches_never_plan_sample_or_acquire_a_generator(monkeypatch, graphs):
    with closing(make_engine(tiny_ouro_model("cuda"), graphs=graphs)) as e:
        runner = e.speculative_runner

        def forbidden(*args, **kwargs):
            raise AssertionError("an all-greedy speculative round used the random path")

        monkeypatch.setattr(e.rng, "acquire", forbidden)
        monkeypatch.setattr(runner, "_uniforms", forbidden)
        plans = []
        plan_random_sampling = async_speculative.plan_random_sampling

        def spying_plan(*args, **kwargs):
            plans.append(plan_random_sampling(*args, **kwargs))
            return plans[-1]

        monkeypatch.setattr(async_speculative, "plan_random_sampling", spying_plan)
        tickets = spy_tickets(monkeypatch, runner)
        lengths = {"a": 13, "b": 7, "c": 1}
        for i, (rid, length) in enumerate(lengths.items()):
            e.add_request(rid, [2 + i, 3, 4], greedy_params(length))
        outputs = drain(e)
        assert {rid: len(o.token_ids) for rid, o in outputs.items()} == lengths
        assert plans and all(plan is None for plan in plans)
        assert tickets and all(not ticket.error_flags for ticket in tickets)
        assert e.rng.peek("a") is None
        assert_reclaimed(e)
