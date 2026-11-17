"""Random sampling in asynchronous block speculation, through the public entrypoint.

Everything here goes through ``vllm_rlt.LLM``: it launches a spawned ``StageWorker`` process
behind a ``StageClient`` and talks to it over the control pipe, as a user would. All tests need
``--run-gpu`` (async speculation requires CUDA).

The PD decode-stage guard that rejects random sampling under async speculation
(``vllm_rlt/pd/engine.py``) is unchanged by this feature and is not exercised here.

Observability note. The worker process exposes no speculative statistics: ``StageEngine.stats()``
(the only stats control call) returns just ``requests`` and ``used_blocks``, and ``RequestOutput``
carries tokens and exit depths only. The one signal that crosses the process boundary is the
stream of ``RequestOutput`` snapshots. In async speculation the first token comes from the
bootstrap CODA step and every later token is committed by a delivered speculative round, which
emits exactly one snapshot per round. So the snapshot sequence of a request is
``[1, 1 + d1, 1 + d1 + d2, ...]`` with ``1 <= d_i <= K + 1``, and it proves that rounds were
submitted and delivered. A delta above 1 additionally proves that a delivered round accepted
at least one draft, since a plain CODA step commits exactly one token. It cannot show the
round took the random branch: for a request with ``temperature > 0`` a round is always
planned as random, which unit tests cover
(``test_async_speculative_sampling_plan.py``, ``test_async_speculative_lifecycle.py``).
A device-side counter of random rounds would need a new field in ``StageEngine.stats()``.
"""

import math

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)

K = 3
VOCAB = tiny_ouro_config().vocab_size
PROMPT = [2, 3, 4]
MAX_TOKENS = 12


def random_params(**overrides):
    values = dict(
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        top_k=16,
        top_p=0.95,
        seed=11,
        ignore_eos=True,
    )
    values.update(overrides)
    return SamplingParams(**values)


def make_llm(mode="block"):
    """The public facade over one CUDA worker process, FP32 so greedy rows are reproducible."""
    # The worker initializes weights from LLM's seed (torch.initial_seed()): pin it.
    torch.manual_seed(123)
    return LLM(
        tiny_ouro_config(),
        device="cuda",
        dtype=torch.float32,
        attention_backend="triton",
        cache_config=CacheConfig(256, 2, enable_prefix_caching=False, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=16, prefill_chunk_size=2
        ),
        exit_config=ExitConfig("ouro"),
        execution_config=ExecutionConfig(async_scheduling=True, multi_stream=True),
        speculative_config=SpeculativeConfig(K, mode=mode),
    )


def stream(llm, requests):
    """Submit (id, prompt, params) triples and keep every RequestOutput snapshot per request.

    Like ``LLM.generate`` this loops until the client reports no unfinished request; a dead
    worker raises from ``step()`` instead of hanging, so no wall-clock bound is involved.
    """
    client = llm.engine
    for request_id, prompt, params in requests:
        client.add_request(request_id, prompt, params)
    history = {request_id: [] for request_id, _, _ in requests}
    while client.has_unfinished_requests():
        for output in client.step():
            history[output.request_id].append(output)
    return history


def assert_valid_tokens(token_ids, count):
    assert len(token_ids) == count
    assert all(type(t) is int and 0 <= t < VOCAB for t in token_ids)


def assert_closed_cleanly(client):
    assert client.closed and client.failure is None
    assert not client.process.is_alive()
    assert client.process.exitcode == 0
    assert client.stats() == dict(requests=0, used_blocks=0)


# 1. Entry contract -----------------------------------------------------------------------------


@pytest.mark.gpu
def test_async_block_engine_accepts_random_sampling():
    params = random_params(max_tokens=4, temperature=0.8, top_k=8, top_p=0.9, seed=7)
    with make_llm() as llm:
        # stream() would raise ValueError here if the worker rejected the request.
        history = stream(llm, [("r", PROMPT, params)])["r"]
        final = history[-1]
        assert final.finished and final.finish_reason == "length"
        assert_valid_tokens(final.token_ids, 4)
    assert_closed_cleanly(llm.engine)


@pytest.mark.gpu
def test_async_pipeline_engine_still_rejects_random_sampling():
    with make_llm("pipeline") as llm:
        client = llm.engine
        with pytest.raises(ValueError, match="greedy"):
            client.add_request("r", PROMPT, random_params())
        # Rejection leaves no client-side trace and does not take the worker down.
        assert client.request_ids() == ()
        assert not client.has_unfinished_requests()
        assert client.process.is_alive()
    assert_closed_cleanly(llm.engine)


# 2. Public end-to-end ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_random_request_completes_through_speculative_rounds_and_closes():
    # Make draft acceptance near-certain on the random-weight tiny model: at temperature 1.5
    # without top-k/top-p both the draft (2 loops) and target (4 loops) distributions are close
    # to uniform over 64 tokens, so sum_x min(p, q) is high for each draft. With 24 tokens
    # (at least 6 rounds of up to 3 drafts) the chance of no accepted draft anywhere is negligible.
    tokens = 24
    params = random_params(max_tokens=tokens, temperature=1.5, top_k=-1, top_p=1.0)
    with make_llm() as llm:
        client = llm.engine
        history = stream(llm, [("r", PROMPT, params)])["r"]
        final = history[-1]
        assert final.finished and final.finish_reason == "length"
        assert_valid_tokens(final.token_ids, tokens)
        assert all(not o.finished for o in history[:-1])

        # Proof of speculation: one snapshot per delivered round (see the module docstring).
        lengths = [len(o.token_ids) for o in history]
        deltas = [b - a for a, b in zip([0] + lengths, lengths)]
        assert deltas[0] == 1, "the bootstrap CODA step commits exactly the first token"
        assert all(0 <= d <= K + 1 for d in deltas[1:])
        assert sum(deltas[1:]) == tokens - 1
        rounds = sum(d > 0 for d in deltas[1:])
        assert rounds >= math.ceil((tokens - 1) / (K + 1)) >= 1
        # Only a delivered speculative round that accepted a draft can commit more than one token.
        assert max(deltas[1:]) > 1
        # Every snapshot extends the previous one: rounds only append.
        assert all(
            a.token_ids == b.token_ids[: len(a.token_ids)] for a, b in zip(history, history[1:])
        )

        # Secondary evidence that sampling was random rather than silently greedy.
        greedy = stream(llm, [("g", PROMPT, SamplingParams(max_tokens=tokens, ignore_eos=True))])
        assert greedy["g"][-1].token_ids != final.token_ids

        # Close while a random request is live, with rounds possibly still in flight.
        client.add_request("close-live", PROMPT, random_params(max_tokens=60, seed=5))
    assert_closed_cleanly(llm.engine)


# 3. Same-seed replay -----------------------------------------------------------------------------


@pytest.mark.gpu
def test_same_seed_replays_identically_through_fresh_llms():
    def run(seed):
        with make_llm() as llm:
            output = llm.generate([PROMPT], random_params(seed=seed))[0]
        assert_valid_tokens(output.token_ids, MAX_TOKENS)
        return output.token_ids

    first = run(11)
    assert run(11) == first
    # A different seed must change the stream, so equality above is not vacuous.
    assert run(12) != first


# 4. Greedy and random requests together -----------------------------------------------------------


@pytest.mark.gpu
def test_concurrent_greedy_and_random_requests_complete_and_greedy_is_unchanged():
    greedy = SamplingParams(max_tokens=9, ignore_eos=True)
    sampled = random_params(max_tokens=13, seed=3)
    greedy_prompt, random_prompt = [2, 3, 4, 5], [7, 8]
    with make_llm() as llm:
        alone = llm.generate([greedy_prompt], greedy)[0]
        assert_valid_tokens(alone.token_ids, 9)
        mixed = llm.generate([greedy_prompt, random_prompt], [greedy, sampled])
        assert [o.finished for o in mixed] == [True, True]
        assert_valid_tokens(mixed[0].token_ids, 9)
        assert_valid_tokens(mixed[1].token_ids, 13)
        # FP32 tiny model: the argmax of a greedy request does not depend on batch composition
        # (the repo's async-vs-sync tests already rely on this across batches).
        assert mixed[0].token_ids == alone.token_ids
    assert_closed_cleanly(llm.engine)
