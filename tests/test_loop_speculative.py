"""Mixed-depth schedule, round equivalence and pipeline rollback contracts."""

import argparse
from collections import Counter

import pytest
import torch

from tests.helpers import LocalLLM as LLM
from tests.test_speculative import drain, engine, model
from vllm_rlt import CacheConfig, SamplingParams, SpeculativeConfig
from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
from vllm_rlt.entrypoints.runtime_args import add_runtime_args, runtime_configs
from vllm_rlt.request import Request, RequestPhase
from vllm_rlt.worker.speculative import loop_waves


@pytest.mark.parametrize("draft_loops,target_loops", [(1, 4), (2, 4), (3, 4), (1, 2)])
@pytest.mark.parametrize("counts", [(1,), (5,), (5, 3, 1), (2, 9)])
def test_loop_waves_cover_each_depth_once_after_its_history(counts, draft_loops, target_loops):
    waves = loop_waves(counts, draft_loops, target_loops)
    assert len(waves) == max((c - 1) * draft_loops + target_loops for c in counts)
    seen = Counter()
    for wave, (rows, drafts) in enumerate(waves):
        for i, offset, depth in rows:
            assert depth == wave - offset * draft_loops
            # The previous position at this depth ran in an earlier wave.
            assert offset == 0 or seen[i, offset - 1, depth]
            # This position's previous depth ran in the previous wave.
            assert depth == 0 or seen[i, offset, depth - 1]
        seen.update(rows)
        assert list(drafts) == [
            row
            for row, (i, offset, depth) in enumerate(rows)
            if depth == draft_loops - 1 and offset + 1 < counts[i]
        ]
        # One draft per request and wave keeps proposals in offset order.
        assert len({rows[row][0] for row in drafts}) == len(drafts)
    assert seen == Counter(
        (i, offset, depth)
        for i, count in enumerate(counts)
        for offset in range(count)
        for depth in range(target_loops)
    )
    if max(counts) > 1 and target_loops > draft_loops:
        assert any(len({depth for _, _, depth in rows}) > 1 for rows, _ in waves)


def _round(m, pipelined, draft_loops, sampling):
    e = engine(m, k=4, pipelined=pipelined, cache_config=CacheConfig(128, 2))
    runner = e.speculative_runner
    runner.config = SpeculativeConfig(
        4, draft_loops=draft_loops, target_loops=4, loop_level_pipelining=pipelined
    )
    cache = e.cache_manager
    spans = []
    for i, count in enumerate((5, 3, 1)):
        rid = str(i)
        prefix = [2, 3 + i]
        cache.allocate(rid, 16)
        for pos, token in enumerate(prefix):
            h = m.prelude(torch.tensor([token]))
            for depth in range(4):
                h, _ = m.recurrent(h, [rid], [depth], [pos], cache)
        params = SamplingParams(
            temperature=0.8 if sampling and i != 1 else 0,
            top_k=7,
            top_p=0.8,
            seed=42,
            ignore_eos=True,
        )
        request = Request(rid, prefix, params, generated_token_ids=[7 + i])
        spans.append(ScheduledRequest(request, len(prefix), count))
    calls = []
    core = m.recurrent_prepared

    def record(hidden, batch, cache, **kwargs):
        calls.append([(id(a), depth, pos) for a, depth, pos in batch.rows])
        return core(hidden, batch, cache, **kwargs)

    heads = []
    hook = m.lm_head.register_forward_hook(
        lambda module, inputs, output: heads.append(output.detach().clone())
    )
    m.recurrent_prepared = record
    try:
        results = runner.submit(SchedulerOutput(RequestPhase.SPECULATIVE, spans)).collect()
    finally:
        m.recurrent_prepared = core
        hook.remove()
    kv = [
        tensor.clone()
        for span in spans
        for depth in range(4)
        for layer in range(m.config.num_hidden_layers)
        for tensor in cache.read(
            layer, span.request.request_id, depth, span.token_start + span.token_count
        )
    ]
    return results.speculative, heads[-1], kv, runner.stats, calls


@pytest.mark.parametrize("draft_loops", [1, 2, 3])
@pytest.mark.parametrize("sampling", [False, True])
def test_loop_round_matches_block_round_and_batches_mixed_depths(draft_loops, sampling):
    m = model()
    block = _round(m, False, draft_loops, sampling)
    loop = _round(m, True, draft_loops, sampling)
    # Same proposals, verification rows, KV and same-seed sampling decisions.
    assert loop[0] == block[0]
    torch.testing.assert_close(loop[1], block[1], atol=4e-5, rtol=4e-5)
    for a, b in zip(loop[2], block[2]):
        torch.testing.assert_close(a, b, atol=4e-5, rtol=4e-5)
    assert loop[3] == block[3]
    calls = loop[4]
    work = Counter(row for call in calls for row in call)
    assert len(work) == sum(work.values()) == (5 + 3 + 1) * 4
    assert len(calls) == 4 * draft_loops + 4
    assert any(len({depth for _, depth, _ in call}) > 1 for call in calls)


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_loop_round_runs_only_draft_heads_inside_the_loop(temperature, monkeypatch):
    m = model()
    e = engine(m, k=3, pipelined=True)
    e.add_request(
        "r", [2, 3], SamplingParams(max_tokens=9, ignore_eos=True, temperature=temperature)
    )
    while not e.step():
        pass
    runner = e.speculative_runner
    rounds, heads = [], []
    submit, coda = runner._submit_loop_round, runner._coda
    monkeypatch.setattr(runner, "_submit_round", None)
    monkeypatch.setattr(
        runner, "_submit_loop_round", lambda *args: rounds.append(1) or submit(*args)
    )
    monkeypatch.setattr(runner, "_coda", lambda hidden: heads.append(len(hidden)) or coda(hidden))
    e.step()
    # K single-row draft heads, then one head over all K + 1 verification rows.
    assert rounds == [1] and heads == [1, 1, 1, 4]


def test_loop_generation_agrees_at_eight_concurrent_requests():
    m = model()
    prompts = [[2 + i, 7 - (i % 4), 11 + (i % 3)] for i in range(8)]
    params = [SamplingParams(max_tokens=7, ignore_eos=True) for _ in prompts]
    native = LLM(m).generate(prompts, params)
    for k in (1, 2, 4, 8):
        actual = LLM(
            m,
            speculative_config=SpeculativeConfig(k, loop_level_pipelining=True),
            cache_config=CacheConfig(256, 2),
        ).generate(prompts, params)
        assert [o.token_ids for o in actual] == [o.token_ids for o in native]
        assert [o.exit_depths for o in actual] == [o.exit_depths for o in native]


def test_loop_sampling_matches_block_round_for_the_same_seed():
    # Both rounds consume each request's generator in the same order: drafts by
    # offset, then verification. Same-seed outputs therefore agree.
    m = model()
    prompts = [[2, 3], [7, 8, 9], [4]]
    params = SamplingParams(max_tokens=11, temperature=0.8, top_k=7, top_p=0.8, seed=5)

    def run(pipelined):
        return LLM(
            m, speculative_config=SpeculativeConfig(3, loop_level_pipelining=pipelined)
        ).generate(prompts, params)

    assert [o.token_ids for o in run(True)] == [o.token_ids for o in run(False)]


def test_pipeline_config_and_cli():
    assert not SpeculativeConfig(3).loop_level_pipelining
    for invalid in (None, 0, 1, "true"):
        with pytest.raises(ValueError, match="loop_level_pipelining"):
            SpeculativeConfig(3, loop_level_pipelining=invalid)
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    parser.set_defaults(num_blocks=128, block_size=16)
    args = parser.parse_args(["--speculative-tokens", "3", "--loop-level-pipelining"])
    assert runtime_configs(args)["speculative_config"].loop_level_pipelining
    args = parser.parse_args(["--loop-level-pipelining"])
    with pytest.raises(ValueError, match="requires --speculative-tokens"):
        runtime_configs(args)
    with pytest.raises(ValueError, match="mutually exclusive"):
        SpeculativeConfig(3, mode="pipeline", loop_level_pipelining=True)


def test_pipeline_partial_execution_failure_releases_kv(monkeypatch):
    m = model()
    e = engine(m, pipelined=True)
    e.add_request("r", [2, 3], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    core = m.recurrent_prepared
    calls = 0

    def fail_after_write(*args, **kwargs):
        nonlocal calls
        result = core(*args, **kwargs)
        calls += 1
        if calls == 3:
            raise RuntimeError("partial pipeline failure")
        return result

    monkeypatch.setattr(m, "recurrent_prepared", fail_after_write)
    with pytest.raises(RuntimeError, match="partial pipeline failure"):
        e.step()
    assert not e.has_unfinished_requests()
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("rejected", [0, 1, 2, 3])
def test_loop_rejection_rolls_back_and_next_round_matches_native(rejected, monkeypatch):
    m = model()
    e = engine(m, k=3, pipelined=True, cache_config=CacheConfig(128, 2))
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    coda = m.coda
    calls = 0

    def controlled(h):
        nonlocal calls
        logits = torch.full((len(h), m.config.vocab_size), -100.0)
        if calls < 3:
            logits[:, 10 + calls] = 100
        else:
            for i in range(4):
                logits[i, 10 + i if i < rejected else 20 + i] = 100
        calls += 1
        return logits

    monkeypatch.setattr(m, "coda", controlled)
    out = e.step()[0]
    assert out.token_ids[-rejected - 1 :] == list(range(10, 10 + rejected)) + [20 + rejected]
    monkeypatch.setattr(m, "coda", coda)
    expected = LLM(m).generate(
        [out.prompt_token_ids + out.token_ids],
        SamplingParams(max_tokens=12 - len(out.token_ids), ignore_eos=True),
    )[0]
    assert drain(e)["r"].token_ids[len(out.token_ids) :] == expected.token_ids
    assert e.cache_manager.num_used_blocks == 0
