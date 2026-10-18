"""Mixed-depth execution, round equivalence and pipeline rollback contracts."""

import argparse
from collections import Counter

import pytest
import torch

from tests.test_speculative import drain, engine, model
from vllm_rlt import LLM, CacheConfig, SamplingParams, SpeculativeConfig
from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput
from vllm_rlt.entrypoints.runtime_args import add_runtime_args, runtime_configs
from vllm_rlt.request import Request, Stage
from vllm_rlt.worker.speculative import SpeculativeResult


@pytest.mark.parametrize("draft_depth", [1, 2, 3])
@pytest.mark.parametrize("sampling", [False, True])
def test_pipeline_round_matches_baseline_and_batches_mixed_depths(draft_depth, sampling):
    m = model()
    snapshots = []
    for pipelined in (False, True):
        e = engine(m)
        runner = e.speculative_runner
        runner.config = SpeculativeConfig(4, draft_depth, 4, pipelined)
        cache = e.cache_manager
        items = []
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
            items.append(ScheduledItem(request, len(prefix), count))
        calls, work, heads = [], Counter(), []
        core, verify = runner._core, runner._verify

        def record(hidden, ids, positions, depth, **kwargs):
            depths = [row[1] for row in kwargs["batch"].rows]
            calls.append(list(zip(ids, positions, depths)))
            work.update(zip(ids, positions, depths))
            return core(hidden, ids, positions, depth, **kwargs)

        def record_verify(items, drafted, proposals, logits):
            heads.append(logits.clone())
            return verify(items, drafted, proposals, logits)

        runner._core, runner._verify = record, record_verify
        results = runner.execute(SchedulerOutput(Stage.SPECULATIVE, items))
        expected_work = Counter(
            (item.request.request_id, pos, depth)
            for item in items
            for pos in range(item.token_start, item.token_start + item.token_count)
            for depth in range(4)
        )
        assert work == expected_work  # Each position/depth is computed once.
        if pipelined:
            assert any(len({d for _, _, d in call}) > 1 for call in calls)
            # History at a given depth must finish before its later position.
            completed = set()
            for call in calls:
                for rid, pos, depth in call:
                    assert pos == 2 or (rid, pos - 1, depth) in completed
                completed.update(call)
            assert len(calls) == draft_depth * 4 + 4
        kv = [
            tensor.clone()
            for item in items
            for depth in range(4)
            for layer in range(m.config.num_hidden_layers)
            for tensor in cache.read(
                layer, item.request.request_id, depth, item.token_start + item.token_count
            )
        ]
        snapshots.append((results, heads[0], kv, runner.stats))
    baseline, pipeline = snapshots
    assert baseline[0] == pipeline[0]
    torch.testing.assert_close(baseline[1], pipeline[1], atol=4e-5, rtol=4e-5)
    for a, b in zip(baseline[2], pipeline[2]):
        torch.testing.assert_close(a, b, atol=4e-5, rtol=4e-5)
    assert baseline[3] == pipeline[3]


@pytest.mark.parametrize("rejected", [0, 1, 2, 3])
@pytest.mark.parametrize("temperature", [0, 1])
def test_pipeline_correction_bonus_and_next_round(rejected, temperature, monkeypatch):
    m = model()
    e = engine(m, pipelined=True, cache_config=CacheConfig(128, 2))
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    request = e.scheduler.requests["r"]
    request.sampling_params = SamplingParams(
        max_tokens=12, ignore_eos=True, temperature=temperature, seed=42
    )
    start = request.position
    runner = e.speculative_runner
    core, coda = runner._core_mixed, runner._coda
    head_rows = []

    def record(hidden, ids, positions, depths):
        nonlocal head_rows
        output = core(hidden, ids, positions, depths)
        # Force proposal/target distributions by semantic position and depth,
        # independent of which other rows share this batched LM-head call.
        head_rows = [
            (p - start, False) for p, d in zip(positions, depths) if d == 1 and p - start < 3
        ]
        head_rows += [(p - start, True) for p, d in zip(positions, depths) if d == 3]
        return output

    def controlled(hidden):
        assert len(hidden) == len(head_rows)
        logits = torch.full((len(hidden), m.config.vocab_size), -torch.inf)
        for row, (offset, target) in enumerate(head_rows):
            token = 20 + offset if target and offset >= rejected else 10 + offset
            logits[row, token] = 0
        return logits

    monkeypatch.setattr(runner, "_core_mixed", record)
    monkeypatch.setattr(runner, "_coda", controlled)
    output = e.step()[0]
    count = rejected + 1
    assert output.token_ids[-count:] == list(range(10, 10 + rejected)) + [20 + rejected]
    assert runner.stats.accepted_tokens == rejected
    for plane in e.cache_manager._get_allocation("r").written:
        assert all(w.prefix == start + count and not w.pending for w in plane)
    monkeypatch.setattr(runner, "_core_mixed", core)
    monkeypatch.setattr(runner, "_coda", coda)
    # Replay a common committed prefix with native greedy to expose stale KV.
    request.sampling_params = SamplingParams(max_tokens=12, ignore_eos=True)
    expected = LLM(m).generate(
        [output.prompt_token_ids + output.token_ids],
        SamplingParams(max_tokens=12 - len(output.token_ids), ignore_eos=True),
    )[0]
    assert drain(e)["r"].token_ids[len(output.token_ids) :] == expected.token_ids
    assert e.cache_manager.num_used_blocks == 0


def test_greedy_early_rejection_prunes_unstarted_suffix(monkeypatch):
    m = model()
    e = engine(m, pipelined=True, cache_config=CacheConfig(128, 2))
    runner = e.speculative_runner
    runner.config = SpeculativeConfig(4, 2, 4, True, True)
    cache = e.cache_manager
    request = Request(
        "r", [2, 3], SamplingParams(max_tokens=12, ignore_eos=True), generated_token_ids=[7]
    )
    cache.allocate(request.request_id, 16)
    for pos, token in enumerate(request.prompt_token_ids):
        hidden = m.prelude(torch.tensor([token]))
        for depth in range(4):
            hidden, _ = m.recurrent(hidden, [request.request_id], [depth], [pos], cache)
    item = ScheduledItem(request, len(request.prompt_token_ids), 4)

    calls = []
    core, coda = runner._core_mixed, runner._coda
    head_rows = []

    def record(hidden, ids, positions, depths):
        nonlocal head_rows
        calls.append([(p - item.token_start, d) for p, d in zip(positions, depths)])
        output = core(hidden, ids, positions, depths)
        head_rows = [(p - item.token_start, False) for p, d in zip(positions, depths) if d == 1]
        head_rows += [(p - item.token_start, True) for p, d in zip(positions, depths) if d == 3]
        return output

    def controlled(hidden):
        assert len(hidden) == len(head_rows)
        logits = torch.full((len(hidden), m.config.vocab_size), -torch.inf)
        for row, (offset, target) in enumerate(head_rows):
            token = 20 if target and offset == 0 else 10 + offset
            logits[row, token] = 0
        return logits

    monkeypatch.setattr(runner, "_core_mixed", record)
    monkeypatch.setattr(runner, "_coda", controlled)
    result = runner.execute(SchedulerOutput(Stage.SPECULATIVE, [item]))[0]

    assert result == SpeculativeResult([20], 0, 1)
    assert runner.stats.verified_rows == 1
    assert runner.stats.drafted_tokens == 1
    assert all(offset < 2 for call in calls for offset, _ in call)
    assert any(offset == 1 and depth == 1 for call in calls for offset, depth in call)
    assert not any(offset == 1 and depth >= 2 for call in calls for offset, depth in call)
    assert not any(offset >= 2 for call in calls for offset, _ in call)
    monkeypatch.setattr(runner, "_core_mixed", core)
    monkeypatch.setattr(runner, "_coda", coda)


def test_pipeline_config_and_cli():
    assert not SpeculativeConfig(3).loop_level_pipelining
    for invalid in (None, 0, 1, "true"):
        with pytest.raises(ValueError, match="loop_level_pipelining"):
            SpeculativeConfig(3, loop_level_pipelining=invalid)
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    parser.set_defaults(num_blocks=128, block_size=16)
    args = parser.parse_args(
        ["--speculative-tokens", "3", "--loop-level-pipelining", "--early-rejection-pruning"]
    )
    config = runtime_configs(args)["speculative_config"]
    assert config.loop_level_pipelining and config.early_rejection_pruning
    args = parser.parse_args(["--loop-level-pipelining"])
    with pytest.raises(ValueError, match="requires --speculative-tokens"):
        runtime_configs(args)


def test_pipeline_partial_execution_failure_releases_kv(monkeypatch):
    e = engine(model(), pipelined=True)
    e.add_request("r", [2, 3], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    runner = e.speculative_runner
    core = runner._core_mixed
    calls = 0

    def fail_after_write(*args):
        nonlocal calls
        result = core(*args)
        calls += 1
        if calls == 3:
            raise RuntimeError("partial pipeline failure")
        return result

    monkeypatch.setattr(runner, "_core_mixed", fail_after_write)
    with pytest.raises(RuntimeError, match="partial pipeline failure"):
        e.step()
    assert not e.has_unfinished_requests()
    assert e.cache_manager.num_used_blocks == 0
