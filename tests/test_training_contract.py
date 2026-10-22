"""Full-publication recovery and selected-token scores across stage refactoring."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.helpers import LocalLLM, tiny_ouro_model
from tests.reference import dense_reference
from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SpeculativeConfig,
)
from vllm_rlt.engine.llm_engine import StageClient
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine


@pytest.mark.parametrize("asynchronous,speculative", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
@pytest.mark.parametrize("temperature", [0, 0.7])
def test_selected_scores_match_dense_target_and_keep_tokens(
    asynchronous, speculative, mode, temperature
):
    model = tiny_ouro_model()
    llm = LocalLLM(
        model,
        cache_config=CacheConfig(256, 2),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
        exit_config=ExitConfig("ouro_delayed" if asynchronous else "ouro"),
        speculative_config=SpeculativeConfig(2) if speculative else None,
        logprobs_mode=mode,
    )
    params = SamplingParams(max_tokens=7, ignore_eos=True, temperature=temperature, logprobs=0)
    prompts = [[2, 3], [4, 5, 6]]
    try:
        outputs = llm.generate(prompts, params)
        ordinary = llm.generate(prompts, replace(params, logprobs=None))
        assert [o.token_ids for o in outputs] == [o.token_ids for o in ordinary]
        for prompt, output in zip(prompts, outputs):
            assert len(output.logprobs) == len(output.token_ids) == 7
            prefix = list(prompt)
            for token, actual in zip(output.token_ids, output.logprobs):
                logits = dense_reference(model, prefix, model.config.total_ut_steps)[-1][2][-1]
                if mode == "processed_logprobs" and temperature > 0:
                    logits = logits / temperature
                expected = logits.float().log_softmax(-1)[token].item()
                assert actual == pytest.approx(expected, abs=2e-5)
                prefix.append(token)
    finally:
        llm.close()


def test_full_publication_invalidates_prefix_and_preserves_parameter_addresses():
    model = tiny_ouro_model()
    engine = StageEngine(model, cache_config=CacheConfig(128, 2, enable_prefix_caching=True))
    llm = LLM(engine)
    params = SamplingParams(max_tokens=2, ignore_eos=True)
    try:
        llm.generate([[2, 3, 4, 5]], params)
        assert engine.cache_manager.lookup_prefix([2, 3, 4, 5])
        addresses = {name: value.data_ptr() for name, value in model.named_parameters()}
        weights = [
            (name, value.detach().clone() * 0.95) for name, value in model.named_parameters()
        ]
        llm.start_weight_update(3)
        assert not engine.cache_manager.lookup_prefix([2, 3, 4, 5])
        with pytest.raises(RuntimeError, match="paused"):
            engine.add_request("blocked", [1], params)
        llm.update_weights(weights[:1])
        with pytest.raises(ValueError, match="every physical"):
            llm.finish_weight_update()
        llm.update_weights(weights[1:])
        llm.finish_weight_update()
        assert llm.get_weight_version() == 3
        assert addresses == {name: value.data_ptr() for name, value in model.named_parameters()}
        for (name, value), (_, expected) in zip(model.named_parameters(), weights):
            torch.testing.assert_close(value, expected)
        output = llm.generate([[2, 3]], params)[0]
        assert output.weight_version == 3
        assert output.sampling_params == params
    finally:
        llm.close()


@pytest.mark.parametrize("failure", ["duplicate", "shape", "iteration"])
def test_failed_chunk_blocks_generation_until_new_complete_transaction(failure):
    model = tiny_ouro_model()
    engine = StageEngine(model, cache_config=CacheConfig(64, 2))
    weights = [(name, value.detach().clone()) for name, value in model.named_parameters()]

    def interrupted_chunk():
        yield weights[0]
        raise RuntimeError("interrupted source")

    chunk = {
        "duplicate": [weights[0], weights[0]],
        "shape": [(weights[0][0], torch.empty(1))],
        "iteration": interrupted_chunk(),
    }[failure]
    try:
        engine.start_weight_update(1)
        original = dict(model.named_parameters())[weights[0][0]].detach().clone()
        with pytest.raises((ValueError, RuntimeError)):
            engine.update_weights(chunk)
        torch.testing.assert_close(dict(model.named_parameters())[weights[0][0]], original)
        for call in (engine.step, engine.resume_generation, engine.finish_weight_update):
            with pytest.raises(RuntimeError):
                call()
        with pytest.raises(ValueError, match="increase"):
            engine.start_weight_update(1)
        engine.start_weight_update(2)
        engine.update_weights(weights)
        assert engine.finish_weight_update() == 2
        engine.resume_generation()
        engine.add_request("recovered", [1], SamplingParams(max_tokens=1))
        engine.abort_request("recovered")
    finally:
        engine.close()


def test_publication_waits_for_requests_and_remote_transfer_leases():
    engine = StageEngine(tiny_ouro_model(), cache_config=CacheConfig(64, 2))
    try:
        engine.add_request("live", [1, 2], SamplingParams(max_tokens=1))
        engine.pause_generation()
        with pytest.raises(RuntimeError, match="idle"):
            engine.start_weight_update()
        engine.abort_request("live")
        engine.connector = SimpleNamespace(has_pending=True)
        with pytest.raises(RuntimeError, match="idle"):
            engine.start_weight_update()
        engine.connector = None
        engine.start_weight_update()
    finally:
        engine.close()


@pytest.mark.parametrize("speculative", [False, True])
def test_explicit_stop_applies_with_ignore_eos_and_scores_align(speculative):
    llm = LocalLLM(
        tiny_ouro_model(),
        speculative_config=SpeculativeConfig(2) if speculative else None,
        cache_config=CacheConfig(128, 2),
    )
    try:
        params = SamplingParams(max_tokens=7, ignore_eos=True, logprobs=0)
        full = llm.generate([[2, 3]], params)[0]
        stop = full.token_ids[2]
        stopped = llm.generate([[2, 3]], replace(params, stop_token_ids=(stop,)))[0]
        count = full.token_ids.index(stop) + 1
        assert stopped.token_ids == full.token_ids[:count]
        assert stopped.logprobs == pytest.approx(full.logprobs[:count], abs=2e-5)
        assert stopped.finish_reason == "stop"
    finally:
        llm.close()


def test_cpu_stage_clients_publish_to_all_pd_peers_and_recover_control_errors():
    model = tiny_ouro_model()
    clients = [
        StageClient(model, device="cpu", dtype="float32", cache_config=CacheConfig(64, 2))
        for _ in range(2)
    ]
    pd = object.__new__(PDEngine)
    pd.config = PDConfig()
    pd.peers = {str(i): SimpleNamespace(client=c) for i, c in enumerate(clients)}
    pd.requests, pd.transfers = {}, {}
    pd.closed, pd.failure = False, None
    pd._weight_version, pd._update_version, pd._update_failed, pd._paused = 0, None, False, False
    try:
        clients[0].pause_generation()
        with pytest.raises(RuntimeError, match="paused"):
            clients[0].add_request("paused", [1], SamplingParams(max_tokens=1))
        assert clients[0].process.is_alive() and clients[0].failure is None
        clients[0].resume_generation()
        pd.start_weight_update(1)
        # One rank rejects a duplicate while another has already copied its chunk.
        first = next(iter(model.named_parameters()))
        clients[1].update_weights([first])
        with pytest.raises(RuntimeError, match="Duplicate"):
            pd.update_weights([first])
        assert pd.get_weight_version() == 0
        assert pd._update_failed
        assert all(c.process.is_alive() and c.failure is None for c in clients)
        assert all(not c.pending_controls and not c.replies for c in clients)
        with pytest.raises(RuntimeError):
            pd.resume_generation()
        pd.start_weight_update(2)
        pd.update_weights(model.named_parameters())
        pd.finish_weight_update()
        assert pd.get_weight_version() == 2
        assert [c.get_weight_version() for c in clients] == [2, 2]
        llm = LLM(clients[0])
        output = llm.generate([[2, 3]], SamplingParams(max_tokens=2, logprobs=0))[0]
        assert output.weight_version == 2
        assert len(output.logprobs) == len(output.token_ids)
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_mixed_score_admission_and_unspecified_seed_replay(asynchronous):
    model = tiny_ouro_model()
    llm = LocalLLM(
        model,
        cache_config=CacheConfig(128, 2),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
        exit_config=ExitConfig("ouro_delayed" if asynchronous else "ouro"),
    )
    try:
        scored = SamplingParams(
            max_tokens=4,
            temperature=0.8,
            logprobs=0,
            seed=None,
            top_k=model.config.vocab_size + 5,
            ignore_eos=True,
        )
        outputs = llm.generate([[2, 3], [4, 5]], [replace(scored, logprobs=None), scored])
        assert outputs[0].logprobs is None
        assert len(outputs[1].logprobs) == 4
        effective = outputs[1].sampling_params
        assert effective == replace(scored, seed=effective.seed)
        assert 0 <= effective.seed < 2**63
        replay = llm.generate([[4, 5]], effective)[0]
        assert replay.token_ids == outputs[1].token_ids
        assert replay.logprobs == pytest.approx(outputs[1].logprobs, abs=2e-5)
    finally:
        llm.close()
