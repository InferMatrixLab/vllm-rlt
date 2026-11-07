# SPDX-License-Identifier: Apache-2.0
"""CPU checks of deferred decoding against dense, causal Ouro equations."""

import pytest
import torch

from tests.reference import dense_reference as functional_reference
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM
from vllm_rlt.worker.deferred_ouro import DeferredOuroRunner, generate_deferred


def make_model(kv_heads=2, loops=4, max_positions=64):
    torch.manual_seed(19)
    return OuroForCausalLM(
        OuroConfig(
            vocab_size=31,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=kv_heads,
            head_dim=8,
            max_position_embeddings=max_positions,
            total_ut_steps=loops,
        )
    ).eval()


@torch.inference_mode()
def dense_reference(model, tokens):
    outputs, planes = functional_reference(model, torch.tensor(tokens), 4, return_kv=True)
    return [output[0] for output in outputs], planes


@pytest.mark.parametrize("interval", [1, 2, 4, 8])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("defer", [False, True])
@torch.inference_mode()
def test_readouts_and_all_committed_kv_match_dense(interval, kv_heads, defer):
    model = make_model(kv_heads)
    prompts = [[3, 5, 7], [11, 4, 9, 6, 8]]
    prompt_lengths = [len(prompt) for prompt in prompts]
    with DeferredOuroRunner(
        model, [32, 32], interval=interval, backend="torch", defer=defer
    ) as runner:
        hidden = runner.prefill(prompts)
        for index, prompt in enumerate(prompts):
            expected, _ = dense_reference(model, prompt)
            torch.testing.assert_close(hidden[index], expected[3][-1], atol=2e-5, rtol=2e-5)
        for step in range(9):
            tokens = [3 + step, 17 + step]
            hidden, depths = runner.step([0, 1], tokens)
            for index, token in enumerate(tokens):
                prompts[index].append(token)
                expected, planes = dense_reference(model, prompts[index])
                torch.testing.assert_close(
                    hidden[index], expected[depths[index] - 1][-1], atol=2e-5, rtol=2e-5
                )
                for depth in range(4):
                    for layer in range(model.config.num_hidden_layers):
                        actual = runner.cache.read(layer, runner.requests[index].request_id, depth)
                        committed = (
                            prompt_lengths[index] + (step + 1) // interval * interval
                            if defer and depth >= 2
                            else len(prompts[index])
                        )
                        for cached, reference in zip(actual, planes[depth][layer], strict=True):
                            assert len(cached) == committed
                            torch.testing.assert_close(
                                cached, reference[: len(cached)], atol=2e-5, rtol=2e-5
                            )
        assert depths == ([4, 4] if 9 % interval == 0 else [2, 2])
    assert runner.cache.num_free_blocks == runner.cache.num_blocks
    assert all(not request.pending_hidden and not request.live for request in runner.requests)


@torch.inference_mode()
def test_staggered_blocks_and_request_release():
    model = make_model()
    with DeferredOuroRunner(model, [32, 32], interval=4, backend="torch") as runner:
        runner.prefill([[3, 5], [7, 9, 11]])
        runner.step([0], [4])
        runner.step([0], [6])
        runner.step([0, 1], [8, 12])
        hidden, depths = runner.step([0, 1], [10, 14])
        assert depths == [4, 2]
        expected, _ = dense_reference(model, [3, 5, 4, 6, 8, 10])
        torch.testing.assert_close(hidden[0], expected[3][-1], atol=2e-5, rtol=2e-5)
        runner.release([0])
        runner.step([1], [16])
        hidden, depths = runner.step([1], [18])
        assert depths == [4]
        expected, _ = dense_reference(model, [7, 9, 11, 12, 14, 16, 18])
        torch.testing.assert_close(hidden[0], expected[3][-1], atol=2e-5, rtol=2e-5)


@torch.inference_mode()
def test_stop_tokens_processed_scores_and_batch_shrink(monkeypatch):
    model = make_model()
    samples = iter([2, 7, 8, 0])
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: torch.tensor([next(samples)]))
    outputs = generate_deferred(model, [[3, 5], [4, 6]], [42, 43], max_tokens=8, backend="torch")
    assert [output.token_ids for output in outputs] == [[2], [7, 8, 0]]
    assert [output.finish_reason for output in outputs] == ["stop", "stop"]
    assert [output.readout_depths for output in outputs] == [[4], [4, 2, 2]]
    for prompt, output in zip([[3, 5], [4, 6]], outputs, strict=True):
        for index, (token, depth) in enumerate(
            zip(output.token_ids, output.readout_depths, strict=True)
        ):
            levels, _ = dense_reference(model, prompt + output.token_ids[:index])
            expected = model.coda(levels[depth - 1][-1]).float().log_softmax(-1)[token]
            assert output.logprobs[index] == pytest.approx(float(expected), abs=2e-5)


@torch.inference_mode()
def test_length_limit_and_repeatable_rng():
    model = make_model()
    args = dict(
        prompts=[[3, 5], [4, 6]], seeds=[42, 43], max_tokens=7, stop_token_ids=(), backend="torch"
    )
    first = generate_deferred(model, **args)
    second = generate_deferred(model, **args)
    assert first == second
    assert all(len(output.token_ids) == 7 and output.finish_reason == "length" for output in first)
    assert all(output.readout_depths == [4, 2, 2, 2, 4, 2, 2] for output in first)


def test_invalid_configuration_and_step_are_rejected():
    with pytest.raises(ValueError, match="four-loop"):
        DeferredOuroRunner(make_model(loops=3), [16], backend="torch")
    with pytest.raises(ValueError, match="interval"):
        DeferredOuroRunner(make_model(), [16], interval=3, backend="torch")
    with DeferredOuroRunner(make_model(), [3], backend="torch") as runner:
        with pytest.raises(ValueError, match="Invalid request index"):
            runner.release([-1])
        assert runner.requests[0].live
        with pytest.raises(ValueError, match="after prefill"):
            runner.step([0], [1])
        runner.prefill([[3, 4, 5]])
        with pytest.raises(ValueError, match="capacity"):
            runner.step([0], [1])


@pytest.mark.parametrize("interval", [1, 2, 4, 8])
@torch.inference_mode()
def test_deferred_groups_deep_rows_without_reducing_committed_work(interval, monkeypatch):
    records = []
    for defer in (False, True):
        model = make_model()
        calls = []
        original = model.recurrent

        def record(hidden, request_ids, depths, positions, cache, **kwargs):
            calls.append((depths[0], len(request_ids)))
            return original(hidden, request_ids, depths, positions, cache, **kwargs)

        monkeypatch.setattr(model, "recurrent", record)
        with DeferredOuroRunner(model, [32, 32], interval=interval, defer=defer) as runner:
            runner.prefill([[3, 5], [7, 9]])
            calls.clear()
            for step in range(8):
                runner.step([0, 1], [3 + step, 15 + step])
        records.append(calls)
    for depth in (2, 3):
        eager = [rows for loop, rows in records[0] if loop == depth]
        deferred = [rows for loop, rows in records[1] if loop == depth]
        assert eager == [2] * 8
        assert deferred == [2 * interval] * (8 // interval)
        assert sum(eager) == sum(deferred) == 16


@torch.inference_mode()
def test_pending_states_own_only_their_request_rows():
    model = make_model()
    with DeferredOuroRunner(model, [16] * 12) as runner:
        runner.prefill([[3, 5]] * 12)
        runner.step(list(range(12)), [7] * 12)
        runner.release(range(1, 12))
        retained = runner.requests[0].pending_hidden[0]
        assert retained.untyped_storage().nbytes() == retained.numel() * retained.element_size()
        assert retained.shape == (1, model.config.hidden_size)


def test_recurrent_failure_releases_allocations(monkeypatch):
    model = make_model()
    original = model.recurrent

    def fail(hidden, request_ids, depths, positions, cache, **kwargs):
        if depths[0] == 2:
            raise RuntimeError("injected recurrent failure")
        return original(hidden, request_ids, depths, positions, cache, **kwargs)

    monkeypatch.setattr(model, "recurrent", fail)
    with DeferredOuroRunner(model, [16, 16]) as runner:
        with pytest.raises(RuntimeError, match="injected recurrent failure"):
            runner.prefill([[3, 5], [7, 9]])
        assert runner.cache.num_free_blocks == runner.cache.num_blocks
        assert all(not request.live and not request.pending_hidden for request in runner.requests)


def test_last_sample_needs_no_extra_kv_position():
    model = make_model(max_positions=16)
    output = generate_deferred(
        model, [[3, 5]], [42], max_tokens=15, stop_token_ids=(), backend="torch"
    )[0]
    assert len(output.token_ids) == 15
    assert output.finish_reason == "length"
    with pytest.raises(ValueError, match="position limit"):
        generate_deferred(model, [[3, 5]], [42], max_tokens=16, backend="torch")


def test_request_rng_is_independent_of_batch_members():
    model = make_model()
    settings = dict(max_tokens=7, stop_token_ids=(), backend="torch")
    grouped = generate_deferred(model, [[3, 5], [4, 6]], [42, 43], **settings)
    for prompt, seed, combined in zip([[3, 5], [4, 6]], [42, 43], grouped, strict=True):
        alone = generate_deferred(model, [prompt], [seed], **settings)[0]
        assert combined.token_ids == alone.token_ids
        assert combined.readout_depths == alone.readout_depths
        assert combined.finish_reason == alone.finish_reason
        torch.testing.assert_close(
            torch.tensor(combined.logprobs), torch.tensor(alone.logprobs), atol=2e-5, rtol=2e-5
        )
