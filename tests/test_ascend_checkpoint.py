"""Opt-in local-checkpoint KV and lifecycle regressions on Ascend."""

import os
from pathlib import Path

import pytest
import torch

from tests.test_ascend import assert_bf16_close, recurrent_states
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM, load_tokenizer
from vllm_rlt.request import RequestPhase

pytestmark = pytest.mark.npu


@pytest.fixture(scope="module")
def checkpoint():
    source = os.environ.get("VLLM_RLT_OURO_CHECKPOINT")
    if not source:
        pytest.skip("set VLLM_RLT_OURO_CHECKPOINT to a prepared local Ouro checkpoint")
    if not Path(source).is_dir():
        pytest.fail("VLLM_RLT_OURO_CHECKPOINT must be a prepared local directory")
    resolve_device("npu")
    torch.npu.set_device(0)
    model = OuroForCausalLM.from_pretrained(source, device="npu", dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    yield model, load_tokenizer(source)
    torch.npu.synchronize()


@torch.inference_mode()
def test_checkpoint_incremental_pages_preserve_hidden_logits_and_kv(checkpoint):
    model, tokenizer = checkpoint
    tokens = tokenizer.encode(
        "The capital of France is Paris. The capital of Japan is Tokyo. "
        "A language model predicts the next token in a sequence.",
        add_special_tokens=False,
    )[:21]
    assert len(tokens) == 21
    expected, expected_kv = recurrent_states(model, [19, 1, 1], tokens=tokens)
    actual, actual_kv = recurrent_states(model, [19, 1, 1], tokens=tokens, incremental=True)
    pairs = {}
    for key in expected:
        for name, a, b in zip(("hidden", "logits"), actual[key], expected[key], strict=True):
            pairs[f"{key}.{name}"] = (a, b)
    for key in expected_kv:
        for name, a, b in zip(("key", "value"), actual_kv[key], expected_kv[key], strict=True):
            pairs[f"{key}.{name}"] = (a, b)
    for a, b in pairs.values():
        assert_bf16_close(a, b)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("chunk_size", [127, 128])
@torch.inference_mode()
def test_checkpoint_chunked_prefill_packed_matches_per_query(checkpoint, chunk_size):
    model, tokenizer = checkpoint
    prompt = tokenizer.encode(
        "The capital of France is Paris. The capital of Japan is Tokyo. " * 20,
        add_special_tokens=False,
    )[:145]
    assert len(prompt) == 145
    expected_chunks = [(0, chunk_size), (chunk_size, len(prompt) - chunk_size)]
    loops = model.config.total_ut_steps
    params = SamplingParams(
        max_tokens=4, temperature=0, min_loops=loops, max_loops=loops, ignore_eos=True
    )

    def run(packed_prefill):
        engine = StageEngine(
            model,
            attention_backend="ascend_eager",
            cache_config=CacheConfig(num_blocks=64, block_size=16),
            scheduler_config=SchedulerConfig(
                max_num_seqs=1, max_num_batched_tokens=len(prompt), prefill_chunk_size=chunk_size
            ),
            execution_config=ExecutionConfig(packed_prefill=packed_prefill, use_uva=False),
        )
        logits, chunks = [], []

        def capture_logits(_module, _inputs, output):
            logits.append(output[0].detach().cpu())

        try:
            with model.lm_head.register_forward_hook(capture_logits):
                engine.add_request("chunked", prompt, params)
                for _ in range(64):
                    outputs = engine.step()
                    batch = engine.last_schedule
                    if batch is not None and batch.phase == RequestPhase.PREFILL:
                        chunks.extend(
                            (item.token_start, item.token_count) for item in batch.requests
                        )
                    for output in outputs:
                        if output.finished:
                            assert chunks == expected_chunks
                            assert output.finish_reason == "length"
                            assert output.exit_depths == [loops] * params.max_tokens
                            assert len(logits) == params.max_tokens
                            assert engine.cache_manager.num_used_blocks == 0
                            return output.token_ids, torch.stack(logits)
                pytest.fail("chunked request did not finish")
        finally:
            engine.close()

    # Hold the projection shapes fixed so this checks packing and cached KV,
    # independent of BF16 differences across different Linear row counts.
    expected_ids, expected_logits = run(False)
    actual_ids, actual_logits = run(True)
    assert actual_ids == expected_ids
    torch.testing.assert_close(actual_logits, expected_logits, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["torch", "ascend_eager"])
@torch.inference_mode()
def test_checkpoint_greedy_replay_and_request_cleanup(checkpoint, backend):
    model, tokenizer = checkpoint
    engine = StageEngine(
        model,
        attention_backend=backend,
        cache_config=CacheConfig(num_blocks=64),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=32),
    )
    prompts = ["The capital of France is", "One plus one equals"]
    loops = model.config.total_ut_steps
    params = SamplingParams(
        max_tokens=4, temperature=0, min_loops=loops, max_loops=loops, ignore_eos=True
    )
    expected = None
    try:
        # Repeat the same packed workload and reuse IDs after all pages are freed.
        # Expected IDs come from this checkpoint, not a particular weight file.
        for _ in range(2):
            finished = {}
            for index, prompt in enumerate(prompts):
                engine.add_request(
                    str(index), tokenizer.encode(prompt, add_special_tokens=False), params
                )
            for _ in range(200):
                for output in engine.step():
                    if output.finished:
                        assert len(output.token_ids) == params.max_tokens
                        assert all(
                            0 <= token < model.config.vocab_size for token in output.token_ids
                        )
                        assert output.exit_depths == [loops] * params.max_tokens
                        assert output.finish_reason == "length"
                        finished[output.request_id] = list(output.token_ids)
                if not engine.has_unfinished_requests():
                    break
            assert set(finished) == {str(index) for index in range(len(prompts))}
            assert engine.cache_manager.num_used_blocks == 0
            if expected is None:
                expected = finished
            else:
                assert finished == expected
    finally:
        engine.close()
