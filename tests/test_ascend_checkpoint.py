"""Pinned real-weight NPU checks for explicit Torch attention, not fused CANN."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from tests.test_ascend import assert_bf16_close, recurrent_states
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM, load_tokenizer

pytestmark = pytest.mark.npu


@pytest.fixture(scope="module")
def checkpoint():
    source = os.environ.get("VLLM_RLT_OURO_CHECKPOINT")
    if not source:
        pytest.skip("set VLLM_RLT_OURO_CHECKPOINT to the pinned local Ouro-1.4B checkpoint")
    resolve_device("npu")
    torch.npu.set_device(0)
    assert torch.npu.get_device_name(0) == "Ascend910B3"
    with (Path(source) / "model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
    model = OuroForCausalLM.from_pretrained(source, device="npu", dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    yield model, load_tokenizer(source)
    torch.npu.synchronize()


def save_record(name, record):
    folder = os.environ.get("VLLM_RLT_ASCEND_RESULTS")
    if folder:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(json.dumps(record, indent=2))


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
    save_record(
        "checkpoint-numerics.json",
        dict(
            tokens=tokens,
            states=len(pairs),
            exact=all(torch.equal(a, b) for a, b in pairs.values()),
            finite=all(
                bool(torch.isfinite(a).all() and torch.isfinite(b).all()) for a, b in pairs.values()
            ),
        ),
    )
    for a, b in pairs.values():
        assert_bf16_close(a, b)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@torch.inference_mode()
def test_checkpoint_greedy_generation_and_cleanup(checkpoint):
    model, tokenizer = checkpoint
    engine = StageEngine(
        model,
        attention_backend="torch",
        cache_config=CacheConfig(num_blocks=64),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=32),
    )
    prompts = ["The capital of France is", "One plus one equals"]
    finished = {}
    try:
        for index, prompt in enumerate(prompts):
            engine.add_request(
                str(index),
                tokenizer.encode(prompt, add_special_tokens=False),
                SamplingParams(max_tokens=4, temperature=0, ignore_eos=True),
            )
        for _ in range(200):
            for output in engine.step():
                if output.finished:
                    assert output.exit_depths == [4] * 4
                    assert output.finish_reason == "length"
                    finished[output.request_id] = dict(
                        tokens=output.token_ids, text=tokenizer.decode(output.token_ids)
                    )
            if not engine.has_unfinished_requests():
                break
        assert set(finished) == {"0", "1"}
        assert finished["0"]["tokens"] == [7042, 30, 198, 198]
        assert finished["1"]["tokens"] == [827, 30, 198, 198]
        assert engine.cache_manager.num_used_blocks == 0
        save_record("checkpoint-generation.json", finished)
    finally:
        engine.close()
