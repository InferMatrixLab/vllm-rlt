# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint BF16 agreement for Dory; needs the pinned official snapshot.

RFC #176 PR 1 requires evidence on the actual weights, not only tiny random
models. Set ``DORY_CHECKPOINT`` to the pinned snapshot directory and run on one
CUDA device:

    DORY_CHECKPOINT=/path/to/dory-4b CUDA_VISIBLE_DEVICES=0 \
        python -m pytest tests/test_dory_real_checkpoint.py -q --run-gpu -s

The GPU side runs the staged reference schedule on the Triton backend in BF16;
the CPU side runs the independent dense oracle in FP32. BF16 and FP32 do not
produce bitwise-identical logits, so the assertions compare decision-relevant
quantities (top-1, top-5 overlap, the largest deviation among the top logits,
and the generated tokens) instead of assuming equality.
"""

import gc
import os

import pytest
import torch

from tests.reference.dory import dense_dory_reference
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import DoryForCausalLM, load_tokenizer

CHECKPOINT = os.environ.get("DORY_CHECKPOINT")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not CHECKPOINT, reason="set DORY_CHECKPOINT to the pinned Dory snapshot directory"
    ),
]

PROMPT_TEXT = "The capital of France is"
LOOP_COUNTS = (1, 2, 4)
GENERATED_TOKENS = 4
# BF16 (GPU) versus FP32 (CPU) logits: 4 significant digits, ~15-wide logit
# spread on this checkpoint. Observed margins from this run are reported below;
# these bounds leave room for backend arithmetic without hiding a wrong model.
TOP_K = 20
# BF16 (GPU) versus FP32 (CPU) logits on this checkpoint: measured 0.0422,
# 0.0656 and 0.0712 at 1, 2 and 4 loops with 20/20 top-20 overlap. The bound
# leaves ~3x margin for backend arithmetic without hiding a wrong model.
MAX_TOP_DEVIATION = 0.25


def load(device, dtype, loops):
    return DoryForCausalLM.from_pretrained(CHECKPOINT, device=device, dtype=dtype, loops=loops)


def staged_logits(model, tokens, backend="triton"):
    """Reference schedule: one token per step, staged prelude/core/coda."""
    config = model.config
    device = next(model.parameters()).device
    cache = KVCacheManager(
        num_layers=config.num_kv_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=64,
        block_size=16,
        max_loops=config.total_ut_steps,
        device=device,
        dtype=next(model.parameters()).dtype,
        backend=backend,
        layout=config.kv_cache_layout,
        recurrent_layers=model.recurrent_kv_layers,
    )
    ids = ["request"] * len(tokens)
    assert cache.allocate("request", len(tokens))
    logits = []
    for position in range(len(tokens)):
        boundary = cache._prepare_batch(ids[:1], [0], [position])
        hidden = model.prelude_prepared(tokens[position : position + 1].to(device), boundary, cache)
        for depth in range(config.total_ut_steps):
            core = cache._prepare_batch(ids[:1], [depth], [position])
            hidden, _ = model.recurrent_prepared(hidden, core, cache)
        logits.append(model.coda_prepared(hidden, boundary, cache)[0].float())
    cache.free("request")
    assert cache.num_used_blocks == 0
    return torch.stack(logits)


def oracle_logits(model, tokens):
    return dense_dory_reference(model, tokens)["logits"]


def greedy_from_logits(model, tokens, count, device=None):
    tokens = list(tokens)
    for _ in range(count):
        step = torch.tensor(tokens)
        if device is not None:
            logits = staged_logits(model, step, backend="triton")[-1]
        else:
            logits = oracle_logits(model, step)[-1]
        tokens.append(int(logits.argmax()))
    return tokens[len(tokens) - count :]


def prompt_tokens():
    tokenizer = load_tokenizer(CHECKPOINT)
    return tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids[0], tokenizer


def wide_cpu_threads():
    """The FP32 oracle is CPU-bound and the session fixture pins one thread."""
    torch.set_num_threads(min(32, os.cpu_count() or 1))


def report(loops, gpu, cpu, tokens):
    gpu_logits = staged_logits(gpu, tokens)
    cpu_logits = oracle_logits(cpu, tokens)
    top_gpu = torch.topk(gpu_logits[-1], TOP_K)
    top_cpu_ids = torch.topk(cpu_logits[-1], TOP_K).indices
    overlap = len(set(top_gpu.indices.tolist()) & set(top_cpu_ids.tolist()))
    reference_ids = top_cpu_ids.to(gpu_logits.device)
    gpu_top = gpu_logits[-1][reference_ids].float().cpu()
    cpu_top = cpu_logits[-1][top_cpu_ids].float()
    deviation = (gpu_top - cpu_top).abs().max().item()
    gpu_greedy = greedy_from_logits(gpu, tokens.tolist(), GENERATED_TOKENS, device="cuda")
    cpu_greedy = greedy_from_logits(cpu, tokens.tolist(), GENERATED_TOKENS)
    print(
        f"loops={loops} tokens={len(tokens)} "
        f"top1_equal={int(top_gpu.indices[0]) == int(top_cpu_ids[0])} "
        f"top{TOP_K}_overlap={overlap}/{TOP_K} "
        f"max|dlogit|={deviation:.4f} "
        f"greedy_gpu={gpu_greedy} greedy_cpu={cpu_greedy}",
        flush=True,
    )
    return {
        "gpu_logits": gpu_logits,
        "cpu_logits": cpu_logits,
        "top1_equal": int(top_gpu.indices[0]) == int(top_cpu_ids[0]),
        "overlap": overlap,
        "deviation": deviation,
        "gpu_greedy": gpu_greedy,
        "cpu_greedy": cpu_greedy,
    }


def test_real_checkpoint_bf16_matches_fp32_reference_in_both_directions():
    wide_cpu_threads()
    tokens, _ = prompt_tokens()
    results = {}
    for loops in LOOP_COUNTS:
        gpu = load("cuda", torch.bfloat16, loops)
        cpu = load("cpu", torch.float32, loops)
        try:
            results[loops] = report(loops, gpu, cpu, tokens)
        finally:
            del gpu, cpu
            gc.collect()
            torch.cuda.empty_cache()

    for loops, result in results.items():
        assert result["top1_equal"], f"loops={loops}: reference top-1 disagrees"
        assert result["overlap"] >= TOP_K - 2, f"loops={loops}: top-{TOP_K} overlap too small"
        assert result["deviation"] <= MAX_TOP_DEVIATION, f"loops={loops}: logit deviation"
        assert result["gpu_greedy"] == result["cpu_greedy"], f"loops={loops}: greedy mismatch"

    # A loop count that never reaches the model would show identical outputs.
    assert not torch.allclose(
        results[1]["cpu_logits"][-1], results[4]["cpu_logits"][-1], atol=1e-3
    ), "one and four loops produced the same reference logits"


@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 5e-3), (torch.bfloat16, 0.25)])
def test_real_checkpoint_backend_parity_on_gpu(dtype, tolerance):
    """Triton and Torch kernels must agree; BF16 rounding stacks on top."""
    tokens, _ = prompt_tokens()
    model = load("cuda", dtype, 2)
    try:
        triton = staged_logits(model, tokens, backend="triton")
        torch_backend = staged_logits(model, tokens, backend="torch")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    difference = (triton - torch_backend).abs().max().item()
    print(f"backend parity ({dtype}): max|dlogit|={difference:.6f}", flush=True)
    assert difference <= tolerance, difference


def test_real_checkpoint_generation_stops_on_eos_or_limit():
    tokens, tokenizer = prompt_tokens()
    model = load("cuda", torch.bfloat16, 2)
    eos_ids = tuple(int(value) for value in model.config.eos_token_id)
    try:
        engine = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=512, block_size=16),
            scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=16),
        )
        engine.add_request(
            "request",
            tokens.tolist(),
            SamplingParams(max_tokens=GENERATED_TOKENS, min_loops=2, max_loops=2),
        )
        finished = None
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    finished = output
        assert finished is not None
        print(
            f"generated={finished.token_ids} reason={finished.finish_reason} "
            f"exit_depths={finished.exit_depths}",
            flush=True,
        )
        assert finished.exit_depths == [2] * len(finished.token_ids)
        assert engine.cache_manager.num_used_blocks == 0
        if finished.finish_reason == "stop":
            assert finished.token_ids[-1] in eos_ids
            assert tokenizer.decode(finished.token_ids) is not None
        else:
            assert finished.finish_reason == "length"
            assert len(finished.token_ids) == GENERATED_TOKENS
    finally:
        del model
        torch.cuda.empty_cache()
