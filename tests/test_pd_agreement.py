"""P->D agreement on a real checkpoint (RFC #111 M9 validation).

The handoff contract promises that moving prefill to P changes nothing the
request can observe: at equal forward geometry (same prefill chunk, one live
request), token ids, learned exit depths, finish reasons and D-owned sampling
RNG must match colocated execution exactly. The tiny-model tests in
test_pd.py pin these invariants with trace-driven depths; this harness pins
them where KV fidelity actually decides the outcome — a real checkpoint with
the learned exit policy, prompts crossing prefill-chunk and KV-block
boundaries. Bitwise equality under concurrent load is deliberately NOT
asserted: the colocated engine alone flips threshold decisions and seeded
samples when batch composition or chunk size changes (bf16 tiling ulps), so
that would test engine numerics, not the handoff.

Opt in with --run-gpu on a host with 3 free GPUs (P, D, reference). The
checkpoint defaults to ByteDance/Ouro-1.4B and is overridable through
RLT_AGREEMENT_MODEL.
"""

import os
from dataclasses import replace

import pytest
import torch

from tests.test_pd import drain
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, load_tokenizer
from vllm_rlt.pd.config import PDConfig

MODEL = os.environ.get("RLT_AGREEMENT_MODEL", "ByteDance/Ouro-1.4B")
# Lengths straddle the prefill chunk (64) and KV block (16) boundaries:
# sub-chunk, chunk+1, block-aligned multi-chunk, and a long many-chunk prompt.
PROMPT_LENGTHS = (19, 65, 256, 641)
GREEDY = SamplingParams(max_tokens=64, min_loops=1, ignore_eos=True)
_TEXT = (
    "The measurement of time began with the observation of regular motion: "
    "the turning sky, the shadow of a stick, water dripping from a marked "
    "bowl. Each device traded accuracy for convenience, and each improvement "
    "changed what people expected a day to contain. "
)


def prompts(tokenizer):
    # Natural text, trimmed to exact lengths: realistic hidden states so the
    # learned exit policy actually varies, identical ids on both engines so
    # any divergence comes from the handoff, not the inputs.
    ids = tokenizer(_TEXT * 40, add_special_tokens=False)["input_ids"]
    assert len(ids) >= max(PROMPT_LENGTHS)
    return [ids[:n] for n in PROMPT_LENGTHS]


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(MODEL)


@pytest.fixture(scope="module")
def reference_model():
    if torch.cuda.device_count() < 3:
        pytest.skip("requires 3 visible GPUs (P, D, reference)")
    return AutoModelForCausalLM.from_pretrained(MODEL, device="cuda:2", dtype=torch.bfloat16)


def create_real_pd(exit_config):
    pytest.importorskip("nixl")
    from vllm_rlt.pd.engine import PDEngine

    return PDEngine(
        MODEL,
        pd_config=PDConfig(
            prefill_devices=(0,),
            decode_devices=(1,),
            startup_timeout=600,
            request_timeout=300,
        ),
        prefill_cache_config=CacheConfig(4096, 16),
        decode_cache_config=CacheConfig(4096, 16),
        prefill_scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=64, prefill_chunk_size=64
        ),
        decode_scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=8),
        exit_config=exit_config,
        execution_config=ExecutionConfig(async_scheduling=True),
        attention_backend="triton",
        seed=0,
    )


def create_reference(model, exit_config):
    # Batch geometry differs from PD on purpose — agreement must not depend on
    # batch composition. The prefill chunk, however, matches PD's P exactly:
    # chunk size changes attention tiling by a few bf16 ulps, which greedy
    # argmax survives but the exit-threshold comparison does not (measured on
    # this checkpoint: chunk 256 vs 64 colocated flips one depth decision
    # while producing identical tokens). Agreement is therefore defined at
    # equal prefill geometry; KV fidelity does not imply geometry invariance.
    return LLMEngine(
        model,
        cache_config=CacheConfig(4096, 16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8, max_num_batched_tokens=512, prefill_chunk_size=64
        ),
        exit_config=exit_config,
        attention_backend="triton",
    )


def agree(pd, reference, requests):
    # One request at a time: with a single live request the forward batches on
    # P, D and the reference are identical, so any divergence is handoff
    # evidence. Bitwise agreement under concurrent load is not a handoff
    # property — see test_pd_mixed_batch_completes.
    for request_id, prompt, params in requests:
        for target in (pd, reference):
            target.add_request(request_id, prompt, params)
        got, want = drain(pd), drain(reference)
        a, b = got[request_id], want[request_id]
        assert a.token_ids == b.token_ids, request_id
        assert a.exit_depths == b.exit_depths, request_id
        assert (a.finished, a.finish_reason) == (b.finished, b.finish_reason), request_id


@pytest.mark.gpu
def test_pd_agreement_fixed_depth_greedy(reference_model, tokenizer):
    exit_config = ExitConfig("ouro_delayed")
    with create_real_pd(exit_config) as pd:
        reference = create_reference(reference_model, exit_config)
        inputs = prompts(tokenizer)
        agree(pd, reference, [(f"g{n}", p, GREEDY) for n, p in zip(PROMPT_LENGTHS, inputs)])


@pytest.mark.gpu
def test_pd_agreement_learned_exit_and_seeded_sampling(reference_model, tokenizer):
    # exit_threshold < 1 engages the learned (delayed) cumulative-hazard policy, so the
    # per-token depths are decided by hidden states built on transferred KV.
    exit_config = ExitConfig("ouro_delayed")
    adaptive = replace(GREEDY, exit_threshold=0.5, min_loops=1)
    seeded = replace(adaptive, temperature=0.7, top_k=32, seed=947)
    with create_real_pd(exit_config) as pd:
        reference = create_reference(reference_model, exit_config)
        inputs = prompts(tokenizer)
        agree(
            pd,
            reference,
            [(f"a{n}", p, adaptive) for n, p in zip(PROMPT_LENGTHS, inputs)]
            + [("seeded", inputs[2], seeded)],
        )


@pytest.mark.gpu
def test_pd_mixed_batch_completes(reference_model, tokenizer):
    # Under concurrent load, bitwise agreement with a colocated engine is not
    # a handoff property: measured on this checkpoint, the colocated engine
    # alone flips a seeded sample with batch composition and an exit decision
    # with prefill chunk size (bf16 tiling moves logits a few ulps; argmax
    # usually survives, threshold comparisons and multinomial do not). The
    # mixed-batch PD run is therefore held to liveness and shape invariants,
    # not bitwise equality.
    exit_config = ExitConfig("ouro_delayed")
    adaptive = replace(GREEDY, exit_threshold=0.5, min_loops=1)
    with create_real_pd(exit_config) as pd:
        inputs = prompts(tokenizer)
        for n, p in zip(PROMPT_LENGTHS, inputs):
            pd.add_request(f"g{n}", p, GREEDY)
            pd.add_request(f"a{n}", p, adaptive)
        outputs = drain(pd)
        assert len(outputs) == 2 * len(PROMPT_LENGTHS)
        max_depth = reference_model.config.total_ut_steps
        for o in outputs.values():
            assert len(o.token_ids) == GREEDY.max_tokens, o.request_id
            assert len(o.exit_depths) == len(o.token_ids), o.request_id
            assert all(1 <= d <= max_depth for d in o.exit_depths), o.request_id


@pytest.mark.gpu
def test_pd_agreement_depths_vary(reference_model, tokenizer):
    exit_config = ExitConfig("ouro_delayed")
    adaptive = replace(GREEDY, exit_threshold=0.5, min_loops=1, max_tokens=32)
    reference = create_reference(reference_model, exit_config)
    inputs = prompts(tokenizer)
    reference.add_request("vary", inputs[1], adaptive)
    depths = {d for o in drain(reference).values() for d in o.exit_depths}
    assert len(depths) > 1, "learned exit never varied; agreement test is vacuous"
