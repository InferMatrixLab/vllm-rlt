"""Controlled synthetic CDB replay benchmark; no pretrained quality claims.

Run: python -m benchmarks.cdb_runtime --device cpu --output /tmp/cdb.json
CUDA: set CUDA_VISIBLE_DEVICES and pass --device cuda. Model/data are local.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM


def drive(engine):
    outputs, steps = {}, []
    start = time.perf_counter()
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
        batch = engine.last_schedule
        if batch is not None:
            steps.append(
                dict(
                    stage=batch.phase.value,
                    effective_rows=batch.num_tokens,
                    submitted_rows=engine.model_runner.last_submitted_size,
                )
            )
    engine.model_runner.synchronize()
    return outputs, steps, time.perf_counter() - start


def _toy_ouro_config() -> OuroConfig:
    return OuroConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        total_ut_steps=4,
        bos_token_id=0,
        eos_token_id=0,
    )


def benchmark(device="cpu", layout="last_exited", trace=None, seed=123):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = OuroForCausalLM(_toy_ouro_config()).to(device=device, dtype=dtype)
    prompts = {"A": [2, 3, 4, 5], "B": [6, 7], "C": [8, 9, 10]}
    parameters = {
        rid: SamplingParams(max_tokens=6, exit_threshold=q, ignore_eos=True)
        for rid, q in zip(prompts, [0.0, 1.0, 0.5])
    }
    fingerprint = dict(
        model_config=model.config.to_dict(),
        seed=seed,
        dtype=str(dtype),
        layout=layout,
        prompts=prompts,
        sampling={rid: asdict(p) for rid, p in parameters.items()},
    )
    backend = "triton" if device == "cuda" else "torch"

    def make(exits, mode="refill", execution=None):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(128, 2, layout),
            exit_config=exits,
            execution_config=execution,
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=3, prefill_chunk_size=2, mode=mode
            ),
            attention_backend=backend,
        )
        for rid, prompt in prompts.items():
            engine.add_request(rid, prompt, parameters[rid])
        return engine

    if trace is None:
        outputs, _, _ = drive(make(ExitConfig("random_lookahead", seed=7)))
        trace = dict(
            fingerprint=fingerprint,
            depths_by_request={rid: o.exit_depths for rid, o in outputs.items()},
            token_ids={rid: o.token_ids for rid, o in outputs.items()},
        )
    if trace["fingerprint"] != fingerprint:
        raise ValueError("trace model, dtype, KV layout or workload does not match")
    exits = ExitConfig("trace", depths_by_request=trace["depths_by_request"])
    results = []
    for mode in ("no_refill", "refill"):
        for label, execution in (
            ("sync_eager", ExecutionConfig()),
            ("sync_padded", ExecutionConfig(static_buffers=True, pad_to_power_of_two=True)),
            (
                "async_single",
                ExecutionConfig(
                    async_scheduling=True,
                    multi_stream=False,
                    static_buffers=True,
                    pad_to_power_of_two=True,
                ),
            ),
            (
                "async_multi",
                ExecutionConfig(
                    async_scheduling=True, static_buffers=True, pad_to_power_of_two=True
                ),
            ),
        ):
            # Warm up each variant separately; initialization is outside timing.
            drive(make(exits, mode, execution))
            engine = make(exits, mode, execution)
            if device == "cuda":
                torch.cuda.synchronize()
            outputs, steps, elapsed = drive(engine)
            actual_tokens = {rid: o.token_ids for rid, o in outputs.items()}
            actual_depths = {rid: o.exit_depths for rid, o in outputs.items()}
            if actual_depths != trace["depths_by_request"] or actual_tokens != trace["token_ids"]:
                raise AssertionError(f"replay mismatch in {mode}/{label}")
            results.append(
                dict(
                    mode=mode,
                    execution=label,
                    seconds=elapsed,
                    tokens_per_second=sum(len(o.token_ids) for o in outputs.values()) / elapsed,
                    steps=steps,
                    kv_pool_bytes=engine.cache_manager.num_blocks
                    * engine.cache_manager.bytes_per_block,
                )
            )
    return trace, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--layout", choices=["last_exited", "shared"], default="last_exited")
    parser.add_argument("--trace-in", type=Path)
    parser.add_argument("--trace-out", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace_in.read_text()) if args.trace_in else None
    trace, results = benchmark(args.device, args.layout, trace)
    if args.trace_out:
        args.trace_out.write_text(json.dumps(trace, indent=2))
    args.output.write_text(
        json.dumps(
            dict(
                trace=trace, results=results, note="synthetic tiny-model replay; not paper speedups"
            ),
            indent=2,
        )
    )
    for row in results:
        print(row["mode"], row["execution"], f"{row['tokens_per_second']:.1f} tokens/s")


if __name__ == "__main__":
    main()
