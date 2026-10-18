"""Matched synchronous baseline/pipeline feasibility check on a local Ouro model.

Exclude prefill, interleave trials, and retain raw timings and generated tokens.
This small experiment is not the full RFC #43 accuracy/performance acceptance matrix.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM, load_tokenizer
from vllm_rlt.worker.speculative import SpeculativeStats


@torch.inference_mode()
def trial(engine, prompts, new_tokens, temperature, top_p, top_k):
    params = SamplingParams(
        max_tokens=new_tokens + 1,
        ignore_eos=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, params)
    # Hold decode until all requests have crossed the first-output barrier.
    while any(not r.generated_token_ids for r in engine.scheduler.requests.values()):
        engine.step()
    engine.speculative_runner.stats = SpeculativeStats()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output.token_ids
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    committed = len(prompts) * new_tokens
    stats = asdict(engine.speculative_runner.stats)
    if stats["committed_tokens"] != committed:
        raise RuntimeError(
            f"benchmark committed {stats['committed_tokens']} tokens, expected {committed}; "
            f"output lengths={[len(outputs[str(i)]) for i in range(len(prompts))]}"
        )
    return dict(
        seconds=elapsed,
        committed_tokens=committed,
        tokens_per_second=committed / elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        stats=stats,
        tokens=[outputs[str(i)] for i in range(len(prompts))],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Existing local checkpoint")
    parser.add_argument("--prompts", type=Path, help="JSON list of prompt strings")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["triton", "flash_attn_4"], default="triton")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--draft-loops", type=int, default=2)
    parser.add_argument(
        "--early-rejection-pruning",
        action="store_true",
        help="Enable greedy early rejection pruning for the pipeline engine",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--cuda-graph-max-batch-size", type=int, default=128)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be a complete local checkpoint directory")
    if min(args.concurrency, args.new_tokens, args.repeats, args.k) < 1:
        parser.error("concurrency, new-tokens, repeats and k must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be nonnegative and top-p must be in (0, 1]")
    if args.top_k == 0 or args.top_k < -1:
        parser.error("top-k must be -1 or positive")
    if args.cuda_graph_max_batch_size < 1:
        parser.error("cuda-graph-max-batch-size must be positive")
    torch.set_num_threads(1)
    texts = (
        json.loads(args.prompts.read_text())
        if args.prompts
        else [
            "Explain why the sky appears blue during the day.",
            "Write a Python function that returns the Fibonacci sequence.",
        ]
    )
    if not texts or not all(isinstance(t, str) and t for t in texts):
        parser.error("prompts must be a nonempty JSON list of nonempty strings")
    tokenizer = load_tokenizer(str(args.model))
    prompts = [tokenizer.encode(texts[i % len(texts)]) for i in range(args.concurrency)]
    model = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    blocks = model.config.total_ut_steps * sum(
        (len(p) + args.new_tokens + 15) // 16 for p in prompts
    )
    scheduler = SchedulerConfig(
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=max(
            sum(map(len, prompts)),
            args.concurrency * max(map(len, prompts)),
            args.concurrency * (args.k + 1),
        ),
        prefill_chunk_size=max(map(len, prompts)),
        max_prefill_batches_before_decode=1024,
    )
    execution = ExecutionConfig(
        cuda_graphs=args.cuda_graphs,
        cuda_graph_max_batch_size=args.cuda_graph_max_batch_size,
    )
    engines = {
        name: LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=blocks, block_size=16),
            scheduler_config=scheduler,
            execution_config=execution,
            attention_backend=args.backend,
            speculative_config=SpeculativeConfig(
                num_speculative_tokens=args.k,
                draft_loops=args.draft_loops,
                target_loops=model.config.total_ut_steps,
                loop_level_pipelining=enabled,
                early_rejection_pruning=enabled and args.early_rejection_pruning,
            ),
        )
        for name, enabled in [("baseline", False), ("pipeline", True)]
    }
    raw = {name: [] for name in engines}
    try:
        for engine in engines.values():
            trial(
                engine,
                prompts,
                args.new_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
            )
        for repeat in range(args.repeats):
            order = list(engines) if repeat % 2 == 0 else list(reversed(engines))
            for name in order:
                result = trial(
                    engines[name],
                    prompts,
                    args.new_tokens,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                )
                raw[name].append(result)
                print(
                    f"{name} repeat={repeat}: {result['tokens_per_second']:.2f} tokens/s",
                    flush=True,
                )
    finally:
        for engine in engines.values():
            engine.close()
    summary = {
        name: dict(
            median_seconds=statistics.median(r["seconds"] for r in trials),
            min_seconds=min(r["seconds"] for r in trials),
            max_seconds=max(r["seconds"] for r in trials),
        )
        for name, trials in raw.items()
    }
    agreement = []
    for baseline, pipeline in zip(raw["baseline"], raw["pipeline"]):
        agreement.append(
            [
                next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
                for left, right in zip(baseline["tokens"], pipeline["tokens"])
            ]
        )
    report = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        tracked_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        dtype="bfloat16",
        checkpoint_config_sha256=hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        prompt_token_ids=prompts,
        scheduler=asdict(scheduler),
        execution=asdict(execution),
        summary=summary,
        speedup=summary["baseline"]["median_seconds"] / summary["pipeline"]["median_seconds"],
        first_divergent_output_positions=agreement,
        trials=raw,
        limitations="Small greedy feasibility check; not BF16 acceptance validation. "
        "Peak memory includes both resident engines. Checkpoint config hash does not hash weights.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(dict(summary=summary, speedup=report["speedup"], divergence=agreement), indent=2)
    )


if __name__ == "__main__":
    main()
