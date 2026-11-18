"""Engine-level end-to-end comparison for sparse decode attention.

Runs the real StageEngine with asynchronous self-speculation and compares dense
against sparse decode at full and reduced retention: throughput, peak memory and
token agreement. The per-layer sweeps say what one attention call costs; this
says whether that survives scheduling, sampling and delivery.

The tiny Ouro architecture is used because no trained checkpoint is available on
this host, so the numbers are a scheduler-level check, not a serving result.
"""

import argparse
import statistics
import time

import torch

from tests.helpers import tiny_ouro_model
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SparseDecodeAttentionConfig,
    SpeculativeConfig,
)
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import AutoModelForCausalLM


def build(model, concurrency, sparse, blocks=2048):
    return StageEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(blocks, 16, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=concurrency,
            max_num_batched_tokens=8 * concurrency,
            prefill_chunk_size=16,
        ),
        execution_config=ExecutionConfig(async_scheduling=True, multi_stream=True),
        speculative_config=SpeculativeConfig(3),
        sparse_decode_attention_config=sparse,
    )


def run(model, concurrency, sparse, *, prompt, max_tokens, repeats, blocks=2048):
    """One configuration: returns median wall time, tokens/s and peak bytes."""
    outputs = None
    walls = []
    peaks = []
    for _ in range(repeats):
        engine = build(model, concurrency, sparse, blocks)
        torch.cuda.reset_peak_memory_stats()
        params = [
            SamplingParams(max_tokens=max_tokens, ignore_eos=True) for _ in range(concurrency)
        ]
        prompts = [
            [2 + (index + position) % 60 for position in range(prompt)]
            for index in range(concurrency)
        ]
        start = time.perf_counter()
        for index in range(concurrency):
            engine.add_request(f"r{index}", prompts[index], params[index])
        collected = {}
        for _ in range(100000):
            for output in engine.step():
                collected[output.request_id] = output
            if len(collected) == concurrency and all(o.finished for o in collected.values()):
                break
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - start)
        peaks.append(torch.cuda.max_memory_allocated())
        outputs = {key: list(value.token_ids) for key, value in collected.items()}
    tokens = concurrency * max_tokens
    return {
        "wall": statistics.median(walls),
        "tokens_per_s": tokens / statistics.median(walls),
        "peak_gib": max(peaks) / 2**30,
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", default="1,8,32")
    parser.add_argument("--retentions", default="1.0,0.1")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--cache-blocks", type=int, default=2048)
    parser.add_argument("--sparse-context", type=int, default=64)
    args = parser.parse_args()

    device = args.device
    if args.model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device=device, dtype=torch.bfloat16
        )
    else:
        model = tiny_ouro_model(device)
    concurrency_levels = [int(value) for value in args.concurrency.split(",")]
    retentions = [float(value) for value in args.retentions.split(",")]

    label = args.model_path or "tiny Ouro"
    print(
        f"engine e2e — {label} ({model.config.num_hidden_layers}L, ctx "
        f"{model.config.max_position_embeddings}), prompt {args.prompt}, "
        f"max_tokens {args.max_tokens}, median of {args.repeats}"
    )
    header = f"{'conc':>5}{'case':>14}{'tok/s':>10}{'wall ms':>10}{'peak GiB':>10}{'match':>10}"
    print(header)
    for concurrency in concurrency_levels:
        dense = run(
            model,
            concurrency,
            None,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
            blocks=args.cache_blocks,
        )
        print(
            f"{concurrency:>5}{'dense':>14}{dense['tokens_per_s']:>10.2f}"
            f"{dense['wall'] * 1e3:>10.1f}{dense['peak_gib']:>10.2f}{'--':>10}"
        )
        for retention in retentions:
            sparse = run(
                model,
                concurrency,
                SparseDecodeAttentionConfig(
                    enabled=True,
                    key_retention=retention,
                    max_context_tokens=args.sparse_context,
                ),
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                repeats=args.repeats,
                blocks=args.cache_blocks,
            )
            same = sum(
                1
                for key, tokens in sparse["outputs"].items()
                if tokens == dense["outputs"].get(key)
            )
            print(
                f"{'':>5}{f'sparse r={retention}':>14}{sparse['tokens_per_s']:>10.2f}"
                f"{sparse['wall'] * 1e3:>10.1f}{sparse['peak_gib']:>10.2f}"
                f"{f'{same}/{concurrency}':>10}"
            )


if __name__ == "__main__":
    main()
