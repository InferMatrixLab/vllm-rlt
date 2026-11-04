"""Engine e2e with a real-text prompt, for a trained checkpoint.

Synthetic token-id prompts drive a trained model into a fixed point (one token
repeated), which makes token agreement meaningless: every implementation agrees
when the argmax is constant. This encodes real text with the checkpoint's own
tokenizer and reports throughput, peak memory and per-request token agreement.

Usage:
    python benchmarks/sparse_decode_engine_e2e_text.py \
        --model-path /path/to/Ouro-1.4B --concurrency 1,8
"""

import argparse
import statistics
import time

import torch
from tokenizers import Tokenizer

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

TEXT = (
    "The history of computing is a history of abstraction. Each generation of "
    "hardware made a new kind of program possible, and each new kind of program "
    "demanded a new way of thinking about machines. When memory was scarce, "
    "programmers counted bytes by hand; when it became plentiful, they wrote "
    "interpreters that treated memory as infinite. The same shift is happening "
    "again with inference: models that once fit on a single device now span "
    "racks, and the scheduler rather than the kernel decides how fast a request "
    "finishes. Attention is the expensive part, because every token must look at "
    "every earlier token, and that cost grows with the square of the context. "
    "Sparse attention tries to keep only the keys that matter, which is easy to "
    "state and hard to make fast, because deciding what matters is itself as "
    "expensive as attending to everything."
)


def build(model, concurrency, sparse, blocks):
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


def run(model, concurrency, sparse, *, prompt_ids, max_tokens, repeats, blocks):
    walls = []
    peaks = []
    outputs = None
    for _ in range(repeats):
        engine = build(model, concurrency, sparse, blocks)
        torch.cuda.reset_peak_memory_stats()
        params = [
            SamplingParams(max_tokens=max_tokens, ignore_eos=True) for _ in range(concurrency)
        ]
        start = time.perf_counter()
        for index in range(concurrency):
            engine.add_request(f"r{index}", list(prompt_ids), params[index])
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
        "tokens_per_s": tokens / statistics.median(walls),
        "wall": statistics.median(walls),
        "peak_gib": max(peaks) / 2**30,
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", default="1,8")
    parser.add_argument("--blocks", type=int, default=768)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device="cuda", dtype=torch.bfloat16
    )
    tokenizer = Tokenizer.from_file(f"{args.model_path}/tokenizer.json")
    ids = tokenizer.encode(TEXT).ids
    while len(ids) < args.prompt:
        ids = ids + ids
    prompt_ids = ids[: args.prompt]
    print(f"real text prompt: {args.prompt} tokens, head={prompt_ids[:8]}")

    print(f"{'conc':>5}{'case':>16}{'tok/s':>10}{'wall ms':>10}{'peak GiB':>10}{'match':>10}")
    for concurrency in [int(v) for v in args.concurrency.split(",")]:
        dense = run(
            model,
            concurrency,
            None,
            prompt_ids=prompt_ids,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
            blocks=args.blocks,
        )
        print(
            f"{concurrency:>5}{'dense':>16}{dense['tokens_per_s']:>10.2f}"
            f"{dense['wall'] * 1e3:>10.1f}{dense['peak_gib']:>10.2f}{'--':>10}"
        )
        for retention in (1.0, 0.1):
            sparse = run(
                model,
                concurrency,
                SparseDecodeAttentionConfig(enabled=True, key_retention=retention),
                prompt_ids=prompt_ids,
                max_tokens=args.max_tokens,
                repeats=args.repeats,
                blocks=args.blocks,
            )
            same = sum(
                1
                for key, tokens in sparse["outputs"].items()
                if tokens == dense["outputs"].get(key)
            )
            print(
                f"{'':>5}{f'sparse r={retention}':>16}{sparse['tokens_per_s']:>10.2f}"
                f"{sparse['wall'] * 1e3:>10.1f}{sparse['peak_gib']:>10.2f}"
                f"{f'{same}/{concurrency}':>10}"
            )
        print(f"  dense sample tokens: {dense['outputs']['r0'][:8]}")


if __name__ == "__main__":
    main()
