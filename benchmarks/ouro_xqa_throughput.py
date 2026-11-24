"""Matched Ouro-1.4B BF16 eager throughput with per-row prefill on Triton and XQA."""

import argparse
import gc
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch

from benchmarks.preemption import checkpoint_identity
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM

PROMPT, OUTPUT, PAGE, LOOPS = 512, 128, 16, 4


def drive(engine, prompts, concurrency, params):
    """Time initial admission, continuous replacement, prefill and final drain."""
    pending = iter(enumerate(prompts))
    completed = {}

    def admit_next():
        item = next(pending, None)
        if item is not None:
            index, ids = item
            engine.add_request(str(index), ids, params)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(concurrency):
        admit_next()
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                completed[output.request_id] = output
                admit_next()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if len(completed) != len(prompts) or engine.cache_manager.num_used_blocks:
        raise RuntimeError("Incomplete drain or retained KV blocks")
    for output in completed.values():
        if (
            len(output.token_ids) != OUTPUT
            or output.exit_depths != [LOOPS] * OUTPUT
            or output.finish_reason != "length"
        ):
            raise RuntimeError(
                f"Unexpected token count, exit depth or stop for {output.request_id}"
            )
    return elapsed


def measure(model, backend, concurrency, pair):
    blocks = concurrency * LOOPS * ((PROMPT + OUTPUT + PAGE - 1) // PAGE)
    engine = StageEngine(
        model,
        attention_backend=backend,
        cache_config=CacheConfig(num_blocks=blocks, block_size=PAGE),
        scheduler_config=SchedulerConfig(
            max_num_seqs=concurrency, max_num_batched_tokens=PROMPT, prefill_chunk_size=PROMPT
        ),
        execution_config=ExecutionConfig(async_scheduling=False, packed_prefill=False),
        exit_config=ExitConfig("ouro"),
    )
    try:
        cache = engine.cache_manager
        if cache.backend != backend or cache.key_cache.dtype != torch.bfloat16:
            raise RuntimeError("Unexpected attention backend or KV dtype")
        params = SamplingParams(
            max_tokens=OUTPUT, min_loops=LOOPS, max_loops=LOOPS, ignore_eos=True
        )

        def prompts(seed, count):
            return torch.randint(
                3,
                model.config.vocab_size,
                (count, PROMPT),
                generator=torch.Generator().manual_seed(seed),
            ).tolist()

        drive(engine, prompts(9000, concurrency), concurrency, params)
        tokens = 2 * concurrency * OUTPUT
        elapsed = drive(engine, prompts(pair, 2 * concurrency), concurrency, params)
        return dict(
            pair=pair,
            backend=backend,
            attention=cache.attention_info,
            concurrency=concurrency,
            kv_blocks=blocks,
            output_tokens=tokens,
            seconds=elapsed,
            output_tokens_per_second=tokens / elapsed,
        )
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if min(args.pairs, *args.concurrency) < 1 or len(set(args.concurrency)) != len(
        args.concurrency
    ):
        parser.error("use positive pair counts and distinct positive concurrency levels")
    if torch.cuda.device_count() != 1:
        parser.error("select one GPU with CUDA_VISIBLE_DEVICES")
    model = OuroForCausalLM.from_pretrained(
        str(args.model), device="cuda", dtype=torch.bfloat16
    ).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        model=str(args.model.resolve()),
        checkpoint=checkpoint_identity(args.model),
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        triton=version("triton"),
        flashinfer=version("flashinfer-python"),
        input_tokens=PROMPT,
        output_tokens=OUTPUT,
        loops=LOOPS,
        page_size=PAGE,
        dtype=str(next(model.parameters()).dtype),
        packed_prefill=False,
        pairs=args.pairs,
        concurrency=args.concurrency,
        requests_per_trial="2x concurrency",
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    medians = []
    with (args.output / "trials.jsonl").open("x") as raw, torch.inference_mode():
        for concurrency in args.concurrency:
            rates = {"triton": [], "flashinfer": []}
            for pair in range(args.pairs):
                order = ("triton", "flashinfer") if pair % 2 == 0 else ("flashinfer", "triton")
                for backend in order:
                    row = measure(model, backend, concurrency, pair)
                    print(json.dumps(row), file=raw, flush=True)
                    rates[backend].append(row["output_tokens_per_second"])
                    print(
                        f"C{concurrency} pair {pair} {backend}: {rates[backend][-1]:.1f} tok/s",
                        flush=True,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
            medians.append(
                {"concurrency": concurrency, **{k: statistics.median(v) for k, v in rates.items()}}
            )
    (args.output / "medians.json").write_text(json.dumps(medians, indent=2) + "\n")


if __name__ == "__main__":
    main()
