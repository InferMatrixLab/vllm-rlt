"""Finite closed-loop native E2E for fixed-depth Ouro LoopCD."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    LoopCDParams,
    SamplingParams,
    SchedulerConfig,
)
from vllm_rlt.request import Stage


def distribution(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[math.ceil(0.95 * len(values)) - 1],
        "p99": ordered[math.ceil(0.99 * len(values)) - 1],
        "min": ordered[0],
        "max": ordered[-1],
    }


def graph_counts(engine):
    bank = engine.model_runner.graphs
    return (
        dict(captures=bank.captures, replays=bank.replays, fallbacks=bank.fallbacks)
        if bank
        else dict(captures=0, replays=0, fallbacks=0)
    )


def synchronize(engine):
    if engine.model_runner.device.type == "cuda":
        torch.cuda.synchronize(engine.model_runner.device)


@torch.inference_mode()
def trial(engine, prompts, sampling, concurrency):
    """Refill reused IDs only after consuming the complete output batch."""
    if engine.has_unfinished_requests() or engine.cache_manager.num_used_blocks:
        raise ValueError("trial requires a drained engine")
    if not 1 <= concurrency <= len(prompts) or sampling.max_tokens < 2:
        raise ValueError("require 1 <= C <= requests and at least two output tokens")
    if not sampling.ignore_eos or sampling.min_loops != sampling.max_loops:
        raise ValueError("fixed work requires ignore_eos and fixed decode depth")
    runner, cache = engine.model_runner, engine.cache_manager
    before_graph = graph_counts(engine)
    before_work = dict(runner.loopcd_stats)
    synchronize(engine)
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    start = time.perf_counter()
    active, traces, free = {}, {}, list(range(concurrency))
    next_request = kv_peak = 0
    previous_decode_rows = 0
    resident, outstanding, core_rows = Counter(), Counter(), Counter()
    phases = Counter()

    def fill():
        nonlocal next_request
        while free and next_request < len(prompts):
            slot = free.pop()
            rid = f"slot-{slot}"
            prompt = prompts[next_request]
            engine.add_request(rid, prompt, sampling)
            active[rid] = next_request
            traces[next_request] = dict(
                ordinal=next_request,
                request_id=rid,
                prompt=prompt,
                submitted_s=time.perf_counter() - start,
                admitted_s=None,
                token_ids=[],
                exit_depths=[],
                delivery_s=[],
            )
            next_request += 1

    fill()
    bound = len(prompts) * (
        max(map(len, prompts)) * engine.prefill_depth
        + sampling.max_tokens * (sampling.max_loops + 4)
        + 10
    )
    for _ in range(bound):
        outstanding[len(active)] += 1
        outputs = engine.step()
        now = time.perf_counter() - start
        requests = engine.scheduler.requests
        resident[
            sum(r.stage not in (Stage.WAITING, Stage.RECEIVING) for r in requests.values())
        ] += 1
        for rid, request in requests.items():
            row = traces[active[rid]]
            if request.stage not in (Stage.WAITING, Stage.RECEIVING) and row["admitted_s"] is None:
                row["admitted_s"] = now
        schedule = engine.last_schedule
        if schedule is not None:
            phases[f"{schedule.stage.value}:{len(schedule.items)}"] += 1
        work = runner.loopcd_stats
        submitted = work["decode_core_rows"] - before_work["decode_core_rows"]
        # Counts are taken from actual model runner row counters, not B/C settings.
        if submitted != previous_decode_rows:
            core_rows[submitted - previous_decode_rows] += 1
            previous_decode_rows = submitted
        kv_peak = max(kv_peak, cache.num_used_blocks)
        released = []
        for output in outputs:
            row = traces[active[output.request_id]]
            count = len(row["token_ids"])
            if (
                output.prompt_token_ids != row["prompt"]
                or output.token_ids[:count] != row["token_ids"]
            ):
                raise RuntimeError("request identity or output prefix changed")
            row["delivery_s"].extend([now] * (len(output.token_ids) - count))
            row["token_ids"], row["exit_depths"] = output.token_ids, output.exit_depths
            if output.finished:
                expected = [engine.prefill_depth] + [sampling.max_loops] * (sampling.max_tokens - 1)
                if (
                    len(output.token_ids) != sampling.max_tokens
                    or output.exit_depths != expected
                    or output.finish_reason != "length"
                ):
                    raise RuntimeError("fixed work/depth/finish contract failed")
                row.update(finished_s=now, finish_reason=output.finish_reason)
                released.append(int(output.request_id.split("-")[1]))
                del active[output.request_id]
        free.extend(sorted(released))
        fill()
        if not active:
            break
    else:
        raise RuntimeError("finite work bound exhausted")
    synchronize(engine)
    elapsed = time.perf_counter() - start
    if engine.has_unfinished_requests() or cache.num_used_blocks or runner.loopcd_references:
        raise RuntimeError("KV/reference/request leak after drain")
    rows = [traces[i] for i in range(len(prompts))]
    after_graph = graph_counts(engine)
    return dict(
        elapsed_s=elapsed,
        tps=len(prompts) * sampling.max_tokens / elapsed,
        rps=len(prompts) / elapsed,
        requests=rows,
        ttft_s=distribution([r["delivery_s"][0] - r["submitted_s"] for r in rows]),
        latency_s=distribution([r["finished_s"] - r["submitted_s"] for r in rows]),
        tpot_s=distribution(
            [(r["delivery_s"][-1] - r["delivery_s"][0]) / (sampling.max_tokens - 1) for r in rows]
        ),
        itl_s=distribution(
            [b - a for r in rows for a, b in zip(r["delivery_s"], r["delivery_s"][1:])]
        ),
        admission_wait_s=distribution([r["admitted_s"] - r["submitted_s"] for r in rows]),
        resident_histogram=dict(resident),
        outstanding_histogram=dict(outstanding),
        decode_effective_rows_histogram=dict(core_rows),
        schedule_histogram=dict(phases),
        kv_peak_blocks=kv_peak,
        kv_pool_bytes=cache.num_blocks * cache.bytes_per_block,
        loopcd_work={
            k: v - before_work[k]
            for k, v in runner.loopcd_stats.items()
            if k != "reference_peak_bytes"
        },
        reference_peak_bytes=runner.loopcd_stats["reference_peak_bytes"],
        graph={k: after_graph[k] - before_graph[k] for k in after_graph},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(runner.device)
        if runner.device.type == "cuda"
        else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(runner.device)
        if runner.device.type == "cuda"
        else None,
        drained=True,
    )


def signature(result):
    return [
        (r["ordinal"], r["request_id"], r["token_ids"], r["exit_depths"], r["finish_reason"])
        for r in result["requests"]
    ]


def run(engine, prompts, sampling, concurrency, output, warmups=5, repetitions=5):
    output.mkdir(parents=True, exist_ok=False)
    expected = None
    summary = []
    for index in range(warmups + repetitions):
        result = trial(engine, prompts, sampling, concurrency)
        result.update(index=index, phase="warmup" if index < warmups else "measured")
        current = signature(result)
        if expected is not None and current != expected:
            result["repeat_parity"] = False
        else:
            result["repeat_parity"] = True
        expected = current
        (output / f"trial-{index:02d}.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["repeat_parity"]:
            raise RuntimeError("repeat token/depth/ID/finish mismatch; raw result preserved")
        if index >= warmups and result["graph"]["captures"]:
            raise RuntimeError("cold graph capture during measured work; raw result preserved")
        summary.append({k: v for k, v in result.items() if k != "requests"})
        print(
            json.dumps(
                {"index": index, "phase": result["phase"], "tps": result["tps"], "drained": True}
            ),
            flush=True,
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="574fa66cb8bf5abdc979642d01cf2b79b16bfab1")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--input-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--strength", type=float, choices=[0.0, 0.3], default=0.3)
    parser.add_argument("--kv-gib", type=float, required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batch <= args.concurrency <= 128 or min(args.input_lengths) < 1:
        parser.error("require 1 <= B <= C <= 128 and positive input lengths")
    if not Path(args.model).is_dir():
        parser.error("--model must be a verified local checkpoint directory")
    import vllm_rlt

    source_root = Path(__file__).resolve().parents[1]
    if Path(vllm_rlt.__file__).resolve().parents[1] != source_root:
        raise RuntimeError("imported vllm_rlt belongs to another checkout")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=source_root, check=True)
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    llm = LLM(
        args.model,
        revision=args.revision,
        device="cuda",
        dtype=torch.bfloat16,
        attention_backend="triton",
        cache_config=CacheConfig(kv_cache_memory_bytes=int(args.kv_gib * 1024**3)),
        scheduler_config=SchedulerConfig(
            max_num_seqs=args.batch,
            max_num_batched_tokens=max(128, args.batch),
            prefill_chunk_size=128,
        ),
        execution_config=ExecutionConfig(
            loopcd=True, prefill_depth=4, cuda_graphs=args.graph, cuda_graph_max_graphs=64
        ),
    )
    from vllm_rlt.models import load_tokenizer

    tokenizer = load_tokenizer(args.model, revision=args.revision)
    seed_ids = tokenizer.encode(
        "The scientist studied the stars and recorded the results. ", add_special_tokens=False
    )
    prompts = [
        (seed_ids * math.ceil(length / len(seed_ids)))[:length]
        for length in (
            args.input_lengths * math.ceil(4 * args.concurrency / len(args.input_lengths))
        )[: 4 * args.concurrency]
    ]
    sampling = SamplingParams(
        max_tokens=args.output_tokens,
        min_loops=args.depth,
        max_loops=args.depth,
        ignore_eos=True,
        temperature=0.0,
        loopcd=LoopCDParams(strength=args.strength) if args.strength else None,
    )
    manifest = dict(
        source_sha=source_sha,
        args={**vars(args), "output": str(args.output)},
        sampling=asdict(sampling),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        memory_plan=llm.engine.memory_plan,
        scope="native fixed work; no task quality or HTTP claim",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run(llm.engine, prompts, sampling, args.concurrency, args.output)
    llm.close()


if __name__ == "__main__":
    main()
