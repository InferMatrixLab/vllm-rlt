"""Paired local-preemption traffic benchmarks; profiling runs are separate from timing."""

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import time
import traceback
from pathlib import Path

CASES = [
    "_".join(parts)
    for parts in itertools.product(("equal", "mixed"), ("uniform", "mixed"), ("low", "high"))
]


def workload(case, *, count, slots, seed, rate, vocab_size):
    priority, lengths, _ = case.split("_")
    rng = random.Random(seed)
    requests = []
    arrival = 0.0
    for index in range(count):
        short = lengths == "mixed" and index >= slots and index % 2 == 0
        prompt, output = (32, 16) if short else ((256, 128) if lengths == "mixed" else (128, 64))
        requests.append(
            dict(
                request_id=f"r{index:04d}",
                arrival_s=arrival,
                prompt_token_ids=[rng.randrange(2, vocab_size) for _ in range(prompt)],
                max_tokens=output,
                priority=0 if priority == "equal" or (index >= slots and index % 2 == 0) else 10,
            )
        )
        arrival += rng.uniform(0.75, 1.25) / rate
    return requests


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def metrics(records):
    def summarize(rows):
        return {
            name: dict(
                mean=statistics.mean(values),
                p50=statistics.median(values),
                p95=percentile(values, 0.95),
            )
            for name in ("ttft_s", "e2e_s", "enqueue_ttft_s", "enqueue_e2e_s")
            if (values := [row[name] for row in rows])
        }

    return dict(
        all=summarize(records),
        by_priority={
            str(priority): summarize([row for row in records if row["priority"] == priority])
            for priority in sorted({row["priority"] for row in records})
        },
        by_length={
            str(length): summarize(
                [row for row in records if len(row["prompt_token_ids"]) == length]
            )
            for length in sorted({len(row["prompt_token_ids"]) for row in records})
        },
    )


def drive(engine, requests, state):
    from vllm_rlt import SamplingParams

    start = time.perf_counter()
    pending = iter(requests)
    next_request = next(pending, None)
    records = {}
    state.update(requests=[], max_waiting=0, steps=0)
    while next_request is not None or engine.has_unfinished_requests():
        now = time.perf_counter() - start
        while next_request is not None and next_request["arrival_s"] <= now:
            row = dict(next_request, enqueued_s=time.perf_counter() - start)
            records[row["request_id"]] = row
            state["requests"].append(row)
            engine.add_request(
                row["request_id"],
                row["prompt_token_ids"],
                SamplingParams(
                    max_tokens=row["max_tokens"],
                    priority=row["priority"],
                    ignore_eos=True,
                ),
            )
            next_request = next(pending, None)
        state["max_waiting"] = max(
            state["max_waiting"],
            sum(
                request.status.value == "waiting" for request in engine.scheduler.requests.values()
            ),
        )
        if engine.has_unfinished_requests():
            for output in engine.step():
                row = records[output.request_id]
                delivered = time.perf_counter() - start
                if output.token_ids and "first_output_s" not in row:
                    row["first_output_s"] = delivered
                if output.finished:
                    row.update(
                        finished_s=delivered,
                        token_ids=output.token_ids,
                        exit_depths=output.exit_depths,
                        finish_reason=output.finish_reason,
                    )
                    row["ttft_s"] = row["first_output_s"] - row["arrival_s"]
                    row["e2e_s"] = delivered - row["arrival_s"]
                    row["enqueue_ttft_s"] = row["first_output_s"] - row["enqueued_s"]
                    row["enqueue_e2e_s"] = delivered - row["enqueued_s"]
            state["steps"] += 1
        elif next_request is not None:
            time.sleep(min(0.002, max(0, next_request["arrival_s"] - now)))
    engine.model_runner.synchronize()
    state.update(
        wall_s=time.perf_counter() - start,
        preemptions=engine.preemption.preemptions,
        resumptions=engine.preemption.resumptions,
        kv_used_blocks=engine.cache_manager.num_used_blocks,
        suspended_requests=len(engine.preemption.snapshots),
    )
    if state["kv_used_blocks"] or state["suspended_requests"]:
        raise AssertionError("completed traffic retained request KV or suspended state")
    for row in state["requests"]:
        if len(row["token_ids"]) != row["max_tokens"] or row["finish_reason"] != "length":
            raise AssertionError("traffic did not produce the fixed output length")
    state["tokens_per_s"] = (
        sum(len(row["token_ids"]) for row in state["requests"]) / state["wall_s"]
    )
    state["metrics"] = metrics(state["requests"])


def checkpoint_identity(folder):
    names = [folder / "config.json", *sorted(folder.glob("*.safetensors"))]
    if (index := folder / "model.safetensors.index.json").is_file():
        names.append(index)
    if len(names) < 2:
        raise ValueError("a complete local safetensors checkpoint is required")
    result = {}
    for path in names:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        after = path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise RuntimeError(f"checkpoint changed while hashing: {path.name}")
        result[path.name] = dict(size=after.st_size, sha256=digest.hexdigest())
    return result


def trial(model, args, requests, output_dir, label, state, raw):
    import torch

    from vllm_rlt import (
        CacheConfig,
        ExecutionConfig,
        ProfileConfig,
        SchedulerConfig,
        SpeculativeConfig,
    )
    from vllm_rlt.engine.llm_engine import LLMEngine

    engine = None
    try:
        engine = LLMEngine(
            model,
            attention_backend=args.attention_backend,
            cache_config=CacheConfig(
                kv_cache_memory_bytes=args.kv_gib * 1024**3, incremental_allocation=True
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=64,
                prefill_chunk_size=64,
                policy="priority",
                enable_preemption=state["enable_preemption"],
            ),
            execution_config=ExecutionConfig(cuda_graphs=args.cuda_graphs),
            speculative_config=SpeculativeConfig(3) if args.decode == "speculative" else None,
        )
        # Same token workloads warm every arm before measurement; arrival time is
        # zero only for warmup, so low-load idle time is not multiplied by warmup.
        for index in range(args.warmups):
            warm = dict(label=f"{label}/warmup-{index}", kind="warmup")
            drive(engine, [dict(row, arrival_s=0.0) for row in requests], warm)
            raw.write(json.dumps(warm) + "\n")
            raw.flush()
        engine.preemption.preemptions = engine.preemption.resumptions = 0
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if state["kind"] == "profile":
            engine.start_profile(
                ProfileConfig(
                    output_dir=str(output_dir / "profiles" / label),
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True,
                    warmup=1,
                    active=args.profile_steps,
                ),
                scheduled=True,
            )
        drive(engine, requests, state)
    finally:
        if engine is not None:
            engine.close()  # Stop capture and naturally join the artifact worker.
            if state["kind"] == "profile":
                state["profile"] = engine.profile_status()
            if args.device == "cuda":
                state["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                state["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if state["kind"] == "profile" and not state["profile"]["success"]:
        raise RuntimeError("profiling artifacts failed; inspect the retained profile status")


def paired_summary(rows):
    result = {}
    for case in sorted({row["case"] for row in rows}):
        trials = [row for row in rows if row["case"] == case and row["kind"] == "measurement"]
        if not trials:
            continue
        pairs = {}
        for row in trials:
            pairs.setdefault(row["pair"], {})[row["enable_preemption"]] = row
        ratios = {}
        for scope in trials[0]["metrics"]:
            groups = [None] if scope == "all" else list(trials[0]["metrics"][scope])
            for group in groups:
                for metric, statistic in itertools.product(
                    ("ttft_s", "e2e_s"), ("mean", "p50", "p95")
                ):

                    def value(row):
                        data = row["metrics"][scope]
                        return (data if group is None else data[group])[metric][statistic]

                    values = [value(pair[True]) / value(pair[False]) for pair in pairs.values()]
                    ratios[f"{scope}/{group}/{metric}/{statistic}"] = dict(
                        per_pair=values,
                        mean=statistics.mean(values),
                        median=statistics.median(values),
                    )
        result[case] = dict(pairs=len(pairs), on_over_off=ratios)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="existing local Ouro checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--low-rps", type=float, required=True)
    parser.add_argument("--high-rps", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--decode", choices=("native", "speculative"), default="speculative")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--kv-gib", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--profile-case", choices=CASES)
    parser.add_argument("--profile-only", action="store_true", help="skip the timing matrix")
    parser.add_argument("--profile-steps", type=int, default=128)
    parser.add_argument(
        "--plan-only", action="store_true", help="print matrix without importing Torch"
    )
    args = parser.parse_args()
    if not (
        math.isfinite(args.low_rps)
        and math.isfinite(args.high_rps)
        and 0 < args.low_rps < args.high_rps
    ):
        parser.error("finite positive low-rps < high-rps required")
    if (
        min(
            args.max_num_seqs,
            args.requests,
            args.pairs,
            args.warmups,
            args.kv_gib,
            args.profile_steps,
        )
        < 1
    ):
        parser.error("counts and capacities must be positive")
    if args.requests <= args.max_num_seqs:
        parser.error("requests must exceed max-num-seqs to exercise dynamic arrivals")
    if args.profile_only and args.profile_case is None:
        parser.error("profile-only requires profile-case")
    cases = list(dict.fromkeys(args.case or CASES))
    plan = dict(
        cases=cases,
        low_rps=args.low_rps,
        high_rps=args.high_rps,
        pairs=args.pairs,
        requests=args.requests,
        seed=args.seed,
        decode=args.decode,
        timing="scheduled arrival to CPU output delivery; warmup/loading excluded",
        default_enable_preemption=False,
        scope="controlled local traffic; no quality score",
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if not args.model.is_dir():
        parser.error("model must be an existing local checkpoint directory")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = checkpoint_identity(args.model)
    import torch

    from vllm_rlt.models import OuroForCausalLM

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = OuroForCausalLM.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    )
    plan.update(
        checkpoint=identity,
        torch=torch.__version__,
        source_sha256={
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                Path(__file__).resolve(),
                *Path(__file__).resolve().parents[1].joinpath("vllm_rlt").rglob("*.py"),
            ]
        },
        device_name=torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        model_config=model.config.to_dict(),
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    with (args.output / "raw.jsonl").open("x") as raw:
        runs = [] if args.profile_only else [(case, "measurement") for case in cases]
        if args.profile_case:
            runs.append((args.profile_case, "profile"))
        for case, kind in runs:
            for pair in range(1 if kind == "profile" else args.pairs):
                rate = args.low_rps if case.endswith("_low") else args.high_rps
                requests = workload(
                    case,
                    count=args.requests,
                    slots=args.max_num_seqs,
                    seed=args.seed + pair,
                    rate=rate,
                    vocab_size=model.config.vocab_size,
                )
                outputs = {}
                for enabled in (False, True) if pair % 2 == 0 else (True, False):
                    label = f"{kind}-{case}-{pair}-{'on' if enabled else 'off'}"
                    state = dict(
                        case=case, pair=pair, kind=kind, label=label, enable_preemption=enabled
                    )
                    try:
                        trial(model, args, requests, args.output, label, state, raw)
                    except BaseException:
                        state["error"] = traceback.format_exc()
                        raw.write(json.dumps(state) + "\n")
                        raw.flush()
                        raise
                    raw.write(json.dumps(state) + "\n")
                    raw.flush()
                    rows.append(state)
                    outputs[enabled] = {
                        row["request_id"]: (row["token_ids"], row["exit_depths"])
                        for row in state["requests"]
                    }
                if outputs[False] != outputs[True]:
                    raw.write(
                        json.dumps(dict(case=case, pair=pair, error="paired token/depth mismatch"))
                        + "\n"
                    )
                    raw.flush()
                    raise AssertionError("paired traffic changed generated token/depth output")
    summary = dict(
        cases=paired_summary(rows),
        full_matrix={row["case"] for row in rows if row["kind"] == "measurement"} == set(CASES),
        default_enable_preemption=False,
        profiles_excluded_from_timing=True,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
