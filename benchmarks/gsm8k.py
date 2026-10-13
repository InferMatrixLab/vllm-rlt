"""Compare Ouro with the released Transformers model on a fixed GSM8K subset."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

DATA_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PACKAGES = ("torch", "transformers", "lm-eval", "datasets", "tokenizers", "triton")
DEFAULT_CASE = Path(__file__).parent / "fixtures/gsm8k-87.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def source():
    return {
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def make_task(config=None, split="test"):
    import lm_eval
    import yaml
    from lm_eval.api.task import ConfigurableTask

    if config is None:
        task_file = Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k-cot.yaml"
        config = yaml.safe_load(task_file.read_text())
        config.update(
            dataset_path="openai/gsm8k",
            dataset_kwargs={"revision": DATA_REVISION},
            num_fewshot=3,
            test_split=split,
        )
    task = ConfigurableTask(config=config)
    task.set_fewshot_seed(1234)
    return task


def select_doc_ids(population, *, limit, seed, split):
    """Select without inspecting answers or model outputs; retain dataset row IDs."""
    if population < 1 or (limit is not None and not 1 <= limit <= population):
        raise ValueError("--limit must be between 1 and the dataset size")
    ranked = sorted(
        range(population),
        key=lambda index: hashlib.sha256(
            f"{DATA_REVISION}:{split}:{seed}:{index}".encode()
        ).digest(),
    )
    return sorted(ranked[:limit])


def checkpoint_loops(config):
    """Full recurrent depth of an Ouro checkpoint, from its config.json."""
    loops = config.get("total_ut_steps")
    if config.get("model_type") != "ouro" or type(loops) is not int or loops < 1:
        raise ValueError("This evaluation requires an Ouro checkpoint with total_ut_steps >= 1")
    return loops


def fixed_exit(loops):
    """The original recipe: every output token runs the checkpoint's full depth."""
    return {
        "mode": "ouro",
        "threshold": 1.0,
        "min_loops": loops,
        "max_loops": loops,
        "async_scheduling": False,
    }


def is_fixed(exit_policy):
    return exit_policy["threshold"] == 1


def baseline_fingerprint(protocol):
    """Identify the exact questions, prompts, scoring, model and generation recipe."""
    return digest(
        {
            **{
                key: protocol[key]
                for key in (
                    "model_revision",
                    "model_files",
                    "task_config",
                    "records",
                    "max_new_tokens",
                    "max_length",
                    "dtype",
                    "loops",
                    "batch_size",
                    "add_special_tokens",
                    "apply_chat_template",
                )
            },
            "packages": protocol["source"]["packages"],
        }
    )


def build_records(task, tokenizer, *, split, limit, seed, max_new_tokens, max_length, doc_ids=None):
    ids = (
        select_doc_ids(len(task.eval_docs), limit=limit, seed=seed, split=split)
        if doc_ids is None
        else list(doc_ids)
    )
    if not ids or ids != sorted(set(ids)) or any(i < 0 or i >= len(task.eval_docs) for i in ids):
        raise ValueError("Question IDs must be unique, sorted and present in the dataset")
    task.build_all_requests(samples=ids, rank=0, world_size=1)
    records = []
    for instance in task.instances:
        prompt, kwargs = instance.args
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) + max_new_tokens > max_length:
            raise ValueError("Prompt would be truncated; increase --max-length")
        # lm-eval 0.4.9.2 renumbers explicit samples from zero. Restore the
        # original dataset ID so paired reports identify the actual question.
        records.append(
            {
                "id": ids[instance.doc_id],
                "doc": instance.doc,
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "generation_kwargs": kwargs,
            }
        )
    if [row["id"] for row in records] != ids:
        raise ValueError("The task did not produce exactly one request per selected question")
    return records, {
        "method": "sha256-ranked-source-ids-v1" if doc_ids is None else "fixed-source-ids-v1",
        "seed": seed if doc_ids is None else None,
        "split": split,
        "dataset_revision": DATA_REVISION,
        "population": len(task.eval_docs),
        "ids": ids,
    }


def prepare(args):
    import torch
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from transformers import AutoTokenizer

    if importlib.metadata.version("lm-eval") != "0.4.9.2":
        raise ValueError("Install the pinned evaluation dependencies first")
    model = Path(args.model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not 0 < args.max_new_tokens < args.max_length or args.max_regression_pp < 0:
        raise ValueError("Invalid context/output budget or regression threshold")
    loops = checkpoint_loops(json.loads((model / "config.json").read_text()))
    task = make_task(split=args.split)
    case = None
    if args.limit is None and not args.all:
        if args.split != "test" or args.seed != 0:
            raise ValueError(
                "Use --limit or --all for a custom split/seed; the default case is fixed"
            )
        case = json.loads(DEFAULT_CASE.read_text())
        if case["dataset_revision"] != DATA_REVISION or case["split"] != args.split:
            raise ValueError("Default case does not match the pinned dataset")
    records, selection = build_records(
        task,
        tokenizer,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        doc_ids=case["source_ids"] if case else None,
    )
    files = [
        p
        for p in model.iterdir()
        if p.is_file() and p.suffix in (".json", ".py", ".safetensors", ".txt")
    ]
    hashes = {}
    for path in sorted(files):
        metadata = get_hf_file_metadata(
            hf_hub_url(
                args.model_repo,
                path.name,
                revision=args.model_revision,
            )
        )
        hashes[path.name] = file_digest(path)
        if len(metadata.etag) == 64:
            observed = hashes[path.name]
        else:
            data = path.read_bytes()
            observed = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if observed != metadata.etag or metadata.commit_hash != args.model_revision:
            raise ValueError(f"File does not match the pinned HF release: {path.name}")
    protocol = {
        "format_version": 1,
        "model": str(model),
        "model_revision": args.model_revision,
        "model_files": hashes,
        "task_config": task.dump_config(),
        "records": records,
        "selection": selection,
        "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length,
        "dtype": "bfloat16",
        # Fingerprinted: the stored four-loop baseline applies only when this is 4.
        "loops": loops,
        "batch_size": 1,
        "add_special_tokens": False,
        "apply_chat_template": False,
        "max_regression_pp": args.max_regression_pp,
        "min_reference_accuracy_pct": args.min_reference_accuracy_pct,
        "source": source(),
    }
    if case:
        if baseline_fingerprint(protocol) != case["baseline"]["protocol_fingerprint"]:
            raise ValueError(
                "Default case settings differ from its HF baseline; use --limit or --all "
                "for a custom experiment"
            )
        protocol["selection"].update(case_id=case["case_id"], case_sha256=file_digest(DEFAULT_CASE))
        protocol["baseline"] = case["baseline"]
    if torch.cuda.is_initialized():
        raise RuntimeError("Preparation unexpectedly initialized CUDA")
    write_json(args.output, protocol)
    print(
        json.dumps(
            {
                "examples": len(records),
                "split": args.split,
                "prompt_tokens_max": max(len(r["prompt_ids"]) for r in records),
                "protocol_sha256": file_digest(args.output),
                "cuda_initialized": False,
            }
        )
    )


def exit_settings(
    backend, loops, mode="ouro", threshold=1.0, min_loops=None, async_scheduling=False
):
    """Validate one run's exit policy against the checkpoint's full depth ``loops``.

    The default reproduces the fixed-depth recipe.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("--exit-threshold must be in [0, 1]")
    if threshold == 1:
        if (mode, min_loops, async_scheduling) != ("ouro", None, False):
            raise ValueError("Exit options require --exit-threshold below 1")
        return fixed_exit(loops)
    if min_loops is None or not 1 <= min_loops <= loops:
        raise ValueError(f"Adaptive exit requires an explicit --min-loops between 1 and {loops}")
    if async_scheduling and mode != "ouro_delayed":
        raise ValueError("--async-scheduling requires --exit-mode ouro_delayed")
    if backend == "transformers" and (mode, min_loops) != ("ouro", 1):
        # The release applies its threshold from the first loop and has no delayed mode.
        raise ValueError("Transformers adaptive exit supports only --exit-mode ouro --min-loops 1")
    return {
        "mode": mode,
        "threshold": threshold,
        "min_loops": min_loops,
        "max_loops": loops,
        "async_scheduling": async_scheduling,
    }


def depth_summary(rows):
    """Summarize native exit depths; the first output token is produced by prefill."""
    if any("exit_depths" not in row for row in rows):
        return None
    decode = [row["exit_depths"][1:] for row in rows]
    tokens = sum(len(depths) for depths in decode)
    loops = [sum(depths) for depths in decode]
    histogram = {}
    for depths in decode:
        for depth in depths:
            histogram[str(depth)] = histogram.get(str(depth), 0) + 1
    return {
        "decode_tokens": tokens,
        "mean_decode_depth": sum(loops) / tokens if tokens else None,
        "decode_depth_histogram": dict(sorted(histogram.items())),
        "decode_loops_per_question_mean": statistics.mean(loops),
    }


def first_difference(a, b):
    """First index where two sequences differ; the shorter length if one is a prefix."""
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def sequence_differences(pairs):
    """Diagnostic token and exit-depth sequence differences per question (not gated).

    Matching answers and accuracy can hide different executions, e.g. synchronous and
    asynchronous runs of one exit policy. Each sequence is compared only where both runs
    recorded it. Exit depth k is chosen with the prompt plus tokens[:k] as context, so a
    depth difference at k <= the first token difference, with both runs having a depth at k,
    means the two runs made different exit decisions on identical context
    ("identical_context").
    """
    tokens, depths = [], []
    token_compared = depth_compared = 0
    for x, y in pairs:
        tx, ty = x.get("token_ids"), y.get("token_ids")
        token_first = None
        if tx is not None and ty is not None:
            token_compared += 1
            if tx != ty:
                token_first = first_difference(tx, ty)
                tokens.append({"id": x["id"], "first_divergence": token_first})
        dx, dy = x.get("exit_depths"), y.get("exit_depths")
        if dx is not None and dy is not None:
            depth_compared += 1
            if dx != dy:
                first = first_difference(dx, dy)
                shared = token_first if token_first is not None else min(len(dx), len(dy))
                # A difference only in length (one run stopped earlier) is not a decision.
                decided = first < min(len(dx), len(dy))
                depths.append(
                    {
                        "id": x["id"],
                        "first_divergence": first,
                        "identical_context": decided and first <= shared,
                    }
                )

    def block(differing, compared):
        firsts = [d["first_divergence"] for d in differing]
        return {
            "questions_compared": compared,
            "questions_differing": len(differing),
            "earliest_first_divergence": min(firsts, default=None),
            "median_first_divergence": statistics.median(firsts) if firsts else None,
            "questions": differing,
        }

    depth_block = block(depths, depth_compared)
    depth_block["questions_differing_on_identical_context"] = sum(
        d["identical_context"] for d in depths
    )
    return {"token_ids": block(tokens, token_compared), "exit_depths": depth_block}


def score(task, row, text):
    from lm_eval.api.instance import Instance

    instance = Instance(request_type="generate_until", doc=row["doc"], arguments=(), idx=0)
    instance.resps = [text]
    for pipeline in task._filters:
        pipeline.apply([instance])
    answer = instance.filtered_resps["strict-match"]
    correct = task.process_results(row["doc"], [answer])["exact_match"]
    return {"answer": answer, "correct": bool(correct), "unparseable": answer == "[invalid]"}


def baseline_result(protocol, rows):
    baseline = protocol["baseline"]
    expected = protocol["records"]
    if len(rows) != baseline["examples"] or [r["id"] for r in rows] != [r["id"] for r in expected]:
        raise ValueError("Default accuracy case is incomplete or uses different questions")
    if any(
        row["prompt_sha256"] != digest(record["prompt_ids"]) for row, record in zip(rows, expected)
    ):
        raise ValueError("Default accuracy case prompts differ from the protocol")
    delta = 100 * (sum(r["correct"] for r in rows) - baseline["correct"]) / len(rows)
    reference_accuracy = 100 * baseline["correct"] / baseline["examples"]
    floor = protocol["min_reference_accuracy_pct"]
    return {
        "backend": baseline["backend"],
        "examples": baseline["examples"],
        "reference_correct": baseline["correct"],
        "reference_accuracy_pct": reference_accuracy,
        "delta_pp": delta,
        "max_regression_pp": protocol["max_regression_pp"],
        "min_reference_accuracy_pct": floor,
        "passes_observed_accuracy_gate": (
            delta >= -protocol["max_regression_pp"]
            and (floor is None or reference_accuracy >= floor)
        ),
    }


def run(args):
    import torch
    from transformers import AutoTokenizer

    from benchmarks.gsm8k_backends import Generator

    protocol = json.loads(Path(args.protocol).read_text())
    if (
        "baseline" in protocol
        and baseline_fingerprint(protocol) != protocol["baseline"]["protocol_fingerprint"]
    ):
        raise ValueError("Default case protocol changed after its HF baseline was frozen")
    loops = protocol["loops"]
    exit_policy = exit_settings(
        args.backend,
        loops,
        args.exit_mode,
        args.exit_threshold,
        args.min_loops,
        args.async_scheduling,
    )
    if protocol["source"]["packages"] != source()["packages"]:
        raise ValueError("Evaluation package versions changed after preparation")
    model = Path(protocol["model"])
    for name, expected in protocol["model_files"].items():
        if file_digest(model / name) != expected:
            raise ValueError(f"Model file changed: {name}")
    if checkpoint_loops(json.loads((model / "config.json").read_text())) != loops:
        raise ValueError("Checkpoint depth differs from the protocol")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise RuntimeError("Run with one scheduler-assigned GPU")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "backend": args.backend,
        "attention_backend": args.attention_backend,
        "protocol_sha256": file_digest(args.protocol),
        "source": source(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": str(torch.cuda.get_device_properties(0)),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_cuda": torch.version.cuda,
        "split": protocol["task_config"]["test_split"],
        "expected_examples": len(protocol["records"]),
        "max_regression_pp": protocol["max_regression_pp"],
        "min_reference_accuracy_pct": protocol["min_reference_accuracy_pct"],
        "loops": loops,
        "exit": exit_policy,
    }
    write_json(output / "metadata.json", metadata)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    task = make_task(protocol["task_config"])
    start = time.monotonic()
    generator = Generator(
        args.backend, str(model), tokenizer, protocol["max_length"], exit_policy,
        attention_backend=args.attention_backend,
    )
    load_seconds = time.monotonic() - start
    rows = []
    with (output / "samples.jsonl").open("x", buffering=1) as stream:
        for row in protocol["records"]:
            stops = row["generation_kwargs"]["until"] + [tokenizer.eos_token]
            start = time.monotonic()
            result = generator.generate(row["prompt_ids"], protocol["max_new_tokens"], stops)
            result.update(score(task, row, result["text"]))
            result.update(
                id=row["id"],
                prompt_sha256=digest(row["prompt_ids"]),
                seconds=time.monotonic() - start,
            )
            stream.write(json.dumps(result) + "\n")
            rows.append(result)
            print(
                f"{args.backend} {len(rows)}/{len(protocol['records'])}: "
                f"correct={sum(r['correct'] for r in rows)}",
                flush=True,
            )
    summary = {
        **metadata,
        "complete": True,
        "examples": len(rows),
        "accuracy": statistics.mean(r["correct"] for r in rows),
        "correct": sum(r["correct"] for r in rows),
        "unparseable": sum(r["unparseable"] for r in rows),
        "length_limited": sum(r["finish_reason"] == "length" for r in rows),
        "generated_tokens_per_question_mean": statistics.mean(len(r["token_ids"]) for r in rows),
        "depth": depth_summary(rows),
        "load_seconds": load_seconds,
        "generation_and_scoring_seconds": sum(r["seconds"] for r in rows),
        "samples_sha256": file_digest(output / "samples.jsonl"),
    }
    # The stored baseline describes fixed-depth generation only; its fingerprint fixes loops.
    if args.backend == "native" and "baseline" in protocol and is_fixed(exit_policy):
        summary["baseline_comparison"] = baseline_result(protocol, rows)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def compare(args):
    paths = [Path(args.transformers), Path(args.native)]
    summaries = [json.loads((p / "summary.json").read_text()) for p in paths]
    a, b = summaries
    if {a["backend"], b["backend"]} - {"transformers", "native"}:
        raise ValueError("Unknown backend")
    for key in (
        "protocol_sha256",
        "expected_examples",
        "split",
        "max_regression_pp",
        "min_reference_accuracy_pct",
    ):
        if a[key] != b[key]:
            raise ValueError(f"Unmatched comparison: {key}")
    samples = []
    for path, summary in zip(paths, summaries):
        if not summary["complete"] or summary["examples"] != summary["expected_examples"]:
            raise ValueError("Incomplete evaluation")
        if file_digest(path / "samples.jsonl") != summary["samples_sha256"]:
            raise ValueError("Samples changed after scoring")
        rows = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
        if len(rows) != summary["examples"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Missing or duplicate examples")
        if sum(r["correct"] for r in rows) != summary["correct"]:
            raise ValueError("Summary score does not match samples")
        if statistics.mean(r["correct"] for r in rows) != summary["accuracy"]:
            raise ValueError("Summary accuracy does not match samples")
        samples.append(rows)
    pairs = list(zip(*samples))
    if any((x["id"], x["prompt_sha256"]) != (y["id"], y["prompt_sha256"]) for x, y in pairs):
        raise ValueError("Examples/prompts are not paired")
    differences = [int(y["correct"]) - int(x["correct"]) for x, y in pairs]
    delta = 100 * statistics.mean(differences)
    result = {
        "examples": a["examples"],
        "split": a["split"],
        # Summaries written before exit options existed used the fixed recipe, and the
        # harness then accepted only four-loop checkpoints.
        "reference": {
            "backend": a["backend"],
            "exit": a.get("exit", fixed_exit(4)),
            "depth": a.get("depth"),
        },
        "candidate": {
            "backend": b["backend"],
            "exit": b.get("exit", fixed_exit(4)),
            "depth": b.get("depth"),
        },
        "reference_accuracy_pct": 100 * a["accuracy"],
        "candidate_accuracy_pct": 100 * b["accuracy"],
        "reference_correct": a["correct"],
        "candidate_correct": b["correct"],
        "delta_pp": delta,
        "paired_delta_stderr_pp": 100 * statistics.stdev(differences) / len(pairs) ** 0.5
        if len(pairs) > 1
        else None,
        "reference_correct_candidate_wrong": sum(d == -1 for d in differences),
        "reference_wrong_candidate_correct": sum(d == 1 for d in differences),
        "answer_disagreements": [x["id"] for x, y in pairs if x["answer"] != y["answer"]],
        # Diagnostic only; the accuracy gate below does not use it.
        "sequence_differences": sequence_differences(pairs),
        "max_regression_pp": a["max_regression_pp"],
        "passes_observed_accuracy_gate": (
            delta >= -a["max_regression_pp"]
            and (
                a["min_reference_accuracy_pct"] is None
                or 100 * a["accuracy"] >= a["min_reference_accuracy_pct"]
            )
        ),
        "min_reference_accuracy_pct": a["min_reference_accuracy_pct"],
    }
    if (a["backend"], b["backend"]) == ("transformers", "native"):
        # Earlier key names, kept only for the original pairing where they are accurate.
        result.update(
            transformers_accuracy_pct=result["reference_accuracy_pct"],
            native_accuracy_pct=result["candidate_accuracy_pct"],
            transformers_correct=result["reference_correct"],
            native_correct=result["candidate_correct"],
            reference_correct_native_wrong=result["reference_correct_candidate_wrong"],
            reference_wrong_native_correct=result["reference_wrong_candidate_correct"],
        )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Freeze data/prompts and hashes without using CUDA")
    p.add_argument(
        "--model",
        required=True,
        help="Local pinned Ouro checkpoint including official Python files",
    )
    p.add_argument("--model-repo", required=True, help="HF repository used to verify local files")
    p.add_argument(
        "--model-revision", required=True, help="Exact HF commit used to verify local files"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    subset = p.add_mutually_exclusive_group()
    subset.add_argument(
        "--limit", type=int, help="Custom sample size; default is the fixed 87-question case"
    )
    subset.add_argument("--all", action="store_true", help="Use the complete split")
    p.add_argument("--seed", type=int, default=0, help="Fixed subset selection seed (default: 0)")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-regression-pp", type=float, default=1.0)
    p.add_argument(
        "--min-reference-accuracy-pct",
        type=float,
        help="Optional absolute floor; by default use the measured Transformers baseline only",
    )
    p = commands.add_parser("run")
    p.add_argument("--backend", choices=["transformers", "native"], required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--attention-backend",
        choices=["torch", "triton", "flashinfer"],
        default="triton",
        help="Native engine attention kernel; recorded in the run metadata",
    )
    p.add_argument("--exit-mode", choices=["ouro", "ouro_delayed"], default="ouro")
    p.add_argument(
        "--exit-threshold",
        type=float,
        default=1.0,
        help="Cumulative exit probability; 1 keeps the fixed full-depth recipe",
    )
    p.add_argument("--min-loops", type=int, help="Required when --exit-threshold is below 1")
    p.add_argument("--async-scheduling", action="store_true", help="Native ouro_delayed only")
    p = commands.add_parser("compare")
    p.add_argument("--transformers", "--reference", dest="transformers", required=True)
    p.add_argument("--native", "--candidate", dest="native", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"prepare": prepare, "run": run, "compare": compare}[args.command](args)
    if args.command == "compare" and not result["passes_observed_accuracy_gate"]:
        raise SystemExit(1)
    if args.command == "run" and not result.get("baseline_comparison", {}).get(
        "passes_observed_accuracy_gate", True
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
