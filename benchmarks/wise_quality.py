"""Answer quality of wise-cached-decode through the engine, paired with dense decode.

Runs the fixed 200 Ouro plain-completion HotpotQA/2Wiki prompts greedily for 32
tokens in both arms (same engine settings, Triton, always 4 loops) and scores
the first line with the WISE authors' benchmark_scores (needs wise-reference on
PYTHONPATH). Reports per-row predictions, F1/EM and token agreement.
"""

import argparse
import json

import torch
from transformers import AutoTokenizer
from wise.metrics import benchmark_scores

from vllm_rlt import LLM, SamplingParams
from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.wise import WiseConfig


class InProcessLLM(LLM):
    """LLM facade over an in-process StageEngine, so the script can read engine state."""

    def __init__(self, model, **options):
        self.tokenizer = None
        self._tokenizer_source = None
        self._next_request_id = 0
        self.engine = StageEngine(model, **options)


def run_arm(a, rows, arm):
    model = OuroForCausalLM.from_pretrained(a.model, device="cuda", dtype=torch.bfloat16)
    # EOS stops generation; scoring cuts at <|im_end|> too and keeps the first line.
    params = SamplingParams(max_tokens=a.max_tokens, exit_threshold=1.0, min_loops=4, max_loops=4)
    longest = max(len(r["token_ids"]) for r in rows) + a.max_tokens + 32
    seqs = 32
    # Both arms: the benchmark's best configuration (async scheduling + CUDA graphs).
    llm = InProcessLLM(
        model,
        attention_backend=a.backend,
        cache_config=CacheConfig(num_blocks=seqs * ((longest + 15) // 16) * 4, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=seqs, max_num_batched_tokens=4096, prefill_chunk_size=2048
        ),
        execution_config=ExecutionConfig(async_scheduling=True, cuda_graphs=True),
        exit_config=ExitConfig(mode="ouro_delayed"),
        wise_config=WiseConfig(max_model_len=longest, max_slots=seqs + 8, kernel=a.wise_kernel)
        if arm == "wise"
        else None,
    )
    outputs = llm.generate([r["token_ids"] for r in rows], params)
    wise = llm.engine.cache_manager.wise
    return [o.token_ids for o in outputs], wise.mean_density() if wise is not None else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--backend", default="flash_attn_4", help="prefill and dense decode backend")
    p.add_argument("--wise-kernel", default="triton", choices=("auto", "triton"))
    p.add_argument("--arm", choices=("dense", "wise"), help=argparse.SUPPRESS)
    a = p.parse_args()
    rows = [json.loads(x) for x in open(a.prompts)]
    if a.arm:
        tokens, density = run_arm(a, rows, a.arm)
        with open(a.out, "x") as f:
            json.dump({"tokens": tokens, "density": density}, f)
        return
    import subprocess
    import sys
    import tempfile

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for arm in ("dense", "wise"):
            path = f"{tmp}/{arm}.json"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--model",
                    a.model,
                    "--prompts",
                    a.prompts,
                    "--out",
                    path,
                    "--max-tokens",
                    str(a.max_tokens),
                    "--backend",
                    a.backend,
                    "--wise-kernel",
                    a.wise_kernel,
                    "--arm",
                    arm,
                ],
                check=True,
            )
            results[arm] = json.load(open(path))
    with open(a.out, "x") as f:
        for row, dense, wise in zip(rows, results["dense"]["tokens"], results["wise"]["tokens"]):
            record = {
                "id": row["id"],
                "benchmark": row["benchmark"],
                "format": row.get("format"),
                "identical_tokens": dense == wise,
                "first_divergence": next(
                    (i for i, (u, v) in enumerate(zip(dense, wise)) if u != v),
                    min(len(dense), len(wise)),
                ),
            }
            end = tokenizer.convert_tokens_to_ids("<|im_end|>")
            for arm, tokens in (("dense", dense), ("wise", wise)):
                tokens = tokens[: tokens.index(end)] if end in tokens else tokens
                text = tokenizer.decode(tokens, skip_special_tokens=True).split("\n", 1)[0].strip()
                f1, em = benchmark_scores(row["benchmark"], text, row["gold_answer"])
                record[arm] = {"prediction": text, "f1": f1, "em": em, "tokens": tokens}
            f.write(json.dumps(record) + "\n")
        f.write(
            json.dumps(
                {
                    "summary": True,
                    "wise_mean_support_density": results["wise"]["density"],
                    "backend": a.backend,
                    "wise_kernel": a.wise_kernel,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
