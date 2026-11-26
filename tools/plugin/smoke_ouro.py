"""Smoke test: Ouro-1.4B through the vllm-rlt plugin on one GPU.

    CUDA_VISIBLE_DEVICES=0 python bench/smoke_ouro.py [--exit 0.9] [--wcpb] [--graphs]

Prints the first tokens; run it with and without the loop features to compare.
"""

from __future__ import annotations

import argparse
import json
import sys

MODEL = "/home/zjy/code/david/b_workspace/models/Ouro-1.4B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit", type=float, default=1.0, help="rlt_exit_threshold (1.0 = off)")
    ap.add_argument("--wcpb", action="store_true")
    ap.add_argument("--graphs", action="store_true", help="rlt_graphs: per-op loop graphs")
    ap.add_argument("--cg", action="store_true", help="vLLM CUDA graphs on, loop graphs off")
    ap.add_argument("--spec", type=int, default=0, help="self-spec K (0 = off)")
    ap.add_argument("--passes", type=int, default=2, help="draft passes")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--only", type=int, default=None, help="run just prompt N")
    ap.add_argument("--sync", action="store_true", help="async_scheduling=False")
    ap.add_argument("--long", type=int, default=0, help="prompts of about N tokens")
    ap.add_argument("--mbt", type=int, default=0, help="max_num_batched_tokens")
    ap.add_argument("--mixed", action="store_true", help="rlt_allow_mixed_depth")
    ap.add_argument("--plain", action="store_true", help="vLLM without the plugin")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    extra = {"rlt_wcpb": args.wcpb, "rlt_graphs": args.graphs, "rlt_allow_mixed_depth": args.mixed}
    kwargs = dict(
        model=MODEL,
        dtype=args.dtype,
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.4,
        disable_log_stats=False,
        enforce_eager=not (args.graphs or args.cg),
        scheduler_cls="vllm_rlt.plugin.scheduler.LoopScheduler",
        worker_cls="vllm_rlt.plugin.worker.LoopWorker",
        additional_config=extra,
    )
    if args.mbt:
        kwargs["max_num_batched_tokens"] = args.mbt
        kwargs["long_prefill_token_threshold"] = args.mbt
        kwargs["enable_chunked_prefill"] = True
    if args.plain:
        kwargs.pop("scheduler_cls")
        kwargs.pop("worker_cls")
        kwargs.pop("additional_config")
    if args.sync:
        kwargs["async_scheduling"] = False
    if args.spec:
        kwargs["speculative_config"] = {
            "method": "rlt_self",
            "num_speculative_tokens": args.spec,
            "speculator_options": {"draft_passes": args.passes},
        }
    llm = LLM(**kwargs)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.tokens,
        extra_args={"rlt_exit_threshold": args.exit} if args.exit < 1.0 else None,
    )
    prompts = ["The capital of France is", "def fib(n):", "Water boils at"]
    if args.long:
        base = "The history of the city is long and the river runs through it. "
        prompts = [p0 + " " + base * (args.long // 14) + "In summary," for p0 in prompts]
    if args.only is not None:
        prompts = prompts[args.only : args.only + 1]
    outs = llm.generate(prompts, params)
    if args.spec:
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                print("METRIC", m.name, getattr(m, "value", getattr(m, "values", None)))
    for out in outs:
        o = out.outputs[0]
        print(
            json.dumps(
                {
                    "prompt": out.prompt,
                    "text": o.text,
                    "finish": o.finish_reason,
                    "stop": str(o.stop_reason),
                    "ids": list(o.token_ids)[:12],
                }
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
