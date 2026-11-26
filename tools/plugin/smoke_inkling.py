"""Inkling smoke: plain vLLM vs plugin (passthrough / split ops / WCPB / graphs)."""
import argparse
import json
import os

os.environ.setdefault("VLLM_KV_CACHE_LAYOUT", "LBHNC")
ROOT = "/dev/shm/zjy-inkling2-mini"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="varloop", choices=["varloop", "fixed"])
    ap.add_argument("--mode", default="plain", choices=["plain", "pass", "split", "wcpb", "graphs"])
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--spec", type=int, default=0)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--long", type=int, default=0)
    ap.add_argument("--mbt", type=int, default=1024)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=f"{ROOT}/{args.kind}",
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=args.mbt,
        gpu_memory_utilization=0.5,
        block_size=16,
        kv_cache_dtype="bfloat16",
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,
        language_model_only=True,
        generation_config="vllm",
        disable_log_stats=False,
        enforce_eager=args.mode != "graphs",
    )
    if args.mbt < 1024:
        kwargs["long_prefill_token_threshold"] = args.mbt
    if args.kind == "varloop":
        kwargs["hf_overrides"] = {
            "text_config": {"loop_num_passes": args.passes, "loop_share_cache": False}
        }
    if args.mode != "plain":
        kwargs["scheduler_cls"] = "vllm_rlt.plugin.scheduler.LoopScheduler"
        kwargs["worker_cls"] = "vllm_rlt.plugin.worker.LoopWorker"
        kwargs["additional_config"] = {
            "rlt_split_ops": args.mode in ("split", "wcpb", "graphs"),
            "rlt_wcpb": args.mode == "wcpb",
            "rlt_graphs": args.mode == "graphs",
        }
    if args.spec:
        kwargs["speculative_config"] = {
            "method": "rlt_self",
            "num_speculative_tokens": args.spec,
            "speculator_options": {"draft_passes": 1},
        }
    if args.sync:
        kwargs["async_scheduling"] = False
    llm = LLM(**kwargs)
    prompts = ["The capital of France is", "def fib(n):", "Water boils at"]
    if args.long:
        base = "The history of the city is long and the river runs through it. "
        prompts = [p + " " + base * (args.long // 14) + "In summary," for p in prompts]
    params = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)
    outs = llm.generate(prompts, params)
    if args.spec:
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                print("METRIC", m.name, getattr(m, "value", getattr(m, "values", None)))
    for o in outs:
        c = o.outputs[0]
        print(json.dumps({"text": c.text[:80], "ids": list(c.token_ids)[:16]}))


if __name__ == "__main__":
    main()
