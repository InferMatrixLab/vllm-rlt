"""High-concurrency decode throughput of Ouro with and without WISE (wise-cached-decode).

Each case runs C requests of exactly L real prompt tokens (Ouro-tokenized
HotpotQA/2Wiki plain prompts, concatenated and left-truncated to L) with greedy
decoding, ignore_eos and a fixed number of output tokens, always 4 loops.
Decode throughput = (tokens generated after t_first) / (t_done - t_first), where
t_first is the first engine step after which every request has its first token
(prefill complete) and t_done is when all have finished. Tokens that early requests
generate while later prompts still prefill fall before t_first and are not counted.
Arms run in fresh processes, interleaved per repeat.

Arms (all Triton unless noted; graph arms use CUDA graphs up to the case batch)
  dense_eager  synchronous eager decode
  dense_graph  synchronous decode with CUDA graphs
  dense_best   async scheduling + CUDA graphs
  dense_fa3    FlashAttention-3 with async scheduling + CUDA graphs
  wise_eager   WISE decode, eager (ouro4: discover at depths 0-1, reuse at 2-3)
  wise         WISE decode with CUDA graphs
  wise_best    WISE decode with async scheduling + CUDA graphs
  wise_all     WISE discovery and selection with CUDA graphs, every block kept
  split_dense  CUDA graphs, every row dense through WISE's split-KV kernel (its dense baseline)
  fixed33      CUDA graphs, no discovery, fixed one-in-three block support at reuse depths
  spec_trunc_kN  CUDA graphs, self-speculation: N draft tokens read out after 2 loops
  spec_wise_kN   CUDA graphs, self-speculation: N draft tokens from WISE loops 3-4; both
                 speculative arms verify densely, so greedy output equals dense decode
Any arm may end in @fa2, @fa3 or @fa4 to use FlashAttention 2/3/4 instead of Triton
(FA2 pages are 256 tokens, others 16); dense_fa3 is dense_best@fa3. A WISE arm with a
"_tk" suffix (e.g. wise_best_tk@fa3) keeps FlashAttention for prefill but decodes
WISE with the Triton split-KV kernel.
"""

import argparse
import json
import math
import os
import time

import torch

from vllm_rlt import LLM, SamplingParams
from vllm_rlt.config import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.wise import WiseConfig

ARMS = (
    "dense_eager",
    "dense_graph",
    "dense_best",
    "dense_fa3",
    "wise_eager",
    "wise",
    "wise_all",
    "fixed33",
    "spec_trunc_k2",
    "spec_trunc_k4",
    "spec_wise_k2",
    "spec_wise_k4",
)


def prompts_of_length(rows, length, count):
    out = []
    for i in range(count):
        tokens, j = [], i
        while len(tokens) < length:
            tokens = rows[j % len(rows)]["token_ids"] + tokens
            j += 1
        out.append(tokens[-length:])
    return out


class InProcessLLM(LLM):
    """LLM facade over an in-process StageEngine, so the benchmark can read engine state."""

    def __init__(self, model, **options):
        self.tokenizer = None
        self._tokenizer_source = None
        self._next_request_id = 0
        self.engine = StageEngine(model, **options)

    def close(self):
        self.engine.close()


def build(model, arm, concurrency, length, output_tokens):
    wise = None
    if arm == "dense_fa3":
        arm = "dense_best@fa3"
    arm, _, tag = arm.partition("@")
    triton_kernel = arm.endswith("_tk")
    arm = arm.removesuffix("_tk")
    backend = {"": "triton", "fa2": "flash_attn_2", "fa3": "flash_attn_3", "fa4": "flash_attn_4"}[
        tag
    ]
    page = 256 if tag == "fa2" else 16
    graphs = arm not in ("dense_eager", "wise_eager")
    asynchronous = arm in ("dense_best", "wise_best")
    execution = ExecutionConfig(
        async_scheduling=asynchronous,
        cuda_graphs=graphs,
        cuda_graph_max_batch_size=max(128, concurrency),
    )
    speculative = None
    if arm.startswith("spec_"):
        k = int(arm.rsplit("_k", 1)[1])
        draft = "wise" if arm.startswith("spec_wise") else "truncate"
        speculative = SpeculativeConfig(k, draft_attention=draft)
        if draft == "wise":
            wise = WiseConfig(
                max_model_len=length + output_tokens + 32 + k, max_slots=concurrency + 8
            )
    elif arm.startswith("wise") or arm in ("fixed33", "split_dense"):
        override = {
            "wise": None,
            "wise_best": None,
            "wise_eager": None,
            "wise_all": "all",
            "fixed33": 1 / 3,
            "split_dense": "dense",
        }[arm]
        wise = WiseConfig(
            max_model_len=length + output_tokens + 32,
            max_slots=concurrency + 8,
            support_override=override,
            kernel="triton" if triton_kernel else "auto",
        )
    return InProcessLLM(
        model,
        attention_backend=backend,
        # Identical, case-sized KV pool for every arm: all C requests fit at once.
        cache_config=CacheConfig(
            num_blocks=math.ceil(
                concurrency * math.ceil((length + output_tokens + 32) / page) * 4 * 1.05
            ),
            block_size=page,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=concurrency, max_num_batched_tokens=4096, prefill_chunk_size=2048
        ),
        execution_config=execution,
        # Async needs a delayed exit policy and speculation the fixed-depth ouro policy;
        # threshold 1.0 never exits, so every arm runs 4 loops.
        exit_config=ExitConfig(mode="ouro" if speculative else "ouro_delayed"),
        speculative_config=speculative,
        wise_config=wise,
    )


def timed(llm, prompts, tokens, tag="r"):
    params = SamplingParams(
        max_tokens=tokens, ignore_eos=True, exit_threshold=1.0, min_loops=4, max_loops=4
    )
    engine = llm.engine
    ids = [f"{tag}{i}" for i in range(len(prompts))]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for request_id, prompt in zip(ids, prompts):
        engine.add_request(request_id, list(prompt), params)
    first, outputs = None, {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output.token_ids
        if first is None:
            live = engine.scheduler.requests.values()
            if all(len(r.generated_token_ids) >= 1 for r in live):
                torch.cuda.synchronize()
                first = time.perf_counter()
                # Early requests may decode while later prompts still prefill; those
                # tokens belong to the prefill interval, not to the decode count.
                requests = engine.scheduler.requests
                before = sum(
                    len(outputs[i]) if i in outputs else len(requests[i].generated_token_ids)
                    for i in ids
                )
    torch.cuda.synchronize()
    done = time.perf_counter()
    tokens_out = [outputs[i] for i in ids]
    assert all(len(t) == tokens for t in tokens_out)
    return first - start, done - first, tokens_out, len(ids) * tokens - before


def run_one(a):
    """One (case, arm, repeat) in a fresh process, so every arm starts from empty GPU memory."""
    rows = [json.loads(x) for x in open(a.prompts)]
    length, concurrency = map(int, a.cases.split("x"))
    prompts = prompts_of_length(rows, length, concurrency)
    model = OuroForCausalLM.from_pretrained(a.model, device="cuda", dtype=torch.bfloat16)
    llm = build(model, a.arms, concurrency, length, a.output_tokens)
    blocks = llm.engine.cache_manager.num_free_blocks
    # Same batch size, short prompts: compiles kernels and captures decode graphs untimed.
    timed(llm, [p[-64:] for p in prompts], 8, tag="warm")
    wise = llm.engine.cache_manager.wise
    if wise is not None:
        wise.stats.zero_()  # support density of the timed cohort only
    prefill, decode, tokens, decoded = timed(llm, prompts, a.output_tokens)
    record = {
        "length": length,
        "concurrency": concurrency,
        "arm": a.arms,
        "repeat": a.repeats,
        "output_tokens": a.output_tokens,
        "prefill_s": prefill,
        "decode_s": decode,
        "decode_tokens": decoded,
        "decode_tok_s": decoded / decode,
        "end_to_end_tok_s": concurrency * a.output_tokens / (prefill + decode),
        "free_blocks_at_start": blocks,
        "support_density": wise.mean_density() if wise is not None else None,
        "attention": llm.engine.cache_manager.attention_info.get("selected_backend"),
        "speculative": (
            vars(llm.engine.speculative_runner.stats).copy()
            if llm.engine.speculative_runner is not None
            else None
        ),
        "tokens": tokens,
    }
    with open(a.out, "x") as f:
        json.dump(record, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True, help="JSONL with token_ids (Ouro plain prompts)")
    p.add_argument(
        "--cases", required=True, help="comma list LENGTHxCONCURRENCY, e.g. 1024x32,2048x16"
    )
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.single:
        return run_one(a)
    import subprocess
    import sys
    import tempfile

    arms = a.arms.split(",")
    with open(a.out, "x") as f, tempfile.TemporaryDirectory() as tmp:
        for case in a.cases.split(","):
            for repeat in range(1, a.repeats + 1):  # each process warms up before timing
                order = arms if repeat % 2 else arms[::-1]
                records = {}
                for arm in order:
                    path = os.path.join(tmp, f"{case}-{arm}-{repeat}.json")
                    command = [
                        sys.executable,
                        __file__,
                        "--single",
                        "--model",
                        a.model,
                        "--prompts",
                        a.prompts,
                        "--cases",
                        case,
                        "--arms",
                        arm,
                        "--output-tokens",
                        str(a.output_tokens),
                        "--repeats",
                        str(repeat),
                        "--out",
                        path,
                    ]
                    done = subprocess.run(command, capture_output=True, text=True)
                    if done.returncode:
                        records[arm] = {
                            "case": case,
                            "arm": arm,
                            "repeat": repeat,
                            "error": done.stderr[-2000:],
                        }
                    else:
                        records[arm] = json.load(open(path))
                base = records.get("dense_eager", {}).get("tokens")
                for arm in order:
                    record = records[arm]
                    mine = record.pop("tokens", None)
                    if base is not None and mine is not None:
                        record["sequences_identical_to_dense_eager"] = sum(
                            x == y for x, y in zip(mine, base)
                        )
                        record["first_divergence_mean"] = sum(
                            next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), len(x))
                            for x, y in zip(mine, base)
                        ) / len(base)
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    print(json.dumps(record)[:400], flush=True)


if __name__ == "__main__":
    main()
