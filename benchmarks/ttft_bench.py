"""Engine-level TTFT / ITL / throughput comparison: dense vs sparse decode.

Prefill rows never take the sparse decode correction (RFC #156 section 6.3), so
enabling the feature must not change prefill work. Any TTFT difference therefore
comes from the first speculative round, which does run the decode path.

Usage:
    python benchmarks/ttft_bench.py --device cuda --prompt 64 --max-tokens 32
    python benchmarks/ttft_bench.py --device cuda --retention 0.5
"""

import argparse
import json
import statistics
import time

import torch

from tests.helpers import make_speculative_engine, tiny_ouro_model
from vllm_rlt import SamplingParams, SparseDecodeAttentionConfig


def measure(engine, prompt, max_tokens, repeats):
    """TTFT and inter-token latency of sequential single-request decodes."""
    ttfts, itls, totals = [], [], []
    for repetition in range(repeats):
        request_id = f"r{repetition}"
        start = time.perf_counter()
        engine.add_request(
            request_id,
            prompt,
            SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        )
        first = None
        finished = None
        while True:
            if not engine.has_unfinished_requests():
                break
            for output in engine.step():
                if output.request_id != request_id:
                    continue
                if first is None:
                    first = time.perf_counter() - start
                if output.finished:
                    finished = time.perf_counter() - start
        assert first is not None and finished is not None
        ttfts.append(first * 1e3)
        totals.append(finished * 1e3)
        itls.append((finished - first) * 1e3 / max(1, max_tokens - 1))
    return dict(
        ttft_median_ms=round(statistics.median(ttfts), 2),
        ttft_min_ms=round(min(ttfts), 2),
        itl_median_ms=round(statistics.median(itls), 3),
        e2e_median_ms=round(statistics.median(totals), 2),
        tokens_per_s=round(max_tokens / (statistics.median(totals) / 1e3), 1),
    )


def run(device, sparse, prompt, max_tokens, repeats, seed=123):
    torch.manual_seed(seed)
    model = tiny_ouro_model(device)
    engine = make_speculative_engine(model, sparse=sparse)
    # Warmup: compile-free but pays one-time kernel/allocator costs.
    measure(engine, prompt, 4, 1)
    result = measure(engine, prompt, max_tokens, repeats)
    engine.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    prompt = [2 + (index % 60) for index in range(args.prompt)]
    cases = {
        "dense": None,
        "sparse r=1.0": SparseDecodeAttentionConfig(enabled=True, key_retention=1.0),
        "sparse r=0.5": SparseDecodeAttentionConfig(enabled=True, key_retention=0.5),
    }
    report = {}
    for name, sparse in cases.items():
        report[name] = run(args.device, sparse, prompt, args.max_tokens, args.repeats)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    header = f"{'case':<14}{'TTFT med':>10}{'TTFT min':>10}{'ITL':>9}{'e2e':>9}{'tok/s':>9}"
    print(
        f"TTFT / ITL — {args.device}, prompt {args.prompt}, max_tokens {args.max_tokens}, "
        f"repeats {args.repeats}"
    )
    print(header)
    for name, values in report.items():
        print(
            f"{name:<14}{values['ttft_median_ms']:>9.2f}ms{values['ttft_min_ms']:>9.2f}ms"
            f"{values['itl_median_ms']:>8.3f}ms{values['e2e_median_ms']:>8.1f}ms"
            f"{values['tokens_per_s']:>9.1f}"
        )
    base = report["dense"]["ttft_median_ms"]
    for name in list(report)[1:]:
        delta = report[name]["ttft_median_ms"] / base
        print(f"  TTFT {name} vs dense: {delta:.2f}x")


if __name__ == "__main__":
    main()
