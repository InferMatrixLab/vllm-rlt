"""SiLU-and-Mul accuracy and latency against eager Torch and vLLM.

CUDA_VISIBLE_DEVICES=<allocated GPU> python -m benchmarks.kernels.silu_and_mul \
    --output artifacts/silu-and-mul/results.json
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm_rlt.kernels._extension import load_extension
from vllm_rlt.kernels.activation import silu_and_mul


def errors(actual, expected):
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    delta = (actual[finite].double() - expected[finite].double()).abs()
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    sign = 2 ** (31 if integer == torch.int32 else 15)

    def ordered(x):
        bits = x[finite].view(integer).long()
        return torch.where(bits < 0, -sign - bits, bits)

    nonfinite_match = (actual == expected) | (torch.isnan(actual) & torch.isnan(expected))
    return {
        "max_abs": delta.max().item() if delta.numel() else 0,
        "rmse": delta.square().mean().sqrt().item() if delta.numel() else 0,
        "max_ulp": (ordered(actual) - ordered(expected)).abs().max().item() if delta.numel() else 0,
        "unequal_fraction": ((actual != expected) & ~(torch.isnan(actual) & torch.isnan(expected)))
        .float()
        .mean()
        .item(),
        "nonfinite_mismatches": ((~finite) & ~nonfinite_match).sum().item(),
    }


def make_functions(gate, up, vllm_op):
    # vLLM receives prepacked input and a preallocated output outside timing.
    packed = torch.cat((gate, up), dim=-1)
    output = torch.empty_like(gate)

    def vllm_prepacked():
        vllm_op(output, packed)
        return output

    return {
        "torch_eager": lambda: F.silu(gate) * up,
        "rlt": lambda: silu_and_mul(gate, up),
        "vllm_prepacked": vllm_prepacked,
    }


def measure(functions, repeats, iterations):
    graphs = {}
    unroll = 32
    for name, function in functions.items():
        for _ in range(10):
            function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(unroll):
                function()
        graphs[name] = graph

    result = {name: {"graph_us": [], "eager_us": []} for name in functions}
    rng = random.Random(42)
    for _ in range(repeats):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            graph = graphs[name]
            graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            result[name]["graph_us"].append(start.elapsed_time(end) * 1000 / (iterations * unroll))

            torch.cuda.synchronize()
            before = time.perf_counter()
            for _ in range(iterations):
                functions[name]()
            torch.cuda.synchronize()
            result[name]["eager_us"].append((time.perf_counter() - before) * 1e6 / iterations)
    for measurements in result.values():
        for mode in ("graph", "eager"):
            measurements[f"{mode}_median_us"] = statistics.median(measurements[f"{mode}_us"])
    return result


def profile(functions, output):
    output.mkdir(parents=True, exist_ok=True)
    for full in (False, True):
        mode = "full" if full else "ops-only"
        for function in functions.values():
            function()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=full,
            profile_memory=full,
            with_stack=full,
            experimental_config=torch.profiler._ExperimentalConfig(verbose=full),
        ) as prof:
            for name, function in functions.items():
                with torch.profiler.record_function(name):
                    function()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(output / f"{mode}.json"))
        (output / f"{mode}.txt").write_text(prof.key_averages().table(row_limit=40))
        if full:
            prof.export_stacks(str(output / "stacks.txt"), "self_cuda_time_total")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-library", type=Path)
    parser.add_argument("--vllm-namespace", default="_C")
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--tokens", default="1,16,128,1024,4096")
    parser.add_argument("--widths", default="5632,5633")
    parser.add_argument("--dtypes", default="bfloat16,float16,float32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    load_extension()  # Do not benchmark the Torch fallback as a native kernel.
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if args.vllm_library:
        torch.ops.load_library(str(args.vllm_library.resolve()))
        baseline = {"library": str(args.vllm_library.resolve())}
    else:
        extension = importlib.import_module("vllm._C_stable_libtorch")
        baseline = {"version": importlib.metadata.version("vllm"), "library": extension.__file__}
    baseline["sha256"] = hashlib.sha256(Path(baseline["library"]).read_bytes()).hexdigest()
    if args.vllm_source:
        baseline["source_commit"] = subprocess.check_output(
            ["git", "-C", str(args.vllm_source), "rev-parse", "HEAD"], text=True
        ).strip()
    vllm_op = getattr(getattr(torch.ops, args.vllm_namespace), "silu_and_mul")
    patch = subprocess.check_output(["git", "diff", "--binary"])
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff_sha256": hashlib.sha256(patch).hexdigest(),
        "vllm": baseline,
        "seed": 42,
        "gate_std": 3,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "graph_unroll": 32,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".patch").write_bytes(patch)
    with torch.inference_mode():
        for dtype_name in args.dtypes.split(","):
            dtype = getattr(torch, dtype_name)
            for tokens in map(int, args.tokens.split(",")):
                for width in map(int, args.widths.split(",")):
                    gate = torch.randn(tokens, width, device="cuda", dtype=dtype) * 3
                    up = torch.randn_like(gate)
                    functions = make_functions(gate, up, vllm_op)
                    expected = F.silu(gate) * up
                    g64 = gate.double()
                    activated = (g64 / (1 + torch.exp(-g64))).to(dtype)
                    oracle = (activated.double() * up.double()).to(dtype)
                    accuracy = {}
                    for name, function in functions.items():
                        actual = function()
                        torch.testing.assert_close(actual, expected)
                        accuracy[name] = {
                            "vs_torch": errors(actual, expected),
                            "vs_staged_fp64": errors(actual, oracle),
                        }
                    timing = measure(functions, args.repeats, args.iterations)
                    row = {
                        "dtype": dtype_name,
                        "tokens": tokens,
                        "width": width,
                        "accuracy": accuracy,
                        "timing": timing,
                    }
                    report["rows"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    medians = {
                        name: round(value["graph_median_us"], 3) for name, value in timing.items()
                    }
                    print(dtype_name, tokens, width, medians, flush=True)
        if args.profile_dir:
            gate = torch.randn(128, 5633, device="cuda", dtype=torch.bfloat16)
            up = torch.randn_like(gate)
            profile(make_functions(gate, up, vllm_op), args.profile_dir)


if __name__ == "__main__":
    main()
