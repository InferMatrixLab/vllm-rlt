"""Matched RMSNorm / RMSNorm-then-residual accuracy and latency comparison.

The optional vLLM library must expose the source-built ``vllm_rms_review``
namespace. Normalization and weight multiplication use FP32 before conversion
to the output dtype; an optional residual is added after that conversion.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.silu_and_mul import errors, measure, profile
from vllm_rlt.kernels._extension import load_extension
from vllm_rlt.kernels.normalization import rms_norm, rms_norm_reference


def oracle(x, weight, epsilon, residual):
    value = x.double()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    result = (normalized * weight.double()).to(x.dtype)
    return result if residual is None else (residual.double() + result.double()).to(x.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-library", type=Path)
    parser.add_argument("--vllm-namespace", default="vllm_rms_review")
    parser.add_argument(
        "--vllm-source-commit", help="Source revision used to build the vLLM library"
    )
    parser.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256,1024")
    parser.add_argument("--widths", default="128,768,2048,2049,4096")
    parser.add_argument("--dtypes", default="bfloat16,float16,float32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    load_extension()
    vllm_op = None
    if args.vllm_library is not None:
        torch.ops.load_library(str(args.vllm_library.resolve()))
        vllm_op = getattr(getattr(torch.ops, args.vllm_namespace), "rms_norm")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
    # New kernels often have not been added to the Git index yet. Include their
    # complete contents rather than recording a misleading, incomplete hash.
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"]
    ).split(b"\0")
    for path in filter(None, untracked):
        diff = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", os.fsdecode(path)],
            capture_output=True,
            check=False,
        )
        if diff.returncode not in (0, 1):
            raise RuntimeError(diff.stderr.decode(errors="replace"))
        patch += diff.stdout
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".patch").write_bytes(patch)
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff_sha256": hashlib.sha256(patch).hexdigest(),
        "vllm_library": str(args.vllm_library.resolve()) if args.vllm_library else None,
        "vllm_library_sha256": (
            hashlib.sha256(args.vllm_library.read_bytes()).hexdigest()
            if args.vllm_library
            else None
        ),
        "vllm_source_commit": args.vllm_source_commit,
        "seed": 42,
        "epsilon": 1e-6,
        "tolerances": {
            "float32": {"rtol": 2e-6, "atol": 2e-6},
            "float16": {"rtol": 2e-3, "atol": 2e-3},
            "bfloat16": {"rtol": 1e-2, "atol": 1e-2},
        },
        "repeats": args.repeats,
        "iterations": args.iterations,
        "graph_unroll": 32,
        "device_status_before": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,clocks.sm",
                "--format=csv",
            ],
            text=True,
        ),
        "rows": [],
    }
    args.output.write_text(json.dumps(report, indent=2))
    with torch.inference_mode():
        for dtype_name in args.dtypes.split(","):
            dtype = getattr(torch, dtype_name)
            for width in map(int, args.widths.split(",")):
                for tokens in map(int, args.tokens.split(",")):
                    x = torch.randn(tokens, width, dtype=dtype, device="cuda")
                    residual = torch.randn_like(x)
                    weight = (torch.randn(width, device="cuda") * 0.1 + 1).to(dtype)

                    def vllm():
                        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
                        vllm_op(output, x, weight, 1e-6)
                        return output

                    for add in (False, True):
                        res = residual if add else None
                        funcs = {
                            "torch": lambda: rms_norm_reference(x, weight, 1e-6, res),
                            "rlt_fused": lambda: rms_norm(x, weight, 1e-6, res),
                        }
                        if vllm_op is not None:
                            funcs["vllm"] = (lambda: vllm() + residual) if add else vllm
                        if add:
                            funcs["rlt_separate"] = lambda: rms_norm(x, weight, 1e-6) + residual
                        ref = funcs["torch"]()
                        exact = oracle(x, weight, 1e-6, res)
                        accuracy = {}
                        for name, fn in funcs.items():
                            result = fn()
                            torch.testing.assert_close(
                                result, ref, **report["tolerances"][dtype_name]
                            )
                            accuracy[name] = {
                                "vs_torch": errors(result, ref),
                                "vs_fp64": errors(result, exact),
                            }
                        row = {
                            "dtype": dtype_name,
                            "tokens": tokens,
                            "width": width,
                            "residual": add,
                            "accuracy": accuracy,
                            "timing": measure(funcs, args.repeats, args.iterations),
                        }
                        report["rows"].append(row)
                        args.output.write_text(json.dumps(report, indent=2))
                        print(
                            dtype_name,
                            width,
                            tokens,
                            "add" if add else "norm",
                            {k: round(v["graph_median_us"], 3) for k, v in row["timing"].items()},
                            flush=True,
                        )
                    del x, residual, weight, result, ref, exact, funcs
        if args.profile_dir:
            x = torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16)
            weight = torch.ones(2048, device="cuda", dtype=x.dtype)
            residual = torch.randn_like(x)

            def vllm_profile():
                output = torch.empty_like(x)
                vllm_op(output, x, weight, 1e-6)
                return output + residual

            functions = {
                "torch_norm_add": lambda: rms_norm_reference(x, weight, 1e-6, residual),
                "rlt_norm_add": lambda: rms_norm(x, weight, 1e-6, residual),
            }
            if vllm_op is not None:
                functions["vllm_norm_add"] = vllm_profile
            profile(functions, args.profile_dir)
    report["device_status_after"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,clocks.sm", "--format=csv"],
        text=True,
    )
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
