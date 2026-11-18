"""Decode-attention kernel timing for WISE modes (CUDA events, one layer call).

Ouro-1.4B shapes (16 heads, head_dim 128, bf16). Sparse rows use a fixed
support with the given density, newest block always kept. FlashAttention
sparse rows attend the compacted page table of the same support (head union,
identical across heads here, so the union equals each head's support).
"""

import argparse
import json
import math

import torch

from vllm_rlt.kernels.triton_attention import paged_attention
from vllm_rlt.kernels.wise_attention import (
    MODE_DENSE,
    MODE_DISCOVER,
    MODE_SPARSE,
    wise_paged_attention,
    wise_scratch,
)
from vllm_rlt.kernels.wise_flash import compact_page_tables


def timeit(fn, iters):
    """GPU time per call from CUDA-graph replays (no Python launch overhead, as in the engine)."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(10):
            fn()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    replays = max(1, iters // 10)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (replays * 10) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="64x1024,40x2048,20x4096")
    p.add_argument("--densities", default="0.17,0.33")
    p.add_argument("--iters", type=int, default=50)
    a = p.parse_args()
    heads, dim = 16, 128
    for case in a.cases.split(","):
        rows, length = map(int, case.split("x"))
        for page, backends in ((16, ("triton", 3, 4)), (256, (2,))):
            pages = math.ceil(length / page)
            k = torch.randn((rows * pages, page, heads, dim), device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            q = torch.randn((rows, heads, dim), device="cuda", dtype=torch.bfloat16)
            tables = (
                torch.randperm(rows * pages, device="cuda")
                .to(torch.int32)
                .view(rows, pages)
                .contiguous()
            )
            lengths = torch.full((rows,), length, device="cuda", dtype=torch.int32)
            max_blocks = 1 << math.ceil(math.log2(math.ceil(pages * page / 32)))
            slots = torch.arange(rows, device="cuda", dtype=torch.int32)
            scratch = wise_scratch(rows, heads, dim, max_blocks, "cuda")
            stats = torch.zeros(2, device="cuda", dtype=torch.int64)
            out = {"rows": rows, "length": length, "page": page}
            for backend in backends:
                if backend == "triton":
                    out["triton_dense"] = timeit(
                        lambda: paged_attention(q, k, v, tables, lengths), a.iters
                    )
                    for mode, name in (
                        (MODE_DENSE, "wise_split_dense"),
                        (MODE_DISCOVER, "wise_discover"),
                    ):
                        modes = torch.full((rows,), mode, device="cuda", dtype=torch.int32)
                        support = torch.ones(
                            (rows, 1, heads, max_blocks), device="cuda", dtype=torch.uint8
                        )
                        out[name] = timeit(
                            lambda: wise_paged_attention(
                                q, k, v, tables, lengths, modes, slots, support, 0, scratch, stats
                            ),
                            a.iters,
                        )
                from vllm_rlt.kernels.flash_attention import FlashPagedAttention

                flash = None
                if backend != "triton":
                    try:
                        flash = FlashPagedAttention(
                            torch.device("cuda"), torch.bfloat16, dim, page, f"flash_attn_{backend}"
                        )
                    except (ImportError, ValueError) as error:
                        out[f"fa{backend}"] = str(error)
                        continue
                    out[f"fa{backend}_dense"] = timeit(
                        lambda: flash(q, k, v, tables, lengths), a.iters
                    )
                    out[f"fa{backend}_dense_lse"] = timeit(
                        lambda: flash(q, k, v, tables, lengths, return_lse=True), a.iters
                    )
                for density in map(float, a.densities.split(",")):
                    blocks = torch.arange(max_blocks, device="cuda")
                    count = math.ceil(length / 32)
                    stride = max(1, round(1 / density))
                    chosen = ((blocks % stride == 0) | (blocks == count - 1)) & (blocks < count)
                    support = (
                        chosen[None, None, None, :]
                        .expand(rows, 1, heads, -1)
                        .to(torch.uint8)
                        .contiguous()
                    )
                    modes = torch.full((rows,), MODE_SPARSE, device="cuda", dtype=torch.int32)
                    if backend == "triton":
                        out[f"wise_sparse_{density}"] = timeit(
                            lambda: wise_paged_attention(
                                q, k, v, tables, lengths, modes, slots, support, 0, scratch, stats
                            ),
                            a.iters,
                        )
                    else:
                        ot, ol = torch.empty_like(tables), torch.empty_like(lengths)

                        def run():
                            t, n = compact_page_tables(
                                tables, lengths, modes, slots, support, 0, page, ot, ol
                            )
                            return flash(q, k, v, t, n)

                        out[f"fa{backend}_sparse_{density}"] = timeit(run, a.iters)
                        out[f"fa{backend}_compact_only_{density}"] = timeit(
                            lambda: compact_page_tables(
                                tables, lengths, modes, slots, support, 0, page, ot, ol
                            ),
                            a.iters,
                        )
            print(
                json.dumps(
                    {
                        key: round(val, 1) if isinstance(val, float) else val
                        for key, val in out.items()
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
