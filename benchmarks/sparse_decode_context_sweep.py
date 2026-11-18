"""Dense-baseline context sweep and memory breakdown for sparse decode attention.

The acceptance baseline is the dense Triton kernel, not the scalar sparse
reference: a four-loop decode costs ``4 x dense`` with the feature off and
``dense + capture + 2 x consume`` with it on. This script measures both on
identical paged KV at increasing context, reports the ratio, and breaks memory
down into KV storage, the persistent reference pool and transient workspace.

Usage:
    python benchmarks/sparse_decode_context_sweep.py --device cuda
    python benchmarks/sparse_decode_context_sweep.py --contexts 512,2048,8192 --rows 32
"""

import argparse
import json
import math
import statistics
import time

import torch

from vllm_rlt.attention.sparse import (
    CAPTURE,
    CONSUME,
    reference_paged_attention,
    sparse_paged_attention,
)
from vllm_rlt.kernels.paged_attention import triton_paged_attention
from vllm_rlt.worker.attention_reference import AttentionReferenceStore


def build(context, rows, heads, head_dim, kv_heads, layers, block_size, retention, device):
    blocks = math.ceil(context / block_size)
    generator = torch.Generator().manual_seed(0)
    shape = (blocks, block_size, kv_heads, head_dim)
    key = torch.randn(shape, generator=generator).to(device, torch.bfloat16)
    value = torch.randn(shape, generator=generator).to(device, torch.bfloat16)
    table = torch.arange(blocks, device=device, dtype=torch.int32).repeat(rows, 1)
    lengths = torch.full((rows,), context, device=device, dtype=torch.int32)
    q = torch.randn((rows, heads, head_dim), generator=generator).to(device, torch.bfloat16)
    capacity = math.ceil(context * retention)
    store = AttentionReferenceStore(
        slots=rows,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        key_bucket=capacity,
        device=device,
    )
    return key, value, table, lengths, q, store


def timed(fn, repeats, device):
    for _ in range(2):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def transient_bytes(fn, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    fn()
    torch.cuda.synchronize(device)
    return max(0, torch.cuda.max_memory_allocated(device) - before)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contexts", default="256,512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--retention", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    contexts = [int(value) for value in args.contexts.split(",")]

    rows = []
    for context in contexts:
        try:
            key, value, table, lengths, q, store = build(
                context,
                args.rows,
                args.heads,
                args.head_dim,
                args.kv_heads,
                args.layers,
                args.block_size,
                args.retention,
                device,
            )
            lease = store.reserve("r", 0, args.rows, generation=1)
            owners = [(lease, row) for row in range(args.rows)]
            seed = store.build_plan(owners, [CAPTURE] * args.rows, key_retention=args.retention)
            sparse_paged_attention(
                q, key, value, table, lengths, seed, layer=0, dense_attention=triton_paged_attention
            )
            capture_plan = store.build_plan(
                owners, [CAPTURE] * args.rows, key_retention=args.retention
            )
            consume_plan = store.build_plan(
                owners, [CONSUME] * args.rows, key_retention=args.retention
            )
            dense_ms = timed(
                lambda: triton_paged_attention(q, key, value, table, lengths),
                args.repeats,
                device,
            )
            capture_ms = timed(
                lambda: sparse_paged_attention(
                    q,
                    key,
                    value,
                    table,
                    lengths,
                    capture_plan,
                    layer=0,
                    dense_attention=triton_paged_attention,
                ),
                args.repeats,
                device,
            )
            consume_ms = timed(
                lambda: sparse_paged_attention(
                    q, key, value, table, lengths, consume_plan, layer=0
                ),
                args.repeats,
                device,
            )
            scalar_consume_ms = timed(
                lambda: reference_paged_attention(
                    q, key, value, table, lengths, consume_plan, layer=0
                ),
                args.repeats,
                device,
            )
            dense_total = 4 * dense_ms
            sparse_total = dense_ms + capture_ms + 2 * consume_ms
            rows.append(
                dict(
                    context=context,
                    retained=math.ceil(context * args.retention),
                    dense_ms=round(dense_ms, 3),
                    capture_ms=round(capture_ms, 3),
                    consume_ms=round(consume_ms, 3),
                    scalar_consume_ms=round(scalar_consume_ms, 3),
                    dense_4loop_ms=round(dense_total, 3),
                    sparse_4loop_ms=round(sparse_total, 3),
                    sparse_over_dense=round(sparse_total / dense_total, 3),
                    capture_transient_mib=round(
                        transient_bytes(
                            lambda: sparse_paged_attention(
                                q,
                                key,
                                value,
                                table,
                                lengths,
                                capture_plan,
                                layer=0,
                                dense_attention=triton_paged_attention,
                            ),
                            device,
                        )
                        / 1024**2,
                        1,
                    ),
                    consume_transient_mib=round(
                        transient_bytes(
                            lambda: sparse_paged_attention(
                                q, key, value, table, lengths, consume_plan, layer=0
                            ),
                            device,
                        )
                        / 1024**2,
                        1,
                    ),
                    reference_pool_mib=round(rows_pool_bytes(args, context) / 1024**2, 1),
                    kv_mib=round(key.numel() * key.element_size() * 2 / 1024**2, 1),
                )
            )
            store.release(lease)
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            rows.append(dict(context=context, status="oom"))
            torch.cuda.empty_cache()

    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(
        f"dense-baseline sweep — {device}, rows={args.rows}, layers={args.layers}, "
        f"heads={args.heads}, head_dim={args.head_dim}, retention={args.retention}"
    )
    print(
        f"{'ctx':>7}{'k':>7}{'dense':>9}{'capture':>9}{'consume':>9}"
        f"{'4-loop dense':>13}{'4-loop sparse':>14}{'ratio':>8}"
    )
    for row in rows:
        if row.get("status") == "oom":
            print(f"{row['context']:>7}  OOM")
            continue
        print(
            f"{row['context']:>7}{row['retained']:>7}{row['dense_ms']:>9.2f}"
            f"{row['capture_ms']:>9.2f}{row['consume_ms']:>9.2f}"
            f"{row['dense_4loop_ms']:>13.2f}{row['sparse_4loop_ms']:>14.2f}"
            f"{row['sparse_over_dense']:>8.2f}x"
        )
    print()
    print(f"{'ctx':>7}{'capture ws':>12}{'consume ws':>12}{'ref pool':>11}{'KV':>9}")
    for row in rows:
        if row.get("status") == "oom":
            continue
        print(
            f"{row['context']:>7}{row['capture_transient_mib']:>10.1f}Mi"
            f"{row['consume_transient_mib']:>10.1f}Mi"
            f"{row['reference_pool_mib']:>9.1f}Mi{row['kv_mib']:>7.1f}Mi"
        )


def rows_pool_bytes(args, context):
    """Persistent reference pool for this sweep's row count."""
    capacity = math.ceil(context * args.retention)
    per_slot = capacity * 4 + 4 + 4 * 2 * args.head_dim + 8
    return args.rows * args.layers * args.heads * per_slot


if __name__ == "__main__":
    main()
