"""A/B the scalar and batched sparse decode paths inside the merged tree.

Both implementations live in ``vllm_rlt.attention.sparse`` and share the plan and
store contract, so this isolates the batching gain from the numerical contract:

* ``scalar``  ``reference_paged_attention`` (row loop, the oracle),
* ``batched`` ``batched_paged_attention`` (equal-length row groups, CUDA).

Capture rows are also measured with and without the dense-backend handoff, which
the full-retention control requires.

Usage:
    python benchmarks/sparse_decode_batched_ab.py --device cuda --rows 32 --context 512
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
    batched_paged_attention,
    reference_paged_attention,
)
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.worker.attention_reference import AttentionReferenceStore


def build(rows, context, heads, head_dim, kv_heads, layers, block_size, retention, device):
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
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--retention", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)

    report = {
        "device": str(device),
        "rows": args.rows,
        "context": args.context,
        "retention": args.retention,
        "layers": args.layers,
    }
    for mode, name in ((CAPTURE, "capture"), (CONSUME, "consume")):
        key, value, table, lengths, q, store = build(
            args.rows,
            args.context,
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
        # Seed a capture pass so consume rows have state.
        seed_plan = store.build_plan(owners, [CAPTURE] * args.rows, key_retention=args.retention)
        reference_paged_attention(
            q,
            key,
            value,
            table,
            lengths,
            seed_plan,
            layer=0,
            dense_attention=torch_paged_attention,
        )
        plan = store.build_plan(owners, [mode] * args.rows, key_retention=args.retention)
        dense = torch_paged_attention
        scalar = timed(
            lambda: reference_paged_attention(
                q, key, value, table, lengths, plan, layer=0, dense_attention=dense
            ),
            args.repeats,
            device,
        )
        batched = timed(
            lambda: batched_paged_attention(
                q, key, value, table, lengths, plan, layer=0, dense_attention=dense
            ),
            args.repeats,
            device,
        )
        batched_oracle = timed(
            lambda: batched_paged_attention(q, key, value, table, lengths, plan, layer=0),
            args.repeats,
            device,
        )
        report[name] = {
            "scalar_ms": round(scalar, 3),
            "batched_ms": round(batched, 3),
            "batched_no_dense_handoff_ms": round(batched_oracle, 3),
            "speedup": round(scalar / batched, 2),
        }
        store.release(lease)

    dense_ms = timed(
        lambda: torch_paged_attention(q, key, value, table, lengths), args.repeats, device
    )
    report["dense_attention_ms"] = round(dense_ms, 3)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(
        f"sparse decode path A/B — {device}, rows={args.rows}, context={args.context}, "
        f"retention={args.retention}, layers={args.layers}"
    )
    for name in ("capture", "consume"):
        values = report[name]
        print(
            f"  {name:8s} scalar {values['scalar_ms']:7.3f} ms  "
            f"batched {values['batched_ms']:7.3f} ms  "
            f"speedup {values['speedup']:5.2f}x  "
            f"(batched w/o dense handoff {values['batched_no_dense_handoff_ms']:7.3f} ms)"
        )
    print(f"  dense attention reference: {dense_ms:.3f} ms")


if __name__ == "__main__":
    main()
