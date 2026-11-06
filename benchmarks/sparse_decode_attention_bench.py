"""Sparse decode attention reference-path benchmark (RFC #156, section 3.2).

Measures the reference implementation of the decode correction against the
dense reference path on identical paged-KV inputs:

* ``dense``   one loop of full attention over ``N`` keys,
* ``capture`` the reference loop: dense attention plus retained statistics,
* ``consume`` a later loop: correction over ``k = ceil(N * retention)`` keys.

A four-loop decode therefore costs ``4 x dense`` in the baseline and
``dense + capture + 2 x consume`` with sparse decode attention. This measures
the algorithm on the reference path, not an optimized kernel: the reported
speedup is a work proxy, not a serving claim (RFC #156 section 9).

Usage:
    python benchmarks/sparse_decode_attention_bench.py --device cuda
    python benchmarks/sparse_decode_attention_bench.py --device cpu --context 512
"""

import argparse
import json
import math
import time

import torch

from vllm_rlt.attention.sparse import (
    reference_statistics,
    retained_count,
    sparse_correction,
)
from vllm_rlt.kernels.paged_attention import torch_paged_attention


def build_paged(context, heads, head_dim, kv_heads, block_size, device, dtype):
    blocks = math.ceil(context / block_size)
    shape = (blocks, block_size, kv_heads, head_dim)
    generator = torch.Generator(device="cpu").manual_seed(0)
    key_cache = torch.randn(shape, generator=generator, dtype=torch.float32).to(device, dtype)
    value_cache = torch.randn(shape, generator=generator, dtype=torch.float32).to(device, dtype)
    table = torch.arange(blocks, device=device, dtype=torch.int32).reshape(1, blocks)
    lengths = torch.tensor([context], device=device, dtype=torch.int32)
    q = torch.randn((1, heads, head_dim), generator=generator, dtype=torch.float32)
    return key_cache, value_cache, table, lengths, q.to(device, dtype)


def timed(function, repeats, device):
    """Median wall time of ``repeats`` calls, after warmup."""
    for _ in range(5):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--retention", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    key_cache, value_cache, table, lengths, q = build_paged(
        args.context, args.heads, args.head_dim, args.kv_heads, args.block_size, device, dtype
    )
    block_size = args.block_size
    positions = torch.arange(args.context, device=device)
    blocks = table[0, positions // block_size]
    offsets = positions % block_size
    keys = key_cache[blocks, offsets]
    values = value_cache[blocks, offsets]
    retained = retained_count(args.context, args.retention)

    dense_ms = timed(
        lambda: torch_paged_attention(q, key_cache, value_cache, table, lengths),
        args.repeats,
        device,
    )
    capture_ms = timed(
        lambda: reference_statistics(
            q[0], keys, values, key_retention=args.retention, key_bucket=retained
        ),
        args.repeats,
        device,
    )
    output, contribution, mass, selected = reference_statistics(
        q[0], keys, values, key_retention=args.retention, key_bucket=retained
    )
    groups = args.heads // args.kv_heads
    retained_keys = key_cache[
        table[0, selected.long() // block_size],
        selected.long() % block_size,
        (torch.arange(args.heads, device=device) // groups)[:, None].expand_as(selected),
    ]
    retained_values = value_cache[
        table[0, selected.long() // block_size],
        selected.long() % block_size,
        (torch.arange(args.heads, device=device) // groups)[:, None].expand_as(selected),
    ]
    consume_ms = timed(
        lambda: sparse_correction(q[0], retained_keys, retained_values, mass, output, contribution),
        args.repeats,
        device,
    )
    corrected = sparse_correction(q[0], retained_keys, retained_values, mass, output, contribution)
    # The reference loop and the later loop hold different KV, so the retained
    # set chosen at loop 2 is applied to loop 3 keys. Compare against dense
    # attention over that later KV to measure the approximation, not the
    # same-loop identity.
    later_keys = keys * 1.05 + 0.02
    later_values = values * 0.97 - 0.01
    later_retained_keys = retained_keys * 1.05 + 0.02
    later_retained_values = retained_values * 0.97 - 0.01
    dense_later = reference_statistics(q[0], later_keys, later_values, key_retention=1.0)[0]
    dense_reference = reference_statistics(q[0], keys, values, key_retention=1.0)[0]
    corrected_later = sparse_correction(
        q[0], later_retained_keys, later_retained_values, mass, output, contribution
    )
    divergence = float((corrected_later - dense_later).abs().max())
    same_loop_error = float((corrected - dense_reference).abs().max())

    dense_total = args.layers * args.loops * dense_ms
    sparse_total = args.layers * (dense_ms + capture_ms + (args.loops - 2) * consume_ms)
    element = 2 if args.dtype == "bfloat16" else 4
    store_bytes = args.layers * args.heads * (retained * 4 + 4 + element * args.head_dim * 2)
    kv_bytes = args.layers * 2 * args.kv_heads * args.head_dim * element

    report = dict(
        device=str(device),
        dtype=args.dtype,
        context=args.context,
        retained=retained,
        retention=args.retention,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        dense_ms=round(dense_ms, 4),
        capture_ms=round(capture_ms, 4),
        consume_ms=round(consume_ms, 4),
        dense_decode_ms=round(dense_total, 3),
        sparse_decode_ms=round(sparse_total, 3),
        decode_speedup=round(dense_total / sparse_total, 3),
        keys_dense=args.context,
        keys_sparse=retained,
        key_ratio=round(retained / args.context, 4),
        reference_state_bytes_per_slot=store_bytes,
        kv_bytes_per_token=kv_bytes,
        state_over_kv_ratio=round(store_bytes / kv_bytes, 3),
        same_loop_identity_error=same_loop_error,
        later_loop_max_abs_divergence=divergence,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"sparse decode attention reference path — {report['device']} {args.dtype}")
    print(
        f"  context={args.context} retained={retained} retention={args.retention} "
        f"layers={args.layers} heads={args.heads} head_dim={args.head_dim}"
    )
    print(f"  dense row   {dense_ms:8.4f} ms")
    print(f"  capture row {capture_ms:8.4f} ms")
    print(f"  consume row {consume_ms:8.4f} ms  (k={retained})")
    print(
        f"  {args.loops}-loop decode: dense {dense_total:8.3f} ms  "
        f"sparse {sparse_total:8.3f} ms  speedup {dense_total / sparse_total:5.2f}x"
    )
    print(f"  keys touched per row: {args.context} -> {retained} ({retained / args.context:.1%})")
    print(
        f"  reference state {store_bytes / 1024:.1f} KiB/slot vs KV "
        f"{kv_bytes / 1024:.1f} KiB/token ({store_bytes / kv_bytes:.2f}x)"
    )
    print(f"  same-loop identity error (must be ~0): {same_loop_error:.3e}")
    print(f"  later-loop divergence at this retention: {divergence:.3e}")


if __name__ == "__main__":
    main()
