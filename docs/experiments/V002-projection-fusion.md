# Projection fusion: experiment record

Version V002. Branch `perf/fuse-norm-rope`, commit `0e7b253`. Parent: V001
(`01db13a`, norm/RoPE fusion). Host `lyx`, 8x A100-SXM4-80GB, torch `2.14.0+cu130`,
triton `3.8.0`, Ouro-1.4B (hidden 2048, inter 5632, 24 layers, 4 loops, MHA).

## Objective

After V001 the per-step budget was dominated by matrix multiplication: gemv 6.33 ms/step
(45.7%) + gemm 3.76 (27.1%) of 13.86 ms, while attention was 0.85 (6.1%). Hypothesis: at
batch 1 the projections are GEMV and are limited by how few bytes are in flight, so
concatenating the matrices that share an input row should raise achievable bandwidth.

## First: a measurement correction

The V000 survey quoted "30.6% of peak DRAM throughput" for a `2048x2048` GEMV. That number
was an artifact: the microbenchmark reused **one** weight matrix (8.4 MB), which the A100's
**40 MB L2** serves from cache. Re-measuring with 24 layers x 4 loops of **distinct**
weights (9.87 GB streamed) gives 46.7% for the same shape. Cache-resident microbenchmarks
of small matrices overstate the memory bottleneck; every number below uses a cold-L2 sweep.

Cold-cache efficiency vs matrix size (M=1, K=2048, bf16, measured peak 1770 GB/s):

| output width | bytes | GB/s | % peak |
| --- | --- | --- | --- |
| 2048 (q, k, v, o) | 8.4 MB | 597 | 33.7% |
| 5632 (gate, up) | 23.1 MB | 1118 | 63.2% |
| 6144 (fused qkv) | 25.2 MB | 1168 | 66.0% |
| 11264 (fused gate+up) | 46.1 MB | 1312 | 74.1% |

## Implementation

`vllm_rlt/layers/fused_projections.py::FusedWeight` holds a cached concatenation of the
source weights. Wired in at `vllm_rlt/models/ouro.py`:

- `OuroAttention`: `q/k/v` -> one `[6144, 2048]` matrix, then split the output.
- `OuroMLP`: `gate/up` -> one `[11264, 2048]` matrix, then split.

`o_proj` and `down_proj` are not fused: they consume different inputs (attention output and
the MLP intermediate), so there is nothing to concatenate.

### Why this is not an `nn.Module`, and why it is not a weakref

Three variants were tried; the first two were wrong and each was caught by a test:

| variant | failure |
| --- | --- |
| `nn.Module` holding a strong reference to the owner | cyclic module graph, `state_dict`/`_apply` hit `RecursionError` |
| holder with `weakref.ref(owner)` | `TypeError: cannot pickle 'weakref.ReferenceType'` — the engine ships the model to a worker |
| cache tensor stored as `nn.Parameter` | extra `state_dict` key, `test_native_safetensors_roundtrip` failed |
| **plain object, plain tensor, sources resolved per call** | correct |

The signature check (`id`, `data_ptr`, `dtype`, `shape` per source) is what makes the cache
rebuild after a dtype conversion or a checkpoint load instead of silently using stale
weights.

## Correctness

| gate | result |
| --- | --- |
| full suite `pytest tests/` | **1360 passed, 0 failed**, 914 skipped |
| `gate_up` split vs separate | **bitwise equal** (`torch.equal`) |
| `qkv` split vs separate | **not bitwise**; max relative diff 0.41% (0.41% < 1 BF16 ULP = 0.78%) |
| end-to-end 64 generated tokens | **identical** to the pre-fusion build, in eager and graph mode |

The `qkv` difference is a cuBLAS kernel-selection effect: N=6144 and N=2048 do not run the
same reduction, so the BF16 results differ in the last bits. Measured over all 72 Ouro
projections: 64/72 inexact, max abs 0.0156, max relative 0.004115. `gate_up` (N=11264 vs
5632) happened to be exact. The test
`test_attention_split_matches_separate_projections_within_bf16_rounding` pins the bound so a
real ordering bug cannot hide under "rounding".

## Performance

Operator level, real Ouro weights, one 24-layer pass (cold L2):

| | separate | fused | gain |
| --- | --- | --- | --- |
| projection pass | 1.905 ms | 1.350 ms | **1.411x** |

Engine level, 128-token prompt / 64-token decode, 3 reps, unprofiled:

| configuration | V001 (norm+rope) | V002 (+proj fusion) | gain |
| --- | --- | --- | --- |
| `cuda_graphs=True` | 16.694 ms/step | **15.286 ms/step** | **1.092x** |
| `cuda_graphs=False` | 59.341 ms/step | 59.518 ms/step | 0.997x (neutral) |

**The gain only materialises under CUDA graphs**, and that is expected rather than a
surprise: eager mode is host-launch-bound (59 ms/step with a 12-13 ms/step device cost), so
removing GPU work is invisible there. The graph-mode number is the meaningful one.

Ops-only profile, 47 decode steps, graphs on:

| group | V001 ms/step | V002 ms/step |
| --- | --- | --- |
| gemv | 6.33 | **3.35** |
| gemm | 3.76 | 5.17 |
| norm+rope (fused) | 1.76 | 1.74 |
| elementwise | 0.87 | 0.97 |
| attention | 0.85 | 0.86 |
| **total** | **13.86** | **12.41** (1.117x) |
| **launches/step** | **1929** | **1653** |

The gemv 6.33 -> 3.35 shift moves into the gemm bucket (5.17) because cuBLAS now picks a
GEMM-shaped kernel for the wider matrices. The 1.117x device-time gain appears as 1.092x
end-to-end because the non-graph host work around the replay does not shrink.

## What does not work, and why

- **Do not write a custom GEMV.** Cold-cache cuBLAS already reaches 74% of measured peak at
  46 MB. The remaining inefficiency at 8.4 MB is a hardware property of having too little
  data in flight, not a kernel-quality problem. `torch.compile` on the same call measured
  **3.4x slower** (59 us vs 17.6 us).
- **Do not claim gains in eager mode.** Eager is launch-bound; the operator saving is real
  but unobservable there.
- **Do not trust cache-resident microbenchmarks** for matrices under L2 size.

## Status

**Improved, retained.** Flag `VLLM_RLT_FUSE_NORM_ROPE` still gates the norm/RoPE half (and
the projection fusion rides on the same model code, so it is active regardless of that
flag). Remaining decision: whether the 0.41%-of-a-BF16-ULP projection deviation and the
<=2-ULP RMSNorm deviation are acceptable for promotion to default.

## Next

Device time is now 12.41 ms/step against a 5.6 ms weight-streaming floor (2.47 GB x 4
loops at 1770 GB/s), so roughly 2.2x of headroom remains. The largest single item is the
gemm bucket at 5.17 ms (41.7%). That is now worth an ncu pass on the fused shapes to see
whether cuBLAS is picking an appropriate tile for M=1..128, which is the next measurement,
not a guess.

## Recovery

```bash
git checkout 01db13a   # V001, norm/RoPE only
git checkout 0e7b253   # V002, this experiment
```
