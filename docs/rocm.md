# AMD ROCm

vllm-rlt runs on AMD Instinct GPUs through PyTorch's ROCm build and the Triton
attention backend. The runtime, scheduler, serving entrypoints, asynchronous
scheduling, static buffers, CUDA Graph replay and recurrent-loop features are
shared with the CUDA path. NVIDIA-only packages are not required.

## Requirements

- Linux with a ROCm 7.x driver and a `gfx9` Instinct GPU (MI300/MI350 class).
- Python 3.10+ and a ROCm-enabled PyTorch build. The validated stack is
  PyTorch 2.14.1 + ROCm 7.2 with Triton 3.5.1 on an MI350X (`gfx950`).
- `rocminfo`/`rocm-smi` must report the GPU and `torch.cuda.is_available()`
  must be `True` inside the environment.

Install PyTorch from the ROCm wheel index, then the project from the
repository root. FlashAttention and NIXL are not needed.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install -e .
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
```

`torch.version.hip` must be set. `vllm-rlt` does not build a CUDA extension, so
there is no `nvcc` step and no separate ROCm package.

## Run

Use the Triton backend explicitly, or `auto` (ROCm auto selects Triton and then
Torch; it never tries FlashAttention):

```bash
python -m vllm_rlt.entrypoints.cli \
  --model /models/Ouro-1.4B \
  --device cuda --dtype bfloat16 \
  --attention-backend triton \
  --prompt "The capital of France is" --max-tokens 32
```

The HTTP server and the Python `LLM` facade take the same arguments
(`attention_backend="triton"` in Python). `CUDA_VISIBLE_DEVICES` is still the
device selector because PyTorch exposes ROCm devices through `torch.cuda`.

## Feature support

| Feature | ROCm | Notes |
| --- | --- | --- |
| Ouro-1.4B | Yes | Full validation and A/B benchmark |
| Ouro-2.6B | Yes | Real-checkpoint generation; shared Ouro loader |
| Ouro-1.4B/2.6B-Thinking | Yes | Real-checkpoint generation (bounded) |
| Nanbeige4.2-3B | Yes | Two-loop inference |
| Huginn-0125 | Yes | Synchronous fixed-depth, LAST_EXITED KV, graph replay 32 tokens |
| Triton paged attention | Yes | Decode and prefill |
| Torch reference attention | Yes | Correctness fallback |
| FlashAttention 2/3/4 | No | NVIDIA-only kernels |
| Async scheduling | Yes | `ouro_delayed`, `random_lookahead`, `trace` |
| CUDA Graph replay | Yes | Triton backend, static buffers |
| Multi-stream execution | Yes | `multi_stream` |
| UVA host metadata | Yes (probed) | `hipHostGetDevicePointer` must equal the host address |
| Packed prefill / `prefill_uva` | Opt-in | FP32 greedy matches per-row exactly; BF16 attention differs by <=0.0156 and greedy can diverge, so it stays out of `all` until an accuracy check |
| Wavefront chunked prefill | Yes | `--wavefront-chunked-prefill` |
| Prefix cache | Yes | `enable_prefix_caching=True` |
| Priority scheduling | Yes | `priority=0` is highest; high-priority latency 7.80s → 2.22s |
| Incremental KV | Yes | +3.8% at batch 64 |
| Sparse decode (#157) | ROCm tests pass | `--sparse-decode-attention --sparse-decode-key-retention` on the #157 branch; 137 passed |
| Preemption and recovery | Yes | `enable_preemption=True`, async recovery |
| Token pipeline | Opt-in | **2.14x at 1.4B (256 out)**, 1.63x on Ouro-2.6B; short-output loses |
| Synchronous self-speculation | Opt-in | 1.41x at 1.4B (256 out); 1.18x on Ouro-2.6B |
| Asynchronous self-speculation | Opt-in | 1.48x at 1.4B (256 out); async suite |
| PD (prefill/decode split) | Not validated | Depends on the NIXL/UCX transport |
| HTTP serving | Yes | `/v1/completions` |

Mapped host metadata (UVA) is probed once per device: Triton receives host
pointers directly, so `hipHostGetDevicePointer` must return the same address.
Set `VLLM_RLT_UVA=0` to force the H2D copy path, or `VLLM_RLT_UVA=1` to skip
the probe. The decision is logged at worker startup.

## Validation

The following was measured on one MI350X (288 GiB HBM, `gfx950`) with ROCm 7.2,
PyTorch 2.14.1 and Triton 3.5.1, BF16, greedy decoding, 128 concurrent
requests with 64 output tokens each and an 80-token prompt:

| Model | Output tokens/s | Requests/s |
| --- | --- | --- |
| Ouro-1.4B | 855 | 13.4 |
| Nanbeige4.2-3B | 1012 | 15.8 |

Configuration: `--attention-backend triton --async-scheduling --static-buffers
--pad-to-power-of-two --cuda-graphs --cuda-graph-max-graphs 64 --max-num-seqs
128 --max-num-batched-tokens 16384`. Numbers are one run, not a benchmark
contract; the first request includes Triton compilation.

### Baseline and all-features result

The baseline is the same commit, model, device and wheel set with
`attention_backend="triton"`, `ExecutionConfig()` defaults, no optional
execution features, the same exit mode and `VLLM_RLT_UVA=0`. There was no
earlier ROCm baseline, so every comparison here is ROCm-relative; CUDA numbers
were not measured on this lease. One warm-up `generate` is discarded, then the
median of three runs is reported (same `ouro_delayed` exit mode for both arms):

| Workload | Baseline tok/s | All features tok/s | Speedup |
| --- | ---: | ---: | ---: |
| short, batch 1, 64 out | 18.4 | 32.8 | **1.78x** |
| short, batch 32, 64 out | 446 | 916 | **2.05x** |
| short, batch 128, 64 out | 1148 | 2909 | **2.53x** |
| short, batch 256, 64 out | 1606 | 3270 | **2.04x** |
| short, batch 512, 64 out | 1864 | 3919 | **2.10x** |
| long, batch 8, 32 out | 84.8 | 171.0 | **2.02x** |
| long, batch 32, 32 out | 153.8 | 544.4 | **3.54x** |
| long, batch 64, 32 out | 179.8 | 866.4 | **4.82x** |

Long-context prefill throughput rises from 2652 to 5350 tok/s at batch 8,
4811 to 17029 at batch 32, and 5626 to 27102 at batch 64. The long-context
high-concurrency regime is where WCPB, async scheduling and graph replay
compound.

`all` enables async scheduling, CUDA Graphs, static buffers, power-of-two
padding, UVA, WCPB and prefix cache. Packed prefill is excluded because its
engine parity is still failing (see below). The gain is graph replay plus
batching; short runs are prefill-heavy, so the 256-output decode run is the
conservative 1.72x result (baseline 6.3 ms/step, all 3.4 ms/step at batch 128).

Micro-tuning on the same `all` configuration (batch 128, no packed):

| Setting | tok/s |
| --- | ---: |
| default | 2798 |
| `HIP_FORCE_DEV_KERNARG=1` | 2781 |
| `HSA_ENABLE_INTERRUPT=0` | 2843 |
| both | 2827 |

PyTorch already reports `preferred_blas_library=hipblaslt`; forcing it
explicitly did not improve throughput, so no rocBLAS fallback is used.

### Per-feature warmed A/B

Batch 128, 32 output tokens, one warm-up discarded, median of three runs.
Non-spec configs use `baseline_delayed` (970 tok/s) as the reference; spec
configs use the `ouro` baseline (994 tok/s), which is the only legal exit mode
for them.

| Config | tok/s | vs reference |
| --- | ---: | ---: |
| baseline (non-spec) | 970 | 1.00x |
| async scheduling | 1025 | 1.06x |
| CUDA Graphs | 986 | 1.02x |
| static buffers + pad2 | 1056 | 1.09x |
| UVA | 1019 | 1.05x |
| WCPB | 1402 | **1.45x** |
| prefix cache | 1105 | 1.14x |
| all features | 2798 | **2.88x** |
| baseline (spec family) | 994 | 1.00x |
| sync self-speculation | 510 | 0.51x |
| token pipeline | 947 | 0.95x |
| async self-speculation | 883 | 0.89x |

Priority and incremental KV are positive: with 64 low- and 8 high-priority
requests at 16 slots, high-priority median completion drops from 7.80s (fcfs)
to 2.22s (priority) while total time moves 7.81s to 8.15s; incremental KV gives
817 -> 848 tok/s at batch 64 (+3.8%).

Speculation is **workload-dependent**, not simply negative. At 32 output
tokens the 80-token prefill dominates and it loses (best 0.95x); with 256
output tokens at batch 128 the decode-heavy path wins on Ouro-1.4B:

| Config | tok/s | vs baseline |
| --- | ---: | ---: |
| baseline | 1527 | 1.00x |
| pipeline K4/draft1 | 3270 | **2.14x** |
| pipeline K6/draft1 | 3249 | 2.13x |
| pipeline K2/draft1 | 2365 | 1.55x |
| pipeline K4/draft2 | 2254 | 1.48x |
| pipeline K2/draft2 | 2251 | 1.47x |
| block K2/draft1 | 2148 | **1.41x** |
| pipeline K1/draft1 | 1905 | 1.25x |
| async K2/draft1 | 2314 | 1.48x |

A confirmation run measured pipeline K4/draft1 at 2842 tok/s (1.82x), block
K2/draft1 at 2251 (1.44x), and async K2/draft1 at 2314 (1.48x), with greedy
token match 0.66-0.74 against baseline.

On Ouro-2.6B (batch 64, 128 output): pipeline K2/draft1 814 vs 500 (1.63x),
K4/draft1 782 (1.56x), block K2/draft1 591 (1.18x). Greedy token match against
the non-speculative baseline is 0.65-0.80, and a 20-question GSM8K check scored
0/20 for both arms, so there is no accuracy signal yet; speculation stays
opt-in and is not a default.

### Not done on ROCm

- **Packed prefill (`prefill_uva`)**: root cause located as BF16 accumulation,
  not metadata: FP32 greedy output matches per-row exactly, while BF16
  per-call attention differs by <=0.0156 and can flip greedy tokens. It stays
  opt-in until an accuracy evaluation, and is excluded from `all`.
- **PD/KV transfer**: needs two GPUs and a ROCm NIXL/UCX transport.
- **KV quantization / INT4 residual cache**: not present on the current main
  runtime; tracked by RFC #156. Sparse decode from #157 passes its ROCm suites
  on a stacked branch but is not shipped here.
- **Profiling flags**: `--profile` / `--profile-activities` are untested on
  ROCm; `rocprofv3` is available for external traces.
- **FlashAttention**: NVIDIA-only by construction.
- **Multi-node**: out of scope for this device.

The CPU regression suite passes unchanged on ROCm. GPU tests use the same
`--run-gpu` flag; FlashAttention, FA4-specific packed-prefill and multi-GPU PD tests
skip because those implementations or devices are unavailable.

## Troubleshooting

- `torch.version.hip` is `None`: a CUDA PyTorch wheel is installed. Reinstall
  with the ROCm index above.
- `HIP error: invalid device function` or an illegal memory access inside
  Triton: verify the GPU architecture is supported by the installed ROCm and
  Triton versions, and that `torch.cuda.get_device_name(0)` reports the device.
- `flash_attn currently requires an NVIDIA CUDA device`: use
  `--attention-backend triton`; FlashAttention cannot run on ROCm.
