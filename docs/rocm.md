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
| Packed prefill / `prefill_uva` | Yes (Triton) | Single-sequence validated; multi-sequence packed is not qualified |
| Wavefront chunked prefill | Yes | `--wavefront-chunked-prefill` |
| Prefix cache | Yes | `enable_prefix_caching=True` |
| Preemption and recovery | Yes | `enable_preemption=True`, async recovery |
| Token pipeline | Yes | `SpeculativeConfig(mode="pipeline")` |
| Synchronous self-speculation | Yes | Block mode, fixed depth |
| Asynchronous self-speculation | Yes | Async suite coverage |
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
median of three runs is reported (32 output tokens, same `ouro_delayed` exit
mode for both arms):

| Workload | Baseline tok/s | All features tok/s | Speedup |
| --- | ---: | ---: | ---: |
| short, batch 128 | 977 | 2733 | **2.80x** |
| short, batch 256 | 1136 | 2757 | **2.43x** |

`all` enables async scheduling, CUDA Graphs, static buffers, power-of-two
padding, UVA, packed prefill, WCPB and prefix cache. The gain is mostly graph
replay plus batching: per-token host work, not GEMM throughput, was the
bottleneck.

Micro-tuning on the same `all` configuration (batch 128):

| Setting | tok/s |
| --- | ---: |
| default | 2733 |
| `HIP_FORCE_DEV_KERNARG=1` | 2781 |
| `HSA_ENABLE_INTERRUPT=0` | 2843 |
| both | 2827 |

PyTorch already reports `preferred_blas_library=hipblaslt` on this stack;
forcing it explicitly did not help. `rocBLAS` fallback was not needed.

### Per-feature cold matrix

Single cold run per cell, no warm-up, 64 output tokens for short and 16 for
long; values include JIT and graph-capture cost and are useful for direction,
not as steady-state numbers.

| Config | short b1 | short b32 | short b128 | long b8 prefill |
| --- | ---: | ---: | ---: | ---: |
| baseline | 7 | 205 | 580 | 1162 |
| async | 7 | 224 | 581 | 1158 |
| graphs | 8 | 201 | 511 | 1076 |
| static | 7 | 226 | 596 | 938 |
| uva | 7 | 212 | 565 | 1159 |
| packed (`prefill_uva`) | 8 | 192 | 556 | 931 |
| wcpb | 7 | 228 | 562 | 1046 |
| prefix cache | 6 | 214 | 613 | 1292 |
| sync self-speculation | 8 | 242 | 615 | 1205 |
| all features | 7 | 216 | 763 | 1128 |

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
