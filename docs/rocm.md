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
| Ouro-1.4B, Ouro-2.6B | Yes | BF16, native checkpoint loading |
| Nanbeige4.2-3B | Yes | Two-loop inference |
| Triton paged attention | Yes | Decode and prefill |
| Torch reference attention | Yes | Correctness fallback |
| FlashAttention 2/3/4 | No | NVIDIA-only kernels |
| Async scheduling | Yes | `ouro_delayed`, `random_lookahead`, `trace` |
| CUDA Graph replay | Yes | Triton backend, static buffers |
| Multi-stream execution | Yes | `multi_stream` |
| UVA host metadata | Yes (probed) | `hipHostGetDevicePointer` must equal the host address |
| Packed prefill / `prefill_uva` | No | FA4-specific; ROCm uses per-row prefill |
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
| Ouro-1.4B | 1176 | 18.4 |
| Nanbeige4.2-3B | 1988 | 31.1 |

Configuration: `--attention-backend triton --async-scheduling --static-buffers
--pad-to-power-of-two --cuda-graphs --cuda-graph-max-graphs 64 --max-num-seqs
128 --max-num-batched-tokens 16384`. Numbers are one run, not a benchmark
contract; the first request includes Triton compilation.

The CPU regression suite passes unchanged on ROCm. GPU tests use the same
`--run-gpu` flag; FlashAttention, packed-prefill and multi-GPU PD tests skip
because those implementations or devices are unavailable.

## Troubleshooting

- `torch.version.hip` is `None`: a CUDA PyTorch wheel is installed. Reinstall
  with the ROCm index above.
- `HIP error: invalid device function` or an illegal memory access inside
  Triton: verify the GPU architecture is supported by the installed ROCm and
  Triton versions, and that `torch.cuda.get_device_name(0)` reports the device.
- `flash_attn currently requires an NVIDIA CUDA device`: use
  `--attention-backend triton`; FlashAttention cannot run on ROCm.
