# FlashInfer Attention Backend

The `flashinfer` backend runs paged decode attention through FlashInfer's XQA
kernels (`flashinfer-python >= 0.7.0`, `trtllm_batch_decode_with_kv_cache`,
`backend="xqa"`). It is the validated attention path for NVIDIA consumer
Blackwell (SM120, e.g. RTX 5090), where the pinned FlashAttention releases
reject paged KV.

## Scope

- XQA batches one query position per row for decode and prefill. Each row
  attends to its own visible KV prefix. The backend supports CUDA Graph
  capture and asynchronous scheduling.
- XQA declares `packed_prefill=False`. Triton can also run packed prefill.
  Disable packing for both backends when comparing their single-query paths.
- No prefill/decode disaggregation; PD entrypoints reject the backend
  explicitly.
- On SM120, `--kv-cache-dtype fp8_e4m3` selects E4M3 KV with a fixed
  scale calibrated at startup. Query and output remain BF16.
- Model dtype is `bfloat16` only. `float16` is rejected at startup: the kernels compute
  correctly, but flashinfer-python 0.7.0.post1 crashes the process (host
  SIGFPE) while building the f16 JIT module on first in-process use.
- `--block-size` must be one of `16, 32, 64, 128` (compiled XQA page-size
  variants) and `head_dim` must be divisible by 16 within `[16, 512]`.
  For `head_dim > 256`, XQA requires at most 16 query heads per KV head.

## Installation and runtime requirements

```bash
pip install -e ".[flashinfer]"
```

The XQA kernels are compiled just in time on first use, which imposes two
environment requirements beyond the package install:

1. **A CUDA toolkit with `nvcc >= 12.9` reachable via `CUDA_HOME`.** SM120
   kernels are built for the `compute_120f` architecture family, which needs
   CUDA 12.9 or newer. Without `nvcc`, startup falls back to comparing against
   the CUDA version PyTorch was built with, and SM 12.x devices are rejected
   with `SM 12.x requires CUDA >= 12.9`.
2. **That toolkit's `lib` directory on `LD_LIBRARY_PATH`.** The compiled
   modules link `libcudart.so.13`; pip-installed toolkits keep it outside the
   default loader path. With the pip toolkit layout
   (`nvidia-cuda-nvcc`/`nvidia-cuda-runtime`/`nvidia-cuda-cccl`/...), the
   unified tree lives under `site-packages/nvidia/cu13` and may need
   `lib64/libcudart.so` and `lib64/stubs/libcuda.so` developer symlinks, which
   the wheels do not ship.

The first call on a machine compiles for a few seconds and caches under
`~/.cache/flashinfer/<package-version>/<arch>/`; later calls, including CUDA
Graph capture warmups, reuse the cache.

## Running

```bash
vllm-rlt --model "$MODEL" --revision "$REVISION" --prompt "What is 2 + 2?" \
    --device cuda --dtype bfloat16 --attention-backend flashinfer
```

The adapter holds one 128 MiB zero-initialized workspace for the kernel's
scratch and softmax statistics. The first nonempty call allocates it and later
calls reuse it. Each call still prepares an output tensor, page table and
sequence lengths. Padded block-table columns are
clamped non-negative before the call: the engine produces `-1` padding on the
synchronous path, `0` on the asynchronous path, and stale-but-valid ids in
CUDA Graph replay beyond the current width. The kernel bounds reads by
`seq_lens`, so all three padding states stay invisible.

## Choosing between Triton and FlashInfer

The inference CLI defaults to Torch. The HTTP server and native GSM8K harness
default to Triton. With `--attention-backend auto`, the engine considers
FlashInfer on SM120 (compute capability 12.0) ahead of Triton. Other devices
accepted by the adapter require explicit selection and log a validation warning.

Both XQA and Triton can split a single query's KV prefix. Triton's single-query
kernel launches programs over rows, heads and KV splits, then reduces partial
outputs. Measure performance at the batch size and context length of interest.

## Validation status

The RTX 5090 GSM8K run below used the Ouro-1.4B checkpoint. The GPU regression
tests use a tiny Ouro model for cross-backend comparisons of tokens and
prescribed exit depths in synchronous, asynchronous and multi-stream runs.
The graph tests compare eager execution and replay within each backend.
Attention operator tests compare numerical outputs against a paged reference
with tolerances.

## Accuracy note (GSM8K-87, single greedy pass)

At commit `cd368ce` on RTX 5090, both native backends used Ouro-1.4B,
BF16 weights and KV, page size 16, four fixed loops, greedy decoding, per-row
prefill and the frozen strict scorer. FlashInfer `0.7.5.post1` scored **58/87**
and Triton **62/87**. The generated token sequences matched on **61/87**.
All generated tokens exited at depth four.

| Question | XQA final answer | Triton final answer | Strict scoring |
| --- | --- | --- | --- |
| 203 | `27` | `31` | XQA gave the wrong value |
| 426 | `$11` | `11` | XQA amount was correct, extraction failed |
| 745 | `$1100` | `1100` | XQA amount was correct, extraction failed |
| 819 | `$6,250` | `6,250` | XQA amount was correct, extraction failed |

At the first differing attention row on question 203, rounding the unnormalized
softmax weights to BF16 in both numerator and denominator reproduced all 2,048
XQA output components. The Triton row matched the FP64 reference rounded to BF16.
This experiment identifies the rounding behavior at that row. The three
currency-format losses were established by replaying the scorer.
See the [matched benchmark commands and accuracy gate](accuracy.md#attention-backend-runs).
