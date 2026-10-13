# FlashInfer Attention Backend

The `flashinfer` backend runs paged decode attention through FlashInfer's XQA
kernels (`flashinfer-python >= 0.7.0`, `trtllm_batch_decode_with_kv_cache`,
`backend="auto"`). It is the validated attention path for NVIDIA consumer
Blackwell (SM120, e.g. RTX 5090), where the pinned FlashAttention releases
reject paged KV.

## Scope

- Decode and per-token prefill, identical semantics to the Triton path,
  including CUDA Graph capture and asynchronous scheduling.
- No packed prefill: the engine keeps the FA4-only packed path closed, so
  prefill runs per token exactly as with Triton.
- No prefill/decode disaggregation; PD entrypoints reject the backend
  explicitly.
- `bfloat16` only. `float16` is rejected at startup: the kernels compute
  correctly, but flashinfer-python 0.7.0.post1 crashes the process (host
  SIGFPE) while building the f16 JIT module on first in-process use.
- `--block-size` must be one of `16, 32, 64, 128` (compiled XQA page-size
  variants) and `head_dim` must be divisible by 16 within `[16, 512]`.

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
vllm-rlt --model <ouro-checkpoint> --revision <revision> \
    --device cuda --dtype bfloat16 --attention-backend flashinfer
```

The adapter holds one 128 MiB zero-initialized workspace for the kernel's
scratch and softmax statistics; it is allocated once at startup and reused, so
graph capture performs no new allocations. Padded block-table columns are
clamped non-negative before the call: the engine produces `-1` padding on the
synchronous path, `0` on the asynchronous path, and stale-but-valid ids in
CUDA Graph replay beyond the current width. The kernel bounds reads by
`seq_lens`, so all three padding states stay invisible.

## Choosing between Triton and FlashInfer

SM120 is the only validated target for this backend; on it, FlashInfer decodes
noticeably faster at low concurrency (single-request decode measured ~2x
against the Triton kernel at 256-token contexts) because the XQA kernel
parallelizes a single query row across KV chunks, while the Triton kernel
spawns one program per (row, head). At high concurrency with long contexts the
two are comparable; choose Triton when its determinism properties matter more
and FlashInfer for single-user or low-batch latency. SM9/SM10 devices are
accepted but unvalidated — prefer FlashAttention there until validated
configurations are recorded.

## Validation status

Validated on RTX 5090 (SM120, 32 GB) with Ouro-1.4B BF16, `block_size=16`,
fixed four-loop decode: oracle parity against the reference paged attention,
token/exit-depth equality against the Triton backend across synchronous,
asynchronous, and multi-stream execution with static buffers and graph replay,
and end-to-end greedy generation.

## Accuracy note (GSM8K-87, single greedy pass)

On the frozen GSM8K-87 protocol (HF baseline 59/87), a fixed four-loop greedy
pass scores 58/87 with FlashInfer versus 61/87 with Triton. 64 of 87
generations are token-identical between the two backends; the remaining 23
diverge at scattered positions with unchanged exit depths, and the net
correct-answer flips are 3 against, 0 for. Kernel-level comparison against a
float32 oracle at 352-1024 token contexts shows the Triton kernel is
near-exact (max deviation up to 5e-4) while XQA's split reductions deviate by
up to 4e-3 — within normal BF16 kernel variation, but enough to flip
near-tied greedy argmax decisions during long chain-of-thought decoding.
Triton remains the default backend; treat this note as the expected greedy
variation when opting into FlashInfer on SM120.

