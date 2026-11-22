# FlashInfer Attention Backend

The `flashinfer` backend runs paged decode attention through FlashInfer's XQA
kernels (`flashinfer-python >= 0.7.0`, `trtllm_batch_decode_with_kv_cache`,
`backend="xqa"`). The RTX 5090 (SM120) runs this path with paged KV.
The [GSM8K-87 results](#accuracy-note-gsm8k-87-single-greedy-pass) give its
current model accuracy.

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
  Automatic selection skips XQA if the caller does not know that ratio.

## Installation and runtime requirements

```bash
pip install -e ".[flashinfer]"
```

Set `CUDA_HOME` to the toolkit that supplies `bin/nvcc` for the first JIT
build. The dynamic loader must be able to find that toolkit's CUDA runtime.
For a non-system toolkit, add its library directory to `LD_LIBRARY_PATH`.

| FlashInfer on SM120 | CUDA toolkit used by the XQA JIT |
| --- | --- |
| `0.7.0.post1` | Selects `compute_120f`, which requires CUDA 12.9 or newer. |
| `0.7.5.post1` | Builds `sm_120a` with CUDA 12.8 and selects `compute_120f` with CUDA 12.9 or newer. The CUDA 12.8 build links `libcudart.so.12`. |

At startup the adapter checks whether the installed FlashInfer compilation
context includes SM120 and whether the toolkit supports the selected target.
This also applies to explicit `FLASHINFER_CUDA_ARCH_LIST` overrides.
An incompatible target is reported immediately for
explicit selection. With `auto`, selection continues to the next backend.

The first call on a machine compiles for a few seconds and caches under
`~/.cache/flashinfer/<package-version>/<arch>/`; later calls, including CUDA
Graph capture warmups, reuse the cache.

## Running

The [Ouro-1.4B RTX 5090 recipe](recipes/ouro_xqa_sm120.md) covers the pinned
checkpoint, BF16 and calibrated E4M3 launch options, and the measured BF16
accuracy and throughput.

The adapter holds one 128 MiB zero-initialized workspace for the kernel's
scratch and softmax statistics. The first nonempty call allocates it and later
calls reuse it. The adapter makes a dense copy only when Q has non-contiguous
strides. Each call prepares an output tensor, page table and sequence lengths.
Padded block-table columns are
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
Operator tests compare XQA with FP32 paged attention and check identical
results under a permutation of rows. Ouro-1.4B has 16 Q heads, 16 KV heads
and head dimension 128. With page size 16 and 1,025–1,031 visible tokens,
reducing a four-row batch to two rows changed the XQA BF16 output by up to
`0.001953125` on the RTX 5090. XQA's KV split count depends on batch size.
The tests also cover a separate 16-Q/4-KV grouped-query configuration.

## Accuracy note (GSM8K-87, single greedy pass)

At commit `cd368ce` on RTX 5090, both native backends used Ouro-1.4B,
BF16 weights and KV, page size 16, four fixed loops, greedy decoding, per-row
prefill and the frozen strict scorer. FlashInfer `0.7.5.post1` scored **58/87**
and Triton **62/87**. The generated token sequences matched on **61/87**.
All generated tokens exited at depth four. On 2026-11-22 the same frozen run
with dense query normalization and the SM120 JIT selection fix reproduced
all 87 token sequences for each respective backend. The stored HF reference
scored 59/87. One question out of 87 exceeds the allowed one-percentage-point
regression, so XQA still fails the accuracy gate.

| Question | XQA final answer | Triton final answer | Strict scoring |
| --- | --- | --- | --- |
| 203 | `27` | `31` | XQA gave the wrong value |
| 426 | `$11` | `11` | XQA amount was correct, extraction failed |
| 745 | `$1100` | `1100` | XQA amount was correct, extraction failed |
| 819 | `$6,250` | `6,250` | XQA amount was correct, extraction failed |

At the first differing attention row on question 203, rounding the unnormalized
softmax weights to BF16 in both numerator and denominator reproduced all 2,048
XQA output components. The Triton row matched the FP64 reference rounded to BF16.
All 1,632 XQA calls through the first divergent token used contiguous Q.
Forcing a single KV split in a diagnostic reproduced the same two-key row,
including its 151 differences from the BF16-rounded FP64 reference, and the
same fifteenth token (`5`). The three currency-format losses were established
by replaying the strict scorer.
See the [matched benchmark commands and accuracy gate](accuracy.md#attention-backend-runs).
