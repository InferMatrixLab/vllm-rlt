# FlashInfer FA2 backend (Ada / SM89)

`--attention-backend flashinfer_fa2` runs Ouro's paged decode through
FlashInfer's **FA2** paged-decode kernel. It targets NVIDIA Ada (SM89,
L20/L40S/RTX 40), where FlashAttention-2 rejects the page size 16 used by
Ouro's cache and FlashInfer XQA (PR #112) rejects SM89.

## Scope

- Ouro-1.4B BF16, fixed four-loop, decode-only.
- `block_size=16`, `head_dim=128`, NHD KV layout, BF16.
- LAST_EXITED and SHARED KV layouts.
- Eager execution, CUDA Graphs, and asynchronous scheduling.
- No packed prefill, prefill/decode disaggregation or speculative paths.
- Explicit opt-in only; `auto` keeps its existing FA4/FA3/FA2/Triton/Torch order.

## Requirements

- CUDA toolkit with `nvcc >= 12.9` reachable through `CUDA_HOME`, and that
  toolkit's `lib` directory on `LD_LIBRARY_PATH` (FlashInfer JIT).
- `flashinfer-python>=0.7.2,<0.8`:

```bash
pip install -e ".[flashinfer-fa2]"
```

The first call compiles the FA2 module for SM89 and caches it under
`~/.cache/flashinfer/<version>/89/`.

## Usage

```python
with LLM(
    model_path,
    device="cuda",
    dtype=torch.bfloat16,
    attention_backend="flashinfer_fa2",
    cache_config=CacheConfig(num_blocks=512, block_size=16, layout="last_exited"),
    scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=128),
    execution_config=ExecutionConfig(cuda_graphs=True, async_scheduling=True),
) as llm:
    ...
```

or `--attention-backend flashinfer_fa2` on the offline CLI/server, combined with
the engine's ordinary CUDA Graph and asynchronous-scheduling options.

## Design

- **Tensor-core, graph-compatible wrapper.** The backend uses
  `CUDAGraphBatchDecodeWithPagedKVCacheWrapper(... use_tensor_cores=True)` for
  both eager and graph execution. The CUDA-core decode wrapper bakes its work
  schedule (request/tile indices) into plan-time state; on SM89 a plan built for
  full capacity and replayed with a short live context stalls indefinitely once
  the planned table is wide enough (reproduced at 64+ pages, GQA and MHA). The
  tensor-core path plans against the planned width but takes the schedule from
  the live CSR buffers.
- **Split-KV is disabled.** With FlashInfer's default splitting, a row's
  reduction order depends on the planned width and on batch composition
  (verified on L20: the same 1,500-token row differs between a 128- and a
  256-page plan, and alone versus inside a batch of eight). `disable_split_kv`
  makes the schedule per-row: eager, graph replay, plan width and batch
  composition are bit-identical. The cost is low-batch long-context throughput
  (fewer CTAs); measured kernel-only at batch 8 it is unchanged.
- **Plan cache.** One plan per `(rows, q_heads, kv_heads, width bucket)`, where
  the width is rounded up to a power of two. A growing eager context therefore
  crosses at most one plan per bucket (~7 plans over a 1K-token request) instead
  of one per page boundary. Each plan owns its CSR buffers and a workspace sized
  by `workspace_size(...)`; `plan()`/`workspace_size()` inputs are pinned host
  tensors, so creating a plan does not copy device metadata back to the host.
  Plans captured by a CUDA Graph are pinned; eager plans are evicted first when
  the cache (32 entries) is full. `plan_creations` reports plan count next to
  `plan_misses`.
- **Kernel pinned by assertion.** The wrapper is constructed with
  `backend="auto"` (the graph wrapper exposes no backend argument); after the
  first `plan()` the adapter requires `resolved_backend == "fa2"` and records it
  in `info`, so a patch release that routes to cuDNN fails loudly instead of
  changing arithmetic.
- **Per-traversal metadata.** `prepare_attention` / `ModelRuntime.forward` prime
  the backend once per traversal; the graph paths prime the *graph entry's*
  private metadata after copying the caller's tables in. CSR construction
  (`indptr`, compact `indices`, `last_page_len`) is fixed-shape and device-only,
  and a GPU regression test runs it under `torch.cuda.set_sync_debug_mode` so a
  step with device-owned context lengths cannot regain a host read.
- **Replay refreshes the plan buffers.** `__call__` copies the staging CSR into
  the plan's buffers; graph capture records those copies, so every replay
  consumes the metadata written by `prepare_step` immediately before it. The
  staging buffers are referenced weakly by their metadata tensors and stay
  alive as long as a graph entry does.
- **Padding.** Rows with `context_lengths == 0` are masked to exact zeros after
  the kernel call.
- K/V must have identical strides. The engine's `key_cache[:, layer]` and
  `value_cache[:, layer]` do; a contiguous K with a strided V is rejected by
  FlashInfer.
- No `prefill` entry point: the default Ouro path never calls it, and
  `packed_prefill` stays false. Asynchronous *speculative* scheduling requires
  `device_context_lengths`, which this backend does not declare and is
  therefore rejected at selection.

## Validation (L20, 2026-11-01/02)

Environment: 1×L20 (sm_89, 46 GB, driver 580.126.20), **torch 2.13.0**
(metadata `2.13.0`, CUDA 13.0), FlashInfer 0.7.2.post1, Triton 3.7.1, and the
protocol's pinned `transformers 4.55.0` / `lm-eval 0.4.9.2` / `datasets 3.6.0`
/ `tokenizers 0.21.4`. FlashInfer's JIT requires a CUDA >= 12.9 toolkit; the
official run used the pip `cuda-toolkit[nvcc]==13.0.3` tree.

- `tests/test_flashinfer_fa2_attention.py` (22 cases: 16 GPU, 6 CPU): selection
  and rejection paths; strided/GQA/MHA KV against both Torch and Triton kernels;
  padding tri-state; zero-length rows; batch composition; vectorized CSR
  against a reference construction (including the empty-row last-page length);
  no host synchronization in `prepare_step`/`__call__` across a width-bucket
  change; bounded plan creation over a 1K-token eager decode; bitwise
  width/batch invariance and graph-replay equality at a 128-page plan; and
  tiny-Ouro runs where eager, graphs, async and async+graphs produce identical
  token/exit-depth sequences for LAST_EXITED and SHARED.
- Real Ouro-1.4B BF16, greedy, trace depth 4, three requests: Triton eager,
  FA2 eager, FA2 graphs, and FA2 graphs+async produce **identical token
  sequences**; `plan_misses == 0`, one graph capture, 60 replays.
- Official GSM8K-87 (default protocol, HF baseline 59/87):

| run | correct | accuracy | gate |
|---|---:|---:|---|
| Triton eager | 61/87 | 70.11% | pass |
| FA2 eager (tensor cores) | 59/87 | 67.82% | pass (+0.00 pp) |
| FA2 + CUDA Graphs | 59/87 | 67.82% | pass (+0.00 pp) |

  Eager and graphs are **bit-identical: 87/87 token sequences**, with no
  length-limited runaway (mean 106.7 tokens; eager 717.8 s, graphs 279.6 s).
  Against Triton, 67/87 sequences are identical; four answers flip (203, 426,
  819 correct → incorrect; 871 incorrect → correct). The no-split arithmetic
  reproduces the earlier split-KV *graph* run on all 87 questions; the earlier
  split-KV eager run differs on 18 questions because its reduction order
  depended on the narrow eager plan width. Disabling split-KV moved the eager
  score from 60/87 to 59/87, exactly at the baseline.
- The earlier CUDA-core variant of this branch scored 61/87 with a different
  BF16 accumulation order; it is superseded by the tensor-core kernel that
  graph execution requires.
- An earlier 10-question custom subset on torch 2.8 was used as a smoke:
  3/10 with both backends and 8/10 identical sequences.

## Performance (diagnostic, not a benchmark)

Matched fixed work, real Ouro-1.4B BF16, greedy depth 4, 64 output tokens,
32-token prompts, one wave of N requests, second run after a warmup that
captures the graph. All five columns come from one session:

| concurrency | Triton eager | Triton graphs | FA2 eager | FA2 graphs | FA2 async+graphs |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.20 s (12.3 tok/s) | 1.52 s (42.2) | 5.49 s (11.7) | 1.59 s (40.2) | 1.58 s (40.6) |
| 8 | 5.34 s (95.8) | 1.76 s (290.2) | 5.55 s (92.2) | 1.84 s (278.1) | 1.81 s (282.8) |
| 32 | 6.26 s (327.3) | 2.20 s (930.1) | 6.32 s (324.1) | 2.24 s (915.4) | 2.15 s (952.3) |

Eager FA2 is at parity with Triton eager. CUDA Graphs remove most per-step
launch overhead for **both** backends (2.8–3.5× over eager on this small
model); Triton graphs is then 2–5% faster than FA2 graphs, so the graph gain is
dispatch reduction, not the FA2 kernel. Asynchronous scheduling adds 1–4% over
FA2 graphs at these batch sizes. The official GSM8K-87 runs agree in direction
(eager 717.8 s, graphs 279.6 s, identical generated lengths). These are
single-GPU, single-wave diagnostics, not production throughput claims.

Disabling split-KV costs attention time only at low batch and long context
(kernel-only, one L20, page 16, random KV):

| rows | ctx | split-KV ms | no split ms | ratio |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 0.023 | 0.049 | 2.13× |
| 1 | 8192 | 0.028 | 0.092 | 3.35× |
| 8 | 4096 | 0.375 | 0.372 | 0.99× |
| 8 | 8192 | 0.741 | 0.739 | 1.00× |

Ouro runs 24 attention layers per depth view, so at batch 1 and 8K context the
no-split schedule adds roughly 6 ms per generated token; at batch 8 it is
unchanged. The determinism above is the reason for accepting that cost.

Kernel-only figures from the earlier CUDA-core adapter (FA2/Triton 0.39–0.98
across B 1–32 and context 256–8192) are superseded by the tensor-core kernel
used here and are not reproduced.

## Follow-ups

- Auto-selection on SM89 after broader validation.
- More page sizes/head dimensions, packed prefill, PD and speculative paths.
- Revisit the CUDA-core decode path if FlashInfer makes its plan schedule
  independent of plan-time lengths; it is bit-closer to the Torch reference.
- Write the live CSR directly into the plan buffers at prime time (currently
  three per-layer copies, about 0.5 MiB per layer at 32 rows and 4096 pages).
- Revisit split-KV for low-batch long-context workloads if FlashInfer offers a
  deterministic split schedule; today the deterministic path costs 2.1–3.4×
  kernel time there.
- Add a CPU test for the GSM8K `--cuda-graphs` / `--attention-backend` flags.
- Reconcile with #112 if XQA lands first; this backend keeps a distinct
  `flashinfer_fa2` name and module.
