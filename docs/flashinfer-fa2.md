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
  tensor-core path plans against the full table capacity but takes the schedule
  from the live CSR buffers and masks unused splits, which is the graph
  contract FlashInfer documents for this wrapper.
- **Plan cache.** One plan per `(rows, q_heads, kv_heads, table_width)`, created
  on first use with a synthetic full-capacity table. Each plan owns its CSR
  buffers and a workspace sized by `workspace_size(...)`; FlashInfer keeps
  plan-time tables in that workspace, so plans cannot share it. Plans captured
  by a CUDA Graph are pinned; eager plans are evicted first when the cache
  (32 entries) is full. Every plan is created outside graph capture.
- **Per-traversal metadata.** `prepare_attention` / `ModelRuntime.forward` prime
  the backend once per traversal; the graph paths prime the *graph entry's*
  private metadata after copying the caller's tables in. CSR construction
  (`indptr`, compact `indices`, `last_page_len`) is fixed-shape and device-only,
  so a step with device-owned context lengths does not read back to the host.
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

- `tests/test_flashinfer_fa2_attention.py` (15 GPU cases): selection and
  rejection paths; strided/GQA/MHA KV against both Torch and Triton kernels;
  padding tri-state; zero-length rows; batch composition; vectorized CSR
  against a reference construction; adapter-level CUDA Graph capture/replay
  with refreshed metadata; and tiny-Ouro engine runs where eager, graphs,
  async, and async+graphs produce identical token/exit-depth sequences for
  LAST_EXITED and SHARED.
- Real Ouro-1.4B BF16, greedy, trace depth 4, three requests: Triton eager,
  FA2 eager, FA2 graphs, and FA2 graphs+async produce **identical token
  sequences**; `plan_misses == 0`, one graph capture, 60 replays.
- Official GSM8K-87 (default protocol, HF baseline 59/87):

| run | correct | accuracy | gate |
|---|---:|---:|---|
| Triton eager | 61/87 | 70.11% | pass |
| FA2 eager (tensor cores) | 60/87 | 68.97% | pass (+1.15 pp) |
| FA2 + CUDA Graphs | 59/87 | 67.82% | pass (+0.00 pp) |

  Eager and graphs share **69/87** token sequences; 18 diverge at a BF16 tie
  because padded graph rows and the full-width plan change the tensor-core
  split schedule, and each divergence changes the whole continuation. The
  graphs run has no length-limited runaway (mean 106.7 tokens); the eager run
  hits one (question 182, mean 117.1 tokens).
- The earlier CUDA-core variant of this branch scored 61/87 with a different
  BF16 accumulation order; it is superseded by the tensor-core kernel that
  graph execution requires.
- An earlier 10-question custom subset on torch 2.8 was used as a smoke:
  3/10 with both backends and 8/10 identical sequences.

## Performance (diagnostic, not a benchmark)

Matched fixed work, real Ouro-1.4B BF16, greedy depth 4, 64 output tokens,
32-token prompts, one wave of N requests, second run after a warmup that
captures the graph:

| concurrency | Triton eager | FA2 eager | FA2 graphs | FA2 async+graphs |
|---:|---:|---:|---:|---:|
| 1 | 4.94 s (13.0 tok/s) | 5.24 s (12.2) | 1.67 s (38.4) | 1.66 s (38.7) |
| 8 | 5.50 s (93.1) | 5.78 s (88.6) | 1.92 s (267.4) | 1.88 s (272.1) |
| 32 | 6.12 s (334.5) | 6.22 s (329.4) | 2.32 s (881.7) | 2.23 s (918.8) |

Eager FA2 is at parity with Triton; CUDA Graphs remove most per-step launch
overhead on this small model (roughly 2.6–3.1× over FA2 eager and 3.0–3.4×
over Triton eager). The official GSM8K-87 runs agree in direction (eager
784.6 s, graphs 281.5 s, with different generated lengths). Asynchronous
scheduling adds no measurable gain at these batch sizes. These are single-GPU,
single-wave diagnostics, not production throughput claims.

Kernel-only figures from the earlier CUDA-core adapter (FA2/Triton 0.39–0.98
across B 1–32 and context 256–8192) are superseded by the tensor-core kernel
used here and are not reproduced.

## Follow-ups

- Auto-selection on SM89 after broader validation.
- More page sizes/head dimensions, packed prefill, PD and speculative paths.
- Revisit the CUDA-core decode path if FlashInfer makes its plan schedule
  independent of plan-time lengths; it is bit-closer to the Torch reference.
- Size plan workspaces per graph-entry width; full-capacity plans allocate the
  widest table even for short contexts.
- Reconcile with #112 if XQA lands first; this backend keeps a distinct
  `flashinfer_fa2` name and module.
