# FlashInfer FA2 backend (Ada / SM89)

`--attention-backend flashinfer_fa2` runs Ouro's paged decode through
FlashInfer's **FA2** `BatchDecodeWithPagedKVCacheWrapper`. It targets NVIDIA
Ada (SM89, L20/L40S/RTX 40), where FlashAttention-2 rejects the page size 16
used by Ouro's cache and FlashInfer XQA (PR #112) rejects SM89.

## Scope

- Ouro-1.4B BF16, fixed four-loop, decode-only.
- `block_size=16`, `head_dim=128`, NHD KV layout, BF16.
- LAST_EXITED and SHARED KV layouts.
- Synchronous eager execution.
- No CUDA Graphs, async scheduling, packed prefill, PD or speculative paths.
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
    execution_config=ExecutionConfig(),
) as llm:
    ...
```

or `--attention-backend flashinfer_fa2` on the offline CLI/server.

## Design

- `prepare_attention` / `ModelRuntime.forward` prime the backend once per
  traversal (`KVCacheManager.prepare_attention`), so the CSR metadata
  (`indptr`, `indices`, `last_page_len`) is built once per depth view and reused
  across all decoder layers.
- The FA2 plan is created once per `(rows, q_heads, kv_heads, table_width)` with
  a synthetic full-capacity table; later steps only refresh device buffers.
- K/V must have identical strides. The engine's `key_cache[:, layer]` and
  `value_cache[:, layer]` do; a contiguous K with a strided V is rejected by
  FlashInfer.
- Padding rows (`context_lengths == 0`) are masked to exact zeros.
- No `prefill` entry point: the default Ouro path never calls it, and
  `packed_prefill` stays false.

## Validation (L20, 2026-11-01)

Environment: 1×L20 (sm_89, 46 GB, driver 580.126.20), torch 2.8.0, FlashInfer
0.7.2.post1, Triton 3.7.1. The L20 pod's CUDA 12.1 needed a local removal of
FlashInfer's CUDA-12.9-only `--compress-mode=size` flag; use CUDA 12.9 for a
clean build.

- Unit tests: `tests/test_flashinfer_fa2_attention.py` covers selection,
  rejection paths, strided/GQA KV, padding tri-state, zero-length rows, batch
  composition, empty batches, and tiny-Ouro token/exit-depth parity against
  Triton for LAST_EXITED and SHARED.
- Real Ouro-1.4B: a 10-question custom GSM8K subset produced the same correct
  count (3/10) with Triton and FA2; 8/10 generated token sequences were
  identical. The first divergences sit on questions both backends answered
  incorrectly, at near-tied logits (top1/top2 margins 0.0–0.0625).
- The official GSM8K-87 fingerprint requires torch 2.13.0; the L20 pod ran
  torch 2.8, so the stored 59/87 baseline was not claimed. Run the official
  protocol on a torch-2.13 environment before quoting a 87-question score.

## Performance (diagnostic, not a benchmark)

Kernel-only, Ouro attention shapes, page 16, random KV, single L20:

| B | ctx | FA2 ms | Triton ms | FA2/Triton |
|---:|---:|---:|---:|---:|
| 1 | 256 | 0.0225 | 0.0584 | 0.39 |
| 1 | 2048 | 0.0334 | 0.0609 | 0.55 |
| 1 | 8192 | 0.0243 | 0.0587 | 0.41 |
| 8 | 2048 | 0.1856 | 0.1933 | 0.96 |
| 32 | 8192 | 2.9407 | 2.9881 | 0.98 |

FA2 is ~2–3× faster at low concurrency/short context and at parity at high
concurrency/long context. End-to-end, the 10-question real-model run took
84.8 s with Triton and 79.7 s with FA2; the model forward dominates, so the
attention gain is small. Do not treat these as production speedup claims.

## Follow-ups

- CUDA Graph: prime metadata before capture and enable `cuda_graphs`.
- Async scheduling/device-owned metadata.
- Auto-selection on SM89 after broader validation.
- More page sizes/head dimensions, packed prefill, PD and speculative paths.
- Reconcile with #112 if XQA lands first; this backend keeps a distinct
  `flashinfer_fa2` name and module.
