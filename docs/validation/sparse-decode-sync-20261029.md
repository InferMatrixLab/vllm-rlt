# Sparse decode: ordinary and synchronous integration validation

This is the historical report for runtime `10373ab`. The subsequent
[numerical diagnosis and regression report](sparse-decode-numerics-20261029.md)
documents the backend-preserving fix and its separate validation. The timings
and numerical limitation below describe the original implementation.

This contribution extends the shared [#157](https://github.com/InferMatrixLab/vllm-rlt/pull/157) reference plan/store into ordinary synchronous decode and synchronous block self-speculation. It is a functional reference integration, with measured costs and an unresolved BF16 full-retention sequence difference. It does not claim an optimized sparse kernel or deployment quality.

## Source and checks

- Immediate base: `048b9d7c79a5711857ce3287812502ad8f274ad0` on `kunkun/rfc156-sparse-decode-async`.
- Measured runtime: `10373ab6d63156535b55e4b9be1c7b8ba7319421`. Additional same-input tests: `985c309`; runtime files are identical between these revisions. Source SHA256 inventories are in the archive.
- Full pre-commit: Ruff lint, Ruff format and CPU regression passed; **919 passed, 475 skipped**.
- Applicable GPU regression: **312 passed, 1 skipped, 16 deselected**. The skip needs a second GPU; FA4 cases are excluded on SM89. The original unfiltered run's 12 unsupported-FA4 failures are retained.
- New integration module: 26 CPU tests plus 5 CUDA cases. Includes ordinary/padded routing, prompt exclusion, draft depth 1/2, each rejection boundary, same-position replacement, abort/ID reuse, failed submissions, stream-owned retirement and active-request close. Per-depth hidden, logits and populated KV match serial same-input sparse execution at FP32 atol/rtol `4e-5`.
- The negative stream control restores the unsafe caller-stream release only inside one test process and fails as expected. Corrected production code passes.
- Independent review found release-index stream ownership and shutdown-drain defects; both were fixed and covered. Re-review found no further concrete production defect.
- Upstream FlashLoop `aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6`, `dense_source_selector` plus `cached_mass_sparse_delta(force_torch=True)`: **18 CUDA fixed-input cases passed**, with equal retained sets; lengths 1/13/128, retention .1/.5/1, FP32 and BF16. Predeclared attention-output atol/rtol were `2e-5` and `.02`, respectively, accommodating the gold path's BF16 subset/output storage. This is separate from model-level quality.

## Real-model protocol

One RTX4090, 24,564 MiB; driver 580.105.08; torch 2.12.1+cu130, Triton 3.7.1. Released Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, BF16 weights/activations/KV; weight SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.

Native StageEngine, LAST_EXITED KV, refill, Triton dense backend, synchronous scheduling, multistream on, graphs off. Block K=2, draft depths 1/2, fixed four loops, greedy and ignore EOS. Each fresh process warms 4 output tokens at its target arrival concurrency, then measures three 128-input/16-output bursts. Request i uses `[1 + (j * 137 + i * 53) % 49000 for j in range(128)]`. Context bound is explicitly **512** for sparse, while model positional configuration remains unchanged. Token budget `max(128, 3*C)`, prefill chunk 128, KV blocks `40*C + 32`, block size 16. This does not qualify the model's 65,536 context limit.

Timing includes enqueue, prefill, decode, delivery, drain and CUDA synchronization; model loading and warmup are excluded. Profiles are separate and excluded. GPU clocks and CPU affinity were not pinned for the primary sweep. All raw repeats, outputs, depths, acceptance counters, TTFT/E2E observations and memory values are archived. Speculative output arrives in chunks; chunk gaps are not per-token ITL. The short bursts do not establish steady-state HTTP serving.

**81 primary bursts / 1,107 completed requests / 17,712 output tokens**; no primary OOM, timeout or failed request. Every measured run drained KV blocks and reference slots to zero and repeated its token/depth sequences exactly.

Output throughput: median of three runs, with min-max in parentheses. Full-retention agreement counts one repeat, and was identical across all three.

| Mode | Concurrent arrivals | Dense tok/s | Full retention tok/s | r=.1 tok/s | Full vs dense tokens |
|---|---:|---:|---:|---:|---:|
| Ordinary | 1 | 14.90 (14.67-14.98) | 9.66 (8.89-9.79) | 9.30 (8.86-9.37) | 16/16 |
| Ordinary | 8 | 71.60 (71.59-71.60) | 21.59 (21.53-21.62) | 21.73 (21.59-21.77) | 128/128 |
| Ordinary | 32 | 124.42 (124.41-124.49) | 27.13 (27.01-27.31) | 26.84 (26.52-27.24) | 512/512 |
| Sync, draft 1 | 1 | 17.46 (17.21-17.68) | 9.10 (8.63-9.12) | 13.90 (12.72-14.01) | 16/16 |
| Sync, draft 1 | 8 | 60.53 (57.33-61.12) | 18.01 (17.83-19.32) | 25.54 (25.50-25.64) | 128/128 |
| Sync, draft 1 | 32 | 82.91 (81.09-83.10) | 19.58 (19.54-19.81) | 27.48 (27.05-27.57) | 498/512 |
| Sync, draft 2 | 1 | 16.56 (15.37-16.69) | 9.49 (9.06-9.70) | 11.87 (11.13-11.91) | 16/16 |
| Sync, draft 2 | 8 | 50.27 (50.16-50.57) | 18.55 (17.54-18.58) | 22.40 (22.23-22.42) | 128/128 |
| Sync, draft 2 | 32 | 68.08 (67.59-68.30) | 20.31 (20.20-20.38) | 24.84 (24.70-24.90) | 512/512 |

The shared PyTorch reference path is slower than dense Triton in every covered condition. Peak allocated memory reached **6.67 GiB**. Retention .1 changes outputs, and its speculation acceptance changes are preserved in `summary.json`; token agreement is not a task-quality score.

## Full-retention discrepancy

At C=32, draft depth 1, full retention matches **498/512** dense output tokens. The first divergence is request 9, output index 2 (zero-based), input position 129: dense chooses 293, full retention chooses 92. Other full-retention conditions in this sweep match all tokens; this does not prove broader equivalence.

Instrumented replays reproduce the mismatch with **identical 130-token input prefixes**. Loop-1 hidden and loop-2 input hidden are exact. The first observed hidden drift is already at loop-2 capture output (max .01953125), before any reference consume; later-loop input differences follow. Recorded layer-0 Q/K/V and attention are exact at the capture loop; their same-input full-retention attention replays match Torch. Final logits differ by at most .125. Dense ranks `(293, 8.875)` above `(92, 8.8125)`; the reference path reverses those values. This localizes the observed drift to the capture computation before reuse, consistent with backend numerical differences, but does not localize the first physical-layer operator or establish task quality. No lossless/full-retention-at-scale claim is made. The existing async discrepancy in #157 is also not resolved by this contribution.

## Disabled-feature drift control

Initial unpinned C8 base/candidate controls showed timing drift, so a separate CPU-pinned **ABBA** check was run for sync draft 2, C8, feature off. Each arm uses a fresh process, the same workload and three measurements; CPUs are `[0, 1, 2, 3]`. Tokens and exit depths match the base exactly in all arms.

- Base: median 52.06, range 51.90-53.91 tok/s.
- Candidate: median 52.19, range 50.63-52.54 tok/s.
- Candidate/base median ratio: 1.0027. These pinned observations are a separate control, not replacements for the primary unpinned numbers or a general performance guarantee.

## Separate profiling

Twelve readable CPU+CUDA captures cover ordinary C1/C8 and sync draft-2 C8, dense versus r=.1, each in independent ops-only/full processes. Full enables shapes, memory and stacks together. Every full capture has shape events, memory events and a nonempty exported stack inventory. The profile window spans a further output from each active request after the first outputs; raw shapes and row work remain available for checking scheduling/acceptance differences. Diagnostic count ratios are not throughput ratios.

| C8 arrival workload | Active requests dense / r=.1 | GPU kernels dense / r=.1 | Scalar item reads dense / r=.1 | Argsorts dense / r=.1 |
|---|---:|---:|---:|---:|
| Ordinary | 8 / 8 | 5,600 / 26,220 | 0 / 384 | 0 / 192 |
| Sync, draft 2 | 7 / 5 | 11,319 / 43,242 | 0 / 720 | 0 / 360 |

Ordinary C8 windows contain eight active requests in both arms. The synchronous windows contain seven versus five active requests because progress differs; they support operator diagnosis, not a matched-work count ratio.

The source chain is `reference_plan -> sparse_paged_attention -> reference_statistics/read_reference`: per-row gather, selection and scalar count reads add launches and waits. Pool operations also add retirement work. The measured slowdown is therefore not explained by retained-key count alone. This contribution retains the common reference oracle; it does not duplicate or replace the pending attention optimization discussed on #157.

![Aligned single-request diagnostic decode windows](https://github.com/Levius-Fubuki/vllm-rlt/releases/download/pr157-sync-validation-20261029/single-request-timeline.png)

## Reproduction and evidence

The [validation release](https://github.com/Levius-Fubuki/vllm-rlt/releases/tag/pr157-sync-validation-20261029) contains scripts, original traces, stack exports, raw outputs, source inventories, the negative control, failed diagnostic-harness attempts and the first unsupported-FA4 run. The SHA256 inventory covers the package contents; `SHA256SUMS` covers the downloadable archive and timeline.

From the corresponding source checkout with the recorded dependencies and a reserved GPU:

```bash
pre-commit run --all-files
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m pytest \
  tests/test_sparse_decode_sync.py tests/test_sparse_decode_attention.py \
  tests/test_speculative.py tests/test_async_speculative.py \
  tests/test_async_speculative_lifecycle.py tests/test_backend_selection.py \
  --run-gpu -k 'not flash_attn_4' -q
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python scripts/real_model_bench.py \
  --model /path/to/pinned/Ouro-1.4B --mode sync2 --concurrency 32 \
  --retention .1 --repeats 3 --output /tmp/sync2-c32-r01.json
```

Use `--mode ordinary|sync1|sync2`, `--retention 0|1|.1`, `--concurrency 1|8|32`; retention 0 disables the feature. The script contains the exact engine startup and per-request parameters. Use `--profile ops|full --repeats 1` only for separate diagnostics. The orchestrators retain their original task directory for exact provenance; the individual benchmark CLI accepts arbitrary model/output locations.

Graphs, token pipelines, mixed scheduling, preemption and ordinary async decode are explicitly rejected. PD, prefix-cache composition, adaptive-exit quality, long generations, held-out task quality and other hardware are not qualified here. Sparse remains disabled by default. This is a Draft collaboration contribution to the existing shared feature PR, with numerical/performance limits retained for review.
