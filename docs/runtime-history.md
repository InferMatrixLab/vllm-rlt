# Historical runtime validation

These records were moved from the runtime, asynchronous speculation and KV
management guides. They describe earlier implementations, not validation of the
current checkout. Source revisions, environments, failed checks and qualification
limits are preserved below. Records without a pinned revision remain historical
observations and cannot establish a regression baseline by themselves.

Current mechanisms and reproduction entry points live in the
[runtime guide](runtime.md#replay-and-validation-commands),
[async speculation guide](async-speculative.md#validation) and
[KV guide](kv_cache_walkthrough.md#validation-entry-points).

## Historical validation (2026-09-15)

These results describe the CDB runtime implementation against base commit
`6e8abbc405e1`, before the unified-stage refactor. CUDA Graphs and Huginn were
deferred at that time; this is not their current support status. The recorded
results are not fresh validation of the current working tree.

Final full suite on an available NVIDIA B300, with `CUDA_VISIBLE_DEVICES=0` and
`--run-gpu`: **199 passed, 11 skipped**. The skips require optional `lm_eval`;
GPU tests ran. Ruff and `git diff --check` passed. The CUDA auto-budget,
shared/last-exited async comparisons, cancellation/reuse and synthetic overlap
checks all passed. The documented CLI example completed. SHARED's eight replay
variants preserved recorded output IDs and exit depths. Tiny-model timings do
not demonstrate throughput gains; no pretrained accuracy or paper speedup is claimed.

## Ouro delayed validation

The original record did not specify a date or source commit. In particular,
the BF16 multi-stream difference below was not resolved by these observations.

Recorded validation for `ouro_delayed`: full CPU/GPU suite **215 passed, 11 skipped**
(optional lm_eval), Ruff and diff checks passed. Tests cover cumulative hazards
(0.3 + 0.3 => CDF 0.51), trigger-round minimum, hard maximum, threshold 1,
per-token reset and use of the final extra round's hidden state. Both KV layouts
passed tiny-model CUDA sync/async comparisons including padding and sampling.

Real Ouro-1.4B was tested on one B300. Three prompts with thresholds 0, 0.9 and 1, generating 16 tokens each,
matched token IDs and depths between sync and async in FP32. A BF16 single
Chinese request matched sync, async single-stream and async multi-stream.
However, BF16 three-request multi-stream execution showed a token divergence
on the French-capital prompt at output index 7 despite matching exit depths.
Single-stream matched the synchronous reference in that run; adding diagnostic
logit copies also removed the divergence. A separate synchronous single/batch
comparison did not reproduce it. Reduced-precision or schedule-dependent
numerics are a hypothesis, not a confirmed root cause. Therefore this check
does not establish exact BF16 multi-stream output equivalence or rule out all
async issues. FP32 provides the currently verified real-model comparison for
this workload. No throughput gain or task-accuracy result is claimed here.

## Historical validation: M4 refactor

The following record predates the unified-stage refactor; it is retained as
historical evidence, not current-branch qualification.

CPU tests in [test_kv_cache.py](../tests/test_kv_cache.py):

- The views report membership, held pages under incremental growth, per-plane tables, and lease state. SHARED exposes one plane, and depth-indexed access still works for every depth.
- `exclusive_prefix_blocks` counts only the blocks the prefix cache holds, before and after a live request claims them, including a partially claimed prefix.
- `token_written` requires every layer. `mark_finalized` marks deeper planes without touching their bytes, refuses unwritten tokens, records nothing new at the deepest loop, and leaves SHARED state unchanged.
- `snapshot` and `restore` round-trip pages and written state through a disturbed pool in both layouts, refuse capacity and page-count mismatches, refuse a non-empty target, and do not alias the live allocation.
- `snapshot` refuses leased allocations.
- A claim that cannot succeed leaves prefix entries and free-block accounting untouched. A later claim that can succeed still evicts (F6).

Existing tests in `test_prefix_growth.py` (preemption, resumption, prefix pricing), `test_pd.py`, `test_async_pipeline.py`, `test_engine.py`, and `test_cdb_runtime.py` cover the migrated callers. Counts are in the pull request.

GPU, one RTX PRO 6000 Blackwell, torch 2.13+cu129, Triton 3.7.1, no FlashAttention or NIXL installed. `pytest --run-gpu -m gpu` gives 71 passed and the same 30 failures on `main` and on this branch. Every failure is a missing FlashAttention, FA4, or NIXL backend. Ouro-1.4B BF16 runs with 64 requests, 256-token prompts, and 128 outputs, alternating `main` and this branch for three or more trials, produce identical token ids and exit depths in all seven configurations tried. That includes preemption under KV pressure with prefix caching, and the async finalize path with early exits. Throughput differences stay inside the trial spread, between -3% and +5%. FA4, packed prefill, and PD transfer were not verified.

## Recorded GPU validation (2026-10-10)

This dated record is retained from the earlier implementation. It is not a fresh
validation or performance measurement of the unified-stage refactor.


Runtime source: `d977817d4e174cbc2fcc9aa7ef5893c35da2ec89`, incorporating main
`d704aee654235f6a13e8c5bf13b41aaf80fbb7d9`. Validation used
one scheduler-reserved NVIDIA H20-3e, TP=1, Python 3.10.20,
PyTorch 2.11.0+cu128, CUDA 12.8 and Triton 3.6.0. Real Ouro-1.4B weights and KV
were BF16, with greedy fixed-depth `d=2, D=4`, `last_exited` KV and eager execution.

The async test module passed **26 tests** (22 CUDA cases and four CPU helpers).
Related speculation, async state/pipeline, HTTP serving and CUDA Graph regression
passed **148 tests**, with 15 FlashAttention cases deselected. This includes both
synchronous speculative ragged CUDA Graph capture/replay cases. The initial
related run had 12 failures importing the absent FA3 `flash_attn` package and
three FA4 cases deselected; these environment failures are retained in the logs,
and neither FlashAttention backend is qualified by the Triton rerun.

A subsequent isolated FA3 installation on the same runtime/test source was
validated separately: all original 12 graph failures passed with zero
skips, the related suite passed 160 tests (three explicit FA4 cases deselected),
eight additional FA3 paged/mixed-depth cases passed, and the async Triton module
again passed 26 tests. The original missing-package failures remain preserved;
this follow-up does not qualify asynchronous speculation on FA3, which still uses
the supported Triton path.

The Mac CPU suite passed **477 tests**, with 164 skips: 135 GPU cases not enabled,
26 missing optional `lm_eval`, and three Nanbeige cases missing official assets.
Ruff lint/format passed. Pyright on changed runtime modules reported the same
49 diagnostics as integrated main, with none in the async module; this is baseline
parity rather than a clean whole-project type check.

### Matched BF16 measurements

All **80 unprofiled measured trials** and their warmups matched the fixed-depth
native async baseline's output token IDs exactly. The four modes share the model,
prompt IDs, output budget, backend and KV capacity. Native async uses `ouro_delayed`
at threshold 1; speculation uses fixed-depth `ouro`, K=2. Each cell has five
observations after warmup; mode order reverses on alternate repetitions. EOS is
ignored, and the fixed English prompt is repeated/truncated to the listed lengths.
Maximum sequences is 32, token budget 4096 and prefill chunk size 128. KV capacity
per case is `32 * ceil((prompt + output) / 16) * 4` blocks, matched across modes.

`Async full` is an ablation on the current source that prepares/transfers the
full 32 x 4 x 4096 table; `Async active` uses the committed contiguous active
region. Both wrappers expose the same logical table extents to draft/verify and
have the same wrapper overhead. This isolates preparation/transfer capacity;
it is not a performance measurement of the old PR head.

Committed decode throughput excludes each request's first token. Timing begins
after all first outputs and bootstrap GPU work complete, and ends after final GPU
work completes. Values are median `[min, max]` tokens/s over five observations;
the ranges are descriptive, not confidence intervals. Profiler is disabled.

| Prompt/output | Concurrency | Native async | Sync spec | Async full | Async active |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64/8 | 1 | 19.62 [18.96, 19.89] | 28.72 [28.56, 28.98] | 29.36 [29.08, 29.51] | 29.02 [28.36, 29.53] |
| 64/32 | 1 | 19.51 [19.07, 19.58] | 29.85 [29.25, 30.22] | 30.14 [29.92, 30.34] | 30.32 [29.48, 30.44] |
| 64/32 | 8 | 154.23 [153.57, 155.85] | 236.45 [234.29, 237.60] | 242.53 [240.97, 243.44] | 243.00 [242.51, 243.51] |
| 512/128 | 1 | 19.87 [19.83, 19.92] | 27.49 [26.60, 28.14] | 28.40 [28.22, 28.44] | 28.53 [28.23, 28.71] |

Engine E2E covers request admission through final completed GPU work. It includes
prefill and first token, but excludes client/network latency. Median `[min, max]`
milliseconds under the same protocol:

| Prompt/output | Concurrency | Native async | Sync spec | Async full | Async active |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64/8 | 1 | 411.28 [406.44, 430.49] | 298.40 [295.59, 301.16] | 292.00 [290.69, 294.45] | 295.40 [290.49, 300.41] |
| 64/32 | 1 | 1644.06 [1637.03, 1683.32] | 1092.83 [1079.53, 1115.31] | 1083.81 [1077.01, 1090.74] | 1077.13 [1073.00, 1109.07] |
| 64/32 | 8 | 1718.17 [1701.54, 1724.73] | 1132.40 [1126.94, 1144.91] | 1106.91 [1103.24, 1112.90] | 1104.71 [1101.74, 1106.22] |
| 512/128 | 1 | 6655.23 [6639.07, 6669.83] | 4858.11 [4750.69, 5013.35] | 4708.12 [4700.69, 4736.14] | 4688.06 [4657.09, 4736.26] |

Async active's median decode throughput exceeds sync speculation by 1.1–3.8%
in these cells. The active/full E2E differences are small, mixed in direction and
have overlapping observed ranges; these five observations do not establish a
stable E2E gain from narrowing the table. The largest allocated-memory peak was
19.03 GiB for either async variant and 18.71 GiB for sync speculation. No memory
reduction is claimed; reserved and allocated peaks are retained per observation.

### Active-region preparation and transfer

A separate warmed microexperiment used 1000 CPU preparations and 100 CUDA-event
H2D observations per variant, with prompt/output 64/32. Both buffers are reused,
and copies complete before the CPU mutates pinned storage. Times below are medians;
they are not complete-round or E2E timings.

| Concurrency | Variant | H2D bytes | CPU prepare (us) | H2D event (us) |
| --- | --- | ---: | ---: | ---: |
| 1 | Full capacity | 2,097,152 | 78.51 | 43.55 |
| 1 | Active (1 x 4 x 5) | 80 | 14.22 | 7.42 |
| 8 | Full capacity | 2,097,152 | 98.45 | 43.49 |
| 8 | Active (8 x 4 x 5) | 640 | 34.45 | 7.23 |

This supports the reduced host preparation and transfer work in `_prepare_round()`
and the contiguous table copy in `submit()`. It does not imply that the same
microsecond savings appear on the complete request's critical path.

### Separate profiling diagnostics

Warmed prompt/output 64/8, concurrency 1, K=2 was captured separately with CPU
and CUDA activity. Ops-only disables shapes, memory and stacks; full enables all
three and exports actual nonempty CPU/GPU stack and memory records. Full-request
ops traces cover prefill, first token and decode. Matching single-round ops/full
captures cover the first complete draft/verify/acceptance/result-transfer round
after warmed bootstrap, ending after its GPU work completes. The initial
full-request full trace was valid but about 468 MiB, exceeding the 200 MiB capture
budget; it was preserved, and the remaining full captures were narrowed rather
than repeating that oversized window.

The CPU `user_annotation` decode range bounds the union of GPU kernels/copies.
A gap is the remainder of that range, not SM utilization or proof of a host
bottleneck. These single ops-only observations are diagnostic and include profiler
overhead; they do not replace the unprofiled table above.

| Ops-only observation | Async full | Async active | Sync spec |
| --- | ---: | ---: | ---: |
| Request wall (ms) | 500.82 | 513.64 | 513.32 |
| Decode wall (ms) | 407.43 | 422.25 | 424.59 |
| Decode GPU active union (ms) | 64.92 | 64.81 | 64.28 |
| Decode GPU gaps (ms) | 342.51 | 357.44 | 360.31 |
| H2D bytes / copies (whole request) | 6,303,248 / 13 | 12,032 / 13 | 12,864 / 32 |
| H2D summed copy duration (us) | 175.90 | 10.59 | 25.31 |
| D2H copies (whole request) | 4 | 4 | 4 |
| CUDA synchronization calls | 12 | 12 | 10 |
| CPU `aten::item` calls | 0 | 0 | 0 |

The active/full pair reduces copied bytes while copy counts and synchronization
counts remain unchanged. GPU active time changes little. The candidate trace's
longer wall/gap time does not isolate a regression, just as lower transfer time
does not establish E2E improvement. The current synchronous runner also has no
`aten::item` events here; historical scalar-read counts must not be applied to
this merged source. Retiring a collected ticket omits the redundant compute-event
host wait because its copy-completion event already depends on compute completion;
partial-failure drains retain both owned streams. The regression tests check
ordering and lifetime behavior independently of this performance diagnosis.

### Evidence and reproduction

Run-specific commands, artifact locations and source/model/tokenizer identities
are recorded in [PR #66](https://github.com/InferMatrixLab/vllm-rlt/pull/66).
The measurements above specify the workload, mode order, warmup, timing boundaries
and comparison controls. The commands in [Validation](async-speculative.md#validation) remain the supported GPU
regression entry points; the experiment's diagnostic scripts are not supported
benchmark entry points.

## Historical validation and measured results

The measurements below belong to the 2026-09-23 snapshot: PR44 commit
`d9fca507e766e81f5d89f90d598881c12d7d8397` plus the asynchronous implementation.
They do not qualify current main or the review refactor. The 2026-10-10 validation
above is a separate experiment with its own source revision and narrower workload.
Historical log locations are recorded separately in the PR description.

The real Ouro-1.4B checkpoint used BF16 weights, activations and KV, Triton,
one H20-3e, TP=1 and `d=2, D=4`. It has 24 shared layers,
hidden size 2048, 16 attention/KV heads and head dimension 128. Source and
model/config hashes were recorded with the results. Native async used
`ouro_delayed` at threshold 1 (full depth); speculative modes used fixed-depth
`ouro`. The same fixed English prompt was repeated/truncated across requests,
with EOS ignored for matched output lengths.

Historical CPU regression passed 289 tests with 11 skips. Related Triton GPU
regression passed 145 tests (3 deselected), and the final focused GPU run passed
24 tests. An earlier all-backend run failed two existing FA4 tests because the
package was unavailable; those failures are excluded from Triton qualification.
The GPU tests exercise every rejection position, two submissions before CPU
readback, FP32/BF16 per-layer/per-depth KV against serial replay, ragged bounds,
prefix reuse, arrivals, EOS, cancellation, ID reuse, partial failure and shutdown.
Token IDs require exact agreement; KV tolerances are FP32 `atol=rtol=4e-5` and
BF16 `atol=0.08, rtol=0.04`. These are targeted correctness checks, not general
model-quality evidence.

For a current rerun, use the [validation commands](async-speculative.md#validation)
on a scheduler-reserved CUDA device.
Compare native async, synchronous speculation and async speculation with the
same BF16 model, prompts, token budget, backend, capacity and GPU. Warm up each
mode, alternate measurement order and retain five unprofiled observations per
cell. Measure committed decode tokens after all first outputs through completed
GPU work; preserve output IDs, delivery timestamps, memory peaks and failures.
Use short contexts (64/32 prompt/output, concurrency 1/8, K=1/2/4/8) and longer
contexts (512/128, concurrency 1/32, K=2). Separately capture warmed ops-only and
full profiles at concurrency 1, K=2. For the active-table change, compare CPU
preparation time, H2D bytes/time and unprofiled end-to-end latency against the
previous full-capacity transfer, especially small batches and short contexts.
The removed historical benchmark is not a supported repository entry point.

## Historical results

All **150 measured trials** (120 short + 30 longer) completed with exact token
agreement against the matched native path. Three additional observations in the
profile job also matched; they are excluded from the table below to preserve the
five-repeat protocol. All runtime and benchmark hashes were checked against the
2026-09-23 measured working tree.

Throughput is committed decode tokens/s, shown as median `[min, max]` over five
observations. The final column compares async speculation with sync speculation.
These ranges describe the observed runs, not confidence intervals.

| Prompt/output | Concurrency | K | Native async | Sync spec | Async spec | Async/sync gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64/32 | 1 | 1 | 19.82 [19.74, 20.09] | 26.80 [26.68, 27.04] | 27.70 [27.04, 27.83] | +3.4% |
| 64/32 | 1 | 2 | 19.91 [19.83, 19.98] | 30.06 [29.67, 30.13] | 30.72 [30.58, 30.83] | +2.2% |
| 64/32 | 1 | 4 | 19.90 [19.83, 20.00] | 33.46 [33.30, 33.50] | 34.46 [34.37, 34.60] | +3.0% |
| 64/32 | 1 | 8 | 19.92 [19.89, 20.01] | 36.48 [36.23, 36.83] | 37.87 [37.47, 38.07] | +3.8% |
| 64/32 | 8 | 1 | 156.69 [154.97, 158.18] | 207.06 [206.47, 211.72] | 216.65 [215.69, 222.13] | +4.6% |
| 64/32 | 8 | 2 | 159.04 [156.63, 159.69] | 234.64 [230.22, 235.66] | 248.41 [245.01, 248.59] | +5.9% |
| 64/32 | 8 | 4 | 156.09 [154.57, 157.24] | 256.23 [254.39, 257.54] | 271.33 [270.83, 272.62] | +5.9% |
| 64/32 | 8 | 8 | 156.51 [154.94, 157.05] | 277.20 [277.11, 277.84] | 295.62 [294.83, 295.76] | +6.6% |
| 512/128 | 1 | 2 | 19.56 [19.50, 19.84] | 27.07 [26.83, 27.26] | 27.90 [27.68, 28.04] | +3.1% |
| 512/128 | 32 | 2 | 563.26 [561.55, 566.09] | 731.83 [726.46, 732.91] | 837.98 [836.99, 843.34] | +14.5% |

The largest observed allocated-memory peaks were 18.342 GiB for sync speculation
and 18.657 GiB for async speculation (longer case, concurrency 32, K=2). This
implementation retains device state and pinned/device buffers; it does not claim a
memory reduction. JSON also contains reserved-memory peaks for every observation.

## Historical profiling analysis

The diagnostic capture is the warmed short workload at concurrency 1, K=2.
`measured_decode` is selected from the CPU `user_annotation` track; PyTorch also
emits a GPU annotation with the same name. GPU-active time is the union of kernel
and copy intervals clipped to that CPU decode range. GPU gap is the remainder,
including dispatch gaps and boundary waiting. These profiled durations include
instrumentation overhead and must not replace the unprofiled throughput table.

| Decode observation | Sync spec | Async spec |
| --- | ---: | ---: |
| Wall time (ms) | 1716.04 | 1668.54 |
| GPU active union (ms) | 274.40 | 274.61 |
| GPU gaps (ms) | 1441.64 | 1393.93 |
| Kernel count | 116295 | 116977 |
| CUDA launch calls | 100136 | 100829 |
| CUDA synchronization calls | 503 | 25 |
| CPU time inside CUDA synchronization (ms) | 4.389 | 0.083 |
| CPU `aten::item` calls | 51 | 0 |
| H2D copies | 451 pageable | 12 pinned |
| D2H copies | 51 | 10 |

Prefill plus first-token preparation occupies 87.37/83.62 ms in the two profiled
captures; its GPU active time is 19.73 ms in both. No prefill speed claim is made.

GPU active time is essentially unchanged. The device-owned frontier removes
candidate/acceptance scalar reads and replaces repeated pageable metadata uploads
with a per-round pinned table snapshot. That matches the reduced synchronization
and copy counts. Extra device bookkeeping slightly increases launch/kernel counts.
The captured decode gap falls by 47.71 ms, whereas time blocked inside CUDA
synchronization falls by only 4.31 ms: the entire difference cannot be attributed
to blocking waits alone. Reduced host coordination is consistent with the measured
speedup; one profiled pair does not isolate every component of that difference.
The eager path remains dominated by launch/dispatch gaps in this trace.

Layer reports use `_paged_attention_kernel` as one anchor per physical layer and
24 anchors per core pass. The selected pass 5 is the second draft loop of the
first decode round, after the four prefill passes; this is a warmed capture,
regardless of the generic analyzer's automatic `cold-start` label on pass 0.
Anchor spans cross module boundaries and are navigation aids, not exact module
hook timings. Representative kernel reports and all-pass tables are retained.

## Qualification limits

These historical engine microbenchmarks do not establish production HTTP
capacity, TTFT or general model quality. Repeated identical prompts do not
represent all serving traffic; concurrency 32 is a tested point, not a saturation
limit. The 2026-10-10 runs cover K=2 and concurrency 1/8; they do not validate K=1/4/8,
concurrency 32, production capacity, other GPUs/checkpoints, random sampling,
adaptive depth, asynchronous CUDA Graphs, FA4, preemption or PD. Keep the
synchronous path until the missing capabilities are covered and validated.
