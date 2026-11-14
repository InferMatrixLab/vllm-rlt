# RMSNorm launch-policy validation — 2026-11-10

The selected resource-based policy removes the dedicated H=2048 kernel and adjusts thread/cache choices using row count, dtype element size and the actual device SM count. On one B300, it improves the two investigated FP32 operator regressions. Its measured cost includes up to 11.6% latency on a tested H=2048 BF16 residual shape after removing the specialization. A separately expanded numerical matrix exposed two **pre-existing** BF16 residual tolerance failures; the new policy produced exactly the same values as the prior CUDA implementation in both. The failures are explained below and are **not a passing result** for that expanded contract.

The final source under test is HEAD `566aef88353ff3e51f2b4beb9aa063281f6f3459` plus the launch-policy, test and documentation diff identified in [final/manifest.json](final/manifest.json). That manifest records source, binary, model, dataset and result hashes. The tested new CUDA source SHA-256 is `542310f69b2453adb48fc4f98087ff28f3a2aa233d32dd0b7cb8007a4c640397`; the loaded extension is `3c132ba489156b36876fdb971e68305583df76bf839ed2f8770ae828a0d5af04`. SM80 and SM103 binaries were built; execution was tested only on SM103. Driver 610.43.02, CUDA runtime 13.0, nvcc 13.4, PyTorch 2.13.0+cu130, Triton 3.7.1, Python 3.12.3, one NVIDIA B300 SXM6 AC with 148 SMs. The GPU was otherwise idle and experiments were serial. Clocks were not locked for timing.

## Selection and measured tradeoffs

For H≤256 the implementation retains 32 threads. Otherwise, start with 256 threads and count 16-byte vectors per thread. If there are fewer rows than SMs and a 256-thread block would need more than four vectors per thread, use 512 threads when the device supports them. After selecting thread count, cache one, two or four vectors per thread, or reread if more are needed. At two or more rows per SM, reread rather than retain more than two vectors per thread. The 16-byte vector holds four FP32 or eight FP16/BF16 elements. There is no exact-width, model-name or GPU-model rule; the earlier static H=2048 template is removed.

Three candidate policies and a static-removal-only control were frozen before the comparison. [POLICIES.md](POLICIES.md) and [contract.json](contract.json.gz) state their complete rules and the 342-case grid: three dtypes, both residual modes, core widths and rows plus adjacent width/SM-count boundaries. Each cell used shape warmup and five randomly interleaved, unprofiled CUDA-event repetitions of 32-operation Graphs; the candidate arms and the `replica_static` old control shared the same output tensor. `replica_static` was checked bitwise against the prior production binary. Ratios below are **old `replica_static` time / candidate time**; values above one favor the candidate, with equal weight per cell, not production-workload weighting. [policy-summary.json](policy-summary.json.gz) records the inverse convention, candidate/old latency.

| policy | geometric mean old/new | >5% faster | within ratio 0.95–1.05 | >5% slower |
|---|---:|---:|---:|---:|
| Remove static path only | 0.995× | 0 | 328 | 14 |
| Conservative | 1.012× | 21 | 306 | 15 |
| **Moderate, selected** | **1.015×** | **25** | **303** | **14** |
| Balanced | 1.030× | 57 | 261 | 24 |

Moderate retains four-pack caching until two rows per SM, avoiding some losses at the one-SM boundary while still helping underfilled wide rows. Representative Graph medians from the same shared-output comparison are below. These are operator timings, not end-to-end throughput.

| dtype / H / rows / residual | old µs | selected µs | observation |
|---|---:|---:|---|
| FP32 / 4096 / 1024 / no | 6.616 | 5.312 | reduce cache pressure |
| FP32 / 8192 / 4 / no | 3.650 | 2.690 | 256→512 threads; cache choice also changes after thread selection |
| FP32 / 8192 / 128 / no | 3.970 | 3.074 | 256→512 threads |
| BF16 / 8192 / 148 / no | 2.818 | 2.818 | balanced candidate was 3.074 µs |
| BF16 / 2048 / 1024 / yes | 2.755 | 3.074 | +11.6%, static path removed |
| FP16 / 2048 / 1024 / yes | 2.765 | 3.074 | +11.2%, static path removed |
| FP16 / 8192 / 296 / no | 3.097 | 3.317 | +7.1%, two-SM cache threshold |

The 14 cells with old/new ratio below 0.95 include 13 H=2048 cells and the FP16 H=8192/296 example. Small differences and the equal-cell geometric mean are descriptive only. The raw per-repeat data and exact capture order are in [policy-results.json.gz](policy-results.json.gz). After final compilation, [final/verify-policy.json.gz](final/verify-policy.json.gz) confirms the new production extension matched the evaluated moderate arm **bitwise on all 342 generated inputs**. Compared to prior static CUDA, all BF16 and FP16 cases were bitwise equal; nine FP32 cases differed after changing the reduction tree, with maximum absolute new/old difference 9.54e-7 and all nine passing rtol=atol=2e-6 against old CUDA. This does not override the separate reference failures below.

## Why the original FP32 shapes regressed

In a separately warmed, seven-repetition shared-output ablation on the previous implementation, H=4096/1024 rows used 6.596 µs with four cached 16-byte vectors per thread and 5.373 µs when rereading with the same 256 threads; vLLM was 5.215 µs. NCU observed 40 vs 32 registers/thread and a register-limited maximum of 6 vs 8 blocks/SM, with 66.17% vs 72.29% active warps on active SMs. The reread path issued 50.33 MB rather than 33.55 MB of global load requests. In this **warm profiled** sample the extra requests were largely served by cache rather than HBM; the measurements do not imply that rereading costs no memory bandwidth in other workloads.

For H=8192/4 rows, holding reread behavior fixed gave 3.648 µs at 256 threads and 2.944 µs at 512 threads; vLLM's 1024-thread result was 3.073 µs. The grid has only four row blocks, limiting the number of concurrently used SMs, while more threads increase parallelism inside each block. Thread count also changes reduction and scheduling, so this is not a single-instruction attribution.

A diagnostic forcing six blocks/SM via unused dynamic shared memory made the reread variant slower, but also reduced observed L1 hits substantially; its occupancy and cache effects are confounded. BF16 H=4096/1024 is an important opposite control: cached/streamed measured 3.268/3.801 µs. Likewise, a scalar-vs-paired-FP32 arithmetic ablation affected latency but does not independently quantify the scheduling, cache and occupancy contributions. [regression-ncu-summary.json](regression-ncu-summary.json) and [regression-sass-summary.json](regression-sass-summary.json) contain profiler/resource summaries. Profiler timings are diagnostic and do not supply the throughput figures above.

## Numerical checks and known tolerance boundaries

The predeclared comparisons against the new FP32 Torch reference use rtol/atol `2e-6` FP32, `2e-3` FP16, `1e-2` BF16. The final production binary completed the **original 420-cell** seed-42 dtype/width/row/residual matrix, including vLLM, and all arms passed those per-element checks. The separately seeded, expanded **342-cell** policy matrix did not: prior CUDA and all candidate arms each had two BF16 residual cells with one failing element apiece. vLLM had two failing cells, one of which was different. No candidate-only failing cell was observed. All exceptions, indices and values remain in [policy-summary.json](policy-summary.json.gz) and [policy-results.json.gz](policy-results.json.gz); no tolerances were relaxed.

The two old-and-new CUDA failures were H=4095/1024 rows and H=8192/296 rows. At H=4095, an instrumented copy produced the same complete output as the unmodified old binary and exposed a weighted FP32 value of `2.4453127384`, while Torch's was `2.4453122616`; the BF16 midpoint is `2.4453125`. The standalone rounded values are therefore `2.453125` and `2.4375`; adding BF16 residual `-2.546875` gives `-0.09375` vs `-0.109375`. At H=8192, the instrumented old cached and new reread versions had the same reduction result and output; their pre-cast value `-2.0703127384` and Torch's midpoint `-2.0703125` round to adjacent BF16 values. Residual cancellation exposes the same absolute `0.015625` difference. In both cases the fused residual output was bitwise equal to standalone RMSNorm followed by a separate BF16 add. The evidence supports a reduction/rounding boundary rather than an observed residual-fusion or indexing error, but it does not establish the declared tolerance for every input.

[numeric-diagnosis.json](numeric-diagnosis.json) retains the input/weight/residual values, complete-row FP64 sum, Torch FP32 calculation, standalone and residual outputs. [numeric-probe.json](numeric-probe.json) records the instrumented reduction. CPU `float64 → bfloat16` conversion can double-round through FP32 near a midpoint, so the diagnosis separately reports the actual CPU conversion and a direct decision from the FP64 value and adjacent BF16 midpoint. FP64 is diagnostic; the specified kernel arithmetic remains FP32.

The final GPU suite produced [378 passed](final/gpu-tests.txt), including rank 1–4, three dtypes, independent offsets, with/without residual and wide rows. Additional FP32 H=4095 tests used 299/302/300 rows in rank 2/3/4, different input/residual strides and input/weight/residual address offsets of 4/12/8 bytes; both residual modes passed the same FP32 tolerance. A short trace confirmed the nonaligned 256-thread streaming kernel actually executed. Results are in [final/stride-check.json](final/stride-check.json). Ruff lint/format and the CPU regression hook passed, as recorded in [root-checks.txt](root-checks.txt).

## Final unprofiled model comparison

The performance baseline here is the **previous committed generic CUDA extension**, not the older Torch expression. Old and new extensions ran in separate workers, each verifying its binary SHA-256. Both models were resident on one GPU, but only one submitted work at a time. Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, BF16 weights/activations/KV, Triton attention, fixed four loops, greedy, Graph enabled, `ignore_eos=True`, exactly 64 outputs per request. KV was 1536×16 blocks, max sequences 8, max batched tokens 1024, prefill chunk 256. Prompt IDs were repeated/truncated to the stated lengths. Each shape was warmed before three interleaved `AB BA AB` runs. All requests completed with equal token count and depth; tokens and exit depths matched in every paired request.

Throughput is successful output tokens divided by the local first-submit-to-last-completion window, including scheduler, IPC, prefill and decode but excluding network. TTFT/TPOT/E2E are local client observations; TPOT is `(last − first) / 63`. Ranges below are across runs, with p50 values the median of per-run p50s.

| workload | prior CUDA tok/s median (range) | new CUDA tok/s median (range) | prior→new p50 TTFT / TPOT / E2E ms |
|---|---:|---:|---|
| c1, input 128, 4 requests, 256 output tokens/run | 93.325 (93.089–93.344) | 94.071 (93.949–94.311) | 25.374/10.482/685.600 → 24.674/10.401/680.055 |
| c8, input 256, 16 requests, 1024 output tokens/run | 512.513 (511.330–514.494) | 515.086 (512.276–515.099) | 84.250/14.011/971.914 → 84.246/13.986/964.621 |

Median changes are +0.8% and +0.5%; c8 ranges overlap, and these short runs do not establish a significant model speedup, steady-state saturation or tail SLO. Request/KV allocation returned to zero after each run; Graph fallback counts were zero. Each c8 worker's PyTorch peak allocated/reserved memory was 7.423/7.648 GiB, not total co-resident GPU occupancy or peak useful KV. Full repetitions, p95 and hashes are in [final/model-summary.json](final/model-summary.json.gz) and [final/model-ab.json.gz](final/model-ab.json.gz).

## Natural-generation quality smoke

The original fixed seed-0 SHA-ranked GSM8K 10-question protocol was reused without changing IDs, prompts, model, strict scorer or generation controls: three-shot greedy BF16, one request at a time, ≤512 new tokens, harness stop sequences or EOS. New CUDA was rerun for fixed four loops and adaptive threshold 0.7 / loops 2–4; the earlier prior-CUDA, old-Torch and new-Torch-reference results are reused only as the recorded same-protocol baselines.

| mode | prior CUDA | new CUDA | paired old→new loss / gain | token / depth differing questions |
|---|---:|---:|---:|---:|
| Fixed four loops | 7/10 | 7/10 | 0 / 0 | 0 / 0 |
| Adaptive | 7/10 | 7/10 | 0 / 0 | 0 / 0 |

Each run had three unparseable answers counted wrong and no length-limited responses. Raw responses were re-scored and prompt/sample hashes checked. Comparing to the earlier *old Torch expression*, both new results also scored 7/10 with no paired loss/gain, but four adaptive questions differ in token sequence and four have a different exit depth on identical token context; the old Torch and CUDA arithmetic are not generally bitwise equivalent. This ten-question smoke screen does not qualify the full 87-question case, other models or broader quality. [final/quality-summary.json](final/quality-summary.json), [adaptive prior-CUDA comparison](final/compare-adaptive-prior-native.json), [fixed prior-CUDA comparison](final/compare-fixed-prior-native.json), [adaptive old-Torch comparison](final/compare-adaptive-old-torch.json) and [fixed old-Torch comparison](final/compare-fixed-old-torch.json) record the counts.

## Final 420-cell comparison and reproduction

The final 420-cell original matrix used BF16/FP16/FP32 × widths 128,256,768,2048,2049,4096,8192 × rows 1,2,4,8,16,32,64,128,256,1024 × residual off/on, seed 42, five interleaved repeats × 50 iterations × Graph32 after warmup. vLLM source is `2b678ac739166bd0356883048673bbd1c34a3b0e`; with residual it performs RMSNorm followed by a separate add, not vLLM's different add-before-norm fused operation. The production benchmark uses each implementation's original output allocation behavior; do not mix its ratios with the shared-output policy table.

| dtype | vLLM/new geometric mean, no residual | vLLM/new geometric mean, residual |
|---|---:|---:|
| BF16 | 1.109× | 1.636× |
| FP16 | 1.130× | 1.660× |
| FP32 | 1.068× | 1.552× |

These are equal-cell geometric means, not workload-weighted. FP32 still has regressions: H=2049, rows=8, no residual measured 2.690 µs new vs 2.497 µs vLLM, about 7.7% higher latency. [final/micro-summary.json](final/micro-summary.json.gz) and [final/micro.json.gz](final/micro.json.gz) retain all cases and repetitions. Final ops-only/full operator profiles were captured separately; prior model profiles describe the earlier fusion mechanism and are historical, not the source of the new-binary performance result. The full raw traces, scripts, selected failing input rows and source-derived diagnostic variants are retained locally outside this compact bundle.

Reproduce on one verified idle GPU with the pinned environment and source hashes:

```bash
PY=/home/zjy/code/david/b_workspace/.b_rdma/bin/python
ART=artifacts/rmsnorm-policy-20261110
TORCH_CUDA_ARCH_LIST='8.0;10.3' MAX_JOBS=2 "$PY" setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=3 "$PY" -m pytest tests/test_normalization.py tests/test_ouro.py \
  tests/test_activation.py tests/test_native_build.py --run-gpu -q
CUDA_VISIBLE_DEVICES=3 "$PY" -m benchmarks.kernels.rms_norm \
  --vllm-library artifacts/rmsnorm-generic-20261109/vllm-source-build/build/vllm_rms_review.so \
  --vllm-source-commit 2b678ac739166bd0356883048673bbd1c34a3b0e \
  --tokens 1,2,4,8,16,32,64,128,256,1024 --widths 128,256,768,2048,2049,4096,8192 \
  --dtypes bfloat16,float16,float32 --repeats 5 --iterations 50 \
  --output "$ART/final/micro-rerun.json" --profile-dir "$ART/final/micro-rerun-profiles"
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 "$PY" "$ART/final/stride_check.py"
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 "$PY" "$ART/final/model_compare.py" \
  --mode graph --repeats 3 --output "$ART/final/model-rerun.json"
PYTHONPATH=. HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 RLT_RMS_QUALITY=native \
  "$PY" artifacts/rmsnorm-generic-20261110/accuracy.py run --backend native \
  --protocol artifacts/rmsnorm-generic-20261110/gsm8k-protocol.json \
  --output "$ART/final/quality-rerun-adaptive" --exit-threshold 0.7 --min-loops 2
```

Omit the adaptive flags for fixed four-loop generation. The local model comparison requires both preserved old and newly built extension binaries; it verifies their hashes. The source-built vLLM library and exact frozen prompts are needed as recorded in the manifest. The report's compact archive intentionally does not contain the large traces, local scripts or model weights. The earlier compatibility-kernel and pre-policy reports are historical and must not be substituted for the final policy measurements here.
