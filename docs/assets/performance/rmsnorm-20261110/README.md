# RMSNorm validation: B300, 2026-11-10

This is a compact shareable summary of the generic CUDA RMSNorm experiment. The frozen code was PR head `59f964cc741d38643fa22edaafb931da67291f8e` plus a working-tree patch with SHA-256 `7550ee866c404cafa9536eaa254d6e4bd64e35fd43168c72ef1f2b3e4a775a4e`; source and library hashes are in [manifest.json](manifest.json). The native implementation uses FP32 reduction, scale and weight multiplication, converts to output dtype, then adds residual in that dtype. It intentionally need not match the old Torch expression bit for bit.

The experiments used one NVIDIA B300 SXM6 AC (compute 10.3), driver 610.43.02, PyTorch 2.13.0+cu130, Triton 3.7.1 and Python 3.12.3. The frozen implementation passed 354 GPU tests. The local complete report and raw data are retained separately; this compact set omits large traces, raw prompts and raw responses.

## vLLM operator comparison

The vLLM baseline was built from commit `2b678ac739166bd0356883048673bbd1c34a3b0e`. The 420-case contiguous matrix covered BF16/FP16/FP32, H=128/256/768/2048/2049/4096/8192, rows 1–1024 and both residual modes, with 5 randomized-interleaved repetitions after warmup. CUDA event timings were taken from 32-operation Graph replays. With residual, vLLM runs RMSNorm followed by a separate add; its add-before-norm fused operator has different semantics. All matrix outputs passed the frozen dtype-specific Torch-reference tolerances in [contract.json](contract.json). Offset and strided layouts were exercised by the separate GPU tests rather than vLLM's vectorized comparison.

Equal-cell geometric means below are ratios `vLLM time / RLT time` and are not weighted by a production workload. The complete aggregates and extreme shapes are in [micro-summary.json](micro-summary.json); [all 420 cases with raw repetitions and errors](micro-raw.json.gz) are also included.

| dtype | no residual | residual |
|---|---:|---:|
| BF16 | 1.113× | 1.645× |
| FP16 | 1.134× | 1.670× |
| FP32 | 1.036× | 1.519× |

For BF16 H=2048, rows=128, RLT/vLLM were 1.794/2.049 µs without residual and 1.857/3.330 µs with residual. FP32 has meaningful regressions: H=4096, rows=1024 without residual was 6.597/5.197 µs, a 26.9% RLT latency increase. Small ratios near parity should be interpreted cautiously.

## Unprofiled end-to-end speed

Ouro-1.4B BF16, Triton attention, one GPU, fixed four loops, greedy, Graph enabled, ignored EOS and exactly 64 output tokens per request. Each workload was warmed, then three repetitions were measured with old Torch, new native plus separate residual, and new native plus fused residual in `ABC/CBA/BAC` order. The three models were resident at once, but only one submitted GPU work at a time. Inputs were repeated/truncated question tokens, fixed to the input lengths. Throughput includes local scheduler, IPC, prefill and decode; it excludes network time. All requests completed with the specified token and loop count.

| workload | old Torch median tok/s (range) | new CUDA median tok/s (range) | ratio |
|---|---:|---:|---:|
| c1, input 128, 4 requests | 69.17 (69.02–69.29) | 93.72 (93.07–93.72) | 1.355× |
| c8, input 256, 16 requests | 369.48 (366.60–370.51) | 515.79 (515.37–515.99) | 1.396× |

This compares the original Torch expression to the new CUDA path, including their numerical change; it is not a direct measurement against the older PR HEAD's Torch-compatible CUDA implementation. The new-native separate-add medians were 89.22 and 472.11 tok/s. Request latency summaries, Graph counters, per-process memory, and all three observations are in [model-summary.json](model-summary.json). These short workloads do not establish saturation throughput or tail-latency SLOs. One c1 request first diverged at generated token index 47 in all three repetitions, while retaining the fixed work contract.

## Natural-generation quality smoke

Before generation, ten GSM8K test source IDs were selected independently of answers via the repository's SHA-256 selection, seed 0, dataset revision `740312add88f781978c0658806c59bc2815b9866`: `[182, 392, 436, 526, 587, 617, 645, 853, 1094, 1257]`. Greedy BF16 generation used the repository's 3-shot GSM8K strict scorer, one request at a time, maximum 512 new tokens, natural EOS or harness stopping sequences. Every case stopped by a harness sequence; none hit the limit. Three invalid parses were counted wrong in each arm.

| execution | old Torch | new Torch reference | new CUDA | old→CUDA paired losses / gains |
|---|---:|---:|---:|---:|
| fixed four loops | 7/10 | — | 7/10 | 0 / 0 |
| adaptive threshold 0.7, loops 2–4 | 7/10 | 7/10 | 7/10 | 0 / 0 |

In adaptive mode, old→new CUDA differed in tokens on four questions and in exit depth on identical token context on four. New Torch reference→CUDA differed in tokens on two and in exit depth on identical token context on five. On question 182, both new paths first differ from old at zero-based token 2; all three responses were unparseable by the strict scorer. Outputs and exit decisions are therefore not guaranteed identical even though this small paired score is unchanged. This is a smoke screen, not the full 87-question evaluation or evidence for all models. See [quality-summary.json](quality-summary.json) and the paired comparison JSON files for counts and first-divergence IDs.

## Attribution and reproducibility

Paired ops-only and full model profiles used a warmed, complete 128-input/2-output, fixed-four-loop Graph request. The full trace showed 11,004 vs 5,188 GPU kernels old vs new; the first decode replay contained 1,386 vs 659 nodes and spanned 3.361 vs 2.330 ms on the observed GPU stream. Profiled timings are diagnostic, not performance measurements. The [timeline](rmsnorm-graph-timeline.png) uses aligned scales; [profile-summary.json](profile-summary.json) records the capture settings and counts. The profile is a short diagnostic and does not cover all 64 tokens of the primary speed workload.

Reproduce the operator case with `python -m benchmarks.kernels.rms_norm` and the matrix/repeat controls in [contract.json](contract.json), using a source-built vLLM library from the recorded revision. The full local report gives exact commands, source hashes, selector scripts, frozen prompts and raw outputs. It also preserves the separate full and ops-only trace archives; they are intentionally excluded from this compact bundle.
