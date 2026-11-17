# Ouro on Ascend

Run Ouro on one Ascend NPU with BF16, synchronous greedy decoding and four
recurrent loops. Use the explicit `ascend_eager` attention backend.

Sections 1–5 describe the synchronous path and its historical measurements.
Section 6 covers the optional asynchronous engine plan. Sections 7–8 record
successive serving campaigns and their qualification limits. Sections 7–8 record
successive serving campaigns and their qualification limits.

## 1. Prepare the environment and model

Use a compatible Torch, Torch-NPU and CANN environment with the project's
Python dependencies installed. Keep the vendor Torch pair; CUDA/Triton
packages are not needed for this backend.

~~~bash
source /path/to/cann/set_env.sh
source /path/to/npu-environment/bin/activate
python -m pip install --no-deps --no-build-isolation -e .
export MODEL=/path/to/prepared/Ouro-1.4B
~~~

Use the checkpoint's own config and tokenizer. Model loading uses the shared
native Ouro loader, including its supported sharded checkpoint layout.

## 2. Start the server

Select an allocated physical NPU using `ASCEND_RT_VISIBLE_DEVICES`.

~~~bash
ASCEND_RT_VISIBLE_DEVICES=<allocated-device> \
python -m vllm_rlt.entrypoints.serve \
  --model "$MODEL" --tokenizer "$MODEL" --served-model-name ouro \
  --device npu --dtype bfloat16 --attention-backend ascend_eager \
  --num-blocks 1024 --max-num-seqs 1 \
  --max-num-batched-tokens 4096 --prefill-chunk-size 4096 \
  --host 127.0.0.1 --port 8123
~~~

Explicit KV capacity is required. Adjust it for the model, available memory,
prompt/output lengths and concurrency. To compare with full-prefill inference,
both the chunk size and batched-token budget must fit the whole prompt.

## 3. Send a request

~~~bash
curl --fail http://127.0.0.1:8123/health
curl --fail http://127.0.0.1:8123/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ouro","prompt":"The capital of France is","max_tokens":8,"temperature":0,"min_loops":4,"max_loops":4,"exit_threshold":1}'
~~~

Wait for HTTP 200 before sending requests. Add `"stream": true` for SSE.
The `ascend_eager` backend follows HF-style BF16 attention rounding;
`--attention-backend torch` selects the existing FP32-intermediate reference.

## 4. Run checks

With an assigned NPU visible to the test process:

~~~bash
pre-commit run --all-files
VLLM_RLT_OURO_CHECKPOINT="$MODEL" python -m pytest \
  tests/test_ascend.py tests/test_ascend_checkpoint.py \
  tests/test_ascend_eager.py tests/test_ascend_kv.py \
  tests/test_ascend_serving.py --run-npu -q
~~~

These checks cover causal attention, cross-page KV, request reuse and HTTP/SSE.
They do not establish model quality or performance; use the
[accuracy guide](../accuracy.md) and [serving benchmark guide](../serving.md)
for those evaluations.

Graphs, speculation, PD, preemption, prefix caching and adaptive exit are
unsupported on this path. See section 6 for fixed-depth asynchronous execution.

## 5. Attention batching and device-side KV writes

Two follow-up changes reduce host submissions on this path without changing the
BF16 arithmetic contract:

- `ascend_kv.py` builds the Scatter ND write index from the batch's staged
  device tensors (`write_blocks` / `write_offsets`) instead of a Python loop
  over `batch.rows`; a fail-closed check rejects a row/address count mismatch.
- `ascend_eager.py` merges equal-length whole-prompt prefill into one
  `[size, width]` attention call, and merges independent single-row decode
  groups into one ragged gather. Write slots, ragged plans and the prefill mask
  are cached per batch. A single decode row keeps the original per-group path
  so it does not pay for plan construction.

The arithmetic is unchanged: BF16 score scaling, additive BF16 causal mask,
FP32 softmax cast back to BF16 probabilities, then a BF16 matmul. The default
backend is still `AscendEagerAttention`; no fused CANN attention operator is
used, because those operators do not reproduce the released HF eager rounding.

### Measurement conditions

One Ascend 910B3, CANN 9.0.0, Torch 2.9.0+cpu with torch_npu 2.9.0,
Ouro-1.4B BF16, fixed four recurrent loops, synchronous greedy decoding,
full prefill, frozen prompt IDs, forced 32-token output, profiler off.
Three arms share one process on the same card (NPU 6, `taskset -c 64-71`):
official HF eager as reference, the pre-change per-group eager attention,
and the merged implementation. Each arm gets one warmup and three timed
repetitions; throughput, TTFT and TPOT are medians over request samples.
P2048/C8 uses a 16384 token budget to preserve full-prefill geometry.

### Before/after results (same hardware, different card assignment)

| Input / output / concurrency | Arm | Output tok/s | TTFT (ms) | TPOT (ms) |
| --- | --- | ---: | ---: | ---: |
| 128 / 32 / 8 | HF eager | 50.792 | 158.9 | 157.5 |
| | before | 14.611 | 630.7 | 544.8 |
| | **merged** | **37.724** | **273.0** | **209.9** |
| 2048 / 32 / 1 | HF eager | 6.339 | 371.2 | 150.9 |
| | before | 4.784 | 531.3 | 198.7 |
| | **merged** | **4.789** | **534.7** | **198.3** |

A second matched run on longer/higher-concurrency workloads (same protocol,
different process, so absolute values are not cross-comparable with the table
above):

| Input / output / concurrency | Arm | Output tok/s | TTFT (ms) | TPOT (ms) |
| --- | --- | ---: | ---: | ---: |
| 2048 / 32 / 8 | HF eager | 35.835 | 2489.5 | 150.1 |
| | before | 12.426 | 3702.4 | 545.1 |
| | **merged** | **25.007** | **3591.4** | **213.7** |
| 4096 / 32 / 1 | HF eager | 5.812 | 1091.1 | 142.4 |
| | before | 4.170 | 1532.7 | 196.3 |
| | **merged** | **4.244** | **1534.5** | **193.7** |

The gain is in decode/overall throughput, not long-prompt TTFT: prefill is
dominated by the dense `[W, W]` attention and by host submission, and merging
calls reduces launches rather than arithmetic. Relative to HF eager, C8
throughput moves from 0.288x to 0.743x at input 128; HF eager remains faster
in absolute terms.

An earlier, separate A/B isolated the device-side write index alone (different
harness, older configuration, directional only): P2048/C1 TTFT 790.7 -> 531.9 ms
(-32.7%) and P128/C8 TTFT 691.7 -> 619.8 ms (-10.4%).

### Full W1–W7 serving campaign

`docs/unified_execution.md` defines the project's full serving benchmark: W1–W7,
three unprofiled repetitions per workload, 60 seconds of closed-loop admission
followed by full drain, median aggregate output throughput, pooled TTFT / TPOT /
ITL / E2E percentiles, fixed lengths for W1–W5, and a fixed seeded ShareGPT
selection with natural EOS and a 512-token cap for W6/W7. The merged candidate
ran as a single arm on the frozen campaign actor with no runtime monkeypatch on
NPU 5/6/7. The before column is the `d921719` baseline published in PR #168 on
2026-11-06; card assignment differs between the two runs, so this is a
same-hardware, not same-run, comparison.

| Workload | Input / output / C | before tok/s | merged tok/s | before TTFT p95 (s) | merged TTFT p95 (s) | before TPOT p95 (ms) | merged TPOT p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 128 / 128 / 1 | 3.944 | 4.653 | 0.426 | 0.205 | 265.8 | 221.1 |
| W2 | 128 / 128 / 32 | 4.802 | 96.726 | 9.737 | 6.827 | 6642.6 | 285.9 |
| W3 | 4096 / 128 / 1 | 3.727 | 4.368 | 2.167 | 1.721 | 254.3 | 222.1 |
| W4 | 4096 / 128 / 32 | 4.201 | 24.761 | 747.440 | 137.225 | 1794.4 | 236.0 |
| W5 | 1024 / 512 / 8 | 4.575 | 30.744 | 4.676 | 2.551 | 1743.0 | 256.0 |
| W6 | ShareGPT / ≤512 / 8 | 4.660 | 21.417 | 3.572 | 1.689 | 1666.7 | 317.5 |
| W7 | ShareGPT / ≤512 / 32 | 4.760 | 27.970 | 15.765 | 7.289 | 6523.6 | 1071.7 |

All seven workloads completed three unprofiled repetitions with zero failed
requests. Successful requests over the three runs (before → merged): W1 6→9,
W2 96→192, W3 6→9, W4 96→120, W5 24→24, W6 27→33, W7 99→105.

The W2 ratio is dominated by the baseline's lack of concurrency scaling rather
than by a per-request speedup: on `d921719`, raising concurrency from 1 to 32
moved aggregate throughput only from 3.944 to 4.802 tok/s (1.22×) while TPOT
p95 grew from 265.8 ms to 6642.6 ms, whereas the merged path reaches 20.79× the
C1 throughput with TPOT p95 of 285.9 ms. The same near-flat baseline scaling
appears in W4, W5, W6 and W7. Under the W1–W7 protocol a faster configuration
also covers more inputs during the same 60-second admission window (W2
completed 32 requests per run before versus 64 after, with drain falling from
about 792 s to 25 s), so the aggregate ratio reflects both a shorter drain and
a larger completed-request count.

The fresh campaign quality gate (GSM8K-87 three-shot strict and the paired
same-history logit vectors) has not been re-run on this change. The earlier
`kv-read-full-20261106` campaign recorded 63/87 for both HF eager and the
target, 87/87 exact sequences and 8952/8952 exact logit vectors with
`max_abs=0.0` on `d921719` and a different KV-read candidate; those figures do
not qualify this change.

### Correctness gates

- CPU synthetic parity in BF16: regular prefill, single-request prefill, ragged
  decode, a mixed-length ragged batch with `-1` padded block tables, single
  decode and an irregular grid all match the per-group path exactly
  (`atol=0, rtol=0`); path-activation counters are asserted.
- Ascend selection suite on the merged candidate: `test_ascend.py`,
  `test_ascend_eager.py`, `test_ascend_kv.py`, `test_ascend_serving.py`,
  `test_ascend_checkpoint.py` with `--run-npu`: **31 passed, 0 skipped,
  0 failures** (66.6 s, NPU 6), including the new batched-path parity cases.
- Generated sequences against official HF: 24/24 at C8 and 3/3 at C1.
- Pre-lm-head vectors: 256/256 at C8 and 32/32 at C1, `max_abs=0.0`,
  for both `merged vs HF` and `merged vs before`.
- In the timed C8 run, the batched-prefill path activated 384 times and the
  ragged decode path 11904 times, with zero fallbacks.

### Limits

- Batched prefill requires a strict regular grid. **Chunked prefill does not
  activate it**, so the documented case-203 chunk128 sequence mismatch
  (27 vs 31, first divergence at the 15th output token) is **still unresolved**;
  this follow-up does not change chunked-prefill behavior or claim a fix.
- Multi-row non-initial groups still use the original path. Adaptive exit,
  long conversations, C16+ concurrency and models other than Ouro-1.4B are
  untested.
- Three timed repetitions per arm on a single card; only 32-token outputs.
- Only the full-prefill synchronous path is qualified. Async scheduling,
  graphs, speculation, PD, preemption, prefix caching and adaptive exit
  remain unsupported here.


## 6. Fixed-depth asynchronous engine plan

This opt-in path reduces host submissions and KV gathering without replacing
model operators or replaying graphs. It reuses stable batch allocation/range
plans, combines fixed-depth decode submissions, and consumes owned KV in
`[batch, kv_head, token, head_dim]` layout. Submitted-write coverage is distinct
from device completion: stream events still protect readers, cancellation,
request-ID reuse and release. Generic cache access exports live KV into paged
storage and can temporarily require both representations.

Use `CacheConfig(contiguous_kv=True)` and
`ExecutionConfig(async_scheduling=True, multi_stream=False, use_uva=False,
coalesce_fixed_prelude=True, native_eager_plan=True)`, with
`attention_backend="ascend_eager"`, `ExitConfig("ouro_delayed")` and greedy
`SamplingParams(min_loops=4, max_loops=4, exit_threshold=1.0, temperature=0.0)`.
Allocate explicit KV blocks and enough prefill tokens for complete prompts.
Graphs, static buffers, padding, prefix sharing, incremental allocation,
speculation, PD and preemption are unsupported in this configuration. Arbitrary
chunked/mixed-length prefill and adaptive-depth quality are not qualified.

A checkpoint CLI smoke command is:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 TASK_QUEUE_ENABLE=2 OMP_NUM_THREADS=1 \
python -m vllm_rlt.entrypoints.cli \
  --model /work/models/Ouro-1.4B --device npu --dtype bfloat16 \
  --attention-backend ascend_eager --prompt "Explain why the sky is blue." \
  --max-tokens 4 --min-loops 4 --max-loops 4 --exit-mode ouro_delayed \
  --async-scheduling --single-stream --no-uva --contiguous-kv \
  --coalesce-fixed-prelude --native-eager-plan --num-blocks 256 \
  --prefill-chunk-size 4096 --max-num-batched-tokens 4096
```

`worker/ascend.py` owns the NPU runtime, token/hidden batch reuse and batched
sampling. CUDA/ROCm retain the shared runner, runtime, sampler, workspace and
model implementation. The optional native executor in that module and
`worker/csrc/ascend_eager_plan.cpp` dispatches individual eager ATen operations
with the Python GIL released: Q/K/V and gate/up projections stay separate, with
explicit RMSNorm, RoPE, matmul and softmax. Each call borrows live weights and
returns owned outputs; unsupported module schemas, hooks/custom forwards or
paged storage use Python. Changes to Ouro arithmetic require revalidating this
executor too. The older experimental paged KV read cache has been removed.

Native dispatch requires a C++17 compiler, Ninja, PyTorch headers and a writable
cache on first enabled startup. `VLLM_RLT_NATIVE_BUILD_DIR` selects that cache;
`MAX_JOBS=1` limits build parallelism. Its identity includes source, Torch version,
C++ ABI and Python SOABI. Disable `native_eager_plan` to avoid compilation.
CPU/CUDA/ROCm do not import or build this extension. The FP32 CPU generation of
RoPE constants is an explicit NPU-only numerical correction to match HF;
CUDA/ROCm retain their original device arithmetic.

With refill or mixed scheduling, pending prompts or runnable recurrent work
temporarily disable full-depth decode prelude coalescing. Keeping the ordinary
PRELUDE-to-RECURRENT boundary lets other requests make progress; stable batches
without competing work still use the coalesced path.

The November 17 refinement integrates upstream `c338aaa`, preserving its sliding
windows, automatic `packed_prefill` policy and CUDA state-pool changes. Runtime
source manifests, frozen inputs, pinned HF source, raw request records and
reproduction scripts are in `/work/vllm-rlt-engine-20261117` inside `kelei-rlt`
on `ssh npu`. Fixed and natural-input quality checks are separate from timing;
they do not establish a task-accuracy score or W1–W7 qualification.

### Prior O32 validation (`028f9c6`)

These measurements predate the refill follow-up below and use different output
lengths and token budgets from section 7.

One NPU per case, CANN 9.0.0, Torch/Torch-NPU 2.10, Transformers 4.54.1,
Ouro-1.4B BF16, UT4, greedy output, no graphs, one execution stream. All arms
set `TASK_QUEUE_ENABLE=2` and OMP/MKL threads to one before import. Each has a
complete shape-matched warmup, then 60 seconds of closed-loop admission and full
drain. Rates include drain, exclude startup/compilation/warmup, and are medians of
three unprofiled runs. Orders alternate native/HF, HF/native, native/HF on the
same card and CPU affinity. These are engine-actor timings, not HTTP latency.

| Prompt / output / client C | Resident batch | Token budget / chunk / blocks | HF tok/s | Native tok/s | vs HF |
| --- | ---: | --- | ---: | ---: | ---: |
| 128 / 32 / 8 | 8 | 1024 / 1024 / 336 | 50.732 | 80.625 | +58.92% |
| 128 / 32 / 64 | 64 | 8192 / 1024 / 2576 | 348.660 | 512.346 | +46.95% |
| 4096 / 32 / 8 | 4 | 16384 / 4096 / 4144 | 13.636 | 17.210 | +26.21% |

Blocks hold 16 tokens. Inputs are frozen complete equal-length prompts; timing
forces 32 output tokens with EOS ignored. Native exceeds HF in all nine paired
case/repeat comparisons. All 18 runs have zero failed requests; all 3,588 native
request outputs (114,816 tokens) match the independent HF oracle, with no native
fallbacks. This is not a chunked-prefill, saturation or W1–W7 campaign.

Throughput does not imply faster first output. Pooled TTFT p50 is 222.4 vs
157.5 ms at S8 and 634.5 vs 595.5 ms at S64 (native vs HF). L8 includes queuing
at eight clients/four resident requests: 11.68 vs 13.54 s. Raw request records,
variability, latency percentiles and peak allocations are in `slim-summary.json`
and `slim-results` in the experiment bundle.

The prior refinement passed 1,117 CPU regression tests (800 device tests skipped),
81 targeted Ascend tests, three additional real-checkpoint regressions and three
HTTP/SSE/disconnect regressions (Torch sync, Ascend sync and Ascend async).
Separate fixed-input quality checks pass 192/192 sequences; 16 untruncated natural
inputs also match HF under normal EOS with a 32-token cap (all reach the cap).
Ruff and source/wheel packaging checks pass. Staged CUDA compatibility runs on
`ict-90` pass 1,736 unique tests, including 619 GPU cases (181 skipped), and five
paired real-checkpoint configurations match the baseline. These runs do not
qualify unsupported SM120 paged FlashAttention or NIXL configurations. ROCm
hardware validation is deferred.

## 7. Unified workload follow-up (`394f3b1`, quota1) (`394f3b1`, quota1)

The refill follow-up preserves a RECURRENT scheduling opportunity when full-depth
PRELUDE would otherwise bypass pending prompt or recurrent work. Bounded CPU
regressions reproduce both starvation cases before the fix; the final tree passes
1,121 CPU cases (802 skipped), 51 targeted CPU/NPU cases, and full-tree Ruff checks.
The NPU-only option defaults off; its disabled CUDA/ROCm path is unchanged.

This compares the whole enabled combination with official HF eager using the seven
workloads and measurement rules in [unified execution](../unified_execution.md#protocol-and-comparison-rules).
All 42 selected runs complete: three unprofiled 60-second closed-loop admission
windows per arm/workload, each followed by full drain. They contain 1,087 successful
requests and zero runtime failures. Model loading, compilation and full-output
cohort warmup are excluded. These are in-process actor measurements, without HTTP
or StageClient IPC.

BF16, fixed UT4, greedy, one execution stream, graphs disabled. The engine uses
async/refill, budget/chunk 1024, contiguous KV and native host plans with fixed-depth
decode/prelude coalescing. W1–W5 ignore EOS; W6/W7 replay the frozen seed-0 selection
of 256 untruncated ShareGPT first-human messages, without a chat template, using
natural EOS and cap 512. Both arms use TASK_QUEUE_ENABLE=2, one Torch/OMP/MKL thread,
and one fixed card/CPU affinity per workload; order is native/HF, HF/native,
native/HF. The guide's CUDA FA4/graphs/PD/speculation modes are outside this NPU run.

The 64 GiB device requires a capacity adjustment: W4 retains 32 clients with four
resident requests, W6/W7 at most eight residents. Both arms use the same limits.
Logical KV blocks are `max(256, residents * ceil((input_bound + output_cap)/16) * 4 + 16)`,
with a 4096-token input bound for natural requests. HF uses ordinary batched eager
and retains finished rows until its active cohort drains.

Median output tok/s [three-run min–max]; throughput includes drain:

| Workload | Input / output / clients | Residents | HF eager | Engine combination | vs HF |
| --- | --- | ---: | ---: | ---: | ---: |
| W1 | 128 / 128 / 1 | 1 | 6.152 [5.998–6.464] | 10.191 [8.951–10.274] | +65.65% |
| W2 | 128 / 128 / 32 | 32 | 192.660 [184.098–204.178] | 131.489 [113.793–148.783] | -31.75% |
| W3 | 4096 / 128 / 1 | 1 | 6.198 [6.028–6.318] | 8.379 [5.606–9.152] | +35.20% |
| W4 | 4096 / 128 / 32 | 4 | 20.449 [20.192–20.910] | 16.045 [15.719–16.325] | -21.54% |
| W5 | 1024 / 512 / 8 | 8 | 51.508 [50.409–52.387] | 40.692 [39.713–40.696] | -21.00% |
| W6 | ShareGPT / EOS cap 512 / 8 | 8 | 25.353 [23.451–25.831] | 35.410 [33.534–37.133] | +39.67% |
| W7 | ShareGPT / EOS cap 512 / 32 | 8 | 35.557 [35.320–35.800] | 35.961 [34.726–36.356] | +1.14% |

W7's 1.14% median difference lies within the observed run ranges. W3 also has
substantial engine variability; one paired run is slower than HF. This matrix
supports case-dependent performance, not a universal speedup.

Native layer execution is active throughout W1/W3 with zero fallback. W2/W4/W5/W6/W7
materialize paged KV during warmup and have zero native-executor layer calls in
the timed windows; those layers use the Python eager fallback. Enabling the flag
does not establish native-path coverage for these mixed/chunked geometries.

Complete token sequences and finish reasons match the paired HF requests as follows:

| Workload | Identical / paired requests | Supplemental HF reference coverage |
| --- | ---: | --- |
| W1 | 10/10 | All 15 engine requests match HF |
| W2 | 221/224 | None |
| W3 | 9/9 | All 13 engine requests match HF |
| W4 | 93/108 | None |
| W5 | 18/24 | None |
| W6 | 15/27 | None |
| W7 | 45/99 | None |

The supplemental checks cover three additional fixed inputs after all timing ends.
A bounded natural-input diagnostic also finds different HF outputs at batch 1 and
batch 8; this does not explain all engine/HF differences. Full greedy equivalence
and equal-quality performance acceptance have not passed. Natural EOS output
lengths differ: mean HF/engine output is 458.33/452.44 tokens for W6 and
457.15/466.58 for W7. Fresh GSM8K-87 task qualification has not been run.

Latency triples are mean / p95 / p99, pooling successful requests across all three
runs. TTFT and E2E include queuing; TPOT is first-to-last arrival divided by N−1.
ITL pools adjacent token delivery intervals, with zero intervals within a delivered
chunk. Small-sample tail percentiles are descriptive.

| Workload / arm | TTFT (ms) | TPOT (ms) | ITL (ms) | E2E (s) |
| --- | ---: | ---: | ---: | ---: |
| W1 / HF | 158.26 / 164.42 / 164.72 | 160.67 / 174.41 / 177.74 | 160.67 / 175.26 / 292.54 | 20.56 / 22.31 / 22.73 |
| W1 / engine | 210.43 / 252.72 / 313.55 | 101.54 / 119.08 / 128.71 | 101.54 / 153.61 / 161.31 | 13.11 / 15.38 / 16.66 |
| W2 / HF | 315.94 / 316.78 / 316.87 | 164.36 / 174.60 / 174.60 | 164.36 / 174.36 / 241.96 | 21.19 / 22.49 / 22.49 |
| W2 / engine | 807.75 / 1535.41 / 1569.18 | 237.03 / 330.89 / 334.54 | 237.03 / 365.28 / 448.81 | 30.91 / 42.86 / 43.11 |
| W3 / HF | 1126.36 / 1127.27 / 1127.31 | 154.25 / 159.01 / 159.36 | 154.25 / 164.21 / 166.66 | 20.72 / 21.32 / 21.37 |
| W3 / engine | 1038.28 / 1129.21 / 1134.81 | 121.93 / 172.03 / 174.34 | 121.93 / 182.16 / 190.80 | 16.52 / 22.92 / 23.25 |
| W4 / HF | 108201.82 / 184760.44 / 185035.97 | 162.69 / 185.42 / 199.48 | 162.69 / 197.72 / 215.71 | 128.86 / 204.86 / 205.16 |
| W4 / engine | 128120.96 / 228911.74 / 232203.32 | 228.93 / 241.58 / 242.65 | 228.93 / 449.12 / 640.13 | 157.19 / 258.43 / 262.65 |
| W5 / HF | 948.24 / 949.16 / 949.17 | 154.02 / 157.16 / 157.16 | 154.02 / 164.57 / 181.10 | 79.65 / 81.26 / 81.26 |
| W5 / engine | 1716.00 / 3022.56 / 3034.33 | 194.26 / 198.29 / 198.68 | 194.26 / 205.56 / 256.37 | 100.98 / 103.11 / 103.14 |
| W6 / HF | 9446.60 / 73948.53 / 74441.99 | 157.54 / 158.20 / 176.94 | 157.82 / 162.33 / 211.88 | 81.62 / 153.98 / 164.74 |
| W6 / engine | 900.40 / 1247.62 / 1247.82 | 205.59 / 217.85 / 218.08 | 205.19 / 214.00 / 358.37 | 93.53 / 112.26 / 112.26 |
| W7 / HF | 135933.80 / 259319.34 / 335009.35 | 162.34 / 165.82 / 167.64 | 162.53 / 169.73 / 218.95 | 210.07 / 343.37 / 416.61 |
| W7 / engine | 151301.80 / 320714.48 / 333012.61 | 207.59 / 222.27 / 222.30 | 207.95 / 413.73 / 441.94 | 248.12 / 425.01 / 438.91 |

Source/protocol manifests, per-request arrivals, memory peaks, fallback counters
and the full summary are retained in
`/work/vllm-rlt-engine-20261117/unified-fairness-20261117`. The initial pre-fix
campaign is retained separately as an aborted defect diagnostic and contributes
no selected performance runs above.

## 8. Profile-guided high-batch scheduling

The runtime follow-up is two scheduler-accounting lines: a full-depth coalesced
PRELUDE has performed recurrent work and must reset the prefill fairness counter.
Embedding-only PRELUDE retains the previous counter. The new regression fails
on `394f3b1` (one failure, two controls pass) and passes on `7b4537f`.
Global defaults and the disabled CUDA/ROCm path are unchanged.

At budget/chunk 1024, quota 1 interleaves decode with P128/C32 prompt cohorts:
observed KV lengths become 130–133. Nonuniform ranges export owned KV to paged
storage and force Python layer execution. The existing prefill quota 32 preserves
a uniform resident cohort and the native eager host path. Higher quotas can
postpone decode for existing requests; the global default remains one.

Independent P128/C32 ops-only/full profiles cover one resident UT4 decode
(96 layer calls), without model hooks. Full enables shapes, memory, stacks and
CPU/NPU activities. Instrumented durations are not throughput measurements.

| Decode path | Python calls | KV index_select | contiguous | linear |
| --- | ---: | ---: | ---: | ---: |
| Quota1 / paged Python fallback | 74323 | 192 | 192 | 677 |
| Quota32 / owned KV + native host | 31447 | 0 | 4 | 677 |
| HF eager | 26754 | 0 | 0 | 677 |

The configuration recovers the owned-KV/native-host path; this is not an isolated
speedup from the two accounting lines. No fused model operators or graph replay
are introduced. An HF ops-only export lacking hardware events is retained beside
its successful post-timing recapture. These are decode/C32 profiles; a fresh
full-request C1 before/after timeline was not collected for this increment.

### Full unified W1-W7 validation

**The complete matrix has not passed equal-output acceptance.** W4/W6/W7 have
unresolved sequence differences, so their throughput figures remain diagnostic.

All seven workloads were rerun on the same final configuration from `7b4537f`:
budget 8192, chunk 1024, prefill quota 32, async/refill, owned contiguous KV, native
eager host dispatch and fixed-depth prelude/decode coalescing. No per-case tuning
or older C32/C64 result is substituted into this matrix.

One 910B3 per arm, CANN 9, Torch/Torch-NPU 2.10, Transformers 4.54.1, Ouro-1.4B BF16,
UT4 greedy. Graphs, multi-stream, UVA, padding and static buffers are off. Both
arms set TASK_QUEUE_ENABLE=2 and Torch/OMP/MKL threads to one. Each case uses the
same card/CPU affinity with native/HF, HF/native, native/HF order.

All **42** new selected runs complete with **1422 successful requests and zero
runtime failures**. Every process warms a complete output cohort; loading,
compilation and warmup are excluded. Each run admits closed-loop traffic for
60 seconds and fully drains. Output throughput includes drain; values are
median [min–max] tok/s. These are in-process actor timings, not HTTP/IPC latency
or a saturation sweep. This covers the guide's applicable single-stage W1–W7
HF comparison, not its historical B300/FA4/graphs/PD/speculation mode matrix.

W1–W5 ignore EOS. W6/W7 retain the same frozen seed-0 selection of 256 ShareGPT
first-human messages, without truncation or a chat template, using natural EOS
and cap 512. The 64GiB capacity adaptation is identical in both arms: W4 has 32
clients/four residents; W6/W7 have 8/32 clients with at most 8 residents.

| Workload | Input / output / clients | Residents | HF eager tok/s | Engine tok/s | vs HF | Complete output matches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W1 | 128 / 128 / 1 | 1 | 6.334 [5.284–6.496] | 9.464 [9.403–10.394] | +49.42% | 15/15 |
| W2 | 128 / 128 / 32 | 32 | 201.314 [200.099–204.437] | 305.496 [293.376–316.662] | +51.75% | 480/480 |
| W3 | 4096 / 128 / 1 | 1 | 6.254 [6.206–6.268] | 7.569 [6.110–9.215] | +21.03% | 12/12 |
| W4 | 4096 / 128 / 32 | 4 | 21.211 [21.012–21.580] | 31.312 [31.135–31.405] | +47.62% | 120/132 |
| W5 | 1024 / 512 / 8 | 8 | 50.885 [50.234–51.362] | 79.791 [76.843–80.666] | +56.80% | 48/48 |
| W6 | ShareGPT / EOS cap 512 / 8 | 8 | 26.343 [26.060–26.385] | 37.992 [37.906–38.879] | +44.22% | 18/27 |
| W7 | ShareGPT / EOS cap 512 / 32 | 8 | 36.341 [35.576–36.931] | 36.194 [35.808–36.679] | -0.41% | 45/99 |

Complete tokens, finish reasons and depth 4 are checked, including supplemental
independent HF references after timing; no engine output remains unverified.
W1/W2/W3/W5 pass these covered-input checks. W4/W6/W7 do not. W7 ranges overlap
and do not establish a speedup; one W3 pair is slower than HF. This is not a
fresh GSM8K task-accuracy qualification or a universal equal-quality speedup.

W1–W5 retain native execution and owned KV without timed-window fallback.
W6/W7 materialize paged KV during warmup and have zero native layer calls
in the timed windows. Their enabled flag does not imply fast-path coverage.

Latency statistics pool successful requests across repetitions; values below
are mean / p95 / p99. TTFT and E2E include queuing, TPOT is first-to-last output
/(N−1), and ITL uses adjacent delivery intervals with zero intra-chunk gaps.

| Workload / arm | TTFT (ms) | TPOT (ms) |
| --- | ---: | ---: |
| W2 / HF | 314.90 / 315.98 / 316.09 | 157.04 / 160.28 / 160.28 |
| W2 / engine | 386.18 / 616.33 / 617.22 | 102.73 / 112.46 / 112.46 |
| W7 / HF | 132566.71 / 255707.79 / 326208.73 | 158.90 / 163.43 / 164.37 |
| W7 / engine | 149321.74 / 318700.31 / 321899.41 | 205.49 / 209.90 / 210.49 |

The W2 throughput gain trades off first-output latency. At W7, mean E2E is
245.42/205.07 seconds and peak allocated memory is 30.056/19.536GiB (engine/HF).
Allocator peaks reset after warmup; these are not startup peaks or total HBM.

### Earlier full-cohort C32/C64 check

The separate, previously declared P128/O128 preset used the same 8192/1024/quota 32
settings. Its 12 selected runs recorded +54.60% at C32 and +45.36% at C64, with
1440/1440 complete engine outputs matching independent HF references. Those
runs remain separate and contribute no selected result to the 42-run table.

For the earlier input44/token0 divergence, Python/native logits match HF batch8
exactly; full-cohort engine prefill matches HF batch32 logits and head inputs
with max_abs=0. This bounded batch-geometry finding does not explain the
unchecked W4/W6/W7 differences. They remain unresolved.

The runtime passed 1124 CPU tests (802 skipped), 54 targeted CPU/NPU tests and
full-tree Ruff checks. The native option defaults off; this increment does not
claim a fresh CUDA/ROCm hardware run. Earlier CUDA validation retains its
original scope.

The full report retains all latency distributions, natural length/stop counts,
memory and native/fallback counters, exact commands and source/input hashes in
`/work/vllm-rlt-engine-20261117/unified-final-8192-20261117`. Prior captures and
bounded logits diagnostics remain in the separate `high-batch-20261117` bundle.
