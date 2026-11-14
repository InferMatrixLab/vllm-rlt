# Ouro on Ascend

Run Ouro on one Ascend NPU with BF16, synchronous greedy decoding and four
recurrent loops. Use the explicit `ascend_eager` attention backend.

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

Async scheduling, graphs, speculation, PD, preemption, prefix caching and
adaptive exit are currently unsupported on this path.

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
