# Synchronous self-speculation evaluation on H800

This report evaluates the implementation in [PR #44](https://github.com/InferMatrixLab/vllm-rlt/pull/44)
against native synchronous Ouro decoding. It is the first measurement deliverable
for the [accuracy and performance work in RFC #43](https://github.com/InferMatrixLab/vllm-rlt/issues/43).
The comparison isolates speculation within the same synchronous eager execution
mode. It does not compare to asynchronous scheduling or CUDA Graphs.

## Coverage relative to PR #44

PR #44 already tests greedy output, fixed-chain reuse on a tiny model,
rejection/rollback positions, EOS and length limits, RNG isolation, and GPU
smoke cases. This evaluation adds a pinned real checkpoint, a repeated matched
decode matrix, real-model hidden/KV/logit measurements, filtered-distribution
sampling checks, and HTTP measurements that account for failed streams and
transport chunking. The experiments did not change production decoding.

## Pinned setup

| Item | Value |
| --- | --- |
| Implementation and native baseline | PR #44 head `d9fca507e766e81f5d89f90d598881c12d7d8397`; native mode omits `SpeculativeConfig` |
| Checkpoint and tokenizer | `ByteDance/Ouro-1.4B` revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| GPU | NVIDIA H800 PCIe 80 GB, SM 9.0; one exclusive device |
| Driver / CUDA / Torch | 595.71.05 / 12.8 / 2.9.1+cu128 |
| Python / Transformers / Triton / aiohttp | 3.10.8 / 4.57.3 / 3.5.1 / 3.14.3 |
| Attention / precision | FA4 4.0.0b30, BF16; CUDA matmul TF32 disabled |
| Execution | synchronous eager, `last_exited` KV, `d=2`, `D=4`, refill scheduler, greedy, `ignore_eos=True` |

The `code_sha` in the archived measurements pins the **implementation
baseline** in PR #44. The [experiment snapshot](https://github.com/Levius-Fubuki/vllm-rlt/tree/5040d2b209942ed6535a5cfe28cd09b465bde36f)
retains the benchmark code, tests and raw measurements for reproduction and
audit.

## Method

The archived offline benchmark, `python -m benchmarks.speculative`, uses eight
distinct, semantic questions with a fixed context, clipped at the *front* to
make exact 32- and 128-token prompts while retaining each question. Repeating a
short prompt to fill a context caused Ouro to echo that prompt and accept every
draft; those synthetic echo results are excluded from the primary comparison.

For each prompt length (32, 128), requested output length (32, 128), concurrency
(1, 8), and K (1, 2, 4, 8), the script warms both paths and runs five paired
trials with alternating execution order. Both modes share one resident BF16
model, scheduler settings, KV capacity, prompt IDs, greedy parameters and
output length. A CUDA synchronization follows once all requests have produced
their first token; another follows the last completion. Decode time covers every engine step
in between, including draft, final shallow fill, verification, sampling,
host synchronization, scheduler work, commit and rollback. Reported throughput
is `(output_length - 1) × concurrency / decode_seconds`; the initial token and
matched prefill are excluded. Each paired run stores full output token IDs,
timings, draft/accept/verification counts and exact agreement.
The five repeats estimate timing variability; because the same deterministic
inputs are reused, they are not independent accuracy samples.

The archived `python -m benchmarks.speculative_accuracy` replays a fixed
candidate chain through the reuse verifier and a serial full-depth oracle on
the same BF16 model. It compares hidden states and KV at every depth and final logits, using
FP32 differences for max absolute and RMS error. The unit tests also exercise
greedy, rejected suffix rollback, EOS/length boundaries and the rejection
sampler, including temperature/top-k/top-p filters.

The archived HTTP client, `python -m benchmarks.speculative_serving`, sends the
same fixed-length prompts and output limit to separately started native and
speculative servers. It counts only completed `[DONE]` streams with usage
matching the requested tokens; native streams additionally require one choice
event per token. PR #44 intentionally emits one choice event per speculative
round, which may contain several tokens. TTFT starts at the actual HTTP send
after the client's concurrency semaphore. E2E ends at the last choice event.
The client records inter-choice-event intervals and choice events per transport
chunk. Native intervals are token ITL; speculative intervals are round-level
and must not be compared as token ITL. Usage divided by choice events gives
the mean tokens per event, but the current HTTP payload does not expose each
event's exact token count.

## Reproduction

Check out the [experiment snapshot](https://github.com/Levius-Fubuki/vllm-rlt/tree/5040d2b209942ed6535a5cfe28cd09b465bde36f),
install its development, serving and FA4 dependencies, and obtain the pinned
model on a compatible Hopper GPU. Run the following commands from that
snapshot:

> **Snapshot required:** The `benchmarks.speculative*` modules below exist only
> in the pinned fork snapshot, not in this repository's current tree. Run these
> reproduction commands from that snapshot.

```bash
python -m benchmarks.speculative \
  --model /path/to/Ouro-1.4B \
  --model-revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --output artifacts/spec-eval/decode.json \
  --prompt-lengths 32 128 --output-lengths 32 128 \
  --concurrencies 1 8 --ks 1 2 4 8 --trials 5

python -m benchmarks.speculative_accuracy \
  --model /path/to/Ouro-1.4B \
  --model-revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --output artifacts/spec-eval/accuracy.json --prefix-tokens 8 --ks 1 2 4 8

python -m benchmarks.speculative_sampling \
  --output artifacts/spec-eval/sampling.json --trials 6000 --seeds 431 982 2026

python -m benchmarks.speculative_profile \
  --model /path/to/Ouro-1.4B \
  --model-revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --output artifacts/spec-eval/profile.json \
  --prompt-tokens 128 --output-tokens 64 --concurrency 1 --k 4
```

The archived benchmark scripts also accept `--help` for all options. Exact
server settings and aggregate results are recorded below.

## Results

### Fixed-chain and free-generation accuracy

On the pinned real model, fixed-chain reuse and serial full-depth execution
matched **bit for bit** for the tested 8-token prefix and K=1/2/4/8: max and
RMS errors were zero for every depth's hidden state and KV, and for final
logits. There were zero argmax differences across the 2/3/5/9 compared
positions, respectively. This is evidence for those fixed chains, not a BF16
error bound over arbitrary prefixes.

In free generation, 15 of 32 workload configurations had exact agreement for
every request. Every configuration produced the same match pattern across its
five deterministic repeats. For example, with 32-token input, 32-token output,
concurrency 8 and K=1, request 2 first diverged at zero-based output index 23
(native ID 7071, speculative ID 314). A fixed-chain replay at that prefix had
zero hidden/KV/logit error between reuse and a serial oracle, whose next-token
top two were tied. Native concurrent replay selected ID 7071 with a 0.125
top-1/top-2 margin. For the 32/128/concurrency-8/K=1 case, five requests first
diverged at indices 23, 66, 78, 81 and 97. All recorded native prefixes
matched on replay; at two of the later positions, the replayed top two tied
and the replayed top-1 differed from the original run. These observations
indicate shape-sensitive BF16 behavior and ties are relevant. They do not
establish exact greedy equivalence, nor rule out every state bug. The archived
token IDs and [short replay](https://github.com/Levius-Fubuki/vllm-rlt/blob/5040d2b209942ed6535a5cfe28cd09b465bde36f/docs/results/speculative-h800/replay-short.json)
and [long replay](https://github.com/Levius-Fubuki/vllm-rlt/blob/5040d2b209942ed6535a5cfe28cd09b465bde36f/docs/results/speculative-h800/replay-long.json)
diagnostics permit further investigation.

The filtered rejection sampler was checked for three temperature/top-k/top-p
settings with three independent seeds and 6,000 draws per setting and seed.
Across these nine experiments, the largest absolute empirical deviation from
the target probability was **0.0119** and the largest standardized deviation
was **1.98**. These small-vocabulary checks do not validate real-model
sampling quality over full generated sequences.

### Decode throughput

Each row reports median native and speculative committed decode tokens/s,
the median of five *paired* time ratios with the observed interquartile range,
draft acceptance (`accepted_tokens / drafted_tokens`), and exact matching
requests in the first trial. Match counts were stable across repeats.

| Input | Output | Concurrency | K | Native tok/s | Spec tok/s | Paired speedup [Q1, Q3] | Accept | Exact requests |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 1 | 1 | 13.3 | 16.6 | 1.26 [1.25, 1.26] | 87.5% | 1/1 |
| 32 | 32 | 1 | 2 | 13.0 | 18.1 | 1.39 [1.37, 1.40] | 86.4% | 1/1 |
| 32 | 32 | 1 | 4 | 13.3 | 17.4 | 1.28 [1.28, 1.28] | 71.0% | 1/1 |
| 32 | 32 | 1 | 8 | 13.2 | 16.0 | 1.22 [1.21, 1.22] | 62.5% | 1/1 |
| 32 | 32 | 8 | 1 | 101.0 | 127.0 | 1.26 [1.25, 1.26] | 93.7% | 7/8 |
| 32 | 32 | 8 | 2 | 103.0 | 125.9 | 1.22 [1.22, 1.24] | 89.7% | 7/8 |
| 32 | 32 | 8 | 4 | 100.9 | 126.3 | 1.25 [1.24, 1.25] | 83.8% | 7/8 |
| 32 | 32 | 8 | 8 | 100.1 | 116.9 | 1.17 [1.17, 1.17] | 82.1% | 7/8 |
| 32 | 128 | 1 | 1 | 13.4 | 17.5 | 1.30 [1.29, 1.31] | 92.4% | 1/1 |
| 32 | 128 | 1 | 2 | 13.5 | 19.4 | 1.44 [1.44, 1.44] | 92.1% | 1/1 |
| 32 | 128 | 1 | 4 | 13.5 | 19.3 | 1.43 [1.43, 1.44] | 81.5% | 1/1 |
| 32 | 128 | 1 | 8 | 13.4 | 18.0 | 1.36 [1.33, 1.37] | 70.9% | 0/1 |
| 32 | 128 | 8 | 1 | 103.2 | 123.3 | 1.19 [1.19, 1.20] | 89.2% | 3/8 |
| 32 | 128 | 8 | 2 | 104.4 | 128.5 | 1.24 [1.20, 1.25] | 84.2% | 3/8 |
| 32 | 128 | 8 | 4 | 103.4 | 115.9 | 1.13 [1.12, 1.13] | 72.5% | 3/8 |
| 32 | 128 | 8 | 8 | 103.0 | 97.8 | 0.95 [0.95, 0.95] | 64.9% | 2/8 |
| 128 | 32 | 1 | 1 | 13.6 | 17.8 | 1.32 [1.31, 1.33] | 93.8% | 1/1 |
| 128 | 32 | 1 | 2 | 13.5 | 18.1 | 1.34 [1.34, 1.34] | 82.6% | 1/1 |
| 128 | 32 | 1 | 4 | 13.4 | 17.6 | 1.31 [1.31, 1.34] | 71.9% | 1/1 |
| 128 | 32 | 1 | 8 | 13.5 | 16.0 | 1.19 [1.18, 1.19] | 61.0% | 1/1 |
| 128 | 32 | 8 | 1 | 102.1 | 126.3 | 1.23 [1.22, 1.24] | 93.7% | 6/8 |
| 128 | 32 | 8 | 2 | 102.9 | 129.5 | 1.26 [1.25, 1.26] | 87.0% | 6/8 |
| 128 | 32 | 8 | 4 | 103.2 | 108.9 | 1.06 [1.05, 1.06] | 72.1% | 7/8 |
| 128 | 32 | 8 | 8 | 103.9 | 89.8 | 0.86 [0.86, 0.87] | 57.8% | 5/8 |
| 128 | 128 | 1 | 1 | 13.5 | 18.0 | 1.34 [1.33, 1.34] | 98.4% | 1/1 |
| 128 | 128 | 1 | 2 | 13.6 | 20.2 | 1.48 [1.48, 1.49] | 95.4% | 1/1 |
| 128 | 128 | 1 | 4 | 13.7 | 21.4 | 1.57 [1.55, 1.57] | 91.7% | 1/1 |
| 128 | 128 | 1 | 8 | 13.5 | 22.0 | 1.62 [1.61, 1.64] | 87.4% | 1/1 |
| 128 | 128 | 8 | 1 | 104.1 | 129.6 | 1.25 [1.24, 1.26] | 93.5% | 5/8 |
| 128 | 128 | 8 | 2 | 104.1 | 141.2 | 1.36 [1.36, 1.36] | 91.1% | 5/8 |
| 128 | 128 | 8 | 4 | 103.9 | 139.6 | 1.34 [1.34, 1.34] | 84.5% | 5/8 |
| 128 | 128 | 8 | 8 | 103.6 | 115.2 | 1.11 [1.10, 1.11] | 71.5% | 4/8 |

K=8 helped most for the 128/128 single-request workload (1.62×), but
regressed at 128/32, concurrency 8 (0.86×). In that regression, only 57.8%
of drafted candidates were accepted. K therefore needs workload-specific
selection; the sweep does not establish a universal default.

### Memory and phase costs

With one engine resident at a time, the 128/128, concurrency-8 decode peak
allocation exceeded the same-capacity native path by **1.6 MB at K=1**, **4.5 MB
at K=4**, and **7.6 MB at K=8**. The model and fixed KV cache dominate these
process-level PyTorch allocation peaks; CUDA driver and external library
allocations are outside this metric. The K=1 model baseline changed after the
first warmup, so small differences should be treated as indicative.

Out-of-band instrumentation of 128-token input, 64-token output and K=4
recorded the following sums of CPU wall time spent in each phase. Nested
execute time and CUDA event spans overlap these phase sums and must not be
added to them. The hooks were absent from the throughput sweep.

| Concurrency | Draft core | Final shallow core | Target core | Commit |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,176 ms | 571 ms | 580 ms | 1.7 ms |
| 8 | 2,470 ms | 664 ms | 675 ms | 10.9 ms |

### Serving

The two servers used the same model/tokenizer, BF16/FA4, 640 KV blocks,
`max_num_seqs=8`, `max_num_batched_tokens=1024`, and 128-token prefill chunks.
The speculative server alone added `--speculative-tokens 4 --draft-loops 2
--target-loops 4`. Each request used a 128-token prompt, a 64-token output
limit, greedy fixed D=4 and `ignore_eos=True`. The serial workload sent eight
requests at max concurrency 1; burst and finite-rate workloads sent 16 at max
concurrency 8. Finite-rate arrivals were scheduled at two requests per second.
Server startup and one client warmup were excluded. Burst values are medians of
three independent workload runs; serial and finite-rate values are one run
each. Every measured request completed with `[DONE]` and correct usage.

| Workload | Mode | Complete tok/s | TTFT p50 / p95 ms | E2E p50 / p95 ms | Tokens / choice event | Exact text |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Serial | Native | 12.9 | 95 / 98 | 4,931 / 4,994 | 1.00 | — |
| Serial | K=4 | 17.8 | 95 / 96 | 3,642 / 4,016 | 3.76 | 6/8 |
| Burst | Native | 94.3 | 316 / 343 | 5,414 / 5,489 | 1.00 | — |
| Burst | K=4 | 111.8 | 489 / 524 | 4,315 / 4,779 | 3.85 | 10/16 |
| 2 req/s | Native | 70.2 | 112 / 116 | 5,808 / 5,902 | 1.00 | — |
| 2 req/s | K=4 | 83.5 | 317 / 437 | 4,610 / 5,005 | 3.74 | 10/16 |

The three burst throughput runs spanned 90.8–95.1 tok/s natively and
110.5–113.9 tok/s with speculation. K=4 improved completion throughput and
E2E latency in these workloads, while burst and finite-rate TTFT worsened.
The client records choice-event intervals, but speculative events represent
rounds, so their intervals are **not** token ITL. Loopback transport delivered
one choice event per HTTP chunk in the burst runs. A separate natural-EOS
attempt finished at the 64-token length limit for every request in both modes;
it did not test early termination behavior.

To reproduce from the experiment snapshot, start
`python -m vllm_rlt.entrypoints.serve` with the shared settings above, first
without speculative flags and then with them. For each
server, run `python -m benchmarks.speculative_serving` with `--mode native` or
`--mode speculative`, `--base-url http://127.0.0.1:8001`,
`--tokenizer /path/to/Ouro-1.4B`, the pinned `--model-revision`, and the
workload-specific `--num-requests`, `--max-concurrency`, and `--request-rate`
settings stated above. Use `--prompt-tokens 128 --output-tokens 64` and a
distinct `--output` file for every run.

The server and client commands below also target the pinned fork snapshot;
`benchmarks.speculative_serving` is unavailable in the current repository tree.

The native server command used for these runs was:

```bash
python -m vllm_rlt.entrypoints.serve \
  --model /path/to/Ouro-1.4B --tokenizer /path/to/Ouro-1.4B \
  --served-model-name ouro --device cuda --dtype bfloat16 \
  --attention-backend flash_attn_4 --num-blocks 640 \
  --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --prefill-chunk-size 128 --host 127.0.0.1 --port 8001
```

For the speculative server, append `--speculative-tokens 4 --draft-loops 2
--target-loops 4`. Run the servers separately. For example, the burst client
command is:

```bash
python -m benchmarks.speculative_serving \
  --base-url http://127.0.0.1:8001 --served-model ouro \
  --tokenizer /path/to/Ouro-1.4B \
  --model-revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --mode native --output artifacts/spec-eval/serve-native-burst.json \
  --num-requests 16 --max-concurrency 8 --request-rate inf \
  --prompt-tokens 128 --output-tokens 64
```

### Archived measurements

The [pinned results directory](https://github.com/Levius-Fubuki/vllm-rlt/tree/5040d2b209942ed6535a5cfe28cd09b465bde36f/docs/results/speculative-h800)
contains all 160 paired decode trials, complete output token IDs, fixed-chain
accuracy, sampling, memory, phase timings, divergent-prefix replays and ten
HTTP serving runs. The decode and serving records are gzip-compressed JSON.

## Validation

On the pinned H800 environment, using the archived benchmark scripts and tests:

```text
pytest -q tests/test_speculative_accuracy.py tests/test_speculative_benchmark.py \
  tests/test_speculative_profile.py tests/test_speculative_replay.py \
  tests/test_speculative_serving_benchmark.py tests/test_speculative.py --run-gpu
71 passed, 1 skipped

pytest -q tests/test_serving.py
47 passed

ruff check <all changed Python files>
All checks passed
ruff format --check <all changed Python files>
11 files already formatted
```

The saved files passed JSON/gzip parsing and link checks; all 32 decode cases
contain five paired trials with complete output lengths, and all ten serving
runs report zero failed requests.

## Scope and limits

The asynchronous speculative implementation is separate and is not evaluated
here. The RFC's BF16 acceptance tolerance remains a maintainer decision, so
numeric errors and exact greedy agreement are reported without declaring a
general losslessness threshold. No Nsight Compute/Systems executables were
available in this container; MFU and memory
bandwidth are not estimated from throughput alone. Serving results use fixed
output length and `ignore_eos=True`; the natural-EOS attempt did not encounter
an early EOS, so that behavior remains unmeasured with this checkpoint/workload.
