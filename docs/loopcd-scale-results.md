# Audited Ouro LoopCD results

The tested source is `e8aa1ee57734bc5c1118c143ebdb68d0b53210a1`, a direct
child of the original Task 0 source `d700ecca`. That increment adds the
mixed-depth continuation oracle, scale-validation entry point and documentation;
it does not change the production readout, recurrent update or feature guards.
The checkpoint is `ByteDance/Ouro-1.4B` at
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

## Public loading and full-model correctness

The public model entry passed the frozen tokenizer and parameter mapping checks
for FP32 and BF16: 52 tokenizer records, 269 parameter mappings and four facade
cases per dtype. Loading success is separate from mixed-depth numerical
qualification.

Full-model FP32 mixed-depth qualification passed with both separately qualified
Torch and Triton attention paths. The original Triton protocol includes 12
cases, 108 readouts, 96 forced continuation inputs, all four historical KV
planes and 21 ownership/lifecycle checks at unchanged `atol=rtol=1e-4`.
The same original FP32/Triton protocol subsequently passed on one H100, with
independent source/raw evidence review and selected-logprob scalar recomputation.

The full-model BF16/Triton mixed-depth gate failed at its first P4 prefill
readout: 998/2048 hidden-state elements were outside the frozen tolerance,
with maximum absolute discrepancy 0.23046875. Subsequent checks were not run.
This failure is retained; neither loading checks nor tiny-model tests qualify
the failed full-model BF16 path. The earlier BF16 `linear_fused` rejection also
remains in force. `two_head` stays the reference, default-off implementation.

## Completed C32 native fixed-work comparisons

All arms use FP32/eager Triton, prefill depth 4, B16, initial-total C32,
512 prompt tokens and 128 emitted tokens per request. Each profile uses its
own unchanged common KV pool. Observed resident A and active decode S are 16.
Two fresh starts reverse A/B/C and C/B/A order; each arm/process includes
five warmups and five measurements. Each complete profile contains six
processes, 60 trials, 1920 requests and 245760 output tokens. The measured
subset contains 30 trials, 960 requests and 122880 tokens.

| Profile | Policy | Mean tokens/s | TTFT mean ms | TPOT mean ms | Native request latency mean ms |
| --- | --- | ---: | ---: | ---: | ---: |
| H20 | D4, true off | 144.177 | 11945.174 | 74.588 | 21417.862 |
| H20 | D3, true off | 167.121 | 10934.280 | 59.380 | 18475.506 |
| H20 | D3, two_head/ref1/strength0.3 | 164.701 | 11076.655 | 60.521 | 18762.769 |
| One H100 | D4, true off | 179.28 | 9133.83 | 63.18 | 17157.87 |
| One H100 | D3, true off | 212.99 | 8191.65 | 49.25 | 14446.54 |
| One H100 | D3, two_head/ref1/strength0.3 | 207.10 | 8371.48 | 51.02 | 14851.28 |

The H20 comparison uses the mean of ten matched elapsed-time ratios across
the two fresh starts: D3-off/D4 is 1.159342x and D3-guided/D4 is 1.142596x.
Guidance reduces paired throughput by 1.436% relative to the same-depth
D3-off control. The H100 comparison uses ratios of mean TPS: D3-off/D4
is +18.80%, D3-guided/D4 is +15.52%; guided fresh-process gains are
15.39%-15.64%. These statistics are different estimators and the two-start
range is not a confidence interval. Device profiles are not pooled.

Request IDs, emitted tokens/depths, finish state, retained references, drain,
native timestamps and KV accounting passed the original immutable raw auditor.
The complete results were independently verified off device. Guidance adds
128 KiB peak allocated reference state and 2 MiB peak reserved memory in these
common-pool comparisons; there is no measured VRAM saving.

## Quality and remaining scope

The separate FP32 development screen generated and scored all 128 questions
under six policies. D4-off and selected D3-guided both scored 80/128, but paired
uncertainty did not establish the frozen one-percentage-point non-inferiority
criterion. Natural-stop D3-guided timing retained two output-limit hits and
was slower than D4 in that screen; fixed-length cost cannot remove those tails.

The independent H100 confirmation retains all 505 GSM8K and 148 HumanEval+
prompt IDs and three arms. It completed 1827/1959 outputs, with 132 D3-off
code outputs not run within the original budget. D4-off mathematics scored
316/505 and D3-guided 301/505: -2.970 percentage points, frozen paired M2
interval [-9.009, +3.129]. Code scored 1/148 versus 0/148 under the unchanged
cooperative scorer. Both primary quality comparisons remain uncertain;
the incomplete matched-off code subset is not scored as a full arm.

Completed C32 results qualify this instrumented native fixed-work scope.
They do not establish full-model C64/C128, steady 4C refill, HTTP serving,
engine-async, new CUDA Graph performance, training convergence or quality
non-inferiority. Tiny BF16 eager/Graph C32/C64/C128 checks are retained as
separate tiny-model qualification. The broader Task 1-4 matrix remains open.

Downstream preemption, async and P/D work should retain the ownership and
numerical gates before enabling each feature. Complete the frozen confirmation
and qualify feature-specific full-model parity before extending the final
concurrency/context campaign. Existing successful and failed results remain
separate from any new candidate or runtime.

[Machine-readable scientific summary](../benchmarks/results/loopcd-c32/summary.json)
and [scale-validation protocol](loopcd-scale.md).
