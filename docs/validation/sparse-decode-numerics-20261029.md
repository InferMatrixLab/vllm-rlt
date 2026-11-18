# Sparse decode: backend numerical diagnosis

This follow-up fixes the full-retention discrepancy in the
[ordinary/synchronous integration report](sparse-decode-sync-20261029.md), within
the same Draft collaboration PR targeting `kunkun/rfc156-sparse-decode-async`.
The immediate feature base remains `048b9d7c79a5711857ce3287812502ad8f274ad0`;
the pre-fix contribution head is `7f22af3901416096ea6c1c0e0a5536f440f662f5`.
The fixed runtime/test commit is `1bbe6fdd24be3e9bbea6a9dd1440d4b64f8dd89b`.

## Reproduced cause

The original BF16 Ouro-1.4B workload reproduces 498/512 token agreement at
concurrent arrivals 32 and synchronous draft depth 1. Request 9 first differs
at output index 2: dense chooses 293, the original sparse reference chooses 92.
The 14 differences include subsequent autoregressive continuation, rather than
14 independent initial errors.

The new diagnostic retains the original 128-input/16-output protocol, including
four-token warmup. It freezes the complete accepted verification batch's input
hidden states, logical row/depth/position metadata, position IDs and context
lengths, all physical-layer module outputs, full packed queries and attention
outputs, and the watched row's causal K/V. It filters out rejected executions of
the same position. The accepted loop-2 batch here has **22 rows**, not the
theoretical 96-row maximum; both arms have exactly equal packed input states,
request histories, shapes, positions and lengths. A shorter-output exploratory
run changed the batch geometry and is retained separately, not substituted for
this reproduction.

For the watched row, physical layers 0 and 1 agree. At physical layer **2**
(zero-based), Q, all 130 legal keys and values are identical, but the attention
output differs at one BF16 element by **0.000030517578125**. The next `o_proj`
differs at two elements, max **0.000244140625**. Across the complete batch,
attention differences already exist at physical layer 0. Later layers and
recurrences amplify these differences. This places the first observed watched
operator discrepancy at attention, before reference consumption.

A separate replay uses the frozen operands with no model or scheduler. Triton
split-KV reproduces the captured dense output exactly; the Torch
einsum/softmax reference reproduces the captured sparse output exactly. Changing
only the backend reproduces the layer-2 difference. Disabling BF16 reduced
precision reduction does not change that reference result; its reductions
already use FP32 operands. TF32 is disabled throughout. Equal packed shapes
rule out a batch-shape change as the explanation for this captured attention
discrepancy. No global precision flags or model GEMM implementations are changed.

## Fix and numerical contract

`KVCacheManager` passes its selected dense attention adapter into the shared
`sparse_paged_attention` dispatcher. The adapter receives the original packed
queries, tables and lengths, without slicing into different batch shapes.

- DENSE and CAPTURE rows use that backend's actual output. Capture stores that
  output as FP32 **after** the backend's output-dtype rounding, so `O_ref` is
  the value entering recurrence. Retained selection, mass and contribution
  remain the FP32 Torch reference arithmetic.
- At retention 1, CONSUME rows also use the configured dense backend, while
  still checking reference ownership/payload availability and recording
  consumption. This is explicitly a dense control; it does not qualify the
  Torch correction as bitwise equal to Triton.
- At retention below 1, later loops retain `O_ref - C_ref + m * C_new`. This
  changes the original reference's source-output rounding contract, not the
  selection or sparse delta formula. Approximate quality remains unqualified.
- Omitting the adapter when calling the standalone paged oracle forces the
  original Torch capture and correction, including at retention 1. Known-answer
  paged cases test changed later keys/values at retention .5 and 1; the full-retention
  control cannot hide a regression in that independent arithmetic path.

A full-retention-only bypass would leave partial-retention capture rows using
a different dense algorithm; preserving the selected backend at every dense
source row fixes that inconsistency too. No new kernel or global precision
setting is introduced. The fallback oracle remains independently available.

## Regression evidence

Four CUDA BF16 cases cover MHA/GQA, retention .1/1, a mixed dense/capture plan,
stored `O_ref`, changed later-loop operands and reference consumption. All four
fail on the original implementation with backend-output differences, then pass
with the fix using exact equality. The standalone paged known-answer cases
return 23.5 and 40.0 after changing later keys/values, exercising partial and
full Torch correction without the backend control.

Full CPU regression: **921 passed, 479 skipped**. Full pre-commit passes Ruff
lint, Ruff format and the CPU suite. Independent read-only review found no
actionable defect in the four-file runtime/test delta; its targeted CPU check
passed 48 cases, with 22 GPU cases deselected.

Applicable CUDA regression: **318 passed, 1 skipped, 16 deselected**. The skip
requires two GPUs; FA4 cases remain excluded on this SM89 device. A fixed
instrumented replay now matches the original dense replay exactly at every
recorded depth: complete packed inputs/outputs, all physical module outputs,
attention queries/outputs and the watched row's causal K/V. This comparison is
separate from the uninstrumented generation matrix.

The uninstrumented BF16 regression matrix covers ordinary decode and synchronous
draft depths 1/2, simultaneous arrivals 1/8/32, and retention off/1/.1. Each of
the 27 fresh-process configurations warms four output tokens, then measures
three bursts using the original 128-input/16-output protocol. All **81 bursts,
1,107 requests and 17,712 output tokens** complete, with KV/reference state
drained. Tokens and exit depths repeat exactly within each configuration.

| Mode | C=1 full vs dense | C=8 full vs dense | C=32 full vs dense |
|---|---:|---:|---:|
| Ordinary | 16/16 | 128/128 | 512/512 |
| Sync, draft 1 | 16/16 | 128/128 | 512/512 |
| Sync, draft 2 | 16/16 | 128/128 | 512/512 |

Every cell matches in all three repetitions, including exit depths: **5,904/5,904
full-retention tokens** in total. In particular, the original C32/draft-1 case
improves from 498/512 to 512/512 in each repetition. This verifies the explicit
dense-backend control; the forced Torch correction remains a separate oracle.
Retention .1 completes and drains, but its token agreement is not a quality score.

## Reproduction and archive

The [numerical validation release](https://github.com/Levius-Fubuki/vllm-rlt/releases/tag/pr157-numerics-validation-20261029)
contains the frozen tensor snapshots, source and SHA256 inventories, Git bundle,
same-input replay, original failures, red/green regression logs, full raw matrix,
environment metadata and reproduction commands. Model weights are not included.
The original unsupported-FA4 failures and performance profiles remain in the
[earlier release](https://github.com/Levius-Fubuki/vllm-rlt/releases/tag/pr157-sync-validation-20261029).

Use `trace_first_operator.py` against `7f22af3` for the dense and pre-fix capture,
then against `1bbe6fd` for the fixed capture; `replay_attention.py` tests the
frozen operands independently. `compare_fixed_trace.py` checks every recorded
state, and `run_regression_matrix.py` / `summarize_matrix.py` reproduce and verify
the complete generation matrix. These local experiment scripts preserve their
recorded root directory; the archive README explains relocation and exact CLI
commands. Diagnostic timings are excluded from uninstrumented results.

## Scope

The model weights match the prior SHA256
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
The follow-up server reports one RTX4090 (SM89) with 49,140 MiB, driver
595.71.05, torch 2.12.1+cu130 and Triton 3.7.1. This is a different instance
from the original 24,564-MiB server with driver 580.105.08;
timings across these instances are not a matched performance comparison.

These are numerical and execution regressions on synthetic fixed-token inputs,
not a held-out task-quality evaluation. Partial retention, adaptive exits,
long contexts, other GPUs and untested feature combinations remain outside the
qualification. Existing asynchronous lifecycle regressions are included, but
no new real-model asynchronous quality claim is made. The original performance
profiles remain historical evidence;
this correctness fix makes no serving-speed improvement claim.
