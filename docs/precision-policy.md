# BF16 inference and numerical validation policy

BF16 weights, ordinary activations and KV storage are the primary inference
target. Full-model FP32 is an optional diagnostic/reference configuration.
Historical results retain their original dtype, bounds and limitations.

## Inference precision

Follow the pinned Ouro release, distinguishing storage/output dtype from
accumulation precision. The [source notes](https://github.com/hsliuustc0106/vllm-rlt/blob/0f091a985e13a207f4fa127a08d51a625d130ef5/docs/paper-notes.md#original-ouro-paper-and-precision-guidance)
identify author guidance and project differences.

| Operation | Precision requirement |
| --- | --- |
| Projections, MLP and LM head | BF16 operands and outputs; record the backend's accumulation/reduction mode. FP32 accumulation is compatible with BF16 inference. |
| Attention | BF16 Q/K/V and output. The official eager path uses native-dtype QK/PV products and FP32 softmax cast back to query dtype; fused-kernel FP32 reductions/statistics/accumulation are explicit backend choices to validate. |
| RMSNorm | Retain FP32 variance/reduction arithmetic, returning to the activation dtype at the defined boundary. |
| RoPE | Retain FP32 frequencies, phase and trigonometric computation; cast the positional factors to the activation dtype as specified by the model. |
| Gate and sampling probabilities | Use the pinned release as the reference. Its exit-distribution code has no explicit FP32 promotion; the current runner's FP32 sigmoid/host cumulative update is a project difference to validate. The sampling implementation is recorded in [`worker/sampler.py`](../vllm_rlt/worker/sampler.py): the non-greedy branch promotes logits to FP32 and processes them as `float()/temperature -> top-k -> top-p -> multinomial`, where top-k keeps `min(k, vocab_size)` candidates and top-p may trim that set further. Sampling itself adds no explicit host readback; `ModelRunner.submit()` stages CUDA results into pinned host buffers; `Submission.collect()` waits for completion and converts the result to a host list. |
| Cache metadata and KV copies | Integer addresses/lengths; copy BF16 KV without changing its values. |

## Validation requirements

Keep cache ownership, initialized history, address mapping, KV copies and RNG
ownership exact. Metadata-only changes with identical arithmetic should preserve
selected tensors, tokens and exit decisions exactly.

For changed arithmetic, compare independent references on matched inputs and
histories with justified numerical bounds. Record rounding/accumulation choices,
token margins and gate disagreements; task-quality claims need task evaluation.
Passing FP32 or matching a few generated tokens does not qualify BF16 generally.

## Measurements

Match dtype, hardware, workload and timed work across performance comparisons.
Report preparation separately, retain raw observations and variability, and keep
profiling outside throughput timing. Frozen milestone-specific budgets and
thresholds belong to their [historical records](https://github.com/hsliuustc0106/vllm-rlt/releases/tag/implementation-notes-archive-20260913).
