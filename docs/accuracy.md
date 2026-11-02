# Ouro GSM8K-87 accuracy regression case

The default case uses **87 fixed questions from GSM8K `main`/`test`**. Its measured
original Hugging Face Transformers baseline is **59/87 (67.82%)**. Native vllm-rlt
also scored **59/87**, with identical per-question correctness on this case.

The [checked-in fixture](../benchmarks/fixtures/gsm8k-87.json) pins the exact source
row IDs, dataset revision, HF score and recipe fingerprint. These are the 87
questions from the original seed-0 100-question case that had completed on both
backends when the case was adopted on September 12, 2026. The choice was made
after inspecting results, so this is a regression case for future changes, not
a new held-out estimate of model quality. The observed native score on the
original 100-question case was 69/100; its protocol and results are preserved.
Adopting the 87-question case required no new GPU inference.

## Protocol and baseline

- Pinned `ByteDance/Ouro-1.4B` checkpoint and official release code; BF16,
  fixed four loops, greedy decoding, one request at a time.
- lm-eval-harness 0.4.9.2's `gsm8k_cot`, first three demonstrations and strict
  answer extraction/scoring, following the Ouro paper's stated 3-shot CoT setup.
- No chat template or added special tokens. Natural EOS and identical harness
  stop sequences; at most 1,024 new tokens within a 2,048-token total context.
- Exact-answer accuracy with unparseable answers counted as incorrect. Save
  raw text, token IDs, extracted answers, correctness and stopping reason.
- Default regression threshold: native may lose at most one percentage point
  against the measured HF baseline. One question out of 87 is about 1.15 points,
  so the default gate requires **at least 59 correct**. This is a regression
  screen, not proof of population equivalence.

The reference calls `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` and the released model's `generate()`. It uses eager
attention and the standard Transformers `DynamicCache` with one slot per layer and
loop (96 here), plus `exit_at_step = total_ut_steps - 1` (3) for last-loop logits.
This cache setup accommodates the
pinned release's older cache interface; the model source is not patched.
The candidate uses the existing native engine and Triton attention. Both use
BF16 weights and activations, retaining each backend's existing FP32 reductions
and other operations that need higher precision. The evaluator changes no
inference arithmetic.

The recorded environment used torch 2.13.0, Transformers 4.55.0, lm-eval 0.4.9.2,
datasets 3.6.0, tokenizers 0.21.4 and Triton 3.7.1. The fixture records these
versions and hashes of the source HF protocol and selected raw outputs. The
baseline fingerprint covers model files, prompts/token IDs, questions, task
configuration, generation controls and package versions. Changing the checkout
SHA or checkpoint directory is allowed; changing the recipe requires a fresh
HF comparison using a custom protocol.

## Prepare and run the default case

Reuse the prepared accuracy environment when available. For a new environment,
install the project and evaluation requirements, using the package versions
recorded above to reuse the stored baseline:

```bash
pip install -e .
pip install -r benchmarks/requirements-accuracy.txt
hf download ByteDance/Ouro-1.4B \
  --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /path/to/ouro
python -m benchmarks.gsm8k prepare --model /path/to/ouro \
  --output /path/to/gsm8k-87-protocol.json
```

Preparation runs without CUDA. It verifies the local checkpoint against the
pinned Hub release, loads the pinned dataset, and freezes the questions, prompts,
token IDs, controls and package/model hashes. Oversized prompts fail preparation;
questions are never replaced. The default also verifies the recipe fingerprint
and embeds the stored HF baseline. Keep generated protocols and results outside
Git.

Before inference, declare the GPU/CPU affinity and run budget. Use two disjoint
training questions (`prepare --split train --limit 2`) for feasibility, then one
greedy pass over the 87 test questions for this regression screen. This run budget
does not measure repeat-run variability or throughput. On a host with the GPU
scheduler, use an available exact device ID and run the command below through
`gpu run --gpu-ids <available-id> --timeout 2h --note "GSM8K-87 accuracy" --`.

```bash
python -m benchmarks.gsm8k run --backend native \
  --protocol /path/to/gsm8k-87-protocol.json --output /path/to/native
```

The native summary includes `baseline_comparison`, with HF accuracy, native-minus-HF
percentage points on the fixed cohort and the gate result. The command saves the
summary and exits nonzero if the gate fails. Exceptions and non-finite logits
stop execution while preserving completed records; do not reduce the denominator.
An optional `--min-reference-accuracy-pct` floor and `--max-regression-pp` changes
must be declared during preparation. There is no default paper-score floor.

## Fresh comparisons and custom cases

To remeasure HF on the same default case, run both backends from the same protocol
and compare their saved results. Keep the exact GPU, CPU/NUMA affinity and controls
fixed across backends. Budget one feasibility pass and one measured greedy pass
per backend before starting; this budget does not establish repeat-run variability.

```bash
python -m benchmarks.gsm8k run --backend transformers \
  --protocol /path/to/gsm8k-87-protocol.json --output /path/to/transformers
python -m benchmarks.gsm8k compare --transformers /path/to/transformers \
  --native /path/to/native --output /path/to/comparison.json
```

The comparison reports correct counts, accuracies, percentage-point difference,
paired losses/gains and answer disagreements. It rejects incomplete or mismatched
runs. Loading and generation durations are recorded separately.

Explicit `--limit 100 --seed 0` recreates the original 100-question selection;
`--all` uses the complete split. Custom subsets use the SHA-256 ranking of dataset
revision, split, seed and source row ID, independent of answers. In particular,
`--limit 87` samples a different set from the fixed default case. Custom protocols
do not reuse the stored 59/87 baseline: run HF and native, then use `compare`.

## Thinking checkpoints

`--prompt-mode thinking` prepares zero-shot chat cases for the Ouro-Thinking
checkpoints: each record renders the raw question through
`vllm_rlt.entrypoints.chat_template.render_chat_prompt` with
`enable_thinking=True`, encodes with the checkpoint's normal special-token
behavior (including BOS), and stops only at EOS. The few-shot task scaffolding
and its stop strings do not apply. Scoring uses `thinking_final_v1`: the section
after the last generated `</think>` token is decoded independently. Extract the
last `\boxed{...}` occurrence if its braces balance, otherwise a terminal
`####` answer. The boxed form need not be at the end; a nonnumeric last box is
unparseable (no fallback to an earlier answer). Numeric signs, decimals,
commas, surrounding whitespace/dollar signs and a trailing period are normalized;
expressions, fractions and nonfinite values are not evaluated. Missing closing
markers and unparseable final answers count as incorrect, including on length
truncation. A parseable matching answer may count even on a length finish; length
counts remain reported separately, and no row is excluded from the denominator.
This zero-shot protocol is a project choice, not an official paper-score recipe.
Custom subsets only: the stored raw-prompt baseline never applies, so establish
a Thinking baseline by running both backends from one protocol and using `compare`.

```bash
python -m benchmarks.gsm8k prepare --model /path/to/Ouro-1.4B-Thinking \
  --model-repo ByteDance/Ouro-1.4B-Thinking --model-revision <pinned-commit> \
  --prompt-mode thinking --limit 20 --seed 1 \
  --max-new-tokens 1024 --max-length 2048 \
  --output /path/to/thinking-protocol.json
```

## Adaptive-exit runs

The default recipe runs the checkpoint's full depth for every token: `prepare` reads
`total_ut_steps` from its `config.json` (4 for Ouro-1.4B) into the protocol's `loops`,
and `run` takes the depth from there. `loops` is part of the baseline fingerprint, so the
stored GSM8K-87 baseline applies only to a four-loop protocol. `run` also accepts an
adaptive exit policy, so accuracy can be measured under the same protocol:

```bash
python -m benchmarks.gsm8k run --backend native \
  --protocol /path/to/gsm8k-87-protocol.json --output /path/to/native-ouro-t05 \
  --exit-mode ouro --exit-threshold 0.5 --min-loops 1
```

- `--exit-threshold` is the cumulative exit probability. `1` (the default) keeps
  the fixed full-depth recipe, and the loop/mode options are rejected with it.
- Below 1, `--min-loops` is required, between 1 and the protocol's depth.
  `--exit-mode ouro_delayed` selects the delayed gate, and `--async-scheduling`
  (native only) requires it.
- The Transformers release accepts `--exit-mode ouro --min-loops 1` only: it has no
  minimum loop count or delayed mode. It runs every loop and selects the exited
  loop's hidden state, so its KV stays full-depth, while native execution skips the
  remaining loops and uses the configured KV layout. The release would also apply the
  threshold to the prefill forward; the harness forces that call to full depth, as
  native prefill always is, so the first output token is chosen the same way.
- The release computes the exit rule in BF16 and compares it with the threshold
  rounded to BF16 (0.2 becomes 0.2002); native uses FP32 gate scores. Comparisons of
  the two backends therefore include exit-arithmetic and kernel differences, not only
  the KV layout.

Metadata and summaries record the exit settings. Adaptive summaries add `depth`: decode
token count, mean decode depth, a depth histogram and mean decode loops per
question. For the Transformers release these are the selected exit loops, recorded
with its own rule; it still computes every loop for every token. The first output
token comes from full-depth prefill and is excluded from these statistics. The stored
GSM8K-87 baseline describes fixed-depth generation, so adaptive native runs report no
`baseline_comparison` and no gate.

`compare --reference A --candidate B` (aliases of `--transformers`/`--native`)
accepts any pair of backends from the same protocol and reports each side's backend,
exit settings and depth summary alongside the paired accuracy result. Result keys use
`reference_*`/`candidate_*`; the earlier `transformers_*`/`native_*` keys are also
written only when the reference is the Transformers release and the candidate is native.

Matching answers can hide different executions, for example between synchronous and
asynchronous runs of one exit policy. `sequence_differences` therefore reports, for the
generated token sequences and for the exit-depth sequences (where both runs record
them), how many questions differ and where each first diverges. An exit-depth difference
counts as `identical_context` when the two runs chose different depths for a position
whose preceding tokens were still identical. These fields are diagnostic; the accuracy
gate does not use them.

Dataset: [GSM8K](https://huggingface.co/datasets/openai/gsm8k).
The [Ouro evaluation settings](https://arxiv.org/html/2510.25741v5#A3.T16) do not
pin the exact harness revision, demonstrations or token limits, so the settings
above are explicit project choices rather than an exact paper reproduction.

## Attention-backend runs

`benchmarks.gsm8k run` accepts `--attention-backend {torch,triton,flashinfer_fa2}`
for the native backend; the default stays `triton`. `--cuda-graphs` selects the
native recurrent-core CUDA Graphs for that run. Both flags are recorded in the
summary and do not change the frozen baseline fingerprint.

On one L20 (SM89, torch 2.13.0, CUDA 13.0, FlashInfer 0.7.2.post1, Triton
3.7.1), the default GSM8K-87 protocol (HF baseline 59/87) produced:

| run | correct | accuracy | gate |
|---|---:|---:|---|
| triton | 61/87 | 70.11% | pass (+2.30 pp) |
| flashinfer_fa2 eager | 59/87 | 67.82% | pass (+0.00 pp) |
| flashinfer_fa2 + CUDA Graphs | 59/87 | 67.82% | pass (+0.00 pp) |

Eager and graph FA2 are **bit-identical (87/87 token sequences)**. This backend
disables FlashInfer's split-KV: with splitting, a row's reduction order depends
on the planned width and batch composition, which made the earlier eager and
graph runs differ on 18 questions. Without splits, 67/87 sequences match
Triton; four answers flip (203, 426, 819 correct → incorrect; 871 incorrect →
correct). There is no length-limited runaway (mean 106.7 tokens; eager 717.8 s,
graphs 279.6 s). The no-split arithmetic reproduces the earlier split-KV graph
run on all 87 questions; the earlier split-KV eager run scored 60/87.

The `flashinfer_fa2` backend runs FlashInfer's tensor-core, CUDA-Graph-
compatible decode wrapper; the FA2 guide records why the CUDA-core path is not
graph-safe on SM89. CUDA Graphs remove most per-token launch overhead on this
small model for both attention backends: a matched 64-token fixed-work run at
concurrency 1/8/32 went from 5.49/5.55/6.32 s (FA2 eager) to 1.59/1.84/2.24 s
(FA2 graphs), and from 5.20/5.34/6.26 s (Triton eager) to 1.52/1.76/2.20 s
(Triton graphs). Triton graphs is 2–5% faster than FA2 graphs here, so the gain
is graph dispatch reduction rather than the FA2 kernel; async+graphs is 1–4%
faster than FA2 graphs. That is a launch-overhead result on one L20, not a
production throughput qualification. A GPU test keeps `prepare_step` and
`__call__` free of host synchronization across a plan-width bucket change (it
runs them under `torch.cuda.set_sync_debug_mode`), and a CPU test bounds eager
plan creation over a 1K-token decode.

The stored GSM8K-87 baseline fingerprints `torch==2.13.0`; a run on another torch
version cannot reuse its 59/87 baseline. A 10-question custom subset on torch 2.8
was used only as an early smoke (3/10 with both backends, 8/10 identical).


The [full 1,319-question comparison](https://github.com/InferMatrixLab/vllm-rlt/blob/d2f7db20593cccfe666b3c04c3ee50abc8c39af7/docs/benchmarks/gsm8k-bf16-20260912.md) recorded
**62.02% native versus 61.64% Transformers** with strict matching. It includes
per-question audits and a separate extraction diagnostic. That experiment's
frozen 75.92% reference floor failed; the current default remains the measured
GSM8K-87 baseline described above.
