# Ouro and Nanbeige validation and reproducibility index

This index answers [issue #124](https://github.com/InferMatrixLab/vllm-rlt/issues/124)
for **ByteDance/Ouro-1.4B** and **Nanbeige/Nanbeige4.2-3B**. It separates available
test code, historical reports, downloadable artifacts and missing evidence.
It does not extend those results to other checkpoints or execution modes.

Audit date: **2026-11-01**. The inspected code is
[`989921ac1d3897c5c43e27ec8cddbc91fa953194`](https://github.com/InferMatrixLab/vllm-rlt/tree/989921ac1d3897c5c43e27ec8cddbc91fa953194).
Relative source links below follow the reader's checkout; historical links pin
their own revisions. This documentation audit ran no checkpoint inference or new
GPU/NPU experiments. Historical passes and unexecuted recipes are not validation
of the audited runtime.
Model revisions below come from the cited records; no checkpoint files were
downloaded or reverified against Hugging Face in this audit.

## Reference versus native checks

| Model / entry point | What it checks | Evidence boundary |
| --- | --- | --- |
| Ouro: [CPU tests](../tests/test_ouro.py), [dense oracle](../tests/reference/models.py) | Tiny random models: per-depth hidden states, gates and logits, packed/incremental execution, mixed depths and checkpoint loading | Internal numerical consistency; these are not official HF checkpoint comparisons |
| Ouro: [GSM8K evaluator](../benchmarks/gsm8k.py), [generation adapters](../benchmarks/gsm8k_backends.py), [protocol](accuracy.md) | Released HF `generate()` versus native generation from identical prompts; BF16, fixed four loops, greedy, batch one by default | Task accuracy and saved sequence diagnostics, not bitwise logits equivalence; historical results are listed below |
| Nanbeige: [CPU tests](../tests/test_nanbeige.py), [dense oracle](../tests/reference/models.py) | Tiny random models with two loops, GQA variants, packed/incremental execution and loading | Internal consistency only |
| Nanbeige: [`test_parity_against_official_hf_transformers`](../tests/test_nanbeige.py) | Official HF remote-code architecture, downsized to two layers / hidden size 256 with random weights; FP32 parameter shapes and logits, `atol=1e-5`, `rtol=1e-4` | Needs cached official config/code; skips without them. Does not load the released 3B weights |
| Nanbeige: [`test_parity_against_real_checkpoint_hf_transformers`](../tests/test_nanbeige.py) | Real local weights, BF16: last-position logits for three prompts (Chinese, factual English, code); exact top-1 and top-5 set, cosine similarity ≥0.9998; five greedy tokens for the English prompt | Current script uses **CPU/Torch**, not CUDA. Requires `artifacts/models/Nanbeige4.2-3B`; skips without weights. It initializes the HF RoPE `inv_freq` buffer explicitly and uses uncached HF full-prefix forwards |

### Historical external-reference reports

- **Ouro GSM8K:** [accuracy guide](accuracy.md),
  [fixed 87-question fixture](../benchmarks/fixtures/gsm8k-87.json), and
  [full-split report at `d2f7db2`](https://github.com/InferMatrixLab/vllm-rlt/blob/d2f7db20593cccfe666b3c04c3ee50abc8c39af7/docs/benchmarks/gsm8k-bf16-20260912.md).
  The fixture stores HF **59/87**, source IDs, package pins and hashes, not the
  per-question generations. The guide reports native 59/87 with the same
  correctness labels. The cohort was selected after inspecting the original
  100-question run; it is a regression screen, not a held-out quality estimate.
- **Nanbeige HF parity:** [PR #63's author report](https://github.com/InferMatrixLab/vllm-rlt/pull/63#issuecomment-5905825526)
  and [follow-up with token traces](https://github.com/InferMatrixLab/vllm-rlt/pull/63#issuecomment-5906226255)
  report BF16 top-token/logit agreement and five-step greedy agreement on an
  RTX 5080, torch 2.6.0+cu124, Transformers 5.17.0. They do not pin the exact
  executed code SHA or HF model revision. PR #63's final head is
  `40670cee67b8ab2cd6a12ff07f8a50b870629f7f`; that metadata is not an experiment
  source pin. The follow-up contains inconsistent English token IDs/traces
  between its own sections. No complete machine-readable logits/output archive
  was linked in the inspected discussion. The GPU wording also differs from
  the current CPU test above. Treat these as attributed reports, not a fully
  reproducible GPU qualification or a locally confirmed result.

## Accuracy evidence and per-example outputs

The full Ouro comparison reports **818/1,319 native (62.02%)** versus
**813/1,319 HF (61.64%)**, strict match. Its executed source was
[`080821e6a64e4a5051981f5e4185e6788d66d976`](https://github.com/InferMatrixLab/vllm-rlt/tree/080821e6a64e4a5051981f5e4185e6788d66d976),
not the later report commit.
The model revision was `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` and GSM8K
revision `740312add88f781978c0658806c59bc2815b9866`.

Recorded controls: one NVIDIA L20X, torch 2.13.0+cu130, Transformers 4.55.0,
Triton 3.7.1, lm-eval 0.4.9.2; BF16, four loops, greedy batch-one inference,
three CoT demonstrations, no chat template/added BOS, 1,024 new tokens within
2,048 total tokens. HF used eager attention and native used Triton. The
predeclared **75.92% reference floor failed**; the combined historical gate
remains failed. Post-hoc flexible-extractor scores do not replace strict scoring
or reproduce the paper's score.

The report links an
[original raw-evidence archive](https://github.com/hsliuustc0106/vllm-lt/releases/download/gsm8k-bf16-evidence-20260912/gsm8k-bf16-evidence-20260912.tar.gz)
with recorded SHA256
`e749f3245f150940db977f47f23b7e8747aa414fdb32cfea8baa6a01a6701bac`.
It describes protocols, raw outputs, hashes, commands, environment and audit
records. **The release-tag API returned HTTP 404 during this audit** after the
old repository resolved to `InferMatrixLab/vllm-rlt`; the archive could not be
retrieved or checksum-verified. The report and fixture remain public, but their
hashes alone do not recover the missing per-example files.

No Nanbeige task-accuracy evaluator/result package or complete per-example
quality dataset was located in the inspected sources. Its three-prompt parity
check and PD agreement tests are not task-accuracy evaluations. The Ouro GSM8K
adapter contains Ouro-specific cache and exit handling; do not relabel it as a
validated Nanbeige evaluator.

## Historical integration and performance evidence

These records answer different questions. A native single-engine reference is
not the official HF implementation, and PD agreement is not model accuracy.

| Record | Source, model and configuration | Public evidence and limits |
| --- | --- | --- |
| [Ouro async/FA4, PR #30](https://github.com/InferMatrixLab/vllm-rlt/pull/30) | Report links source `c0d7a5ad4f3c5987f0d429acf3967d59069d90f2`; final PR head `488a0386573811f529aee30f320942ba8e5d925b`. B300, BF16, FA4 4.0.0b30, LAST_EXITED, 1K–64K inputs, 128 outputs; fixed and adaptive exit policies | Public tables and protocol; model revision not pinned. Raw timings, output IDs, depths and experiment scripts are explicitly local/unattached. The 612-trial campaign includes 196 trials with token or depth differences; it does not establish full BF16 equivalence |
| [Ouro PD, PR #31](https://github.com/InferMatrixLab/vllm-rlt/pull/31) | Final PR head `6bd9760be70e79ef41c5250b98f03a26d561a3da`; measured SHA/model revision not separately pinned. Four GPUs, BF16/FA4, async + graphs, 512 ShareGPT prompts, 128 outputs, 4/8 requests/s; replicas versus PD ratios | Public summary and protocol; no downloadable raw log/result bundle located. Mixed concurrent runs and isolated reruns, with phase-percentile averaging; not uniformly isolated measurements |
| [Nanbeige PD correctness, PR #77](https://github.com/InferMatrixLab/vllm-rlt/pull/77), [corrected self-review](https://github.com/InferMatrixLab/vllm-rlt/pull/77#issuecomment-6095406235) | Self-review names `829588b15758a3da9431cd049aac4c34f0aa1922`; final head `cedaedf7e05c8db69c8a7909004c66b3b24e7248`. Model `b82e54bd609793562a75cbf9337970a93369eab5`; 2×RTX 4080 SUPER as reported, torch 2.7.0+cu128, Triton 3.3.0, NIXL 1.4.1/UCX; BF16, fixed two loops, Triton, LAST_EXITED, async, graphs off | Six GPU tests in official-checkpoint mode: **five official-weight cases plus one controlled tiny EOS case**. Token/finish agreement, cancellation, ID reuse and cleanup; reference is native single-engine. No raw archive linked. Earlier collection-only self-review was superseded; final full Linux rerun was not confirmed |
| [Nanbeige performance comment](https://github.com/InferMatrixLab/vllm-rlt/pull/77#issuecomment-6160260065) | `cedaedf7e05c8db69c8a7909004c66b3b24e7248`, official weights (comment does not separately repin revision); 2×B300/NVLink, BF16/Triton, two loops, async, graphs/prefix cache off. Two replicas versus 1P1D; alternating 128/4096 inputs, 64 outputs, Poisson arrivals for 60 seconds, three repeats | Public tables and profiling interpretation; no linked raw logs/traces. Lower decode latency trades off against TTFT/throughput. At 1 request/s, 6/166 replica-baseline requests diverged; that point remains diagnostic |
| [Ouro real-checkpoint PD, draft PR #119](https://github.com/InferMatrixLab/vllm-rlt/pull/119) | Inspected draft head `e1a94eac540f75e6fee53c09a1668ee261143827`; report does not separately pin measured SHA/model revision. Three H100s, BF16/Triton, NIXL; matched prefill chunk 64, fixed/learned exit and seeded sampling | Reports four passes; [frozen test](https://github.com/InferMatrixLab/vllm-rlt/blob/e1a94eac540f75e6fee53c09a1668ee261143827/tests/test_pd_agreement.py). Solo checks compare native PD and colocated execution. Mixed batches assert liveness/shape, not exact equality. **Draft/unmerged and absent from audited main**; no raw result archive linked |
| [Ouro Triton kernels, PR #128](https://github.com/InferMatrixLab/vllm-rlt/pull/128) | Measured A=`9d601ce`, B=`43aa99e` (not final PR head `6b02350a`); Ouro `574fa66c`, BF16, H100/Triton, four loops. ABBA at 2K/8K inputs, C=1/4/16, 128 outputs; separate warmed 2K/64 profile | Public per-arm timing tables and **downloadable traces** below. Includes greedy differences versus Torch and a small-batch eager regression. Its native A/B GSM8K screen is separate from an HF baseline; do not transfer the fixture's 59/87 to different package pins |
| [October serving campaign](unified_execution.md#measured-serving-performance), [runtime history](runtime-history.md), [speculation measurements](async-speculative.md) | Each section records its own workloads, configuration and source information; missing pins remain missing | Useful summary/methodology links, not a complete current-checkpoint validation bundle. Serving has unresolved cross-configuration output differences. Some speculation records say raw JSON/logs are available from the reviewer on request; that is not a public download |

### Available raw profiling artifacts

[PR #128's evidence release](https://github.com/0z5a/vllm-rlt/releases/tag/pr-evidence-triton-kernels-20261015)
contains two Chrome/Perfetto traces. Both files were downloaded in this audit;
SHA256 matched GitHub's asset digest and gzip decompression completed without
error. This verifies artifact integrity, not the historical timing conclusions.
These are diagnostic profiles, not unprofiled benchmark samples or per-question
accuracy outputs.

| Artifact | SHA256 |
| --- | --- |
| [A: `graphs-main.trace.json.gz`](https://github.com/0z5a/vllm-rlt/releases/download/pr-evidence-triton-kernels-20261015/graphs-main.trace.json.gz) | `ac33252ca91806e193bcd30ffcfa487207c750e56ac7f5400b75dbadb74f8dad` |
| [B: `graphs-pr-43aa99e.trace.json.gz`](https://github.com/0z5a/vllm-rlt/releases/download/pr-evidence-triton-kernels-20261015/graphs-pr-43aa99e.trace.json.gz) | `f886401f6968428e05d395d33760c5d3326fca88aacf52a5db16c6b323a9cb96` |

The [upstream Actions artifact listing](https://api.github.com/repos/InferMatrixLab/vllm-rlt/actions/artifacts)
returned zero artifacts on the audit date. The
[CPU workflow](../.github/workflows/cpu-tests.yml) runs lint/format and CPU
regressions; it does not upload a real-checkpoint validation bundle. The
[upstream releases](https://github.com/InferMatrixLab/vllm-rlt/releases) inspected
were v0.1.0/v0.2.0 distribution packages and checksums, not experiment evidence.
This inventory covers those releases, the repository/history and discussions
linked above; it does not assert that private or unlinked artifacts do not exist.

## Reproduction entry points

All checkpoint/GPU commands below are **unexecuted recipes from this audit**.
Prepare a separate environment and reserve the required devices before running
them. Keep generated artifacts outside Git. Pin the checkout, checkpoint,
tokenizer, packages, inputs and execution configuration; a fresh run measures
that checkout, not the historical implementation automatically.

### CPU implementation checks

After [installing the project](launching.md), these need no pretrained assets:

```bash
python -m pytest tests/test_ouro.py tests/test_nanbeige.py -q -rs \
  -k 'not official_nanbeige and not parity_against'
```

For the Nanbeige official-code/real-weight checks, the current test requires
the vendor Python files as well as config/tokenizer/weights. It executes that
HF code with `trust_remote_code=True`. The following downloads the revision
recorded by **PR #77**, not a recovered revision for PR #63:

```bash
hf download Nanbeige/Nanbeige4.2-3B \
  --revision b82e54bd609793562a75cbf9337970a93369eab5 \
  --local-dir artifacts/models/Nanbeige4.2-3B
python -m pytest tests/test_nanbeige.py -v -rs -k 'official_nanbeige or parity_against'
```

The real-weight test needs enough host RAM for the 3B checkpoint and activations.
Inspect skip reasons: successful collection or a skipped test is not a parity
pass. The three-prompt checks do not establish full task accuracy.

### Ouro HF/native accuracy comparison

Follow [the accuracy guide](accuracy.md) for environment pins, preparation,
reservation and run budget. A complete command sequence is:

```bash
hf download ByteDance/Ouro-1.4B \
  --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /path/to/ouro
python -m benchmarks.gsm8k prepare --model /path/to/ouro \
  --model-repo ByteDance/Ouro-1.4B \
  --model-revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --output /path/to/protocol.json
# Run each backend inside a reservation exposing exactly one CUDA device.
python -m benchmarks.gsm8k run --backend transformers \
  --protocol /path/to/protocol.json --output /path/to/hf-results
python -m benchmarks.gsm8k run --backend native \
  --protocol /path/to/protocol.json --output /path/to/native-results
python -m benchmarks.gsm8k compare --transformers /path/to/hf-results \
  --native /path/to/native-results --output /path/to/comparison.json
```

`prepare` requires explicit `--model-repo` and `--model-revision`; it fetches
dataset/model metadata without CUDA. The default fixture requires its recorded
recipe/package fingerprint. For a fresh custom comparison, use `--all` or an
explicit `--limit` and run **both** backends; do not substitute a different
recipe into the stored baseline. Each result directory contains `metadata.json`,
`samples.jsonl` (text, token IDs, answers, correctness and finish reasons) and,
on completion, `summary.json`; retain the common protocol and stdout/stderr too.
Incomplete runs preserve completed samples and must not be scored on a reduced
denominator. This evaluator records durations but is not a serving benchmark.

### Benchmark and historical PD commands

- [Serving startup and one request](serving.md) and [CLI usage](launching.md)
  provide current smoke-test entry points; a successful response is not an
  HF comparison or performance measurement.
- [Preemption benchmark](preemption-benchmarks.md) and
  [`benchmarks.preemption`](../benchmarks/preemption.py) preserve current Ouro
  paired workload commands and `raw.jsonl`/`summary.json` outputs. Its example
  arrival rates are placeholders; it is not the script used for #30/#31/#77.
- [`benchmarks.cdb_runtime`](../benchmarks/cdb_runtime.py) is a tiny random-model
  replay benchmark. Its CPU/CUDA timing is not Ouro-1.4B or Nanbeige performance.
- [Profiling](profiling.md) explains current collection. Keep unprofiled timing
  separate from diagnostic profiles and record capture settings/windows.
- To investigate the **historical** #77 tests, use a separate checkout at
  `cedaedf7e05c8db69c8a7909004c66b3b24e7248` and its dependencies. Set
  `VLLM_RLT_NANBEIGE_CHECKPOINT=/path/to/pinned/nanbeige`, then run
  `python -m pytest tests/test_pd.py --run-gpu -k 'nanbeige and gpu' -v -rs`
  with two reserved CUDA devices and NIXL. Without that environment variable,
  the [historical fixture](https://github.com/InferMatrixLab/vllm-rlt/blob/cedaedf7e05c8db69c8a7909004c66b3b24e7248/tests/test_pd.py#L351)
  generates tiny weights; even with it, the EOS case stays tiny.
  Those six tests are not retained as that suite in audited main.
- To investigate **draft** #119, use its frozen head and three reserved CUDA
  devices, NIXL and a pinned local `RLT_AGREEMENT_MODEL`, then run
  `python -m pytest tests/test_pd_agreement.py --run-gpu -v -rs`.
  The reported H100 command uses
  `UCX_TLS=tcp,sm,self,cuda_copy,cuda_ipc` when accelerated IB is unavailable.
  It is not a command available on audited main or an official-HF test.

For any new evidence package, retain the executed source SHA/diff, exact model
and tokenizer revisions/file hashes, dependency and device versions, backend,
dtype/KV/loop/exit/sampling settings, prompts/token IDs, dataset revision,
warmup/repetition and timing definitions, all failures, commands/logs, raw
per-request/per-example results and checksums. Missing historical fields need
the original author's artifacts or a separately labeled new experiment; they
cannot be reconstructed from a summary table.

## Checks actually rerun for this audit

On macOS ARM64, Python 3.12.14, torch 2.13.0, Transformers 4.55.0 and Ruff
0.13.2, `pre-commit run --all-files --verbose` passed lint, format and CPU
regression checks: **911 passed, 584 skipped**. Hub access was offline and GPU
tests were disabled. Skips were 549 GPU cases, 28 cases needing optional
`lm_eval`, and seven cases needing official tokenizer/code/weight assets.
An initial sandboxed run had 21 local socket/shared-memory permission failures;
the same checks passed after allowing those operations.

Relative links/anchors, shell syntax and required GSM8K command arguments were
checked. The preemption command accepted `--plan-only` and produced eight cases
without inference. The two public profiling archives passed the integrity checks
described above. None of these checks reran the historical checkpoint accuracy,
PD integration or performance experiments.
