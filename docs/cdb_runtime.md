# CDB runtime implementation and verification

Implemented 2026-09-15 against base commit `6e8abbc405e1`. CUDA graphs and Huginn remain deferred. The lookahead head is intentionally random; no distillation or pretrained quality claim is made.

## Feature checklist

| Feature | Implementation | Verification |
| --- | --- | --- |
| SHARED KV | `core/kv_cache_manager.py`: one physical storage plane, overwrites at every loop; finalization validates without copying | Shared overwrite/read/free tests; chunk/batch invariance; CUDA comparison |
| Random lookahead head | `worker/model_runner.py`: separate frozen linear head, independent CPU seed; base checkpoint untouched | Same seed reproduces weights; CPU/CUDA RNG preservation; base state names unchanged |
| Delayed exit routing | `engine/llm_engine.py`: synchronous pending depth or async previous-submission signal | Signal at round 2 exits after round 3; final hidden and all depth bounds checked |
| CPU/GPU pipeline | `_step_async()` submits round r before collecting round r-1 signals | Submission-order test; GPU outputs/depths compared with sync |
| Multiple CUDA streams | `ModelRunner.submit/finalize/release`: core/boundary streams and per-request completion events | CUDA cancellation/reuse; event test demonstrates overlap with artificially extended core |
| Reusable execution buffers | `worker/buffers.py`, runner state pool and readback leases | Static/dynamic comparison; slot reuse; readback ownership; GPU tests |
| Power-of-two padding | Inactive rows have zero context, no KV write address, no sampling/routing | Free-page sentinels; padded/unpadded outputs; torch/Triton attention regression |
| Automatic KV capacity | `core/memory.py`: CUDA peak probe, memory fraction, headroom, overlap/static budget, useful-capacity cap | CPU byte-budget checks; actual CUDA profiling and admission tests |
| Admission fairness | `Scheduler._admit`: bounded scan and bounded bypass count | Short request admitted behind blocked long request; protection prevents further bypass |
| Chunked prefill policy | Per-request chunk ceiling, queue rotation and bounded prefill batches before decode | Short prompt completes before long prefill; existing decode starvation and chunk consistency tests |
| Exit-trace replay | `ExitConfig(mode="trace")` bypasses gate compute/readback | Variable-depth replay, gate hook rejects accidental execution, all replay variants match |

This implements runtime mechanisms. It does not reproduce the paper's complete accuracy ablations, workload distributions, roofline fit or published speedups.

## Configuration

```python
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, LLM, SchedulerConfig

llm = LLM(
    model,
    cache_config=CacheConfig(layout="shared", gpu_memory_utilization=0.9),
    scheduler_config=SchedulerConfig(
        prefill_chunk_size=128,
        max_prefill_batches_before_decode=1,
        admission_scan_limit=64,
        max_admission_bypasses=8,
    ),
    exit_config=ExitConfig(mode="random_lookahead", seed=7),
    execution_config=ExecutionConfig(
        async_scheduling=True, static_buffers=True, pad_to_power_of_two=True,
    ),
    attention_backend="triton",  # CUDA async requires this backend
)
```

The default exit mode remains `ouro`, with the original cumulative hazard policy. For random lookahead, sigmoid is a direct score for exiting after one more loop, not a hazard/CDF. `SamplingParams.exit_threshold` compares this score directly. Threshold 1 disables adaptive exit; `max_loops` always caps execution. Adaptive lookahead cannot exit before round 2, although an explicitly configured hard maximum of 1 is honored. All prompt tokens still run the model's full depth.

Async execution requires `ouro_delayed`, `random_lookahead` or `trace`. It automatically enables reusable static inputs to avoid blocking device metadata transfers. CPU supports the same scheduling state machine for tests, with immediately completed operations; it does not simulate GPU parallel execution. `multi_stream=False` shares the core stream for controlled comparisons.

CLI flags are shared by offline and serving entrypoints:

```bash
OMP_NUM_THREADS=1 python -m vllm_rlt.entrypoints.cli --toy --dtype float32 \
  --max-tokens 4 --exit-mode random_lookahead --lookahead-seed 7 \
  --exit-threshold 0.5 --async-scheduling --static-buffers \
  --pad-to-power-of-two --kv-layout shared --prefill-chunk-size 2
```

For a task-assigned GPU, set `CUDA_VISIBLE_DEVICES` and add `--device cuda --attention-backend triton`. Use `--single-stream` for the single-stream pipeline. `--num-blocks` overrides automatic sizing, or specify `--kv-cache-memory-bytes`; these two overrides are mutually exclusive. CPU auto mode keeps a 256-block default. `engine.memory_plan` records how the capacity was selected.

## Fixed-loop self-speculative decoding

Use shallow loops to draft tokens, then reuse their hidden states and KV to
verify them at full depth. Only verified output is committed; rejected suffixes
are discarded and corrected by the target. The feature is opt-in.

```python
from vllm_rlt import LLM, SamplingParams, SpeculativeConfig

llm = LLM(
    "../models/Ouro-1.4B",
    device="cuda",
    attention_backend="flash_attn_4",
    speculative_config=SpeculativeConfig(
        num_speculative_tokens=4, draft_loops=2, target_loops=4,
    ),
)
outputs = llm.generate(
    ["Explain why the sky is blue."],
    SamplingParams(max_tokens=64, max_loops=4, exit_threshold=1.0),
)
```

Equivalent offline CLI (select an available GPU through `CUDA_VISIBLE_DEVICES`):

```bash
python -m vllm_rlt.entrypoints.cli \
  --model ../models/Ouro-1.4B --device cuda --dtype bfloat16 \
  --attention-backend flash_attn_4 --speculative-tokens 4 \
  --draft-loops 2 --target-loops 4 --max-loops 4 --exit-threshold 1.0 \
  --max-tokens 64 --prompt "Explain why the sky is blue."
```

`num_speculative_tokens` / `--speculative-tokens` is K, the maximum candidates
per round; K=4 here is an example, not a universal optimum. Draft depth defaults
to 2 and must be below target depth. Target depth must equal the full model depth
(4 for Ouro-1.4B). K shrinks to fit the token budget and remaining output length.

Requires `last_exited`, fixed-depth `ouro` exits, eager execution,
and refill scheduling. Cross-round async greedy decoding is available with CUDA
and Triton; see [async self-speculation](async-speculative.md). CUDA Graphs,
preemption, no-refill, and PD are not
supported. Greedy and temperature/top-k/top-p sampling are supported; sampling
preserves the target distribution in exact arithmetic, not identical text for
the same seed. BF16 batch shapes can introduce numerical differences.

The same flags work with the single-device server. A streaming event may contain
multiple committed tokens. See [RFC #43](https://github.com/ThinkFlowLab/vllm-rlt/issues/43)
for the design.

### Alternate-branch experiment (PR #59)

[PR #59](https://github.com/ThinkFlowLab/vllm-rlt/pull/59) tested whether drafting
one alternate suffix could reduce generation time. Small gains at concurrency 1
did not extend to concurrency 4 or 8, so the maintainers recommended stopping
this approach and keeping standard speculation. The implementation was not
merged; the results are recorded here for future work.

The gate reads the shallow full-softmax top two probabilities `p1` and `p2` of
the draft logits. Two policies were evaluated: an original policy that forks
when `p1 - p2 <= 0.2`, and a revised policy that additionally requires
`p1 + p2 >= 0.4`. A fork opens a private alternate KV stem and candidate
suffix, drafted alongside the primary chain. The alternate is verified at the
same full target depth as the primary, inside the same packed verification
batch, and is committed only when the first primary rejection lands exactly at
the fork and the target's corrected token matches the stored alternate;
otherwise it is discarded at round end.

#### L40S results (contributor)

Source: [public PR #59 comment](https://github.com/ThinkFlowLab/vllm-rlt/pull/59#issuecomment-6096514596);
raw runs in the linked [results.zip](https://github.com/prettygirlisnotme/vllm-rlt/releases/download/pr59-review-evidence-20261010/results.zip),
profiles in the linked [profiles.zip](https://github.com/prettygirlisnotme/vllm-rlt/releases/download/pr59-review-evidence-20261010/profiles.zip).
The tests used Ouro-1.4B at `574fa66` and runtime `07bf8cb` (based on main
`ecb1f8b`): BF16/Triton, synchronous eager, greedy, target depth 4, four draft
tokens, and 64 fixed prompts (32 GSM8K + 32 HumanEval). Each request generated
128 tokens with EOS ignored. After warmup, plain and branch runs alternated
for five repeats. E2E throughput below is mean ± SD in tokens/s, including
queuing, prefill and decoding for 64 simultaneously submitted requests.

| Draft / concurrency | E2E plain | E2E original gap policy |
| --- | ---: | ---: |
| d1 / 1 | 38.06 ± 0.99 | 39.75 ± 1.21 |
| d1 / 4 | 138.54 ± 1.35 | 134.34 ± 2.04 |
| d1 / 8 | 252.38 ± 3.88 | 231.09 ± 1.94 |
| d2 / 1 | 29.83 ± 0.33 | 30.61 ± 0.42 |
| d2 / 4 | 109.46 ± 0.82 | 109.23 ± 1.19 |
| d2 / 8 | 195.85 ± 1.71 | 191.72 ± 1.47 |

Gains were confined to low concurrency and reversed at C8. The d2/C8
decode-only gain was +6.76% (cohort-bootstrap 95% CI: +1.20% to +14.55%), while the same
condition's E2E was -2.11%. Decode-only and E2E are different workloads, so
that gap is not evidence that queuing or prefill caused the reversal.
Full-quality runs gave identical task scores for plain and branch (GSM8K
61.71%/62.09%, HumanEval 69.51% at d1/d2), but equal scores on these finite
datasets do not establish BF16 losslessness; the separate BF16 diagnostics
retain sequence divergences.

#### B300 results (maintainer)

Source: [maintainer comment](https://github.com/ThinkFlowLab/vllm-rlt/pull/59#issuecomment-6139829875).
Ouro-1.4B on a single B300, BF16, target depth 4, draft depths 1/2 with 4 draft
tokens, 64 requests (32 GSM8K + 32 HumanEval), and 128 output tokens each. The
table reports median throughput in tokens/s across five runs, including
queuing, prefill and decoding but excluding HTTP/network overhead. The source
comment does not specify code/checkpoint revisions or backend/execution mode;
these are separate measurements, not a controlled comparison with the L40S.

| Draft / concurrency | No branch | Original policy | Revised policy |
| --- | ---: | ---: | ---: |
| d1 / 1 | 65.46 | 67.91 | 66.36 |
| d1 / 4 | 234.49 | 225.29 | 226.66 |
| d1 / 8 | 393.05 | 361.17 | 364.84 |
| d2 / 1 | 49.48 | 50.95 | 50.46 |
| d2 / 4 | 185.05 | 184.02 | 181.43 |
| d2 / 8 | 322.17 | 317.81 | 310.24 |

The revised policy also increased request completion p95 in five of six
configurations compared with the original policy.

#### Why this was stopped

- Fewer speculative rounds do not mean less total work. On B300 at d1/concurrency 8, the
  original policy cut rounds from 2,109 to 1,983 but raised verification token
  rows from 10,396 to 12,568.
- A small gap does not predict a useful branch: earlier draft tokens may be
  rejected before the fork, rejection may fall elsewhere, or the target may not
  pick the second candidate. The alternate token is itself an ordinary target
  correction, so a higher hit rate is not saved work.
- The revised policy raised the hit rate from 21.1% to 29.7% but reduced
  selected branches from 193 to 131 and increased rounds from 1,983 to 2,041.
- Maintaining an alternate suffix adds drafting, verification and KV copies;
  the gate also requires CPU readbacks. In the L40S C8 ops-only profile window,
  stream synchronizations rose from 2 to 261, with about 59.4 MB of
  device-to-device copies versus zero. The profiles show extra operations, but
  do not isolate their contribution to unprofiled runtime.

This is a negative result for the two tested policies on these workloads, not a
general impossibility claim about tree or multi-branch decoding.

## SHARED prefill semantics

SHARED is not equivalent to LAST-EXITED. A later token reads every earlier token's final KV at all depths. To make that meaning independent of chunk boundaries, the shared prefill runner completes all loops for one position before advancing that request. Different requests can still be batched at each position wave. This is a correctness-oriented implementation and can be slower than packed depth-major prefill. It is not a claim to reproduce an unpublished optimized shared-prefill kernel.

SHARED reduces physical planes from `total_ut_steps` to one; block accounting and reclaim follow the physical plane count. Full-lifetime position reservation is retained. Incremental reservation/preemption and prefix sharing remain outside this change.

## Submission lifetime and overlap

`Submission` retains its result/readback storage until completion. For lookahead, the engine launches the next required loop before collecting the previous signal. The current loop is guaranteed necessary under this signal convention, so the pipeline does not intentionally execute an extra speculative loop. Coda submissions remain pending while other requests can run recurrent work.

Request hidden states and events enforce cross-stream dependencies. The host loop counter represents submitted progress in the async path; an event establishes that the corresponding GPU state is actually complete. Finalization waits on the last core event, and coda waits on finalization. Cancellation/completion waits on the request's last event before recycling its KV and hidden slot. Partial submission failures drain owned streams before cleanup. Old coda results are rejected by Request object identity after ID reuse; previous signals also carry position and depth.

Core/boundary workspaces rotate two banks per group. Host metadata is pinned and copied nonblocking; a bank cannot be refilled while its earlier DMA/device work is using it. Readback buffers are preallocated separately, leased to submissions, and released only after consumption or safe retirement. LAST-EXITED uses basic-index device copies instead of constructing a blocking device index tensor during finalization.

Static mode preallocates inputs, metadata, persistent request states and async readback buffers. Intermediate model activations still use PyTorch allocation; this is not CUDA-graph capture or a claim of an allocation-free forward. Padding affects the compute batch; only real rows write KV, produce signals and sample tokens. `last_effective_size` and `last_submitted_size` expose the distinction.

## Capacity and admission

CUDA automatic sizing profiles a synthetic maximum-row core/coda pass. It reserves the measured extra peak (doubled conservatively for overlap), static device buffers, metadata, long-context torch-reference scratch when relevant, and `memory_reserve_bytes`. The memory-utilization budget is also bounded by startup free memory. The resulting pool is capped at the blocks usable by all concurrent full-context requests, avoiding enormous useless pools for tiny models. Profiling includes temporary KV, so the estimate is deliberately conservative; it is not a promise against other processes changing GPU memory usage after startup.

Admission scans up to `admission_scan_limit` waiting items and preserves the order of deferred requests. A later admission increments bypass counts of earlier blocked requests. Once a request reaches `max_admission_bypasses`, later admissions stop bypassing it until enough active work drains. Existing active requests can finish because lifetime KV is already reserved. Chunk sizes and prefill/decode interleaving are independent from admission and apply in both layouts.

## Replay and validation commands

Trace mode expects `depths_by_request`, with one depth per output: full prefill depth first, then each decode input's exit depth. Admission validates coverage and loop bounds. Request IDs must match the trace. The public engine replay fixes depth decisions; token generation still follows the configured sampler.

```python
ExitConfig(mode="trace", depths_by_request={"A": [4, 2, 4]})
```

The synthetic benchmark records prompts, output IDs, depths, model configuration/seed, dtype, layout and sampling configuration. It validates this fingerprint before replay and rejects output/depth mismatches. It compares refill/no-refill with eager, padded, async single-stream and async multi-stream execution. Initialization and a warmup are excluded from timing; per-step trace collection is included equally for all variants.

```bash
python -m benchmarks.cdb_runtime --device cpu --output /tmp/cdb-cpu.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.cdb_runtime --device cuda \
  --trace-out /tmp/cdb-trace.json --output /tmp/cdb-gpu.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.cdb_runtime --device cuda \
  --trace-in /tmp/cdb-trace.json --output /tmp/cdb-replay.json
CUDA_VISIBLE_DEVICES=0 python -m pytest -q --run-gpu
```

Use only a GPU available to the task. The overlap regression deliberately extends GPU core duration to test stream dependencies; it is not a throughput benchmark. The tiny replay benchmark also cannot establish real Ouro speedups. Random gate quality, real-model workload performance and the paper's accuracy results require separate evaluation.

## Recorded validation (2026-09-15)

Final full suite on an available NVIDIA B300, with `CUDA_VISIBLE_DEVICES=0` and
`--run-gpu`: **199 passed, 11 skipped**. The skips require optional `lm_eval`;
GPU tests ran. Ruff and `git diff --check` passed. The CUDA auto-budget,
shared/last-exited async comparisons, cancellation/reuse and synthetic overlap
checks all passed. The documented CLI example completed. SHARED's eight replay
variants preserved recorded output IDs and exit depths. Tiny-model timings do
not demonstrate throughput gains; no pretrained accuracy or paper speedup is claimed.


## Reusing the trained Ouro gate: `ouro_delayed`

`ExitConfig(mode="ouro_delayed")` uses the checkpoint's existing
`model.early_exit_gate` directly, without creating or copying an auxiliary head.
Its per-round sigmoid output remains a hazard: after consuming round r,
`remaining_probability *= (1 - hazard_r)`. When `1 - remaining_probability`
reaches the threshold and r >= min_loops, execution exits after round r+1.
The hard max_loops still wins, and threshold 1 disables adaptive exits.
Prefill always runs full depth; probability state resets for every decode token.

For example, hazards 0.3, 0.3 give cumulative probabilities 0.3, 0.51.
With threshold 0.5 and min_loops 2, ordinary Ouro exits at round 2;
`ouro_delayed` exits at round 3. It preserves the original gate's probability
interpretation but changes the exit depth. This is a delayed original-gate
heuristic, not a trained predictor of the next round's exit quality.

Synchronous execution stores a pending exit depth. Async submits round r+1
before collecting round r's signal, accumulating each consumed hazard once.
Both modes use the same policy. Select both with `--exit-mode ouro_delayed`;
add `--async-scheduling` for async, which automatically enables static buffers.
CUDA async requires `--attention-backend triton`. Padding is optional; use
`--static-buffers --pad-to-power-of-two` to enable it.

Fish startup with the local checkpoint (GPU 0 must be available):

```fish
cd /home/zjy/code/david/b_workspace
source .b_rdma/bin/activate.fish
cd vllm-rlt
env CUDA_VISIBLE_DEVICES=0 python -m vllm_rlt.entrypoints.serve \
  --model /home/zjy/code/david/b_workspace/models/Ouro-1.4B \
  --served-model-name ouro --device cuda --dtype bfloat16 \
  --attention-backend triton --exit-mode ouro_delayed --async-scheduling \
  --static-buffers --pad-to-power-of-two --kv-layout last_exited \
  --max-num-seqs 8 --host 127.0.0.1 --port 8000
```

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ouro","prompt":"请解释一下什么是矩阵乘法。","max_tokens":512,"temperature":0,"min_loops":2,"max_loops":4,"exit_threshold":0.9,"stream":false}' \
  | python -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["text"] if "choices" in r else json.dumps(r,ensure_ascii=False))'
```

For a scheduling comparison keep exit mode, threshold, padding, cache and
workload identical, removing only `--async-scheduling` for sync. Comparing
ordinary `ouro` to async `ouro_delayed` also changes the exit policy, confounding
a scheduling speed comparison. A single request verifies functionality;
multiple concurrent requests are needed to exercise cross-request overlap.


Validation for `ouro_delayed`: full CPU/GPU suite **215 passed, 11 skipped**
(optional lm_eval), Ruff and diff checks passed. Tests cover cumulative hazards
(0.3 + 0.3 => CDF 0.51), trigger-round minimum, hard maximum, threshold 1,
per-token reset and use of the final extra round's hidden state. Both KV layouts
passed tiny-model CUDA sync/async comparisons including padding and sampling.

Real local Ouro-1.4B was tested on B300 GPU 5, keeping the user's GPU 0 service
untouched. Three prompts with thresholds 0, 0.9 and 1, generating 16 tokens each,
matched token IDs and depths between sync and async in FP32. A BF16 single
Chinese request matched sync, async single-stream and async multi-stream.
However, BF16 three-request multi-stream execution showed a token divergence
on the French-capital prompt at output index 7 despite matching exit depths.
Single-stream matched the synchronous reference in that run; adding diagnostic
logit copies also removed the divergence. A separate synchronous single/batch
comparison did not reproduce it. Reduced-precision or schedule-dependent
numerics are a hypothesis, not a confirmed root cause. Therefore this check
does not establish exact BF16 multi-stream output equivalence or rule out all
async issues. FP32 provides the currently verified real-model comparison for
this workload. No throughput gain or task-accuracy result is claimed here.
