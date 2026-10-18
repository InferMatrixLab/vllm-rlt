# Loop-level speculative execution

This opt-in synchronous path implements the mixed-depth batching portion proposed
in [Issue #43](https://github.com/ThinkFlowLab/vllm-rlt/issues/43): advance shallow work
for later token positions in the same recurrent call as deep work for earlier
positions. It retains the existing finite speculative window and atomic round
commit. The default remains the original block verifier.

## Enable

```python
SpeculativeConfig(
    num_speculative_tokens=4,
    draft_loops=2,
    target_loops=4,
    loop_level_pipelining=True,
)
```

The offline and serving CLI accept `--speculative-tokens 4 --loop-level-pipelining`.
Greedy-only early suffix pruning can be enabled separately with
`--early-rejection-pruning`; it is off by default. Top-p and top-k sampling are
supported by the same loop pipeline, but sampling still verifies the complete
finite window before applying rejection sampling. Early pruning is intentionally
greedy-only because greedy acceptance is a deterministic candidate/argmax
comparison, while sampling acceptance requires the target and proposal
distributions, random draws and ordered correction sampling.
Existing restrictions apply: `last_exited`, full fixed target depth, synchronous
refill scheduling, and no preemption or PD.

The implementation has three execution choices:

| Mode | Recurrent schedule | Verification | Intended use |
| --- | --- | --- | --- |
| Baseline | Complete all draft positions, then complete target depth | One batched verifier window | Reference path and default |
| Loop | Advance active positions one depth per mixed-depth wave | Reorder completed target rows into the common verifier | Measure mixed-depth batching |
| Loop + greedy prune | Loop schedule with target-ready rejection checks | Stop future suffix branches after the first mismatch | Optional greedy experiment |

The loop mode is a synchronous pipeline. It does not add CUDA streams, reduce the
number of recurrent rows, or retain hidden state across speculative rounds. Its
possible benefit comes from the shape and timing of mixed-depth batches; metadata,
Python scheduling, Coda fragmentation and dynamic CUDA Graph shapes are additional
costs.

## Execution and state

### Loop-level flow

The loop path keeps one branch for each active token position. A branch carries
its request index, token offset, recurrent hidden state and current loop depth.
Each wave advances every active branch by one depth. When a branch reaches the
draft depth, its coda output starts a child branch while the parent continues
toward the target depth. When a branch reaches the target depth, its target
logits are saved for the existing verifier.

The baseline executes all draft positions before block target verification. In
the loop schedule, after A reaches the draft depth, B starts at depth 0 while A
continues at a deeper level; C starts two waves later. Target rows are reordered
into the common verifier only after their corresponding branches reach target
depth. The waves are still launched sequentially on the current host path.

This PR currently implements only the synchronous finite-window path. Mixed-
depth waves are launched sequentially from the host; there are no CUDA streams,
asynchronous scheduler overlap, cross-round hidden-state retention, preemption,
or PD execution. The existing `SpeculativeResult` and atomic KV commit/rollback
boundary are intentionally preserved.

For K=2, d=2 and D=4, A is the last committed input, B and C are proposals.
Numbers below are zero-based depths; columns are successive recurrent calls.

| Input position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 1 | 2 | 3 | | | | |
| B | | | 0 | 1 | 2 | 3 | | |
| C | | | | | 0 | 1 | 2 | 3 |

After A completes depth 1, its shallow head proposes B. Call 3 batches A at
depth 2 and B at depth 0. After B completes depth 1, it proposes C while A's
full-depth head produces the distribution that verifies B. These head rows are
batched too. C's full-depth distribution supplies the bonus token.

Each branch stores its item index, position offset, hidden state and completed
loop count. A child starts d calls behind its parent. Therefore, at every layer,
the same-depth KV for earlier positions is already available before the child's
attention. `_prepare_batch` supplies a separate depth plane and causal context
length for every row. The pipeline uses single-query paged attention, not packed
multi-query prefill. No shallow KV is copied into uncomputed deep planes.

Each position/depth is computed exactly once. Proposals remain on the device and
greedy target IDs are read back once per mixed-depth wave. For sampling, the
filtered proposal distribution `q` is retained for each draft and the common
verifier applies rejection sampling with the request's generator from the shared
`RngRegistry`. Different GEMM shapes and the changed draft/verify interleaving can
change BF16 results or the fixed-seed token sequence; same-seed agreement is not a
general sampling correctness guarantee.

Completed target logits are reordered into item/position order for the common
verifier. `SpeculativeResult` and engine commit stay unchanged: keep KV through
the initial input and accepted candidates; invalidate every depth's rejected
suffix; leave correction/bonus unforwarded. Engine EOS/length handling and
exception cleanup remain responsible for the request lifecycle.

## Scope and benefit

The default finite-window path does **not** implement cross-round hidden-state
retention or the complete continuous LoopSpec algorithm. The optional greedy
pruning path checks a target row when its coda output is ready and skips later
suffix branches after rejection. Work that already entered the mixed-depth wave
cannot be cancelled. Sampling still runs the complete window before
verification. Greedy target and draft IDs are read back once per mixed-depth
wave; the host-side rejection decision remains the pruning synchronization
boundary. No new CUDA streams or asynchronous scheduler are introduced.

### Why pruning is greedy-only

For greedy decoding, a target row produces one deterministic token ID. Once that
ID is available, the runner can compare it with the candidate and stop creating
later branches for that request after the first mismatch. The already submitted
rows still finish, but suffix rows that have not entered a wave are avoided.

Top-p and top-k use rejection sampling. Acceptance depends on the filtered target
distribution `p`, the filtered draft distribution `q`, and a random uniform draw;
on rejection, the correction token is drawn from `normalize(max(p - q, 0))`.
A target argmax mismatch is therefore not equivalent to rejection: a candidate
can differ from the target argmax and still be accepted, while a candidate that
matches the argmax can still be rejected by the random draw. Applying the full
decision as soon as each target row is ready would also require retaining every
proposal distribution, preserving the request's RNG order, sampling a residual
correction token, and invalidating all descendants generated from the rejected
candidate. The current sampling path instead computes the finite window and
applies the existing verifier in order, preserving its distribution and RNG
behavior. Device-side batched acceptance can support sampling pruning in a later
change.

For a uniform window, both the original block verifier and this pipeline require
`d*K + D` recurrent calls and `(K+1)*D` recurrent rows. Pipelining changes batch
shapes and head batching, not total recurrent work or that critical path. Its
cost also includes extra per-step metadata preparation and loss of the original
verifier's packed FA4 attention. Speedup must be measured; mixing depths alone
does not guarantee an improvement. Reducing round bubbles or wasted suffix work
would require a follow-up design for acceptance and state lifetime.

## Correctness checks for Issue #43

The fixed-input oracle test runs the same token chain through native serial
full-depth execution and the reuse path. It compares every loop-2 hidden row,
every depth/layer KV row and the final target logits, and checks max absolute
error, RMS error and target argmax mismatch rate in the assertion contract. The
FP32 case is used to diagnose indexing or dependency mistakes; the BF16 case uses
relaxed numerical tolerances to assess real model behavior.

Free greedy generation is checked at `K=1/2/4/8`, with the existing ragged
single-request cases and an eight-request batch. The test requires exact token
and exit-depth agreement with native generation and reports the first divergent
position in the failure message. Sampling is checked independently with small
analytical distributions after applying the actual top-p and top-k filters; the
empirical samples must match the filtered target distribution within the
sampling tolerance. Fixed-seed text equality is not required for sampling.

In the focused 16,000-draw check, the maximum absolute deviation from the
filtered target distribution was 0.003085 for top-p and 0.002593 for top-p plus
top-k; the corresponding RMS deviations were 0.001781 and 0.001368. Both are
below the test tolerance of 0.02.

```bash
python -m pytest -q tests/test_speculative.py \
  -k 'reused_hidden_logits or issue43_greedy or filtered_top_p'
```

## Parameters and results

The reference measurements use an Ouro-1.4B checkpoint on an NVIDIA A100-SXM4-40GB,
BF16, Triton attention, CUDA Graphs, `K=4`, `draft_loops=2`, `target_loops=4`,
32 generated tokens and three timed repetitions per point. The benchmark compares
the original block verifier with the loop pipeline using the same prompts and
prefill barrier. Throughput is aggregate committed decode throughput across all
concurrent requests, measured in output tokens per second (`output tok/s`).
It counts generated output tokens only, excludes prompt/prefill tokens and is
not a per-request rate.

For C=1 through C=64, the CUDA Graph limit is 128. The C=128 rows use limit 256
because the loop's largest mixed-depth wave is approximately `2C` rows. The C=64
wave is at most 128 rows and therefore remains within the default limit.

### Greedy

| Concurrency | Baseline (output tok/s) | Loop (output tok/s) | Change | Loop + greedy prune (output tok/s) | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 58.6 | 56.9 | -2.8% | 66.7 | +13.9% |
| 4 | 217.1 | 214.7 | -1.1% | 215.5 | -0.7% |
| 8 | 420.3 | 409.8 | -2.5% | 412.0 | -2.0% |
| 16 | 790.7 | 780.3 | -1.3% | 782.8 | -1.1% |
| 32 | 1319.5 | 1473.7 | +11.7% | 1465.0 | +10.4% |
| 64 | 2428.5 | 2648.4 | +9.1% | 2628.8 | +7.9% |
| 128 | 4085.1 | 4229.6 | +3.5% | 4203.6 | +3.1% |

The additional greedy counters show where pruning saves work:

| Concurrency | Loop verified rows / committed tokens | Loop + prune verified rows / committed tokens | Loop peak allocated |
| ---: | ---: | ---: | ---: |
| 1 | 40 / 32 | 32 / 32 | 2.99 GiB |
| 4 | 144 / 128 | 128 / 128 | 3.23 GiB |
| 8 | 288 / 256 | 256 / 256 | 3.54 GiB |
| 16 | 576 / 512 | 512 / 512 | 4.18 GiB |
| 32 | 1152 / 1024 | 1024 / 1024 | 5.42 GiB |
| 64 | 2304 / 2048 | 2048 / 2048 | 7.90 GiB |
| 128 | 4608 / 4096 | 4096 / 4096 | 12.88 GiB |

### Sampling

Sampling uses `temperature=0.8`, `top_p=0.8`, with `top_k=-1` for top-p and
`top_k=7` for top-p + top-k. Early pruning is disabled. The C=128 rows use
CUDA Graph max 256; all other rows use max 128.

| Concurrency | Top-p baseline (output tok/s) | Top-p loop (output tok/s) | Change | Top-p + top-k baseline (output tok/s) | Top-p + top-k loop (output tok/s) | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 61.9 | 58.5 | -5.5% | 51.6 | 46.8 | -9.4% |
| 4 | 205.1 | 159.9 | -22.0% | 139.7 | 137.8 | -1.3% |
| 8 | 295.4 | 330.2 | +11.8% | 236.3 | 254.2 | +7.5% |
| 16 | 523.8 | 492.9 | -5.9% | 417.2 | 393.2 | -5.8% |
| 32 | 662.2 | 723.9 | +9.3% | 519.3 | 532.2 | +2.5% |
| 64 | 876.7 | 914.1 | +4.3% | 625.2 | 646.8 | +3.5% |
| 128 | 1046.6 | 940.0 | -10.2% | 727.1 | 801.2 | +10.2% |

The benchmark JSON also records median/min/max decode time, peak allocated
memory, recurrent rounds, drafted/accepted tokens, verified rows and the first
divergent output position. Greedy runs have identical outputs at every point;
sampling divergence is expected when the same request generator is consumed in
a different draft/verify order. It is not used as a distribution-correctness
criterion.

The results are workload-dependent. Loop-level scheduling does not reduce the
number of recurrent rows; its potential benefit comes from mixed-depth batch
shapes and device utilization. Metadata preparation, Python scheduling, Coda
fragmentation and graph capture shape can outweigh that benefit. For example,
the greedy loop reduces verified rows from 4608 to 4096 at C=128 only when
early pruning is enabled, while the non-pruned loop keeps the same recurrent
row count as baseline. The sampling rows also show that acceptance rate and
round count change with the sampled path, so throughput should be read together
with the recorded counters.

In summary, the synchronous loop path is a clear win only at selected higher
concurrency points: greedy throughput improves by 3-12% from C=32 in this
benchmark, while low-concurrency runs regress by about 1-3%. Greedy pruning is
strongest for a single request with an early mismatch (+13.8% at C=1), but adds
little at larger batches because work already submitted in the current wave
cannot be cancelled. Sampling remains workload-sensitive, ranging from a 22%
regression to a 12% improvement for top-p and from a 9% regression to a 10%
improvement for top-p plus top-k. These results show that mixed-depth scheduling
alone does not guarantee a speedup; asynchronous overlap, state retention,
device-side acceptance and more efficient dynamic compaction remain follow-up
work.

## Validation

```bash
OMP_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
python -m ruff format --check vllm_rlt tests benchmarks

# On an allocated CUDA device with Triton:
python -m pytest -q tests/test_speculative.py --run-gpu -m gpu -k 'not flash_attn_4'
```

Tests exercise mixed depths including non-divisible d/D, actual hidden/logit/KV
agreement against a serial reference, every rejection boundary and bonus,
greedy/sampling batches, K=1/2/4/8, shortened windows, EOS, prefix reuse,
incremental allocation, cancellation and partial execution failure. CUDA coverage
checks eager/graph parity and one greedy readback per round.

The focused Issue #43 checks can be run without the full suite:

```bash
python -m pytest -q tests/test_speculative.py \
  -k 'reused_hidden_logits or issue43_greedy or filtered_top_p'
```

For a bounded real-checkpoint feasibility run:

```bash
python -m benchmarks.loop_speculative \
  --model /path/to/local/Ouro-1.4B \
  --cuda-graphs --k 4 --concurrency 1 --new-tokens 32 --repeats 3 \
  --output /tmp/loop-speculative.json

# Sampling variant (top-p + top-k):
python -m benchmarks.loop_speculative \
  --model /path/to/local/Ouro-1.4B \
  --cuda-graphs --k 4 --concurrency 8 --new-tokens 32 --repeats 3 \
  --temperature 0.8 --top-p 0.8 --top-k 7 \
  --output /tmp/loop-speculative-top-p-top-k.json
```

The harness warms both engines, completes a common prefill barrier before timing,
ignores EOS to commit equal work, synchronizes timing boundaries, alternates trial
order, and saves raw timings, outputs, first divergences and runtime metadata.
For large loop batches, `--cuda-graph-max-batch-size 256` allows Graph capture of
the 2C mixed-depth wave when C=128; the default is 128.
`--prompts` accepts a JSON list of strings; `--backend flash_attn_4` requires
compatible hardware. Peak memory includes both resident engines; the recorded
checkpoint config hash is not a weight hash. These checks cover fixed-input
reuse and sampling distribution contracts, but do not replace Issue #43's
long-context, serving-latency or full BF16 acceptance matrix.

Raw benchmark JSON is stored under `artifacts/loop-speculative/`; the extended
run notes include the runtime metadata and per-repetition timings. This work is
an opt-in implementation of the loop-level direction in Issue #43; it does not
claim that mixed-depth scheduling reduces total recurrent work or improves every
hardware and workload combination.
