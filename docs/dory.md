# Dory baseline

This page documents the native Dory implementation added for
[NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
and the exact scope that is qualified today. It describes a **baseline
increment**, not full support: the plan for continuous serving, cache
management, execution and speculation is tracked in
[RFC #176](https://github.com/ThinkFlowLab/vllm-rlt/issues/176).

## Model layout

Dory is one physical stack of 72 sublayers. Each sublayer holds a single mixer,
either attention or MLP, plus an NGPT slerp residual:

| Stage | Sublayers | KV |
| --- | --- | --- |
| Input (`n_input_layers`) | 16 (8 attention, 8 MLP) | boundary KV at depth 0 |
| Recurrent (`n_recurrent_layers`) | 40 (20 attention, 20 MLP) | per loop depth, or one shared plane |
| Output (`n_output_layers`) | 16 (8 attention, 8 MLP) | boundary KV at depth 0 |

`layer_types` maps every physical sublayer to `sliding_attention`,
`full_attention` or `mlp`, and the released artifact lists 27 sliding and 9
full attention sublayers: every fourth attention pair is full attention. When
`layer_types` is omitted the constructor generates exactly that pattern, and
refuses anything that is not attention/MLP pairs so a custom stack cannot fall
back to an all-sliding architecture by accident. `sliding_window` (1024 in the
artifact) applies only to `sliding_attention` sublayers.

The recurrent stage runs `n_recurrent_loops` times. Before every loop the
retained input-stage output is combined with the state and re-normalized, so
the runner carries `[recurrent state, injection input]` and
`DoryConfig.hidden_size` is the doubled carry width while
`DoryConfig.d_model` is the artifact width. `config.json`'s `hidden_size` maps
to `d_model`.

NGPT arithmetic is preserved explicitly: FP32 `justnorm` accumulation, learned
per-channel scales stored without their constant factors, post-RoPE Q/K
normalization, the output gate before the projection, the MLP gate/up scaling,
and slerp residuals with `ngpt_alpha_init`. Logits are scaled by the learned
vocabulary-wise `lm_head.logit_scale`.

## KV modes

`recurrent_kv_cache_mode` selects between two different semantics, and each maps
to one cache layout:

| Mode | Runtime layout | Recurrent attention history |
| --- | --- | --- |
| `per_loop` (default) | `last_exited` | Independent KV per loop depth |
| `last_loop` | `shared` | One plane per layer, overwritten by each loop |

Boundary stages always write depth 0, so input and output KV are never
replicated per loop. In `last_loop` mode a token's current-step KV is overwritten
by the next loop, so results depend on the chunk schedule; the qualified
schedule is one token per step. `per_loop` gives the same result for
token-at-a-time and all-positions-at-once execution.

Only the 36 attention sublayers own KV. The cache pool is sized by that compact
count (`DoryConfig.num_kv_layers`), not by all 72 physical sublayers, and
`StageEngine`/`plan_cache` select it through `num_kv_layers` when a model
declares one. Attention layers therefore reach the cache through a packed
layer id rather than their physical sublayer index.

## Supported today

- Native strict safetensors loading with the artifact's per-projection key
  names (`q/k/v/g/o_proj`, `gate_proj`/`up_proj`) and the learned scale tensors.
  There is no fused-key remapping: a checkpoint that packs `qkv_proj` or
  `gate_up_proj` fails loudly instead of loading approximately.
- Synchronous eager execution: full-prompt prefill followed by ordinary token
  decoding, with the configured fixed loop count.
- `loops=` at load time to fix 1, 2 or 4 loops.
- Greedy and sampled decoding, request seeds, EOS (`2`, `11`) and length
  stopping through the existing engine, covered by
  `test_engine_stops_on_the_configured_eos_token`.
- CUDA execution on the Triton backend, including sliding-window masking; `auto`
  selects Triton because the Flash adapters do not implement masking.
- CUDA Graphs (`ExecutionConfig(cuda_graphs=True)`) for decode: the input
  stage, every recurrent loop and the output stage each replay one graph per
  batch size. The window is a constant of the captured attention kernel; page
  tables and context lengths stay graph inputs. Prompt prefill runs eager,
  because its row count follows the prompt rather than the batch.
- Ordinary asynchronous scheduling (`ExecutionConfig(async_scheduling=True)`),
  with or without graphs. Dory's loop count is fixed, so no exit score is read
  back and any exit mode is accepted; the sampled token stays on the device for
  the next input stage and only its readback is deferred. The two switches are
  independent: sync eager, sync Graph, async eager and async Graph are tested
  separately.
- Self-speculation in block and token-pipeline mode, in the same four execution
  modes. Asynchronous rounds take greedy requests only. See
  [Self-speculation](#self-speculation).
- The official tokenizer, including the Hub's `tokenizer.json` and chat
  template, resolved without requiring transformers to accept the checkpoint
  config (its generic validator rejects Dory's `mlp` sublayer entries).

## Packed Triton prefill

With `attention_backend="triton"`, the default `packed_prefill=None` automatically
enables packed prefill when execution and backend support it. Set
`ExecutionConfig(packed_prefill=True)` / `--packed-prefill` to require support,
or `packed_prefill=False` / `--no-packed-prefill` to keep per-query prefill.
An unsupported explicit enable request fails at startup.

This path uses BF16, `per_loop` KV, ordinary prefill, and synchronous eager
single-stage execution. Speculation, preemption, CUDA graphs and asynchronous
scheduling are outside its scope.

On NVIDIA CUDA, the packed path uses a 32 × 32 tile. At startup, Triton compiles
the full-attention kernel and, if the model uses a sliding window, the
sliding-window kernel. It checks their shared-memory requirements against the
device limit. Automatic mode uses per-query attention if the check fails;
explicit enable requests fail through backend capability validation.
The result is cached by device and attention geometry within the process.
The check runs before KV memory planning and uses tensor metadata only.
The check establishes that the tile fits, not that it is fastest. The fixed
32 × 32 tile was validated on RTX 4060 and RTX 5090; other GPUs and workloads
need measurements to determine their best tile.
The full BF16 checkpoint and long-context service validation require enough
device memory for the model weights, runtime buffers and KV pool.

A batch uses packed attention across PRELUDE, recurrence and the CODA
writes for prompt history when every prompt is longer than 64 tokens.
The CODA that samples the first token uses per-query attention. A batch containing
a shorter prompt or a 64-token prompt uses per-query attention. The
cutoff uses the full prompt length. Decode keeps the per-query path.

Packed full and sliding-window attention use the same causal ranges as
per-query attention. The matrix products and softmax reductions use a
different order and intermediate rounding. The value product compensates
for the BF16 rounding of softmax weights. Remaining differences can
propagate through Dory's recurrent layers and change a greedy choice.
For reproducible comparisons, use per-query attention and hold the
batch shape fixed. Changes in arrival timing can change the batch shape
even with per-query attention.

## Rejected today

These fail fast rather than degrade silently:

- WCPB, mixed or staged prefill, and PD.
- Prefix caching.
- Self-speculation combined with preemption or `last_loop` KV, sampled
  token-pipeline speculation, and sampled asynchronous rounds. The other forms
  are supported; see below.
- Per-request loop counts other than the configured full depth: Dory has no
  trained exit gate, so the runtime refuses to invent one, including via
  `min_loops` with `exit_threshold=0`.

Preemption and snapshots are **not qualified** rather than enforced: the
generic KV snapshot mechanism copies every plane, so it may well work, but no
restoration test exists for Dory's boundary state. Keep `enable_preemption`
off until 2E covers it.

Sliding-window masking exists in the Torch reference backend and in the Triton
decode kernel (`window=` on `KVCacheManager.attend`). Backend selection requires
the `sliding_window` capability, so a Flash-backed Dory engine fails at
construction instead of attending full history. Masking accepts host-planned,
graph-private and device-owned metadata; storage reclamation of out-of-window
pages stays with 2B.

## Self-speculation

`SpeculativeConfig(K, draft_loops=d, target_loops=T, mode=...)` drafts with the
first `d` recurrent loops and verifies with all `T`, where `T` is the loop count
the model was loaded with. It needs `per_loop` KV and fixed-depth requests
(`min_loops == max_loops == T`). Qualified depth pairs are 1→2, 1→4 and 2→4.
This section describes synchronous eager rounds first;
[asynchronous rounds and CUDA Graphs](#asynchronous-rounds-and-cuda-graphs) keep
the same KV contract.

| Mode | Sampling | Round |
| --- | --- | --- |
| `block` | greedy, or temperature/top-k/top-p with rejection sampling | `K` draft tokens one after another, then one verification of `K + 1` positions |
| `pipeline` | greedy | every submission advances each retained token by `d` loops; `d` must divide `T` |

What a draft shares with the target, and what it does not:

- **Input stage.** Prelude KV depends only on the input tokens. A candidate's
  prelude is the target's prelude whenever the candidate is accepted, so it is
  computed once.
- **Recurrent loops.** With `per_loop` KV, loop `i` of a position reads only
  loop `i` of earlier positions. The draft's loops `0..d-1` are the target's
  first `d` loops; verification continues from the retained state with loops
  `d..T-1`.
- **Output stage.** CODA attention reads the state it is given, so draft CODA
  KV (after `d` loops) differs from target CODA KV (after `T` loops) at the
  same position. A draft CODA stores at its own position, which is never below
  the committed frontier, and the target CODA of that position stores over it
  before any target row attends it. Target rows therefore read target history
  only. Draft rows read target history for committed positions and draft
  history for positions still in flight; that affects acceptance, never the
  committed output.
- **Rollback.** After a rejection the scheduler cuts every plane, boundary
  layers included, back to the committed frontier (`truncate_suffix`); the next
  round stores over the stale slots.

Sliding-window attention takes one query row per token, so verification uses
per-row attention rather than a packed call.

**Round metadata.** A block round prepares its metadata on the host before any
GPU work. One `MetadataBuilder.build` covers every row of the round (each
request's committed token plus its `K` candidates, grouped by request in
position order) at every target depth. Each operation stages a selection of
that layout (`MetadataBuilder.select_rows`):

- draft offset `j`: the requests still drafting at `j`, at depths `0..d-1`
  (depth 0 also serves the prelude);
- draft CODA: the subset that emits a candidate at `j`, at depth 0. The same
  `coda_rows` index both the CODA hidden rows and their KV metadata;
- verification: every row at depths `d..T-1`, packed when the attention backend
  declares `packed_matches_decode` and the model allows packed queries;
- target CODA: every row at depth 0.

Selections keep the caller's row order and take positions and write addresses
from the source layout; attention tables are re-planned for the rows selected.
All selections go into one upload per dtype, and `_prepare_inputs` returns them
as a `PreparedRound`. Only `PreparedRound.layouts` stores bound KV metadata.
Each draft step's `views` maps PRELUDE, RECURRENT and CODA to indices in that
store; `target` uses the same index contract for verification and target CODA.
There are no separately owned CODA or verification metadata fields. Repeated
row/depth/packed selections reuse the same layout index, including a full-row
draft CODA and its PRELUDE/recurrent depth-0 view. Empty selections have no
indices. `coda_rows` still selects the matching hidden rows.

Layout tensors borrow the runner's typed upload snapshots until the submission's
final consumer completes. The pipeline mode uses the same preparation-local
layout store and operation indices for its new-token, recurrent and endpoint
views; all views are bound before execution.

### Asynchronous rounds and CUDA Graphs

`ExecutionConfig(async_scheduling=True)` submits the next round before the host
reads the previous one (see `docs/async-speculative.md`). The actual positions
then exist only on the device. Thus the runner builds device metadata
(`DeviceKVMetadata`) at depth 0 for PRELUDE and CODA, as it does for the
recurrent loops. The KV contract does not change: all KV below the start of the
next round is the serial target history.

- **Block rounds.** Each draft offset gives its rows one depth-0 view for
  PRELUDE and the draft CODA. A partial draft CODA gets its own view of the
  drafting rows. Verification gives every row one depth-0 view for the target
  CODA.
- **Pipeline steps.** Every row runs both boundary stages, so each call keeps one
  shape (the fixed-slot calls of `docs/async-speculative.md`). Write masks select
  the rows that store KV. PRELUDE stores only for rows at depth 0. CODA stores
  only for rows that reach the draft or the target depth. A row between these
  depths (1→4) keeps the CODA KV of its draft pass, as in a synchronous step.
  Both stages use one depth-0 view and differ only in the write mask, as in
  synchronous steps and asynchronous block rounds.
  The runtime discards the PRELUDE output of rows that are not at depth 0, and
  the CODA output of rows that reach neither depth.
- **Sampling.** Asynchronous rounds take greedy requests only. Asynchronous
  block rounds can sample for models without boundary KV, but that combination
  is not qualified for Dory. Such a request fails when it is added.

The asynchronous pipeline keeps all its slots full: when the root reaches the
target depth, the newest draft row gets its successor in the same step. The
synchronous pipeline appends only inside the scheduled span. The two modes
therefore run different step sequences, and their acceptance can differ. Both
commit the target tokens.

`ExecutionConfig(cuda_graphs=True)`:

- **Asynchronous rounds** replay bucketed graphs for PRELUDE, every recurrent
  loop and CODA (`DeviceBoundaryGraphs`, `DeviceRecurrentGraphs`). Positions,
  write addresses, write masks and context lengths are graph inputs. The engine
  captures all buckets at construction. A row count above the largest bucket
  runs eager.
- **Synchronous rounds** use the host-planned boundary graphs of ordinary decode
  (`PreludeGraphs`, and `CodaGraphs` with the cache). These capture one graph
  per row count, up to `cuda_graph_max_graphs`; other row counts run eager.

## Validation status

`tests/test_dory.py` and `tests/test_dory_gpu.py` hold the regression suites;
`tests/test_dory_real_checkpoint.py` holds the pinned-weight run. Measured
results are below, including what they do not cover.
### CPU suite

`tests/test_dory.py` (44 cases) covers configuration validation, strict loading
with single-file and sharded checkpoints, stage/KV contracts, sliding-window
masking, KV layer compaction, engine scheduling, EOS stopping, and numerical
agreement with an independent dense oracle (`tests/reference/dory.py`) for both
KV modes — including greedy generation against that oracle. The full CPU run is
945 passed with only the pre-existing failures of the reviewed baseline
(`test_ouro` packed prefill, one `test_pd`, eleven `test_serving`).

### CUDA

`tests/test_dory_gpu.py` (66 cases) checks the Triton backend against the Torch
reference: windowed decode for `head_dim` 8/128/256, windows 1/3/16/64 and
lengths 1/7/63/64/65; unwindowed decode unchanged; both KV modes against the
dense oracle; a greedy engine run whose tokens match the oracle; and
`plan_cache`'s CUDA profiling branch, which exercises the windowed boundary
path.

### Real checkpoint

`tests/test_dory_real_checkpoint.py` runs the pinned snapshot
`f6c832e6963ddb74f62c75813a3e8c74b1af67b1` on one `A100-SXM4-80GB`
(torch 2.13.0+cu130, triton 3.7.1, Python 3.12.15) with the prompt
"The capital of France is" (5 tokens):

| Loops | Top-1 | Top-20 overlap | max abs Δlogit, BF16 GPU vs FP32 CPU | Greedy 4 tokens |
| --- | --- | --- | --- | --- |
| 1 | equal | 20/20 | 0.0422 | identical |
| 2 | equal | 20/20 | 0.0656 | identical |
| 4 | equal | 20/20 | 0.0712 | identical |

Backend parity on the same weights: `max|Δlogit|` 2.8e-5 in FP32 (kernel
arithmetic only) and 0.172 in BF16 (rounding dominates). Generation stops with
`finish_reason=length` at `max_tokens=4`, `exit_depths=[2,2,2,2]`, and releases
the KV allocation.

**The reference is the dense oracle, not vLLM's executable.** The installed vLLM
predates Dory, so a runtime-to-runtime comparison was not possible on this host;
the oracle is an independent implementation of the pinned upstream math, which
is weaker than comparing two runtimes. It is recorded here rather than implied
to be complete.

### Execution modes

`tests/test_dory_gpu.py` adds 36 cases on a small FP32 model. For each of sync
eager, sync Graph, async eager and async Graph, with 1, 2 and 3 concurrent
requests of different prompt lengths, the generated tokens equal the dense
oracle's, and the Graph modes must replay the input-stage, recurrent and
output-stage graphs. Cancellation (before the first step, after prefill, in the
middle of decode) and a failed submission are checked in every mode: the KV
allocation returns to zero and a following request still matches the oracle.

Real checkpoint, one GH200 (torch 2.14.0+cu130, triton 3.8.0), BF16, two loops,
greedy, 64 generated tokens per request, prompts cut from tokenized Python
source, `max_num_batched_tokens=2048`. Each workload runs twice in one process
and the second pass is reported. Decode throughput in tokens/s, counted after
the last request's first token:

| Requests x prompt | sync eager | sync Graph | async eager | async Graph | Peak GiB |
| --- | --- | --- | --- | --- | --- |
| 128 x 468 | 688 | 2274 | 752 | 2356 | 44.8 |
| 64 x 937 | 384 | 1341 | 358 | 1843 | 26.4 |
| 32 x 1875 | 198 | 775 | 217 | 1054 | 26.4 |
| 16 x 3750 | 122 | 512 | 135 | 572 | 26.4 |
| 8 x 7500 | 64.6 | 288 | 69.2 | 318 | 26.1 |
| 4 x 15000 | 35.1 | 152 | 31.8 | 166 | 25.9 |
| 2 x 32256 | 15.8 | 77.6 | 18.0 | 85.2 | 25.9 |
| 1 x 32256 | 9.4 | 41.2 | 8.5 | 44.8 | 16.9 |

Graphs raise decode throughput 3.3x to 5.3x and cut the median inter-token
latency from 107-182 ms to 22-56 ms; no graph call fell back to eager. Async
alone is within run-to-run noise of sync in eager mode (0.9x to 1.1x) and adds
4% to 37% on top of Graphs. Time to first token is prompt prefill, which stays
eager and does not change with the mode: 7.2-7.9 s for 128 x 468, 25.6-25.8 s
for 4 x 15000, 48.7-49.7 s for 2 x 32256. These are single runs on a shared
host.

Token agreement on the real checkpoint: in FP32 the four modes generate
identical tokens for every request (32 x 937 and 64 x 468, the second with the
async Graph run batching its decode waves differently from the sync runs). In
BF16 the four modes are identical up to 16 requests and at every prompt length
measured there; from 32 requests on, sync eager, sync Graph and async eager
still agree while async Graph differs, because it merges the two prefill
cohorts into one decode wave and BF16 greedy tokens depend on batch
composition. FP32 is therefore the exact check; BF16 agreement is not promised
across batch shapes.

### Self-speculation

`tests/test_dory_speculative.py` (54 CPU cases) checks block and token-pipeline
speculation for the depth pairs 1→2, 1→4 and 2→4 against the dense oracle:
greedy tokens for ragged batches, a forced rejection at every draft position,
all-accepted rounds with the bonus token, EOS inside an accepted prefix, output
and context limits, cancellation with ID reuse, cleanup after a failed round,
seeded sampling replay, the rejection-sampling output distribution, and the
rejected combinations. After every round the committed KV equals the serial
target history and nothing is written above the committed frontier. On one
GH200 the file passes together with `tests/test_dory_gpu.py` and
`tests/test_speculative.py` (251 passed; the only failures are 12
`flash_attn_4` cases whose optional package is not installed there).

Real checkpoint, Triton, synchronous eager, greedy, WikiText prompts, one GH200
(torch 2.14.0+cu130). Speculative tokens are compared with the non-speculative
baseline of the same process:

| Dtype | Loops | Shape (requests × prompt + generated) | Arms | Requests with identical tokens |
| --- | --- | --- | --- | --- |
| FP32 | 2 | 8 × 512 + 64 | block 1→2 K=1/3/4, pipeline 1→2 K=1/3 | 8/8 every arm |
| FP32 | 4 | 8 × 512 + 64 | block and pipeline, 1→4 and 2→4, K=3 | 8/8 every arm |
| FP32 | 2 | 32 × 937 + 128 | block 1→2 K=3, pipeline 1→2 K=1/3 | 32/32 every arm |
| FP32 | 4 | 16 × 512 + 96 | block and pipeline, 1→4 and 2→4, K=3 | 16/16 every arm |
| BF16 | 2 and 4 | 8 × 512 + 64 | same arms as the first two rows | 3–6 of 8 per arm |

BF16 is not token-exact, and that is not specific to speculation: serving the
same eight BF16 requests without speculation in waves of four or one instead of
eight also leaves only 2–4 of 8 requests token-identical, while the FP32
baseline is identical across wave sizes (32/32). Speculation changes the batch
shapes of the same per-row arithmetic, BF16 rounding then differs, and greedy
near-ties flip; every later token of that request diverges. FP32 is therefore
the exact check, and BF16 output is a valid greedy continuation up to rounding,
not a bit-reproduction of the baseline.

Acceptance (accepted / drafted, BF16): 1→2 0.78–0.93, 2→4 0.90–0.98, 1→4
0.72–0.93.

Whole-run wall time in seconds (all requests submitted at once, prefill
included; BF16; one pinned host core; arms interleaved, mean of two repetitions
except where marked †; decode-phase committed tokens/s in parentheses):

| Loops | Shape | Baseline | Block d=1 K=3 | Pipeline d=1 K=3 | Pipeline d=1 K=1 |
| --- | --- | --- | --- | --- | --- |
| 2 | 128 × 468 + 128 | 34.5 (697) | 31.9 (1133) | 23.0 (1249) | 27.1 (948) |
| 2 | 64 × 937 + 128 | 37.6 (330) | 34.8 (618) | 24.0 (678) | 28.2 (488) |
| 2 | 32 × 1875 + 128 | 39.5 (171) | 38.4 (298) | 25.4 (359) | 29.9 (256) |
| 2 | 8 × 7500 + 128 | 41.3 (45) | 31.9 (83) | 29.8 (85) | 33.0 (66) |
| 2 | 1 × 8000 + 128 | 18.3 (8.3) | 17.6 (8.9) | 12.1 (13.2) | 14.8 (10.3) |
| 2 | 2 × 32000 + 96 | 60.1 (12.9) | 52.6 (19.9) | 49.7 (24.8) | 52.2 (18.6) |

| Loops | Shape | Baseline | Block d=1 K=3 † | Block d=2 K=3 | Pipeline d=1 K=3 † | Pipeline d=2 K=3 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 64 × 937 + 96 | 49.0 (197) | 39.8 (496) | 47.1 (412) | 33.0 (397) | 32.1 (470) |
| 4 | 8 × 7500 + 96 | 59.9 (25.5) | 44.9 (58) | 46.8 (53) | 45.3 (50) | 42.8 (61) |
| 4 | 1 × 8000 + 96 | 23.3 (5.0) | 14.5 (9.2) | 17.7 (7.1) | 15.9 (8.1) | 15.4 (8.4) |

Reading the numbers:

- The pipeline is the faster mode in every two-loop shape (wall time −25 % to
  −39 % up to 8K prompts). Block speculation gains little wall time at high
  concurrency (−3 % to −8 % for 32–128 requests): its first round runs K draft
  steps before the first token is delivered, which adds 5–11 s to the time until
  every request has a first token, so its decode-phase rate overstates the
  benefit and wall time is the figure to use.
- At 2 × 32000 the run is dominated by ~42 s of prompt processing that
  speculation does not touch; the generated part is short (96 tokens).
- Single runs vary: the two single-request block repetitions were 21.2 s and
  13.9 s, the baseline 20.1 s and 16.4 s. Eager Dory is bound by host-side
  per-step cost, which is why fewer engine steps per committed token helps at
  all; these ratios should not be carried over to a graph-captured path.
- The longest context exercised is 32,096 tokens (two requests). Peak allocated
  memory was 74–82 GiB with the automatically sized KV pool; every arm released
  its KV blocks.

### Asynchronous self-speculation and CUDA Graphs

`tests/test_dory_speculative_gpu.py` (97 CUDA cases, small FP32 model, Triton)
checks the execution modes against the dense oracle:

- Greedy tokens for ragged requests in block and pipeline mode, for 1→2, 1→4 and
  2→4 and K = 1 and 3, in async eager, async Graph and sync Graph.
- A forced rejection at every draft position of an async block round, and forced
  acceptance patterns of an async pipeline. After every collected round, every
  plane below the committed frontier equals the serial target history while the
  next round is in flight.
- After every async pipeline step, a retained row keeps its PRELUDE KV and a row
  between the draft and target depths keeps the CODA KV of its draft pass, bit
  for bit.
- No host read of a device value during submission, cancellation with two
  rounds in flight and ID reuse, cleanup after a failed round, and the greedy
  and preemption rejections.

The Graph cases also check that all three graph families replay; async Graph
rounds have no eager fallback. On one H100 (torch 2.15.0+cu132, Triton 3.9), the
file passes together with `tests/test_dory_speculative.py`,
`tests/test_dory_gpu.py`, `tests/test_speculative.py`, `tests/test_cuda_graph.py`
and the `tests/test_async_*` files. The `flash_attn` cases were not selected,
because that package is not installed there.

Real checkpoint (pinned revision), K = 3, greedy, Triton, one H100, torch
2.15.0+cu132. Prompts are disjoint windows of the WikiText-103 test split.

**FP32 token agreement.** 32 requests × 512 prompt tokens + 128 generated
tokens. Every speculative arm (sync and async eager pipeline and block, sync
Graph pipeline, async Graph pipeline and block) is compared with synchronous
eager decoding without speculation at the same loop count:

| Loops | Draft → target | Requests with identical tokens, every arm | Acceptance, pipeline | Acceptance, block |
| --- | --- | --- | --- | --- |
| 2 | 1 → 2 | 32/32 | 0.85 | 0.85 |
| 4 | 1 → 4 | 32/32 | 0.81 sync, 0.73 async | 0.85 |
| 4 | 2 → 4 | 32/32 | 0.93 | 0.92 |

At 1 → 4 the asynchronous pipeline runs a different step sequence (see
[Asynchronous rounds and CUDA Graphs](#asynchronous-rounds-and-cuda-graphs)),
so its acceptance differs. Its tokens do not.

**GSM8K, BF16.** The first 200 test questions, 8-shot prompt from the first
eight training examples, at most 256 generated tokens. An answer is correct
when the number after `####` equals the reference:

| Arm | Correct |
| --- | --- |
| Baseline: async Graph, no speculation | 161/200 |
| Async Graph pipeline | 164/200 |
| Async Graph block | 163/200 |

BF16 tokens depend on the batch composition, so most sequences differ from the
baseline after some token. Accuracy does not drop.

**BF16 wall time.** Two loops, a one-loop draft. All requests are sent at the
same time. Each arm runs one warmup, then five timed repetitions, all in one
container. Wall times are the mean whole-run time in seconds, prefill included.
Every Graph arm uses `cuda_graph_max_graphs=128`, so no arm runs eager because
its graph cache is full; every Graph arm had zero graph fallbacks. BF16
acceptance is 0.85–0.88. Tables 1–4 were measured before PRELUDE and CODA of a
pipeline step shared one depth-0 view.

Table 1: with CUDA Graphs, concurrency sweep. Prompt 1024 tokens, 128 generated
tokens for each request. Percentages are against the baseline.

| Requests | Baseline (async Graph, no spec) | Baseline, default graph cache | Sync Graph pipeline | Async Graph pipeline | Async Graph block |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.50 | 2.49 (−0.1 %) | 1.83 (−27 %) | **1.78 (−29 %)** | 1.91 (−23 %) |
| 4 | 3.17 | 3.18 (+0.4 %) | 2.66 (−16 %) | **2.63 (−17 %)** | 2.95 (−7 %) |
| 16 | 5.89 | 5.90 (+0.1 %) | 5.51 (−6 %) | **5.30 (−10 %)** | 5.63 (−5 %) |
| 64 | 15.60 | 25.72 (+65 %) | 16.00 (+3 %) | **15.10 (−3 %)** | 16.45 (+5 %) |

Table 2: latency with CUDA Graphs, same workload. TTFT is the time to the first
token; TPOT is the time per output token after the first token. Percentages are
against the baseline at the same concurrency.

| Requests | Arm | TTFT p50 (s) | TTFT p90 (s) | TPOT p50 (ms) | TPOT p90 (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline (async Graph, no spec) | 0.18 | 0.18 | 18.3 | 18.3 |
| 1 | Async Graph pipeline | 0.17 (−2 %) | 0.20 (+13 %) | 12.6 (−31 %) | 12.6 (−31 %) |
| 1 | Async Graph block | 0.17 (−1 %) | 0.19 (+7 %) | 13.7 (−25 %) | 13.7 (−25 %) |
| 4 | Baseline (async Graph, no spec) | 0.67 | 0.67 | 20.0 | 20.0 |
| 4 | Async Graph pipeline | 0.66 (−1 %) | 0.67 (−0.8 %) | 16.6 (−17 %) | 17.4 (−13 %) |
| 4 | Async Graph block | 0.69 (+3 %) | 0.69 (+3 %) | 18.1 (−10 %) | 19.0 (−5 %) |
| 16 | Baseline (async Graph, no spec) | 2.01 | 2.75 | 35.2 | 40.9 |
| 16 | Async Graph pipeline | 1.68 (−17 %) | 2.70 (−2 %) | 28.8 (−18 %) | 36.1 (−12 %) |
| 16 | Async Graph block | 1.79 (−11 %) | 2.93 (+6 %) | 30.1 (−15 %) | 36.1 (−12 %) |
| 64 | Baseline (async Graph, no spec) | 6.21 | 10.58 | 76.6 | 108.5 |
| 64 | Async Graph pipeline | 5.78 (−7 %) | 10.02 (−5 %) | 70.9 (−7 %) | 102.5 (−5 %) |
| 64 | Async Graph block | 6.43 (+3 %) | 11.54 (+9 %) | 68.1 (−11 %) | 96.4 (−11 %) |

Table 3: without CUDA Graphs (eager), sync against async, same workload. There
is no eager baseline without speculation, so each async percentage is against
the sync value of the same mode.

| Requests | Pipeline sync | Pipeline async | Block sync | Block async |
| --- | --- | --- | --- | --- |
| 1 | 6.80 | 6.42 (−6 %) | 6.83 | 7.16 (+5 %) |
| 4 | 7.80 | 8.24 (+6 %) | 9.64 | 9.71 (+0.8 %) |
| 16 | 10.74 | 10.57 (−2 %) | 13.24 | 13.25 (+0.1 %) |
| 64 | 21.22 | 20.49 (−3 %) | 28.02 | 27.29 (−3 %) |

Table 4: long context. 8 requests, prompt 8192 tokens, 128 generated tokens for
each request.

| Arm | Wall (s) | TTFT p50 (s) | TTFT p90 (s) | TPOT p50 (ms) | TPOT p90 (ms) |
| --- | --- | --- | --- | --- | --- |
| Baseline (async Graph, no spec) | 23.73 | 18.60 | 20.71 | 53.2 | 68.9 |
| Sync Graph pipeline | 23.52 (−0.9 %) | 17.85 (−4 %) | 20.73 (+0.1 %) | 51.2 (−4 %) | 72.5 (+5 %) |
| Async Graph pipeline | 23.18 (−2 %) | 17.69 (−5 %) | 20.58 (−0.6 %) | 49.4 (−7 %) | 70.3 (+2 %) |
| Async Graph block | 23.13 (−3 %) | 17.60 (−5 %) | 20.42 (−1 %) | 49.9 (−6 %) | 67.9 (−1 %) |
| Sync eager pipeline | 28.83 | 18.04 | 21.08 | 86.1 | 111.4 |
| Async eager pipeline | 28.52 (−1 % vs sync) | 18.10 (+0.3 % vs sync) | 21.15 (+0.3 % vs sync) | 87.2 (+1 % vs sync) | 108.0 (−3 % vs sync) |

Reading the numbers:

- The gain of speculation falls as concurrency rises. At one request the GPU
  has spare capacity, so draft and verify work is almost free. At 64 requests
  the GPU is busy, and the extra verify rows cancel most of the saved steps.
- In pipeline mode, every row runs both boundary stages, so that each call keeps
  one shape. PRELUDE also runs on rows that already hold their input-stage KV,
  and CODA also runs on rows that reach neither the draft nor the target depth.
  The runtime discards these outputs. At 64 requests, this extra work can
  explain a part of the small gain.
- At 8 × 8192, prefill takes about 18 of 24 seconds. Speculation shortens only
  decode, so the whole run is 2–3 % faster.
- The baseline with the default graph cache (16 shapes) runs eager in most calls
  at 64 requests and takes 25.72 s instead of 15.60 s. Comparisons against that
  baseline overstate the gain of speculation.
- These runs use a different harness and host than the 2D and 3A tables above,
  so the absolute numbers do not compare with those tables.
- Not covered: HTTP serving and open-loop arrival traces.

**Known issues.**

- Async Graph block is 5 % slower than the baseline at 64 requests. Its time per
  output token is 11 % lower, but its time to the first token is 3–9 % higher.
  The async Graph pipeline is 3 % faster there, so it is the better mode at high
  concurrency.
- Without CUDA Graphs, async rounds at 1 and 4 requests are between 6 % faster
  and 6 % slower than sync rounds (Table 3). A dedicated run alternated a sync
  and an async engine in ABBA order, 10 timed repetitions each, at one request
  in pipeline mode. Async was 2.7 % and 3.9 % slower in two containers. The
  profiler counted 5503 kernel launches per generated token for sync and 5626
  for async (+124, +2.2 %). Async rounds keep positions on the GPU, so the GPU
  computes the KV metadata, the PRELUDE and CODA write masks and the slot-pool
  update; sync rounds do this work on the CPU. Eager Dory is bound by host-side
  launch cost, so the extra launches give the extra time. At 16 and 64 requests,
  async eager is 0–3 % faster than sync eager.
- After PRELUDE and CODA of a pipeline step shared one depth-0 view, async makes
  5615 launches per generated token (+113 against sync). The same ABBA run gave
  −0.2 % and +0.4 % in two new containers, and async was slower in 5 of 10
  rounds in each. The time between containers varies by a few percent, so these
  runs do not show that the gap is closed. A fused kernel for the per-step
  update and metadata would remove more launches. It also changes the Ouro path,
  so it is a separate change.

### Not yet performed

- A runtime-to-runtime comparison against vLLM once a Dory-capable build is
  available.
- Long-context behaviour beyond 32K tokens: the longest measured prompt is
  32,256 tokens, far below the artifact's 256K configuration limit, which is
  not a qualified deployment length. Dense-oracle logit parity is still only
  qualified on the 5-token prompt above.
- Windowed Flash-backed kernels.
- Graph-captured prefill, open-loop arrival traces and sampled decoding on the
  real checkpoint in Graph or async mode. The measurements above are
  closed-loop, greedy, and their time to first token is eager prefill.
- Self-speculation with preemption, sampled asynchronous rounds, sampled
  token-pipeline, adaptive exit, or a draft width above the fixed K.
- Sampled speculation on the real checkpoint (the rejection-sampling contract is
  qualified on the tiny CPU model only).
