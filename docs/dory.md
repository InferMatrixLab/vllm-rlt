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
replicated per loop. In `last_loop` mode a token's KV is overwritten by each of
its loops, so the runtime completes every loop of a token before the next one,
also inside a prefill chunk: later tokens read final-loop KV whatever the chunk
size. `per_loop` preserves the same mathematical
computation for token-at-a-time and all-positions-at-once execution; floating-point
rounding can still vary with chunk and batch shapes.

Only the 36 attention sublayers own KV. The cache pool is sized by that compact
count (`DoryConfig.num_kv_layers`), not by all 72 physical sublayers, and
`StageEngine`/`plan_cache` select it through `num_kv_layers` when a model
declares one. Attention layers therefore reach the cache through a packed
layer id rather than their physical sublayer index.

## Grouped storage and optional SWA reclamation

Dory's ordinary synchronous eager execution uses one pool of attention-layer
pages, grouped by window policy and boundary/recurrent plane domain.
`reclaim_swa` defaults to `False`. To return expired sliding-window pages to the
pool, enable it together with incremental allocation on an already loaded model:

```python
from vllm_rlt import CacheConfig
from vllm_rlt.config import ExecutionConfig
from vllm_rlt.engine.llm_engine import LLMEngine

engine = LLMEngine(
    model,
    attention_backend="torch",
    execution_config=ExecutionConfig(use_uva=False),
    cache_config=CacheConfig(
        num_blocks=256,
        layout=model.kv_cache_layout,
        incremental_allocation=True,
        reclaim_swa=True,
    ),
)
```

`num_blocks` preserves the existing all-layer KV byte budget: the native pool
converts it to `num_blocks * model.config.num_kv_layers` single-layer pages.
Reclamation supports ordinary synchronous fixed-loop `ouro` execution. Each
`per_loop` plane advances independently; shared recurrent history in
`last_loop` advances only after the token's final loop. Pages retire only after
all layers in their group have consumed the range and the submission completes.
Full-attention history remains resident.

CUDA Graphs and asynchronous scheduling borrow device-owned KV metadata;
self-speculation needs suffix truncation after a rejection; packed prefill does
not record which SWA positions it read; FP8 KV keeps per-layer scales. Grouped
pages do not support these yet. In all five cases the engine uses all-layer
blocks, including for prefix caching, and rejects `reclaim_swa`.

Native SWA attention stages only the needed page-aligned read span. Logical token
positions, write addresses and public layer maps remain absolute; the backend's
read-table context lengths are local to that span. Native attention calls must
match the layer's declared window, including `None` for full-attention layers.

For read-only group policy, `cache.layer_group(kv_layer_id)` returns
`(group_id, policy)`: a manager-local group ID and the immutable `KVGroup` with
compact layer IDs, window, recurrent domain and plane count. The layer ID is a
compact KV index, not a physical sublayer index.

For read-only cache inspection, `cache.retained_range(request_id, kv_layer_id,
depth)` returns `(resident_start, written_end)` in absolute token positions.
`cache.layer_block_table(request_id, kv_layer_id, depth)` returns the absolute
logical-page map, with `-1` for reclaimed pages. Here `cache` is
`engine.cache_manager`, and `kv_layer_id` is a compact KV index. Reading a
reclaimed full prefix is rejected; these queries do not retain page references
or expose mutable allocator state.

Grouped prefix caching publishes one entry per token page holding every
owned (group, plane, layer) page, ordered as the allocator splits a claim.
The entry takes its own reference, so the publishing request can finish or
release its pages without freeing the cached ones. Publication stops at the
first token page with any owned page missing; each step publishes its prompt
pages before retiring expired ones. With `reclaim_swa`, a hit takes
request references only to SWA pages still within the next token's window.
The prefix cache can hold older pages for reuse by shorter hits. Each request
is budgeted for its own peak KV use so sharing does not reserve less memory
than the requests may need as they advance.

## Supported today

- Native strict safetensors loading with the artifact's per-projection key
  names (`q/k/v/g/o_proj`, `gate_proj`/`up_proj`) and the learned scale tensors.
  There is no fused-key remapping: a checkpoint that packs `qkv_proj` or
  `gate_up_proj` fails loudly instead of loading approximately.
- Synchronous eager execution: ordinary bounded-chunk prefill followed by token
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
- The official tokenizer, including the Hub's `tokenizer.json` and chat
  template, resolved without requiring transformers to accept the checkpoint
  config (its generic validator rejects Dory's `mlp` sublayer entries).
- Prefix caching (`enable_prefix_caching=True`). In `per_loop` mode a
  published page carries boundary KV from plane 0 and recurrent KV from every
  plane; boundary layers never write deeper planes, so they are not required
  there. In `last_loop` mode the single shared plane holds each cached token's
  final-loop KV, which is what a cold prefill leaves; this mode is qualified
  for synchronous eager execution only. A hit restores exactly that coverage
  and the scheduler resumes prefill from the cached offset.
  Prefix keys hash tokens only. That is sound because the loop count is fixed
  when the model loads and the layout is fixed by `CacheConfig`, and each
  engine owns its own KV manager, so pages from different loop counts or
  layouts never share a cache. A cache shared across loop counts would have to
  add both to the key. Evicted prefixes fall back to a full cold prefill.

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
- Prefix caching in `last_loop` mode with CUDA Graphs or asynchronous
  scheduling.
- Preemption and snapshot/restore with grouped KV.
- Self-speculation combined with CUDA Graphs, asynchronous scheduling,
  preemption, `reclaim_swa` or `last_loop` KV, and sampled token-pipeline
  speculation. The synchronous eager form is supported; see below.
- Per-request loop counts other than the configured full depth: Dory has no
  trained exit gate, so the runtime refuses to invent one, including via
  `min_loops` with `exit_threshold=0`.

For ordinary decoding, the all-layer fallback used with CUDA Graphs or
asynchronous scheduling accepts `enable_preemption`. Dory preemption in these
modes has not been qualified on GPU.

Sliding-window masking exists in the Torch reference backend and in the Triton
decode kernel (`window=` on `KVCacheManager.attend`). Backend selection requires
the `sliding_window` capability, so a Flash-backed Dory engine fails at
construction instead of attending full history. Masking accepts host-planned,
graph-private and device-owned metadata. Physical reclamation of out-of-window
pages is the optional grouped path described above.

## Synchronous self-speculation

`SpeculativeConfig(K, draft_loops=d, target_loops=T, mode=...)` drafts with the
first `d` recurrent loops and verifies with all `T`, where `T` is the loop count
the model was loaded with. It needs `per_loop` KV, synchronous eager execution
and fixed-depth requests (`min_loops == max_loops == T`). Qualified depth pairs
are 1→2, 1→4 and 2→4.

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

## Validation status

`tests/test_dory.py` and `tests/test_dory_gpu.py` hold the regression suites;
`tests/test_dory_real_checkpoint.py` holds the pinned-weight run. Measured
results are below, including what they do not cover.

### CPU suite

`tests/test_dory.py` covers configuration validation, strict loading
with single-file and sharded checkpoints, stage/KV contracts, sliding-window
masking, KV layer compaction, engine scheduling, EOS stopping, and numerical
agreement with an independent dense oracle (`tests/reference/dory.py`) for both
KV modes — including greedy generation against that oracle. The full CPU run on
2026-11-16 passed 1109 tests with no failures; GPU-marked tests were skipped.

### Prefix reuse

Prefix reuse is checked at loops 1, 2 and 4: cold and cache-hit prefill
logits match the oracle, cached prefixes longer than the sliding window,
block-aligned prompts, diverging suffixes, concurrent sharers and a
re-request after its prefix was evicted all reproduce the cold greedy tokens,
and publication refuses pages missing any owned plane. Prefix pages stay
pinned for the life of the cache entry. Prefix tests run on both the legacy
all-layer blocks and grouped storage. On a small FP32 model, prefix hits and
diverging branches match the oracle in sync eager, sync Graph, async eager and
async Graph modes. The checks above are `per_loop`. `last_loop` checks the
cache-hit logits against the oracle on both storages, and block-aligned,
diverging and concurrent requests after a hit against the cold greedy tokens,
with prefill chunks of 3 and 4 tokens; on CUDA with Triton, a hit and a
diverging branch match the oracle with chunks of 3 and 16. With `reclaim_swa`,
in both modes, hits of 2 and 8 tokens match the oracle logits and reclamation
advances past the hit, including submissions longer than the window.

Real checkpoint, one L20, two loops, 16 greedy tokens, `prefill_chunk_size=256`,
prompts cut from the model card. Each cell compares prefix caching on and off in
a fresh engine; time to first token in seconds:

| Prompt tokens | Request | TTFT off | TTFT on | Tokens equal (BF16) |
| --- | --- | --- | --- | --- |
| 1100 | repeat | 15.3 | 0.30 | yes |
| 1100 | branch | 15.2 | 0.60 | yes |
| 4096 | repeat | 60.4 | 0.41 | yes |
| 4096 | branch | 60.3 | 0.78 | yes |

An evicted prefix falls back to a cold prefill with equal tokens, and a control
without eviction hits 1088 tokens. In BF16, one of four concurrent sharers and a
seeded sampled (temperature 0.8) repeat differed from the uncached run; in FP32
all of them match. A hit resumes prefill at the cached offset, which changes
the chunk split, and with caching off, changing `prefill_chunk_size` alone
already changes BF16 greedy tokens. FP32 tokens matched in these checks; BF16
agreement is not promised across chunk splits. These are single runs on a
shared host.

The same checks were repeated on one RTX PRO 5000 Blackwell (torch 2.13+cu130,
Torch backend), BF16, at loops 1, 2 and 4:

| Loops | Prompt tokens | TTFT off | TTFT on (repeat) |
| --- | --- | --- | --- |
| 1 | 1100 | 6.6 | 0.13 |
| 1 | 4096 | 27.9 | 0.20 |
| 2 | 1100 | 9.3 | 0.18 |
| 2 | 4096 | 40.9 | 0.29 |
| 4 | 1100 | 16.4 | 0.31 |
| 4 | 4096 | 69.9 | 0.49 |

Repeat tokens were equal everywhere; eviction and its 1088-token control
behaved as on the L20. At two loops and 1100 tokens in FP32, all eight
concurrent requests, the seeded sampled repeat, every `prefill_chunk_size`
from 16 to 256 and the cache hit produced identical tokens.

With the Triton backend at two loops and 1100 tokens, sync eager, sync Graph,
async eager and async Graph each produced the uncached tokens for the cold,
repeat and branch requests, with the hit cutting time to first token from
0.54-0.79 s to 0.06-0.10 s. Through the OpenAI-compatible HTTP server (1516
prompt tokens), every request succeeded and hits cut time to first token from
1.5-7 s to 0.09-0.35 s; the branch and one concurrent response differed in
BF16 text, which is the chunk-split effect above.

### CUDA

`tests/test_dory_gpu.py` checks the Triton backend against the Torch
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

`tests/test_dory_gpu.py` checks a small FP32 model. For each of sync
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
composition. FP32 tokens matched in these measured workloads; BF16 agreement is
not promised across batch shapes.

### Self-speculation

`tests/test_dory_speculative.py` checks block and token-pipeline
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
near-ties flip; later tokens of that request can then diverge. FP32 tokens matched
in these measured workloads; BF16 does not reproduce the baseline token for token
across these batch shapes.

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
- Self-speculation with CUDA graphs, asynchronous scheduling, preemption,
  sampled token-pipeline, adaptive exit, or a draft width above the fixed K.
- Sampled speculation on the real checkpoint (the rejection-sampling contract is
  qualified on the tiny CPU model only).
