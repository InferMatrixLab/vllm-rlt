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
- The official tokenizer, including the Hub's `tokenizer.json` and chat
  template, resolved without requiring transformers to accept the checkpoint
  config (its generic validator rejects Dory's `mlp` sublayer entries).

## Rejected today

These fail fast rather than degrade silently:

- WCPB, mixed or staged prefill; packed prefill metadata (`prefill_uva`); PD.
- Prefix caching.
- Self-speculation.
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

### Not yet performed

- A runtime-to-runtime comparison against vLLM once a Dory-capable build is
  available.
- Long-context behaviour beyond 32K tokens: the longest measured prompt is
  32,256 tokens, far below the artifact's 256K configuration limit, which is
  not a qualified deployment length.
- Windowed Flash-backed kernels.
- Graph-captured prefill, open-loop arrival traces and sampled decoding on the
  real checkpoint in Graph or async mode. The measurements above are
  closed-loop, greedy, and their time to first token is eager prefill.
