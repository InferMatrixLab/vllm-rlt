# vllm-rlt as a vLLM plugin (unverified on GPU)

This branch adds `vllm_rlt/plugin/`: the loop-aware parts of vllm-rlt running
inside vLLM's own engine process. Serving, tokenizer, executor, parallelism,
kernels, KV allocation, the sampler, the rejection sampler and structured output
are vLLM's; this package only adds what needs to know a token is a loop.

Development target: vLLM `0.34.0rc1.dev1149+g3146a9df4`. There is no runtime
version guard.

## Use

```bash
pip install -e . --no-deps          # registers the vllm.general_plugins entry point
vllm serve <model> \
  --scheduler-cls vllm_rlt.plugin.scheduler.LoopScheduler \
  --worker-cls vllm_rlt.plugin.worker.LoopWorker
# with async scheduling use LoopAsyncScheduler instead
```

Without those two options a vLLM run is unaffected, apart from `Request`
being the `LoopRequest` subclass (see below).

Loop features are switched on in `--additional-config` (JSON):

| key | effect |
| --- | --- |
| `rlt_split_ops` | Inkling: run prelude, each core pass and coda as separate ops (needed by the three below except self-spec) |
| `rlt_wcpb` | wavefront chunked prefill: chunks of one prompt run at different depths in the same step |
| `rlt_graphs` | per-op CUDA graphs for one-token decode steps |
| `rlt_draft_passes` | passes the self-speculative draft runs (default N/2) |
| `rlt_allow_mixed_depth` | let one CORE step hold rows at different depths (Ouro only) |
| `rlt_wait_limit`, `rlt_wcpb_depth` | starvation bound; chunks of a prompt in flight |

Self-speculation uses vLLM's speculative config with the plugin's method:

```bash
--speculative-config '{"method":"rlt_self","num_speculative_tokens":4,
                       "speculator_options":{"draft_passes":2}}'
```

Per-request parameters go in `vllm_xargs`: `rlt_exit_threshold`,
`rlt_min_loops`, `rlt_max_loops`, `rlt_target_loops`.

## Inkling: Fixed and VarLoop

Both share the architecture name `Inkling2LoopForConditionalGeneration`; the
plugin tells them apart by `text_config.loop_config.is_varloop`.

* **Fixed** (`is_varloop` false) was trained at exactly `num_passes` passes. Every
  token runs all of them; `rlt_target_loops` is ignored; self-speculation is
  refused (a shallow pass is not a usable draft).
* **VarLoop** (`is_varloop` true) was trained to run 1..N passes. The pass count
  of a deployment is `--hf-overrides '{"text_config": {"loop_num_passes": N}}'`.
  It supports self-speculation, and, when served with `loop_share_cache`, a
  per-request `rlt_target_loops` (all passes read and write one KV cache, so
  the last executed pass already stands in for the skipped ones).

Inkling has no exit gate. "Early exit" there means leaving after a configured
pass count, never a predicted exit like Ouro's.

## Layout

| file | role |
| --- | --- |
| `__init__.py` | entry point `register()` |
| `patch.py`, `request.py` | the one patch: replace `Request` with `LoopRequest` everywhere (as vllm-omni does); `Span` |
| `scheduler.py` + `sched_*.py`, `carry.py`, `plan.py` | `LoopScheduler(Scheduler)`, `LoopAsyncScheduler(LoopScheduler, AsyncScheduler)` |
| `program.py`, `output.py` | loop ops/steps, `LoopSchedulerOutput`, `LoopModelRunnerOutput` |
| `parallel.py` | DP alignment of the op kind (`choose_op`, `Aligner`), dummy loop steps |
| `exit.py`, `runner_exit.py`, `kvfill.py` | early-exit policy (Ouro gate), its evaluation, LAST_EXITED KV fill |
| `runner.py`, `worker.py` | `LoopModelRunner(GPUModelRunner)`, `LoopWorker(Worker)` |
| `loopgraph.py` | per-(op, depth) CUDA graphs on vLLM's `ModelCudaGraphManager` |
| `selfspec.py` | `rlt_self` speculative method: shallow-pass drafts, vLLM verifies |
| `ouro.py`, `ouro_layers.py`, `ouro_config.py`, `inkling.py`, `models.py` | models and their loop layout |

## How it works

* One EngineCore step is one loop step (mapping A). Every step holds one op
  kind: PRELUDE, CORE at one depth, CODA, or FULL.
* vLLM's `schedule()` still does admission, KV allocation, preemption and token
  accounting. It opens a token span; the loop scheduler walks the span through
  its ops while the request is hidden from vLLM's running list.
* `update_from_output` reaches vLLM only when a span's last op returns.
* A model vLLM already runs natively (Inkling) uses `LoopLayout.full_only` unless
  `rlt_split_ops`, `rlt_wcpb` or `rlt_graphs` is on: the scheduler passes straight
  through to vLLM, so with loop features off the baseline is vLLM itself.
* **Early exit (Ouro)** follows the reference: `remaining *= 1 - sigmoid(gate)`,
  exit when `1 - remaining >= threshold`, bounded by min/max loops. The runner
  decides on the device; a skipped pass gets the exit pass's KV copied slot by slot.
* **WCPB.** A prompt that still has chunks left stays visible to vLLM, so its
  next chunk opens while earlier chunks are mid-loop (up to `num_passes + 2`).
  A request has several spans; a step takes at most one per request. Chunk c may
  run an op only after chunk c-1 is past it (`plan.unblocked`), so the KV prefix
  at that depth exists. Each step tells the worker every chunk's start
  (`span_starts`) because vLLM's own counter is not in chunk order. WCPB needs
  one KV cache per pass, and turns mixed-depth CORE steps on for Ouro.
* **Self-speculation.** It is vLLM's speculative decoding with a custom speculator
  (`register_speculative_method`). vLLM schedules a K+1-token verify span, runs it
  at full depth, verifies with its rejection sampler and rolls back
  `num_computed_tokens` and output placeholders on rejection, in sync and async
  scheduling alike. `LoopSpeculator.propose` produces the next K drafts by running
  the model for `draft_passes` core passes one token at a time (greedy). The verify
  rewrites the shallow KV the draft wrote with identical values and fills the deeper
  passes. Draft and verify rows are the same kind of vLLM batch, so there is no
  separate coda batching to do: the verify span's final op is vLLM's own sampling.
* **CUDA graphs.** `loopgraph.py` captures one `ModelCudaGraphManager` per
  (op, depth) after vLLM's own capture, each with that op's loop context. The state
  that flows between ops goes through static buffers (`Sink` for the output,
  `GraphStore.inp` for the input), copied before and after each replay. Only
  uniform one-token decode steps replay; chunks, mixed depths and speculative spans
  run eagerly.

## Status

Tested on CPU (`tests/plugin`, 58 tests, `VLLM_SRC` pointing at a vLLM checkout of
the pinned commit): the loop program, planner (including chunk ordering and one
span per request), exit policy, DP alignment tables; the Fixed/VarLoop layouts;
state helpers (`TokenState`, `Sink`); the scheduler against the real vLLM
`Scheduler` (sync, async, WCPB with several chunks in flight); the `Request` patch
and entry point; the `rlt_self` registration; Ouro's model code on vLLM's CPU setup;
and the Inkling op partition.

**Not run on a GPU.** Nothing below has executed a kernel:

* `runner.py`, `runner_exit.py`, `worker.py` (the code that talks to vLLM's runner
  internals: `execute_model_state`, slot mappings, `_dummy_run`, the CUDA-graph
  manager, `num_computed_tokens` staging);
* `loopgraph.py`: capture order, the static buffers, and whether the attention
  metadata vLLM builds for a replay matches the capture;
* `selfspec.py`: the draft step builds its own attention metadata with
  `build_attn_metadata`; whether that is accepted for a one-token decode on every
  backend is untested. The draft's greedy tokens, rollback and KV rewrite are
  untested;
* `ouro*.py` numerics and `kvfill.py`;
* `inkling.py`: it calls vLLM's private `_run_layer`. Its first check on a GPU
  must be that running prelude, the core passes and coda as separate ops gives
  the same final hidden state as vLLM's single forward.

## Known gaps

* Inkling mixed depths: the AttnRes state differs per depth (block counts), so an
  Inkling CORE step holds one depth; `state_cat` refuses to merge otherwise. WCPB
  on Inkling therefore overlaps chunks of different requests, not of different depths
  in one step, until that state is made depth-independent.
* The exit decision is made on the device but read back by the scheduler, one
  step late under async scheduling; gated rows therefore advance synchronously.
* `LAST_EXITED` fill assumes a `(2, blocks, block_size, ...)` KV layout and
  raises on any other.
* The draft step is eager and one token at a time; a CUDA graph for it and a
  batched draft/verify CODA are not done.
* The decode pipeline (tokens of one request at different depths in one step) is not
  implemented. A span is one program walk.
* No data-parallel run: `DPOpAlign` and the idle-rank path are unexercised.
* No accuracy check of early exit against Ouro's reference.
