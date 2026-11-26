# vllm-rlt as a vLLM plugin

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
```

Without those two options a vLLM run is unaffected, apart from `Request`
being the `LoopRequest` subclass (see below). `LoopScheduler` builds itself as
`LoopAsyncScheduler` when vLLM's async scheduling is on (its default), so the
same `--scheduler-cls` works in both modes.

Options in `--additional-config` (JSON). The loop program always runs; these
only choose what the loop scheduler does on top of it:

| key | effect |
| --- | --- |
| `rlt_wcpb` | wavefront chunked prefill: chunks of one prompt run at different depths in the same step |
| `rlt_graphs` | per-op CUDA graphs for one-token decode steps |
| `rlt_draft_passes` | passes the self-speculative draft runs (default N/2) |
| `rlt_allow_mixed_depth` | **to be removed** (see "Decided, not yet in code"): mixed depth becomes the default for Ouro and Inkling |
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
  kind: PRELUDE, CORE at one depth, or CODA. Every model, Inkling included,
  runs as this program; no path hands a token to a model's single forward.
* Scheduler as it is in the code today (the subclass design; a reimplemented
  scheduler is preferred, see "Decided, not yet in code"): vLLM's `schedule()`
  does admission, KV allocation, preemption and token accounting. It opens a token span; the loop scheduler walks the span through
  its ops while the request is hidden from vLLM's running list.
* `update_from_output` reaches vLLM only when a span's last op returns.
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

Tested on CPU (`tests/plugin`, 61 tests, `VLLM_SRC` pointing at a vLLM checkout of
the pinned commit): the loop program, planner (chunk ordering, one span per
request), exit policy, DP alignment tables; the Fixed/VarLoop layouts; state
helpers (`TokenState`, `Sink`, mixed-depth merge); the scheduler against the real
vLLM `Scheduler` (sync, async, WCPB with several chunks in flight, late return of
an async step); the `Request` patch and entry point; the `rlt_self` registration;
Ouro's model code on vLLM's CPU setup; and the Inkling op partition.

### GPU runs (one B300, vLLM `0.34.0rc1.dev1149`, greedy, three prompts)

Scripts are in `tools/plugin/`: `smoke_ouro.py`, `smoke_inkling.py`, `hf_ref.py`, and the
benchmark `run_job.sh`, `queue.sh`, `bench.py`, `jobs_plugin.txt`.

**Ouro-1.4B** (12 tokens per prompt unless noted):

| run | result |
| --- | --- |
| HF transformers reference, bf16 and fp32 | identical ids in both dtypes |
| plugin, fp32, sync and async | identical to HF on all three prompts |
| plugin, bf16, sync | identical to HF on prompts 2 and 3; prompt 1 differs from token 8 (`216, 33` vs HF `216, 34`) |
| plugin, bf16, async | prompt 1 differs from token 5; prompts 2 and 3 identical to HF |
| vLLM's own `OuroForCausalLM`, bf16 (no plugin) | prompt 1 differs from token 5, prompt 2 from token 4, prompt 3 from token 1 |

fp32 matching HF exactly on every token, sync and async, is the evidence that the
loop scheduler, per-pass KV and runner compute the model correctly. The bf16
differences are most likely rounding: they disappear in fp32, and the sync and
async bf16 runs differ from each other in the same way. vLLM's native Ouro is
further from HF in bf16 than the plugin is.

| feature | result |
| --- | --- |
| self-speculation `rlt_self`, K=3, 2 draft passes, 32 tokens | 39 rounds, 59 of 117 draft tokens accepted (50%); the first 12 ids equal HF on all three prompts; sync and async give identical ids |
| early exit, threshold 0.5 | 287 gated tokens, 93 exited, 72 of them before the last pass and so filled; for all 72, layer 0 of the last pass holds the same KV as the exit pass (checked on the device) |
| WCPB, 200-token prompts in 64-token chunks | all 4 chunks of a prompt in flight at once, at different ops; 8 ids per prompt identical to the run without WCPB, sync and async |
| `rlt_allow_mixed_depth`, async | 8 ids per prompt equal HF except prompt 1's 8th (the same bf16 difference as the sync run); on long prompts identical to the run without it |
| `rlt_graphs` | five loop graphs (CORE 0..3, CODA) capture and replay; 12 ids equal the same configuration without loop graphs |

**Inkling-2 mini** (`/dev/shm/zjy-inkling2-mini`, 16 tokens):

| run | result |
| --- | --- |
| VarLoop, loop path, sync and async | identical to plain vLLM |
| Fixed, loop path | identical to plain vLLM (Fixed) |
| VarLoop, `rlt_graphs` | identical to plain vLLM |
| VarLoop, WCPB, 200-token prompts in 64-token chunks | identical to plain vLLM on the same prompts |
| VarLoop, `rlt_self` K=3, 1 draft pass | 14 rounds, 38 of 42 drafts accepted; prompts 1 and 2 identical to plain vLLM, prompt 3 differs from token 5 (cause not checked) |

**Throughput**, output tok/s (Inkling VarLoop, 3 passes, one B300; `tools/plugin`:
`run_job.sh` + `queue.sh`, median of 3 waves, same server flags as vLLM's row;
W1C1/W3C16: fixed prompts, 128 tokens, ignore_eos; W7C32: 512 tokens):

| engine | W1C1 | W3C16 | W7C32 |
| --- | --- | --- | --- |
| vLLM | 262.7 | 1045.6 | 3699.3 |
| plugin (rlt loop path), eager | 17.8 (7%) | 211.9 (20%) | 474.7 (13%) |
| plugin (rlt loop path), `rlt_graphs` | 168.9 (64%) | 640.4 (61%) | not run |
| existing rlt engine (reference) | 256.0 | 510.1 | 2191.2 |

These rows are the baseline: the rlt loop path with no loop gain exercised. The
≥95% target is **not met**; the best row is 61–64% of vLLM. The existing rlt
engine, running the same loop program, reaches 97% at W1C1, so the gap is in the
plugin. Each op is a full EngineCore step (schedule, input preparation, sampler
bookkeeping) for a fraction of the model, and without `rlt_graphs` every op runs
eagerly (vLLM's graphs were captured for the whole model and are not replayed for
loop ops).

Bugs these runs found and fixed: drafts never reached vLLM (a sync scheduler
class under vLLM's default async engine); every CODA ran twice under async
scheduling (a late return cleared the next op's in-flight mark); a mixed-depth
step dropped rows without carried state and shifted the others; the KV fill
assumed a `(2, blocks, ...)` layout; loop graph output was read after the graph
manager was switched back.

Not run: data parallelism (`DPOpAlign`, idle-rank dummies); the accuracy of
early exit on a task (only its mechanics are checked); Inkling with
`loop_share_cache`.

## Decided, not yet in code

* **Mixed depth from the first step, no switch.** A CORE step holds rows at
  different depths for Ouro and Inkling alike; `rlt_allow_mixed_depth` goes away.
  Weight work that does not depend on the pass (projections, MLP/MoE, the message
  controllers) runs once for the whole batch; work that does runs once per depth
  present. For Inkling that is attention KV (mask the slots in
  `attn_metadata[layer].slot_mapping`, which is what its kernels write through),
  the short-conv state `conv_states[pass]` (mask the conv metadata's slots), the
  message blend (only rows at depth > 0 have a bank), and the last core layer's
  next query (rows on their last pass mix for the CODA). The carried state is saved
  after the next pass's restart (bank cut, stack back to `prelude_blocks`), so every
  depth enters CORE with the same block count and states merge along the token axis;
  depth-0 rows get a zero bank. Mixed-depth steps run eagerly at first.
* **Scheduler: reimplement it (preferred), keep the subclass as a control.**
  Both plug in through `scheduler_cls` and are compared before one is kept.
  1. *Reimplement* (preferred, not written yet): a scheduler that implements
     `SchedulerInterface` directly, without subclassing `Scheduler`. Scheduling has
     one level: each `schedule()` picks one op's rows from the ready loop steps, and
     admission, KV allocation, preemption and op choice are decided in the same
     place. Queues, token budget, chunked prefill, preemption, spec-token accounting
     and rollback, output placeholders and the loop state are rlt's;
     `KVCacheManager`, `StructuredOutputManager`, the encoder cache manager, the KV
     connector and vLLM's output types are used by composition. Sync and async are
     two modes of one class, chosen by `async_scheduling`. vLLM features it does not
     cover (KV connector async loads, encoder budget, DP prefill throttling, LoRA
     limits, priority scheduling) raise at startup in the first version. EngineCore
     only calls the interface (`schedule`, `update_from_output`,
     `update_draft_token_ids[_in_output]`, `get_grammar_bitmask`, `add_request`,
     `finish_requests`, `has_[unfinished_]requests`, `make_stats`, the reset and
     connector getters, `get_request_counts`, `get_kv_cache_usage`, `pause_state`,
     `shutdown`), and `get_scheduler_cls` does not check the base class.
     Why preferred: the subclass works around `Scheduler`'s one-span-per-request
     model by hiding mid-loop requests from `running`, handing vLLM a restricted
     `SchedulerOutput` at a span's last op, and keeping a span table beside vLLM's;
     the two scheduler bugs the GPU runs found were at that seam. It also runs the
     full `Scheduler.schedule()` on every step that opens a span.
  2. *Subclass* (current code, control): `LoopScheduler(Scheduler)` /
     `LoopAsyncScheduler(LoopScheduler, AsyncScheduler)`, two-level scheduling as
     described under "How it works". Structured output, KV connector, encoder
     budget, DP prefill throttling, preemption and async placeholders come with the
     base class; it depends on `Scheduler` internals.

  Both share `LoopRequest`, `LoopSchedulerOutput`, the loop program, the runner and
  the tests. They are compared on throughput with loop features off (≥95% of vLLM),
  per-step CPU time of `schedule()` and `update_from_output` under nsys, the same
  correctness cases (the applicable vLLM scheduler CPU tests plus the plugin's
  sync/async/WCPB/self-spec/exit cases), and how many vLLM internals each depends on.

## Known gaps

* Inkling mixed depths are not implemented yet (see above): today an Inkling CORE
  step holds one depth and `state_cat` refuses to merge states with different block
  counts, so WCPB on Inkling overlaps chunks of different requests only.
* The exit decision is made on the device but read back by the scheduler, one
  step late under async scheduling; gated rows therefore advance synchronously.
* The draft step is eager and one token at a time; a CUDA graph for it and a
  batched draft/verify CODA are not done.
* The decode pipeline (tokens of one request at different depths in one step) is
  not implemented. A span is one program walk.
