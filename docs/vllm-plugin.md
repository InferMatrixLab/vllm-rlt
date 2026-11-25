# vllm-rlt as a vLLM plugin (v0, unverified on GPU)

This branch adds `vllm_rlt/plugin/`: the loop-aware parts of vllm-rlt running
inside vLLM's own engine process. Serving, tokenizer, executor, parallelism,
kernels, KV allocation, the sampler and structured output are vLLM's; this
package only adds what needs to know a token is a loop.

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

Per-request loop parameters go in `vllm_xargs`:
`rlt_exit_threshold`, `rlt_min_loops`, `rlt_max_loops`, `rlt_target_loops`.

## Layout

| file | role |
| --- | --- |
| `__init__.py` | entry point `register()` |
| `patch.py`, `request.py` | the one patch: replace `Request` with `LoopRequest` everywhere (as vllm-omni does) |
| `scheduler.py` + `sched_*.py`, `carry.py`, `plan.py` | `LoopScheduler(Scheduler)`, `LoopAsyncScheduler(LoopScheduler, AsyncScheduler)` |
| `program.py`, `output.py` | loop ops/steps, `LoopSchedulerOutput`, `LoopModelRunnerOutput` |
| `parallel.py` | DP alignment of the op kind (`choose_op`, `Aligner`), dummy loop steps |
| `exit.py`, `kvfill.py` | early-exit policy (Ouro gate) and LAST_EXITED KV fill |
| `runner.py`, `worker.py` | `LoopModelRunner(GPUModelRunner)`, `LoopWorker(Worker)` |
| `ouro.py`, `ouro_layers.py`, `models.py` | Ouro on vLLM layers; per-model loop layout |

## How it works

* One EngineCore step is one loop step (mapping A). Every step holds one op
  kind: PRELUDE, CORE at one depth, CODA, or FULL.
* vLLM's `schedule()` still does admission, KV allocation, preemption and token
  accounting. It opens a token span; the loop scheduler walks the span through
  its ops while the request is hidden from vLLM's running list.
* `update_from_output` reaches vLLM only when a span's last op returns.
* A model vLLM already runs natively (Inkling) uses `LoopLayout.full_only`: the
  scheduler passes straight through to vLLM, so with loop features off the
  baseline is vLLM itself.
* Early exit follows Ouro: `remaining *= 1 - sigmoid(gate)`, exit when
  `1 - remaining >= threshold`, bounded by min/max loops. The runner decides on
  the device; a skipped pass gets the exit pass's KV copied slot by slot.

## Status

Tested on CPU (`tests/plugin`, 46 tests, `VLLM_SRC` pointing at a vLLM checkout
of the pinned commit): the loop program, planner, exit policy and DP alignment
tables; the scheduler against the real vLLM `Scheduler` (sync and async); the
`Request` patch and entry point; Ouro's model code built on vLLM's CPU setup
(layer names, parameter names and loading against the checkpoint layout, per-pass
KV layers, depth-masked slot mapping); and the Inkling op partition (every op
visits exactly vLLM's layers, once, in order).

**Not run on a GPU.** Nothing below has executed a kernel:

* `runner.py`, `worker.py` (the code that talks to vLLM's runner internals:
  `execute_model_state`, slot mappings, `_dummy_run`, the CUDA-graph manager);
* `ouro*.py` numerics and `kvfill.py`;
* `inkling.py`: it calls vLLM's private `_run_layer`. Its first check on a GPU
  must be that running prelude, the core passes and coda as separate ops gives
  the same final hidden state as vLLM's single forward.

What was added since v0: DP op alignment in the workers (an all-reduce of each
rank's candidate table over vLLM's DP CPU group, and an idle rank joins it before
running a dummy of the agreed op); depth-mixed CORE steps for Ouro (projections
run once, attention once per depth with a slot mapping that only writes that
depth's rows); Ouro's HF config registered so vLLM can load the checkpoint; and
Inkling split into ops (`rlt_split_ops`), with the AttnRes state carried between
ops.

## Known gaps

* Per-op CUDA graphs: the runner switches graphs off for a loop step. This is the
  largest open performance item.
* Inkling mixed depths: the AttnRes state differs per depth (block counts), so an
  Inkling CORE step holds one depth; `state_cat` refuses to merge otherwise.
* The exit decision is made on the device but read back by the scheduler, one
  step late under async scheduling; gated rows therefore advance synchronously.
* `LAST_EXITED` fill assumes a `(2, blocks, block_size, ...)` KV layout and
  raises on any other; the real layout of the attention backend in use has not
  been checked.
* No data-parallel run: `DPOpAlign` and the idle-rank path are unexercised.
* No accuracy check of early exit against Ouro's reference.
