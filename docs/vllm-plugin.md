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

Tested on CPU against the real vLLM scheduler (`tests/plugin`, 32 tests, set
`VLLM_SRC` to a vLLM checkout). **Not run on a GPU**: `runner.py`, `worker.py`,
`ouro*.py` and `kvfill.py` are untested. Known gaps:

* CORE steps mix rows of one depth only; mixed depth raises in the runner.
* Inkling runs as one FULL op. Splitting it into per-op steps needs the
  AttnRes state (`stack`, `prelude_partial`, `bank`) to be kept between ops.
* `LoopWorker.execute_dummy_batch` runs a dummy loop step only after
  `set_dummy_step`; nothing in the DP engine calls it yet.
* `DPAlign` needs a gather over the DP group; it is not wired.
* The runner turns CUDA graphs off for loop steps; per-op graphs are not done.
* Async scheduling advances gate-free rows when a step is emitted; the interplay
  with vLLM's output placeholders is only tested at the scheduler level.
