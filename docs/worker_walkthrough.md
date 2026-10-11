# Worker and ModelRunner Walkthrough

This is the first M5 increment of the [refactor RFC](https://github.com/ThinkFlowLab/vllm-rlt/issues/32). It explains how device execution works today: what `ModelRunner` owns, how the asynchronous path keeps tokens and hidden states on the GPU, which streams and events order the work, and which other modules reach into it. It changes no code. It follows the RFC's seven-part walkthrough layout.

Files in scope:

- [model_runner.py](../vllm_rlt/worker/model_runner.py): stage execution, submissions, readback leases, finalization, slot release.
- [async_state.py](../vllm_rlt/worker/async_state.py) and [kernels/routing.py](../vllm_rlt/kernels/routing.py): resident request slots, routing banks, gather/scatter, the batched finalize kernel.
- [buffers.py](../vllm_rlt/worker/buffers.py), [cuda_graph.py](../vllm_rlt/worker/cuda_graph.py), [prefill_metadata.py](../vllm_rlt/worker/prefill_metadata.py) with [kernels/prefill_metadata.py](../vllm_rlt/kernels/prefill_metadata.py).
- [core/memory.py](../vllm_rlt/core/memory.py): startup KV budget.
- The device half of [kv_cache_manager.py](../vllm_rlt/core/kv_cache_manager.py) (storage tensors, writes, finalize copies), which the RFC moves to M5 later. Backend selection and attention metadata moved to `vllm_rlt/attention` in #46 (M8).
- Callers: [llm_engine.py](../vllm_rlt/engine/llm_engine.py), [preemption.py](../vllm_rlt/engine/preemption.py), [pd/worker.py](../vllm_rlt/pd/worker.py), and the device fields of [request.py](../vllm_rlt/request.py).

## Why the runner looks the way it does

A recurrent model runs the same transformer core several times per token, and different requests in one batch can be at different loop depths. The runner therefore executes four kinds of batches, one stage at a time:

```text
PREFILL    prompt chunk -> embed -> every loop depth            -> keep last hidden
PRELUDE    sampled token -> embed                               -> keep hidden
RECURRENT  kept hidden   -> one loop at each row's own depth    -> keep hidden, return gate score
CODA       kept hidden   -> LM head -> sample                   -> return token
```

The hidden state has to survive between batches, so the runner keeps one per request. In synchronous mode this is easy: run a stage, copy results to the host, let the engine decide what comes next.

Asynchronous mode overlaps the CPU with the GPU. It is built on three ideas:

1. **Tokens never come back to the CPU on the critical path.** CODA scatters sampled IDs into a resident device pool. The next PRELUDE gathers them from there. The host copy is read later, only to report output.
2. **Exit decisions are one loop late.** The engine submits loop `r`, then reads loop `r-1`'s gate score to decide whether `r+1` is needed. No speculative extra loop runs. The policies that allow this are `ouro_delayed`, `random_lookahead`, and `trace`.
3. **Events, not host waits, order the work.** Every request carries the event of its last submitted work. Any later stage on any stream waits for it. The host blocks only to reuse a buffer, to read a result, or to recycle a slot.

Everything else in the scope is machinery for those three ideas: reusable buffers so nothing allocates mid-step, pinned host memory so copies do not serialize streams, and leases so a buffer is not reused while a copy is still reading it.

```text
ModelRunner
  prepare(batch)             snapshot depths/positions/output indices, build routing
  submit(batch)              pick stream, wait request events, _execute, D2H into a leased slot, record event
  _execute(batch, prepared)  four stages x four execution paths (section 4.3)
  finalize / finalize_many   early-exit KV copies into skipped depths
  release(request_id)        wait the request's last event, recycle its slot
  execute(batch)             synchronous path: _execute + .tolist()
```

## 1. Baseline and boundary

Baseline: upstream `main` at `11b1d34`, after [#46](https://github.com/ThinkFlowLab/vllm-rlt/pull/46) (M8). Line numbers below refer to that commit. #46 moved backend selection and attention metadata into `vllm_rlt/attention` and replaced the FA4 `generation == 4` check in `_prefill_tokens` with `cache.attention_capabilities.packed_prefill`. One open PR changes lines in scope and is assumed to land first: [#57](https://github.com/ThinkFlowLab/vllm-rlt/pull/57) (M4) replaces the `written` reads and writes in `finalize_many`.

Not in scope: the sampler arithmetic (M7, [sampling_walkthrough.md](sampling_walkthrough.md)), the model's layer code (M6), attention kernels and backend selection (M8), and `SpeculativeRunner` (RFC #43). `SpeculativeRunner` is a second runner in `worker/` that the engine calls instead of `ModelRunner` for SPECULATIVE batches (`llm_engine.py:110-114`, `253-255`). It uses the same `RecurrentGraphs` class and the cache's `_prepare_batches`, but shares no state with `ModelRunner`. The Executor in section 6 has to dispatch to both.

Who touches the runner and its device state at the baseline:

| Caller | Module | What it uses |
| --- | --- | --- |
| `LLMEngine.__init__` | M2 | Constructs `ModelRunner` directly (`llm_engine.py:115-121`). Calls `plan_cache` before the KV manager exists (`86-92`) |
| `LLMEngine._step` (sync) | M2 | `execute`, `finalize`, `release`, `synchronize` (`231-263`, `336`, `369`) |
| `LLMEngine._update` | M2 | `model_runner.events.get(request_id)` to defer prefix publication (`308`) |
| `LLMEngine._step_async` | M2 | `submit`, `finalize_many`. Reads `boundary_stream` and `core_stream` (`484-485`). Reads `ticket.device_values`, `ticket.batch`, `ticket.output_indices`, `ticket.depths`, `ticket.ready()`, `ticket.collect()` (`443-533`) |
| `PreemptionManager` | M3 | `synchronize`, `release` (`preemption.py:79`, `102`). Reads `hidden_state` and `input_token_tensor` to snapshot, writes them back on resume (`97-108`, `134-137`) |
| `PDWorker` | M9 | `submit` for prefill, then drops the ticket (`pd/worker.py:296`, `321`). Reads `core_stream` (`297`). `release`, `synchronize` (`112`, `350`, `410`). Copies `request.hidden_state` into its own transfer buffer and later points `hidden_state` at that buffer (`307`, `351`, `385`) |
| `Scheduler.finish` | M3 | Clears `hidden_state`, `input_token_tensor`, `generator` (`scheduler.py:92-95`) |

What the runner reads from each `Request`, which is what a typed `SchedulerOutput` (M3) has to carry instead:

| Field | Read at | Purpose |
| --- | --- | --- |
| `request_id` | everywhere | Slot, event, and KV allocation key |
| `loops_done` | `prepare` 327, `_execute` 397, 413, `finalize*` 518, 576 | Zero-based depth of the loop being run. After an exit, `loops_done - 1` is the exit depth |
| `position` | `prepare` 328, `_execute` 398, 414, `finalize*` | Token position. Derived from prompt length and scheduled outputs |
| `num_scheduled_outputs` | `prepare` 329 | Output index, used to check coda delivery order |
| `token_start`, `token_count`, `prompt_token_ids` | `_prefill` 253-267 | Prompt chunk |
| `input_token_id` | `_execute` 378 | Sync PRELUDE input, a host int |
| `input_token_tensor` | `prepare` 332, `_execute` 371-376 | Async PRELUDE input, a device scalar. Cleared by the runner |
| `hidden_state` | `submit` 472, `_gather` 228, `ensure_slot` 164 | Hidden state. Written by the runner (section 2) |
| `sampling_params`, `generator` | `_sample_tensor` 604-605 | Sampling. The generator is created and written back by the runner |

## 2. Classes and state

### ModelRunner

| Field | What it holds | Written by | Lives for |
| --- | --- | --- | --- |
| `model`, `cache_manager`, `device`, `sampler` | Dependencies | `__init__` | Runner |
| `graphs` | `RecurrentGraphs` when `cuda_graphs` | `__init__` 86-95 | Runner |
| `lookahead_head` | Frozen random `Linear(hidden, 1)` for `random_lookahead`, built on a forked CPU RNG | `__init__` 98-110 | Runner |
| `core_stream`, `boundary_stream`, `copy_stream` | Three streams on CUDA async. One shared stream with `multi_stream=False`. `None` otherwise | `__init__` 111-125 | Runner |
| `readback_slots` | Pinned host buffers: `max_num_seqs + 4` per dtype (float32 scores, int64 tokens), each `max_num_batched_tokens` long | `__init__` 126-138. Leased by `_readback_slot`, returned by `Submission` | Runner |
| `submission_events` | Events of the last submissions, used to bound in-flight work to three | `submit` 451-454, 495 | Rolling |
| `events[request_id]` | The event of each request's latest submitted work: a submission, a finalize copy, or a finalize kernel | `submit` 494, `finalize` 581, `finalize_many` 562. Popped by `release` | Request |
| `workspaces`, `workspace_index` | Two `Workspace` per group (`core`, `boundary`) when `static_buffers` | `__init__` 154-158, `_workspace` | Runner |
| `states`, `state_slots`, `free_state_slots` | Hidden-state slot pool. Static pool when `static_buffers`, replaced by aliases into `AsyncState` on CUDA async (179-181) | `_save`, `release` | Runner |
| `async_state` | `AsyncState` on CUDA async | `__init__` 169-183 | Runner |
| `_prefill_banks`, `_prefill_bank_index` | Three `PrefillMetadataBank` for FA4 packed prefill with `prefill_uva`, created lazily | `_prefill_tokens` 283-286 | Runner |
| `last_effective_size`, `last_submitted_size` | Real versus padded row count of the last batch (diagnostics) | `_size` 189 | Rolling |

### Submission (the ticket returned by `submit`)

| Field | What it holds |
| --- | --- |
| `batch` | The `SchedulerOutput` that was submitted |
| `values` | Host view of the result in a leased `ReadbackSlot`. `None` for PREFILL, PRELUDE, and trace-mode RECURRENT. On CPU, the result tensor itself |
| `device_values` | The device result. The engine hands each CODA row to the next PRELUDE |
| `event` | Recorded after the device-to-host copy. `None` on CPU |
| `slot` | The leased `ReadbackSlot`. Returned by `collect()` or by `__del__` |
| `cached` | The list produced by `collect()` |
| `depths`, `output_indices` | Snapshots taken in `prepare`, so delivery does not read state that has already moved on |

The result means different things by stage: nothing, gate probabilities (float32, after sigmoid), or token IDs (int64). The caller has to know the stage to read it (finding F6).

### AsyncState and RoutingBank

| Field | What it holds | Lifetime |
| --- | --- | --- |
| `AsyncState.hidden` `[max_num_seqs, hidden]` | Resident hidden state, one row per slot | Runner |
| `AsyncState.tokens` `[max_num_seqs, 1]` int64 | Resident sampled token, one per slot. Written by the CODA scatter, read by the PRELUDE gather | Runner |
| `AsyncState.tables` `[max_num_seqs, planes, width]` int32 | Device copy of each slot's block tables | Runner |
| `slots[request_id]`, `free` | Slot assignment. `free` is a LIFO list | Admission (first `prepare`) to `release` |
| `owners[request_id]` | `(id(request), id(allocation))`, to catch a slot reused before release | Same |
| `table_versions[request_id]` | The block-table tuple last uploaded | Same |
| `banks` (4) | Routing banks for PREFILL, PRELUDE, RECURRENT, CODA. Rotate round robin | Runner |
| `finalize_banks` (2) | Control-only banks: a pinned descriptor, no device tensors | Runner |
| `RoutingBank.host` `[rows, 3]` pinned int64 | `(slot, depth, position)` per row. Read by the GPU through UVA, or copied to the device on HIP | One use |
| `RoutingBank.slots/positions/blocks/offsets/lengths/tables` | Per-row metadata produced on the device by `metadata_kernel` | One use |
| `RoutingBank.hidden`, `RoutingBank.tokens` | Gather output buffers | One use |
| `RoutingBank.uploads`, `imports` | Pending table uploads and hidden-state imports for new or grown slots | Until `acquire` |
| `RoutingBank.ready_event`, `done` | Metadata ready (on `copy_stream`), last reader finished (on the execution stream) | One use |

A bank is reused only after `acquire()` waits for `done` (`async_state.py:39-44`). A slot is reused only after `release()` waits for `events[request_id]` (`model_runner.py:583-594`).

### Workspace and PrefillMetadataBank

`Workspace` (`buffers.py:29-96`) holds a hidden buffer plus pinned host and device copies of tokens, positions, blocks, offsets, tables, and lengths. `acquire` waits for the event recorded by the last `release`, which protects the pinned source of a non-blocking copy as well as the device scratch. `PrefillMetadataBank` (`prefill_metadata.py:12-110`) packs all FA4 packed-prefill metadata into one pinned buffer, stages it with one Triton copy, then expands per-depth block and offset tensors on the device. Its buffers grow by powers of two at run time.

### Device state on Request

Three tensors that the RFC assigns to the worker still live on `Request`:

| Field | Writers |
| --- | --- |
| `hidden_state` | `ModelRunner._save` 205, 210, `_save_batch` 220, `AsyncState.ensure_slot` 166, `PreemptionManager` 108, 134, `PDWorker` 351, 385, `Scheduler.finish` 92 |
| `input_token_tensor` | `LLMEngine._step_async` 481, `ModelRunner._execute` 376, `PreemptionManager` 108, 135-137, `Scheduler.finish` 93 |
| `generator` | `ModelRunner._sample_tensor` 605, `sampling.generator_for` 24, `Scheduler.finish` 95 |

In sync mode `hidden_state` is either a private clone or a view of a static slot. In async CUDA mode it is a view of `AsyncState.hidden[slot]`. Which one depends on the mode and on whether the request already has a slot.

## 3. Functions and arguments

Shapes use `n` for real rows, `size` for submitted rows (`n` rounded up to a power of two when padding), and `H` for the hidden size.

| Function | Expects | Does | Side effects |
| --- | --- | --- | --- |
| `ModelRunner.prepare(batch)` | A scheduled batch. For async PRELUDE, every request has `input_token_tensor` | Snapshots `depths`, `positions`, `output_indices`. On CUDA async, takes the next routing bank, assigns or refreshes slots, writes descriptors. For RECURRENT, also builds a `_PreparedKVBatch` over the bank's tensors. On CPU async without workspaces, builds RECURRENT metadata with `_prepare_batch` | Rotates the bank (may wait on its `done`). Assigns slots. Queues table uploads and hidden imports. Sets `last_*_size` |
| `ModelRunner.submit(batch)` | Same | Bounds in-flight submissions to three. Picks `core_stream` for PREFILL and RECURRENT, `boundary_stream` for PRELUDE and CODA. Runs routing on `copy_stream`. Waits each request's event. Runs `_execute`. Copies the result into a leased pinned slot. Records one event and assigns it to every request in the batch | Returns a `Submission`. The host blocks only in `_readback_slot`, the three-submission bound, or `bank.acquire` |
| `ModelRunner.execute(batch)` | Synchronous engine | `_execute`, then `.cpu().tolist()` | Host sync per step, by design |
| `ModelRunner._execute(batch, prepared)` | `prepared` from `prepare`, or `None` on the sync path | Section 4.3 | Writes KV. Saves hidden states. Samples. Scatters tokens |
| `ModelRunner.finalize(request)` | The token's KV is written at every layer at depth `loops_done - 1` | On `boundary_stream` after the request's event: `finalize_token` copies the exit-depth KV to every deeper plane. Records a new event | Marks `written` for the deeper planes |
| `ModelRunner.finalize_many(requests)` | Same, for a list | CUDA async LAST_EXITED only: one `finalize_kernel` launch on `boundary_stream`, using the request slots' device tables. Otherwise loops `finalize` | Marks `written`. Sets each copied request's event to the bank's `done` |
| `ModelRunner.release(request_id)` | Any ID | Pops and **synchronizes** the request's last event, then frees its slot | Host block until that request's work completes |
| `ModelRunner.synchronize()` | — | Synchronizes all three streams | Host block |
| `AsyncState.ensure_slot(request, bank)` | The request has a KV allocation | Returns the request's slot, allocating one on first use. Queues a table upload when the block-table tuple changed. On first use, queues a hidden import and repoints `request.hidden_state` at the slot | Raises if the slot belongs to a different `(Request, allocation)` pair, or if no slot is free |
| `RoutingBank.transfer(batch)` | Runs on `copy_stream` | Uploads tables, imports hidden states, launches `metadata_kernel` over `size` rows, records `ready_event`. Returns `batch` with device metadata swapped in | Writes `AsyncState.tables` and `hidden` |
| `RoutingBank.gather(tokens=False)` | `transfer` has run | `[size, H]` hidden, or `[size]` tokens, gathered by slot. Padding rows read 0 | Kernel launch |
| `RoutingBank.scatter(values, tokens=False)` | `values` has `count` real rows first | Writes rows into the slot pool | Kernel launch |
| `Workspace.prepare(cache, ids, depths, positions, size)` | Live allocations | Fills pinned host metadata, copies it non-blocking, returns a `_PreparedKVBatch` over device views | Uses the M4 private `_validate_rows` and `_plane` |
| `plan_cache(model, cache, scheduler, execution, backend)` | Called before `KVCacheManager` exists | Returns `(num_blocks, plan)`. Explicit block or byte overrides first. CPU defaults to 256 blocks. On CUDA: empties the allocator cache, runs a maximum-row prelude, recurrent, and coda pass on a probe cache, then subtracts the peak (doubled for multi-stream async), torch scratch, `execution_buffer_bytes`, metadata, reserve, and graph reserve | Synchronizes the device and resets peak statistics |
| `execution_buffer_bytes(...)` | Same config as the runner | Device bytes of workspaces, static states, routing banks, the async pools, and the finalize descriptors | Pure |

## 4. Line-by-line coverage

### 4.1 Construction (`model_runner.py:69-183`)

| Lines | What happens | Notes |
| --- | --- | --- |
| 78-85 | Takes the device and dtype from the first parameter. Creates `Sampler` | |
| 86-95 | `RecurrentGraphs` when `cuda_graphs`. `compute_gate` is true only for `ouro` and `ouro_delayed` | The engine rejects graphs off CUDA (`llm_engine.py:61-62`); since #46, backend compatibility comes from the selected backend's `cuda_graphs` capability |
| 98-110 | `random_lookahead`: builds the head on CPU inside `fork_rng(devices=[])`, then moves it | Model-weight initialization inside the runner. An M6/Worker concern |
| 111-125 | Three streams on CUDA async. With `multi_stream=False` all three are the same stream. Each waits the current stream | |
| 126-138 | Pinned readback slots, allocated up front, because `cudaHostAlloc` during submission can serialize streams | Section 4.6 explains the count |
| 141-148 | `submission_events`, `events`, empty workspaces, a default slot pool of `max_num_seqs`, `states=None` | |
| 149-167 | `static_buffers`: four workspaces (2 x core, 2 x boundary), padded to a power of two when padding. `states [max_num_seqs, H]`. Streams wait again | |
| 169-183 | CUDA async: `AsyncState` with the same padded row count. **`states`, `state_slots`, `free_state_slots` are rebound to the async pools** | The static `states` from 159 is then unused (F4) |

### 4.2 Submission (`model_runner.py:324-349`, `440-504`)

| Lines | What happens | Branches and failure paths |
| --- | --- | --- |
| 326-329 | Snapshots depths, positions, output indices from the requests | |
| 331-334 | PRELUDE collects `input_token_tensor` | Raises if any is missing: async prelude needs the device sample from CODA |
| 335-343 | CUDA async: `AsyncState.prepare`. `recurrent=True` builds KV metadata as well. PREFILL also goes through here, but only to register the slot (F11) | Bank rotation may block on `done` |
| 344-349 | CPU async: RECURRENT without workspaces gets `_prepare_batch`. Everything else leaves `kv=None` for `_execute` | |
| 451-454 | Drops completed submission events. With three still pending, synchronizes the oldest | Host block. `prepare` runs *before* this (comment at 139-140) |
| 455-459 | Stream by stage: core for PREFILL and RECURRENT, boundary for PRELUDE and CODA | `None` on CPU |
| 460-466 | Routing: if hidden imports are queued, `copy_stream` first waits the current stream, because imported tensors may come from it (a preemption restore, a PD buffer). `transfer` runs on `copy_stream`. The execution stream waits `ready_event` | |
| 467-474 | On the execution stream: wait each request's event, `record_stream` its hidden state | One `wait_event` per request even when requests share an event (F9) |
| 475-479 | `_execute`. `record_done` on the routing bank in `finally`, so the bank is retired even on failure | Exceptions propagate. The engine then synchronizes and aborts everything (`llm_engine.py:231-244`) |
| 480-495 | CUDA: lease a readback slot of the result's dtype, copy non-blocking, record one event, attach it to the slot, to **every request in the batch**, and to `submission_events` | `_readback_slot` raises if exhausted. It also synchronizes the slot's previous event, a safety net for tickets dropped before their copy finished |
| 496-504 | Returns the `Submission` | |
| `Submission.collect` 54-65 | Waits the event, converts to a list, returns the lease | Idempotent |
| `Submission.__del__` 47-49 | Returns the lease if the ticket is dropped without `collect` | Relies on CPython reference counting (F13) |

### 4.3 Execution (`model_runner.py:185-322`, `351-433`)

Four execution paths:

| Path | Selected when | Hidden in | Hidden out | KV metadata |
| --- | --- | --- | --- | --- |
| Sync dynamic | sync, no `static_buffers` | `torch.stack(hidden_state)` | `_save`: clone per request | `model.recurrent` → `_prepare_batch` |
| Sync static | sync, `static_buffers` | `_gather`: zero the workspace, copy per request | `_save`: copy into a static slot | `Workspace.prepare` |
| CPU async | `async_scheduling` on CPU | As sync, by `static_buffers` | As sync | `prepare`: `_prepare_batch` (dynamic) or `Workspace.prepare` (static) |
| CUDA async | `async_scheduling` on CUDA | `routing.gather()` | `routing.scatter()`, then repoint `hidden_state` per request | `AsyncState.prepare` + `metadata_kernel` |

Crossed with the stages:

| Stage | Sync dynamic / static | CUDA async | Graphs |
| --- | --- | --- | --- |
| PREFILL, LAST_EXITED, non-FA4 | `_prefill_tokens` 307-322: prelude, then one `_core` per depth. With a workspace, each depth re-acquires the same workspace | Same code on `core_stream`. Routing is ignored | Not used |
| PREFILL, LAST_EXITED, FA4 | 276-306: packed ragged prefill. With `prefill_uva`, one of three `PrefillMetadataBank` builds every depth's metadata once. Otherwise one `_prepare_batch(packed_prefill=True)` per depth | Same | Not used |
| PREFILL, SHARED | 250-261: position waves. Each request finishes all loops at one position before it advances, batched across requests. Each wave calls `_prefill_tokens` | Same | Not used |
| PRELUDE | Host token IDs → tensor (pinned workspace or `torch.tensor`) | Gather tokens from `AsyncState.tokens`. `record_stream` on the source tensors. Clear `input_token_tensor` | — |
| RECURRENT | `_gather` / `stack` → `_core` | Gather → `prepared.kv` | `graphs.run(hidden[:n], kv)` in any mode. `kv` comes from `prepared` or a fresh `_prepare_batch` |
| CODA | `model.coda` over `size` rows. One `_sample_tensor` call per real row | Same, then scatter tokens into `AsyncState.tokens` | `CodaGraphs` is not used by `ModelRunner` |

| Lines | What happens | Notes |
| --- | --- | --- |
| 185-190 `_size` | Pads to a power of two when enabled | Also records `last_*_size`, a hidden side effect |
| 192-199 `_workspace` | Alternates between the two workspaces of a group and acquires one | `None` without `static_buffers` |
| 201-211 `_save` | Clones when there is no slot pool, or when async exists but the request has no slot. Otherwise copies into the request's slot, allocating one | No in-tree path reaches the clone branch with `async_state` set, because `submit` always runs `ensure_slot` first (F16) |
| 213-220 `_save_batch` | Without routing: `_save` per row. With routing: one scatter, then `request.hidden_state = self.states[slot]` per request | Creates one view per request per step (F9) |
| 222-229 `_gather` | Stack, or zero the workspace rows and copy per request | Per-request copies |
| 231-247 `_core` | Dynamic `model.recurrent`, or `Workspace.prepare` + `recurrent_prepared` | |
| 249-272 `_prefill` | SHARED waves, or one LAST_EXITED batch. Keeps only each item's last row | Writes `hidden_state` |
| 274-322 `_prefill_tokens` | Packed versus generic, chosen by `cache.attention_capabilities.packed_prefill` (276; FA4 is the only packed backend today). In the generic path, lines 316-321 acquire and release the workspace around every depth | Each `acquire` waits for the previous depth's whole forward to finish (F10) |
| 361-386 PRELUDE | Covered above | `tokens.new_zeros` padding only without routing. The routing gather pads with zeros itself |
| 388-421 RECURRENT | Gather, run, save, optional lookahead head. Result is `sigmoid(logits[:n])` in float32, or `None` in trace mode | Padded rows run through the model but write no KV and produce no signal |
| 422-428 CODA | LM head, per-row sampling, token scatter | The sampler may create the request's generator |
| 429-430 | Any other stage raises | |
| 431-432 | Releases the workspace (records its event) | Skipped on exception. The engine's error path synchronizes everything |

### 4.4 Finalization and release (`model_runner.py:506-599`)

| Lines | What happens | Notes |
| --- | --- | --- |
| 509-512 | No async state, or SHARED: call `finalize` per request | SHARED `finalize_token` only validates (`kv_cache_manager.py:679-680`) |
| 515-525 | Checks every layer's `written` at the exit depth. Keeps only requests with deeper planes to fill | #57 moves this to `token_written` |
| 527-533 | Prepares a control-only finalize bank with depth = exit depth | `ensure_slot` may queue uploads that a finalize bank never performs (F8) |
| 534-543 | On `boundary_stream`: wait each request's event, set the descriptor (UVA, or a non-blocking copy) | |
| 544-559 | One launch over a `(requests, max_loops, ceil(layers*channels/256))` grid. A program does nothing when `target <= depth` | Reads `AsyncState.tables`, not the bank's tables |
| 560-566 | Records `done`, makes it each request's event, marks `written` for every deeper plane | Host `written` advances at submission time. Section 4.7 |
| 568-581 `finalize` | Waits the request's event on `boundary_stream`, calls `finalize_token` (basic-index copies, no H2D index tensor), records a new event | On CPU or sync there is no stream and no event |
| 583-594 `release` | Pops and synchronizes the event. Async: `AsyncState.release`. Otherwise returns the static slot | Host block until the request's last work finishes, including work submitted after a delayed EOS |
| 596-599 `synchronize` | All three streams | |
| 601-606 `_sample_tensor` | Delegates to `Sampler`, writes back `request.generator` | M7 owns the arithmetic |

### 4.5 AsyncState and kernels

| Lines | What happens | Notes |
| --- | --- | --- |
| `async_state.py:17-37` | Bank buffers. `control_only` banks have only the pinned descriptor | |
| 39-44 `acquire` | Waits `done`, clears uploads and imports | Pinned upload tables stay referenced until here, so their non-blocking copies cannot read freed memory |
| 46-87 `transfer` | See section 3. Imports call `record_stream` on the source | `descriptor` is the pinned buffer itself under UVA. Without UVA (HIP) it is copied non-blocking |
| 89-91 `record_done` | Event on the current stream | |
| 93-112 `gather`, `scatter` | Grids `(size, ceil(H/256))` and `(count, ceil(H/256))` | Gather fills `size` rows and zeroes those past `count`. Scatter writes only `count` rows |
| 116-142 `AsyncState.__init__` | `width = ceil(max_position_embeddings / block_size)`. `planes` is `total_ut_steps` for LAST_EXITED, 1 for SHARED | `tables` start at zero. Rows are not cleared on release |
| 144-167 `ensure_slot` | Existing slot: check the owner, re-upload the table if the tuple changed. New slot: pop LIFO, record the owner, queue the upload, and if the request already has a hidden state, queue an import and repoint it | Only the hidden state is imported, not `input_token_tensor` (F1). `owners` stores `id()` values of objects it does not keep alive, so a recycled id could mask a missed release |
| 169-209 `prepare` | Rotates banks. `width` from the largest position. Descriptors use depth 0 except for RECURRENT and finalize. RECURRENT also validates rows, rejects duplicate write addresses, and builds a `_PreparedKVBatch` over bank tensors (its `position_ids` is a host view, swapped for the device tensor by `transfer`) | Uses `_validate_rows`, `_plane`, `_get_allocation`, `_PreparedKVBatch` from M4 |
| 211-216 `release` | Drops the bookkeeping, returns the slot | The caller must have synchronized the slot's last event (`ModelRunner.release` does) |
| `routing.py:7-41` `metadata_kernel` | One program per submitted row. Real rows get slot, position, length `pos+1`, write block and offset, and the table row. Padding rows get length 0 and no block or offset store | Padding never gets a KV address (`buffers.py:1` states the same rule) |
| 44-76 gather and scatter | Tiled row copies by slot | |
| 79-109 `finalize_kernel` | Copies one token's K and V for every layer and channel from the exit plane's block to the target plane's block | Same semantics as `finalize_token` |

### 4.6 Buffers, graphs, prefill banks, memory plan

| Lines | What happens | Notes |
| --- | --- | --- |
| `buffers.py:10-26` | Per row: workspaces `H*e + 4*width + 36` (four int64 and one int32 vector, int32 tables). Routing banks `H*e + 4*width + 68` (the same plus an int64 token and a 24-byte fallback descriptor). Async pools `H*e + 8 + planes*width*4` per slot. Finalize descriptors `2 * max_num_seqs * 24` | Matches `Workspace.__init__`, `RoutingBank.__init__`, `AsyncState.__init__`. Does not include `PrefillMetadataBank`, and still counts the static `states` that async replaces (F12) |
| `buffers.py:49-63` | `acquire` waits the last `release`. `tokens` zeroes `size` host rows and copies non-blocking | |
| `buffers.py:65-96` `prepare` | Validates rows, rejects duplicate addresses, zeroes positions, lengths, and tables up to `size`, fills real rows, copies non-blocking | Blocks and offsets are copied for real rows only |
| `cuda_graph.py:10-20` `_capture` | Synchronizes the device, warms up twice on the capture stream, captures | Each first capture of a key drains the device |
| 23-52 `_DeviceCache` | A proxy with only the KV write and attend operations, so the graph captures no Python `written` updates | |
| 65-88 `run` | Key = `(rows, tables, max_seqlen_q, packed)`. Falls back to eager above `cuda_graph_max_batch_size`, for empty batches, or when the graph cache is full | |
| 89-106 | Validates the batch and the written prefix on the host before replay. Falls back for non-contiguous same-request rows | Uses `_require_live_batch` and `_require_prefix` |
| 107-109 | Waits the previous replay's event | Graph-private buffers are shared across streams |
| 110-152 | Builds static entry tensors, copies metadata into them, captures on first use | |
| 153-162 | Replays, clones outputs, records `last_event`, marks `written` | The host marks `written` at replay submission |
| 165-195 `CodaGraphs` | Fixed-row LM head graphs | Used only by `SpeculativeRunner` |
| `prefill_metadata.py:19-88` | Waits `done`. Builds tokens, positions, sequence ids, ends, cumulative lengths, and every plane's tables in one host list. Grows the buffers to a power of two if needed. One staging kernel, then one expand kernel per plane | Reads `cache._allocations` |
| 90-110 | `metadata(depth)` builds a `_PreparedKVBatch` positionally. `release` records `done` | |
| `memory.py:20-129` | Section 3 | Profiles `model.recurrent`, the dynamic path, not the static, async, FA4-packed, or graph paths. Imports `worker.buffers` from `core` |

### 4.7 Device half of the KV manager

| Lines | What happens | Notes |
| --- | --- | --- |
| `kv_cache_manager.py:132-136` | `key_cache` and `value_cache` `[blocks, layers, page, kv_heads, head_dim]` | Allocated by M4, registered directly by NIXL |
| 556-558 `_stage` | Pinned host tensor, non-blocking copy | The caching host allocator keeps the staging buffer alive |
| 599-614 `_write_prepared` | Validates, writes real rows with advanced indexing, **then marks `written` on the host** | |
| 617-624, 640-667 | Attention requires a contiguous `written` prefix through each row's position | |
| 668-694 `finalize_token` | Validates, then for LAST_EXITED copies each deeper plane with basic indexing and marks `written` | |
| 696-717 `read` | Test and debug helper that materializes a prefix | |

`written` therefore records that a write has been **enqueued**, not that it has finished. Every consumer is ordered behind the producer by a stream or an event: same-stream order inside one traversal, `events[request_id]` across stages, `ready_event`/`done` across banks, and the event passed to `publish_prefix` for prefix reuse. This is the contract M4's `written` and M5's events share. A refactor that moves either side has to keep both.

### 4.8 Callers

| Lines | What happens | Notes |
| --- | --- | --- |
| `llm_engine.py:231-244` | Async error path: synchronize all streams, abort every request, clear pending signals, coda, and in-flight tickets | Partial submissions are drained before slots are recycled |
| 245-263 | Sync path: `execute`, then `_update`. On failure, abort only the batch's requests | |
| 306-319 | PREFILL: publish the prefix with the request's event, so publication waits for the writes | The event is non-`None` only on CUDA async |
| 335-337 | Sync RECURRENT exit: `finalize`, then CODA | |
| 403-427 `_deliver_coda` | Drops results for unregistered or replaced `Request` objects. Checks output order against `output_indices`. Uses the snapshot `depths` for `exit_depths` | Object identity guards ID reuse |
| 441-460 | Drops ready in-flight tickets (their leases return), delivers ready coda. Prefers RECURRENT when a core is still running and the previous step submitted a coda on a separate boundary stream | |
| 461-477 | CODA: waits until earlier outputs are delivered, so each request has at most one pending output | |
| 478-492 | Submits. For CODA: one placeholder per request, `input_token_tensor = ticket.device_values[i]`, enqueue PRELUDE | The PRELUDE gather reads `AsyncState.tokens` instead (F1) |
| 496-532 | RECURRENT: submit `r`, increment `loops_done`, read `r-1`'s score (`collect` blocks on that older event only), check that the score's position and depth match, decide exit, keep `r`'s score for the next round, `finalize_many` the exits | |
| `preemption.py:79-112` | Synchronize everything, copy KV pages and the hidden and token tensors to the CPU, `release`, free, re-queue as WAITING | |
| `preemption.py:114-144` | Allocate, copy pages back, install `written`, move hidden and token to the device, synchronize the current stream, re-enqueue the saved stage | |
| `pd/worker.py:276-322` | Prefill role: `submit` a PREFILL batch, copy the final hidden state into the PD buffer on `core_stream`, record per-request events for chunk transfer, drop the ticket | |
| `pd/worker.py:349-353`, `385` | Releases the runner slot after the last prefill event. On decode activation, points `hidden_state` at the PD buffer (imported into a slot on the first `prepare`) | |

## 5. Findings

Classes as in the M4 note. **Behavioral**: a runtime effect that needs its own fix. **Structural**: a boundary problem for this refactor. **Performance**: a cost worth measuring. **Open question**: needs a decision. **Necessary**: complexity that stays.

| ID | Where | Class | What happens | Outcome |
| --- | --- | --- | --- | --- |
| F1 | `AsyncState.ensure_slot`, `_execute` PRELUDE, `PreemptionManager.resume` | Behavioral, confirmed on GPU | See below | Fixed in [#109](https://github.com/ThinkFlowLab/vllm-rlt/pull/109), which waits for #101 |
| F2 | `Request.hidden_state`, `input_token_tensor`, `generator` | Structural | Worker-side tensors live on the scheduler's object. Five modules write them (section 2). Which tensor `hidden_state` refers to depends on the mode | Runner-owned request state, step 3 |
| F3 | `prepare`, `_execute`, `_prefill_tokens`, `Workspace`, `AsyncState`, `PrefillMetadataBank`, `RecurrentGraphs` | Structural | M5 builds `_PreparedKVBatch` in four places and reads `_get_allocation`, `_allocations`, `_validate_rows`, `_plane`, `_require_live_batch`, `_require_prefix`. It writes `written` in `finalize_many` and after graph replay. Each builder repeats the address computation and the duplicate-address check | `finalize_many` is fixed by #57. The rest is M4 step 2 plus step 5 below (M4 F5, F7) |
| F4 | `model_runner.py:145-147`, `159-163`, `179-181`, `591-594` | Structural | Two slot pools. On CUDA async with `static_buffers`, the static `states` tensor is allocated and then replaced by aliases into `AsyncState`, and the two boundary workspaces are never used, because routing replaces them for every non-prefill stage. `release` branches on which pool is live | One pool, step 3 |
| F5 | `llm_engine.py:308`, `484-485`, `491`; `pd/worker.py:296-297`, `321` | Structural | Callers read `events`, the stream objects, and `ticket.device_values`, and rely on dropping a ticket to return a lease | Runner produces typed output (step 3, after #107); the engine stops reading events and streams |
| F6 | `Submission.values` | Structural | Scores, tokens, or nothing, decided by stage. `_update`, `_deliver_coda`, and `_step_async` each decode it again | `ModelRunnerOutput`, step 2 |
| F7 | `LLMEngine.__init__`, `ModelRunner.__init__`, `core/memory.py` | Structural | No Worker or Executor layer. The engine plans memory and constructs the runner. The runner creates streams, buffers, and a model head. Nothing shuts the runner down. `core/memory.py` imports `worker/buffers.py` | Thin Worker and Executor, step 6 |
| F8 | `finalize_many` 527-533, `ensure_slot` 151-154 | Open question | A finalize bank never calls `transfer`, so a table upload queued during finalize is dropped while `table_versions` still advances. Today this cannot happen, because `finalize_many` runs right after the same requests' RECURRENT `prepare` with no growth in between. Nothing enforces that | Finalize looks up existing slots and asserts instead of calling `ensure_slot` |
| F9 | `_save_batch` 219-220, `submit` 468-474 | Performance, unmeasured | Per request per step on the CUDA async path: one `self.states[slot]` view (`aten::select`), one `wait_event`, one `record_stream`. They exist to keep `Request.hidden_state` current and to order requests that usually share one event | Goes away with F2. Measure first (see below) |
| F10 | `_prefill_tokens` 315-321 | Performance, unmeasured | With `static_buffers`, non-FA4 prefill re-acquires one workspace for every depth. Each acquire waits for the previous depth's forward, so the host stalls once per depth. Under async scheduling this blocks the scheduling thread during prefill. Dynamic metadata does not stall | Build all depths' metadata once, as `_prepare_batches` and `PrefillMetadataBank` already do |
| F11 | `prepare` 335-343 for PREFILL | Open question | PREFILL takes a routing bank, uploads tables, and launches `metadata_kernel` with the request's last prompt position, only to register a slot. `_prefill` ignores the routing. The bank rotation can block | Separate slot registration from routing |
| F12 | `execution_buffer_bytes` | Open question | The formula matches the workspaces, routing banks, and async pools exactly. It omits `PrefillMetadataBank` (three banks that grow at run time with `prefill_uva`) and counts static `states` that async replaces. Whether profiling slack covers the first is unverified | Budget the banks, or size them at startup |
| F13 | `Submission.__del__`, `_readback_slot` 440-447 | Necessary, fragile | A lease returns on `collect` or when the ticket is garbage collected. `_readback_slot` also synchronizes the slot's last event, which covers tickets dropped before their copy finished (error path, `del ticket` in PD). An exception traceback that holds a ticket keeps its lease | Keep. Make return explicit in the typed output |
| F14 | `events[request_id]`, `release` | Necessary | One event per request, overwritten by each submission or finalize, gives every later stage on any stream one dependency to wait for. `release` must wait it before recycling the slot. Section 4.7 explains why `written` alone is not enough | Keep. Document as the M4/M5 contract |
| F15 | Banks, workspaces, readback slots, three-submission bound | Necessary | Two core rounds plus one boundary stage can be in flight. Four routing banks cover that plus the bank being prepared. Readback needs one float slot per request with a pending signal and one int slot per request with a pending coda, each at most `max_num_seqs`, plus in-flight margin: `max_num_seqs + 4` | Keep. Write the bounds next to the constants |
| F16 | `_save` 202-205; `cdb_runtime.md:47`, `181` | Open question | The clone branch for "async without a slot" is unreachable from in-tree callers. The runtime doc says async enables static buffers automatically. It does not: CUDA async always uses routing banks, and prefill stays dynamic unless `--static-buffers` is given | Delete the branch or assert. Fix the doc |

**F1 in detail.** On CUDA async, the PRELUDE input comes from `AsyncState.tokens[slot]` (`model_runner.py:365-369`), written by the previous CODA scatter (`427-428`). `request.input_token_tensor` is only used to check presence and for `record_stream` (`332-334`, `371-372`). Preemption snapshots `input_token_tensor` and restores it on resume (`preemption.py:97-100`, `135-137`), but `release` gives the slot back (`async_state.py:211-216`). On resume, `ensure_slot` imports the hidden state only (`164-166`). If the victim was waiting in PRELUDE and gets a different slot, the next prelude embeds whatever token that slot last held, which is another request's sample. Conditions: CUDA, async, `enable_preemption`, a victim in PRELUDE with no pending output (allowed by `_is_preemption_candidate`), and another request taking the freed slot before the victim resumes. `test_async_pressure_preemption_with_resident_state` cannot see this. It runs three requests on three slots, and the LIFO free list returns the victim's own slot, whose token is still intact. Proposed reproduction: four requests on `max_num_seqs=2` slots with incremental allocation and a tight pool, so a newly admitted request takes the victim's slot. Then compare against the same run with a large pool. Fix: import the token in `ensure_slot` the same way as the hidden state ([#109](https://github.com/ThinkFlowLab/vllm-rlt/pull/109)). Longer term, F2 makes the slot the only home of the token.

Confirmed on an RTX 4090 with four requests on two slots, distinct prompts, and a tight versus a large pool. Two requests preempted in PRELUDE resumed in another slot and diverged from the unpreempted run (request `1`: `59` became `53`; request `2`: `22, 22` became `58, 58`). With the fix, both runs match token for token.

The reproduction first hit a separate scheduler bug, outside M5: after admission the refill policy only took PREFILL and RECURRENT, so a request restored directly into PRELUDE or CODA was never scheduled and the engine raised `scheduler made no progress`. [#101](https://github.com/ThinkFlowLab/vllm-rlt/pull/101) fixes this in `core/scheduling_policy.py`; #109 depends on it.

**F9 measurement note.** M4's F11 profile (Ouro-1.4B, async with static buffers) attributed about 300k `aten::select` and `aten::copy_` calls to "per-request hidden-state save and gather loops in `ModelRunner`". On this baseline, CUDA async PRELUDE, RECURRENT, and CODA use the gather and scatter kernels, not `_save` or `_gather`. The per-request work left on that path is line 220 (a `select` per request per step), `ensure_slot`, and the waits in `submit`. Re-profile before batching anything, so the fix targets the actual call sites.

## 6. Where this is going

RFC section 4 makes Executor, Worker, and ModelRunner one refactor unit:

| Layer | Owns | Today |
| --- | --- | --- |
| Executor | Dispatch execution and control calls. Single process, no Ray or RPC | The engine calls `ModelRunner` and `SpeculativeRunner` directly |
| Worker | Device, model load, memory plan, KV storage and buffer allocation, shutdown | Split across entrypoints, `LLMEngine.__init__`, `plan_cache`, `KVCacheManager.__init__`, `ModelRunner.__init__`. No shutdown |
| ModelRunner | Prepare and execute batches, device request state, sampling calls, submissions and results | `ModelRunner`, `AsyncState`, buffers, graphs, plus device fields on `Request` |

Planned increments. Each keeps the old entry points as adapters until every caller has moved, then removes them.

1. **This note.**
2. **F1 fix** ([#109](https://github.com/ThinkFlowLab/vllm-rlt/pull/109), draft until #101 lands). Import the token on slot assignment, with a GPU reproduction test that fails withou3. **Typed output** (F5, F6). M2's [#107](https://github.com/ThinkFlowLab/vllm-rlt/pull/107) adds `ModelRunnerOutput` in `worker/output.py` (stage, progress, prefill ranges, completion events, exit signals carrying request generation and source sequence, sampled tokens) and `Scheduler.update_from_output()`, with an engine-side adapter (`engine/output_adapter.py`) that converts today's `Submission` and value lists. This step makes the runner produce `ModelRunnerOutput` itself and removes the adapter. CUDA events and streams stay inside the runner: completion travels in the output, so the engine no longer reads `events`, `core_stream` or `boundary_stream`; boundary overlap becomes a capability flag; PD stream ordering moves to the worker-side connector (M9). Sampled tokens keep a device handle for the next prelude, and the generation check rejects stale results for a reused ID (R17). #107 decides how results are applied; M5 owns what they contain and how the runner produces them.how results are applied, M5 defines what they contain.
4. **Runner-owned request state** (F2, F4, F8, F9, F16). One slot table keyed by request generation holds hidden state, device token, and generator, for sync and async alike. `Scheduler.finish` stops clearing tensors and calls the runner's release through the engine. Preemption asks the runner to save and restore device state (RFC section 3: the execution side saves and restores, the scheduler picks the victim). PD hands the final hidden state over through a runner operation instead of writing `request.hidden_state`. Coordinate with M3 (#50) and M9 (#64).
5. **One prepared-batch builder** (F3, F10, F11), after #57 and #46. M4 step 2 exposes host row data. M5 owns the single builder that turns it into device tensors, pinned or routed, and `Workspace`, `AsyncState.prepare`, `PrefillMetadataBank`, and the dynamic path become its backends. M8's metadata types define what the attention side consumes. Graph replay calls public write and attend operations.
6. **Thin Worker and Executor** (F7, F12). The Worker owns `plan_cache` and `execution_buffer_bytes` (moved out of `core/`), creates the runner, and takes over storage tensors after M4 step 3. It provides `shutdown()`. The Executor is a plain single-process class with `submit`, `collect`, `release`, `synchronize`.
7. **Measured hot-path work** (F9, F10), each with a matched before/after comparison.

What must not change along the way:

- No `.item()`, `.tolist()`, or host token on the PRELUDE dependency path. Sampled IDs go from CODA to PRELUDE on the device.
- Delayed exit timing: submit `r`, then read `r-1`. Never launch an extra loop.
- PREFILL and RECURRENT on the core stream, PRELUDE, CODA, and finalize on the boundary stream, routing on the copy stream, joined only by events.
- `release` waits the request's last event before a slot, a bank, or KV pages are reused. Banks and workspaces wait their own `done` before reuse.
- Padding rows never get a KV address, a signal, or a sample.
- `written` advances at enqueue time and is consumed behind an event.
- Readback leases outlive the device-to-host copy.
- LAST_EXITED finalize copies the exit plane into every deeper plane. SHARED finalize only validates.
- Results are dropped for a replaced `Request` object, and coda outputs are delivered in order.

## 7. Validation

This change only adds documentation, so there is no behavior to compare. The baseline was exercised as follows.

CPU, macOS arm64, Python 3.12, torch 2.11, no Triton:

```bash
PYTHONPATH=. python -m pytest -q tests/test_async_pipeline.py tests/test_async_state.py \
  tests/test_cdb_runtime.py tests/test_engine.py tests/test_prefix_growth.py \
  tests/test_prepared_kv.py tests/test_kv_cache.py tests/test_cuda_graph.py tests/test_pd.py
```

Result at `11b1d34`: 146 passed, 76 skipped. All 76 skips are GPU-marked tests, so this run covers only the CPU async state machine, not streams, banks, or kernels.

The same files on GPU, one RTX 4090 (SM89), driver 550.144.03, torch 2.5.1+cu124, Triton 3.1.0, with `--run-gpu`: 198 passed, 18 failed, 6 skipped at `11b1d34`. All 18 failures are environmental on SM89: FlashAttention-2 paged KV needs a block size that is a multiple of 256 (12), FlashAttention-4 is unsupported (3), and the PD decode tests need a second visible GPU (3).

Existing tests that cover M5 paths, and which of them need a GPU:

| Area | Tests | Device |
| --- | --- | --- |
| Delayed exit, placeholders, ordered delivery | `test_async_pipeline.py`, `test_cdb_runtime.py::test_async_submits_next_core_before_collecting_previous_signal` | CPU and GPU |
| Device token into prelude | `test_prelude_consumes_device_sample_before_cpu_delivery` | CPU and GPU |
| Bank and slot reuse, abort, ID reuse | `test_async_state.py`, `test_abort_speculative_core_then_reuse_id` | GPU for real streams |
| Batched finalize | `test_batched_exit_copies_only_target_positions_and_depths` | GPU |
| Stream overlap | `test_cuda_boundary_and_core_can_overlap` | GPU |
| Async equals sync | `test_cuda_async_matches_synchronous_with_padding_and_sampling`, `test_cuda_dynamic_async_respects_buffers_and_matches_sync` | GPU |
| Graphs | `test_cuda_graph.py` | GPU |
| Memory plan | `test_explicit_bytes_and_cpu_auto_sizing` (CPU), `test_cuda_auto_memory_plan_and_abort_pending_work`, `test_cuda_memory_plan_reclaims_allocator_residue_on_recreation` (GPU) | Both |
| Async preemption | `test_async_pressure_preemption_with_resident_state`; #109 adds `test_async_preemption_restores_prelude_token_in_a_new_slot` | GPU. The first does not reach F1; the second does |

Not verified here: the FlashAttention paths, HIP (no UVA), FA4 packed prefill, `prefill_uva`, PD transfer, and the F9 and F10 performance claims. F1 is confirmed on GPU (see section 5).
