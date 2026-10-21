# M9 — PD and KV transfer walkthrough

Code evidence for [RFC #111](https://github.com/InferMatrixLab/vllm-rlt/issues/111)
and module M9 in [#32](https://github.com/InferMatrixLab/vllm-rlt/issues/32).
Interface decisions belong to the RFC; this document records the implementation,
its ownership and lifetime contracts, and remaining review/validation work.

## 1. Baseline and scope

Pinned source: `InferMatrixLab/vllm-rlt` at
`21f59c9928f73426dfce0bf9a7fe537758b88ed3`, including the unified-stage refactor
merged in #118.

Follow [unified execution](unified_execution.md) for the architecture:
**stage is the deployment/resource boundary; loop is the local execution and
scheduling unit**. `RequestPhase` describes request progress, `Operation`
describes computation, and `StageConfig` describes deployment capability.

| Source | Responsibility |
| --- | --- |
| `vllm_rlt/pd/engine.py` | `PDEngine`, `Peer`, `Pending`: routing, admission reservations, cross-worker ordering, output mapping and aggregate profiling |
| `vllm_rlt/pd/connector.py` | `PDStageConnector`, `RequestTransfer`: transfer bookkeeping, staging, backpressure, receipt and safe retirement |
| `vllm_rlt/pd/transport.py` | `NixlConnector`, `Transfer`, `kv_segments`, `partition_segments`: registered memory, descriptors and NIXL WRITE lifecycle |
| `vllm_rlt/pd/config.py` | Same-host pools, transfer/control limits and deadlines |
| `vllm_rlt/engine/{llm_engine,worker}.py` | Shared `StageClient` / `StageWorker`: process/channel ownership, generic control and local service loop |
| `vllm_rlt/engine/{stage_engine,connector}.py` | Local execution and transfer operations; `StageConnector` / `TransferConnector` protocols |
| `vllm_rlt/core/{scheduler,kv_cache_manager}.py` | Local admission, queues, allocation, validity and leases |

`LLMEngine` is an alias for `StageEngine`, not a second execution path. Serving
uses `request_ids()`, `has_request()` and `stats()`; `PDEngine` exposes no
scheduler/cache facade.

This is a documentation-only review, not a new feature qualification or a
performance result. The test inventory below distinguishes CPU contracts, GPU
tests in the repository, and external evidence; existence of a test is not a
pass.

## 2. Ownership and local execution

```text
frontend -> PDEngine
              |-- StageClient(P) -- control pipe --> StageWorker(P)
              |                                      |-- StageEngine
              |                                      `-- PDStageConnector
              |                                               |
              |                                          NIXL WRITE
              |                                               v
              `-- StageClient(D) -- control pipe --> StageWorker(D)
                                                     |-- StageEngine
                                                     `-- PDStageConnector
```

Each worker owns a complete model replica and one local execution engine.
`StageWorker.run()` handles bounded control input, advances connector progress,
calls `StageEngine.step()`, and emits outputs. Scheduler selection, submission,
progress application, async retirement, persistent buffers and CUDA Graph
execution stay in the shared path.

| State | Authoritative owner / borrowing rule |
| --- | --- |
| External request ID → transfer generation; P/D assignment; deadlines | `PDEngine`; each request gets a fresh UUID `tid` |
| Process, channel, readiness, generic control replies | `StageClient`; the PD coordinator orders connected-worker shutdown |
| Request registry, phase queues and compute admission | Scheduler; StageEngine coordinates local lifecycle and progress |
| KV mappings, prefix validity, references and transfer leases | KVCacheManager; transfer allocation uses scheduler admission budgets |
| Model execution, reusable inputs/results, stream/event dependencies | Shared executor, runners and controllers; not duplicated by PD |
| Transfer slot, destination mapping, sequences, commit/receipt flags | `PDStageConnector` / `RequestTransfer` |
| Registered tensors, NIXL handles and in-flight byte accounting | `NixlConnector`; retained until local/remote access has ended |

`RequestTransfer.request` borrows the stage-local `Request`; it is not another
copy of its loop/token progress. Connector-owned hidden staging is intentionally
separate from runner storage. P copies the final prompt hidden state into a
staging slot; D's engine binds its received slot for coda. Slot lifetime must
cover both transport access and any local execution borrowing that storage.

### Shared transfer operations

| Operation | Contract at the pinned baseline |
| --- | --- |
| `StageEngine.reserve_transfer(..., receiving=...)` | Creates one local Request, reserves prompt mapping under the scheduler's growth budget, and pins KV. Failure returns `None` without retaining the request/allocation. Reservation grants no compute slot. |
| `PDStageConnector.can_schedule(request)` | Nonblocking export readiness: live identity, not cancelled, no queued chunks, last event complete, and byte pressure below the limit. Does not edit scheduler queues. |
| `prefill_submitted(..., phase, start, progress, event, stream)` | Borrows already-submitted execution under its event/lease. Logical progress is already applied; submission is not device completion. |
| `StageEngine.accept_import(request, hidden, cached_tokens=...)` | Checks identity/state, publishes imported KV validity, binds hidden state and joins the shared admission queue while retaining `RECEIVING` and its lease. |
| Scheduler admission of an imported Request | Grants compute capacity, unpins the transfer lease and queues CODA. Local and imported requests share admission policy. |
| `release_transfer_compute(request, ready_event=...)` | Checks identity and the final staging-read event, releases runner compute state, sets `compute_released`; does not release KV or staging. |
| `suspend_transfer(request)` | Removes future scheduling while retaining resources borrowed by pending work. |
| `release_transfer(request)` | Caller must establish remote-access safety first. Engine retires local state; async speculation may defer completion. Connector frees its slot only after successful retirement. |

The connector reads the scheduler registry to detect retirement. This is a
remaining boundary to review; a replacement operation needs a concrete
ownership/lifetime need, not merely a desire to hide a field.

## 3. Coordinator state and control plane

`Pending.phase` is a coarse routing label:

```text
waiting -> reserving -> prefill -> decode
```

Cancellation and P/D release are independent flags; `received` is a milestone,
not another phase. Finished output can remove the external request before both
workers release. `has_unfinished_requests()` therefore tracks transfers, not
just externally active requests. Once both releases arrive, the coordinator
stores timing metrics and retires the generation.

### Reservation credits

`Peer.reserved_blocks` and `slots` are coordinator reservations, not measured
worker KV usage or local compute occupancy. `_admit()` charges both peers before
sending D's reservation command:

- P: `depths * ceil(prompt_tokens / block_size)` blocks.
- D: `depths * ceil((prompt_tokens + max_tokens - 1) / block_size)` blocks.
- One transfer slot on each peer; eligibility also checks capacity and watermark.

Selection uses block-reservation fraction then slot fraction. A bounded waiting
scan supports priority ordering and age protection. Credits return on each
peer's first `released` event, not on `prefill_complete`, `received`, activation,
or final output. Duplicate releases cannot decrement them twice.

Prefix hits and incremental physical allocation do not make these counters
allocator measurements. `PDEngine.stats()` explicitly reports `reserved_blocks`;
physical usage belongs to worker/cache statistics.

### Message scopes

| Scope | Commands / events | Routing and evidence |
| --- | --- | --- |
| Stage lifecycle | `ready`, `connected`, `fatal`, `stopped`; `connect`, `stop` | StageClient owns common state; connector connects transport peers; coordinator owns cross-peer ordering |
| Generic host control | `control(method, args, kwargs, control_id, session_id)` → `reply` | `submit_control()` / `take_control_reply()` / `discard_control()`; no transfer lookup |
| Transfer reservation | `reserve` → `reserved(tables, slot, cached_tokens)`; `prefill` → `prefill_started` | `tid` identifies the generation; the responding peer must match the assigned pair |
| Transfer progress | `prefill_complete`, `commit(count)`, `received`, `activated`, `ack` | Different evidence/lifetimes; see next section |
| Cancellation / release | `cancel`, `cancel_safe`, `released`, `rejected` | Release is generation-scoped; rejection is treated as a coordinator/worker capacity mismatch |
| Output | `output(request_id=tid, output=...)` | Coordinator replaces the wire generation with the external request ID |

Profiling uses generic `control` / `reply`, not a PD-specific wire protocol.
PDEngine owns fan-out, session checks, aggregate manifests, partial-start
rollback and deadlines. StageClient retains only replies for pending control
IDs; discarded calls do not accumulate late replies.

`_message()` ignores retired transfer generations, rejects a response from a
worker outside the assigned pair, and suppresses cancelled outputs. It does not
constitute a general schema validator for every payload or a promise that every
command is idempotent. For example, duplicate reservations are rejected, while
matching commits and repeated receive notifications are tolerated.

## 4. Handoff and completion evidence

```text
Coordinator                 P stage                         D stage
    | reserve ------------------------------------------------>|
    |<----------------------- reserved(tables, slot, prefix) ---|
    | prefill ---------------->|
    |                          | shared admission / prefill
    |                          | KV chunks + final hidden ---->|
    |<-- prefill_complete -----|                               |
    |<-- commit(count) --------|                               |
    | commit(count) ------------------------------------------>|
    |                          |               all receipts + import binding
    |<--------------------------------------------- received --|
    | ack -------------------->|                               |
    |                          |               shared compute admission
    |<-------------------------------------------- activated --|
    |<-- released -------------|                               |
    |<--------------------------------------- output/released --|
```

Receipt notifications can precede the control-plane commit. P release and D
activation are not ordered relative to each other. In particular,
**`received → ack`, not `activated → ack`**: a receiver waiting for compute
capacity must not unnecessarily hold sender KV.

### Distinct milestones

1. **Submission:** the engine applies logical prompt progress and supplies its
   completion event. The connector must not expose pending writes as ready data.
2. **Staging readiness:** `prefill_submitted()` waits on the producer event in
   the supplied stream; the final chunk copies prompt hidden state and records
   a new event covering that copy. The model event alone is insufficient.
3. **Bounded WRITE submission:** `progress()` queries readiness and posts at
   most one bounded chunk per transfer per tick, with round-robin ordering.
   `can_schedule()` supplies backpressure without a separate PD scheduler.
4. **Commit:** all queued writes have been submitted and the final prefill
   boundary is known. The advertised count is not proof of NIXL completion.
5. **Receipt/import readiness:** D has a valid positive commit count and every
   sequence in `[0, count)`. It rejects wrong-peer/invalid sequences, conflicting
   counts and new sequences after readiness. `accept_import()` makes the received
   prefix valid, binds hidden state and queues ordinary admission; D emits
   `received` independently of capacity.
6. **Activation:** Scheduler admits the received Request to CODA. Connector
   observes it leaving `RECEIVING` and reports `activated`. D owns first-token
   sampling and subsequent RNG use, including local speculative rounds.
7. **P retirement:** requires acknowledgment, committed submission, released
   compute state and no pending NIXL handle for the generation. Cancellation
   uses its separate safe-drain path.
8. **D retirement:** engine completion/cancellation retires local execution;
   connector returns the staging slot and reports `released` only when safe.

`prefill_complete` reports compute-state release after the final staging read,
not delivery. Its boundary/overlap metrics should not be confused with NIXL
handle completion or end-to-end latency.

### Prefix hits, ranges and layouts

P and D may have different valid prefixes. D advertises its hit; P transfers
any additional prefix cached only on P, then required recomputed suffix ranges.
`queue_chunk()` clips transfer start to D's cached frontier. Final hidden state
is still required: cached KV alone cannot produce the first sample.

`plane_block_tables()` is a public logical-KV view. Its outer dimension is
storage planes, not necessarily loop depth: LAST_EXITED has per-depth planes;
SHARED has one. `kv_segments()` walks those tables and `[start, end)` positions:
full blocks cover all layers, partial blocks emit per-layer ranges without
sending unused tails. `partition_segments()` aligns source/target byte lengths
and bounds bytes and descriptor count without an intermediate KV tensor copy.

NIXL registers K, V and hidden storage. Peer startup checks model/weight/config
fingerprints and compatible geometry. `submit()` retains a handle before
posting, including on submission failure. `poll()` releases successful handles,
accounts bytes and reads generation/sequence notifications. A local handle's
completion is distinct from the receiver's receipt/import state.

## 5. Cancellation, failure and lease ordering

| Situation | Required ordering in the implementation |
| --- | --- |
| Still waiting at coordinator | Drop the generation; neither peer owns resources. |
| D reservation outstanding | Cancel D and P. If `reserved` arrives after cancellation, coordinator sends `cancel_safe`: it never dispatched prefill for that reservation. |
| P executing or transferring | Clear unsent chunks and suspend future scheduling; retain staging/KV until both `last_event` and submitted NIXL accesses finish. |
| D not yet fully received | Retain receive resources until P release establishes that remote writers stopped and coordinator sends `cancel_safe`. |
| D already ready/active | Complete receipt already proves remote-write completion; ask engine to retire local work. Async speculative retirement may still delay slot release. |
| Retired generation / external ID reuse | UUID wire identity prevents old commit/receipt/output from targeting the new request; stage operations additionally check Request identity. |
| Transport error / timeout | Raise without ordinary pool reuse/deregistration. Connected worker retains its owners while reporting `fatal`; coordinator terminates/joins P workers before D workers. |
| Graceful close | Cancel and drain transfers, finish profiling/artifacts, request worker stop, then join. Any failed drain takes the fatal shutdown path. |

“Quarantine” here means fail-closed retention and ordered process termination,
not an implemented recoverable quarantine allocator or retry protocol.
`NixlConnector.close()` refuses in-flight handles; connector close refuses
pending transfers. A connected StageClient cannot independently shut down live
transport owners, including through destructor ordering.

Transfer leases defer `KVCacheManager.free()` and exclude allocations from
preemption. Imports retain their leases while awaiting compute admission;
admission unpins them, after which normal supported preemption can apply.

## 6. Remaining work

### Concrete discrepancies and review candidates

- **Model metadata:** `StageWorker` advertises `model_type` and the actual model
  config, but `PDEngine.__init__()` reconstructs every ready payload as
  `OuroConfig`. A Nanbeige config contains `num_loops`; reconstructing
  `OuroConfig(**asdict(NanbeigeConfig()))` raises `TypeError` on this baseline.
  This was reproduced at the CPU metadata boundary, not in a real-checkpoint
  multi-GPU launch. Resolve model-neutral coordinator metadata separately from
  any protocol cleanup.
- **Acknowledgment wording:** `TransferConnector.acknowledge()` describes
  “downstream activation”, while PDEngine acknowledges `received`. Align the
  contract wording with the receipt/admission distinction; do not introduce
  activation-gated sender release.
- **Borrowing boundaries:** connector registry reads and transport geometry
  access (`NixlConnector.__init__` reads cache tensor strides/pointers) deserve
  explicit lifetime/ownership review. They are not, by themselves, demonstrated
  runtime failures or justification for new APIs.
- **Protocol typing:** commands/events are dictionaries and routing phases are
  strings. Keep draft PR #131 only if its revised scope addresses a concrete
  maintenance/validation need. It must target `PDStageConnector`, preserve shared
  control/reply routing, and not duplicate generic profiling. Draft PR #132's
  reservation naming/statistics motivation is already covered by
  `reserved_blocks`/`stats()`; keep only a demonstrably useful remainder.

The next M9 increment should follow these findings and validation gaps. Keep
fixes, structural changes and qualification work separately identifiable.

## 7. Compatibility and validation evidence

The launcher supports same-host P/D GPU pools with complete replicas. Constrained
PD speculation and asynchronous speculation on D are supported; mixed is a local
batching policy, not a new PD role or a promise of arbitrary deployment graphs.

Use [runtime constraints](runtime.md#fixed-loop-self-speculative-decoding) and
[async speculation](async-speculative.md), including LAST_EXITED, fixed target
depth, refill/no-preemption requirements for speculation, and greedy/device-
context requirements for async speculation. SHARED requires atomic prompt
execution; split prefill/mixed require LAST_EXITED. Preserve supported settings,
not an unrestricted cross-product of model/backend/graph/async options.

### Existing test inventory

| Evidence | What it exercises / limitation |
| --- | --- |
| `tests/test_transfer_connector.py` | CPU commit/receipt reordering, shared admission capacity/priority, lease retention, cancellation, stale generations, staging events, late writes and safe close; mocked transport does not establish real CUDA/NIXL ordering. |
| `tests/test_stage_engine.py` | Shared handoff execution, import/export identity, backpressure, growth budgets, client lifecycle/control, connected-owner retention and imported-request preemption; device-specific cases require GPU execution. |
| `tests/test_pd.py` | CPU descriptors/configuration/reservation accounting; GPU PD parity, cancellation/reuse, prefix transfer, split-prefill boundaries/overlap, worker failure and speculative variants. |
| `tests/test_profiling.py` | Generic worker session control, aggregate PD acknowledgments/rollback, artifact and serving control behavior. |
| Draft PR #119, head `e1a94eac` | Author-reported real Ouro-1.4B P→D agreement: four tests on three H100s. Historical evidence, not a rerun on this baseline. |

#119 compares solo requests at matched forward geometry for token IDs, exit
depths, finish semantics and seeded sampling. Concurrent-load tests target
liveness/shape rather than assume BF16 batch/chunk invariance. Future validation
must report the first divergence and distinguish numerics from handoff defects;
non-speculative agreement does not qualify speculative first-round, bonus-token,
rollback or async cancellation behavior.

### Remaining validation

1. Reconcile model startup/support claims, beginning with the metadata mismatch.
2. Audit coordinator routing and shared control correlation, including stale,
   duplicate, reordered and conflicting feedback and exactly-once credits.
3. Reuse existing connector/engine fixtures for cancellation at reservation,
   staging, receipt-wait, activation-wait and speculative retirement boundaries.
4. Update the real-checkpoint harness to this baseline; qualify valid KV ranges,
   final hidden state, sampling/RNG and release ordering under the actual
   supported feature combinations.
5. For hot-path changes only, compare matched unprofiled latency/throughput and
   transfer bytes, with profile attribution where needed. Do not add blocking
   synchronization or abandon buffers/graphs/overlap to simplify an interface.

No new GPU, real-checkpoint or performance qualification is supplied by this
documentation update. Historical results for #119, #131 or #132 must retain their
original revision and configuration scope.
