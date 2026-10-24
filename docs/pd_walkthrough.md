# PD disaggregation and KV transfer

Prefill/decode (PD) disaggregation runs the prompt phase and the generation
phase in separate stage processes on the same host, each with a complete model
replica and its own KV pool. A CPU coordinator routes requests across them;
NIXL WRITEs move KV and the final prompt hidden state directly between
registered worker memory. Control channels carry only metadata and outputs.

Stage is the deployment and resource boundary; loop is the local scheduling
unit. This page covers the cross-stage handoff; local execution follows
[unified execution](unified_execution.md) and the
[loop scheduler](scheduler.md).

## Ownership

| Component | Owns or performs |
| --- | --- |
| `PDEngine` (coordinator) | External request ID → transfer generation mapping, peer assignment, admission reservations, cross-worker ordering, output mapping, aggregate profiling |
| `StageClient` | Worker process, control channel, readiness, generic control replies; connected-worker shutdown is ordered by the coordinator |
| `PDStageConnector` / `RequestTransfer` | Transfer slot, destination mapping, chunk sequences, commit/receipt flags, cancellation state, safe retirement |
| `NixlConnector` | Registered tensors, segment descriptors, NIXL handles, notifications, in-flight bytes |
| Scheduler | Request registry, phase queues, admission, local selection; never invoked directly by transport code |
| KVCacheManager | Allocation, block mappings, prefix validity, references, transfer leases |
| Executor / runners | Execution, persistent buffers, stream/event dependencies, result retirement |

`RequestTransfer.request` borrows the stage-local `Request`; it is not a second
copy of loop/token progress. Connector-owned hidden staging is separate from
runner storage: the prefill stage copies the final prompt hidden state into a
staging slot, and the decode engine binds its received slot for coda.

Code: [coordinator](../vllm_rlt/pd/engine.py),
[connector](../vllm_rlt/pd/connector.py),
[transport](../vllm_rlt/pd/transport.py),
[connector protocol](../vllm_rlt/engine/connector.py),
[stage engine](../vllm_rlt/engine/stage_engine.py).

## Handoff

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

The decode stage reserves first: it allocates its full prompt-plus-output
mapping, pins a transfer lease, and replies with block tables, a hidden receive
slot and its own cached prefix length. Only then does the coordinator dispatch
prefill to the prefill stage.

Completion evidence stays distinct at every step:

1. **Submission.** The engine applies logical prompt progress and supplies a
   completion event. Submission is not device completion.
2. **Staging readiness.** `prefill_submitted()` waits on the producer event in
   the producer's stream; the final chunk copies the prompt hidden state and
   records a new event covering that copy.
3. **Bounded WRITE submission.** The connector posts at most one bounded chunk
   per transfer per tick with round-robin ordering; `can_schedule()` applies
   backpressure without touching scheduler queues.
4. **Commit.** All queued writes have been submitted. The advertised count is
   not proof of NIXL completion.
5. **Receipt and import readiness.** D holds a known commit count and every
   sequence in `[0, count)`, then `accept_import()` publishes KV validity,
   binds hidden state and queues shared admission. D reports `received`,
   which triggers P's `ack` independently of D's compute admission: a receiver
   waiting for capacity must not hold sender KV.
6. **Activation.** The scheduler admits the imported request alongside local
   work, unpins its lease and queues CODA. The connector observes the phase
   change and reports `activated`. D owns first-token sampling and RNG.
7. **P retirement.** Requires the acknowledgment, committed submission,
   released compute state and no outstanding NIXL handles for the generation.
8. **D retirement.** Local execution retires; the staging slot returns only
   when safe.

Receipt notifications may precede the control-plane commit. P release and D
activation are unordered relative to each other.

## Ranges, prefixes and layouts

P and D may hold different valid prefixes. D advertises its hit; P transfers
any additional prefix cached only on P plus the recomputed suffix, and
`queue_chunk()` clips the transfer start to D's frontier. Final hidden state is
always included: cached KV alone cannot produce the first sample.

`plane_block_tables()` exposes one table per storage plane — per loop depth for
LAST_EXITED, a single plane for SHARED. `kv_segments()` maps a valid
`[start, end)` range to byte segments: full blocks cover all layers, partial
blocks emit per-layer ranges so unused tails are never advertised as
initialized. `partition_segments()` bounds bytes and descriptor count per
WRITE without an intermediate KV copy.

Peer startup verifies model/weight/configuration fingerprints and KV geometry;
mismatches abort connection.

## Reservations and statistics

The coordinator charges both peers before dispatching D's reservation:
`depths * ceil(tokens / block_size)` blocks per peer (D additionally counts
`max_tokens - 1` output positions) plus one transfer slot each. These are
coordinator reservations, not measured worker usage; prefix hits and
incremental allocation do not reduce them. Credits return on each peer's first
`released` event and cannot be released twice.

`PDEngine.stats()` reports `reserved_blocks`; physical usage belongs to
worker-side cache statistics. Frontend queries go through `request_ids()`,
`has_request()` and `stats()`.

## Cancellation, failure and leases

| Situation | Ordering |
| --- | --- |
| Waiting at coordinator | Drop the generation; neither peer owns resources. |
| D reservation outstanding | Cancel both. A late `reserved` gets `cancel_safe`: no prefill was dispatched. |
| P executing or transferring | Stop future scheduling; retain staging/KV until the final staging event and submitted NIXL handles finish. |
| D not fully received | Retain receive resources until P's release proves remote writers stopped, then `cancel_safe`. |
| D ready or active | Completed receipt proves remote writes ended; retire local work. Async speculative retirement may still delay slot release. |
| External ID reuse | Wire generations are unique UUIDs; local transitions also check `Request` identity, so retired messages cannot target a reused ID. |
| Transport error or timeout | Fail closed: no pool reuse or deregistration. Connected workers retain owners while reporting the failure; the coordinator terminates and joins P before D. |

Transfer leases defer `KVCacheManager.free()` and exclude allocations from
preemption. Imports keep their leases while awaiting compute admission;
admission unpins them, after which normal supported preemption applies.

## Profiling and control

Generic host operations — profiling sessions, statistics, lifecycle — use the
shared `control`/`reply` channel with control and session identities, never
transfer routing. The coordinator fans profiling control out to all workers,
aggregates per-rank manifests and rolls back partial starts.

## Scope

The launcher supports same-host P/D pools with full model replicas. Decode-side
speculation, including asynchronous speculation, is supported under the
constraints in [runtime](runtime.md#fixed-loop-self-speculative-decoding)
(LAST_EXITED, fixed target depth, refill, no preemption; async adds greedy and
device-context restrictions). SHARED requires atomic prefill; mixed scheduling
excludes speculative rounds. Ouro and Nanbeige checkpoints are supported;
Huginn is not — the coordinator fails fast when a stage's ready report lacks
the limits it validates against.
