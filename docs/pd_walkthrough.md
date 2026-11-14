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
| `StageEngine` | Local request lifecycle, import binding, compute release and asynchronous retirement; coordinates scheduler and runner operations |
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

## Configuration and startup

Deployment entry points create and validate `ModelConfig` before spawning
workers. It retains the concrete model configuration and delegates vocabulary,
loop and context limits, including derived properties. `PDEngine` uses those
limits for request validation and passes the configuration through each
`StageClient` to its worker. Spawned processes receive copies, not shared mutable
configuration.

Checkpoint workers use the supplied concrete configuration to load weights
without reparsing JSON. Config-only inputs construct models; module inputs retain
the supplied weights. Models and `StageEngine` use the concrete configuration.
Unsupported PD model families are rejected explicitly before workers start.

Workers report runtime results in `ready`: actual KV capacity and storage depths,
admission limits, registered transport information and compatibility fingerprints.
The coordinator does not reconstruct model configuration from this message. It
still waits for every worker, checks model/weight/KV compatibility, connects peers
and waits for connection acknowledgment before accepting work. Entry configuration
alone does not establish runtime compatibility.

Code: [model configuration](../vllm_rlt/models/config.py),
[stage client](../vllm_rlt/engine/llm_engine.py),
[worker startup](../vllm_rlt/engine/worker.py).

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
2. **Staging readiness.** At an export boundary, `prefill_submitted()` enqueues
   a stream wait on the producer event; the final chunk copies the prompt hidden
   state and records a new event covering that copy. This is a device dependency,
   not a blocking host wait.
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

## Prefill scheduling and export boundaries

The scheduler chooses execution boundaries through `ExecutionBatch.loop_count`.
Ordinary prompt ranges can traverse their remaining depths in one submission;
staged scheduling exposes intermediate depth boundaries. These are local execution
choices, not additional cross-process scheduling commands.

With `SchedulerConfig(wavefront_chunked_prefill=True)`, WCPB can batch ready ranges
from several chunks and depths. The scheduler applies grouped progress for each
request before the engine notifies the connector. The callback uses
`PrefillProgress.transfer_start` when provided, so its export origin is not inferred
from an arbitrary range in the batch.

`num_prefilled_tokens` tracks the final-depth frontier, while
`prefill_reclaimable_tokens` advances at completed chunk boundaries. Intermediate
prelude/depth submissions do not by themselves queue a transfer. When
`progress.chunk_complete` is set, the connector queues the newly exportable KV
range under the producer event; `progress.prompt_complete` additionally stages
the final hidden state. The connector keeps at most one bounded WRITE submission
per transfer per tick, independently of how many local ranges shared a batch.

Imported prompts initialize the same prefill frontiers before entering D's shared
admission path. WCPB requires LAST_EXITED KV; it does not change the receipt,
acknowledgment or retirement protocol.

See [split and wavefront prefill](split-prefill-wavefront.md) for the hidden-state
ring, dependency frontiers and numerical limitations. Different BF16 batch/chunk
geometries are not guaranteed to produce identical tokens; transfer correctness
and cross-geometry numerical agreement are separate claims.

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

Logical completion is not physical retirement: an asynchronous speculative
request may be `FINISHED` while submitted work still retains its storage. The
connector keeps its staging slot until the engine retires that ownership;
frontend `has_request()` alone is not a storage-lifetime check.

## Profiling and control

Generic host operations — profiling sessions, statistics, lifecycle — use the
shared `control`/`reply` channel with control and session identities, never
transfer routing. The coordinator fans profiling control out to all workers,
aggregates per-rank manifests and rolls back partial starts.

## Scope

The launcher supports same-host P/D pools with full model replicas; this does not
qualify cross-host transport. Model-family admission is checked before spawning,
while backend and execution compatibility are checked during worker initialization.

Decode-side speculation requires LAST_EXITED, full target depth, fixed-depth
`ouro` exits and refill scheduling. Asynchronous speculation additionally requires
CUDA, greedy sampling, a backend supporting device-owned context lengths and no
preemption. Synchronous speculation does not impose that same preemption
rejection; transfer leases still protect receiving/exporting allocations.
Mixed scheduling excludes speculative rounds.

SHARED completes all depths for each prompt position before advancing; staged
mixed/WCPB paths require LAST_EXITED. The prefill-only stage has its own prompt
scheduling policy, so its mode does not imply decode interleaving.

See [runtime](runtime.md) and
[async speculation](async-speculative.md) for execution details and validation
limits. Feature availability is not evidence that every checkpoint, batching
geometry or feature combination has been validated.
