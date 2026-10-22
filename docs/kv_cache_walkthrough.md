# KV management and metadata

Loop-level execution requires both storage capacity and depth-specific validity.
A page being allocated does not mean its contents are ready for attention.
Transfer completion, request progress and GPU completion are also distinct.

## Ownership

| Component | Owns or performs |
| --- | --- |
| `KVCacheManager` | Physical pages and tensors, allocation identities, references, prefix entries, transfer pins, written positions, KV write/attention/finalize operations |
| `Scheduler` | Admission budgets, capacity checks and logical progress for selected token frontiers |
| `StageEngine` | Request lifecycle, pending tickets and coordination of resource retirement |
| `MetadataBuilder` | Resolve validated allocation mappings into host layouts or borrow device-owned descriptors |
| Runner and buffer owners | Materialize metadata tensors, stage transfers, select streams and retain buffers until consumers finish |
| Connector | Transport state and remote-access lifetime; ask the engine to reserve/import/release resources |

The manager still owns KV storage and attention dispatch. This refactor does not
claim a complete split into separate allocator, storage and attention services.
The builder also still accesses validated allocation internals.

Code: [manager](../vllm_rlt/core/kv_cache_manager.py),
[metadata](../vllm_rlt/worker/metadata.py),
[runner](../vllm_rlt/worker/model_runner.py),
[transfer connector](../vllm_rlt/pd/connector.py).

## Capacity, validity and lifetime

`required_blocks(tokens)` counts physical blocks across storage planes.
LAST_EXITED has one plane per recurrent depth; SHARED has one plane. A request's
maximum input extent is prompt length plus `max_tokens - 1`: the last sampled
output is not fed back into the model.

`allocate()` establishes the allocation and initial pages. Non-incremental
allocation covers the full extent; incremental allocation can start with fewer
pages and grow through `ensure_capacity()`. The scheduler budgets existing
requests' future growth before admitting new work or reserving incoming KV.
Without preemption, this includes decode growth for imports awaiting admission.

The watermark is spare capacity within the KV pool used as admission headroom.
It is zero when incremental allocation is disabled, and admission can waive it
for an idle stage. GPU memory utilization instead controls initial pool sizing.
See the [admission budget](scheduler.md#admission-available-memory-is-not-the-whole-budget).

Written-position state records validity per physical plane and layer. Address
preparation does not mark a position written. `_WrittenPositions` can represent
out-of-order writes without pretending an unwritten gap is a valid prefix.
Allocation identity checks reject metadata that refers to a released or replaced
allocation.

`free()` removes a request's ownership, but a transfer pin can defer physical
release. `pin_transfer()`/`unpin_transfer()` protect pages borrowed by transport;
they do not pin CPU memory. Prefix references can also keep pages alive after a
request finishes. GPU readers must be retired by the execution owner before
storage is reused; reference counts alone do not prove CUDA completion.

## One metadata contract for prompt, decode and mixed work

`KVMetadata` carries positions, write block/offset addresses, block tables,
context lengths and, for packed rows, query sequence offsets. Prefill and decode
use this same contract. Their row counts, sequence lengths and storage layouts
can differ without requiring separate metadata types.

`MetadataBuilder.build()` computes layouts from request IDs, depths and token
positions. Buffer owners fill and upload reusable storage, then bind tensor
views to metadata. The runner coordinates CPU preparation, GPU preparation and
execution; the metadata module does not schedule requests or choose streams.
The manager's `_prepare_batch()` convenience path delegates to the same builder.

`DeviceKVMetadata` is the intentional exception for device-owned speculative
positions. `prepare_device()` borrows and validates those tensors without reading
positions back to the CPU. The speculative runner keeps their allocations alive
until GPU consumers finish. A common model interface does not require rebuilding
GPU-resident state on the host.

## Early exit, prefixes and rollback

`finalize_token()` copies the last computed depth into skipped LAST_EXITED planes
and updates validity. The asynchronous copy path uses `mark_finalized()` for the
same bookkeeping after arranging the required device dependencies. SHARED needs
no skipped-plane copies. A finalized logical frontier is not an implicit host
wait for every copy.

`publish_prefix()` can attach a completion event; pending prefixes become reusable
only when ready. `lookup_prefix()` can update prefix/LRU bookkeeping. Reclaimable
prefix-only pages contribute to `num_free_blocks`; `exclusive_prefix_blocks()`
accounts for pages that become non-evictable when a request claims that prefix.
A failed page claim must not evict prefixes when the total reclaimable capacity
is already insufficient.

Synchronous speculative commit can truncate rejected suffix validity through
`truncate_suffix()`. Asynchronous speculation instead advances actual positions
on device and excludes rejected positions through its descriptors. Collecting an
older round must not rewind a newer in-flight round. See
[speculative execution](async-speculative.md#execution-and-ownership).

## Transfer and preemption

A transfer reserves the whole prompt mapping and pins it before compute
admission. The connector checks completion before the engine marks imported
positions valid and queues the local Request for admission. The same Request is
reused across chunk notifications. Receiving all data, granting a compute slot
and releasing source pages are separate transitions.

For cancellation, resources remain protected until local GPU users and remote
access have ended. A transport commit alone is not completion evidence. See the
[PD handoff and protocol](scheduler.md#3-request-handoff-between-reusable-stages).

Preemption excludes transfer-pinned allocations and other unsafe candidates.
`KVSnapshot` retains copied pages and written-position state; `restore()` checks
capacity and page count and restores into a compatible empty allocation. The
preemption manager handles copying, while Scheduler owns queue transitions.

## Validation entry points

Relevant regressions include `tests/test_kv_cache.py`, `tests/test_prefix_growth.py`,
`tests/test_split_prefill.py`, `tests/test_transfer_connector.py`, and
`tests/test_tensor_transfer.py`. CPU tests do not establish GPU overlap or NIXL
transport behavior. Current qualification requires running the relevant tests on the target revision.

Earlier M4-refactor results and their limitations are retained in the
[validation history](runtime-history.md#historical-validation-m4-refactor).
