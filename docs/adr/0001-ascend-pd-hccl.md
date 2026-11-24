# ADR 0001: Use HCCL for Initial Ascend 2P2D PD

- Status: Accepted for the current Ascend PD implementation
- Date: 2026-11-23

## Context

PR #168 provides the Ascend eager execution path, paged KV and NPU stream and
event handling. The existing PD transport registers CUDA allocations through
NIXL. The target Ascend environment provides HCCL point-to-point communication
for NPU tensors.

The PD contract still requires destination reservation before Prefill work,
transfer generations, receipt after destination writes complete, and safe
retirement after all local and remote users release their allocations.

## Decision

Add an explicit Ascend `HCCL` transport for same-host 2P2D. Each worker joins
one HCCL process group. The transport packs descriptor ranges into NPU uint8
buffers, submits asynchronous point-to-point sends and receives, then copies
received buffers into the destination KV and hidden tensors. CUDA deployments
keep the existing NIXL transport.

Workers run a synchronized one-byte `batch_isend_irecv` warmup for every
Prefill/Decode pair before serving requests. HCCL creates pair subgroups for
the subsequent traffic, and the connector serializes transfers across tids so
each subgroup sees the same operation order on both ranks.

The qualified Ascend scope is Ouro-1.4B, BF16, fixed four loops, greedy full
prefill and `last_exited` KV. HCCL receive plans are created after D reserves
its destination and posted lazily after P starts its matching sends. The fixed
full-prefill geometry provides matching source and destination chunk boundaries.

## Consequences

The implementation can validate multi-worker output equality, transfer byte
accounting and resource retirement on one host. It is limited to full prefill
and the fixed BF16 scope above. Cross-host transport and unsupported execution
features require separate validation.
