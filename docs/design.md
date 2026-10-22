# Model and KV execution semantics

The [architecture overview](unified_execution.md) explains stage ownership and
loop-level scheduling. The [scheduler guide](scheduler.md) describes admission
and batch selection; the [runtime guide](runtime.md) covers preparation,
submission, asynchronous execution and supported combinations. This page focuses
on what the selected operations mean for the model and its KV history.

## Operations and request progress

PRELUDE prepares token hidden states. RECURRENT traverses the shared transformer
core at a selected depth. CODA consumes the final hidden state and produces
vocabulary logits. Some models also have attention/KV in prelude or coda.

`ModelRunner.submit()` and `SpeculativeRunner.submit()` return execution tickets whose
`collect()` method yields a typed `ModelRunnerOutput` after readback (`SpeculativeRunner.execute()`
also returns `ModelRunnerOutput` directly). `Scheduler` applies completed results through
`update_from_output()` and records asynchronous submissions through
`update_after_submission()`, while `StageEngine` coordinates submission,
pending tickets, connector hooks, and retirement.

Prompt tokens run the full recurrent depth. Atomic prefill executes a chunk's
prelude and recurrence in one submission; split prefill exposes intermediate
scheduling boundaries. The final prompt hidden state produces the first output
through CODA. That prompt token is not redundantly processed as a decode input.
A sampled token enters KV only when it is fed back through the model.

Each ordinary decode request has one token undergoing recurrence. Requests can
share recurrent execution while retaining different token positions and loop
depths. Speculation instead manages a provisional candidate span, with committed
progress separated from device-side draft/verify state.

## Gate and model contract

Ouro's embedding is the prelude. One recurrent operation runs all physical transformer layers and the shared end-of-loop normalization. The normalized hidden state persists for the next loop. Coda applies the LM head to the exited state.

Decode exits when `1 - product(1 - sigmoid(gate_i)) >= exit_threshold`, subject to the loop bounds. Every loop contributes, including those before the minimum allowed exit depth. Default adaptive bounds are two through four loops; threshold `1.0` selects fixed depth explicitly. Token position and its RoPE phase stay unchanged while loop depth advances.

Checkpoint loading defaults to BF16; explicit dtype overrides are supported. Existing model objects retain their dtype, and engine KV storage uses that same dtype. RMSNorm and RoPE use FP32 intermediates; Triton attention accumulates in FP32.

## Depth-aware KV

In LAST_EXITED, a block table maps each `(depth, logical token block)` to a
physical block. A row reads history at its own depth, so a batch does not need
one shared depth. On early exit, the final computed KV is copied into skipped
depths; later tokens must not encounter holes in those planes.

SHARED uses one physical plane and overwrites it through recurrence. To preserve
its causal meaning, prompt execution finishes all depths for one position
before advancing that request. Its ordering is not interchangeable with packed
depth-major LAST_EXITED prefill.

Allocated capacity is distinct from valid history. Attention lengths must exclude
unwritten or rejected positions, even when their pages exist. Prefix reuse,
incremental growth, transfer pins and snapshot/restore are described in
[KV management](kv_cache_walkthrough.md).

## Correctness entry points

Model tests compare selected operations with reference execution; runtime tests
cover mixed depths, prompt splitting, early-exit copies and request isolation.
See [runtime validation](runtime.md#replay-and-validation-commands) for
commands and explicitly dated historical results. Architectural unification is
not itself evidence of equal throughput or identical reduced-precision outputs.
