"""Stage-local submission dispatch and shared model execution.

Executor routes scheduled batches to their runner. ModelRuntime executes prepared
model calls for ordinary and speculative runners; it never updates Request progress.
"""

from collections.abc import Generator, Iterable
from dataclasses import dataclass, replace

import torch

from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import ExecutionBatch, Operation, RequestPhase, StateSpan


def build_execution_batch(scheduled_batch: SchedulerOutput, total_depths: int) -> ExecutionBatch:
    """Bind selected Request references to immutable operation/position/depth rows.

    This creates scalar execution descriptors, not new Requests or KV metadata.
    The runner prepares its backend-specific metadata once from these bindings.
    """
    spans = []
    operations = set()
    loop_counts = set()
    planned_loops = scheduled_batch.loop_count
    for scheduled_request in scheduled_batch.requests:
        phase = scheduled_batch.phase_for(scheduled_request)
        request = scheduled_request.request
        chunk = phase in (RequestPhase.PREFILLING_PRELUDE, RequestPhase.PREFILLING_RECURRENT)
        prompt_input = phase in (RequestPhase.PREFILL, RequestPhase.PREFILLING_PRELUDE)
        operation = {
            RequestPhase.PREFILL: Operation.PRELUDE,
            RequestPhase.PREFILLING_PRELUDE: Operation.PRELUDE,
            RequestPhase.PRELUDE: Operation.PRELUDE,
            RequestPhase.PREFILLING_RECURRENT: Operation.RECURRENT,
            RequestPhase.RECURRENT: Operation.RECURRENT,
            RequestPhase.CODA: Operation.CODA,
        }.get(phase)
        if operation is None:
            raise ValueError(f"phase {phase} does not describe a model operation")
        depth = (
            scheduled_request.depth if scheduled_request.depth is not None else request.loops_done
        )
        if operation != Operation.RECURRENT:
            # PRELUDE starts from 0. CODA has no recurrent work; 0 addresses
            # boundary-layer KV, not the request's actual completed-loop count.
            depth = 0
        position = (
            scheduled_request.token_start
            if chunk or phase == RequestPhase.PREFILL or scheduled_request.depth is not None
            else request.position
        )
        execution_loops = (
            planned_loops
            if planned_loops is not None
            else total_depths
            if phase == RequestPhase.PREFILL
            else int(operation == Operation.RECURRENT)
        )
        spans.append(
            StateSpan(
                request=request,
                token_start=position,
                token_count=scheduled_request.token_count,
                depth=depth,
                chunk_start=scheduled_request.chunk_start if chunk else None,
                retain_chunk=chunk and depth + execution_loops < total_depths,
                retain_last=scheduled_request.completes_prompt if chunk else True,
                emit_gate=phase == RequestPhase.RECURRENT,
                input_tokens=tuple(
                    request.prompt_token_ids[position : position + scheduled_request.token_count]
                )
                if prompt_input and (not chunk or execution_loops > 0)
                else None,
                prefill_hidden_row_offset=(
                    scheduled_request.prefill_hidden_row_offset if chunk else None
                ),
            )
        )
        operations.add(operation)
        loop_counts.add(execution_loops)
    if len(operations) != 1 or len(loop_counts) != 1:
        raise ValueError("batch requests must share an operation and execution boundary")
    return ExecutionBatch(operations.pop(), tuple(spans), loop_counts.pop())


class Executor:
    """Dispatch local batches, preserving the runner's device-resident paths.

    The round controller shares the model runtime and RNG with the runner.
    It expands a speculative round locally, without a stage handoff.
    """

    def __init__(self, runner, *, speculation=None):
        self.runner = runner
        self.speculation = speculation

    def submit(self, scheduled_batch):
        if scheduled_batch.is_speculative:
            if self.speculation is None:
                raise ValueError("speculative submission requires a configured round controller")
            return self.speculation.submit(scheduled_batch)
        batch = build_execution_batch(scheduled_batch, self.runner.model.config.total_ut_steps)
        return self.runner.submit(scheduled_batch, batch=batch)


@dataclass(frozen=True)
class ForwardBatch:
    """One compatible model call, independent of deployment role.

    CPU preparation may first populate host metadata views. GPU preparation binds
    their device views before execution; the batch is the input description in
    both phases, rather than being reconstructed from a separate loop-input type.

    PRELUDE can include a sequence of RECURRENT views. RECURRENT starts from supplied hidden.
    CODA projects supplied hidden. KV views carry allocation identity and may
    use host-planned or device-owned positions. Graph executors must match the
    operation and gate mode. All tensors/results are borrowed, not snapshots.
    """

    operation: Operation
    hidden: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    metadata: Iterable = ()
    prelude_coda_kv_metadata: object = None
    generators: object = None
    hidden_buffer: torch.Tensor | None = None
    compute_gate: bool = False
    graphs: object = None
    graph_calls: tuple = ()
    compute_logits: bool = True


@dataclass(frozen=True)
class ForwardResult:
    hidden: torch.Tensor | None = None
    gate_logits: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class ModelRuntime:
    def __init__(self, model, cache):
        self.model = model
        self.cache = cache

    def prepare(self, batch: ForwardBatch) -> ForwardBatch:
        """Choose graph/fallback for every call before execution.

        Only the first hidden input exists now. Later recurrent inputs must be
        staged after their producer, so graph-private buffers cannot be overwritten
        by pre-staging multiple loops into the same entry.
        """
        if batch.graphs is None:
            return batch
        if batch.operation == Operation.CODA:
            calls = (batch.graphs.select(batch.hidden, batch.prelude_coda_kv_metadata),)
        else:
            metadata = tuple(batch.metadata)
            batch = replace(batch, metadata=metadata)
            calls = tuple(batch.graphs.select(batch.hidden, kv) for kv in metadata)
        if calls:
            batch.graphs.prepare_inputs(calls[0], batch.hidden)
        return replace(batch, graph_calls=calls)

    def forward(self, batch: ForwardBatch) -> ForwardResult:
        hidden, gate = batch.hidden, None
        try:
            if batch.operation == Operation.CODA:
                if hidden is None:
                    raise ValueError("CODA requires hidden state")
                if batch.graphs is not None:
                    logits = batch.graphs.execute(batch.graph_calls[0], hidden)
                elif batch.prelude_coda_kv_metadata is not None:
                    logits = self.model.coda_prepared(
                        hidden,
                        batch.prelude_coda_kv_metadata,
                        self.cache,
                        compute_logits=batch.compute_logits,
                    )
                else:
                    logits = self.model.coda(hidden) if batch.compute_logits else None
                return ForwardResult(hidden=hidden, logits=logits)
            if batch.operation == Operation.PRELUDE:
                if batch.tokens is None:
                    raise ValueError("PRELUDE requires token IDs")
                if batch.prelude_coda_kv_metadata is not None:
                    hidden = self.model.prelude_prepared(
                        batch.tokens[: len(batch.prelude_coda_kv_metadata.position_ids)],
                        batch.prelude_coda_kv_metadata,
                        self.cache,
                        generators=batch.generators,
                    )
                    if batch.hidden_buffer is not None:
                        batch.hidden_buffer.zero_()
                        batch.hidden_buffer[: len(hidden)].copy_(hidden)
                        hidden = batch.hidden_buffer
                else:
                    hidden = self.model.prelude(batch.tokens)
            elif batch.operation != Operation.RECURRENT:
                raise ValueError(f"unsupported operation {batch.operation}")
            if hidden is None:
                raise ValueError("RECURRENT requires hidden state")
            for index, kv in enumerate(batch.metadata):
                if batch.graphs is not None:
                    call = batch.graph_calls[index]
                    if not call.inputs_ready:
                        batch.graphs.prepare_inputs(call, hidden)
                    hidden, gate = batch.graphs.execute(call, hidden)
                else:
                    hidden, gate = self.model.recurrent_prepared(
                        hidden, kv, self.cache, compute_gate=batch.compute_gate
                    )
            return ForwardResult(hidden, gate)
        finally:
            if isinstance(batch.metadata, Generator):
                batch.metadata.close()
