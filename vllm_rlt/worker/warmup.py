"""Bounded startup execution using the serving runners and scratch KV leases."""

from dataclasses import replace

import torch

from vllm_rlt.config import cuda_graph_rows
from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
from vllm_rlt.request import ExecutionBatch, Operation, Request, RequestPhase, StateSpan
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.cuda_graph import graph_capture_sizes


def graph_owners(runner, speculation=None):
    return tuple(_graph_owners(runner, speculation).values())


def _graph_owners(runner, speculation):
    return {
        f"{label}.{name}": owner
        for label, component in (("runner", runner), ("speculation", speculation))
        if component is not None
        for name in ("graphs", "prelude_graphs", "coda_graphs")
        if (owner := getattr(component, name, None)) is not None
    }


def drain(runner, speculation=None):
    runner.synchronize()
    runner.submissions.drain()
    if speculation is not None:
        if hasattr(speculation, "synchronize"):
            speculation.synchronize()
        elif speculation.compute_stream is not None:
            speculation.compute_stream.synchronize()
        speculation.submissions.drain()
    for owner in graph_owners(runner, speculation):
        stream = getattr(owner, "capture_stream", None) or getattr(owner, "stream", None)
        stream.synchronize()


@torch.inference_mode()
def prepare_runner(runner, scheduler, speculation=None, *, phase=None):
    """Warm valid prefill/decode paths without admitting or publishing requests.

    Coverage is bounded by configured limits and committed KV. Exact graph keys
    not encountered here retain the existing bounded lazy-capture allowance.
    """
    cache, config = runner.cache_manager, runner.model.config
    if cache._allocations:
        raise RuntimeError("runtime preparation requires an idle KV pool")
    named_owners = _graph_owners(runner, speculation)
    owners = tuple(named_owners.values())
    queries, hits = cache.prefix_queries, cache.prefix_hits
    devices = [runner.device.index] if runner.device.type == "cuda" else []
    gaps = []
    decode_workloads = set()
    extra = 2 + (2 * (speculation.config.num_speculative_tokens + 1) if speculation else 0)
    capacity = min(
        config.max_position_embeddings, cache.num_blocks // cache.storage_depths * cache.block_size
    )
    if capacity < 1:
        raise ValueError("KV capacity cannot back one startup token at every depth")
    if capacity <= extra:
        extra = min(extra, max(0, capacity - 1))
        gaps.append("KV capacity limits decode/speculative coverage")
    prefill_limit = min(scheduler.prefill_chunk_size, scheduler.max_num_batched_tokens)
    prefill = tuple(sorted({1, min(prefill_limit, capacity - extra)}))
    batch_history = min(cache.block_size + 1, capacity - extra)
    history_limit = max(
        cache.block_size + 1, prefill_limit, (getattr(config, "sliding_window", 0) or 0) + 1
    )
    history = min(history_limit, capacity - extra)
    max_batch = min(
        scheduler.max_num_seqs,
        scheduler.max_num_batched_tokens,
        cache.num_blocks // cache.required_blocks(batch_history + extra),
    )
    graph_batch = min(max_batch, cuda_graph_rows(runner.execution_config))
    execution = replace(runner.execution_config, cuda_graph_max_batch_size=graph_batch)
    sizes = tuple(sorted({*graph_capture_sizes(execution, (graph_batch,)), max_batch}))
    if max_batch < min(scheduler.max_num_seqs, scheduler.max_num_batched_tokens):
        gaps.append("KV capacity limits decode batch coverage")
    if prefill[-1] < prefill_limit or history < history_limit:
        gaps.append("KV capacity limits prefill/history coverage")
    captured = tuple((owner, owner.replays, owner.fallbacks) for owner in owners)
    counters = (runner.split_prefill_submissions, runner.split_prefill_submission_waits)
    speculative_stats = replace(speculation.stats) if speculation is not None else None
    speculative_counters = {
        name: getattr(speculation, name)
        for name in ("submitted_before_collect", "discarded_rounds", "readback_waits")
        if speculation is not None and hasattr(speculation, name)
    }

    def execute(operation, spans, loops=0):
        ticket = runner.submit(batch=ExecutionBatch(operation, tuple(spans), loops))
        output = ticket.collect()
        if operation == Operation.CODA and ticket.device_values is not None:
            for row, span in enumerate(spans):
                span.request.input_token_tensor = ticket.device_values[row]
        return output

    def run_case(count, length, decode, *, concurrent=False, stochastic=False):
        requests = []
        tickets = []
        try:
            for index in range(count):
                request = Request(
                    f"__warmup_{index}",
                    [0] * length,
                    SamplingParams(
                        max_tokens=max(1, extra),
                        ignore_eos=True,
                        seed=index,
                        temperature=0.8 if stochastic else 0.0,
                        top_k=config.vocab_size if stochastic else -1,
                        top_p=0.9,
                    ),
                )
                requests.append(request)
                if not cache.allocate(request.request_id, length + extra):
                    raise RuntimeError("insufficient KV blocks for startup preparation")
            chunk_size = min(
                prefill_limit, scheduler.max_num_batched_tokens - (count - 1 if concurrent else 0)
            )
            for start in range(0, length, chunk_size):
                end = min(length, start + chunk_size)
                spans = [
                    StateSpan(
                        request,
                        start,
                        end - start,
                        0,
                        chunk_start=start if runner.staged_prefill else None,
                        retain_chunk=runner.staged_prefill,
                        input_tokens=tuple(request.prompt_token_ids[start:end]),
                    )
                    for request in requests
                ]
                # One request at a time respects the prefill token budget.
                for index, span in enumerate(spans):
                    pending = concurrent and end == length and index == count - 1
                    if runner.staged_prefill:
                        execute(Operation.PRELUDE, (span,))
                        operation = Operation.RECURRENT
                    else:
                        operation = Operation.PRELUDE
                    loops = 1 if runner.staged_prefill else config.total_ut_steps
                    if runner.staged_prefill:
                        # Wavefront/mixed work yields at each existing depth boundary.
                        for depth in range(config.total_ut_steps - 1):
                            execute(operation, (replace(span, depth=depth),), loops)
                        span = replace(span, depth=config.total_ut_steps - 1)
                    if pending:
                        # Serving also submits independent boundary work before core.
                        for request in requests[:-1]:
                            request.loops_done = config.total_ut_steps
                        tickets.append(
                            runner.submit(
                                batch=ExecutionBatch(
                                    Operation.CODA,
                                    tuple(StateSpan(r, length - 1, 1, 0) for r in requests[:-1]),
                                )
                            )
                        )
                        tickets.append(
                            runner.submit(batch=ExecutionBatch(operation, (span,), loops))
                        )
                    else:
                        execute(operation, (span,), loops)
            for request in requests:
                request.loops_done = config.total_ut_steps
            spans = [StateSpan(request, length - 1, 1, 0) for request in requests]
            if concurrent:
                # Independent coda on boundary can overlap the final prompt core.
                # Keep both tickets alive until their device readers have retired.
                for ticket in tickets:
                    ticket.collect()
                output = execute(Operation.CODA, spans[-1:])
                request = requests[-1]
                request.generated_token_ids.extend(output.token_ids)
                if extra:
                    span = StateSpan(request, length, 1, 0)
                    execute(Operation.PRELUDE, (span,))
                    for depth in range(config.total_ut_steps - 1):
                        execute(Operation.RECURRENT, (replace(span, depth=depth, emit_gate=True),))
                    # Decode core graphs avoid the host launch cost of fused prefill.
                    tickets.append(
                        runner.submit(
                            batch=ExecutionBatch(
                                Operation.RECURRENT,
                                (replace(span, depth=config.total_ut_steps - 1, emit_gate=True),),
                            )
                        )
                    )
                    execute(Operation.CODA, spans[:-1])
                    tickets[-1].collect()
                    decode_workloads.add((1, length))
                return
            output = execute(Operation.CODA, spans)
            for request, token in zip(requests, output.token_ids):
                request.generated_token_ids.append(token)
            if not decode or not extra:
                return
            ordinary_steps = min(extra, 2)
            for offset in range(ordinary_steps):
                position = length + offset
                execute(
                    Operation.PRELUDE,
                    [StateSpan(r, position, 1, 0) for r in requests],
                )
                for depth in range(config.total_ut_steps):
                    execute(
                        Operation.RECURRENT,
                        [StateSpan(r, position, 1, depth, emit_gate=True) for r in requests],
                    )
                output = execute(Operation.CODA, [StateSpan(r, position, 1, 0) for r in requests])
                for request, token in zip(requests, output.token_ids):
                    request.generated_token_ids.append(token)
                decode_workloads.add((count, position))
            if speculation is not None:
                width = min(
                    speculation.config.num_speculative_tokens + 1,
                    (extra - ordinary_steps) // 2,
                    scheduler.max_num_batched_tokens // count,
                )
                if width < 1:
                    gaps.append("KV capacity cannot cover a speculative round")
                    return
                for request in requests:
                    request.phase = RequestPhase.SPECULATIVE
                pipeline = speculation.config.mode == "pipeline"
                rounds = 2 if hasattr(speculation, "buffers") else 1
                if pipeline:
                    rounds = speculation.config.target_loops // speculation.config.draft_loops
                for offset in range(rounds):
                    batch = SchedulerOutput(
                        RequestPhase.SPECULATIVE,
                        [
                            ScheduledRequest(
                                request,
                                request.position
                                if pipeline
                                else length + ordinary_steps + offset * width,
                                width,
                            )
                            for request in requests
                        ],
                    )
                    ticket = speculation.submit(batch)
                    tickets.append(ticket)
                    if pipeline:
                        output = ticket.collect()
                        for request, result in zip(requests, output.speculative):
                            request.generated_token_ids.extend(result.token_ids)
                        if hasattr(ticket, "retire"):
                            ticket.retire()
                for ticket in tickets:
                    ticket.collect()
        finally:
            drain(runner, speculation)
            for ticket in tickets:
                if hasattr(ticket, "retire"):
                    ticket.retire(discard=True)
            if speculation is not None and hasattr(speculation, "discard_buffers"):
                speculation.discard_buffers()
            for request in requests:
                if speculation is not None:
                    speculation.release(request)
                runner.release(request.request_id)
                cache.free(request.request_id)

    try:
        with torch.random.fork_rng(devices=devices):
            for owner in owners:
                owner.startup = True
            if speculation is not None and hasattr(speculation.graphs, "capture_sizes"):
                if not speculation.graphs.entries:
                    speculation.graphs.prepare()
                    speculation.coda_graphs.prepare()
            for length in prefill:
                run_case(1, length, False)
            for count in sizes:
                run_case(count, batch_history, True)
            if history != batch_history:
                run_case(1, history, True)
            run_case(sizes[-1], batch_history, False, stochastic=True)
            if phase is not None:
                phase("captured")
                run_case(1, prefill[-1], False)
                run_case(sizes[-1], batch_history, True)
                phase("replayed")
            concurrent_batches = ()
            if runner.execution_config.async_scheduling and runner.execution_config.multi_stream:
                concurrent_limit = (
                    min(
                        scheduler.max_num_seqs,
                        scheduler.max_num_batched_tokens,
                        cache.num_blocks // cache.required_blocks(prefill[-1] + extra),
                    )
                    - 1
                )
                if concurrent_limit > 0:
                    concurrent_batches = graph_capture_sizes(
                        replace(execution, cuda_graph_max_batch_size=concurrent_limit),
                        (concurrent_limit,),
                    )
                    # Prepare exact shapes, then isolate concurrent replay peaks.
                    for count in concurrent_batches:
                        run_case(count + 1, prefill[-1], False, concurrent=True, stochastic=True)
                    if phase is not None:
                        phase("concurrent_prepared")
                    for count in concurrent_batches:
                        run_case(count + 1, prefill[-1], False, concurrent=True, stochastic=True)
                    if phase is not None:
                        phase("concurrent")
                else:
                    gaps.append("request/token/KV limits prevent independent stream overlap")
    finally:
        drain(runner, speculation)
        cache.prefix_queries, cache.prefix_hits = queries, hits
        runner.split_prefill_submissions, runner.split_prefill_submission_waits = counters
        if speculation is not None:
            speculation.stats = speculative_stats
            for name, value in speculative_counters.items():
                setattr(speculation, name, value)
        coverage = {
            name: dict(
                keys=tuple(key for key, entry in owner.entries.items() if entry.graph is not None),
                replays=owner.replays - replays,
                fallbacks=owner.fallbacks - fallbacks,
                remaining_entries=max(
                    0, owner.execution.cuda_graph_max_graphs - len(owner.entries)
                ),
            )
            for name, (owner, replays, fallbacks) in zip(named_owners, captured)
        }
        for owner, replays, fallbacks in captured:
            owner.startup = False
            owner.replays, owner.fallbacks = replays, fallbacks
        for buffers in runner.batch_input_buffers:
            buffers.reset()
        for buffers in runner.packed_metadata_buffers:
            buffers.clear()
        if runner.async_state is not None:
            state = runner.async_state
            state.submissions.drain()
            state.finalize_submissions.drain()
            for buffers in (*state.routing_buffers, *state.finalize_buffers):
                buffers.imports.clear()
    if owners:
        gaps.append("unplanned exact graph shapes retain bounded lazy capture and graph reserve")
    gaps.append(
        "contexts and batch/history combinations beyond the reported workloads "
        "retain the estimated transient reserve"
    )
    if speculation is not None and speculation.config.mode == "block":
        gaps.append("random block speculation retains the estimated transient reserve")
    if runner.execution_config.async_scheduling and speculation is not None:
        gaps.append("mixed speculative/prompt overlap retains the estimated transient reserve")
    return dict(
        prefill_tokens=prefill,
        decode_batches=sizes,
        history_tokens=history,
        decode_history_tokens=batch_history,
        decode_workloads=tuple(
            dict(batch_size=count, history_tokens=position)
            for count, position in sorted(decode_workloads)
        ),
        concurrent_batches=concurrent_batches,
        graph_coverage=coverage,
        coverage_gaps=tuple(dict.fromkeys(gaps)),
    )
