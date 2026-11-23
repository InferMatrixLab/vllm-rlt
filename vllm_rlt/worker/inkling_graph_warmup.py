"""Warm native Inkling kernels and graph entries before serving requests."""

import gc
import logging
import time

import torch

from vllm_rlt.config import cuda_graph_rows
from vllm_rlt.kernels.inkling2_vllm import warmup_fa4
from vllm_rlt.request import Operation
from vllm_rlt.worker.packed_graph import packed_capture_sizes

logger = logging.getLogger(__name__)


def _run_graph(owner, inputs, metadata):
    call = owner.select(inputs, metadata)
    owner.prepare_inputs(call, inputs, startup=True)
    return owner.execute(call, inputs)


def _warm_decode(runner, ids, budget):
    cache = runner.cache_manager
    model = runner.model
    limit = len(ids)
    sizes = sorted(range(1, limit + 1), key=lambda size: (size != limit, size != 1, -size))
    positions = [0] * limit
    tokens = torch.zeros(limit, device=cache.device, dtype=torch.long)
    hidden = torch.zeros(
        (limit, model.config.hidden_size), device=cache.device, dtype=runner.state_dtype
    )
    multi_pass = model.config.total_ut_steps > 1
    uniform_depths = (0, 1) if multi_pass else (0,)
    for size in sizes[:budget]:
        batch = cache._prepare_batch(ids[:size], positions[:size], positions[:size])
        _run_graph(runner.prelude_graphs, tokens[:size], batch)
        _run_graph(runner.coda_graphs, hidden[:size], batch)
        for depth in uniform_depths:
            if len(runner.graphs.entries) >= budget:
                break
            metadata = cache._prepare_batch(ids[:size], [depth] * size, positions[:size])
            _run_graph(runner.graphs, hidden[:size], metadata)
    if multi_pass:
        for size in sizes:
            if len(runner.graphs.entries) >= budget:
                break
            if size > 1:
                metadata = cache._prepare_batch(
                    ids[:size], [1] + [0] * (size - 1), positions[:size]
                )
                _run_graph(runner.graphs, hidden[:size], metadata)


def _warm_packed(runner, scheduler, ids, limit, budget):
    owner = runner.packed_graphs
    cache = runner.cache_manager
    model = runner.model
    sizes = sorted(
        packed_capture_sizes(limit),
        key=lambda size: (size != limit, bool(size & (size - 1)), -size),
    )
    tokens = torch.zeros(limit, device=cache.device, dtype=torch.long)
    hidden = torch.zeros(
        (limit, model.config.hidden_size), device=cache.device, dtype=runner.state_dtype
    )
    positions = list(range(limit))
    all_prompt = [ids[0]] * limit
    multi_pass = model.config.total_ut_steps > 1
    mixed = scheduler.mode == "mixed"
    compute_gate = mixed and not owner.speculative_core
    for size in sizes:
        if size < owner.min_rows or len(owner.entries) >= budget:
            continue
        first = cache._prepare_batch(
            all_prompt[:size], [0] * size, positions[:size], packed_prefill=True
        )
        owner.forward(Operation.PRELUDE, tokens[:size], first, startup=True)
        if len(owner.entries) < budget:
            owner.forward(
                Operation.RECURRENT, hidden[:size], first, compute_gate=compute_gate, startup=True
            )
        if multi_pass and len(owner.entries) < budget:
            later = cache._prepare_batch(
                all_prompt[:size], [1] * size, positions[:size], packed_prefill=True
            )
            owner.forward(
                Operation.RECURRENT, hidden[:size], later, compute_gate=compute_gate, startup=True
            )
        if mixed and multi_pass and size > 1 and len(ids) > 1 and len(owner.entries) < budget:
            staggered = cache._prepare_batch(
                [ids[1], *all_prompt[: size - 1]],
                [1] + [0] * (size - 1),
                [0, *positions[: size - 1]],
                packed_prefill=True,
            )
            owner.forward(
                Operation.RECURRENT,
                hidden[:size],
                staggered,
                compute_gate=compute_gate,
                startup=True,
            )
        if len(owner.entries) < budget:
            owner.forward(Operation.CODA, hidden[:size], first, startup=True)


@torch.inference_mode()
def warmup_inkling_graphs(runner, scheduler):
    """Capture a bounded set of graphs using temporary, valid KV allocations."""
    cache = runner.cache_manager
    execution = runner.execution_config
    available = cache.num_blocks // cache.storage_depths
    if available == 0:
        raise ValueError("Inkling CUDA Graph warmup needs KV pages for one request")
    decode_rows = min(
        cuda_graph_rows(execution),
        scheduler.max_num_seqs,
        scheduler.max_num_batched_tokens,
        available,
    )
    packed_rows = min(
        runner.model.config.max_position_embeddings,
        scheduler.max_num_batched_tokens,
        (available - decode_rows + 1) * cache.block_size,
    )
    budget = execution.cuda_graph_max_graphs
    ids = [f"__inkling_warmup_{i}__" for i in range(decode_rows)]
    prefix_queries, prefix_hits = cache.prefix_queries, cache.prefix_hits
    allocated = []
    start = time.monotonic()
    if runner.packed_graphs is not None:
        logger.info("Compiling Inkling FA4 variants before serving")
        with torch.cuda.device(cache.device):
            warmup_fa4(
                runner.model.config, cache, scheduler.max_num_batched_tokens, scheduler.max_num_seqs
            )
    logger.info("Capturing Inkling startup CUDA Graphs")
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for i, request_id in enumerate(ids):
            if not cache.allocate(request_id, packed_rows if i == 0 else 1):
                raise RuntimeError("cannot allocate temporary Inkling warmup KV pages")
            allocated.append(request_id)
        _warm_decode(runner, ids, budget)
        if runner.packed_graphs is not None:
            _warm_packed(runner, scheduler, ids, packed_rows, budget)
    finally:
        try:
            torch.cuda.synchronize(cache.device)
        finally:
            for request_id in allocated:
                cache.free(request_id)
            cache.prefix_queries, cache.prefix_hits = prefix_queries, prefix_hits
            if gc_enabled:
                gc.enable()

    owners = {
        "prelude": runner.prelude_graphs,
        "core": runner.graphs,
        "coda": runner.coda_graphs,
        "mixed": runner.mixed_graphs,
        "packed": runner.packed_graphs,
    }
    for owner in owners.values():
        if owner is not None:
            owner.allow_capture = False
            owner.replays = owner.fallbacks = 0
    return dict(
        elapsed_seconds=time.monotonic() - start,
        entries={name: len(owner.entries) for name, owner in owners.items() if owner is not None},
    )
