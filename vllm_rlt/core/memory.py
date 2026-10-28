"""Startup KV budget: explicit overrides or CUDA peak profiling plus headroom."""

import math

import torch

from vllm_rlt.config import cuda_graph_rows
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.worker.buffers import execution_buffer_bytes
from vllm_rlt.worker.cuda_graph import speculative_capture_sizes


def budget_blocks(available_bytes, bytes_per_block):
    blocks = int(available_bytes) // bytes_per_block
    if blocks < 1:
        raise ValueError(
            "no memory remains for KV blocks; reduce execution limits or memory reserve"
        )
    return blocks


def graph_memory_reserve(model, scheduler, execution, speculative=None):
    """Headroom per pool plus retained speculative coda logits across all entries.

    This is a conservative planning allowance, not an allocator upper bound.
    Explicit KV capacities bypass automatic planning as before.
    """
    if not execution.cuda_graphs:
        return 0
    pools = 1 + bool(getattr(model, "requires_boundary_kv", False))
    retained_logits = 0
    if speculative is not None:
        pools += 2
        rows = min(
            cuda_graph_rows(execution),
            scheduler.max_num_seqs * (speculative.num_speculative_tokens + 1),
            scheduler.max_num_batched_tokens,
        )
        entries = min(rows, execution.cuda_graph_max_graphs)
        # Each entry retains a separate full-vocabulary output, even though the
        # graph allocator pool can reuse intermediate storage between replays.
        retained_rows = (
            sum(speculative_capture_sizes(execution, scheduler, speculative))
            if execution.async_scheduling
            else entries * rows
        )
        retained_logits = (
            retained_rows * model.config.vocab_size * next(model.parameters()).element_size()
        )
    return pools * execution.cuda_graph_memory_reserve_bytes + retained_logits


@torch.inference_mode()
def plan_cache(
    model, cache, scheduler, execution, backend, *, speculative=None, staged_prefill=None
):
    parameter = next(model.parameters())
    config, device = model.config, parameter.device
    per_block = (
        2
        * config.num_hidden_layers
        * cache.block_size
        * config.num_key_value_heads
        * config.head_dim
        * parameter.element_size()
    )
    if cache.num_blocks is not None:
        return cache.num_blocks, {"source": "blocks", "bytes_per_block": per_block}
    if cache.kv_cache_memory_bytes is not None:
        blocks = budget_blocks(cache.kv_cache_memory_bytes, per_block)
        return blocks, {"source": "bytes", "bytes_per_block": per_block}
    if device.type != "cuda":
        return 256, {"source": "cpu_default", "bytes_per_block": per_block}
    torch.cuda.synchronize(device)
    # Return unused allocator segments before measuring driver-free memory.
    # In particular, a previous engine may have left its KV pool cached here.
    # Live allocations and non-releasable segments remain reserved.
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    free_before, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    count = min(
        scheduler.max_num_batched_tokens, scheduler.max_num_seqs * config.max_position_embeddings
    )
    if execution.pad_to_power_of_two:
        count = 1 << (count - 1).bit_length()
    lengths = [
        min(config.max_position_embeddings, count - start)
        for start in range(0, count, config.max_position_embeddings)
    ]
    probe = KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_loops=config.total_ut_steps,
        num_blocks=sum(math.ceil(length / cache.block_size) for length in lengths)
        * config.total_ut_steps,
        block_size=cache.block_size,
        device=device,
        dtype=parameter.dtype,
        backend=backend,
        recurrent_layers=getattr(model, "recurrent_kv_layers", None),
    )
    ids, positions = [], []
    for index, length in enumerate(lengths):
        rid = f"profile-{index}"
        probe.allocate(rid, length)
        ids.extend([rid] * length)
        positions.extend(range(length))
    tokens = torch.zeros(count, dtype=torch.long, device=device)
    if getattr(model, "requires_boundary_kv", False):
        boundary = probe._prepare_batch(ids, [0] * count, positions)
        hidden = model.prelude_prepared(tokens, boundary, probe)
    else:
        hidden = model.prelude(tokens)
    for depth in range(config.total_ut_steps):
        hidden, _ = model.recurrent(hidden, ids, [depth] * count, positions, probe)
    logits = (
        model.coda_prepared(hidden, boundary, probe)
        if getattr(model, "requires_boundary_kv", False)
        else model.coda(hidden)
    )
    torch.cuda.synchronize(device)
    peak = max(0, torch.cuda.max_memory_allocated(device) - allocated)
    del hidden, logits, probe
    # Profiling includes a conservative temporary KV allocation. Also budget for
    # long-context reference attention (Triton's online softmax has bounded scratch).
    context_scratch = 0
    if backend == "torch":
        context_scratch = (
            config.max_position_embeddings
            * config.num_attention_heads
            * (8 * config.head_dim + 8)
            * 4
        )
    buffers = execution_buffer_bytes(
        config,
        scheduler,
        cache,
        execution,
        parameter.element_size(),
        staged_prefill=staged_prefill,
    )
    # Dynamic metadata at the largest supported context is small but not zero.
    metadata_width = math.ceil(config.max_position_embeddings / cache.block_size)
    metadata_rows = scheduler.max_num_batched_tokens
    wide_capacity = metadata_rows * (config.total_ut_steps + 2)
    narrow_capacity = metadata_rows * config.total_ut_steps * (metadata_width + 2) + 1
    metadata = 3 * (
        (1 << (wide_capacity - 1).bit_length()) * 8 + (1 << (narrow_capacity - 1).bit_length()) * 4
    )
    budget = min(free_before, int(total * cache.gpu_memory_utilization) - reserved)
    concurrent_peak = peak * (2 if execution.async_scheduling and execution.multi_stream else 1)
    graph_reserve = graph_memory_reserve(model, scheduler, execution, speculative)
    budget -= (
        concurrent_peak
        + context_scratch
        + buffers
        + metadata
        + cache.memory_reserve_bytes
        + graph_reserve
    )
    # More blocks than all concurrent full-context requests can use only waste
    # memory, particularly for tiny-model GPU tests with very small pages.
    useful_blocks = (
        math.ceil(config.max_position_embeddings / cache.block_size)
        * (config.total_ut_steps if cache.layout == "last_exited" else 1)
        * scheduler.max_num_seqs
    )
    blocks = min(budget_blocks(budget, per_block), useful_blocks)
    return blocks, dict(
        source="cuda_profile",
        bytes_per_block=per_block,
        profile_peak_bytes=peak,
        concurrent_peak_bytes=concurrent_peak,
        max_useful_blocks=useful_blocks,
        cuda_graph_reserve_bytes=graph_reserve,
        static_buffer_bytes=buffers,
        context_scratch_bytes=context_scratch,
        kv_budget_bytes=int(budget),
    )
