"""Startup KV budget: explicit overrides or CUDA peak profiling plus headroom."""

import math

import torch

from vllm_rlt.config import cuda_graph_rows
from vllm_rlt.core.kv_cache_manager import FP8_E4M3_MAX, KVCacheManager, kv_bytes_per_block
from vllm_rlt.worker.buffers import execution_buffer_bytes
from vllm_rlt.worker.cuda_graph import speculative_capture_sizes


def budget_blocks(available_bytes, bytes_per_block):
    blocks = int(available_bytes) // bytes_per_block
    if blocks < 1:
        raise ValueError(
            "no memory remains for KV blocks; reduce execution limits or memory reserve"
        )
    return blocks


def measurement_budget(report):
    """Driver free/total bytes and coverage gaps from a startup preparation report.

    ``report`` is step 1 of RFC #192 (``StageEngine.prepare_runtime``): boundary
    samples taken after draining, with the allocator cache retained. Only the
    last sample sizes capacity, because it is the state after warmup and capture.
    """
    if not report:
        raise ValueError("a measured KV plan requires a startup preparation report")
    phases = report.get("phases") or ()
    if not phases:
        raise ValueError("the startup report contains no phase samples")
    final = phases[-1]
    free, total = final.get("driver_free_bytes"), final.get("device_total_bytes")
    if type(free) is not int or type(total) is not int or total <= 0 or free < 0:
        raise ValueError("the final startup phase lacks valid driver free/total bytes")
    return free, total, tuple(report.get("coverage_gaps") or ())


def measured_kv_capacity(
    *,
    report,
    bytes_per_block,
    useful_blocks,
    gpu_memory_utilization,
    transient_bytes,
    safety_bytes,
):
    """Blocks to commit after warmup and capture, with every margin recorded.

    The physical budget is the driver's remaining memory, which already excludes
    weights, execution buffers, captured graphs and any live mapping, so none of
    them is subtracted a second time. ``transient_bytes`` covers execution above
    that persistent baseline and ``safety_bytes`` the fixed reserve.
    """
    for name, value in (
        ("bytes_per_block", bytes_per_block),
        ("useful_blocks", useful_blocks),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("transient_bytes", transient_bytes),
        ("safety_bytes", safety_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    free, total, gaps = measurement_budget(report)
    resident = total - free
    # What the utilization cap still allows, bounded by measured free memory.
    remaining = min(free, int(total * gpu_memory_utilization) - resident)
    budget = remaining - transient_bytes - safety_bytes
    if budget < bytes_per_block:
        raise ValueError(
            "no KV capacity fits the measured budget: "
            f"remaining={remaining} transient={transient_bytes} "
            f"safety={safety_bytes} block={bytes_per_block}"
        )
    blocks = min(budget // bytes_per_block, useful_blocks)
    return blocks, dict(
        source="measured_warmup",
        bytes_per_block=bytes_per_block,
        driver_free_bytes=free,
        device_total_bytes=total,
        resident_bytes=resident,
        utilization_cap_bytes=int(total * gpu_memory_utilization),
        remaining_instance_bytes=remaining,
        transient_bytes=transient_bytes,
        safety_bytes=safety_bytes,
        kv_budget_bytes=budget,
        max_useful_blocks=useful_blocks,
        coverage_gaps=gaps,
    )


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


def _probe_forward(model, cache, tokens, ids, positions):
    count = len(ids)
    boundary = None
    if getattr(model, "requires_boundary_kv", False):
        boundary = cache._prepare_batch(ids, [0] * count, positions)
        hidden = model.prelude_prepared(tokens, boundary, cache)
    else:
        hidden = model.prelude(tokens)
    for depth in range(model.config.total_ut_steps):
        hidden, _ = model.recurrent(hidden, ids, [depth] * count, positions, cache)
    if boundary is not None:
        return model.coda_prepared(hidden, boundary, cache)
    return model.coda(hidden)


class _CalibrationCache(KVCacheManager):
    """Temporary model-dtype cache that records Q/K/V ranges at the kernel boundary."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.maxima = []

    def _store_layer(self, layer, blocks, offsets, k, v, **kwargs):
        self.maxima.extend((k.abs().amax(), v.abs().amax()))
        super()._store_layer(layer, blocks, offsets, k, v, **kwargs)

    def _attend_prepared(self, layer, batch, q, **kwargs):
        self.maxima.append(q.abs().amax())
        return super()._attend_prepared(layer, batch, q, **kwargs)


@torch.inference_mode()
def calibrate_fp8_scale(model, cache, attention) -> float:
    """Run bounded BF16 prefill and one decode input before allocating the FP8 pool."""
    parameter = next(model.parameters())
    config = model.config
    # P and D use different scheduler limits but must measure the same input.
    length = min(128, max(1, config.max_position_embeddings - 1))
    capacity = min(length + 1, config.max_position_embeddings)
    num_blocks = math.ceil(capacity / cache.block_size)
    if cache.layout == "last_exited":
        num_blocks *= config.total_ut_steps
    probe = _CalibrationCache(
        num_layers=getattr(config, "num_kv_layers", config.num_hidden_layers),
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_loops=config.total_ut_steps,
        num_blocks=num_blocks,
        block_size=cache.block_size,
        layout=cache.layout,
        device=parameter.device,
        dtype=parameter.dtype,
        attention=attention,
        recurrent_layers=getattr(model, "recurrent_kv_layers", None),
    )
    request = "fp8-calibration"
    probe.allocate(request, capacity)
    # Sample vocabulary IDs without touching the engine's or the caller's RNG.
    generator = torch.Generator(device="cpu").manual_seed(0)
    tokens = torch.randint(config.vocab_size, (length,), generator=generator, device="cpu")
    logits = _probe_forward(
        model, probe, tokens.to(parameter.device), [request] * length, list(range(length))
    )
    if capacity > length:
        _probe_forward(model, probe, logits[-1:].argmax(dim=-1), [request], [length])
    amax = torch.stack(probe.maxima).amax().item()
    if not math.isfinite(amax):
        raise ValueError("FP8 startup calibration encountered non-finite Q/K/V values")
    # Leave headroom for inputs outside this brief startup sample. Serve with a
    # single immutable scale so existing KV pages keep their numerical meaning.
    scale = max(amax * 1.1 / FP8_E4M3_MAX, torch.finfo(torch.float32).tiny)
    return 2.0 ** math.ceil(math.log2(scale))


@torch.inference_mode()
def plan_cache(
    model,
    cache,
    scheduler,
    execution,
    backend,
    *,
    speculative=None,
    staged_prefill=None,
    fp8_scale=None,
):
    parameter = next(model.parameters())
    config, device = model.config, parameter.device
    # Models whose physical stack mixes KV-owning and KV-free sublayers (Dory)
    # expose the compact count; everything else keeps the layer count.
    kv_layers = getattr(config, "num_kv_layers", config.num_hidden_layers)
    per_block = kv_bytes_per_block(
        kv_layers,
        cache.block_size,
        config.num_key_value_heads,
        config.head_dim,
        parameter.element_size(),
        cache.kv_cache_dtype,
    )
    if cache.num_blocks is not None:
        return cache.num_blocks, {"source": "blocks", "bytes_per_block": per_block}
    if cache.kv_cache_memory_bytes is not None:
        blocks = budget_blocks(cache.kv_cache_memory_bytes, per_block)
        return blocks, {"source": "bytes", "bytes_per_block": per_block}
    if device.type == "npu":
        raise ValueError("NPU requires explicit num_blocks or kv_cache_memory_bytes")
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
        num_layers=getattr(config, "num_kv_layers", config.num_hidden_layers),
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_loops=config.total_ut_steps,
        num_blocks=sum(math.ceil(length / cache.block_size) for length in lengths)
        * (config.total_ut_steps if cache.layout == "last_exited" else 1),
        block_size=cache.block_size,
        layout=cache.layout,
        device=device,
        dtype=parameter.dtype,
        backend=backend,
        recurrent_layers=getattr(model, "recurrent_kv_layers", None),
        kv_cache_dtype=cache.kv_cache_dtype,
        fp8_scale=fp8_scale,
    )
    ids, positions = [], []
    for index, length in enumerate(lengths):
        rid = f"profile-{index}"
        probe.allocate(rid, length)
        ids.extend([rid] * length)
        positions.extend(range(length))
    tokens = torch.zeros(count, dtype=torch.long, device=device)
    logits = _probe_forward(model, probe, tokens, ids, positions)
    torch.cuda.synchronize(device)
    peak = max(0, torch.cuda.max_memory_allocated(device) - allocated)
    del logits, probe
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
