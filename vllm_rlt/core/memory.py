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
    existing_blocks=0,
    reservation_blocks=None,
):
    """Total blocks the pool may hold after warmup and capture.

    ``free`` memory already excludes weights, execution buffers, captured graphs
    and anything already mapped, including ``existing_blocks`` (the bootstrap
    prefix). The measured capacity is therefore additional, and the cumulative
    ``commit(total)`` of a storage maps only the new part.

    ``transient_bytes`` covers execution above that persistent baseline and
    ``safety_bytes`` the fixed reserve. The result is clamped by the logical
    ``useful_blocks`` and, when the pool cannot grow that far, by
    ``reservation_blocks``; the binding limit and every margin are recorded.

    ``gpu_memory_utilization`` keeps its existing meaning: a fraction of the
    whole device, not of what is free after the baseline. The KV budget is that
    cap minus everything already resident, i.e. weights, execution buffers,
    captured graphs and mapped blocks.
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
        ("existing_blocks", existing_blocks),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if reservation_blocks is not None and (
        type(reservation_blocks) is not int or reservation_blocks < 1
    ):
        raise ValueError("reservation_blocks must be a positive integer")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    free, total, gaps = measurement_budget(report)
    if gaps and not transient_bytes:
        # The preparation reported paths it did not exercise. Sizing them as if
        # measured would publish a pool that cannot serve them.
        raise ValueError(f"coverage gaps require an explicit transient allowance: {gaps[0]}")
    resident = total - free
    # What the utilization cap still allows, bounded by measured free memory.
    remaining = min(free, int(total * gpu_memory_utilization) - resident)
    budget = remaining - transient_bytes - safety_bytes
    additional = max(0, budget // bytes_per_block)
    capacity = existing_blocks + additional
    if capacity < 1:
        raise ValueError(
            "no KV capacity fits the measured budget: "
            f"remaining={remaining} transient={transient_bytes} "
            f"safety={safety_bytes} block={bytes_per_block}"
        )
    limit = "budget"
    if capacity > useful_blocks:
        capacity, limit = useful_blocks, "useful_blocks"
    if reservation_blocks is not None and capacity > reservation_blocks:
        capacity, limit = reservation_blocks, "reservation"
        gaps = (
            *gaps,
            "measured capacity exceeds the reservation; committed capacity truncated",
        )
    return capacity, dict(
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
        existing_blocks=existing_blocks,
        additional_blocks=additional,
        max_useful_blocks=useful_blocks,
        reservation_blocks=reservation_blocks,
        limited_by=limit,
        coverage_gaps=gaps,
    )


def transient_allowance(report, *, estimate_bytes, overlap_factor=1):
    """Allowance for execution above the persistent baseline of a phase.

    A phase records its end state and its own peak, so ``peak - allocated`` is
    what that phase needed on top of what it kept. The largest such delta is
    measured; ``estimate_bytes`` covers paths the preparation did not exercise
    and must stay in force. ``overlap_factor`` covers supported concurrency,
    because a serial peak understates a concurrent envelope. The larger of the
    two is returned, so an unmeasured path is never sized as if measured.
    """
    if type(estimate_bytes) is not int or estimate_bytes < 0:
        raise ValueError("estimate_bytes must be a nonnegative integer")
    if type(overlap_factor) is not int or overlap_factor < 1:
        raise ValueError("overlap_factor must be an integer of at least 1")
    phases = (report or {}).get("phases") or ()
    measured = 0
    sampled = 0
    for phase in phases:
        peak, allocated = phase.get("peak_allocated_bytes"), phase.get("allocated_bytes")
        if type(peak) is not int or type(allocated) is not int:
            continue
        sampled += 1
        measured = max(measured, peak - allocated)
    value = max(measured * overlap_factor, estimate_bytes)
    return value, dict(
        transient_bytes=value,
        measured_transient_bytes=measured,
        estimate_bytes=estimate_bytes,
        overlap_factor=overlap_factor,
        sampled_phases=sampled,
    )


def commit_measured_capacity(
    storage,
    *,
    report,
    bytes_per_block,
    useful_blocks,
    gpu_memory_utilization,
    transient_bytes,
    safety_bytes,
    allow_reservation_truncation=False,
):
    """Plan the measured capacity, map it, and return the publishable block count.

    Enforces the preconditions of the agreed ordering: the report describes the
    pool the storage already backs, the plan never exceeds the reservation, and
    the storage must report at least the planned blocks as backed. The caller
    publishes exactly the returned count and nothing else.

    Two caller policies are checked here rather than assumed. The trial pool must
    already back at least one block, because a mapped pool cannot be built with no
    mapping at all. And its reservation must be sized from ``useful_blocks``, not
    from the pre-warmup estimate: otherwise the measured capacity is clamped back
    to the estimate and the pool can never grow. Pass
    ``allow_reservation_truncation=True`` only when that clamp is intended.
    """
    existing = (report or {}).get("num_blocks")
    if type(existing) is not int or existing < 0:
        raise ValueError("the startup report must record the current pool as num_blocks")
    if existing < 1:
        raise ValueError(
            "the trial pool must back at least one block; size it from the warmup "
            "coverage target before preparing the runtime"
        )
    if existing > storage.max_blocks:
        raise ValueError(
            f"the report describes {existing} blocks but the storage reserves {storage.max_blocks}"
        )
    if storage.committed_blocks != existing:
        # The pool changed after the sample, so the measured free memory no
        # longer describes this storage.
        raise ValueError(
            f"the report describes {existing} blocks but the storage backs "
            f"{storage.committed_blocks}"
        )
    capacity, record = measured_kv_capacity(
        report=report,
        bytes_per_block=bytes_per_block,
        useful_blocks=useful_blocks,
        gpu_memory_utilization=gpu_memory_utilization,
        transient_bytes=transient_bytes,
        safety_bytes=safety_bytes,
        existing_blocks=existing,
        reservation_blocks=storage.max_blocks,
    )
    if record["limited_by"] == "reservation" and not allow_reservation_truncation:
        raise RuntimeError(
            f"the measured capacity ({capacity} blocks) exceeds the "
            f"{storage.max_blocks}-block reservation, so it would be truncated to the "
            "pre-warmup estimate; reserve the useful cap instead of the estimate"
        )
    backed = storage.commit(capacity)
    if type(backed) is not int or backed < capacity:
        raise RuntimeError(f"storage reported {backed} backed blocks after committing {capacity}")
    if backed > storage.max_blocks:
        raise RuntimeError(
            f"storage reported {backed} backed blocks beyond its "
            f"{storage.max_blocks}-block reservation"
        )
    record.update(committed_blocks=backed, storage="mapped")
    return backed, record


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


def torch_attention_scratch_bytes(config, *, pool_element_size):
    """Transient the reference torch attention needs for one long-context row.

    ``torch_paged_attention`` resolves one row at a time: it gathers that row's
    whole history for K and for V in the pool dtype, copies each of them into
    float32, and keeps two score vectors of ``[heads, length]`` floats. Nothing
    else scales with the context, and the tensors are released before the next
    row, so the peak is linear in the length and does not depend on the layer
    count.

    Measured with ``torch_paged_attention`` on an A100 (16 heads, head_dim 128,
    bf16 pool, one row): 100.8 MB at 4096 tokens, 403.1 MB at 16384 and
    1612.2 MB at 65536 — 12 bytes per element of the gathered history, which is
    exactly ``2 * (pool_element_size + 4)``, plus the two score vectors. The
    estimate this replaces used ``8 * head_dim + 8`` floats per token and head,
    which is 2.7 times the measurement, and that excess came straight out of the
    KV budget: 2.5 GiB, or about 860 blocks, on a 65536-token checkpoint.

    The repeated head axis is not measured, because no grouped-query checkpoint
    was available; it adds a float32 copy of the expanded history and is counted
    whenever key-value heads are fewer than attention heads.
    """
    length = config.max_position_embeddings
    gathered = 2 * length * config.num_key_value_heads * config.head_dim * (pool_element_size + 4)
    scores = 2 * length * config.num_attention_heads * 4
    repeat = 0
    if config.num_attention_heads > config.num_key_value_heads:
        repeat = 2 * length * config.num_attention_heads * config.head_dim * 4
    return gathered + scores + repeat


def bootstrap_blocks(config, cache, scheduler):
    """Blocks the trial pool must back before startup warmup can measure anything.

    Warmup runs one forward per shape, so the pool has to hold the largest
    prefill step (``max_num_batched_tokens`` tokens, capped by
    ``max_num_seqs * max_position_embeddings``) plus one growing tail block per
    concurrent sequence for the decode cases that follow. The looped layout
    keeps ``total_ut_steps`` copies of each.

    A pool smaller than this makes warmup skip shapes and report a coverage gap,
    and a measurement with gaps cannot retire the estimate it was meant to
    replace: the measured plan keeps the formula reserve and the extra startup
    work buys nothing. Commit at least this many blocks before
    ``prepare_runtime``.
    """
    tokens = min(
        scheduler.max_num_batched_tokens, scheduler.max_num_seqs * config.max_position_embeddings
    )
    depth = config.total_ut_steps if cache.layout == "last_exited" else 1
    return math.ceil(tokens / cache.block_size) * depth + scheduler.max_num_seqs * depth


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
    probe_blocks = sum(math.ceil(length / cache.block_size) for length in lengths) * (
        config.total_ut_steps if cache.layout == "last_exited" else 1
    )
    probe = KVCacheManager(
        num_layers=getattr(config, "num_kv_layers", config.num_hidden_layers),
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_loops=config.total_ut_steps,
        num_blocks=probe_blocks,
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
    raw_peak = max(0, torch.cuda.max_memory_allocated(device) - allocated)
    # The probe's own KV pool is not runtime workspace: charging it here would
    # take KV bytes out of the KV budget twice (RFC #192, section 5).
    probe_kv_bytes = probe_blocks * per_block
    peak = max(0, raw_peak - probe_kv_bytes)
    del logits, probe
    # Fused attention keeps bounded scratch. The reference torch path gathers a
    # whole history into float32 per row, so its transient scales with the
    # context the deployment supports, not with the prefill batch.
    context_scratch = 0
    if backend == "torch":
        pool_element = 1 if cache.kv_cache_dtype == "fp8_e4m3" else parameter.element_size()
        context_scratch = torch_attention_scratch_bytes(config, pool_element_size=pool_element)
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
        raw_profile_peak_bytes=raw_peak,
        probe_kv_bytes=probe_kv_bytes,
        concurrent_peak_bytes=concurrent_peak,
        max_useful_blocks=useful_blocks,
        cuda_graph_reserve_bytes=graph_reserve,
        static_buffer_bytes=buffers,
        context_scratch_bytes=context_scratch,
        metadata_bytes=metadata,
        kv_budget_bytes=int(budget),
    )
