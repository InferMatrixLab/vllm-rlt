"""Sparse decode attention arithmetic and its per-row plan (RFC #156, section 3.2).

The sparse decode algorithm keeps the attention output of a reference loop and
replaces only the contribution of the retained keys in a later loop. The
reference loop is loop 2 in the RFC's one-based numbering, which is depth index
1 at runtime.

This module holds the arithmetic and the plan that describes one submission:

``reference_statistics``
    the dense reference attention of one row, plus the retained state
    ``(S, m, O_ref, C_ref)``,
``sparse_correction``
    the corrected output of a later loop, ``O = O_ref - C_ref + m * C_new``,
``SparseDecodePlan``
    the row modes and reference slots of one submission.

The arithmetic is deliberately plain PyTorch. The retained state is small, so a
straightforward implementation is easier to check against the worked example in
the RFC than an optimized kernel, and it keeps the contract that the attention
kernels must satisfy visible in one place.

The plan is the named view between the runner and the attention layer. The
runner owns the reference slots and their lifetime; the attention layer only
borrows the plan for the rows of one submission. Row modes and slots are
host-known: the runner decides both before it submits, so a submission never
depends on a device readback.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_rlt.kernels.sparse_capture import TRITON_AVAILABLE, fused_capture_scores

# Row modes for one recurrent submission.
DENSE = 0
CAPTURE = 1
CONSUME = 2

# The reference loop is loop 2 in the RFC's one-based numbering, which is
# zero-based depth index 1 at runtime.
REFERENCE_DEPTH = 1


def reference_mode(depth: int) -> int:
    """Row mode of a recurrent step at the given zero-based loop depth."""
    if depth == REFERENCE_DEPTH:
        return CAPTURE
    if depth > REFERENCE_DEPTH:
        return CONSUME
    return DENSE


@dataclass(frozen=True)
class SparseDecodePlan:
    """The reference plan of one recurrent submission.

    ``modes`` and ``slots`` are host-known: the runner decides both before it
    submits, so a submission never depends on a device readback. ``slots`` holds
    one reference slot per row, or ``-1`` for a row that owns no reference.
    ``incarnations`` identifies the token that occupies each slot; a plan built
    for a superseded incarnation is stale and must be rejected.

    A dense row runs the ordinary attention path. A capture row is the
    reference loop: it computes the dense attention and publishes the retained
    state. A consume row is a later loop: it corrects the reference output with
    the retained keys.

    ``store`` is the runner-owned reference pool, borrowed by the attention
    layer through duck typing so that model code needs no import of the worker
    package.
    """

    modes: tuple[int, ...]
    slots: tuple[int, ...]
    incarnations: tuple[int, ...]
    key_retention: float
    key_bucket: int | None
    slot_ids: torch.Tensor | None = None
    store: object | None = None

    def groups(self) -> dict[int, list[int]]:
        """Rows per mode, in submission order."""
        grouped: dict[int, list[int]] = {}
        for row, mode in enumerate(self.modes):
            grouped.setdefault(mode, []).append(row)
        return {mode: rows for mode, rows in grouped.items() if rows}

    @property
    def reference_depth(self) -> int:
        """The zero-based depth that captures the reference state."""
        return REFERENCE_DEPTH

    def dense(self) -> bool:
        """Whether the submission has no sparse work at all."""
        return all(mode == DENSE for mode in self.modes)


def retained_count(length: int, key_retention: float) -> int:
    """How many keys a query with ``length`` legal keys retains.

    Follows the FlashLoop reference (``aeaadee``, ``select_decode_keys``):
    ``k = min(length, max(1, ceil(length * key_retention)))``. The count is a
    pure function of the retention fraction, so ``key_retention=1.0`` stays an
    exact full-retention control; index-buffer capacity lives in the store and
    never changes the algorithm.
    """
    if key_retention <= 0 or key_retention > 1:
        raise ValueError("key_retention must be in (0, 1]")
    if length < 1:
        return 0
    return min(length, max(1, math.ceil(length * key_retention)))


def expand_kv(tensor: torch.Tensor, groups: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention."""
    if groups == 1:
        return tensor
    return tensor.repeat_interleave(groups, dim=1)


def select_keys(probabilities: torch.Tensor, count: int) -> torch.Tensor:
    """Pick the retained key positions of every head.

    ``probabilities`` is ``[heads, length]``. Equal probabilities are broken by
    the smaller absolute position, so the selection does not depend on the order
    of the rows in a batch.
    """
    if count < 1:
        raise ValueError("a selection needs at least one key")
    if count > probabilities.shape[1]:
        raise ValueError("cannot retain more keys than the query has")
    # A stable descending sort keeps equal probabilities in position order.
    order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
    return order[:, :count].sort(dim=-1).values


def reference_statistics(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    key_retention: float,
    key_bucket: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense reference attention and the retained state of one row.

    ``q`` is ``[heads, dim]``, ``keys`` and ``values`` are
    ``[length, kv_heads, dim]``. Returns ``(output, contribution, mass,
    positions)`` where output and contribution are ``[heads, dim]``, mass is
    ``[heads]`` and positions are the retained absolute key positions.
    """
    heads, dim = q.shape
    length = keys.shape[0]
    groups = heads // keys.shape[1]
    scale = 1.0 / dim**0.5
    expanded_keys = expand_kv(keys, groups).float()
    expanded_values = expand_kv(values, groups).float()
    scores = torch.einsum("hd,lhd->hl", q.float(), expanded_keys) * scale
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hl,lhd->hd", probabilities, expanded_values)
    count = retained_count(length, key_retention)
    # Buffer capacity never truncates the algorithm: a count above it fails
    # loudly instead of silently degrading retention.
    if key_bucket is not None and count > key_bucket:
        raise ValueError(f"reference capacity {key_bucket} cannot hold {count} retained keys")
    positions = select_keys(probabilities, count)
    gathered = probabilities.gather(1, positions)
    # Index the per-head value rows: positions is [heads, count], so the head
    # index broadcasts against it and selects each head's own key rows.
    head_index = torch.arange(heads, device=q.device)[:, None]
    selected_values = expanded_values[positions, head_index]
    contribution = torch.einsum("hk,hkd->hd", gathered, selected_values)
    mass = gathered.sum(dim=-1)
    return output, contribution, mass, positions


def sparse_correction(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mass: torch.Tensor,
    output: torch.Tensor,
    contribution: torch.Tensor,
) -> torch.Tensor:
    """Correct the reference output with the retained keys of a later loop.

    ``keys`` and ``values`` are ``[heads, retained, dim]`` and already carry the
    keys of the later loop. ``mass``, ``output`` and ``contribution`` are the
    retained reference state.
    """
    dim = q.shape[-1]
    scale = 1.0 / dim**0.5
    scores = torch.einsum("hd,hkd->hk", q.float(), keys.float()) * scale
    probabilities = torch.softmax(scores, dim=-1)
    new_contribution = torch.einsum("hk,hkd->hd", probabilities, values.float())
    return output - contribution + mass.float().unsqueeze(-1) * new_contribution


def gather_retained(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected: torch.Tensor,
    block_size: int,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather each query head's retained keys and values from the paged cache.

    ``selected`` holds absolute key positions per query head, so every head
    reads its own retained set. Returns ``[heads, retained, dim]`` tensors.
    """
    index = selected.long()
    blocks = block_table[index // block_size]
    offsets = index % block_size
    kv_index = (torch.arange(index.shape[0], device=index.device) // groups)[:, None]
    kv_index = kv_index.expand_as(index)
    return key_cache[blocks, offsets, kv_index], value_cache[blocks, offsets, kv_index]


def reference_paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    plan: SparseDecodePlan,
    *,
    layer: int,
    dense_attention: Callable | None = None,
) -> torch.Tensor:
    """Reference sparse decode attention over a paged KV cache (RFC #156 3.2).

    This is the numerical oracle for the sparse decode column, not an optimized
    kernel: it walks rows on the host, reads the same paged tensors the dense
    path reads, and publishes or consumes the retained state through the plan's
    store. Dense rows reproduce the ordinary attention path exactly.

    ``key_cache``/``value_cache`` are one layer of
    ``[physical block, token, KV head, dimension]``; ``block_tables`` and
    ``context_lengths`` describe the submission rows exactly as the dense
    backend receives them. When supplied, ``dense_attention`` preserves that
    backend's outputs for dense/capture rows and the full-retention control.
    Omitting it forces the independent Torch arithmetic oracle, including the
    full-retention correction; that path need not be bitwise equal to Triton.
    """
    store = plan.store
    if store is None:
        raise ValueError("a sparse decode plan needs its reference store")
    block_size = key_cache.shape[1]
    heads = q.shape[1]
    groups = heads // key_cache.shape[2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    output = torch.empty_like(q)
    dense_output = None
    if dense_attention is not None and (
        plan.key_retention == 1.0 or any(mode != CONSUME for mode in plan.modes)
    ):
        # Keep the original packed geometry: slicing rows can also change a
        # backend's numerics. Only the rows needing dense output use this result.
        dense_output = dense_attention(q, key_cache, value_cache, block_tables, context_lengths)
    captured = consumed = False
    for row, length in enumerate(context_lengths.tolist()):
        if length == 0:
            output[row].zero_()
            continue
        positions = torch.arange(length, device=q.device)
        blocks = block_tables[row, positions // block_size]
        offsets = positions % block_size
        mode = plan.modes[row]
        if mode == CONSUME:
            selected, mass, reference, contribution = store.read_reference(plan, layer, row)
            if dense_output is not None and plan.key_retention == 1.0:
                output[row] = dense_output[row]
                consumed = True
                continue
            keys, values = gather_retained(
                key_cache, value_cache, block_tables[row], selected, block_size, groups
            )
            output[row] = sparse_correction(q[row], keys, values, mass, reference, contribution).to(
                q.dtype
            )
            consumed = True
            continue
        keys = key_cache[blocks, offsets]
        values = value_cache[blocks, offsets]
        if mode == CAPTURE:
            reference, contribution, mass, selected = reference_statistics(
                q[row],
                keys,
                values,
                key_retention=plan.key_retention,
                key_bucket=store.key_bucket,
            )
            if dense_output is not None:
                # O_ref is the output actually fed into the recurrence. Keep
                # selector probabilities/contributions in the Torch FP32 oracle.
                reference = dense_output[row].float()
            output[row] = reference.to(q.dtype)
            store.write_reference(
                plan,
                layer,
                row,
                selected=selected,
                counts=torch.full((heads,), selected.shape[-1], dtype=torch.int32, device=q.device),
                mass=mass,
                output=reference,
                contribution=contribution,
            )
            captured = True
            continue
        # Dense row: the ordinary attention path.
        if dense_output is not None:
            output[row] = dense_output[row]
            continue
        expanded_keys = expand_kv(keys, groups).float()
        expanded_values = expand_kv(values, groups).float()
        scores = torch.einsum("hd,lhd->hl", q[row].float(), expanded_keys) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum("hl,lhd->hd", probabilities, expanded_values).to(q.dtype)
    if captured:
        store.capture(plan, layer)
    if consumed:
        store.consume(plan, layer)
    return output


def sparse_paged_attention(
    q,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    plan,
    *,
    layer,
    dense_attention: Callable | None = None,
):
    """Use bounded row batches on CUDA, retaining the scalar numerical oracle.

    On CUDA with a dense backend available, the capture rows take the fused
    paged score pass; the scalar oracle stays the definition of the arithmetic,
    and ``batched_paged_attention`` without ``fused_scores`` still reproduces it
    bit for bit.
    """
    if not q.is_cuda:
        return reference_paged_attention(
            q,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            plan,
            layer=layer,
            dense_attention=dense_attention,
        )
    return batched_paged_attention(
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        plan,
        layer=layer,
        dense_attention=dense_attention,
        fused_scores=TRITON_AVAILABLE and dense_attention is not None,
    )


def _rowwise_product(equation, left, right):
    """Keep each row's cuBLAS problem shape equal to the scalar oracle.

    Folding rows into the head dimension can choose a different FP32 reduction
    kernel. Its small error can cross a BF16 rounding boundary and change a
    greedy token, so batch the gathers/selection but retain row-wise products.
    """
    # Prepare the exact views einsum uses once for the group, instead of
    # repeating its permutation/reshape bookkeeping for every row.
    if equation == "hd,lhd->hl":
        right = right.permute(0, 2, 3, 1)
    elif equation == "hl,lhd->hd":
        right = right.transpose(1, 2)
    elif equation == "hd,hkd->hk":
        right = right.transpose(-1, -2)
    elif equation != "hk,hkd->hd":
        raise ValueError(f"unsupported reference product {equation}")
    return torch.stack([torch.bmm(a, b).squeeze(1) for a, b in zip(left.unsqueeze(-2), right)])


def batched_paged_attention(
    q,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    plan,
    *,
    layer,
    dense_attention: Callable | None = None,
    fused_scores: bool = False,
):
    """Batch equal-length rows without padding or changing retained-key selection.

    Each head still uses the same FP32 matrix products, stable sort and softmax
    as the scalar oracle. Grouping exact lengths avoids padded reductions and
    bounds gather scratch independently of the reference pool's key capacity.
    There is one count readback per layer, rather than one per consume row.

    ``dense_attention`` keeps the numerical contract of the scalar path: dense
    and capture rows publish the selected backend's output, the actually
    rounded recurrence input becomes ``O_ref``, and a full-retention consume row
    returns that same backend output as the control. Selector probabilities and
    the retained contribution stay in the Torch FP32 oracle.
    """
    store = plan.store
    if store is None:
        raise ValueError("a sparse decode plan needs its reference store")
    store._require_live(plan)
    lengths = context_lengths.tolist()
    dense_output = None
    if dense_attention is not None and (
        plan.key_retention == 1.0 or any(mode != CONSUME for mode in plan.modes)
    ):
        # Keep the original packed geometry: slicing rows can also change a
        # backend's numerics.
        dense_output = dense_attention(q, key_cache, value_cache, block_tables, context_lengths)
    consume_rows = [r for r, mode in enumerate(plan.modes) if mode == CONSUME and lengths[r]]
    counts = {}
    if consume_rows:
        # The retained count is host-known from the capture. Reading it back
        # from the device would sync every layer, which RFC #156 section 6.2
        # forbids.
        for row in consume_rows:
            count = store.host_count(plan.slots[row], layer)
            if not count:
                raise RuntimeError("a consume row has no captured reference")
            counts[row] = count
    grouped = {}
    for row, length in enumerate(lengths):
        if length:
            mode = plan.modes[row]
            extent = counts[row] if mode == CONSUME else length
            grouped.setdefault((mode, extent), []).append(row)
    output = torch.zeros_like(q)
    heads, dim = q.shape[1:]
    groups = heads // key_cache.shape[2]
    block_size = key_cache.shape[1]
    scale = 1.0 / math.sqrt(dim)
    head_index = torch.arange(heads, device=q.device)[None, :, None]
    kv_index = head_index // groups
    captured = consumed = False
    for (mode, extent), all_rows in grouped.items():
        # The fused path materialises only the score matrix, so its budget is
        # the scores themselves; the expanded path budgets an FP32 K row. Both
        # must allow the whole group at once, or a 32k row degenerates into one
        # launch per row.
        if fused_scores:
            chunk = max(1, min(32, 512 * 1024 * 1024 // (heads * extent * 4)))
        else:
            chunk = max(1, min(32, 64 * 1024 * 1024 // (extent * heads * dim)))
        for start in range(0, len(all_rows), chunk):
            rows = all_rows[start : start + chunk]
            # Building a device tensor from a Python list costs ~2.4 ms (a
            # blocking host-to-device copy per call). Grouped rows are almost
            # always contiguous, so derive them on the device instead.
            if rows[0] + len(rows) == rows[-1] + 1:
                row_ids = torch.arange(rows[0], rows[0] + len(rows), device=q.device)
            else:
                row_ids = torch.tensor(rows, dtype=torch.int64, device=q.device)
            slots = plan.slot_ids[row_ids]
            queries = q[row_ids].float()
            tables = block_tables[row_ids]
            batch_index = torch.arange(len(rows), device=q.device)[:, None, None]
            if mode == CONSUME:
                if dense_output is not None and plan.key_retention == 1.0:
                    # Full retention is the control: same backend as dense.
                    output[row_ids] = dense_output[row_ids]
                    consumed = True
                    continue
                columns = torch.arange(extent, device=q.device)[None, None, :]
                selected = store.selected[slots[:, None, None], layer, head_index, columns].long()
                blocks = tables[batch_index, selected // block_size]
                offsets = selected % block_size
                keys = key_cache[blocks, offsets, kv_index].float()
                values = value_cache[blocks, offsets, kv_index].float()
                scores = _rowwise_product("hd,hkd->hk", queries, keys) * scale
                probabilities = torch.softmax(scores, dim=-1)
                new = _rowwise_product("hk,hkd->hd", probabilities, values)
                corrected = (
                    store.output[slots, layer]
                    - store.contribution[slots, layer]
                    + store.mass[slots, layer].unsqueeze(-1) * new
                )
                output[row_ids] = corrected.to(q.dtype)
                consumed = True
                continue
            positions = torch.arange(extent, device=q.device)
            blocks = tables[:, positions // block_size]
            offsets = positions % block_size
            if mode == DENSE and dense_output is not None:
                output[row_ids] = dense_output[row_ids]
                continue
            if fused_scores and dense_output is not None:
                # One pass over the paged cache: no expanded FP32 K row, and
                # no per-row cuBLAS. The dot product is still FP32; only the
                # reduction order differs from the oracle.
                scores = fused_capture_scores(
                    q[row_ids],
                    key_cache,
                    tables,
                    context_lengths[row_ids],
                    layer=layer,
                    block_size=block_size,
                )
                scores = scores * scale
            else:
                keys = key_cache[blocks, offsets].repeat_interleave(groups, dim=2).float()
                scores = _rowwise_product("hd,lhd->hl", queries, keys) * scale
            probabilities = torch.softmax(scores, dim=-1)
            if dense_output is not None:
                # O_ref is the output actually fed into the recurrence; the
                # selector state below stays in the Torch FP32 oracle.
                reference = dense_output[row_ids].float()
            else:
                values = value_cache[blocks, offsets].repeat_interleave(groups, dim=2).float()
                reference = _rowwise_product("hl,lhd->hd", probabilities, values)
            output[row_ids] = reference.to(q.dtype)
            if mode == CAPTURE:
                count = retained_count(extent, plan.key_retention)
                capacity = store.selected.shape[-1]
                if count > capacity:
                    raise ValueError(
                        f"reference capacity {capacity} cannot hold {count} retained keys"
                    )
                order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
                selected = order[..., :count].sort(dim=-1).values
                gathered = probabilities.gather(-1, selected)
                # Gather the retained values straight from the paged cache: the
                # expanded fp32 copy would be the same bits (bf16 -> fp32 is
                # exact) but reading it scatters through a row-length tensor.
                selected_values = value_cache[
                    tables[batch_index, selected // block_size],
                    selected % block_size,
                    kv_index,
                ].float()
                contribution = _rowwise_product("hk,hkd->hd", gathered, selected_values)
                columns = torch.arange(count, device=q.device)[None, None, :]
                # Explicit indices write only the live prefix, never gather the
                # entire capacity-sized selected tensor to obtain a prefix view.
                store.selected[slots[:, None, None], layer, head_index, columns] = selected.to(
                    torch.int32
                )
                store.counts[slots, layer] = count
                for row in rows:
                    store.note_count(plan.slots[row], layer, count)
                store.mass[slots, layer] = gathered.sum(dim=-1)
                store.output[slots, layer] = reference
                store.contribution[slots, layer] = contribution
                captured = True
    if captured:
        store.capture(plan, layer)
    if consumed:
        store.consume(plan, layer)
    return output
