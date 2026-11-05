"""Fused sparse decode attention for uniform capture or consume submissions.

Opt-in alternative to the batched path in ``vllm_rlt.attention.sparse``. That
path keeps the scalar oracle's cuBLAS reduction shapes, so every layer launches
per-row products and reads row lengths and retained counts back to the host.
Here each layer launches a fixed number of kernels and keeps lengths and
counts on the device.

Numerics differ from the oracle on consume rows: scores and FP32 reductions
run in another order, so a key on the selection boundary can differ from the
oracle's. Retained positions are written in ascending order and equal scores
keep the smaller positions, as in the oracle. Capture rows still return the
dense backend output, which is also published as ``O_ref``, so the recurrence
input of the reference loop is unchanged.

A consume row whose reference was never captured (count 0) yields NaN instead
of raising: raising would need the host readback this path exists to avoid. The
batched and scalar paths keep the explicit error.
"""

import operator

import torch
import triton
import triton.language as tl

from vllm_rlt.attention.sparse import CAPTURE, CONSUME, retained_count
from vllm_rlt.kernels.paged_attention import TritonPagedAttention, triton_paged_attention
from vllm_rlt.kernels.sparse_workspace import CAPTURE_SCRATCH, CONSUME_SCRATCH, RETENTION_SCALARS
from vllm_rlt.kernels.triton_attention import NUM_SPLITS


@triton.jit
def _order_key(scores):
    """Key in [0, 2**32) with the order of FP32 scores (bits of negatives flipped)."""
    bits = scores.to(tl.int32, bitcast=True)
    return (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.int64) + 2147483648


@triton.jit
def _select_and_publish(
    scores_row,
    lse,
    length,
    count,
    slot,
    layer,
    head,
    reference,
    table,
    V,
    SELECTED,
    COUNTS,
    MASS,
    OUTPUT,
    CONTRIBUTION,
    retained_capacity,
    v_block_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Keep the ``count`` largest of ``length`` stored scores and publish ``(S, m, O_ref, C_ref)``.

    Finds the count-th largest score by a radix select over its 32-bit order
    key, four bits per pass, and writes the retained positions in ascending
    order; equal scores keep the smaller positions, like the oracle's stable
    sort. ``reference`` is ``O_ref``.
    """
    kv_head = head // HEAD_GROUPS
    dims = tl.arange(0, BLOCK_D)
    # After eight passes ``prefix`` is the order key of the count-th largest
    # score and ``ties_needed`` how many keys equal to it are still retained.
    # Loop-carried values must keep one type on every iteration.
    prefix = length.to(tl.int64) * 0
    ties_needed = count
    digits = tl.arange(0, 16)
    for level in tl.static_range(8):
        histogram = tl.zeros((16,), dtype=tl.int32)
        for start in range(0, length, BLOCK_T):
            positions = start + tl.arange(0, BLOCK_T)
            live = positions < length
            key = _order_key(tl.load(scores_row + positions, mask=live, other=0))
            candidate = live & ((key >> (32 - 4 * level)) == prefix)
            digit = ((key >> (28 - 4 * level)) & 15).to(tl.int32)
            histogram += tl.sum(
                ((digit[:, None] == digits[None, :]) & candidate[:, None]).to(tl.int32), axis=0
            )
        # Highest digit whose keys, with every higher digit, still cover the rest.
        covered = tl.cumsum(histogram, 0, reverse=True)
        chosen = tl.sum((covered >= ties_needed).to(tl.int32), axis=0) - 1
        ties_needed -= tl.sum(tl.where(digits > chosen, histogram, 0), axis=0)
        prefix = prefix * 16 + chosen
    low = prefix
    selected = SELECTED + ((slot * NUM_LAYERS + layer) * NUM_HEADS + head) * retained_capacity
    written = 0
    ties_seen = 0
    mass = 0.0
    contribution = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start in range(0, length, BLOCK_T):
        positions = start + tl.arange(0, BLOCK_T)
        live = positions < length
        scores = tl.load(scores_row + positions, mask=live, other=0)
        key = _order_key(scores)
        tie = (live & (key == low)).to(tl.int32)
        take = (live & (key > low)) | (
            (tie == 1) & (ties_seen + tl.cumsum(tie, 0) - tie < ties_needed)
        )
        taken = take.to(tl.int32)
        tl.store(selected + written + tl.cumsum(taken, 0) - taken, positions, mask=take)
        probabilities = tl.where(take, tl.exp(scores - lse), 0.0)
        blocks = tl.load(table + positions // PAGE_SIZE, mask=take, other=0).to(tl.int64)
        values = tl.load(
            V
            + blocks[:, None] * v_block_stride
            + (positions % PAGE_SIZE)[:, None] * v_token_stride
            + kv_head * v_head_stride
            + dims[None, :],
            mask=take[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)
        contribution += tl.sum(probabilities[:, None] * values, axis=0)
        mass += tl.sum(probabilities, axis=0)
        written += tl.sum(taken, axis=0)
        ties_seen += tl.sum(tie, axis=0)
    state = (slot * NUM_LAYERS + layer) * NUM_HEADS + head
    tl.store(COUNTS + state, count)
    tl.store(MASS + state, mass)
    tl.store(OUTPUT + state * HEAD_DIM + dims, reference, mask=dims < HEAD_DIM)
    tl.store(CONTRIBUTION + state * HEAD_DIM + dims, contribution, mask=dims < HEAD_DIM)


# A copy of triton_attention._paged_attention_kernel that also stores the raw
# scores. Its arithmetic must stay identical to the dense kernel: the capture
# output is required to equal the dense backend bit for bit (tested).
@triton.jit
def _capture_split_kernel(
    Q,
    K,
    V,
    TABLES,
    LENGTHS,
    PART_M,
    PART_L,
    PART_O,
    SCORES,
    SEGHIST,
    RETENTION,
    STATE,
    TICKETS,
    top,
    score_capacity,
    q_batch_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    q_dim_stride: tl.constexpr,
    k_block_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    k_dim_stride: tl.constexpr,
    v_block_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    v_dim_stride: tl.constexpr,
    table_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SPLITS: tl.constexpr,
    HISTOGRAM: tl.constexpr,
    PARTS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = head // HEAD_GROUPS
    length = tl.load(LENGTHS + row)
    # Equal tile-aligned parts of this row's context; trailing parts may be empty.
    split_tokens = ((length + SPLITS - 1) // SPLITS + BLOCK_T - 1) // BLOCK_T * BLOCK_T
    begin = split * split_tokens
    end = tl.minimum(length, begin + split_tokens)
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        Q + row * q_batch_stride + head * q_head_stride + dims * q_dim_stride,
        mask=dims < HEAD_DIM,
        other=0,
    ).to(tl.float32)
    maximum = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    local = tl.zeros((512,), dtype=tl.int32)
    for start in range(begin, end, BLOCK_T):
        positions = start + tl.arange(0, BLOCK_T)
        valid_tokens = positions < end
        blocks = tl.load(
            TABLES + row * table_stride + positions // PAGE_SIZE,
            mask=valid_tokens,
            other=0,
        ).to(tl.int64)
        # Physical IDs fit in int32, but ID * block_stride can exceed 2**31
        # elements in a large KV pool. Promote BEFORE multiplying, for K and V.
        offsets = positions % PAGE_SIZE
        keys = tl.load(
            K
            + blocks[:, None] * k_block_stride
            + offsets[:, None] * k_token_stride
            + kv_head * k_head_stride
            + dims[None, :] * k_dim_stride,
            mask=valid_tokens[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid_tokens, scores, -float("inf"))
        # The only addition to the dense kernel: keep the raw scores for selection.
        tl.store(
            SCORES + (row * NUM_HEADS + head).to(tl.int64) * score_capacity + positions,
            scores,
            mask=valid_tokens,
        )
        if HISTOGRAM:
            # Selection pass 0 of this part: the top eight bits of the order key.
            digit = tl.where(valid_tokens, (_order_key(scores) >> 24).to(tl.int32), 256)
            # Reorder integer votes into a blocked vector; a histogram is
            # order-independent. This avoids Triton 3.1's sliced-layout bug
            # without changing the dense floating-point reduction.
            digit = tl.reshape(tl.trans(tl.reshape(digit, (4, BLOCK_T // 4))), (BLOCK_T,))
            local += tl.histogram(digit, 512)
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        correction = tl.exp(maximum - next_maximum)
        weights = tl.exp(scores - next_maximum)
        values = tl.load(
            V
            + blocks[:, None] * v_block_stride
            + offsets[:, None] * v_token_stride
            + kv_head * v_head_stride
            + dims[None, :] * v_dim_stride,
            mask=valid_tokens[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
        normalizer = normalizer * correction + tl.sum(weights, axis=0)
        maximum = next_maximum
    # An empty part stores (-inf, 0, 0), which the merge weights by zero.
    part = (row * NUM_HEADS + head) * SPLITS + split
    tl.store(PART_M + part, maximum)
    tl.store(PART_L + part, normalizer)
    tl.store(PART_O + part * BLOCK_D + dims, accumulator)
    if HISTOGRAM:
        rh = row * NUM_HEADS + head
        _, count = _retained(LENGTHS, row, RETENTION, top)
        if count > 0:
            bins = tl.arange(0, 512)
            tl.store(SEGHIST + (rh * PARTS + split) * 4 * 256 + bins, local, mask=bins < 256)
            _close_pass(SEGHIST, STATE, TICKETS, rh, count, 0, SPLITS, PARTS)


# Parallel selection after the split kernel. Every (row, head) splits its
# context into PARTS tile-aligned parts. Four radix passes over the 32-bit order
# key, eight bits each, store one 256-bin histogram per part. The last part to
# finish a pass (an integer ticket per row and head) sums the histograms in a
# fixed order and publishes the state for the next pass, so the result stays
# deterministic and no histogram is zeroed. After the last pass it also derives
# every part's output offset and tie quota, which keeps the ascending order and
# the smaller-position tie rule. The contribution kernel then gathers V over the
# compact retained list only.
SELECT_PARTS = 16
# Tiles of the radix passes and the write (scores only) and of the V gather,
# chosen by a sweep on an RTX 5090 at 4K-32K (16 vs 32 parts, 128-1024 scan,
# 32-128 gather, 4 or 8 warps).
_SCAN_BLOCK = 1024
_SCAN_WARPS = 4
_GATHER_BLOCK = 128
# The split kernel's tile (the dense kernel's), so its splits are whole parts.
_SEGMENT_ALIGN = 32
# The split kernel builds the first pass's histograms while it scores the keys.
_PASS0_IN_SPLIT = True

# Triton's Python dispatch costs several microseconds per launch, and a fused
# capture makes six launches, so at short contexts in eager mode the dispatch,
# not the GPU, sets its cost. A launch whose specialization was seen before
# reuses that compiled kernel directly. The key follows Triton's own
# specialization (dtype and 16-byte alignment of pointers; integers equal to 1,
# divisible by 16, and their width; constexpr values); tests set _LAUNCH_CHECK
# to assert that it always picks the kernel Triton would. Other Triton versions
# keep its dispatch.
try:
    from triton import knobs as _knobs
    from triton.runtime.driver import driver as _driver

    _FAST_LAUNCH = triton.__version__.startswith("3.8.") and hasattr(
        _knobs.runtime, "launch_enter_hook"
    )
except ImportError:  # pragma: no cover - other Triton versions
    _FAST_LAUNCH = False
_LAUNCH_CHECK = False


def _key_function(args, constexpr):
    """Compile the specialization key of one argument layout, as Triton's binder does.

    The layout is the arguments' types; each layout gets its own function.
    """
    terms = []
    for i, value in enumerate(args):
        if i in constexpr:
            terms.append(f"a[{i}]")
        elif isinstance(value, torch.Tensor):
            terms.append(f"a[{i}].dtype, a[{i}].data_ptr() & 15 == 0")
        elif type(value) is int:
            terms.append(f"a[{i}] == 1, a[{i}] & 15 == 0, -2147483648 <= a[{i}] < 2147483648")
        elif type(value) is not float:
            return None
    if not terms:
        return lambda a: ()
    body = ", ".join(terms)
    return eval(f"lambda a: ({body},)")  # noqa: S307 - built from indices only


class _Launcher:
    """``kernel[grid](*args, **meta)`` that skips Triton's dispatch once compiled."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.compiled = {}
        self.keys = {}
        # Positional parameters declared constexpr specialize on their value.
        self.constexpr = frozenset(i for i, p in enumerate(kernel.params) if p.is_constexpr)

    def launch(self, grid, args, meta):
        if not _FAST_LAUNCH:
            self.kernel[grid](*args, **meta)
            return
        layout = (tuple(map(type, args)), len(grid))
        keyed = self.keys.get(layout)
        if keyed is None:
            keyed = self.keys[layout] = _key_function(args, self.constexpr)
        if keyed is None:
            self.kernel[grid](*args, **meta)
            return
        device = _driver.active.get_current_device()
        key = (device, layout, keyed(args), tuple(meta.items()))
        entry = self.compiled.get(key)
        if entry is None:
            compiled = self.kernel[grid](*args, **meta)
            # The launcher takes the bound arguments in the binder's order.
            binder = self.kernel.device_caches[device][4]
            options = dict(
                meta,
                debug=self.kernel.debug or _knobs.runtime.debug,
                instrumentation_mode=_knobs.compilation.instrumentation_mode,
            )
            names = tuple(binder(*args, **options)[0])[len(args) :]
            # None when the call passes its meta in the binder's order.
            if names == tuple(meta):
                order = None
            elif len(names) > 1:
                order = operator.itemgetter(*names)
            else:
                order = lambda meta: tuple(meta[name] for name in names)  # noqa: E731
            self.compiled[key] = (compiled, order)
            return
        compiled, order = entry
        if _LAUNCH_CHECK:
            assert self.kernel.run(*args, grid=grid, warmup=True, **meta) is compiled
        values = args + (tuple(meta.values()) if order is None else order(meta))
        stream = _driver.active.get_current_stream(device)
        compiled.run(
            grid[0],
            grid[1] if len(grid) > 1 else 1,
            grid[2] if len(grid) > 2 else 1,
            stream,
            compiled.function,
            compiled.packed_metadata,
            compiled.launch_metadata(grid, stream, *values),
            _knobs.runtime.launch_enter_hook,
            _knobs.runtime.launch_exit_hook,
            *values,
        )


def _current_stream(device):
    """The raw current stream; Triton's driver query is much cheaper than torch's."""
    if _FAST_LAUNCH:
        return _driver.active.get_current_stream(device.index)
    return torch.cuda.current_stream(device).cuda_stream


_LAUNCHERS: dict = {}


def _launch(kernel, grid, *args, **meta):
    # Keyed by identity: hashing a JITFunction recomputes its cache key.
    launcher = _LAUNCHERS.get(id(kernel))
    if launcher is None or launcher.kernel is not kernel:
        launcher = _LAUNCHERS[id(kernel)] = _Launcher(kernel)
    launcher.launch(grid, args, meta)


@triton.jit
def _retained(LENGTHS, row, RETENTION, top):
    """The row length and its retained count, as the host's ``retained_count``."""
    length = tl.load(LENGTHS + row)
    # retained_count in float64, exactly as the host's math.ceil(length * r).
    real = length.to(tl.float64)
    count = tl.minimum(tl.maximum(tl.ceil(real * tl.load(RETENTION)), 1.0), real)
    return length, tl.minimum(count.to(tl.int32), top)


@triton.jit
def _segment(length, SEGMENTS: tl.constexpr, ALIGN: tl.constexpr):
    """This program's part ``[first, last)`` of a row's context."""
    size = tl.cdiv(tl.cdiv(length, SEGMENTS), ALIGN) * ALIGN
    first = tl.program_id(2) * size
    return first, tl.minimum(first + size, length)


@triton.jit
def _choose(histogram, prefix, ties_needed):
    """Highest digit whose keys, with every higher digit, still cover the rest."""
    digits = tl.arange(0, 256)
    covered = tl.cumsum(histogram, 0, reverse=True)
    chosen = tl.sum((covered >= ties_needed).to(tl.int32), axis=0) - 1
    ties_needed -= tl.sum(tl.where(digits > chosen, histogram, 0), axis=0)
    return prefix * 256 + chosen.to(tl.int64), ties_needed, chosen


@triton.jit
def _last_part(TICKETS, rh, PARTS: tl.constexpr):
    """Whether this program is the last of its row and head to finish; resets the ticket.

    Each part stores its results first; the acquire-release ticket then makes
    every part's stores visible to the last one, which reads them through L2.
    """
    tl.debug_barrier()
    last = tl.atomic_add(TICKETS + rh, 1, sem="acq_rel", scope="gpu") == PARTS - 1
    if last:
        tl.store(TICKETS + rh, 0)
    return last


@triton.jit
def _close_pass(
    SEGHIST, STATE, TICKETS, rh, count, PASS: tl.constexpr, USED: tl.constexpr, PARTS: tl.constexpr
):
    """The last part of a pass publishes the state before the next pass."""
    if _last_part(TICKETS, rh, USED):
        digits = tl.arange(0, 256)
        parts = (rh * PARTS + tl.arange(0, USED)) * 4 + PASS
        histogram = tl.sum(
            tl.load(SEGHIST + parts[:, None] * 256 + digits[None, :], cache_modifier=".cg"), axis=0
        )
        if PASS == 0:
            prefix = count.to(tl.int64) * 0
            ties_needed = count
        else:
            prefix = tl.load(STATE + (rh * 5 + PASS) * 2)
            ties_needed = tl.load(STATE + (rh * 5 + PASS) * 2 + 1).to(tl.int32)
        prefix, ties_needed, _ = _choose(histogram, prefix, ties_needed)
        tl.store(STATE + (rh * 5 + PASS + 1) * 2, prefix)
        tl.store(STATE + (rh * 5 + PASS + 1) * 2 + 1, ties_needed.to(tl.int64))


@triton.jit
def _select_pass_kernel(
    PART_M,
    PART_L,
    PART_O,
    OUT,
    SCORES,
    LENGTHS,
    RETENTION,
    SLOTS,
    COUNTS,
    OUTPUT,
    LSE,
    SEGHIST,
    STATE,
    TICKETS,
    GREATER,
    OFFSETS,
    layer,
    top,
    score_capacity,
    out_batch_stride: tl.constexpr,
    out_head_stride: tl.constexpr,
    out_dim_stride: tl.constexpr,
    PASS: tl.constexpr,
    MERGE: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PARTS: tl.constexpr,
    ALIGN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """One radix pass of one part; the first launched pass also merges the split parts."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    segment = tl.program_id(2)
    rh = row * NUM_HEADS + head
    length, count = _retained(LENGTHS, row, RETENTION, top)
    if MERGE:
        if segment == 0:
            # Identical to triton_attention._paged_attention_merge_kernel.
            parts = rh * SPLITS + tl.arange(0, SPLITS)
            dims = tl.arange(0, BLOCK_D)
            maxima = tl.load(PART_M + parts)
            maximum = tl.max(maxima, axis=0)
            scale = tl.where(maxima > -float("inf"), tl.exp(maxima - maximum), 0.0)
            normalizer = tl.sum(tl.load(PART_L + parts) * scale, axis=0)
            partial = tl.load(PART_O + parts[:, None] * BLOCK_D + dims[None, :])
            accumulator = tl.sum(partial * scale[:, None], axis=0)
            dense = accumulator / tl.maximum(normalizer, 1.0e-30)
            out = OUT + row * out_batch_stride + head * out_head_stride + dims * out_dim_stride
            tl.store(out, dense, mask=dims < HEAD_DIM)
            slot = tl.load(SLOTS + row)
            tl.store(LSE + rh, maximum + tl.log(normalizer))
            if count > 0:
                state = (slot * NUM_LAYERS + layer) * NUM_HEADS + head
                tl.store(COUNTS + state, count)
                # O_ref is the output as rounded to the query dtype: the recurrence input.
                reference = dense.to(OUT.dtype.element_ty).to(tl.float32)
                tl.store(OUTPUT + state * HEAD_DIM + dims, reference, mask=dims < HEAD_DIM)
    if count > 0:
        if PASS == 0:
            prefix = length.to(tl.int64) * 0
        else:
            prefix = tl.load(STATE + (rh * 5 + PASS) * 2)
        first, last = _segment(length, PARTS, ALIGN)
        scores_row = SCORES + rh.to(tl.int64) * score_capacity
        local = tl.zeros((512,), dtype=tl.int32)
        greater = 0
        for start in range(first, last, BLOCK_T):
            positions = start + tl.arange(0, BLOCK_T)
            live = positions < last
            key = _order_key(tl.load(scores_row + positions, mask=live, other=0))
            candidate = live & ((key >> (32 - 8 * PASS)) == prefix)
            digit = ((key >> (24 - 8 * PASS)) & 255).to(tl.int32)
            # Bin 256 collects the non-candidates and is never stored.
            local += tl.histogram(tl.where(candidate, digit, 256), 512)
            if PASS == 3:
                # Keys of this part already above the threshold's first 24 bits.
                greater += tl.sum((live & ((key >> 8) > prefix)).to(tl.int32), axis=0)
        bins = tl.arange(0, 512)
        own = (rh * PARTS + segment) * 4 + PASS
        tl.store(SEGHIST + own * 256 + bins, local, mask=bins < 256)
        if PASS == 3:
            tl.store(GREATER + rh * PARTS + segment, greater)
            if _last_part(TICKETS, rh, PARTS):
                # Fix the threshold, then every part's offset and tie quota.
                digits = tl.arange(0, 256)
                segments = tl.arange(0, PARTS)
                prefix = tl.load(STATE + (rh * 5 + 3) * 2)
                ties_needed = tl.load(STATE + (rh * 5 + 3) * 2 + 1).to(tl.int32)
                last_pass = tl.load(
                    SEGHIST + ((rh * PARTS + segments[:, None]) * 4 + 3) * 256 + digits[None, :],
                    cache_modifier=".cg",
                )
                low, ties_needed, chosen = _choose(tl.sum(last_pass, axis=0), prefix, ties_needed)
                above = tl.load(GREATER + rh * PARTS + segments, cache_modifier=".cg") + tl.sum(
                    tl.where(digits[None, :] > chosen, last_pass, 0), axis=1
                )
                equal = tl.sum(tl.where(digits[None, :] == chosen, last_pass, 0), axis=1)
                ties_before = tl.cumsum(equal, 0) - equal
                taken = above + tl.minimum(tl.maximum(ties_needed - ties_before, 0), equal)
                tl.store(OFFSETS + (rh * PARTS + segments) * 2, tl.cumsum(taken, 0) - taken)
                tl.store(OFFSETS + (rh * PARTS + segments) * 2 + 1, ties_before)
                tl.store(STATE + (rh * 5 + 4) * 2, low)
                tl.store(STATE + (rh * 5 + 4) * 2 + 1, ties_needed.to(tl.int64))
        else:
            _close_pass(SEGHIST, STATE, TICKETS, rh, count, PASS, PARTS, PARTS)


@triton.jit
def _select_write_kernel(
    SCORES,
    LENGTHS,
    RETENTION,
    SLOTS,
    STATE,
    OFFSETS,
    SELECTED,
    layer,
    top,
    score_capacity,
    retained_capacity,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PARTS: tl.constexpr,
    ALIGN: tl.constexpr,
):
    """Write this part's retained positions at its offset, ties up to its quota."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    segment = tl.program_id(2)
    rh = row * NUM_HEADS + head
    length, count = _retained(LENGTHS, row, RETENTION, top)
    if count > 0:
        low = tl.load(STATE + (rh * 5 + 4) * 2)
        ties_needed = tl.load(STATE + (rh * 5 + 4) * 2 + 1).to(tl.int32)
        written = tl.load(OFFSETS + (rh * PARTS + segment) * 2)
        ties_seen = tl.load(OFFSETS + (rh * PARTS + segment) * 2 + 1)
        slot = tl.load(SLOTS + row)
        selected = (
            SELECTED
            + ((slot * NUM_LAYERS + layer) * NUM_HEADS + head).to(tl.int64) * retained_capacity
        )
        scores_row = SCORES + rh.to(tl.int64) * score_capacity
        first, last = _segment(length, PARTS, ALIGN)
        for start in range(first, last, BLOCK_T):
            positions = start + tl.arange(0, BLOCK_T)
            live = positions < last
            key = _order_key(tl.load(scores_row + positions, mask=live, other=0))
            tie = (live & (key == low)).to(tl.int32)
            take = (live & (key > low)) | (
                (tie == 1) & (ties_seen + tl.cumsum(tie, 0) - tie < ties_needed)
            )
            taken = take.to(tl.int32)
            tl.store(selected + written + tl.cumsum(taken, 0) - taken, positions, mask=take)
            written += tl.sum(taken, axis=0)
            ties_seen += tl.sum(tie, axis=0)


@triton.jit
def _select_contribution_kernel(
    SCORES,
    V,
    TABLES,
    LENGTHS,
    RETENTION,
    SLOTS,
    LSE,
    SELECTED,
    PART_MASS,
    PART_C,
    TICKETS,
    MASS,
    CONTRIBUTION,
    layer,
    top,
    score_capacity,
    table_stride,
    retained_capacity,
    v_block_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PARTS: tl.constexpr,
):
    """``(m, C_ref)`` over the retained ranks, in rank chunks summed in a fixed order.

    Gathering V over the compact retained list keeps every lane busy, where a
    scan over all positions would leave most of them masked.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    part = tl.program_id(2)
    rh = row * NUM_HEADS + head
    length, count = _retained(LENGTHS, row, RETENTION, top)
    if count > 0:
        chunk = tl.cdiv(tl.cdiv(count, PARTS), BLOCK_T) * BLOCK_T
        first = part * chunk
        last = tl.minimum(first + chunk, count)
        lse = tl.load(LSE + rh)
        slot = tl.load(SLOTS + row)
        kv_head = head // HEAD_GROUPS
        dims = tl.arange(0, BLOCK_D)
        state = (slot * NUM_LAYERS + layer) * NUM_HEADS + head
        selected = SELECTED + state.to(tl.int64) * retained_capacity
        scores_row = SCORES + rh.to(tl.int64) * score_capacity
        table = TABLES + row * table_stride
        mass = 0.0
        contribution = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(first, last, BLOCK_T):
            ranks = start + tl.arange(0, BLOCK_T)
            live = ranks < last
            positions = tl.load(selected + ranks, mask=live, other=0)
            scores = tl.load(scores_row + positions, mask=live, other=0)
            probabilities = tl.where(live, tl.exp(scores - lse), 0.0)
            blocks = tl.load(table + positions // PAGE_SIZE, mask=live, other=0).to(tl.int64)
            values = tl.load(
                V
                + blocks[:, None] * v_block_stride
                + (positions % PAGE_SIZE)[:, None] * v_token_stride
                + kv_head * v_head_stride
                + dims[None, :],
                mask=live[:, None] & (dims[None, :] < HEAD_DIM),
                other=0,
            ).to(tl.float32)
            contribution += tl.sum(probabilities[:, None] * values, axis=0)
            mass += tl.sum(probabilities, axis=0)
        index = rh * PARTS + part
        tl.store(PART_MASS + index, mass)
        tl.store(PART_C + index * BLOCK_D + dims, contribution)
        if _last_part(TICKETS, rh, PARTS):
            parts = rh * PARTS + tl.arange(0, PARTS)
            total = tl.sum(tl.load(PART_MASS + parts, cache_modifier=".cg"), axis=0)
            summed = tl.sum(
                tl.load(PART_C + parts[:, None] * BLOCK_D + dims[None, :], cache_modifier=".cg"),
                axis=0,
            )
            tl.store(MASS + state, total)
            tl.store(CONTRIBUTION + state * HEAD_DIM + dims, summed, mask=dims < HEAD_DIM)


@triton.jit
def _capture_kernel(
    Q,
    K,
    V,
    TABLES,
    LENGTHS,
    RETENTION,
    SLOTS,
    DENSE,
    SCORES,
    SELECTED,
    COUNTS,
    MASS,
    OUTPUT,
    CONTRIBUTION,
    layer,
    top,
    score_capacity,
    table_stride,
    retained_capacity,
    q_row_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    k_block_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    v_block_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    dense_row_stride: tl.constexpr,
    dense_head_stride: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Select and publish the retained state ``(S, m, O_ref, C_ref)`` of one row and head.

    One program scores every key, finds the ``count``-th largest score by
    bisection on its order key, and writes the retained positions in ascending
    order; equal scores keep the smaller positions, like the oracle's stable
    sort.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(SLOTS + row)
    length = tl.load(LENGTHS + row)
    # retained_count in float64, exactly as the host's math.ceil(length * r).
    real = length.to(tl.float64)
    count = tl.minimum(tl.maximum(tl.ceil(real * tl.load(RETENTION)), 1.0), real)
    # key_bucket covers the serving context, so this never truncates; it only
    # keeps a misconfiguration from writing past the index buffer.
    count = tl.minimum(count.to(tl.int32), top)
    if count > 0:
        kv_head = head // HEAD_GROUPS
        dims = tl.arange(0, BLOCK_D)
        query = tl.load(
            Q + row * q_row_stride + head * q_head_stride + dims, mask=dims < HEAD_DIM, other=0
        ).to(tl.float32)
        table = TABLES + row * table_stride
        scores_row = SCORES + (row * NUM_HEADS + head).to(tl.int64) * score_capacity
        maximum = -float("inf")
        normalizer = 0.0
        for start in range(0, length, BLOCK_T):
            positions = start + tl.arange(0, BLOCK_T)
            live = positions < length
            blocks = tl.load(table + positions // PAGE_SIZE, mask=live, other=0).to(tl.int64)
            keys = tl.load(
                K
                + blocks[:, None] * k_block_stride
                + (positions % PAGE_SIZE)[:, None] * k_token_stride
                + kv_head * k_head_stride
                + dims[None, :],
                mask=live[:, None] & (dims[None, :] < HEAD_DIM),
                other=0,
            ).to(tl.float32)
            scores = tl.where(live, tl.sum(keys * query[None, :], axis=1) * SCALE, -float("inf"))
            tl.store(scores_row + positions, scores, mask=live)
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
            normalizer = normalizer * tl.exp(maximum - next_maximum) + tl.sum(
                tl.exp(scores - next_maximum), axis=0
            )
            maximum = next_maximum
        lse = maximum + tl.log(normalizer)
        # Later passes read scores other threads of this program stored.
        tl.debug_barrier()
        dense = tl.load(
            DENSE + row * dense_row_stride + head * dense_head_stride + dims,
            mask=dims < HEAD_DIM,
            other=0,
        ).to(tl.float32)
        _select_and_publish(
            scores_row,
            lse,
            length,
            count,
            slot,
            layer,
            head,
            dense,
            table,
            V,
            SELECTED,
            COUNTS,
            MASS,
            OUTPUT,
            CONTRIBUTION,
            retained_capacity,
            v_block_stride,
            v_token_stride,
            v_head_stride,
            NUM_LAYERS,
            NUM_HEADS,
            HEAD_DIM,
            HEAD_GROUPS,
            PAGE_SIZE,
            BLOCK_D,
            BLOCK_T,
        )


@triton.jit
def _consume_split_kernel(
    Q,
    K,
    V,
    TABLES,
    LENGTHS,
    SLOTS,
    SELECTED,
    COUNTS,
    PART_M,
    PART_L,
    PART_O,
    layer,
    table_stride,
    retained_capacity,
    q_row_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    k_block_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    v_block_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Online-softmax parts of ``C_new`` over one contiguous chunk of retained ranks.

    Each (row, head) splits its ``count`` retained keys into ``SPLITS`` chunks
    of whole ``BLOCK_T`` tiles; an empty chunk writes the merge's neutral part.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    dims = tl.arange(0, BLOCK_D)
    part = (row * NUM_HEADS + head) * SPLITS + split
    maximum = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if tl.load(LENGTHS + row) > 0:
        slot = tl.load(SLOTS + row)
        kv_head = head // HEAD_GROUPS
        state = (slot * NUM_LAYERS + layer) * NUM_HEADS + head
        count = tl.load(COUNTS + state)
        chunk = tl.cdiv(tl.cdiv(count, SPLITS), BLOCK_T) * BLOCK_T
        first = split * chunk
        last = tl.minimum(first + chunk, count)
        selected = SELECTED + state.to(tl.int64) * retained_capacity
        query = tl.load(
            Q + row * q_row_stride + head * q_head_stride + dims, mask=dims < HEAD_DIM, other=0
        ).to(tl.float32)
        for start in range(first, last, BLOCK_T):
            ranks = start + tl.arange(0, BLOCK_T)
            live = ranks < last
            positions = tl.load(selected + ranks, mask=live, other=0)
            blocks = tl.load(
                TABLES + row * table_stride + positions // PAGE_SIZE, mask=live, other=0
            ).to(tl.int64)
            offsets = positions % PAGE_SIZE
            keys = tl.load(
                K
                + blocks[:, None] * k_block_stride
                + offsets[:, None] * k_token_stride
                + kv_head * k_head_stride
                + dims[None, :],
                mask=live[:, None] & (dims[None, :] < HEAD_DIM),
                other=0,
            ).to(tl.float32)
            scores = tl.where(live, tl.sum(keys * query[None, :], axis=1) * SCALE, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
            correction = tl.exp(maximum - next_maximum)
            weights = tl.exp(scores - next_maximum)
            values = tl.load(
                V
                + blocks[:, None] * v_block_stride
                + offsets[:, None] * v_token_stride
                + kv_head * v_head_stride
                + dims[None, :],
                mask=live[:, None] & (dims[None, :] < HEAD_DIM),
                other=0,
            ).to(tl.float32)
            accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
            normalizer = normalizer * correction + tl.sum(weights, axis=0)
            maximum = next_maximum
    tl.store(PART_M + part, maximum)
    tl.store(PART_L + part, normalizer)
    tl.store(PART_O + part * BLOCK_D + dims, accumulator)


@triton.jit
def _consume_merge_kernel(
    PART_M,
    PART_L,
    PART_O,
    LENGTHS,
    SLOTS,
    MASS,
    OUTPUT,
    CONTRIBUTION,
    OUT,
    layer,
    q_row_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """``O = O_ref - C_ref + m * C_new``, merging the split parts in a fixed order."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    out = OUT + row * q_row_stride + head * q_head_stride + dims
    slot = tl.load(SLOTS + row)
    if tl.load(LENGTHS + row) == 0:
        tl.store(out, tl.zeros((BLOCK_D,), dtype=tl.float32), mask=dims < HEAD_DIM)
    else:
        state = (slot * NUM_LAYERS + layer) * NUM_HEADS + head
        parts = (row * NUM_HEADS + head) * SPLITS + tl.arange(0, SPLITS)
        maxima = tl.load(PART_M + parts)
        maximum = tl.max(maxima, axis=0)
        scale = tl.where(maxima > -float("inf"), tl.exp(maxima - maximum), 0.0)
        normalizer = tl.sum(tl.load(PART_L + parts) * scale, axis=0)
        partial = tl.load(PART_O + parts[:, None] * BLOCK_D + dims[None, :])
        accumulator = tl.sum(partial * scale[:, None], axis=0)
        reference = tl.load(OUTPUT + state * HEAD_DIM + dims, mask=dims < HEAD_DIM, other=0)
        contribution = tl.load(
            CONTRIBUTION + state * HEAD_DIM + dims, mask=dims < HEAD_DIM, other=0
        )
        # count 0 (no captured reference) divides 0 by 0: NaN, never a silent value.
        corrected = reference - contribution + tl.load(MASS + state) * (accumulator / normalizer)
        tl.store(out, corrected, mask=dims < HEAD_DIM)


def _retention(device, key_retention):
    """A float64 device scalar, built once; Triton would pass a Python float as fp32."""
    key = (device, key_retention)
    if key not in RETENTION_SCALARS:
        RETENTION_SCALARS[key] = torch.full((1,), key_retention, dtype=torch.float64, device=device)
    return RETENTION_SCALARS[key]


def _build_capture_workspace(device, rows, heads, capacity, block_d):
    empty = lambda *size, dtype=torch.float32: torch.empty(size, device=device, dtype=dtype)  # noqa: E731
    return (
        empty(rows, heads, capacity),
        empty(rows, heads, NUM_SPLITS),
        empty(rows, heads, NUM_SPLITS),
        empty(rows, heads, NUM_SPLITS, block_d),
        # Per-part radix histograms, the state before each pass (and the final
        # threshold), per-part keys above the threshold and output offsets, the
        # merged lse, the last-part tickets (zero between calls), and the
        # contribution parts.
        empty(rows, heads, SELECT_PARTS, 4, 256, dtype=torch.int32),
        empty(rows, heads, 5, 2, dtype=torch.int64),
        empty(rows, heads, SELECT_PARTS, dtype=torch.int32),
        empty(rows, heads, SELECT_PARTS, 2, dtype=torch.int32),
        empty(rows, heads),
        torch.zeros(rows, heads, device=device, dtype=torch.int32),
        empty(rows, heads, SELECT_PARTS),
        empty(rows, heads, SELECT_PARTS, block_d),
    )


def _capture_workspace(device, rows, heads, capacity, block_d):
    """Capture scratch reused across layers.

    Captures issued on one stream run in order, so each layer may overwrite
    the previous layer's scratch; one entry per stream keeps memory bounded.
    """
    stream = _current_stream(device)
    shape = (rows, heads, capacity, block_d)
    entry = CAPTURE_SCRATCH.get((device, stream))
    if entry is None or entry[0] != shape:
        entry = (shape, *_build_capture_workspace(device, rows, heads, capacity, block_d))
        CAPTURE_SCRATCH[(device, stream)] = entry
    return entry[1:]


def _consume_workspace(device, rows, heads, block_d):
    """Consume split parts, grown per stream and reused across layers."""
    stream = _current_stream(device)
    entry = CONSUME_SCRATCH.get((device, stream))
    if (
        entry is None
        or entry[0].shape[0] < rows
        or entry[0].shape[1] != heads
        or entry[2].shape[-1] != block_d
    ):
        empty = lambda *size: torch.empty(size, device=device, dtype=torch.float32)  # noqa: E731
        entry = (
            empty(rows, heads, NUM_SPLITS),
            empty(rows, heads, NUM_SPLITS),
            empty(rows, heads, NUM_SPLITS, block_d),
        )
        CONSUME_SCRATCH[(device, stream)] = entry
    return (None, *entry)


def supports(plan, q, dense_attention) -> bool:
    """Whether the fused path covers this submission; otherwise use the batched path."""
    modes = set(plan.modes)
    return (
        q.is_cuda
        and q.shape[-1] <= 256
        and plan.key_retention < 1.0
        and (modes == {CONSUME} or (modes == {CAPTURE} and dense_attention is not None))
    )


def capture_workspace(device, rows, heads, capacity, dim):
    """Capture scratch for at most ``rows`` rows, owned by the caller.

    A CUDA graph bakes the scratch addresses in, so its owner keeps one
    workspace alive for every bucket instead of the shape-keyed cache.
    """
    return _build_capture_workspace(device, rows, heads, capacity, triton.next_power_of_2(dim))


def fused_paged_attention(
    q,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    plan,
    *,
    layer,
    dense_attention,
    workspace=None,
):
    """Capture or consume one uniform-mode submission without host readbacks.

    The kernels also write the store's ``valid``/``uses`` marks, so the host
    only checks the plan's liveness instead of launching the bookkeeping.
    ``workspace`` comes from ``capture_workspace`` and must cover the rows
    and table width of the submission; consume uses only its split parts.
    """
    store = plan.store
    store._require_live(plan)
    rows, heads, dim = q.shape
    page = key_cache.shape[1]
    groups = heads // key_cache.shape[2]
    block_d = triton.next_power_of_2(dim)
    scale = dim**-0.5
    for cache in (key_cache, value_cache):
        if cache.stride(-1) != 1:
            raise ValueError("fused sparse attention needs a contiguous head dimension")
    if q.stride(-1) != 1:
        raise ValueError("fused sparse attention needs a contiguous query head dimension")
    selected = store.selected
    if not selected.is_contiguous():
        raise ValueError("fused sparse attention derives index offsets from a contiguous pool")
    if plan.modes[0] == CAPTURE:
        # With the Triton backend, the dense split kernel also keeps the scores
        # (and the first radix histograms), the dense merge runs inside the
        # first selection pass, and selection runs in parallel parts. Other
        # backends keep their call plus one serial selection kernel.
        # Engines pass their backend adapter, which wraps the Triton kernel;
        # callers may also pass the Triton function itself.
        kernel = getattr(dense_attention, "kernel", dense_attention)
        split = isinstance(kernel, TritonPagedAttention) or kernel is triton_paged_attention
        if split:
            output = torch.empty_like(q)
        else:
            output = dense_attention(q, key_cache, value_cache, block_tables, context_lengths)
        if rows == 0:
            return output
        # The round's page tables bound every row's context on the host.
        capacity = block_tables.shape[1] * page
        top = min(retained_count(capacity, plan.key_retention), store.key_bucket or capacity)
        store.ensure_key_capacity(top)
        selected = store.selected
        if workspace is None:
            workspace = _capture_workspace(q.device, rows, heads, capacity, block_d)
        elif workspace[0].shape[1:] != (heads, capacity) or workspace[0].shape[0] < rows:
            raise ValueError("capture workspace does not cover the submission")
        (
            scores,
            part_m,
            part_l,
            part_o,
            seghist,
            state,
            greater,
            offsets,
            lse,
            tickets,
            part_mass,
            part_c,
        ) = workspace if workspace[0].shape[0] == rows else (buffer[:rows] for buffer in workspace)
        if split:
            folded = _PASS0_IN_SPLIT
            retention = _retention(q.device, plan.key_retention)
            _launch(
                _capture_split_kernel,
                (rows, heads, NUM_SPLITS),
                q,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                part_m,
                part_l,
                part_o,
                scores,
                seghist,
                retention,
                state,
                tickets,
                top,
                capacity,
                *q.stride(),
                *key_cache.stride(),
                *value_cache.stride(),
                block_tables.stride(0),
                HEAD_DIM=dim,
                HEAD_GROUPS=groups,
                NUM_HEADS=heads,
                PAGE_SIZE=page,
                SCALE=scale,
                BLOCK_D=block_d,
                BLOCK_T=32,
                SPLITS=NUM_SPLITS,
                HISTOGRAM=folded,
                PARTS=SELECT_PARTS,
                num_warps=4,
            )
            first_pass = 1 if folded else 0
            for level in range(first_pass, 4):
                _launch(
                    _select_pass_kernel,
                    (rows, heads, SELECT_PARTS),
                    part_m,
                    part_l,
                    part_o,
                    output,
                    scores,
                    context_lengths,
                    retention,
                    plan.slot_ids,
                    store.counts,
                    store.output,
                    lse,
                    seghist,
                    state,
                    tickets,
                    greater,
                    offsets,
                    layer,
                    top,
                    capacity,
                    *output.stride(),
                    PASS=level,
                    MERGE=level == first_pass,
                    NUM_LAYERS=store.layers,
                    NUM_HEADS=heads,
                    HEAD_DIM=dim,
                    BLOCK_D=block_d,
                    BLOCK_T=_SCAN_BLOCK,
                    PARTS=SELECT_PARTS,
                    ALIGN=_SEGMENT_ALIGN,
                    SPLITS=NUM_SPLITS,
                    num_warps=_SCAN_WARPS,
                )
            _launch(
                _select_write_kernel,
                (rows, heads, SELECT_PARTS),
                scores,
                context_lengths,
                retention,
                plan.slot_ids,
                state,
                offsets,
                selected,
                layer,
                top,
                capacity,
                selected.shape[-1],
                NUM_LAYERS=store.layers,
                NUM_HEADS=heads,
                BLOCK_T=_SCAN_BLOCK,
                PARTS=SELECT_PARTS,
                ALIGN=_SEGMENT_ALIGN,
                num_warps=_SCAN_WARPS,
            )
            _launch(
                _select_contribution_kernel,
                (rows, heads, SELECT_PARTS),
                scores,
                value_cache,
                block_tables,
                context_lengths,
                retention,
                plan.slot_ids,
                lse,
                selected,
                part_mass,
                part_c,
                tickets,
                store.mass,
                store.contribution,
                layer,
                top,
                capacity,
                block_tables.stride(0),
                selected.shape[-1],
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                NUM_LAYERS=store.layers,
                NUM_HEADS=heads,
                HEAD_DIM=dim,
                HEAD_GROUPS=groups,
                PAGE_SIZE=page,
                BLOCK_D=block_d,
                BLOCK_T=_GATHER_BLOCK,
                PARTS=SELECT_PARTS,
                num_warps=4,
            )
            return output
        _launch(
            _capture_kernel,
            (rows, heads),
            q,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            _retention(q.device, plan.key_retention),
            plan.slot_ids,
            output,
            scores,
            selected,
            store.counts,
            store.mass,
            store.output,
            store.contribution,
            layer,
            top,
            capacity,
            block_tables.stride(0),
            selected.shape[-1],
            q.stride(0),
            q.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            output.stride(0),
            output.stride(1),
            NUM_LAYERS=store.layers,
            NUM_HEADS=heads,
            HEAD_DIM=dim,
            HEAD_GROUPS=groups,
            PAGE_SIZE=page,
            SCALE=scale,
            BLOCK_D=block_d,
            BLOCK_T=64,
            num_warps=4,
        )
        return output
    output = torch.empty_like(q)
    if rows:
        if workspace is None:
            workspace = _consume_workspace(q.device, rows, heads, block_d)
        elif workspace[1].shape[0] < rows or workspace[1].shape[1] != heads:
            raise ValueError("consume workspace does not cover the submission")
        part_m, part_l, part_o = (
            workspace[1:4] if workspace[1].shape[0] == rows else (b[:rows] for b in workspace[1:4])
        )
        _launch(
            _consume_split_kernel,
            (rows, heads, NUM_SPLITS),
            q,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            plan.slot_ids,
            selected,
            store.counts,
            part_m,
            part_l,
            part_o,
            layer,
            block_tables.stride(0),
            selected.shape[-1],
            q.stride(0),
            q.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            NUM_LAYERS=store.layers,
            NUM_HEADS=heads,
            HEAD_DIM=dim,
            HEAD_GROUPS=groups,
            PAGE_SIZE=page,
            SCALE=scale,
            BLOCK_D=block_d,
            BLOCK_T=32,
            SPLITS=NUM_SPLITS,
            num_warps=4,
        )
        _launch(
            _consume_merge_kernel,
            (rows, heads),
            part_m,
            part_l,
            part_o,
            context_lengths,
            plan.slot_ids,
            store.mass,
            store.output,
            store.contribution,
            output,
            layer,
            output.stride(0),
            output.stride(1),
            NUM_LAYERS=store.layers,
            NUM_HEADS=heads,
            HEAD_DIM=dim,
            BLOCK_D=block_d,
            SPLITS=NUM_SPLITS,
            num_warps=4,
        )
    return output
