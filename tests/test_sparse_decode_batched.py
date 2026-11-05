"""Batched CUDA arithmetic versus the original row-wise sparse oracle."""

import pytest
import torch

from tests.test_sparse_decode_attention import make_store
from vllm_rlt.attention.sparse import (
    CAPTURE,
    CONSUME,
    DENSE,
    batched_paged_attention,
    reference_paged_attention,
    slot_span,
)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("retention", [0.1, 1.0])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("use_dense_callback", [False, True])
def test_batched_ragged_gqa_matches_scalar_oracle(
    device, dtype, retention, ties, use_dense_callback
):
    torch.manual_seed(157)
    rows, heads, kv_heads, dim, block_size = 6, 4, 2, 128, 16
    lengths = torch.tensor([0, 17, 17, 31, 31, 257], device=device)
    tables = torch.randperm(rows * 17, device=device).reshape(rows, 17).int()
    keys = torch.randn(rows * 17, block_size, kv_heads, dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    queries = torch.randn(rows, heads, dim, device=device, dtype=dtype)
    if ties:
        queries.zero_()
    stores = [
        make_store(
            slots=rows,
            layers=1,
            heads=heads,
            head_dim=dim,
            key_bucket=257,
            device=device,
            initial_key_capacity=1 if implementation else None,
        )
        for implementation in range(2)
    ]
    leases = [store.reserve("test", 0, rows) for store in stores]
    implementations = [reference_paged_attention, batched_paged_attention]
    for modes in ([CAPTURE] * rows, [DENSE, CAPTURE, CONSUME, CONSUME, DENSE, CONSUME]):
        results = []
        for store, lease, implementation in zip(stores, leases, implementations):
            plan = store.build_plan(
                [(lease, row) for row in range(rows)], modes, key_retention=retention
            )
            # A deliberately distinct backend output verifies that capture
            # stores the rounded recurrence input, not its FP32 oracle PV.
            callback = (lambda q, *args: q * 0.375 + 1.25) if use_dense_callback else None
            results.append(
                implementation(
                    queries,
                    keys,
                    values,
                    tables,
                    lengths,
                    plan,
                    layer=0,
                    dense_attention=callback,
                )
            )
        torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)
        capacity = stores[1].selected.shape[-1]
        assert torch.equal(stores[0].selected[..., :capacity], stores[1].selected)
        assert not torch.count_nonzero(stores[0].selected[..., capacity:])
        for name in ("counts",):
            assert torch.equal(getattr(stores[0], name), getattr(stores[1], name)), name
        for name in ("mass", "output", "contribution"):
            torch.testing.assert_close(
                getattr(stores[0], name), getattr(stores[1], name), rtol=0, atol=0
            )
        queries = queries + 0.25
        keys = keys + 0.125
        values = values - 0.125
    for store, lease in zip(stores, leases):
        store.release(lease)
        assert store.outstanding == 0


def test_batched_rejects_missing_and_stale_reference():
    store = make_store(slots=1, layers=1, heads=2, head_dim=4)
    lease = store.reserve("test", 0, 1)
    plan = store.build_plan([(lease, 0)], [CONSUME], key_retention=0.1)
    args = (
        torch.ones(1, 2, 4),
        torch.ones(1, 16, 2, 4),
        torch.ones(1, 16, 2, 4),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.tensor([1]),
        plan,
    )
    with pytest.raises(RuntimeError, match="no captured reference"):
        batched_paged_attention(*args, layer=0)
    store.release(lease)
    store.reserve("test", 0, 1)
    with pytest.raises(RuntimeError, match="stale attention reference"):
        batched_paged_attention(*args, layer=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_reference_growth_preserves_live_layers_and_budget(device):
    store = make_store(
        slots=3,
        layers=2,
        heads=2,
        head_dim=4,
        key_bucket=17,
        initial_key_capacity=2,
        device=device,
    )
    leases = [store.reserve(str(i), i, 1) for i in range(2)]
    plans = [store.build_plan([(lease, 0)], [CAPTURE], key_retention=1.0) for lease in leases]
    # Simulate queued work on the runner's owner stream before reallocating.
    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        store.stream = stream
        context = torch.cuda.stream(stream)
    else:
        from contextlib import nullcontext

        context = nullcontext()
    budget = store.peak_capacity_bytes
    fixed = store.bytes - store.selected.nbytes
    assert budget == fixed + 3 * 2 * 2 * (2 + 4 + 8 + 16 + 17) * 4
    with context:
        for layer in range(2):
            store.write_reference(
                plans[0],
                layer,
                0,
                selected=torch.tensor([[1, 3], [2, 4]], device=device),
                counts=torch.full((2,), 2, device=device),
                mass=torch.full((2,), 0.5, device=device),
                output=torch.full((2, 4), 0.25, device=device),
                contribution=torch.full((2, 4), 0.125, device=device),
            )
        before = [x.clone() for x in store.read_reference(plans[0], 1, 0)]
        # Grow several times while a different lease remains live.
        for count in [3, 7, 17]:
            store.ensure_key_capacity(count)
            assert store.selected.shape[-1] >= count
            assert store.peak_capacity_bytes == budget
            for actual, expected in zip(store.read_reference(plans[0], 1, 0), before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with pytest.raises(ValueError, match="capacity"):
            store.ensure_key_capacity(18)
    if device == "cuda":
        store.stream.synchronize()
    for lease in leases:
        store.release(lease)
    assert store.outstanding == 0


@pytest.mark.gpu
def test_reference_growth_rejects_foreign_stream():
    store = make_store(key_bucket=17, initial_key_capacity=2, device="cuda")
    store.stream = torch.cuda.Stream()
    with pytest.raises(RuntimeError, match="owner stream"):
        store.ensure_key_capacity(3)
    assert store.selected.shape[-1] == 2


def test_slot_span_requires_every_slot_in_place():
    # Endpoint arithmetic accepts [5, 3, 7] (5 + 3 == 7 + 1) and would then use
    # the slice [5, 6, 7], which is another request's reference.
    assert slot_span([5, 3, 7]) is None
    assert slot_span([7, 6, 5]) is None
    assert slot_span([5, 6, 8]) is None
    assert slot_span([5, 5, 5]) is None
    assert slot_span([3, 4, 5]) == slice(3, 6)
    assert slot_span([5]) == slice(5, 6)


def _fragmented_store():
    """A store whose next three-row reservation yields slots ``[5, 3, 7]``.

    ``reserve`` reverses the tail of the free list, so releasing 7, then 3, then
    5 leaves ``[7, 3, 5]`` and the reservation comes back fragmented. Slots 6 and
    4 stay bound to unrelated live leases, which is what a span read of
    ``[5, 6, 7]`` would corrupt.
    """
    store = make_store(slots=8, layers=1, heads=2, head_dim=4, key_bucket=8)
    held = [store.reserve(f"held{i}", i, 1) for i in range(5)]  # slots 7, 6, 5, 4, 3
    store.release([held[0], held[4], held[2]])
    assert store.free[-3:] == [7, 3, 5]
    lease = store.reserve("fragmented", 0, 3)
    assert lease.slots == (5, 3, 7)
    return store, lease, held[1], held[3]


def test_fragmented_consume_reads_each_rows_own_reference():
    torch.manual_seed(157)
    store, lease, live_six, live_four = _fragmented_store()
    rows, heads, kv_heads, dim, block_size, extent = 3, 2, 2, 4, 8, 8
    # A zero query makes attention uniform, so the correction is identical for
    # every row and each output equals its own slot's reference plus one
    # constant. Reading the span [5, 6, 7] breaks exactly one row, by 3.
    queries = torch.zeros(rows, heads, dim)
    keys = torch.randn(1, block_size, kv_heads, dim)
    values = torch.randn(1, block_size, kv_heads, dim)
    tables = torch.zeros(rows, 1, dtype=torch.int32)
    lengths = torch.full((rows,), extent)

    store.output.copy_(torch.arange(8).float()[:, None, None, None])
    store.contribution.zero_()
    store.mass.fill_(1.0)
    store.counts.fill_(extent)
    store.selected.copy_(torch.arange(extent).expand(8, 1, heads, extent))
    for slot in lease.slots:
        store.note_count(slot, 0, extent)

    plan = store.build_plan(
        [(lease, 0), (lease, 1), (lease, 2)], [CONSUME] * rows, key_retention=0.5
    )
    output = batched_paged_attention(queries, keys, values, tables, lengths, plan, layer=0).float()

    own_reference = torch.tensor([5.0, 3.0, 7.0])[:, None, None]
    correction = output - own_reference
    torch.testing.assert_close(correction, correction[0].expand_as(correction), rtol=0, atol=1e-6)
    # The unrelated owners keep their payload.
    assert float(store.output[6, 0].mean()) == 6.0
    assert float(store.output[4, 0].mean()) == 4.0
    store.release([lease, live_six, live_four])


def test_fragmented_capture_writes_only_the_slots_it_reserved():
    torch.manual_seed(157)
    store, lease, live_six, live_four = _fragmented_store()
    rows, heads, kv_heads, dim, block_size, extent = 3, 2, 2, 4, 8, 8
    queries = torch.randn(rows, heads, dim)
    keys = torch.randn(1, block_size, kv_heads, dim)
    values = torch.randn(1, block_size, kv_heads, dim)
    tables = torch.zeros(rows, 1, dtype=torch.int32)
    lengths = torch.full((rows,), extent)

    # A sentinel per slot, so a write to the wrong slot is visible.
    sentinel = torch.arange(8).float() - 1000.0
    store.output.copy_(sentinel[:, None, None, None])
    store.contribution.copy_(sentinel[:, None, None, None])
    store.mass.copy_(sentinel[:, None, None])
    store.counts.fill_(-1)

    plan = store.build_plan(
        [(lease, 0), (lease, 1), (lease, 2)], [CAPTURE] * rows, key_retention=1.0
    )
    batched_paged_attention(queries, keys, values, tables, lengths, plan, layer=0)

    for slot in lease.slots:
        assert float(store.output[slot, 0].mean()) != float(sentinel[slot]), slot
        assert bool((store.counts[slot, 0] == extent).all()), slot
    # Slot 6 belongs to another request: the span [5, 6, 7] overwrote it and
    # left slot 3 unwritten.
    for slot in (4, 6):
        assert float(store.output[slot, 0].mean()) == float(sentinel[slot]), slot
        assert float(store.mass[slot, 0].mean()) == float(sentinel[slot]), slot
        assert bool((store.counts[slot, 0] == -1).all()), slot
    store.release([lease, live_six, live_four])
