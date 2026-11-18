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
    leases = [store.reserve("test", 0, rows, generation=1) for store in stores]
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
        for name in ("counts", "valid", "uses"):
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
    lease = store.reserve("test", 0, 1, generation=1)
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
    store.reserve("test", 0, 1, generation=2)
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
    leases = [store.reserve(str(i), i, 1, generation=1) for i in range(2)]
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
