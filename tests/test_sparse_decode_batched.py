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
def test_batched_ragged_gqa_matches_scalar_oracle(device, dtype, retention, ties):
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
        make_store(slots=rows, layers=1, heads=heads, head_dim=dim, key_bucket=257, device=device)
        for _ in range(2)
    ]
    leases = [store.reserve("test", 0, rows, generation=1) for store in stores]
    implementations = [reference_paged_attention, batched_paged_attention]
    for modes in ([CAPTURE] * rows, [DENSE, CAPTURE, CONSUME, CONSUME, DENSE, CONSUME]):
        results = []
        for store, lease, implementation in zip(stores, leases, implementations):
            plan = store.build_plan(
                [(lease, row) for row in range(rows)], modes, key_retention=retention
            )
            results.append(implementation(queries, keys, values, tables, lengths, plan, layer=0))
        torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)
        for name in ("selected", "counts", "valid", "uses"):
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
