import pytest
import torch
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager


def dense_attention(q, k, v):
    # Independent dense SDPA oracle: [batch=1, head, query/token, dim].
    groups = q.shape[0] // k.shape[1]
    return F.scaled_dot_product_attention(
        q.float()[None, :, None],
        k.float().repeat_interleave(groups, dim=1).transpose(0, 1)[None],
        v.float().repeat_interleave(groups, dim=1).transpose(0, 1)[None],
    )[0, :, 0].to(q.dtype)


def test_packed_prefill_is_causal_for_repeated_request_ids():
    torch.manual_seed(32)
    cache = KVCacheManager(1, 2, 7, 12, 2, 2)
    assert cache.allocate("a", 5)
    k, v = torch.randn(5, 2, 7), torch.randn(5, 2, 7)
    q = torch.randn(5, 4, 7)
    cache.write(0, ["a"] * 5, [0] * 5, torch.arange(5), k, v)
    actual = cache.attend(0, ["a"] * 5, [0] * 5, torch.arange(5), q)
    expected = torch.stack([dense_attention(q[p], k[: p + 1], v[: p + 1]) for p in range(5)])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def run_mixed_batch_case(device, backend, dtype, head_dim):
    torch.manual_seed(48)
    cache = KVCacheManager(2, 2, head_dim, 48, 3, 3, device=device, dtype=dtype, backend=backend)
    # Free a middle allocation so physical tables are not contiguous globally.
    cache.allocate("retained", 4)
    cache.allocate("hole", 8)
    cache.allocate("a", 7)
    cache.free("hole")
    cache.allocate("b", 6)
    histories = {}
    for request_id, length in [("a", 7), ("b", 6)]:
        for depth in range(3):
            k = torch.randn(length, 2, head_dim, device=device, dtype=dtype) + depth * 0.5
            v = torch.randn_like(k) + (4 if request_id == "a" else -4)
            cache.write(1, [request_id] * length, [depth] * length, list(range(length)), k, v)
            histories[request_id, depth] = k, v
    request_ids, depths, positions = ["a", "b", "a", "b"], [2, 0, 1, 2], [6, 4, 2, 5]
    q = torch.randn(4, 6, head_dim, device=device, dtype=dtype)
    actual = cache.attend(1, request_ids, depths, positions, q)
    expected_rows = []
    for row, (request_id, depth, position) in enumerate(zip(request_ids, depths, positions)):
        k, v = histories[request_id, depth]
        expected_rows.append(dense_attention(q[row], k[: position + 1], v[: position + 1]))
    expected = torch.stack(expected_rows)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("head_dim", [7, 32])
def test_mixed_depth_and_request_batch_matches_dense(head_dim):
    run_mixed_batch_case("cpu", "torch", torch.float32, head_dim)


def run_online_softmax_case(device, backend, dtype):
    torch.manual_seed(89)
    cache = KVCacheManager(1, 1, 32, 30, 5, 2, device=device, dtype=dtype, backend=backend)
    cache.allocate("long", 71)
    k = torch.randn(71, 1, 32, device=device, dtype=dtype) * 8
    v = torch.randn_like(k)
    cache.write(0, ["long"] * 71, [1] * 71, list(range(71)), k, v)
    # Exercise several online-softmax tiles, irregular page boundaries, and
    # different causal lengths within the same physical block table.
    positions = [0, 31, 32, 70]
    q = torch.randn(4, 2, 32, device=device, dtype=dtype) * 8
    actual = cache.attend(0, ["long"] * 4, [1] * 4, positions, q)
    expected = torch.stack(
        [dense_attention(q[row], k[: p + 1], v[: p + 1]) for row, p in enumerate(positions)]
    )
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 2e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_long_context_reference_is_numerically_stable():
    run_online_softmax_case("cpu", "torch", torch.float32)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [7, 32, 64, 128])
def test_triton_matches_dense_on_reserved_gpu(dtype, head_dim):
    run_mixed_batch_case("cuda", "triton", dtype, head_dim)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_online_softmax_on_reserved_gpu(dtype):
    run_online_softmax_case("cuda", "triton", dtype)


def test_empty_batch_and_invalid_gqa():
    cache = KVCacheManager(1, 2, 4, 6, 2, 2)
    empty = torch.empty(0, 2, 4)
    cache.write(0, [], [], [], empty, empty)
    assert cache.attend(0, [], [], [], torch.empty(0, 4, 4)).shape == (0, 4, 4)
    cache.allocate("a", 1)
    cache.write(0, ["a"], [0], [0], torch.ones(1, 2, 4), torch.ones(1, 2, 4))
    with pytest.raises(ValueError, match="positive multiple"):
        cache.attend(0, ["a"], [0], [0], torch.ones(1, 3, 4))


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim,heads,kv_heads", [(64, 4, 2), (96, 8, 8), (128, 16, 16)])
def test_triton_paged_prefill_matches_per_query_reference(head_dim, heads, kv_heads):
    from vllm_rlt.kernels.paged_attention import torch_paged_attention
    from vllm_rlt.kernels.triton_attention import paged_prefill_attention

    torch.manual_seed(0)
    block, blocks = 16, 96
    k = torch.randn(blocks, block, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # (keys visible to the last query, queries in this chunk): prefixes of 32, 0, 0, 60, 9.
    sequences = [(37, 5), (100, 100), (3, 3), (130, 70), (129, 120)]
    order = torch.randperm(blocks).tolist()  # Pages deliberately out of order.
    width = max(-(-length // block) for length, _ in sequences)
    tables = torch.zeros(len(sequences), width, dtype=torch.int32, device="cuda")
    for s, (length, _) in enumerate(sequences):
        pages = [order.pop() for _ in range(-(-length // block))]
        tables[s, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
    total = sum(count for _, count in sequences)
    q = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([length for length, _ in sequences], dtype=torch.int32, device="cuda")
    starts = [0]
    for _, count in sequences:
        starts.append(starts[-1] + count)
    cu = torch.tensor(starts, dtype=torch.int32, device="cuda")
    actual = paged_prefill_attention(
        q, k, v, tables, lengths, cu, max(count for _, count in sequences)
    )
    rows, row_tables, row_lengths = [], [], []
    for s, (length, count) in enumerate(sequences):
        for i in range(count):
            rows.append(starts[s] + i)
            row_tables.append(tables[s])
            row_lengths.append(length - count + i + 1)  # Causal: keys up to own position.
    expected = torch_paged_attention(
        q[rows],
        k,
        v,
        torch.stack(row_tables),
        torch.tensor(row_lengths, dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(actual[rows], expected, rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_split_kv_decode_matches_reference_and_is_batch_invariant(dtype):
    from vllm_rlt.kernels.paged_attention import torch_paged_attention
    from vllm_rlt.kernels.triton_attention import paged_attention

    torch.manual_seed(1)
    block, heads, kv_heads, head_dim = 16, 8, 4, 128
    # Empty, tile and split boundaries, and long rows that use every split.
    lengths = [0, 1, 31, 32, 33, 511, 512, 513, 4097, 10000]
    pages = sum(-(-n // block) for n in lengths)
    k = torch.randn(pages, block, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    order = torch.randperm(pages).tolist()  # Pages deliberately out of order.
    width = max(-(-n // block) for n in lengths)
    tables = torch.zeros(len(lengths), width, dtype=torch.int32, device="cuda")
    for row, n in enumerate(lengths):
        ids = [order.pop() for _ in range(-(-n // block))]
        tables[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    q = torch.randn(len(lengths), heads, head_dim, device="cuda", dtype=dtype)
    context = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    actual = paged_attention(q, k, v, tables, context)
    expected = torch_paged_attention(q, k, v, tables, context)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert torch.count_nonzero(actual[0]) == 0
    # Speculative verification compares rows computed in different batches.
    for row in (1, 5, 8, 9):
        one = slice(row, row + 1)
        alone = paged_attention(q[one], k, v, tables[one], context[one])
        assert torch.equal(alone[0], actual[row])
