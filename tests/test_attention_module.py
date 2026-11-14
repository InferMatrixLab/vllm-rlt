"""M8 contracts independent of logical KV allocation and device ownership."""

from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.attention import (
    AttentionRows,
    BackendCapabilities,
    backend_capabilities,
    build_attention_metadata,
    create_backend,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager


def test_backend_factory_preserves_reference_and_explicit_rejections():
    backend = create_backend("torch", torch.device("cpu"), torch.float32, 8, 2)
    assert callable(backend)
    with pytest.raises(ValueError, match="unknown attention backend"):
        create_backend("unknown", torch.device("cpu"), torch.float32, 8, 2)
    with pytest.raises(ValueError, match="requires a CUDA or ROCm device"):
        create_backend("triton", torch.device("cpu"), torch.float32, 8, 2)
    with pytest.raises(ValueError, match="head_dim <= 256"):
        create_backend("triton", torch.device("cuda"), torch.float32, 257, 2)


def test_backend_name_is_checked_before_other_cache_options():
    with pytest.raises(ValueError, match="unknown attention backend"):
        KVCacheManager(1, 1, 8, 4, 2, 1, dtype=torch.int32, backend="missing")


def test_packed_prefill_capability_is_an_explicit_backend_contract():
    reference = create_backend("torch", torch.device("cpu"), torch.float32, 8, 2)
    assert backend_capabilities(reference).packed_prefill is False
    declared = SimpleNamespace(generation=2, capabilities=BackendCapabilities(packed_prefill=True))
    assert backend_capabilities(declared).packed_prefill is True
    cache = KVCacheManager(1, 1, 8, 4, 2, 1)
    assert cache.attention_capabilities.packed_prefill is False
    cache.attention = declared
    assert cache.attention_capabilities.packed_prefill is True


def test_metadata_is_depth_selected_padded_and_borrowed_for_one_traversal():
    rows = AttentionRows(block_tables=[(4, 7, 8), (2, 3)], positions=[4, 1])
    metadata = build_attention_metadata(rows, block_size=2, device=torch.device("cpu"))
    assert metadata.block_tables.tolist() == [[4, 7, 8], [2, 3, -1]]
    assert metadata.context_lengths.tolist() == [5, 2]
    assert metadata.block_tables.dtype == metadata.context_lengths.dtype == torch.int32
    assert metadata.cu_seqlens_q is None
    assert metadata.max_seqlen_q == 1


def test_packed_metadata_groups_contiguous_query_chunks():
    rows = AttentionRows(
        block_tables=[(4, 7, 8), (4, 7, 8), (2, 3)],
        positions=[2, 3, 1],
        sequence_keys=[("a", 0), ("a", 0), ("b", 1)],
    )
    metadata = build_attention_metadata(
        rows, block_size=2, device=torch.device("cpu"), packed_prefill=True
    )
    assert metadata.block_tables.tolist() == [[4, 7], [2, 3]]
    assert metadata.context_lengths.tolist() == [4, 2]
    assert metadata.cu_seqlens_q.tolist() == [0, 2, 3]
    assert metadata.max_seqlen_q == 2
    with pytest.raises(ValueError, match="contiguous"):
        build_attention_metadata(
            AttentionRows(
                block_tables=[(4, 7, 8), (4, 7, 8)],
                positions=[2, 4],
                sequence_keys=[("a", 0), ("a", 0)],
            ),
            block_size=2,
            device=torch.device("cpu"),
            packed_prefill=True,
        )
    with pytest.raises(ValueError, match="one sequence"):
        build_attention_metadata(
            AttentionRows(
                block_tables=[(4, 7, 8), (2, 3), (4, 7, 8)],
                positions=[2, 1, 3],
                sequence_keys=[("a", 0), ("b", 1), ("a", 0)],
            ),
            block_size=2,
            device=torch.device("cpu"),
            packed_prefill=True,
        )


def test_metadata_rejects_misaligned_row_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        build_attention_metadata(
            AttentionRows(block_tables=[(1,)], positions=[0, 1]),
            block_size=2,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="equal lengths"):
        build_attention_metadata(
            AttentionRows(block_tables=[(1,)], positions=[0], sequence_keys=[("a", 0), ("a", 0)]),
            block_size=2,
            device=torch.device("cpu"),
            packed_prefill=True,
        )


def test_host_attention_plan_builds_packed_values_without_device_work(monkeypatch):
    from vllm_rlt.attention import metadata

    planner = getattr(metadata, "plan_attention_metadata", None)
    assert callable(planner), "M8 must provide host values for the shared staging path"

    def unexpected_tensor(*args, **kwargs):
        pytest.fail("planning semantic metadata must not allocate or copy tensors")

    monkeypatch.setattr(torch, "tensor", unexpected_tensor)
    plan = planner(
        AttentionRows(
            block_tables=[(4, 7, 8), (4, 7, 8), (2,)],
            positions=[2, 3, 1],
            sequence_keys=[("a", 0), ("a", 0), ("b", 1)],
        ),
        block_size=2,
        packed_prefill=True,
    )
    assert plan.block_tables == [4, 7, 2, -1]
    assert (plan.num_rows, plan.table_width) == (2, 2)
    assert plan.context_lengths == [4, 2]
    assert plan.cu_seqlens_q == [0, 2, 3]
    assert plan.max_seqlen_q == 2


@pytest.mark.parametrize("packed", [False, True])
def test_depth_batches_stage_metadata_twice_and_preserve_shared_views(monkeypatch, packed):
    cache = KVCacheManager(1, 1, 8, 48, 2, 4)
    cache.attention = SimpleNamespace(capabilities=BackendCapabilities(packed_prefill=True))
    cache.allocate("a", 6)
    cache.allocate("b", 6)
    original_stage = cache._stage
    stages = []

    def stage(values, dtype):
        stages.append(dtype)
        return original_stage(values, dtype)

    monkeypatch.setattr(cache, "_stage", stage)
    batches = cache._prepare_batches(
        ["a", "a", "b"], [[0, 0, 0], [1, 1, 1], [2, 2, 2]], [2, 3, 1], packed_prefill=packed
    )
    assert stages == [torch.long, torch.int32]
    for name in ("position_ids", "write_offsets", "context_lengths"):
        assert len({getattr(batch, name).data_ptr() for batch in batches}) == 1
    if packed:
        assert len({batch.cu_seqlens_q.data_ptr() for batch in batches}) == 1
        assert batches[0].cu_seqlens_q.tolist() == [0, 2, 3]
        assert batches[0].max_seqlen_q == 2


def test_prepared_attention_keeps_the_legacy_kernel_signature():
    cache = KVCacheManager(1, 1, 4, 4, 2, 1)
    assert cache.allocate("a", 3)
    batch = cache._prepare_batch(["a"] * 3, [0] * 3, [0, 1, 2])
    keys = torch.zeros(3, 1, 4)
    values = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
    query = torch.zeros(3, 2, 4)
    cache._write_prepared(0, batch, keys, values)
    kernel = cache.attention.kernel
    expected = kernel(
        query,
        cache.key_cache[:, 0],
        cache.value_cache[:, 0],
        batch.block_tables,
        batch.context_lengths,
    )
    seen = []

    def legacy(*args):
        seen.append(args)
        return kernel(*args)

    cache.attention.kernel = legacy
    actual = cache._attend_prepared(0, batch, query)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(seen) == 1 and len(seen[0]) == 5
    assert seen[0][0] is query
    assert seen[0][3] is batch.block_tables
    assert seen[0][4] is batch.context_lengths
    cache.free("a")


def test_metadata_attention_borrows_only_a_live_validated_descriptor():
    cache = KVCacheManager(1, 1, 4, 4, 2, 1)
    assert cache.allocate("a", 3)
    batch = cache._prepare_batch(["a"] * 3, [0] * 3, [0, 1, 2])
    keys = torch.zeros(3, 1, 4)
    values = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
    query = torch.zeros(3, 2, 4)
    cache._write_prepared(0, batch, keys, values)
    legacy = cache.attention.kernel
    expected = legacy(
        query,
        cache.key_cache[:, 0],
        cache.value_cache[:, 0],
        batch.block_tables,
        batch.context_lengths,
    )
    seen = []

    class MetadataKernel:
        def __call__(self, *args):
            pytest.fail("metadata-aware kernels must not read device metadata back")

        def attend_metadata(self, q, k, v, descriptor):
            seen.append(descriptor)
            return legacy(q, k, v, descriptor.block_tables, descriptor.context_lengths)

    cache.attention.kernel = MetadataKernel()
    actual = cache._attend_prepared(0, batch, query)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(seen) == 1 and seen[0] is batch
    cache.free("a")
    assert cache.allocate("a", 3)
    with pytest.raises((ValueError, RuntimeError)):
        cache._attend_prepared(0, batch, query)
    assert len(seen) == 1
    cache.free("a")
