"""The eager cohort store is authoritative until an ordered paged materialization."""

from dataclasses import replace

import pytest
import torch

from vllm_rlt.attention.backend import AttentionBackend, BackendCapabilities
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention
from vllm_rlt.worker.metadata import stage_range_metadata


def cache_for(device, enabled=True):
    device = torch.device(device)
    attention = None
    if device.type == "cpu":
        attention = AttentionBackend(
            AscendEagerAttention(),
            BackendCapabilities(sequence_ranges=True, owned_contiguous_kv=True),
            dict(selected_backend="torch", backend="torch"),
        )
    cache = KVCacheManager(
        2,
        2,
        32,
        64,
        4,
        2,
        device=device,
        dtype=torch.bfloat16,
        backend="ascend_eager" if device.type == "npu" else "torch",
        attention=attention,
        contiguous_kv=enabled,
    )
    for rid in ["a", "b"]:
        assert cache.allocate(rid, 16)
    return cache


def metadata(cache, ids, start, count, depth=0):
    layouts = cache.metadata_builder.build_ranges(
        ids, [start] * len(ids), [count] * len(ids), [[depth] * len(ids)]
    )
    batch = stage_range_metadata(layouts, cache._stage)[0]
    return replace(batch, block_tables=batch.table_plan.bind(cache.device))


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu:0", marks=pytest.mark.npu)])
@torch.inference_mode()
def test_cohort_kv_matches_paged_and_materializes_all_layers(device):
    if device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(device)
    torch.manual_seed(41)
    dense, paged = cache_for(device), cache_for(device, False)
    held = snapshot = None
    for start, count in [(0, 5), (5, 3), (8, 1)]:
        for depth in range(2):
            a, b = (
                metadata(dense, ["a", "b"], start, count, depth),
                metadata(paged, ["a", "b"], start, count, depth),
            )
            for layer in range(2):
                k = torch.randn(2 * count, 2, 32, device=device, dtype=torch.bfloat16)
                v, q = torch.randn_like(k), torch.randn_like(k)
                dense._write_prepared(layer, a, k, v)
                paged._write_prepared(layer, b, k, v)
                k.fill_(float("nan"))
                v.fill_(float("nan"))
                out = dense._attend_prepared(layer, a, q)
                expected = paged._attend_prepared(layer, b, q)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                if held is not None:
                    torch.testing.assert_close(held, snapshot, rtol=0, atol=0)
                held, snapshot = out, out.clone()
                assert dense._key_cache is None
    for depth in range(2):
        for layer in range(2):
            for rid in ["a", "b"]:
                for x, y in zip(dense.read(layer, rid, depth, 9), paged.read(layer, rid, depth, 9)):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert dense._contiguous is None


@torch.inference_mode()
def test_cancel_regroup_and_id_reuse_preserve_survivor_history():
    cache = cache_for("cpu")
    histories = {}

    def write(ids, start, count):
        b = metadata(cache, ids, start, count)
        k = torch.randn(len(ids) * count, 2, 32, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        for i, rid in enumerate(ids):
            incoming = (
                k[i * count : (i + 1) * count].clone(),
                v[i * count : (i + 1) * count].clone(),
            )
            histories[rid] = (
                incoming
                if start == 0
                else tuple(torch.cat((old, new)) for old, new in zip(histories[rid], incoming))
            )
        cache._write_prepared(0, b, k, v)
        return b

    stale = write(["a", "b"], 0, 5)
    cache.free("a")
    assert cache.allocate("a", 16)
    with pytest.raises(RuntimeError, match="stale"):
        cache._attend_prepared(0, stale, torch.zeros(10, 2, 32, dtype=torch.bfloat16))
    write(["a"], 0, 5)
    write(["b", "a"], 5, 2)
    assert cache._key_cache is None
    for rid in ["a", "b"]:
        for actual, expected in zip(cache.read(0, rid, 0, 7), histories[rid]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cache.free("a")
    cache.free("b")
    assert cache.num_used_blocks == 0


@torch.inference_mode()
def test_hole_and_overwrite_escape_to_paged_preserves_untouched_suffix():
    cache = cache_for("cpu")
    b = metadata(cache, ["a"], 0, 5)
    k = torch.randn(5, 2, 32, dtype=torch.bfloat16)
    cache._write_prepared(0, b, k, k)
    overwrite = metadata(cache, ["a"], 0, 1)
    patch = torch.randn(1, 2, 32, dtype=torch.bfloat16)
    cache._write_prepared(0, overwrite, patch, patch)
    assert cache._contiguous is None
    expected = k.clone()
    expected[:1] = patch
    torch.testing.assert_close(cache.read(0, "a", 0, 5)[0], expected, rtol=0, atol=0)
    gap = metadata(cache, ["a"], 7, 1)
    cache._write_prepared(0, gap, patch, patch)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache._attend_prepared(0, gap, patch)


@pytest.mark.npu
@torch.inference_mode()
def test_cross_stream_escape_orders_producer_and_keeps_output_owned():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    cache = cache_for("npu:0")
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(producer):
        b = metadata(cache, ["a", "b"], 0, 5)
        k = torch.randn(10, 2, 32, device="npu:0", dtype=torch.bfloat16)
        expected = k.clone()
        cache._write_prepared(0, b, k, k)
        out = cache._attend_prepared(0, b, k)
        snapshot = out.clone()
    with torch.npu.stream(consumer):
        actual, _ = cache.read(0, "a", 0, 5)
        assert cache._contiguous is None
        torch.testing.assert_close(actual, expected[:5], rtol=0, atol=0)
    torch.npu.synchronize()
    torch.testing.assert_close(out, snapshot, rtol=0, atol=0)


@torch.inference_mode()
def test_stranded_group_compaction_exports_correct_survivor_rows():
    cache = cache_for("cpu")
    assert cache.allocate("c", 16)
    initial = metadata(cache, ["a", "b", "c"], 0, 5)
    k = torch.randn(15, 2, 32, dtype=torch.bfloat16)
    cache._write_prepared(0, initial, k, k)
    advance = metadata(cache, ["a"], 5, 1)
    tail = torch.randn(1, 2, 32, dtype=torch.bfloat16)
    cache._write_prepared(0, advance, tail, tail)
    assert cache._contiguous.compactions == 1
    assert cache._key_cache is None
    for i, rid in enumerate(["a", "b", "c"]):
        expected = k[i * 5 : (i + 1) * 5]
        if rid == "a":
            expected = torch.cat((expected, tail))
        actual, _ = cache.read(0, rid, 0, len(expected))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.read(1, "a", 0, 1)


def test_contiguous_configuration_rejects_shared_and_unreserved_modes():
    from vllm_rlt import CacheConfig

    for kwargs in [
        dict(layout="shared"),
        dict(incremental_allocation=True),
        dict(enable_prefix_caching=True),
    ]:
        with pytest.raises(ValueError, match="contiguous KV"):
            CacheConfig(contiguous_kv=True, **kwargs)
    with pytest.raises(ValueError, match="sequence-range"):
        KVCacheManager(1, 1, 32, 8, 4, 1, contiguous_kv=True)


def test_sequence_metadata_capability_does_not_imply_owned_contiguous_kv():
    attention = AttentionBackend(
        AscendEagerAttention(),
        BackendCapabilities(sequence_ranges=True),
        dict(selected_backend="torch", backend="torch"),
    )
    with pytest.raises(ValueError, match="owned contiguous attention"):
        KVCacheManager(2, 2, 32, 16, 4, 2, device="cpu", attention=attention, contiguous_kv=True)
