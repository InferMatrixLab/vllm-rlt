"""Metadata reuse must preserve sparse coverage, ownership and eager values."""

import gc
import weakref

import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _sequence_coverage, _WrittenPositions
from vllm_rlt.kernels.ascend_attention import AscendAttention, _flat_kv_rows


@pytest.mark.parametrize(
    "ranges",
    [
        [(0, 1024), (1024, 4096)],
        [(5, 9), (0, 3), (3, 5)],
        [(9, 12), (4, 7), (0, 6), (6, 10)],
        [(2, 5), (2, 7), (0, 3), (1, 4), (9, 9)],
    ],
)
def test_written_ranges_preserve_sparse_and_overlapping_coverage(ranges):
    reference, actual = _WrittenPositions(), _WrittenPositions()
    for start, end in ranges:
        for position in range(start, end):
            reference.add(position)
        actual.add_range(start, end)
        assert (actual.prefix, actual.pending) == (reference.prefix, reference.pending)


def test_sequence_coverage_keeps_holes_and_plane_identity():
    first, second = object(), object()
    rows = [(first, 0, p) for p in (2, 0, 1, 5)] + [(first, 1, 0), (second, 0, 0)]
    plans = _sequence_coverage(rows, lambda depth: depth)
    assert [(p.allocation, p.plane, p.ranges) for p in plans] == [
        (first, 0, ((0, 3), (5, 6))),
        (first, 1, ((0, 1),)),
        (second, 0, ((0, 1),)),
    ]


def _cache():
    from vllm_rlt.device import resolve_device

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    cache = KVCacheManager(
        2, 2, 64, 64, 16, 4, device=device, dtype=torch.bfloat16, backend="ascend_eager"
    )
    cache.key_cache.fill_(float("nan"))
    cache.value_cache.fill_(float("nan"))
    lengths = [3, 7, 17]
    for index, length in enumerate(lengths):
        rid = str(index)
        assert cache.allocate(rid, length)
        for depth in (0, 1):
            batch = cache._prepare_batch([rid] * length, [depth] * length, list(range(length)))
            for layer in range(2):
                k = torch.randn(length, 2, 64, device=device, dtype=torch.bfloat16)
                v = torch.randn_like(k)
                cache._write_prepared(layer, batch, k, v)
    return cache, lengths


def _reference(kernel, q, keys, values, batch):
    groups = kernel._plan(
        [(id(a), depth) for a, depth, _ in batch.rows],
        [p + 1 for _, _, p in batch.rows],
        q.device,
    )
    return kernel._forward(
        q, keys, values, batch.block_tables, groups, flat_rows=_flat_kv_rows(keys, values)
    )


@pytest.mark.npu
@pytest.mark.parametrize("padded", [False, True])
@torch.inference_mode()
def test_unwritten_nan_padding_is_never_consumed(padded):
    cache, lengths = _cache()
    batch = cache._prepare_batch(
        ["0", "1", "2"],
        [0] * 3,
        [n - 1 for n in lengths],
        for_write=False,
        padded_rows=4 if padded else None,
    )
    # Noncontiguous Q must retain the original contiguous attention layout.
    q = torch.randn(4 if padded else 3, 4, 128, device=cache.device, dtype=cache.dtype)[:, :, ::2]
    for layer in range(2):
        keys, values = cache.key_cache[:, layer], cache.value_cache[:, layer]
        expected = _reference(AscendAttention(), q, keys, values, batch)
        actual = cache._attend_prepared(layer, batch, q)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if padded:
            assert torch.count_nonzero(actual[3]) == 0
    for rid in ("0", "1", "2"):
        cache.free(rid)


@pytest.mark.npu
@torch.inference_mode()
def test_layout_reuse_does_not_reuse_addresses_or_outputs():
    cache, lengths = _cache()
    kernel = cache.attention.kernel
    q = torch.randn(3, 4, 64, device=cache.device, dtype=cache.dtype)
    saved = None
    for depth in (0, 1, 0):
        batch = cache._prepare_batch(
            ["0", "1", "2"], [depth] * 3, [n - 1 for n in lengths], for_write=False
        )
        expected = _reference(
            AscendAttention(), q, cache.key_cache[:, 0], cache.value_cache[:, 0], batch
        )
        actual = cache._attend_prepared(0, batch, q)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if saved is None:
            saved, snapshot = actual, actual.clone()
    assert kernel.layout_reuses >= 2
    torch.testing.assert_close(saved, snapshot, rtol=0, atol=0)
    reference = weakref.ref(batch)
    del batch
    gc.collect()
    assert reference() is None
    for rid in ("0", "1", "2"):
        cache.free(rid)


@pytest.mark.npu
@torch.inference_mode()
def test_cached_validation_keeps_layer_holes_truncation_and_reused_id_checks():
    cache, _ = _cache()
    batch = cache._prepare_batch(["0", "0"], [2, 2], [0, 2])
    k = torch.randn(2, 2, 64, device=cache.device, dtype=cache.dtype)
    q = torch.randn(2, 4, 64, device=cache.device, dtype=cache.dtype)
    cache._write_prepared(0, batch, k, k)
    with pytest.raises(RuntimeError, match="uninitialized KV"):
        cache._attend_prepared(0, batch, q)
    middle = cache._prepare_batch(["0"], [2], [1])
    cache._write_prepared(0, middle, k[:1], k[:1])
    assert torch.isfinite(cache._attend_prepared(0, batch, q)).all()
    with pytest.raises(RuntimeError, match="uninitialized KV"):
        cache._attend_prepared(1, batch, q)
    cache.truncate_suffix("0", 0)
    with pytest.raises(RuntimeError, match="uninitialized KV"):
        cache._attend_prepared(0, batch, q)
    cache.free("0")
    assert cache.allocate("0", 3)
    before = cache.key_cache.clone()
    with pytest.raises(RuntimeError, match="stale prepared KV"):
        cache._write_prepared(1, batch, k, k)
    torch.testing.assert_close(cache.key_cache, before, rtol=0, atol=0, equal_nan=True)
    for rid in ("0", "1", "2"):
        cache.free(rid)


@pytest.mark.npu
@torch.inference_mode()
def test_layout_eviction_and_cross_stream_use_keep_returned_outputs():
    cache, _ = _cache()
    kernel = cache.attention.kernel
    q = torch.randn(3, 4, 64, device=cache.device, dtype=cache.dtype)
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    producer.wait_stream(torch.npu.current_stream())
    batch = cache._prepare_batch(["0", "1", "2"], [0] * 3, [2, 6, 16], for_write=False)
    # The caller owns readiness of KV and descriptor staging; cached layouts
    # are private to each stream and are never overwritten for a later batch.
    torch.npu.synchronize()
    with torch.npu.stream(producer):
        held = cache._attend_prepared(0, batch, q)
        snapshot = held.clone()
        done = torch.npu.Event()
        done.record()
    consumer.wait_event(done)
    with torch.npu.stream(consumer):
        second = cache._attend_prepared(0, batch, q)
        for position in range(2, 14):
            newer = cache._prepare_batch(
                ["0", "1", "2"], [0] * 3, [2, 6, position], for_write=False
            )
            cache._attend_prepared(0, newer, q)
    torch.npu.synchronize()
    torch.testing.assert_close(second, snapshot, rtol=0, atol=0)
    torch.testing.assert_close(held, snapshot, rtol=0, atol=0)
    assert len(kernel._layouts) <= 8
    assert kernel._layout_bytes <= kernel._layout_limit
    for rid in ("0", "1", "2"):
        cache.free(rid)


@pytest.mark.parametrize("kind", ["ragged_decode", "irregular"])
@torch.inference_mode()
def test_failed_plan_preparation_invalidates_the_previous_descriptor(kind, monkeypatch):
    from copy import copy

    from tests.test_ascend_eager import _batched_fixture

    q, keys, values, first = _batched_fixture(kind)
    kernel = AscendAttention()
    expected = kernel.attend_metadata(q, keys, values, first).clone()
    second = copy(first)
    target = "_prepare_decode" if kind == "ragged_decode" else "_plan"
    original = getattr(kernel, target)

    def failed(*args, **kwargs):
        raise RuntimeError("injected plan preparation failure")

    monkeypatch.setattr(kernel, target, failed)
    with pytest.raises(RuntimeError, match="injected plan preparation"):
        kernel.attend_metadata(q, keys, values, second)
    monkeypatch.setattr(kernel, target, original)
    actual = kernel.attend_metadata(q, keys, values, first)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
