"""Tests for fused Triton unquantized graph KV cache writes across CUDA backends."""

from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.kernels import graph_kv
from vllm_rlt.worker.cuda_graph import _DeviceCache


def _reference_write(keys, values, layer, blocks, offsets, k, v, *, active=None, mask=None):
    expected_k = keys.clone()
    expected_v = values.clone()
    limit = len(k) if active is None else int(active.item())
    for row in range(len(k)):
        if row >= limit:
            continue
        if mask is not None and not bool(mask[row].item()):
            continue
        b = int(blocks[row].item())
        o = int(offsets[row].item())
        expected_k[b, layer, o] = k[row]
        expected_v[b, layer, o] = v[row]
    return expected_k, expected_v


def test_can_use_graph_kv_write_rejects_cpu_and_quantized_and_hip(monkeypatch):
    cpu_keys = torch.zeros((4, 2, 4, 4, 8), dtype=torch.bfloat16)
    cpu_values = torch.zeros_like(cpu_keys)
    assert not graph_kv.can_use_graph_kv_write(
        SimpleNamespace(key_cache=cpu_keys, value_cache=cpu_values, backend="flash_attn_4")
    )

    fake_cuda = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    # Non-Tensor mocks must be rejected cleanly.
    assert not graph_kv.can_use_graph_kv_write(
        SimpleNamespace(key_cache=fake_cuda, value_cache=fake_cuda, backend="flash_attn_4")
    )

    # Quantized FP8 pools (key_scale set or fp8 dtype) must be rejected.
    fp8_keys = torch.zeros((2, 1, 4, 4, 8), dtype=torch.float8_e4m3fn)
    assert not graph_kv.can_use_graph_kv_write(
        SimpleNamespace(
            key_cache=fp8_keys,
            value_cache=fp8_keys,
            key_scale=torch.ones(1),
            backend="flash_attn_4",
        )
    )

    # Disabled Triton guard must return False cleanly.
    monkeypatch.setattr(graph_kv, "_TRITON_AVAILABLE", False)
    assert not graph_kv.can_use_graph_kv_write(
        SimpleNamespace(key_cache=cpu_keys, value_cache=cpu_values, backend="triton")
    )


def test_device_cache_cpu_fallback_writes_bit_exact():
    keys = torch.full((4, 2, 4, 4, 8), -1.0, dtype=torch.bfloat16)
    values = torch.full_like(keys, -2.0)
    proxy = _DeviceCache(
        SimpleNamespace(key_cache=keys, value_cache=values, backend="flash_attn_4")
    )
    blocks = torch.tensor([2, 0, 3], dtype=torch.long)
    offsets = torch.tensor([1, 3, 0], dtype=torch.long)
    batch = SimpleNamespace(
        active_rows=None,
        write_mask=None,
        write_blocks=blocks,
        write_offsets=offsets,
    )
    k = torch.randn((3, 4, 8), dtype=torch.bfloat16)
    v = torch.randn((3, 4, 8), dtype=torch.bfloat16)
    expected_k, expected_v = _reference_write(keys, values, 1, blocks, offsets, k, v)
    proxy._write_prepared(1, batch, k, v)
    assert torch.equal(keys, expected_k)
    assert torch.equal(values, expected_v)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(16, 128), (4, 256), (8, 128), (3, 9)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("contiguous", [True, False])
def test_write_graph_kv_bit_exact_shapes_dtypes_and_layouts(shape, dtype, contiguous):
    heads, head_dim = shape
    num_blocks, num_layers, block_size = 16, 3, 16
    layer = 1
    rows = 7
    torch.manual_seed(42)
    keys = torch.full(
        (num_blocks, num_layers, block_size, heads, head_dim), -3.0, dtype=dtype, device="cuda"
    )
    values = torch.full_like(keys, -5.0)
    # Out-of-order block/offset pairs covering multiple blocks and slots.
    blocks = torch.tensor([11, 2, 15, 0, 7, 4, 9], dtype=torch.long, device="cuda")
    offsets = torch.tensor([14, 0, 7, 15, 3, 11, 2], dtype=torch.long, device="cuda")
    if contiguous:
        k = torch.randn((rows, heads, head_dim), dtype=dtype, device="cuda")
        v = torch.randn((rows, heads, head_dim), dtype=dtype, device="cuda")
    else:
        k = torch.randn((rows, head_dim, heads), dtype=dtype, device="cuda").transpose(1, 2)
        v = torch.randn((rows, head_dim, heads), dtype=dtype, device="cuda").transpose(1, 2)
        assert not k.is_contiguous()
    batch = SimpleNamespace(
        active_rows=None,
        write_mask=None,
        write_blocks=blocks,
        write_offsets=offsets,
    )
    expected_k, expected_v = _reference_write(keys, values, layer, blocks, offsets, k, v)
    graph_kv.write_graph_kv(keys[:, layer], values[:, layer], batch, k, v)
    assert torch.equal(keys, expected_k)
    assert torch.equal(values, expected_v)


@pytest.mark.gpu
def test_write_graph_kv_active_and_write_mask_with_duplicate_inactive_slots():
    # Inactive padded rows frequently point to dummy (block=0, offset=0) that collides
    # with an active row; they must never overwrite the active row or untouched slots.
    dtype = torch.bfloat16
    keys = torch.full((6, 2, 8, 4, 256), -1.0, dtype=dtype, device="cuda")
    values = torch.full_like(keys, -2.0)
    blocks = torch.tensor([0, 3, 5, 0, 0, 3], dtype=torch.long, device="cuda")
    offsets = torch.tensor([0, 4, 7, 0, 0, 4], dtype=torch.long, device="cuda")
    active = torch.tensor(4, dtype=torch.int32, device="cuda")
    mask = torch.tensor([True, False, True, False, True, True], dtype=torch.bool, device="cuda")
    torch.manual_seed(11)
    k = torch.randn((6, 4, 256), dtype=dtype, device="cuda")
    v = torch.randn((6, 4, 256), dtype=dtype, device="cuda")
    batch = SimpleNamespace(
        active_rows=active,
        write_mask=mask,
        write_blocks=blocks,
        write_offsets=offsets,
    )
    expected_k, expected_v = _reference_write(
        keys, values, 0, blocks, offsets, k, v, active=active, mask=mask
    )
    graph_kv.write_graph_kv(keys[:, 0], values[:, 0], batch, k, v)
    assert torch.equal(keys, expected_k)
    assert torch.equal(values, expected_v)


@pytest.mark.gpu
def test_write_graph_kv_empty_rows_noop():
    keys = torch.full((4, 2, 4, 4, 64), -1.0, dtype=torch.bfloat16, device="cuda")
    values = torch.full_like(keys, -2.0)
    k = torch.empty((0, 4, 64), dtype=torch.bfloat16, device="cuda")
    v = torch.empty_like(k)
    batch = SimpleNamespace(
        active_rows=None,
        write_mask=None,
        write_blocks=torch.empty((0,), dtype=torch.long, device="cuda"),
        write_offsets=torch.empty((0,), dtype=torch.long, device="cuda"),
    )
    expected_k, expected_v = keys.clone(), values.clone()
    graph_kv.write_graph_kv(keys[:, 0], values[:, 0], batch, k, v)
    assert torch.equal(keys, expected_k)
    assert torch.equal(values, expected_v)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["flash_attn_4", "flash_attn_3", "flash_attn_2", "triton"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_device_cache_routes_all_cuda_backends_in_cuda_graph_without_index_put(backend, dtype):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoIndexPut(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten.index_put_.default, "unexpected separate indexed KV write"
            return func(*args, **(kwargs or {}))

    keys = torch.full((8, 2, 16, 16, 128), -1.0, dtype=dtype, device="cuda")
    values = torch.full_like(keys, -2.0)
    proxy = _DeviceCache(SimpleNamespace(key_cache=keys, value_cache=values, backend=backend))
    assert graph_kv.can_use_graph_kv_write(proxy.cache)

    blocks = torch.tensor([6, 1, 4, 0], dtype=torch.long, device="cuda")
    offsets = torch.tensor([15, 2, 9, 0], dtype=torch.long, device="cuda")
    batch = SimpleNamespace(
        active_rows=None,
        write_mask=None,
        write_blocks=blocks,
        write_offsets=offsets,
    )
    torch.manual_seed(99)
    k = torch.randn((4, 16, 128), dtype=dtype, device="cuda")
    v = torch.randn((4, 16, 128), dtype=dtype, device="cuda")

    with NoIndexPut():
        proxy._write_prepared(1, batch, k, v)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), NoIndexPut():
        proxy._write_prepared(1, batch, k, v)

    # Mutate inputs in place and replay: must update the new slots bit-exactly.
    keys.fill_(-1.0)
    values.fill_(-2.0)
    blocks.copy_(torch.tensor([2, 7, 3, 5], dtype=torch.long, device="cuda"))
    offsets.copy_(torch.tensor([1, 12, 6, 8], dtype=torch.long, device="cuda"))
    k.add_(3.5)
    v.sub_(2.25)
    expected_k, expected_v = _reference_write(keys, values, 1, blocks, offsets, k, v)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(keys, expected_k)
    assert torch.equal(values, expected_v)


@pytest.mark.gpu
def test_device_cache_guards_fp8_and_triton_disabled(monkeypatch):
    # FP8 cache must NOT satisfy can_use_graph_kv_write.
    fp8_keys = torch.zeros((4, 1, 8, 4, 64), dtype=torch.float8_e4m3fn, device="cuda")
    fp8_values = torch.zeros_like(fp8_keys)
    assert not graph_kv.can_use_graph_kv_write(
        SimpleNamespace(
            key_cache=fp8_keys,
            value_cache=fp8_values,
            key_scale=torch.ones(1, device="cuda"),
            backend="flash_attn_4",
        )
    )
    assert not graph_kv.can_use_graph_kv_write(
        SimpleNamespace(key_cache=fp8_keys, value_cache=fp8_values, backend="flash_attn_4")
    )

    # When Triton is disabled, _DeviceCache falls back to index_put_ bit-exactly.
    monkeypatch.setattr(graph_kv, "_TRITON_AVAILABLE", False)
    keys = torch.full((4, 2, 8, 4, 64), -1.0, dtype=torch.bfloat16, device="cuda")
    values = torch.full_like(keys, -2.0)
    proxy = _DeviceCache(
        SimpleNamespace(key_cache=keys, value_cache=values, backend="flash_attn_4")
    )
    blocks = torch.tensor([3, 1], dtype=torch.long, device="cuda")
    offsets = torch.tensor([5, 2], dtype=torch.long, device="cuda")
    k = torch.randn((2, 4, 64), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((2, 4, 64), dtype=torch.bfloat16, device="cuda")
    batch = SimpleNamespace(
        active_rows=None, write_mask=None, write_blocks=blocks, write_offsets=offsets
    )
    expected_k, expected_v = _reference_write(keys, values, 1, blocks, offsets, k, v)
    proxy._write_prepared(1, batch, k, v)
    assert torch.equal(keys, expected_k)
    assert torch.equal(values, expected_v)


@pytest.mark.gpu
def test_write_graph_kv_large_int64_offset_stress():
    """Verify block * cache_block > 2**31 elements does not overflow int32 addressing."""
    free_bytes, _ = torch.cuda.mem_get_info()
    # Need ~4.1 GiB for a flat BF16 buffer spanning > 2**31 elements (4 GiB) plus headroom.
    if free_bytes < 10 * 1024**3:
        pytest.skip("requires >=10 GiB free VRAM for >2^31-element int64 offset stress test")

    # 2 blocks where block 1 begins at element index 2**31 (exceeds signed int32 max).
    block_stride = 1 << 31
    heads, head_dim = 4, 128
    width = heads * head_dim
    total_elems = block_stride + 2 * width
    k_flat = torch.zeros(total_elems, dtype=torch.bfloat16, device="cuda")
    v_flat = torch.zeros(total_elems, dtype=torch.bfloat16, device="cuda")
    try:
        key_cache = k_flat.as_strided(
            size=(2, 2, heads, head_dim), stride=(block_stride, width, head_dim, 1)
        )
        value_cache = v_flat.as_strided(
            size=(2, 2, heads, head_dim), stride=(block_stride, width, head_dim, 1)
        )
        torch.manual_seed(123)
        k = torch.randn((2, heads, head_dim), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((2, heads, head_dim), dtype=torch.bfloat16, device="cuda")
        blocks = torch.tensor([0, 1], dtype=torch.long, device="cuda")
        offsets = torch.tensor([1, 1], dtype=torch.long, device="cuda")
        batch = SimpleNamespace(
            active_rows=None, write_mask=None, write_blocks=blocks, write_offsets=offsets
        )
        graph_kv.write_graph_kv(key_cache, value_cache, batch, k, v)
        torch.cuda.synchronize()
        assert torch.equal(key_cache[0, 1], k[0])
        assert torch.equal(key_cache[1, 1], k[1])
        assert torch.equal(value_cache[0, 1], v[0])
        assert torch.equal(value_cache[1, 1], v[1])
        assert torch.count_nonzero(key_cache[0, 0]).item() == 0
        assert torch.count_nonzero(key_cache[1, 0]).item() == 0
    finally:
        del k_flat, v_flat
        torch.cuda.empty_cache()
