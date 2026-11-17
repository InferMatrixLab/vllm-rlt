"""Stable KV storage keeps the tensor addresses when the physical capacity increases."""

from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.core.kv_storage import PoolSpec, StableKVStorage, vmm_unavailable_reason

# The granule is not a multiple of either block size. The two pools round up differently.
SPECS = (PoolSpec((3, 16, 4, 64), torch.bfloat16), PoolSpec((3, 16, 4, 64), torch.float32))
CUDA = torch.device("cuda")


def test_cpu_reports_why_vmm_is_unavailable():
    assert "cpu" in vmm_unavailable_reason(torch.device("cpu"))


@pytest.mark.gpu
def test_commit_backs_more_blocks_without_moving_or_clearing_them():
    assert vmm_unavailable_reason(CUDA) is None
    storage = StableKVStorage(SPECS, max_blocks=4096, initial_blocks=1, device=CUDA)
    first = storage.committed_blocks
    assert 1 <= first < 4096
    layout = [(t.device, t.data_ptr(), t.shape, t.stride()) for t in storage.tensors]
    written = [torch.randn(t[:first].shape, device=CUDA).to(t.dtype) for t in storage.tensors]
    for tensor, values in zip(storage.tensors, written):
        tensor[:first] = values

    assert storage.commit(3000) >= 3000
    last = storage.committed_blocks - 1
    assert [(t.device, t.data_ptr(), t.shape, t.stride()) for t in storage.tensors] == layout
    for tensor, values in zip(storage.tensors, written):
        tensor[last] = 7
        assert torch.equal(tensor[:first], values)
        assert bool((tensor[last] == 7).all())
    assert storage.commit(1) == last + 1


def index_put(keys, values, block, k, v):
    keys[block, 0, 0] = k
    values[block, 0, 0] = v


def triton_graph_write(keys, values, block, k, v):
    from vllm_rlt.kernels.graph_kv import write_graph_kv

    batch = SimpleNamespace(
        write_blocks=block,
        write_offsets=torch.zeros_like(block),
        active_rows=None,
        write_mask=None,
    )
    write_graph_kv(keys[:, 0], values[:, 0], batch, k, v)


@pytest.mark.gpu
@pytest.mark.parametrize("write", [index_put, triton_graph_write])
def test_graph_captured_before_commit_writes_newly_committed_blocks(write):
    spec = SPECS[0]
    storage = StableKVStorage([spec, spec], max_blocks=4096, initial_blocks=1, device=CUDA)
    keys, values = storage.tensors
    block = torch.zeros(1, dtype=torch.long, device=CUDA)
    k = torch.ones((1, *spec.block_shape[2:]), dtype=spec.dtype, device=CUDA)
    v = torch.full_like(k, 2)
    write(keys, values, block, k, v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write(keys, values, block, k, v)

    storage.commit(4096)
    block.fill_(4095)
    k.fill_(3)
    v.fill_(4)
    graph.replay()
    torch.cuda.synchronize()
    assert bool((keys[4095, 0, 0] == 3).all())
    assert bool((values[4095, 0, 0] == 4).all())


@pytest.mark.gpu
def test_close_returns_memory_the_torch_allocator_never_counted():
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved()
    free = torch.cuda.mem_get_info()[0]
    storage = StableKVStorage(SPECS, max_blocks=4096, initial_blocks=4096, device=CUDA)
    committed = storage.committed_bytes
    assert committed >= 4096 * sum(spec.block_bytes for spec in SPECS)
    held = torch.cuda.mem_get_info()[0]
    assert free - held >= committed
    assert torch.cuda.memory_reserved() == reserved

    storage.close()
    assert torch.cuda.mem_get_info()[0] - held >= committed


@pytest.mark.gpu
def test_views_keep_memory_mapped_after_the_storage_is_dropped():
    storage = StableKVStorage(SPECS[:1], max_blocks=4096, initial_blocks=4096, device=CUDA)
    committed = storage.committed_bytes
    view = storage.tensors[0][-1]
    del storage
    view.fill_(5)
    assert bool((view == 5).all())
    held = torch.cuda.mem_get_info()[0]

    del view
    assert torch.cuda.mem_get_info()[0] - held >= committed
