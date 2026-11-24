"""Stable KV storage keeps the tensor addresses when the physical capacity increases."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.core.kv_storage import (
    PoolSpec,
    StableKVStorage,
    cuda_driver,
    vmm_unavailable_reason,
)

# The granule is not a multiple of either block size. The two pools round up differently.
SPECS = (PoolSpec((3, 16, 4, 64), torch.bfloat16), PoolSpec((3, 16, 4, 64), torch.float32))
CUDA = torch.device("cuda")


def test_cpu_reports_why_vmm_is_unavailable():
    assert "cpu" in vmm_unavailable_reason(torch.device("cpu"))


@pytest.mark.parametrize("initial_blocks", [0, 4097])
def test_initial_blocks_outside_capacity_fail_before_any_reservation(initial_blocks):
    with pytest.raises(ValueError, match="initial_blocks"):
        StableKVStorage(SPECS, max_blocks=4096, initial_blocks=initial_blocks, device=CUDA)


@pytest.mark.gpu
def test_commit_backs_more_blocks_without_moving_or_clearing_them():
    assert vmm_unavailable_reason(CUDA) is None
    storage = StableKVStorage(SPECS, max_blocks=4096, initial_blocks=1, device=CUDA)
    first = storage.committed_blocks
    assert 1 <= first < 4096
    for blocks in (-1, 4097):
        with pytest.raises(ValueError, match="max_blocks"):
            storage.commit(blocks)
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
# A release error in a finalizer is only a warning by default.
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_last_reference_releases_memory_the_torch_allocator_never_counted():
    reserved = torch.cuda.memory_reserved()
    storage = StableKVStorage(SPECS, max_blocks=4096, initial_blocks=4096, device=CUDA)
    assert storage.committed_bytes >= 4096 * sum(spec.block_bytes for spec in SPECS)
    assert torch.cuda.memory_reserved() == reserved
    pools = [weakref.ref(pool) for pool in storage._pools]
    view = storage.tensors[0][-1]

    del storage
    view.fill_(5)
    assert bool((view == 5).all())
    assert [pool() is not None for pool in pools] == [True, False]
    del view
    assert pools[0]() is None


@pytest.mark.gpu
@pytest.mark.parametrize("call", ["map", "set_access"])
def test_failed_commit_returns_the_memory_that_it_mapped(monkeypatch, call):
    storage = StableKVStorage(SPECS, max_blocks=4096, initial_blocks=1, device=CUDA)
    capacity = storage.committed_blocks, storage.committed_bytes
    free = torch.cuda.mem_get_info()[0]
    driver = cuda_driver()
    original = getattr(driver, call)
    calls = []

    def fail_in_second_pool(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("injected driver error")
        return original(*args)

    monkeypatch.setattr(driver, call, fail_in_second_pool)
    with pytest.raises(RuntimeError, match="injected"):
        storage.commit(2048)
    assert (storage.committed_blocks, storage.committed_bytes) == capacity
    assert torch.cuda.mem_get_info()[0] == free

    monkeypatch.undo()
    assert storage.commit(2048) >= 2048
