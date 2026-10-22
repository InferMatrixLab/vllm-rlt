"""Numerical and lifetime contracts of the shared tensor movement layer."""

import argparse

import pytest
import torch

from vllm_rlt import ExecutionConfig
from vllm_rlt.entrypoints.runtime_args import add_runtime_args
from vllm_rlt.worker.tensors import (
    StagedWriteTensor,
    TensorTransfer,
    UvaBackedTensor,
    UvaBuffer,
    copy_tensor,
    pinned_tensor,
    use_uva_on,
)

TRANSFER_MODES = [
    pytest.param("cpu", False, id="cpu"),
    pytest.param("cuda", False, id="cuda-copy", marks=pytest.mark.gpu),
    pytest.param("cuda", True, id="cuda-uva", marks=pytest.mark.gpu),
]
DTYPES = [torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32]


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_backed_tensor_publishes_exact_values_and_keeps_device_address(device, use_uva, dtype):
    backed = UvaBackedTensor(300, dtype, device)
    values = torch.arange(300, dtype=torch.int64)
    if dtype == torch.int64:
        values += 2**54  # Any accidental float conversion loses low token/address bits.
    values = values.to(dtype)
    backed.cpu.copy_(values)
    target = torch.empty_like(values, device=device)
    address = target.data_ptr()
    snapshot = backed.publish()
    assert snapshot.device.type == "cpu"
    assert snapshot.data_ptr() != backed.cpu.data_ptr()
    actual = copy_tensor(snapshot, target, use_uva=use_uva)
    assert torch.equal(actual.cpu(), values)
    backed.cpu[:3].fill_(7)
    actual = copy_tensor(backed.publish(3), target[:3], use_uva=use_uva)
    assert target.data_ptr() == address
    assert torch.equal(actual.cpu(), torch.full((3,), 7, dtype=dtype))
    assert torch.equal(target[3:].cpu(), values[3:])


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
def test_direct_copy_casts_both_outputs_from_source(device, use_uva):
    source = pinned_tensor(5, torch.float32, device)
    source.copy_(torch.tensor([1.5, -2.25, 3.75, 0, 4.5]))
    target = torch.empty(5, dtype=torch.int32, device=device)
    secondary = torch.empty(5, dtype=torch.float32, device=device)
    copy_tensor(source, target, secondary=secondary, use_uva=use_uva)
    assert torch.equal(target.cpu(), source.to(torch.int32))
    assert torch.equal(secondary.cpu(), source)


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_segment_transfer_supports_strides_and_untouched_output(device, use_uva, dtype):
    source = torch.arange(7 * 520, device=device).reshape(7, 520).to(dtype)
    target = torch.full((9, 524), -1, device=device, dtype=dtype)
    transfers = TensorTransfer(2, dtype, device, use_uva=use_uva)
    transfers.append(source[::2, 1::2], target[1:5, 2:522:2])
    transfers.append(source[1, :3], target[7, :3])
    transfers.submit()
    expected = torch.full((9, 524), -1, dtype=dtype)
    expected[1:5, 2:522:2] = source[::2, 1::2].cpu()
    expected[7, :3] = source[1, :3].cpu()
    assert torch.equal(target.cpu(), expected)


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
def test_mixed_host_device_token_sources_preserve_int64(device, use_uva):
    host = pinned_tensor(5, torch.int64, device)
    host.copy_(torch.arange(5) + 2**54)
    resident = torch.tensor([2**54 + 9], device=device, dtype=torch.int64)
    target = torch.full((6,), -1, device=device, dtype=torch.int64)
    transfers = TensorTransfer(2, torch.int64, device, use_uva=use_uva)
    transfers.append(host[1:4], target[:3])
    transfers.append(resident, target[4:5])
    transfers.submit()
    assert target.cpu().tolist() == [2**54 + 1, 2**54 + 2, 2**54 + 3, -1, 2**54 + 9, -1]


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
def test_staged_write_snapshots_only_dirty_ranges(device, use_uva):
    target = torch.full((2, 3, 8), -1, dtype=torch.int32, device=device)
    writes = StagedWriteTensor(target, 3, 8, use_uva=use_uva)
    values = [11, 12]
    writes.stage_write((1, 2), 3, values)
    values[:] = [99, 99]  # Mutating the caller's state must not change submitted values.
    writes.stage_write((0, 0), 7, [42])
    writes.submit()
    expected = torch.full((2, 3, 8), -1, dtype=torch.int32)
    expected[1, 2, 3:5] = torch.tensor([11, 12])
    expected[0, 0, 7] = 42
    assert torch.equal(target.cpu(), expected)
    address = writes.contents.buffers[0].cpu.data_ptr()
    writes.reset()  # .cpu() above retired the GPU consumer.
    writes.stage_write((0, 1), 0, [6])
    writes.submit()
    expected[0, 1, 0] = 6
    assert torch.equal(target.cpu(), expected)
    assert writes.contents.buffers[0].cpu.data_ptr() == address


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
def test_uva_buffer_stage_and_zero_length_transfer(device, use_uva):
    buffer = UvaBuffer(8, torch.int64, device, use_uva=use_uva)
    buffer.np[:] = range(8)
    result = buffer.stage(3)
    assert result.cpu().tolist() == [0, 1, 2]
    if buffer.use_uva or device == "cpu":
        assert result.data_ptr() == buffer.cpu.data_ptr()
    else:
        assert result.is_cuda and result.data_ptr() != buffer.cpu.data_ptr()
    transfers = TensorTransfer(1, torch.int64, device, use_uva=use_uva)
    transfers.append(
        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64, device=device)
    )
    transfers.submit()
    assert not transfers.copies


def test_invalid_copy_range_and_dtype_fail_before_submission():
    target = torch.zeros((2, 4), dtype=torch.int32)
    writes = StagedWriteTensor(target, 1, 4)
    with pytest.raises(ValueError, match="outside"):
        writes.stage_write(0, 3, [1, 2])
    assert not writes.transfers.copies
    with pytest.raises(ValueError, match="dtype"):
        writes.transfers.append(torch.ones(2, dtype=torch.int64), target[0, :2])


@pytest.mark.parametrize("use_uva", [False, True])
@pytest.mark.gpu
def test_rotating_snapshots_allow_cpu_updates_before_gpu_consumes(use_uva):
    from vllm_rlt.worker.submission_queue import SubmissionQueue

    device = torch.device("cuda", torch.cuda.current_device())
    queue = SubmissionQueue(max_pending=2)
    tensor = UvaBackedTensor(257, torch.int64, device, max_concurrency=2)
    target = torch.empty(257, dtype=torch.int64, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    results = []
    # Warm the copy signature so compilation cannot hide the delayed read.
    with torch.cuda.stream(stream):
        tensor.cpu.zero_()
        copy_tensor(tensor.publish(), target, use_uva=use_uva)
    stream.synchronize()
    for value in range(5):
        queue.begin()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            tensor.cpu.fill_(2**54 + value)
            copy_tensor(tensor.publish(), target, use_uva=use_uva)
            # The CPU master is independent of the published pinned snapshot.
            tensor.cpu.fill_(-1)
            results.append(target.clone())
            event = torch.cuda.Event()
            event.record(stream)
            queue.complete(event)
    stream.synchronize()
    for value, result in enumerate(results):
        assert torch.equal(result.cpu(), torch.full((257,), 2**54 + value, dtype=torch.int64))


@pytest.mark.parametrize("use_uva", [False, True])
def test_uva_pool_rotates_independent_storage_without_event_interface(use_uva):
    from vllm_rlt.worker.tensors import UvaBufferPool

    pool = UvaBufferPool(4, torch.int64, "cpu", use_uva=use_uva, max_concurrency=2)
    assert not any(buffer.use_uva for buffer in pool.buffers)
    first = pool.stage(torch.tensor([1, 2]))
    second = pool.stage(torch.tensor([3, 4]))
    assert first.data_ptr() != second.data_ptr()
    assert first.tolist() == [1, 2]
    assert second.tolist() == [3, 4]
    assert pool.stage(torch.tensor([5, 6])).data_ptr() == first.data_ptr()
    assert not any(hasattr(buffer, "event") for buffer in pool.buffers)


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
def test_pool_grows_only_selected_slot_and_reuses_capacity(device, use_uva):
    from vllm_rlt.worker.tensors import UvaBufferPool

    pool = UvaBufferPool((2, 3), torch.int64, device, use_uva=use_uva)
    original = list(pool.buffers)
    values = torch.arange(15, dtype=torch.int64).reshape(5, 3) + 2**54
    first = pool.stage(values)
    assert torch.equal(first.cpu(), values)
    assert pool.buffers[0].cpu.shape == (8, 3)
    assert pool.buffers[1] is original[1]
    first_buffer = pool.buffers[0]
    index = pool.index
    with pytest.raises(ValueError, match="trailing"):
        pool.stage(torch.zeros((4, 2), dtype=torch.int64))
    assert pool.index == index
    second = pool.stage(values[:1])
    assert torch.equal(second.cpu(), values[:1])
    assert pool.buffers[1] is original[1]
    reused = pool.stage(values[:3])
    assert torch.equal(reused.cpu(), values[:3])
    assert pool.buffers[0] is first_buffer
    assert reused.data_ptr() == first.data_ptr()
    empty = pool.stage(values[:0])
    assert empty.shape == (0, 3)
    assert pool.buffers[1] is original[1]


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_staged_writes_grow_contents_and_descriptors_without_losing_updates(device, use_uva, dtype):
    target = torch.full((3, 8), -1, dtype=dtype, device=device)
    writes = StagedWriteTensor(target, 1, 1, use_uva=use_uva)
    values = torch.tensor([11, 12, 13, 14, 15], dtype=dtype)
    writes.stage_write(0, 1, values)
    writes.stage_write(1, 7, [21])
    writes.stage_write(2, 2, [31, 32, 33])
    values.fill_(99)
    writes.submit()
    expected = torch.full((3, 8), -1, dtype=dtype)
    expected[0, 1:6] = torch.tensor([11, 12, 13, 14, 15], dtype=dtype)
    expected[1, 7] = 21
    expected[2, 2:5] = torch.tensor([31, 32, 33], dtype=dtype)
    assert torch.equal(target.cpu(), expected)
    contents = writes.contents.buffers[0]
    assert contents.cpu.numel() == 16
    if device == "cuda" and use_uva:
        assert writes.transfers.descriptors.cpu.shape == (4, 8)
    descriptor = writes.transfers.descriptors
    writes.reset()
    writes.stage_write(0, 0, [42])
    writes.submit()
    expected[0, 0] = 42
    assert torch.equal(target.cpu(), expected)
    assert writes.contents.buffers[0] is contents
    assert writes.transfers.descriptors is descriptor


@pytest.mark.gpu
@pytest.mark.parametrize("use_uva", [False, True])
def test_pool_growth_preserves_inflight_reads_on_other_slots(use_uva):
    from vllm_rlt.worker.submission_queue import SubmissionQueue
    from vllm_rlt.worker.tensors import UvaBufferPool

    device = torch.device("cuda", torch.cuda.current_device())
    pool = UvaBufferPool(1, torch.int64, device, use_uva=use_uva)
    queue = SubmissionQueue(max_pending=2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream(device))
    outputs = []
    with torch.cuda.stream(stream):
        # Compile before delaying consumers, so compilation cannot retire them.
        warm = pinned_tensor(1, torch.int64, device)
        warm.zero_()
        copy_tensor(warm, torch.empty(1, dtype=torch.int64, device=device), use_uva=use_uva)
    stream.synchronize()
    for index, count in enumerate((2, 7, 9, 3, 17)):
        queue.begin()
        values = torch.arange(count, dtype=torch.int64) + 2**54 + index * 100
        output = torch.empty(count, dtype=torch.int64, device=device)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            view = pool.stage(values)
            values.fill_(-1)
            copy_tensor(view, output, use_uva=use_uva)
            event = torch.cuda.Event()
            event.record(stream)
        queue.complete(event)
        outputs.append(output)
    stream.synchronize()
    for index, output in enumerate(outputs):
        expected = torch.arange(output.numel(), dtype=torch.int64) + 2**54 + index * 100
        assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
def test_transfer_descriptor_growth_for_device_sources(device, use_uva):
    source = torch.arange(9, dtype=torch.int64, device=device).reshape(3, 3) + 2**54
    target = torch.full((3, 5), -1, dtype=torch.int64, device=device)
    transfers = TensorTransfer(1, torch.int64, device, use_uva=use_uva)
    for row in range(3):
        transfers.append(source[row], target[row, 1:4])
    transfers.submit()
    expected = torch.full((3, 5), -1, dtype=torch.int64)
    expected[:, 1:4] = source.cpu()
    assert torch.equal(target.cpu(), expected)
    if device == "cuda":
        assert transfers.descriptors.cpu.shape == (4, 8)


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
@pytest.mark.parametrize("preallocated", [False, True])
def test_pool_whole_copy_grows_and_preserves_destination_ownership(device, use_uva, preallocated):
    from vllm_rlt.worker.tensors import UvaBufferPool

    pool = UvaBufferPool(1, torch.int64, device, use_uva=use_uva, max_concurrency=1)
    source = torch.arange(5, dtype=torch.int64) + 2**54
    expected = source.clone()
    storage = torch.full((7,), -1, dtype=torch.int64, device=device)
    out = storage[1:6] if preallocated else None
    result = pool.copy_to_gpu(source, out=out)
    source.fill_(-1)
    assert torch.equal(result.cpu(), expected)
    assert result.device == pool.buffers[0].device
    if preallocated:
        assert result is out
        assert storage[0].item() == storage[-1].item() == -1
    else:
        assert result.data_ptr() != pool.buffers[0].cpu.data_ptr()
    buffer = pool.buffers[0]
    address = result.data_ptr()
    updated = pool.copy_to_gpu(torch.full((5,), 7, dtype=torch.int64), out=result)
    assert updated.data_ptr() == address
    assert updated.cpu().tolist() == [7] * 5
    assert pool.buffers[0] is buffer
    assert buffer.cpu.numel() == 8


def test_pool_whole_copy_rejects_invalid_output_before_rotation():
    from vllm_rlt.worker.tensors import UvaBufferPool

    pool = UvaBufferPool(1, torch.int64, "cpu")
    with pytest.raises(ValueError, match="output"):
        pool.copy_to_gpu(torch.ones(3, dtype=torch.int64), out=torch.empty(2, dtype=torch.int64))
    assert pool.index == 0
    assert pool.buffers[0].cpu.numel() == 1


@pytest.mark.parametrize("device,use_uva", TRANSFER_MODES)
def test_staged_inputs_preserve_slices_and_inflight_snapshots(device, use_uva):
    from vllm_rlt.worker.submission_queue import SubmissionQueue
    from vllm_rlt.worker.tensors import StagedInputTensor

    storage = StagedInputTensor(torch.int64, torch.device(device), use_uva=use_uva)
    queue = SubmissionQueue()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()] if device == "cuda" else [None, None]
    results, expected, allocations = [], [], []
    from contextlib import nullcontext

    # Appending can grow CPU storage after an earlier metadata slice was made.
    # Two submissions run before slot reuse; their snapshots must remain separate.
    for index, count in enumerate((3, 130, 3, 257, 3, 200)):
        queue.begin()
        storage.reset()
        prefix = storage.append([2**54 + index, 2**54 + index + 1])
        tail_values = list(range(count))
        tail = storage.append(tail_values)
        stream = streams[index % 2]
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            if stream is not None:
                torch.cuda._sleep(1_000_000)
            gpu = storage.upload()
            allocations.append(storage.targets[index % 2])
            prefix_gpu = gpu.as_strided(prefix.shape, prefix.stride(), prefix.storage_offset())
            tail_gpu = gpu.as_strided(tail.shape, tail.stride(), tail.storage_offset())
            results.append((prefix_gpu.clone(), tail_gpu.clone()))
            event = None
            if stream is not None:
                event = torch.cuda.Event()
                event.record(stream)
            queue.complete(event)
        expected.append((prefix.clone(), torch.tensor(tail_values)))
        # This is legal immediately after upload, even while the GPU is delayed.
        storage.cpu.fill_(-1)
    queue.drain()
    for actual, wanted in zip(results, expected):
        for result, value in zip(actual, wanted):
            assert torch.equal(result.cpu(), value)
    assert allocations[0] is allocations[2] is allocations[4]
    assert allocations[1] is not allocations[3]
    assert allocations[3] is allocations[5]


def test_uva_configuration_and_capability(monkeypatch):
    assert ExecutionConfig().use_uva
    with pytest.raises(ValueError, match="use_uva"):
        ExecutionConfig(use_uva=1)
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    assert parser.parse_args([]).use_uva
    args = parser.parse_args(["--no-uva", "--prefill-uva"])
    assert not args.use_uva and args.prefill_uva
    monkeypatch.setattr(torch.version, "hip", None)
    assert use_uva_on("cuda")
    assert not use_uva_on("cuda", False)
    assert not use_uva_on("cpu")
    monkeypatch.setattr(torch.version, "hip", "test")
    assert not use_uva_on("cuda")
