"""RLT tensor storage and movement; completion belongs to the enclosing event pool.

UvaBuffer: per-submission host metadata with a kernel-readable view.
UvaBackedTensor: mutable CPU state plus rotating host snapshots.
StagedWriteTensor: pending disjoint updates to caller-owned device state.
Readback and immutable prompt storage remain plain pinned tensors.

These objects never synchronize or decide when their storage may be overwritten.
Callers must acquire the owning buffer before writes and retire its final consumer
before reset/reuse. CUDA graph inputs and hot hidden/KV states stay device-resident.
"""

import math
from functools import cache, cached_property

import torch

from vllm_rlt.worker.submission_queue import MAX_CONCURRENT_SUBMISSIONS

_DTYPES = (torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32)


def use_uva_on(device, enabled=True):
    """Mapped host reads are supported by the NVIDIA CUDA path."""
    return enabled and torch.device(device).type == "cuda" and not torch.version.hip


@cache
def _triton_type(dtype):
    import triton.language as tl

    return dict(zip(_DTYPES, (tl.int32, tl.int64, tl.float16, tl.bfloat16, tl.float32)))[dtype]


def copy_tensor(source, target, *, secondary=None, use_uva=True):
    """Copy contiguous values directly to final storage, optionally casting twice.

    Source and outputs must stay alive through the caller's completion event.
    D2H deliberately uses CUDA's async copy path rather than SM writes to host.
    """
    if source.numel() != target.numel() or not source.is_contiguous() or not target.is_contiguous():
        raise ValueError("copy_tensor requires equal-sized contiguous tensors")
    if secondary is not None and (
        secondary.numel() != source.numel()
        or not secondary.is_contiguous()
        or secondary.device != target.device
    ):
        raise ValueError("secondary output must match target size and device")
    if (
        source.dtype not in _DTYPES
        or target.dtype not in _DTYPES
        or (secondary is not None and secondary.dtype not in _DTYPES)
    ):
        raise ValueError("unsupported transfer dtype")
    if source.is_cuda and target.is_cuda and source.device != target.device:
        raise ValueError("cross-device transfers require an explicit connector")
    if source.numel() == 0:
        return target
    direct = (
        target.is_cuda
        and not torch.version.hip
        and (source.is_cuda or (use_uva_on(target.device, use_uva) and source.is_pinned()))
    )
    if direct:
        from vllm_rlt.kernels.tensor_transfer import tensor_transfer_kernel

        tensor_transfer_kernel[(math.ceil(source.numel() / 256),)](
            None,
            source,
            target,
            secondary,
            source.numel(),
            _triton_type(source.dtype),
            _triton_type(target.dtype),
            False,
            secondary is not None,
            1,
            256,
        )
    else:
        target.view(-1).copy_(source.view(-1), non_blocking=True)
        if secondary is not None:
            secondary.view(-1).copy_(
                target.view(-1) if target.dtype == source.dtype else source.view(-1),
                non_blocking=True,
            )
    return target


def pinned_tensor(shape, dtype, device):
    """Allocate host staging; CPU execution needs no pinned allocation."""
    return torch.empty(
        shape, dtype=dtype, device="cpu", pin_memory=torch.device(device).type == "cuda"
    )


class UvaBuffer:
    """An event-owned host snapshot, read directly or through a device fallback."""

    def __init__(self, shape, dtype, device, *, use_uva=True, target=None):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.use_uva = use_uva_on(self.device, use_uva)
        self.cpu = pinned_tensor(shape, dtype, device)
        if target is not None and (
            target.shape != self.cpu.shape or target.dtype != dtype or target.device != self.device
        ):
            raise ValueError("fallback target must match buffer shape, dtype and device")
        self._target = (
            self.cpu
            if self.use_uva or self.device.type == "cpu"
            else target
            if target is not None
            else None
        )

    @cached_property
    def np(self):
        # BF16 has no NumPy view; callers write that dtype through .cpu instead.
        return self.cpu.numpy()

    def stage(self, count=None):
        """Return kernel-readable data, submitting a copy only on the fallback path.

        count selects the first-dimension prefix. UVA aliases the host storage;
        fallback copies into a reusable target on the current stream.
        This does not wait for completion or make the host storage reusable.
        """
        if self._target is None:
            self._target = torch.empty_like(self.cpu, device=self.device)
        host = self.cpu if count is None else self.cpu[:count]
        target = self._target if count is None else self._target[:count]
        if self._target is not self.cpu:
            copy_tensor(host, target, use_uva=False)
        return target


class UvaBufferPool:
    """Reusable snapshots; each slot grows its first dimension when needed.

    The execution owner must retire GPU readers before a slot is reused. One
    stage per submission and a pool as deep as the submission queue satisfy that
    contract. Repeated stages within one submission need their own bound.
    """

    def __init__(
        self,
        shape,
        dtype,
        device,
        *,
        use_uva=True,
        max_concurrency=MAX_CONCURRENT_SUBMISSIONS,
    ):
        if max_concurrency < 1:
            raise ValueError("UVA pool capacity must be positive")
        self.buffers = [
            UvaBuffer(shape, dtype, device, use_uva=use_uva) for _ in range(max_concurrency)
        ]
        self.index = 0

    def acquire(self, count=None):
        buffer = self.buffers[self.index]
        if count is not None and count > buffer.cpu.shape[0]:
            capacity = 1 << (count - 1).bit_length()
            buffer = UvaBuffer(
                (capacity, *buffer.cpu.shape[1:]),
                buffer.cpu.dtype,
                buffer.device,
                use_uva=buffer.use_uva,
            )
            self.buffers[self.index] = buffer
        self.index = (self.index + 1) % len(self.buffers)
        return buffer

    def _copy_to_buffer(self, source):
        if source.device.type != "cpu":
            raise ValueError("UVA snapshot source must be a CPU tensor")
        count = len(source)
        if source.shape[1:] != self.buffers[0].cpu.shape[1:]:
            raise ValueError("UVA snapshot trailing dimensions do not match")
        buffer = self.acquire(count)
        buffer.cpu[:count].copy_(source)
        return buffer

    def stage(self, source):
        buffer = self._copy_to_buffer(source)
        return buffer.stage(len(source))

    def copy_to_gpu(self, source, out=None):
        """Copy a complete input to a new device tensor or a caller-owned target.

        Uses an independent pinned snapshot and an async tensor copy, without an
        intermediate device buffer. The caller retires readers before pool reuse.
        """
        device = self.buffers[0].device
        if out is not None and (out.shape != source.shape or out.device != device):
            raise ValueError("output must match the input shape and configured device")
        buffer = self._copy_to_buffer(source)
        snapshot = buffer.cpu[: len(source)]
        if out is None:
            return snapshot.to(device=device, non_blocking=True, copy=True)
        return out.copy_(snapshot, non_blocking=True)


class UvaBackedTensor:
    """Mutable CPU state published as rotating host snapshots.

    publish() performs only a CPU copy. The caller owns any device destination,
    chooses how to consume the snapshot, and retires readers before pool reuse.
    """

    def __init__(self, shape, dtype, device, *, max_concurrency=MAX_CONCURRENT_SUBMISSIONS):
        self.cpu = torch.empty(shape, dtype=dtype, device="cpu")
        self.pool = UvaBufferPool(shape, dtype, device, max_concurrency=max_concurrency)

    @cached_property
    def np(self):
        return self.cpu.numpy()

    def publish(self, count=None):
        source = self.cpu if count is None else self.cpu[:count]
        snapshot = self.pool.acquire(len(source)).cpu[: len(source)]
        snapshot.copy_(source)
        return snapshot


class StagedInputTensor:
    """Append CPU inputs, then upload one typed snapshot per submission.

    Returned read-only CPU slices describe offsets in this submission's flat layout. Growth
    preserves those offsets and values. After upload, the caller binds the same
    slices on the returned device tensor; no address lookup is needed.

    reset/append/upload are serial CPU preparation operations. Device snapshots
    rotate independently; the submission owner retires readers before wraparound.
    """

    def __init__(self, dtype, device, *, use_uva=True):
        self.dtype, self.device, self.use_uva = dtype, device, use_uva
        self.cpu = torch.empty(0, dtype=dtype)
        self.pool = UvaBufferPool(0, dtype, device)
        self.targets = [None] * MAX_CONCURRENT_SUBMISSIONS
        self.count = 0

    def reset(self):
        self.count = 0

    def append(self, values):
        if isinstance(values, torch.Tensor) and values.device.type != "cpu":
            raise ValueError("staged input values must be CPU-owned")
        start, end = self.count, self.count + len(values)
        if end > self.cpu.numel():
            capacity = 1 << (end - 1).bit_length()
            storage = torch.empty(capacity, dtype=self.dtype)
            storage[:start].copy_(self.cpu[:start])
            self.cpu = storage
        target = self.cpu[start:end]
        if isinstance(values, torch.Tensor):
            target.copy_(values)
        elif self.dtype == torch.bfloat16:
            target.copy_(torch.tensor(values, dtype=self.dtype))
        else:
            target.numpy()[:] = values
        self.count = end
        return target

    def upload(self):
        slot = self.pool.index
        buffer = self.pool.acquire(self.count)
        snapshot = buffer.cpu[: self.count]
        snapshot.copy_(self.cpu[: self.count])
        target = self.targets[slot]
        if target is None or target.numel() < buffer.cpu.numel():
            target = torch.empty_like(buffer.cpu, device=self.device)
            self.targets[slot] = target
        copy_tensor(snapshot, target[: self.count], use_uva=self.use_uva)
        return target[: self.count]


class TensorTransfer:
    """One typed batch of disjoint 1D/2D copies, with borrowed tensor references.

    Descriptors hold source/destination addresses, shape and strides in elements.
    They belong to the same event-owned slot as the operation using these tensors.
    reset() is only legal after that slot's final consumer completes.
    """

    def __init__(self, capacity, dtype, device, *, target_dtype=None, use_uva=True):
        self.dtype, self.target_dtype = dtype, target_dtype or dtype
        if self.dtype not in _DTYPES or self.target_dtype not in _DTYPES:
            raise ValueError("unsupported transfer dtype")
        # The owning buffer set already rotates and retires this transfer batch.
        self.descriptor_pool = UvaBufferPool(
            (capacity, 8), torch.int64, device, use_uva=use_uva, max_concurrency=1
        )
        self.descriptors = self.descriptor_pool.buffers[0]
        self.reset()

    def reset(self):
        self.copies = []
        self.max_rows = self.max_cols = 0

    def append(self, source, target):
        if source.shape != target.shape or source.ndim not in (1, 2):
            raise ValueError("transfer requires matching 1D/2D shapes")
        if source.dtype != self.dtype or target.dtype != self.target_dtype:
            raise ValueError("transfer dtype does not match its batch")
        if target.device != self.descriptors.device:
            raise ValueError("transfer destination must be on the configured device")
        if source.is_cuda and source.device != self.descriptors.device:
            raise ValueError("transfer source must be on the configured device")
        if source.numel() == 0:
            return
        src = source.unsqueeze(0) if source.ndim == 1 else source
        self.copies.append((source, target))
        self.max_rows = max(self.max_rows, src.shape[0])
        self.max_cols = max(self.max_cols, src.shape[1])

    def submit(self):
        if not self.copies:
            return
        device = self.descriptors.device
        # Host sources require mapped access; GPU-only batches can use copied
        # descriptors on CUDA/HIP. CPU and disabled-UVA host inputs use async copy.
        if device.type == "cpu" or any(
            not src.is_cuda and (not self.descriptors.use_uva or not src.is_pinned())
            for src, _ in self.copies
        ):
            for source, target in self.copies:
                target.copy_(source, non_blocking=True)
            return
        from vllm_rlt.kernels.tensor_transfer import tensor_transfer_kernel

        self.descriptors = self.descriptor_pool.acquire(len(self.copies))
        for index, (source, target) in enumerate(self.copies):
            src = source.unsqueeze(0) if source.ndim == 1 else source
            dst = target.unsqueeze(0) if target.ndim == 1 else target
            self.descriptors.np[index] = (
                src.data_ptr(),
                dst.data_ptr(),
                *src.shape,
                *src.stride(),
                *dst.stride(),
            )
        for source, target in self.copies:
            if source.is_cuda:
                source.record_stream(torch.cuda.current_stream(device))
            target.record_stream(torch.cuda.current_stream(device))
        tensor_transfer_kernel[
            (len(self.copies), math.ceil(self.max_rows / 4), math.ceil(self.max_cols / 256))
        ](
            self.descriptors.stage(len(self.copies)),
            None,
            None,
            None,
            0,
            _triton_type(self.dtype),
            _triton_type(self.target_dtype),
            True,
            False,
            4,
            256,
        )


class StagedWriteTensor:
    """Collect dirty values, then snapshot and transfer them at submission.

    The caller owns the destination. One growable contents slot is sufficient:
    the owning buffer set already rotates and retires readers before reset/reuse.
    """

    def __init__(self, tensor, initial_writes, initial_values, *, use_uva=True):
        self.gpu = tensor
        self.contents = UvaBufferPool(
            initial_values, tensor.dtype, tensor.device, max_concurrency=1
        )
        self.transfers = TensorTransfer(
            initial_writes, tensor.dtype, tensor.device, use_uva=use_uva
        )
        self.reset()

    def stage_write(self, index, start, values):
        count = len(values)
        if not count:
            return
        target_row = self.gpu[index]
        if target_row.ndim != 1 or start < 0 or start + count > target_row.numel():
            raise ValueError("staged write range is outside the target row")
        if isinstance(values, torch.Tensor):
            if values.device.type != "cpu":
                raise ValueError("staged values must be CPU-owned")
            values = values.tolist()
        self.values.extend(values)
        self.targets.append(target_row[start : start + count])

    def submit(self):
        if not self.values:
            return
        buffer = self.contents.acquire(len(self.values))
        snapshot = buffer.cpu[: len(self.values)]
        if snapshot.dtype == torch.bfloat16:
            snapshot.copy_(torch.tensor(self.values, dtype=snapshot.dtype))
        else:
            buffer.np[: len(self.values)] = self.values
        self.transfers.reset()
        offset = 0
        for target in self.targets:
            count = target.numel()
            self.transfers.append(snapshot[offset : offset + count], target)
            offset += count
        self.transfers.submit()

    def reset(self):
        self.values = []
        self.targets = []
        self.transfers.reset()
