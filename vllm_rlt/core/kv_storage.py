"""KV pools that keep a fixed device address when their physical memory increases.

A captured CUDA Graph keeps the cache pointers and strides. A bounds-checked
index operation in a graph also keeps the size of the first dimension. Because
of this, each pool starts with its maximum block count. CUDA virtual memory maps
physical pages into this address range when you commit more capacity. Only the
first ``committed_blocks`` blocks have physical memory. Do not access the blocks
after them.
"""

import ctypes
import math
import sys
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache

import torch

# These values are CUDA driver enums from cuda.h.
_SUCCESS = 0
_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
_ALLOCATION_TYPE_PINNED = 1
_LOCATION_TYPE_DEVICE = 1
_GRANULARITY_MINIMUM = 0
_ACCESS_READWRITE = 3

_Address = ctypes.c_ulonglong
_Handle = ctypes.c_ulonglong


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocationFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


class _Driver:
    def __init__(self) -> None:
        lib = ctypes.CDLL("libcuda.so.1")
        size, pointer = ctypes.c_size_t, ctypes.POINTER
        for name, argtypes in {
            "cuInit": [ctypes.c_uint],
            "cuGetErrorString": [ctypes.c_int, pointer(ctypes.c_char_p)],
            "cuDeviceGetAttribute": [pointer(ctypes.c_int), ctypes.c_int, ctypes.c_int],
            "cuDevicePrimaryCtxRetain": [pointer(ctypes.c_void_p), ctypes.c_int],
            "cuCtxPushCurrent_v2": [ctypes.c_void_p],
            "cuCtxPopCurrent_v2": [pointer(ctypes.c_void_p)],
            "cuMemGetAllocationGranularity": [
                pointer(size),
                pointer(_AllocationProp),
                ctypes.c_int,
            ],
            "cuMemAddressReserve": [pointer(_Address), size, size, _Address, ctypes.c_ulonglong],
            "cuMemAddressFree": [_Address, size],
            "cuMemCreate": [pointer(_Handle), size, pointer(_AllocationProp), ctypes.c_ulonglong],
            "cuMemRelease": [_Handle],
            "cuMemMap": [_Address, size, size, _Handle, ctypes.c_ulonglong],
            "cuMemSetAccess": [_Address, size, pointer(_AccessDesc), size],
            "cuMemUnmap": [_Address, size],
        }.items():
            function = getattr(lib, name)
            function.argtypes = argtypes
            function.restype = ctypes.c_int
        self._lib = lib
        self._contexts: dict[int, ctypes.c_void_p] = {}
        self.check(lib.cuInit(0))

    def check(self, result: int) -> None:
        if result != _SUCCESS:
            message = ctypes.c_char_p()
            self._lib.cuGetErrorString(result, ctypes.byref(message))
            raise RuntimeError(f"CUDA driver error {result}: {(message.value or b'').decode()}")

    def supports_vmm(self, device: int) -> bool:
        flag = ctypes.c_int()
        self.check(
            self._lib.cuDeviceGetAttribute(
                ctypes.byref(flag), _VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device
            )
        )
        return bool(flag.value)

    @contextmanager
    def context(self, device: int) -> Iterator[None]:
        # A VMM call uses the current context of its thread. We push the primary
        # context of torch and pop it after the calls, so the thread context does
        # not change.
        if device not in self._contexts:
            context = ctypes.c_void_p()
            self.check(self._lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device))
            self._contexts[device] = context
        self.check(self._lib.cuCtxPushCurrent_v2(self._contexts[device]))
        try:
            yield
        finally:
            self.check(self._lib.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p())))

    @staticmethod
    def device_memory(device: int) -> _AllocationProp:
        prop = _AllocationProp()
        prop.type = _ALLOCATION_TYPE_PINNED
        prop.location.type = _LOCATION_TYPE_DEVICE
        prop.location.id = device
        return prop

    def granularity(self, device: int) -> int:
        granularity = ctypes.c_size_t()
        self.check(
            self._lib.cuMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(self.device_memory(device)),
                _GRANULARITY_MINIMUM,
            )
        )
        return granularity.value

    def reserve(self, size: int) -> int:
        address = _Address()
        self.check(self._lib.cuMemAddressReserve(ctypes.byref(address), size, 0, 0, 0))
        return address.value

    def free_address(self, address: int, size: int) -> None:
        self.check(self._lib.cuMemAddressFree(address, size))

    def create(self, size: int, device: int) -> int:
        handle = _Handle()
        self.check(
            self._lib.cuMemCreate(
                ctypes.byref(handle), size, ctypes.byref(self.device_memory(device)), 0
            )
        )
        return handle.value

    def release(self, handle: int) -> None:
        self.check(self._lib.cuMemRelease(handle))

    def map(self, address: int, size: int, handle: int) -> None:
        self.check(self._lib.cuMemMap(address, size, 0, handle, 0))

    def set_access(self, address: int, size: int, device: int) -> None:
        access = _AccessDesc()
        access.location.type = _LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = _ACCESS_READWRITE
        self.check(self._lib.cuMemSetAccess(address, size, ctypes.byref(access), 1))

    def unmap(self, address: int, size: int) -> None:
        self.check(self._lib.cuMemUnmap(address, size))


@cache
def cuda_driver() -> _Driver:
    return _Driver()


def device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def vmm_unavailable_reason(device: torch.device) -> str | None:
    """Return why ``StableKVStorage`` cannot operate on ``device``, or None if it can."""
    if device.type != "cuda":
        return f"{device.type} devices have no CUDA virtual memory"
    if torch.version.hip is not None:
        return "ROCm builds do not have the CUDA driver virtual memory API"
    if sys.platform != "linux":
        return f"the CUDA driver binding operates only on Linux, not on {sys.platform}"
    index = device_index(device)
    if not cuda_driver().supports_vmm(index):
        return f"CUDA device {index} does not support virtual memory management"
    return None


@dataclass(frozen=True)
class PoolSpec:
    """The shape and dtype of one block in one pool, for example the keys of all layers."""

    block_shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def block_bytes(self) -> int:
        return math.prod(self.block_shape) * self.dtype.itemsize


def free_pool(
    driver: _Driver,
    device: int,
    base: int,
    reserved_bytes: int,
    chunks: list[tuple[int, int, int]],
) -> None:
    # cuMemUnmap does not always wait for the kernels that use the pages.
    torch.cuda.synchronize(device)
    with driver.context(device):
        for address, size, handle in chunks:
            driver.unmap(address, size)
            driver.release(handle)
        driver.free_address(base, reserved_bytes)


class _Pool:
    def __init__(self, driver: _Driver, device: int, block_bytes: int, reserved_bytes: int) -> None:
        self.block_bytes = block_bytes
        self.reserved_bytes = reserved_bytes
        self.mapped_bytes = 0
        self.base = driver.reserve(reserved_bytes)
        self._driver = driver
        self._device = device
        self._chunks: list[tuple[int, int, int]] = []
        self.free = weakref.finalize(
            self, free_pool, driver, device, self.base, reserved_bytes, self._chunks
        )
        # At process exit, the driver removes all mappings. A driver call during
        # interpreter shutdown can cause an error, because CUDA can stop first.
        self.free.atexit = False

    @property
    def __cuda_array_interface__(self) -> dict[str, object]:
        return {
            "shape": (self.reserved_bytes,),
            "typestr": "|u1",
            "data": (self.base, False),
            "version": 3,
        }

    def grow(self, mapped_bytes: int) -> None:
        if mapped_bytes <= self.mapped_bytes:
            return
        address, size = self.base + self.mapped_bytes, mapped_bytes - self.mapped_bytes
        handle = self._driver.create(size, self._device)
        try:
            # cuMemMap also allocates page tables. It can stop with an
            # out-of-memory error after cuMemCreate is successful.
            self._driver.map(address, size, handle)
        except RuntimeError:
            self._driver.release(handle)
            raise
        self._chunks.append((address, size, handle))
        self._driver.set_access(address, size, self._device)
        self.mapped_bytes = mapped_bytes


class StableKVStorage:
    """Each pool has one ``[max_blocks, *block_shape]`` tensor at a fixed address.

    ``commit`` adds physical memory to more blocks at the start of each pool. It
    does not change an address, a stride or a stored value. Because of this, a
    graph that you captured before a commit also operates correctly on new blocks.

    The PyTorch allocator does not count this physical memory. ``close`` releases
    it. The memory is also released when no reference to the storage or to its
    tensor views remains. Both release paths wait for the device. After the
    release, do not replay a graph that uses these tensors.
    """

    def __init__(
        self,
        specs: Sequence[PoolSpec],
        *,
        max_blocks: int,
        initial_blocks: int,
        device: torch.device,
    ) -> None:
        self.max_blocks = max_blocks
        self._device = device_index(device)
        driver = cuda_driver()
        with driver.context(self._device):
            self.granularity = driver.granularity(self._device)
            self._pools = [
                _Pool(
                    driver,
                    self._device,
                    spec.block_bytes,
                    round_up(max_blocks * spec.block_bytes, self.granularity),
                )
                for spec in specs
            ]
        # PyTorch does not accept an interface pointer that has no physical memory.
        self.commit(initial_blocks)
        self.tensors = tuple(
            torch.as_tensor(pool)
            .view(spec.dtype)[: max_blocks * math.prod(spec.block_shape)]
            .view(max_blocks, *spec.block_shape)
            for pool, spec in zip(self._pools, specs)
        )

    @property
    def committed_bytes(self) -> int:
        return sum(pool.mapped_bytes for pool in self._pools)

    def commit(self, blocks: int) -> int:
        """Add physical memory to at least the first ``blocks`` blocks of each pool.

        Return the number of blocks that have physical memory. The capacity can
        only increase. ``blocks`` must not be more than ``max_blocks``.
        """
        driver = cuda_driver()
        with driver.context(self._device):
            for pool in self._pools:
                pool.grow(round_up(blocks * pool.block_bytes, self.granularity))
        # The granule round-up can add physical memory for more blocks than the request.
        self.committed_blocks = min(
            self.max_blocks, *(pool.mapped_bytes // pool.block_bytes for pool in self._pools)
        )
        return self.committed_blocks

    def close(self) -> None:
        """Release the physical memory immediately, also when tensor views remain."""
        for pool in self._pools:
            pool.free()
