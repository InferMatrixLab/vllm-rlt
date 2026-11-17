"""CUDA VMM block pool with a stable virtual address and growable physical backing.

RFC #192 stage 2 (incremental KV storage commitment). One pool owns the whole
reservation plus every physical mapping; the K and V tensors are views into it,
so there is exactly one lifecycle (reserve -> commit -> close) regardless of how
many tensor planes are exposed.

The tensors always describe the full reservation. Only the committed prefix has
physical backing, so kernels may only address blocks below ``committed_blocks``.
Touching an unmapped block raises a sticky CUDA illegal-address error; the pool
therefore never publishes a block it has not mapped.
"""

from __future__ import annotations

import ctypes

import torch

_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1

_TYPESTR = {
    torch.float32: "<f4",
    torch.float16: "<f2",
    torch.bfloat16: "<f2",
}
# __cuda_array_interface__ has no bfloat16 code, so "<f2" arrives as float16 and
# bfloat16 tensors are reinterpreted afterwards. Same width, same bytes.
_U64 = ctypes.c_uint64
_SIZE = ctypes.c_size_t


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint),
        ("requestedHandleTypes", ctypes.c_uint),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_uint)]


class _Driver:
    """Checked wrapper over the CUDA driver VMM entry points."""

    def __init__(self):
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as error:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "CUDA VMM storage requires libcuda.so.1; disable it on this platform"
            ) from error
        lib = self.lib
        ptr = ctypes.POINTER
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ptr(ctypes.c_char_p)]
        lib.cuMemGetAllocationGranularity.argtypes = [ptr(_SIZE), ptr(_AllocProp), ctypes.c_int]
        lib.cuMemAddressReserve.argtypes = [ptr(_U64), _SIZE, _SIZE, _U64, _U64]
        lib.cuMemAddressFree.argtypes = [_U64, _SIZE]
        lib.cuMemCreate.argtypes = [ptr(_U64), _SIZE, ptr(_AllocProp), _U64]
        lib.cuMemRelease.argtypes = [_U64]
        lib.cuMemMap.argtypes = [_U64, _SIZE, _SIZE, _U64, _U64]
        lib.cuMemUnmap.argtypes = [_U64, _SIZE]
        lib.cuMemSetAccess.argtypes = [_U64, _SIZE, ptr(_AccessDesc), _SIZE]
        for name in (
            "cuMemGetAllocationGranularity",
            "cuMemAddressReserve",
            "cuMemAddressFree",
            "cuMemCreate",
            "cuMemRelease",
            "cuMemMap",
            "cuMemUnmap",
            "cuMemSetAccess",
        ):
            getattr(lib, name).restype = ctypes.c_int

    def check(self, name: str, code: int) -> None:
        if code != 0:
            message = ctypes.c_char_p()
            self.lib.cuGetErrorString(code, ctypes.byref(message))
            text = message.value.decode() if message.value else "unknown"
            raise RuntimeError(f"{name} failed with CUresult {code}: {text}")


_DRIVER: _Driver | None = None


def _driver() -> _Driver:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = _Driver()
    return _DRIVER


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class _ExternalBuffer:
    """Carrier exposing a raw device pointer through __cuda_array_interface__."""

    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (int(ptr), False),
            "version": 2,
            "strides": None,
        }


class GPUKVStorage:
    """One VA reservation holding the K and V planes of a block pool.

    ``max_blocks`` fixes the logical capacity and therefore every tensor shape.
    ``bytes_per_block`` is the K *and* V bytes of one block, matching
    ``KVCacheManager.bytes_per_block``; each plane therefore holds half of it.
    ``commit_blocks`` appends physical memory to both planes; ``close`` releases
    everything. Nothing here knows about requests, block tables, or scheduling.
    """

    def __init__(self, *, max_blocks: int, bytes_per_block: int, device, dtype):
        if type(max_blocks) is not int or max_blocks < 1:
            raise ValueError("max_blocks must be a positive integer")
        if type(bytes_per_block) is not int or bytes_per_block < 1:
            raise ValueError("bytes_per_block must be a positive integer")
        if bytes_per_block % 2:
            raise ValueError("bytes_per_block must cover both K and V, so it must be even")
        if dtype not in _TYPESTR:
            raise ValueError("VMM KV storage supports float32, float16 and bfloat16")
        self.dtype = dtype
        self.max_blocks = max_blocks
        self.bytes_per_block = bytes_per_block
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("VMM KV storage requires a CUDA device")
        self.drv = _driver()
        torch.cuda.set_device(self.device)
        # Establish the primary context before cuMemAddressReserve attaches to it.
        torch.empty(1, device=self.device)
        # The driver only hands out physical memory the allocator has returned.
        # A torch KV pool could reuse its cached segments, but cuMemCreate cannot,
        # so return them before the first commit. Startup profiling leaves one
        # such segment behind (the temporary KV probe).
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        self._prop = _AllocProp()
        self._prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        self._prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        self._prop.location.id = self.device.index or 0
        self._access = _AccessDesc()
        self._access.location = self._prop.location
        self._access.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        granularity = _SIZE()
        self.drv.check(
            "cuMemGetAllocationGranularity",
            self.drv.lib.cuMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(self._prop),
                _CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            ),
        )
        self.allocation_granularity = int(granularity.value)
        # half of bytes_per_block per plane: bytes_per_block already counts K and V.
        self.plane_bytes_per_block = bytes_per_block // 2
        self.plane_bytes = _round_up(
            max_blocks * self.plane_bytes_per_block, self.allocation_granularity
        )
        self.reserve_bytes = 2 * self.plane_bytes

        base = _U64()
        self.drv.check(
            "cuMemAddressReserve",
            self.drv.lib.cuMemAddressReserve(
                ctypes.byref(base), self.reserve_bytes, self.allocation_granularity, 0, 0
            ),
        )
        self.base = int(base.value)
        self._mappings: list[tuple[int, int, int]] = []
        self._mapped_plane_bytes = 0
        self.committed_blocks = 0
        self._closed = False

    # -- capacity ---------------------------------------------------------
    @property
    def committed_bytes(self) -> int:
        """Physical bytes mapped so far, K and V planes together."""
        return 2 * self._mapped_plane_bytes

    def commit_blocks(self, blocks: int) -> int:
        """Map physical memory for ``blocks`` blocks in both planes.

        Returns the committed block count after the call, which can exceed
        ``blocks`` because the last mapping is padded to the VMM granularity.
        Commits only append; an existing mapping is never moved or replaced.
        """
        self._check_open()
        if type(blocks) is not int or blocks < 1:
            raise ValueError("blocks must be a positive integer")
        if blocks > self.max_blocks:
            raise ValueError(
                f"cannot commit {blocks} blocks; the reservation holds {self.max_blocks}"
            )
        if blocks <= self.committed_blocks:
            return self.committed_blocks
        target = _round_up(blocks * self.plane_bytes_per_block, self.allocation_granularity)
        if target > self.plane_bytes:
            raise ValueError(
                f"committing {blocks} blocks needs {target} bytes per plane, "
                f"but the reservation holds {self.plane_bytes}"
            )
        delta = target - self._mapped_plane_bytes
        applied: list[tuple[int, int, int]] = []
        try:
            for offset in (0, self.plane_bytes):
                address = self.base + offset + self._mapped_plane_bytes
                handle = _U64()
                self.drv.check(
                    "cuMemCreate",
                    self.drv.lib.cuMemCreate(
                        ctypes.byref(handle), delta, ctypes.byref(self._prop), 0
                    ),
                )
                try:
                    self.drv.check(
                        "cuMemMap", self.drv.lib.cuMemMap(address, delta, 0, handle.value, 0)
                    )
                except Exception:
                    self.drv.lib.cuMemRelease(handle.value)
                    raise
                try:
                    self.drv.check(
                        "cuMemSetAccess",
                        self.drv.lib.cuMemSetAccess(address, delta, ctypes.byref(self._access), 1),
                    )
                except Exception:  # pragma: no cover - driver dependent
                    self.drv.lib.cuMemUnmap(address, delta)
                    self.drv.lib.cuMemRelease(handle.value)
                    raise
                applied.append((address, delta, handle.value))
        except Exception:
            for address, size, handle in reversed(applied):
                self.drv.lib.cuMemUnmap(address, size)
                self.drv.lib.cuMemRelease(handle)
            raise
        self._mappings.extend(applied)
        self._mapped_plane_bytes = target
        self.committed_blocks = min(target // self.plane_bytes_per_block, self.max_blocks)
        return self.committed_blocks

    # -- tensors ----------------------------------------------------------
    def keys(self, shape) -> torch.Tensor:
        return self._plane_tensor(0, shape)

    def values(self, shape) -> torch.Tensor:
        return self._plane_tensor(self.plane_bytes, shape)

    def _plane_tensor(self, offset: int, shape) -> torch.Tensor:
        self._check_open()
        buffer = _ExternalBuffer(self.base + offset, shape, _TYPESTR[self.dtype])
        tensor = torch.as_tensor(buffer, device=self.device)
        if tensor.dtype != self.dtype:
            # Only bfloat16 reaches here: the interface cannot name it.
            tensor = tensor.view(self.dtype)
        # The mapping outlives every view; closing is the pool's decision alone.
        tensor._vmm_owner = self
        return tensor

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        """Unmap, release handles, then free the reservation. Views become invalid."""
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        for address, size, handle in reversed(self._mappings):
            self.drv.check("cuMemUnmap", self.drv.lib.cuMemUnmap(address, size))
            self.drv.check("cuMemRelease", self.drv.lib.cuMemRelease(handle))
        self._mappings.clear()
        self.drv.check(
            "cuMemAddressFree", self.drv.lib.cuMemAddressFree(self.base, self.reserve_bytes)
        )
        self._mapped_plane_bytes = 0
        self.committed_blocks = 0
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the KV storage pool is closed")
