import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PDConfig:
    prefill_devices: tuple[int, ...] = (0,)
    decode_devices: tuple[int, ...] = (1,)
    max_pending_requests: int = 256
    # KV chunks are submitted, and their completion observed, only between engine
    # steps, so a request whose KV exceeds the in-flight budget waits extra steps
    # after its prefill (the commit tail). 2 GiB covers one 2048-token Ouro-1.4B
    # request (1.6 GB): on a GH200 node the tail fell from ~1 s to 17 ms and 1P1D
    # throughput rose 1.7x. In-flight bytes are not a buffer: NIXL writes straight
    # from the registered prefill pool into the decode pool.
    transfer_chunk_bytes: int = 256 * 1024**2
    max_inflight_bytes: int = 2 * 1024**3
    max_transfer_descriptors: int = 256
    max_control_messages: int = 32
    startup_timeout: float = 300.0
    request_timeout: float = 300.0
    shutdown_timeout: float = 30.0
    backend: str = "UCX"
    max_receiving_requests: int = 32
    max_draining_requests: int = 8

    def __post_init__(self):
        devices = (*self.prefill_devices, *self.decode_devices)
        if not self.prefill_devices or not self.decode_devices:
            raise ValueError("PD requires nonempty prefill and decode device pools")
        if any(type(d) is not int or d < 0 for d in devices) or len(set(devices)) != len(devices):
            raise ValueError("PD devices must be distinct nonnegative CUDA device indices")
        for name in ("max_receiving_requests", "max_draining_requests"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "max_pending_requests",
            "transfer_chunk_bytes",
            "max_inflight_bytes",
            "max_transfer_descriptors",
            "max_control_messages",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.transfer_chunk_bytes > self.max_inflight_bytes:
            raise ValueError("transfer_chunk_bytes exceeds max_inflight_bytes")
        for name in ("startup_timeout", "request_timeout", "shutdown_timeout"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not self.backend:
            raise ValueError("NIXL backend must be specified")
