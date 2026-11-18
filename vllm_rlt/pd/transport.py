"""NIXL WRITE on registered GPU pools; no tensor payloads on the control channel."""

import json
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class Transfer:
    handle: object
    transfer_id: str
    sequence: int
    size: int
    submitted: float


def _kv_region(key, value):
    element = key.element_size()
    return dict(
        key_ptr=key.data_ptr(),
        value_ptr=value.data_ptr(),
        block_bytes=key.stride(0) * element,
        layer_bytes=key.stride(1) * element,
        token_bytes=key.stride(2) * element,
    )


def kv_segments(info, tables, start, end):
    """Map valid token ranges to contiguous byte segments, preserving every depth.

    A full physical block holds all layers. Partial blocks need one range per
    layer; unused tail bytes are never sent or advertised as initialized.
    """
    page = info["block_size"]
    if not 0 <= start < end:
        raise ValueError("invalid KV range")
    scale_region = info.get("scale_region")
    regions = (info,) if scale_region is None else (info, scale_region)
    for table in tables:
        if end > len(table) * page:
            raise ValueError("KV range exceeds allocation")
        for logical in range(start // page, (end - 1) // page + 1):
            block = table[logical]
            if not 0 <= block < info["num_blocks"]:
                raise ValueError("physical block outside registered pool")
            lo = max(start - logical * page, 0)
            hi = min(end - logical * page, page)
            for region in regions:
                block_bytes = region["block_bytes"]
                layer_bytes = region["layer_bytes"]
                token_bytes = region["token_bytes"]
                for base in (region["key_ptr"], region["value_ptr"]):
                    address = base + block * block_bytes
                    if lo == 0 and hi == page:
                        yield (address, block_bytes, info["device"])
                    else:
                        for layer in range(info["num_layers"]):
                            yield (
                                address + layer * layer_bytes + lo * token_bytes,
                                (hi - lo) * token_bytes,
                                info["device"],
                            )


def coalesce_segments(source, target):
    """Merge consecutive segment pairs that are contiguous on both the source and target.

    Physical blocks handed out in order are adjacent in both pools, so a prompt's
    per-block descriptors collapse into a few long ranges. Merging never moves a
    byte: each merged pair covers exactly the bytes of its members, in order.
    """
    from itertools import zip_longest

    current = None
    for src, dst in zip_longest(source, target):
        if src is None or dst is None or src[1] != dst[1]:
            raise ValueError("incompatible source/target KV geometry")
        if (
            current is not None
            and current[0] + current[2] == src[0]
            and current[1] + current[2] == dst[0]
            and current[3] == src[2]
            and current[4] == dst[2]
        ):
            current[2] += src[1]
            continue
        if current is not None:
            yield tuple(current)
        current = [src[0], dst[0], src[1], src[2], dst[2]]
    if current is not None:
        yield tuple(current)


def partition_segments(source, target, byte_limit, descriptor_limit):
    """Bound both transfer size and descriptor count without an intermediate KV copy."""
    local, remote, size = [], [], 0
    for src_address, dst_address, length, src_device, dst_device in coalesce_segments(
        source, target
    ):
        offset = 0
        while offset < length:
            amount = min(length - offset, byte_limit - size)
            local.append((src_address + offset, amount, src_device))
            remote.append((dst_address + offset, amount, dst_device))
            size += amount
            offset += amount
            if size == byte_limit or len(local) == descriptor_limit:
                yield local, remote, size
                local, remote, size = [], [], 0
    if local:
        yield local, remote, size


class NixlConnector:
    def __init__(self, name, cache, hidden, config):
        from nixl._api import nixl_agent, nixl_agent_config

        self.config = config
        if config.backend == "UCX":
            # NIXL's default (peer) error handling requires a peer failure handler,
            # which only the network transports provide, so UCX leaves the
            # shared-memory lanes out of intra-node RMA. A lost worker is reported
            # by the control channel, which owns worker liveness, so transfers do
            # not need a handler of their own. nixl_agent_config cannot pass
            # backend init parameters, so the backend is created explicitly.
            self.agent = nixl_agent(name, nixl_agent_config(backends=[]))
            self.agent.create_backend(config.backend, {"ucx_error_handling_mode": "none"})
        else:
            self.agent = nixl_agent(name, nixl_agent_config(backends=[config.backend]))
        self.storage = (cache.key_cache, cache.value_cache, hidden)
        # Dynamic scales travel with their KV rows. Fixed scales are checked
        # during the handshake and already initialized on the receiving side.
        scale_region = None
        if cache.key_scale is not None and cache.fp8_scale is None:
            self.storage += (cache.key_scale, cache.value_scale)
            scale_region = _kv_region(cache.key_scale, cache.value_scale)
        self.registrations = [self.agent.register_memory(t) for t in self.storage]
        k = cache.key_cache
        self.info = dict(
            **_kv_region(k, cache.value_cache),
            agent=name,
            metadata=self.agent.get_agent_metadata(),
            hidden_ptr=hidden.data_ptr(),
            hidden_bytes=hidden.stride(0) * hidden.element_size(),
            device=k.device.index,
            num_blocks=cache.num_blocks,
            block_size=cache.block_size,
            num_layers=cache.num_layers,
            depths=cache.storage_depths,
            kv_dtype=str(k.dtype),
            fp8_scale=cache.fp8_scale,
            scale_region=scale_region,
        )
        self.peers = {}
        self.pending = deque()
        self.inflight_bytes = 0
        self.bytes_sent = 0
        self.transfers = 0
        self.transfer_seconds = 0.0

    def connect(self, peer):
        strides = ("block_bytes", "layer_bytes", "token_bytes")
        for key in ("block_size", "num_layers", "depths", "hidden_bytes", "kv_dtype", *strides):
            if peer[key] != self.info[key]:
                raise ValueError(f"PD KV geometry mismatch: {key}")
        if peer["fp8_scale"] != self.info["fp8_scale"]:
            raise ValueError("P and D must use the same calibrated FP8 scale")
        local_scales, peer_scales = self.info["scale_region"], peer["scale_region"]
        if (local_scales is None) != (peer_scales is None) or (
            local_scales is not None and any(local_scales[k] != peer_scales[k] for k in strides)
        ):
            raise ValueError("PD KV scale geometry mismatch")
        name = self.agent.add_remote_agent(peer["metadata"])
        name = name.decode() if isinstance(name, bytes) else name
        if name != peer["agent"]:
            raise ValueError(
                f"NIXL agent metadata identity mismatch: {name!r} != {peer['agent']!r}"
            )
        self.peers[name] = peer

    def submit(self, transfer_id, sequence, peer, local, remote, size):
        if size > self.config.transfer_chunk_bytes:
            raise ValueError("transfer exceeds configured chunk size")
        if self.inflight_bytes + size > self.config.max_inflight_bytes:
            return False
        notification = json.dumps([transfer_id, sequence]).encode()
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs(local, "VRAM"),
            self.agent.get_xfer_descs(remote, "VRAM"),
            peer,
            notif_msg=notification,
            backends=[self.config.backend],
        )
        ticket = Transfer(handle, transfer_id, sequence, size, time.monotonic())
        self.pending.append(ticket)  # Keep ownership even if submission fails.
        self.inflight_bytes += size
        if self.agent.transfer(handle) == "ERR":
            raise RuntimeError("NIXL WRITE submission failed")
        return True

    def poll(self):
        remaining = deque()
        finished = []
        for ticket in self.pending:
            status = self.agent.check_xfer_state(ticket.handle)
            if status == "ERR":
                raise RuntimeError("NIXL WRITE failed; registered memory must be quarantined")
            if time.monotonic() - ticket.submitted > self.config.request_timeout:
                raise TimeoutError("NIXL WRITE timed out; registered memory must be quarantined")
            if status != "DONE":
                remaining.append(ticket)
                continue
            self.agent.release_xfer_handle(ticket.handle)
            ticket.handle = None
            self.inflight_bytes -= ticket.size
            self.bytes_sent += ticket.size
            self.transfers += 1
            self.transfer_seconds += time.monotonic() - ticket.submitted
            finished.append(ticket)
        self.pending = remaining
        notifications = []
        for peer, messages in self.agent.get_new_notifs().items():
            peer = peer.decode() if isinstance(peer, bytes) else peer
            for message in messages:
                tid, sequence = json.loads(message)
                notifications.append((peer, tid, sequence))
        return finished, notifications

    def busy(self, transfer_id):
        return any(t.transfer_id == transfer_id for t in self.pending)

    def close(self):
        if self.pending:
            raise RuntimeError("cannot deregister memory with in-flight transfers")
        for peer in self.peers:
            self.agent.remove_remote_agent(peer)
        for registration in self.registrations:
            self.agent.deregister_memory(registration)
        self.registrations.clear()
        self.agent = None
