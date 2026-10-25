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


def kv_segments(info, tables, start, end):
    """Map valid token ranges to contiguous byte segments, preserving every depth.

    ``tables`` has one physical block table per physical plane and
    ``plane_slots`` each plane's live layer slots (default: every layer). A full
    page sends its live slots as one range; partial pages need one range per
    slot. Unused tail bytes and dead layer slots are never sent.
    """
    page = info["block_size"]
    if not 0 <= start < end:
        raise ValueError("invalid KV range")
    slots = info.get("plane_slots") or ((0, info["num_layers"]),) * len(tables)
    if len(slots) != len(tables):
        raise ValueError("one live slot range is required per physical plane")
    for table, (first, last) in zip(tables, slots):
        if end > len(table) * page:
            raise ValueError("KV range exceeds allocation")
        for logical in range(start // page, (end - 1) // page + 1):
            block = table[logical]
            if not 0 <= block < info["num_blocks"]:
                raise ValueError("physical block outside registered pool")
            lo = max(start - logical * page, 0)
            hi = min(end - logical * page, page)
            for base in (info["key_ptr"], info["value_ptr"]):
                address = base + block * info["block_bytes"]
                if lo == 0 and hi == page:
                    yield (
                        address + first * info["layer_bytes"],
                        (last - first) * info["layer_bytes"],
                        info["device"],
                    )
                else:
                    for layer in range(first, last):
                        yield (
                            address + layer * info["layer_bytes"] + lo * info["token_bytes"],
                            (hi - lo) * info["token_bytes"],
                            info["device"],
                        )


def partition_segments(source, target, byte_limit, descriptor_limit):
    """Bound both transfer size and descriptor count without an intermediate KV copy."""
    from itertools import zip_longest

    local, remote, size = [], [], 0
    for src, dst in zip_longest(source, target):
        if src is None or dst is None or src[1] != dst[1]:
            raise ValueError("incompatible source/target KV geometry")
        offset = 0
        while offset < src[1]:
            amount = min(src[1] - offset, byte_limit - size)
            local.append((src[0] + offset, amount, src[2]))
            remote.append((dst[0] + offset, amount, dst[2]))
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
        self.agent = nixl_agent(name, nixl_agent_config(backends=[config.backend]))
        self.storage = (cache.key_cache, cache.value_cache, hidden)
        self.registrations = [self.agent.register_memory(t) for t in self.storage]
        k = cache.key_cache
        element = k.element_size()
        self.info = dict(
            agent=name,
            metadata=self.agent.get_agent_metadata(),
            key_ptr=k.data_ptr(),
            value_ptr=cache.value_cache.data_ptr(),
            hidden_ptr=hidden.data_ptr(),
            hidden_bytes=hidden.stride(0) * hidden.element_size(),
            device=k.device.index,
            num_blocks=cache.num_blocks,
            block_size=cache.block_size,
            num_layers=cache.num_layers,
            depths=cache.storage_depths,
            plane_slots=tuple(tuple(slot) for slot in cache.transfer_slots),
            block_bytes=k.stride(0) * element,
            layer_bytes=k.stride(1) * element,
            token_bytes=k.stride(2) * element,
        )
        self.peers = {}
        self.pending = deque()
        self.inflight_bytes = 0
        self.bytes_sent = 0
        self.transfers = 0
        self.transfer_seconds = 0.0

    def connect(self, peer):
        name = self.agent.add_remote_agent(peer["metadata"])
        name = name.decode() if isinstance(name, bytes) else name
        if name != peer["agent"]:
            raise ValueError(
                f"NIXL agent metadata identity mismatch: {name!r} != {peer['agent']!r}"
            )
        for key in ("block_size", "num_layers", "depths", "block_bytes", "hidden_bytes"):
            if peer[key] != self.info[key]:
                raise ValueError(f"PD KV geometry mismatch: {key}")
        # Slot ranges may arrive as lists after serialization.
        if [list(s) for s in peer["plane_slots"]] != [list(s) for s in self.info["plane_slots"]]:
            raise ValueError("PD KV geometry mismatch: plane_slots")
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
