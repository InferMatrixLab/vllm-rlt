"""PD data transports for registered pools and same-host Ascend staging."""

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass

import torch

from vllm_rlt.device import stream_module


@dataclass
class Transfer:
    handle: object
    transfer_id: str
    sequence: int
    size: int
    submitted: float


def _account_submit(accounts, transfer_id, size):
    entry = accounts.setdefault(transfer_id, dict(transfer_id=transfer_id))
    entry["submitted_bytes"] = entry.get("submitted_bytes", 0) + size
    entry["submitted_chunks"] = entry.get("submitted_chunks", 0) + 1
    entry["status"] = "in_flight"


def _account_complete(accounts, transfer_id, size, *, pending):
    entry = accounts.setdefault(transfer_id, dict(transfer_id=transfer_id))
    entry["completed_bytes"] = entry.get("completed_bytes", 0) + size
    entry["completed_chunks"] = entry.get("completed_chunks", 0) + 1
    entry["status"] = "in_flight" if pending else "complete"


def _account_error(accounts, transfer_id, status):
    entry = accounts.setdefault(transfer_id, dict(transfer_id=transfer_id))
    entry["status"] = status


def _account_receive(accounts, transfer_id, size):
    entry = accounts.setdefault(transfer_id, dict(transfer_id=transfer_id))
    entry["bytes_received"] = entry.get("bytes_received", 0) + size
    entry["received_chunks"] = entry.get("received_chunks", 0) + 1
    entry["status"] = "pending_ready"


def _account_ready(accounts, transfer_id):
    entry = accounts.setdefault(transfer_id, dict(transfer_id=transfer_id))
    entry["ready_chunks"] = entry.get("ready_chunks", 0) + 1
    entry["status"] = (
        "complete" if entry["ready_chunks"] >= entry.get("received_chunks", 0) else "pending_ready"
    )


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


def partition_segments(source, target, byte_limit, descriptor_limit, *, coalesce=True):
    """Bound both transfer size and descriptor count without an intermediate KV copy."""
    from itertools import zip_longest

    local, remote, size = [], [], 0
    if coalesce:
        pairs = coalesce_segments(source, target)
    else:
        pairs = []
        for src, dst in zip_longest(source, target):
            if src is None or dst is None or src[1] != dst[1]:
                raise ValueError("incompatible source/target KV geometry")
            pairs.append((src[0], dst[0], src[1], src[2], dst[2]))
    for src_address, dst_address, length, src_device, dst_device in pairs:
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
            transport="NIXL",
            hidden_rows=hidden.shape[0],
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


@dataclass
class HcclReceive:
    work: object
    future: object
    buffer: torch.Tensor
    descriptors: list
    peer: str
    transfer_id: str
    sequence: int
    size: int
    submitted: float
    event: object = None
    copied: bool = False
    notified: bool = False


class HcclTransport:
    """Asynchronous same-host HCCL point-to-point transport."""

    preserve_segments = True
    # Ascend HCCL matches P2P operations by rank-pair order as well as tag.
    serialize_transfers = True

    @staticmethod
    def _tag(transfer_id, sequence):
        digest = hashlib.sha256(f"{transfer_id}:{sequence}".encode()).digest()
        return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF

    def __init__(self, name, cache, hidden, config, *, rank, world_size, prefill_world_size):
        import torch.distributed as dist

        if not dist.is_initialized() or not dist.is_hccl_available():
            raise RuntimeError("HCCL process group is not initialized")
        self.name = name
        self.cache = cache
        self.hidden = hidden
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.prefill_world_size = prefill_world_size
        self.dist = dist
        self.peers = {}
        self.pair_groups = {}
        self.pair_local_ranks = {}
        decode_world_size = world_size - prefill_world_size
        if decode_world_size != prefill_world_size:
            raise ValueError("Ascend HCCL PD requires equal Prefill and Decode pool sizes")
        for pair_index in range(prefill_world_size):
            prefill_rank, decode_rank = pair_index, prefill_world_size + pair_index
            pair = (prefill_rank, decode_rank)
            group = dist.new_group(ranks=list(pair), backend="hccl")
            if rank in pair:
                self.pair_groups[pair] = group
                self.pair_local_ranks[pair] = 0 if rank == prefill_rank else 1
        self._warmup_pair_groups()
        self.sends = deque()
        self.receive_plans = {}
        self.receive_started = set()
        self.receives = {}
        self.send_keys = set()
        self.inflight_bytes = 0
        self.bytes_sent = 0
        self.transfers = 0
        self.transfer_seconds = 0.0
        self.transfer_stats = {}
        parameter = cache.key_cache
        element = parameter.element_size()
        self.info = dict(
            agent=name,
            rank=rank,
            world_size=world_size,
            transport="HCCL",
            key_ptr=parameter.data_ptr(),
            value_ptr=cache.value_cache.data_ptr(),
            hidden_ptr=hidden.data_ptr(),
            hidden_bytes=hidden.stride(0) * hidden.element_size(),
            hidden_rows=hidden.shape[0],
            device=str(parameter.device),
            num_blocks=cache.num_blocks,
            block_size=cache.block_size,
            num_layers=cache.num_layers,
            depths=cache.storage_depths,
            block_bytes=parameter.stride(0) * element,
            layer_bytes=parameter.stride(1) * element,
            token_bytes=parameter.stride(2) * element,
            kv_dtype=str(parameter.dtype),
            fp8_scale=cache.fp8_scale,
            scale_region=None,
        )

    def _warmup_pair_groups(self):
        """Initialize every pair communicator before request traffic is serialized."""
        for prefill_rank, decode_rank in self.pair_groups:
            group = self.pair_groups[(prefill_rank, decode_rank)]
            if self.rank == prefill_rank:
                buffer = torch.zeros(1, dtype=torch.uint8, device=self.cache.device)
                operation = self.dist.P2POp(
                    self.dist.isend, buffer, decode_rank, group=group, tag=decode_rank
                )
            elif self.rank == decode_rank:
                buffer = torch.empty(1, dtype=torch.uint8, device=self.cache.device)
                operation = self.dist.P2POp(
                    self.dist.irecv,
                    buffer,
                    prefill_rank,
                    group=group,
                    tag=decode_rank,
                )
            else:
                continue
            self.dist.batch_isend_irecv([operation])[0].wait()

    @property
    def pending(self):
        return list(self.sends) + list(self.receive_plans.values()) + list(self.receives.values())

    def connect(self, peer):
        for key in ("block_size", "num_blocks", "num_layers", "depths", "hidden_bytes", "kv_dtype"):
            if peer[key] != self.info[key]:
                raise ValueError(f"PD KV geometry mismatch: {key}")
        if peer.get("fp8_scale") != self.info.get("fp8_scale"):
            raise ValueError("P and D must use the same calibrated FP8 scale")
        if peer["world_size"] != self.world_size:
            raise ValueError("HCCL world size mismatch")
        peer_rank = peer["rank"]
        pair = (
            (self.rank, peer_rank)
            if self.rank < self.prefill_world_size
            else (peer_rank, self.rank)
        )
        try:
            group = self.pair_groups[pair]
            local_rank = self.pair_local_ranks[pair]
        except KeyError as exc:
            raise ValueError("HCCL peer is not in a prefill/decode subgroup") from exc
        self.peers[peer["agent"]] = dict(
            peer, group=group, local_rank=local_rank, peer_local_rank=1 - local_rank
        )

    def _copy_to_buffer(self, descriptors, size):
        buffer = torch.empty(size, dtype=torch.uint8, device=self.cache.device)
        offset = 0
        for address, length, _ in descriptors:
            view = self._address_view(address, int(length))
            buffer[offset : offset + length].copy_(view)
            offset += length
        if offset != size:
            raise ValueError("HCCL source descriptor bytes do not match transfer size")
        return buffer

    def _address_view(self, address, size):
        regions = (self.cache.key_cache, self.cache.value_cache, self.hidden)
        if self.cache.key_scale is not None and self.cache.fp8_scale is None:
            regions += (self.cache.key_scale, self.cache.value_scale)
        for tensor in regions:
            base = tensor.data_ptr()
            capacity = tensor.numel() * tensor.element_size()
            offset = int(address) - base
            if 0 <= offset <= capacity - size:
                return tensor.view(torch.uint8).reshape(-1)[offset : offset + size]
        raise ValueError("HCCL descriptor is outside local storage")

    def _copy_from_buffer(self, buffer, descriptors):
        offset = 0
        for address, length, _ in descriptors:
            target = self._address_view(address, int(length))
            target.copy_(buffer[offset : offset + length])
            offset += length
        if offset != buffer.numel():
            raise ValueError("HCCL target descriptor bytes do not match transfer size")

    def prepare_receive(self, transfer_id, peer, target_tables, target_slot, start, end, final):
        if start != 0:
            raise ValueError("HCCL PD does not support prefix-cached imports")
        if any(key[0] == transfer_id for key in (*self.receive_plans, *self.receives)):
            raise RuntimeError("duplicate HCCL receive generation")
        peer_info = self.peers[peer]
        target = list(kv_segments(self.info, target_tables, start, end))
        chunks = list(
            partition_segments(
                target,
                target,
                self.config.transfer_chunk_bytes,
                self.config.max_transfer_descriptors,
                coalesce=False,
            )
        )
        if final:
            size = self.info["hidden_bytes"]
            hidden = [
                (
                    self.info["hidden_ptr"] + target_slot * size,
                    size,
                    self.info["device"],
                )
            ]
            chunks.extend(
                partition_segments(
                    hidden,
                    hidden,
                    self.config.transfer_chunk_bytes,
                    self.config.max_transfer_descriptors,
                    coalesce=False,
                )
            )
        for sequence, (descriptors, _, size) in enumerate(chunks):
            buffer = torch.empty(size, dtype=torch.uint8, device=self.cache.device)
            self.receive_plans[(transfer_id, sequence)] = dict(
                buffer=buffer,
                descriptors=descriptors,
                peer=peer,
                rank=peer_info["rank"],
                group=peer_info["group"],
                peer_local_rank=peer_info["peer_local_rank"],
                transfer_id=transfer_id,
                sequence=sequence,
                size=size,
                submitted=time.monotonic(),
            )

    def start_receive(self, transfer_id, sequence=None):
        keys = (
            [(transfer_id, sequence)]
            if sequence is not None
            else [key for key in self.receive_plans if key[0] == transfer_id]
        )
        self.receive_started.update(key for key in keys if key in self.receive_plans)

    def submit(self, transfer_id, sequence, peer, local, remote, size):
        if self.inflight_bytes + size > self.config.max_inflight_bytes:
            return False
        if len(local) != len(remote):
            raise ValueError("HCCL source and target descriptor counts differ")
        if any(src[1] != dst[1] for src, dst in zip(local, remote, strict=True)):
            raise ValueError("HCCL source and target descriptor sizes differ")
        peer_info = self.peers[peer]
        if (transfer_id, sequence) in self.send_keys:
            raise RuntimeError("duplicate HCCL transfer sequence")
        buffer = self._copy_to_buffer(local, size)
        work = self.dist.isend(
            buffer,
            dst=peer_info["rank"],
            tag=self._tag(transfer_id, sequence),
            group=peer_info["group"],
        )
        future = work.get_future()
        self.sends.append(
            Transfer((work, buffer, future), transfer_id, sequence, size, time.monotonic())
        )
        self.send_keys.add((transfer_id, sequence))
        self.inflight_bytes += size
        _account_submit(self.transfer_stats, transfer_id, size)
        return True

    def poll(self):
        finished = []
        notifications = []
        now = time.monotonic()
        remaining = deque()
        for ticket in self.sends:
            if now - ticket.submitted > self.config.request_timeout:
                _account_error(self.transfer_stats, ticket.transfer_id, "timeout")
                raise TimeoutError("HCCL send timed out; device memory must be quarantined")
            if not ticket.handle[2].done():
                remaining.append(ticket)
                continue
            ticket.handle[0].wait()
            self.inflight_bytes -= ticket.size
            self.bytes_sent += ticket.size
            self.transfers += 1
            self.transfer_seconds += now - ticket.submitted
            _account_complete(
                self.transfer_stats,
                ticket.transfer_id,
                ticket.size,
                pending=any(
                    other is not ticket and other.transfer_id == ticket.transfer_id
                    for other in self.sends
                ),
            )
            finished.append(ticket)
            self.send_keys.remove((ticket.transfer_id, ticket.sequence))
        self.sends = remaining
        for transfer_id, account in self.transfer_stats.items():
            if account.get("status") == "in_flight" and not any(
                ticket.transfer_id == transfer_id for ticket in self.sends
            ):
                account["status"] = "complete"
        # HCCL's Ascend irecv call waits until the matching isend is launched.
        # Register one planned receive at a time after the coordinator has sent
        # the reservation acknowledgement, which lets P compute and enqueue it.
        receive_plans = getattr(self, "receive_plans", {})
        for key, plan in list(receive_plans.items()):
            if key not in getattr(self, "receive_started", set()):
                continue
            if now - plan["submitted"] > self.config.request_timeout:
                _account_error(self.transfer_stats, plan["transfer_id"], "timeout")
                raise TimeoutError("HCCL receive timed out; device memory must be quarantined")
            work = self.dist.irecv(
                plan["buffer"],
                src=plan["rank"],
                tag=self._tag(plan["transfer_id"], plan["sequence"]),
                group=plan["group"],
            )
            self.receives[key] = HcclReceive(
                work,
                work.get_future(),
                plan["buffer"],
                plan["descriptors"],
                plan["peer"],
                plan["transfer_id"],
                plan["sequence"],
                plan["size"],
                plan["submitted"],
            )
            del receive_plans[key]
            self.receive_started.discard(key)
        for key, receive in list(self.receives.items()):
            if now - receive.submitted > self.config.request_timeout:
                _account_error(self.transfer_stats, receive.transfer_id, "timeout")
                raise TimeoutError("HCCL receive timed out; device memory must be quarantined")
            if not receive.future.done():
                continue
            receive.work.wait()
            if not receive.copied:
                self._copy_from_buffer(receive.buffer, receive.descriptors)
                device_ops = stream_module(self.cache.device)
                receive.event = device_ops.Event()
                receive.event.record(device_ops.current_stream(self.cache.device))
                receive.copied = True
                _account_receive(self.transfer_stats, receive.transfer_id, receive.size)
            if receive.event is not None and not receive.event.query():
                continue
            if receive.notified:
                continue
            _account_ready(self.transfer_stats, receive.transfer_id)
            notifications.append((receive.peer, receive.transfer_id, receive.sequence))
            receive.notified = True
            del self.receives[key]
        return finished, notifications

    def busy(self, transfer_id):
        return (
            any(ticket.transfer_id == transfer_id for ticket in self.sends)
            or any(receive.transfer_id == transfer_id for receive in self.receives.values())
            or any(
                plan["transfer_id"] == transfer_id
                for plan in getattr(self, "receive_plans", {}).values()
            )
        )

    def cancel(self, transfer_id):
        """Drop receive buffers that have not been posted to HCCL yet."""
        for key, plan in list(getattr(self, "receive_plans", {}).items()):
            if plan["transfer_id"] == transfer_id:
                del self.receive_plans[key]
        self.receive_started = {
            key for key in getattr(self, "receive_started", set()) if key[0] != transfer_id
        }

    def close(self):
        if self.sends or getattr(self, "receive_plans", {}) or self.receives:
            raise RuntimeError("cannot close HCCL transport with in-flight transfers")
