"""Per-token attention reference state for sparse decode attention.

A sparse decode loop reuses the reference loop's attention output and only
recomputes the contribution of the retained keys. The retained state is
per token, per layer and per head, so it cannot be stored per request.

This module owns that state. The store keeps one slot per live token, tracks
which request and generation owns it, and publishes the slot views that the
attention kernels borrow. Nothing here reads device state back to the host:
ownership, retirement and invalidation are expressed as device operations so
that an asynchronous round can retire its slots without a host readback.

Slot lifecycle
--------------
``empty`` -> ``source-ready`` -> ``reusable`` -> ``retired``

A slot becomes source-ready when its capture round finished writing the
reference, and reusable once a later loop consumed it. Retirement is ordered
after the last reader of the slot, so a later submission cannot observe a slot
that an earlier round still reads.

Slot identity
-------------
A slot is identified by its request generation, its row inside the request and
an incarnation counter. The incarnation changes whenever the token of that row
changes, including a corrected token that reuses the same position after a
rejection. A plan built for a superseded incarnation is stale and rejected.
"""

from dataclasses import dataclass

import torch

from vllm_rlt.attention.sparse import CAPTURE, CONSUME, DENSE, SparseDecodePlan

__all__ = ["ReferenceLease", "AttentionReferenceStore", "DENSE", "CAPTURE", "CONSUME"]


@dataclass(frozen=True)
class ReferenceLease:
    """One request's reference allocation for one speculative submission.

    ``slots`` maps the request's verification row (token position) to its
    reference slot. The device tensors mirror them so that acceptance-driven
    retirement stays on the device.
    """

    request_id: str
    generation: int
    request_index: int
    slots: tuple[int, ...]
    incarnations: tuple[int, ...]
    slot_ids: torch.Tensor
    row_index: torch.Tensor

    def __len__(self) -> int:
        return len(self.slots)


class AttentionReferenceStore:
    """Device-resident reference state for the live tokens of one stage.

    The store is sized by the engine, which knows how many tokens can be live
    at once. It never allocates per request: a speculative round holds one slot
    per verification row, and several rounds of the same request may be
    outstanding at the same time, so the capacity must cover every outstanding
    round.
    """

    def __init__(self, *, slots, layers, heads, head_dim, key_bucket, device, stream=None):
        if slots < 1 or layers < 1 or heads < 1 or head_dim < 1:
            raise ValueError("reference store dimensions must be positive")
        if key_bucket is not None and key_bucket < 1:
            raise ValueError("key_bucket must be positive when set")
        self.slots = slots
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.key_bucket = key_bucket
        self.device = torch.device(device)
        self.stream = stream
        shape = (slots, layers, heads)
        self.selected = torch.zeros((*shape, key_bucket or 1), dtype=torch.int32, device=device)
        self.counts = torch.zeros(shape, dtype=torch.int32, device=device)
        self.mass = torch.zeros(shape, dtype=torch.float32, device=device)
        self.output = torch.zeros((*shape, head_dim), dtype=torch.float32, device=device)
        self.contribution = torch.zeros((*shape, head_dim), dtype=torch.float32, device=device)
        # A reference is usable per layer while its slot is live.
        self.valid = torch.zeros((slots, layers), dtype=torch.bool, device=device)
        # How many later loops consumed the reference of each slot and layer.
        self.uses = torch.zeros((slots, layers), dtype=torch.int32, device=device)
        # Identity of the token occupying each slot.
        self.position = torch.zeros(slots, dtype=torch.int64, device=device)
        self.incarnation = torch.zeros(slots, dtype=torch.int64, device=device)
        self.accepted = torch.zeros(slots, dtype=torch.int32, device=device)
        self.free = list(range(slots))
        self._bindings: dict[int, tuple[str, int, int]] = {}
        self._incarnations: dict[tuple[str, int], int] = {}
        # Retained counts are host-known: the capture decides them before it
        # submits. Keeping them here lets a reader size its gather without a
        # device readback, which RFC #156 section 6.2 forbids.
        self._counts_host: dict[tuple[int, int], int] = {}

    @property
    def bytes(self) -> int:
        """Payload bytes held by the store, excluding allocator overhead."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.selected,
                self.counts,
                self.mass,
                self.output,
                self.contribution,
                self.valid,
                self.uses,
                self.position,
                self.incarnation,
                self.accepted,
            )
        )

    @property
    def outstanding(self) -> int:
        """Number of slots currently bound to live tokens."""
        return len(self._bindings)

    def reserve(self, request_id: str, request_index: int, rows: int, generation: int):
        """Take one slot per verification row of a request's submission."""
        if rows < 1:
            raise ValueError("a reference reservation needs at least one row")
        if len(self.free) < rows:
            raise RuntimeError("no retired attention reference slots")
        slots = tuple(self.free.pop() for _ in range(rows))
        incarnations = []
        for row, slot in enumerate(slots):
            count = self._incarnations.get((request_id, row), 0) + 1
            self._incarnations[(request_id, row)] = count
            self._bindings[slot] = (request_id, generation, count)
            self.incarnation[slot] = count
            incarnations.append(count)
        return ReferenceLease(
            request_id=request_id,
            generation=generation,
            request_index=request_index,
            slots=slots,
            incarnations=tuple(incarnations),
            slot_ids=torch.tensor(slots, dtype=torch.int64, device=self.device),
            row_index=torch.arange(rows, dtype=torch.int64, device=self.device),
        )

    def release(self, leases) -> None:
        """Return slots to the pool once their last reader completed.

        The caller orders this after the final reader/writer, e.g. by releasing
        from a ticket that already waited for the round's completion event. The
        cleared payload never leaks into a later incarnation.
        """
        if isinstance(leases, ReferenceLease):
            leases = (leases,)
        cleared = []
        for lease in leases:
            for slot in lease.slots:
                binding = self._bindings.get(slot)
                if binding is None or binding[:2] != (lease.request_id, lease.generation):
                    raise RuntimeError("attention reference slot released by a different owner")
                del self._bindings[slot]
                cleared.append(slot)
        if cleared:
            self.free.extend(cleared)
            self._clear(torch.tensor(cleared, dtype=torch.int64, device=self.device))

    def discard(self) -> None:
        """Free every binding without device work; the caller has drained."""
        self._bindings.clear()
        self.free = list(range(self.slots))

    def build_plan(self, owners, modes, *, key_retention: float) -> SparseDecodePlan:
        """Publish the plan of one submission.

        ``owners`` holds one ``(lease, row)`` pair per submission row, in the
        row order of the recurrent step; ``modes`` holds the row mode of every
        submission row. Slots and incarnations are re-read from the live
        bindings, so a plan never aliases a superseded token.
        """
        if len(owners) != len(modes):
            raise ValueError("a reference plan needs one mode per row")
        slots, incarnations = [], []
        for owner, mode in zip(owners, modes):
            if owner is None:
                if mode != DENSE:
                    raise ValueError("a sparse row needs a reference owner")
                slots.append(-1)
                incarnations.append(0)
                continue
            lease, row = owner
            slot = lease.slots[row]
            binding = self._bindings.get(slot)
            if binding is None or binding != (
                lease.request_id,
                lease.generation,
                lease.incarnations[row],
            ):
                raise RuntimeError("attention reference slot released before its plan")
            slots.append(slot)
            incarnations.append(binding[2])
        return SparseDecodePlan(
            modes=tuple(modes),
            slots=tuple(slots),
            incarnations=tuple(incarnations),
            key_retention=key_retention,
            key_bucket=self.key_bucket,
            slot_ids=torch.tensor(slots, dtype=torch.int64, device=self.device),
            store=self,
        )

    def _require_live(self, plan: SparseDecodePlan) -> None:
        """Reject a plan whose slots were recycled or superseded."""
        if plan.store is not self:
            raise RuntimeError("attention reference plan belongs to a different store")
        for slot, incarnation, mode in zip(plan.slots, plan.incarnations, plan.modes):
            if slot == -1 and mode == DENSE:
                continue
            binding = self._bindings.get(slot)
            if binding is None or binding[2] != incarnation:
                raise RuntimeError("stale attention reference plan for a retired token")

    def note_count(self, slot: int, layer: int, count: int) -> None:
        """Record the retained count of one capture row on the host.

        The count is decided by the capture, so it never needs a device
        readback; ``host_count`` is the reader-side counterpart.
        """
        self._counts_host[(slot, layer)] = count

    def host_count(self, slot: int, layer: int) -> int | None:
        """Retained keys of ``(slot, layer)``, or ``None`` before a capture."""
        return self._counts_host.get((slot, layer))

    def capture(self, plan: SparseDecodePlan, layer: int) -> None:
        """Mark the captured rows source-ready for the given layer.

        The attention kernel writes the retained state of a capture row and
        then calls this on the same stream, so a later reader observes a
        complete payload.
        """
        self._require_live(plan)
        rows = plan.groups().get(CAPTURE)
        if not rows:
            return
        # Assign a tensor, not the Python scalar ``True``: a scalar store into
        # a bool tensor through advanced indexing costs milliseconds per call
        # (2.6 ms measured at 32 rows), while a tensor store is ~0.13 ms.
        slots = plan.slot_ids[rows]
        self.valid[slots, layer] = torch.ones(slots.numel(), dtype=torch.bool, device=self.device)

    def consume(self, plan: SparseDecodePlan, layer: int) -> None:
        """Record that the given layer consumed the reference rows."""
        self._require_live(plan)
        rows = plan.groups().get(CONSUME)
        if not rows:
            return
        self.uses[plan.slot_ids[rows], layer] += 1

    def bind_positions(self, plan: SparseDecodePlan, positions: torch.Tensor) -> None:
        """Record the device-owned absolute positions of the plan's rows."""
        self._require_live(plan)
        if positions.shape != plan.slot_ids.shape:
            raise ValueError("reference positions need one entry per plan row")
        rows = [row for row, slot in enumerate(plan.slots) if slot >= 0]
        self.position[plan.slot_ids[rows]] = positions[rows].to(torch.int64)

    def write_reference(
        self,
        plan: SparseDecodePlan,
        layer: int,
        row: int,
        *,
        selected,
        counts,
        mass,
        output,
        contribution,
    ) -> None:
        """Publish the retained state of one capture row (reference path)."""
        self._require_live(plan)
        slot = plan.slots[row]
        count = selected.shape[-1]
        capacity = self.selected.shape[-1]
        if count > capacity:
            raise ValueError(f"reference capacity {capacity} cannot hold {count} retained keys")
        self.selected[slot, layer, :, :count] = selected.to(torch.int32)
        self.counts[slot, layer] = counts.to(torch.int32)
        self.note_count(slot, layer, int(count))
        self.mass[slot, layer] = mass
        self.output[slot, layer] = output
        self.contribution[slot, layer] = contribution

    def read_reference(self, plan: SparseDecodePlan, layer: int, row: int):
        """Borrow the retained state of one consume row (reference path)."""
        self._require_live(plan)
        slot = plan.slots[row]
        count = self.host_count(slot, layer)
        if not count:
            raise RuntimeError("a consume row has no captured reference")
        return (
            self.selected[slot, layer, :, :count].long(),
            self.mass[slot, layer],
            self.output[slot, layer],
            self.contribution[slot, layer],
        )

    def accept(self, leases, accepted: torch.Tensor) -> None:
        """Invalidate the references of rejected rows on the device.

        ``accepted`` is the number of accepted candidates of every request of
        the submission. A verification row of a request survives exactly when
        its row index is at most the accepted count: the root row is always
        kept, and a rejected candidate invalidates its dependent descendants.
        The comparison stays on the device so that acceptance never requires a
        host readback.

        Retirement is memory-bounded: it only clears the per-row metadata that
        gates readers (``valid``, ``counts``, ``mass``), never the retained
        index buffer. An indexed read-modify-write of ``selected`` would
        materialise the whole bucket — at 24 layers, 16 heads and 65,536 keys
        that is tens of GiB for one batch — while a zero ``count`` already makes
        the payload unreadable, and a later capture overwrites the used prefix.
        """
        if isinstance(leases, ReferenceLease):
            leases = (leases,)
        leases = tuple(leases)
        if not leases:
            return
        slot_ids = torch.cat([lease.slot_ids for lease in leases])
        rows = torch.cat([lease.row_index for lease in leases])
        request = torch.cat(
            [torch.full_like(lease.row_index, lease.request_index) for lease in leases]
        )
        limit = accepted.to(device=self.device, dtype=rows.dtype).index_select(0, request)
        self.accepted[slot_ids] = limit.to(torch.int32)
        rejected = slot_ids[rows > limit]
        if rejected.numel():
            # Scalar fills touch only the rejected rows and allocate nothing.
            self.valid[rejected] = False
            self.counts[rejected] = 0
            self.mass[rejected] = 0
            for slot in rejected.tolist():
                for layer in range(self.counts.shape[1]):
                    self._counts_host.pop((slot, layer), None)

    def _clear(self, slots: torch.Tensor) -> None:
        """Retire the metadata of the given slots, ordered on the owner stream.

        Only the tensors that gate readers are cleared. The retained index,
        output and contribution buffers stay allocated and are overwritten by
        the next capture's used prefix, so retiring a slot never writes the
        whole bucket (which would be tens of GiB per batch at full retention).
        """

        def zero():
            for slot in slots.tolist():
                for layer in range(self.counts.shape[1]):
                    self._counts_host.pop((slot, layer), None)
            self.valid[slots] = False
            self.uses[slots] = 0
            self.accepted[slots] = 0
            self.incarnation[slots] = 0
            self.position[slots] = 0
            self.mass[slots] = 0
            self.counts[slots] = 0

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                zero()
        else:
            zero()

    def invalidate(self, slots) -> None:
        """Invalidate the references of the given slots without freeing them."""
        if len(slots) == 0:
            return
        slots = torch.as_tensor(slots, dtype=torch.int64, device=self.device)
        for slot in slots.tolist():
            for layer in range(self.counts.shape[1]):
                self._counts_host.pop((slot, layer), None)
        self.valid[slots] = False
        self.mass[slots] = 0
        self.output[slots] = 0
        self.contribution[slots] = 0
        self.selected[slots] = 0
        self.counts[slots] = 0
