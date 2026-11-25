"""Token-level scheduler metadata that has not reached the runner yet.

vLLM's ``schedule()`` produces new-request data, newly allocated blocks,
finished ids and block copies exactly once. The loop scheduler calls it more
often than it emits a step that needs them, so what cannot be delivered right
away is held here. Everything except per-request new blocks is delivered with
the very next emitted output; new blocks wait for that request's first row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.core.sched.output import SchedulerOutput


@dataclass
class Carry:
    new_reqs: list = field(default_factory=list)
    finished: set = field(default_factory=set)
    preempted: set = field(default_factory=set)
    zero: list = field(default_factory=list)
    copies: list = field(default_factory=list)
    enc_free: list = field(default_factory=list)
    blocks: dict[str, list[list[int]]] = field(default_factory=dict)

    def absorb(self, out: SchedulerOutput) -> None:
        self.new_reqs.extend(out.scheduled_new_reqs)
        self.finished |= set(out.finished_req_ids)
        self.enc_free.extend(out.free_encoder_mm_hashes)
        if out.preempted_req_ids:
            self.preempted |= set(out.preempted_req_ids)
        if out.new_block_ids_to_zero:
            self.zero.extend(out.new_block_ids_to_zero)
        if out.kv_cache_block_copies:
            self.copies.extend(out.kv_cache_block_copies)
        cached = out.scheduled_cached_reqs
        for req_id, new_blocks in zip(cached.req_ids, cached.new_block_ids):
            if new_blocks is None:
                continue
            pending = self.blocks.setdefault(req_id, [[] for _ in new_blocks])
            for group, ids in enumerate(new_blocks):
                pending[group].extend(ids)

    def drain_into(self, out: SchedulerOutput) -> set[str]:
        """Move everything deliverable now into ``out``; returns the ids of the
        requests announced (their block ids are inside the new-request data)."""
        out.scheduled_new_reqs = self.new_reqs
        out.finished_req_ids = self.finished
        out.preempted_req_ids = self.preempted or None
        out.new_block_ids_to_zero = self.zero or None
        out.kv_cache_block_copies = self.copies or None
        out.free_encoder_mm_hashes = self.enc_free
        announced = {d.req_id for d in self.new_reqs}
        self.new_reqs, self.finished, self.preempted = [], set(), set()
        self.zero, self.copies, self.enc_free = [], [], []
        return announced

    def take_blocks(self, req_id: str):
        pending = self.blocks.pop(req_id, None)
        return None if pending is None else tuple(pending)

    def drop(self, req_id: str) -> None:
        """The request ended; forget blocks and undelivered announcements."""
        self.blocks.pop(req_id, None)
        self.new_reqs = [d for d in self.new_reqs if d.req_id != req_id]
