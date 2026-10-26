"""Lossless suspension with resident KV until capacity needs CPU offload.

Scheduler owns victim selection and queue transitions. A saved state with
``kv=None`` retains its existing cache allocation; hidden/token copies own their
storage so the runner slot can be reused. Capacity reclamation can later convert
that same saved state to a CPU snapshot without another logical suspension.
Sampling RNG remains in the runner registry throughout either path.
"""

import torch


class PreemptionManager:
    def __init__(self, engine):
        self.engine = engine
        self.snapshots = {}
        self.preemptions = self.resumptions = 0

    def discard_snapshot(self, request_id: str) -> None:
        """Discard saved state; normal request cleanup releases any resident KV."""
        self.snapshots.pop(request_id, None)

    def suspend(self, victim, *, offload_kv=True):
        """Release a runner slot, optionally reclaiming KV capacity as well.

        An already suspended resident request may be offloaded under later KV
        pressure. Its saved work and RNG must survive that conversion.
        """
        e = self.engine
        cache = e.cache_manager
        previous = self.snapshots.get(victim.request_id)
        if previous is None:
            e.model_runner.synchronize()
            work = e.scheduler.pending_work(victim)
            if work is None:
                raise ValueError("a preemption victim must have pending work")
            work = e._rewind_split_prefill(victim, work)
            hidden, token = victim.hidden_state, victim.input_token_tensor
        else:
            work, hidden, token = previous["work"], previous["hidden"], previous["token"]

        device = torch.device("cpu") if offload_kv else cache.device
        # Build a new record before releasing resources. In particular, a failed
        # offload must leave the prior resident state available for resumption.
        snapshot = dict(
            work=work,
            kv=cache.snapshot(victim.request_id) if offload_kv else None,
            hidden=None if hidden is None else hidden.to(device).clone(),
            token=None if token is None else token.to(device).clone(),
        )
        if not offload_kv and cache.device.type == "cuda":
            # Clones can borrow runner-owned rows. Finish reading them before
            # another stream reuses the released runner slot.
            torch.cuda.current_stream(cache.device).synchronize()
        if previous is None:
            e.model_runner.suspend(victim.request_id)
            victim.hidden_state = victim.input_token_tensor = None
            self.preemptions += 1
        if offload_kv:
            cache.poll_prefixes()
            cache.free(victim.request_id)
        self.snapshots[victim.request_id] = snapshot

    def resume(self, request):
        """Return saved work, None without a snapshot, or False when blocked."""
        state = self.snapshots.get(request.request_id)
        if state is None:
            return None
        cache = self.engine.cache_manager
        kv = state["kv"]
        if kv is not None:
            frontier = min(kv.max_tokens, kv.pages * cache.block_size)
            if not cache.allocate(request.request_id, kv.max_tokens, initial_tokens=frontier):
                return False
            cache.restore(request.request_id, kv)
        request.hidden_state = None if state["hidden"] is None else state["hidden"].to(cache.device)
        request.input_token_tensor = (
            None if state["token"] is None else state["token"].to(cache.device)
        )
        # Resident copies completed before suspension released their runner slot.
        # Restored CPU buffers must be visible to every stream before resumption.
        if kv is not None and cache.device.type == "cuda":
            torch.cuda.current_stream(cache.device).synchronize()
        del self.snapshots[request.request_id]
        self.resumptions += 1
        return state["work"]
