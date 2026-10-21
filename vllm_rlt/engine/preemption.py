"""Lossless pressure preemption for recurrent-depth KV.

CPU snapshots retain exact KV planes and hidden state. The sampling RNG is not
part of the snapshot: ModelRunner.suspend keeps the request's registry slot, so
resuming continues the same random stream. Full-depth replay of generated tokens
is deliberately avoided: it changes RLT semantics. Snapshots are bounded by the
admitted request population.
"""

import torch


class PreemptionManager:
    def __init__(self, engine):
        self.engine = engine
        self.snapshots = {}
        self.preemptions = self.resumptions = 0

    def discard_snapshot(self, request_id: str) -> None:
        """Discard saved CPU state when a request will not be resumed."""
        self.snapshots.pop(request_id, None)

    def suspend(self, victim):
        """Save the selected request's resources; Scheduler owns its queue transition."""
        e = self.engine
        e.model_runner.synchronize()
        e._rewind_split_prefill(victim)
        cache = e.cache_manager
        snapshot = dict(
            phase=victim.phase,
            kv=cache.snapshot(victim.request_id),
            hidden=None if victim.hidden_state is None else victim.hidden_state.cpu().clone(),
            token=None
            if victim.input_token_tensor is None
            else victim.input_token_tensor.cpu().clone(),
        )
        # Suspend rather than release: the victim is not terminating, so its
        # sampling RNG must survive for the resumed sequence to be lossless.
        e.model_runner.suspend(victim.request_id)
        cache.poll_prefixes()
        cache.free(victim.request_id)
        victim.hidden_state = victim.input_token_tensor = None
        self.snapshots[victim.request_id] = snapshot
        self.preemptions += 1

    def resume(self, request):
        """Return the saved phase, None without a snapshot, or False when blocked."""
        state = self.snapshots.get(request.request_id)
        if state is None:
            return None
        cache = self.engine.cache_manager
        kv = state["kv"]
        frontier = min(kv.max_tokens, kv.pages * cache.block_size)
        if not cache.allocate(request.request_id, kv.max_tokens, initial_tokens=frontier):
            return False
        cache.restore(request.request_id, kv)
        request.hidden_state = None if state["hidden"] is None else state["hidden"].to(cache.device)
        request.input_token_tensor = (
            None if state["token"] is None else state["token"].to(cache.device)
        )
        # Restored buffers become visible to every runner stream before resumption.
        if cache.device.type == "cuda":
            torch.cuda.current_stream(cache.device).synchronize()
        del self.snapshots[request.request_id]
        self.resumptions += 1
        return state["phase"]
