"""Lossless pressure preemption for recurrent-depth KV.

CPU snapshots retain exact KV planes, hidden state and request RNG ownership.
Full-depth replay of generated tokens is deliberately avoided: it changes RLT
semantics. Snapshots are bounded by the admitted request population.
"""

from dataclasses import replace

import torch

from vllm_rlt.request import Stage
from vllm_rlt.worker.loopcd import preemption_reference


class PreemptionManager:
    def __init__(self, engine):
        self.engine = engine
        self.snapshots = {}
        self.preemptions = self.resumptions = 0

    def discard_snapshot(self, request_id: str) -> None:
        """Discard saved CPU state when a request will not be resumed."""
        self.snapshots.pop(request_id, None)

    def _is_preemption_candidate(self, request, requester, *, priority_only, excluded):
        """Check safety before considering a victim's priority.

        Pending outputs, transfer leases and the current batch's selected
        requests must remain alive. priority_only additionally forbids evicting
        equal/higher priority work to admit a new arrival.
        """
        e = self.engine
        if (
            request is requester
            or request.stage in (Stage.WAITING, Stage.RECEIVING)
            or request.num_output_placeholders
            or request.request_id in excluded
        ):
            return False
        if e.cache_manager.has_transfer_lease(request.request_id):
            return False
        return not priority_only or (
            request.sampling_params.priority > requester.sampling_params.priority
        )

    def _select_preemption_victim(self, requester, *, priority_only, excluded):
        """Prefer lower priority, then a larger current-position KV footprint.

        Priority 10 loses to priority 0. The footprint tie-breaker uses
        required_blocks(position + 1), not exact reclaimable physical pages.
        Capacity-pressure preemption retains this ranking even in FCFS mode;
        priority_only restricts eligibility only for priority admission.
        """
        candidates = (
            request
            for request in self.engine.scheduler.requests.values()
            if self._is_preemption_candidate(
                request, requester, priority_only=priority_only, excluded=excluded
            )
        )
        return max(
            candidates,
            key=lambda request: (
                request.sampling_params.priority,
                self.engine.cache_manager.required_blocks(request.position + 1),
            ),
            default=None,
        )

    def preempt(self, requester, *, priority_only=False, excluded=frozenset()):
        """Suspend one other request, preserving its recurrent execution state.

        Returns False if no safe victim exists. True means the victim's device
        resources have been released and it has been moved to WAITING; the
        requester must still retry its own allocation.

        excluded names requests this call must not suspend. The Scheduler passes
        the batch it is currently assembling explicitly, rather than exposing
        that selection for the manager to read.
        """
        victim = self._select_preemption_victim(
            requester, priority_only=priority_only, excluded=excluded
        )
        if victim is None:
            return False
        e = self.engine
        e.model_runner.synchronize()
        cache = e.cache_manager
        reference = preemption_reference(
            victim, victim.stage, e.model_runner.loopcd_references.get(victim.request_id)
        )
        snapshot = dict(
            stage=victim.stage,
            loopcd_reference=None
            if reference is None
            else replace(reference, hidden=reference.hidden.cpu().clone()),
            kv=cache.snapshot(victim.request_id),
            hidden=None if victim.hidden_state is None else victim.hidden_state.cpu().clone(),
            token=None
            if victim.input_token_tensor is None
            else victim.input_token_tensor.cpu().clone(),
        )
        e.model_runner.release(victim.request_id)
        cache.poll_prefixes()
        cache.free(victim.request_id)
        for q in e.scheduler.queues.values():
            while victim.request_id in q:
                q.remove(victim.request_id)
        victim.hidden_state = victim.input_token_tensor = None
        self.snapshots[victim.request_id] = snapshot
        e.scheduler.enqueue(victim, Stage.WAITING)
        self.preemptions += 1
        return True

    def resume(self, request):
        """Restore pages/state and re-enqueue the saved stage.

        None means no snapshot; False means insufficient capacity; True means
        restoration and enqueue both succeeded. Scheduler translates this
        callback contract into a named ResumeResult.
        """
        state = self.snapshots.get(request.request_id)
        if state is None:
            return None
        e, cache = self.engine, self.engine.cache_manager
        reference = preemption_reference(request, state["stage"], state["loopcd_reference"])
        kv = state["kv"]
        frontier = min(kv.max_tokens, kv.pages * cache.block_size)
        if not cache.allocate(request.request_id, kv.max_tokens, initial_tokens=frontier):
            return False
        cache.restore(request.request_id, kv)
        request.hidden_state = None if state["hidden"] is None else state["hidden"].to(cache.device)
        request.input_token_tensor = (
            None if state["token"] is None else state["token"].to(cache.device)
        )
        if reference is not None:
            e.model_runner.loopcd_references[request.request_id] = replace(
                reference, hidden=reference.hidden.to(cache.device)
            )
        # Restored buffers become visible to every runner stream before resumption.
        if cache.device.type == "cuda":
            torch.cuda.current_stream(cache.device).synchronize()
        e.scheduler.enqueue(request, state["stage"])
        del self.snapshots[request.request_id]
        self.resumptions += 1
        return True
