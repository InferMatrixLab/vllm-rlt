"""Token level of the loop scheduler: opening spans through vLLM's schedule().

vLLM's ``schedule()`` keeps doing admission, KV allocation, preemption and
token accounting. It hands out token spans; this module turns each into a loop
span that the loop level then walks through PRELUDE, CORE and CODA.

A span is one decode token, a prompt chunk, or a self-speculative round (the
last sampled token plus K draft tokens, which vLLM already counts as one span
of K+1 tokens). With WCPB a prompt that still has chunks left stays visible to
vLLM, so its next chunk is opened while the earlier ones are mid-loop.
"""

from __future__ import annotations

from vllm.v1.core.sched.output import SchedulerOutput


class OpenMixin:
    def _stageable(self, request) -> bool:
        """WCPB: a prompt with chunks left may open another span now."""
        return (
            self.wcpb
            and request.in_loop
            and len(request.spans) < self.wcpb_depth
            and request.num_computed_tokens < request.num_prompt_tokens
        )

    def _needs_token_level(self) -> bool:
        """Whether vLLM's schedule() has something to do right now."""
        if self.waiting or self.kv_holding_waiting or self.async_kv_waiting:
            return True
        # Finished ids are only flushed by schedule(); until they are, vLLM's
        # has_requests() stays true and the engine would spin on empty steps.
        if self.finished_req_ids or self.has_finished_requests():
            return True
        return any(not r.in_loop or self._stageable(r) for r in self.running)

    def _open_spans_via_vllm(self, throttle_prefills: bool) -> SchedulerOutput:
        """Run vLLM's schedule() with mid-loop requests hidden from it, so a
        span already in flight is not scheduled a second time. vLLM counts
        ``len(self.running)`` against its limits, so they shrink by the number
        hidden; otherwise it would admit more than ``max_num_seqs``."""
        hidden = [r for r in self.running if r.in_loop and not self._stageable(r)]
        saved = (self.max_num_running_reqs, self.max_num_active_reqs)
        if hidden:
            ids = {id(r) for r in hidden}
            self.running = [r for r in self.running if id(r) not in ids]
            self.max_num_running_reqs = max(0, saved[0] - len(hidden))
            self.max_num_active_reqs = max(0, saved[1] - len(hidden))
        try:
            return super().schedule(throttle_prefills)
        finally:
            # Hidden requests are never preempted by this call, so they all
            # come back, in front, with their relative order unchanged.
            self.max_num_running_reqs, self.max_num_active_reqs = saved
            if hidden:
                self.running = hidden + self.running

    def _adopt_spans(self, out: SchedulerOutput) -> None:
        """Start a loop span for every span vLLM just scheduled."""
        for rid in out.preempted_req_ids or ():
            request = self.requests.get(rid)
            if request is not None:
                request.end_span()  # its KV is gone; it restarts from its prompt
        # What the worker is told with every step of the span is what vLLM
        # reported at schedule time, before it advanced its own counters.
        info = {r.req_id: (r.num_computed_tokens, 0) for r in out.scheduled_new_reqs}
        cached = out.scheduled_cached_reqs
        for rid, computed, outputs in zip(
            cached.req_ids, cached.num_computed_tokens, cached.num_output_tokens
        ):
            info[rid] = (computed, outputs)
        first = self.layout.first()
        drafts = out.scheduled_spec_decode_tokens
        for rid, num_tokens in out.num_scheduled_tokens.items():
            start, outputs = info[rid]
            request = self.requests[rid]
            # is_prefill_chunk was set by vLLM's _update_after_schedule: True
            # when the span does not reach a position that samples a token.
            request.begin_span(
                num_tokens,
                first,
                start,
                outputs,
                not request.is_prefill_chunk,
                draft=len(drafts.get(rid, ())),
            )
