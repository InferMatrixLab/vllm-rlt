"""Loop level of the loop scheduler: choosing and building the next loop step."""

from __future__ import annotations

from vllm.v1.core.sched.output import CachedRequestData

from vllm_rlt.plugin.output import LoopSchedulerOutput
from vllm_rlt.plugin.plan import Ready, candidates
from vllm_rlt.plugin.program import LoopStep, Op


class EmitMixin:
    def _ready(self) -> list[Ready]:
        rows = []
        for r in self.running:
            if r.in_loop and not r.in_flight:
                rows.append(Ready(r.request_id, r.loop_op, r.loop_depth, r.span_tokens, r.waited))
        return rows

    def _choose(self, ready: list[Ready]):
        """The (op, depth) every rank agrees to run next, or None."""
        local = candidates(ready, self.allow_mixed_depth, self.wait_limit)
        return self.aligner.align(local)

    def _build_step(self, key, rows: list[Ready]) -> LoopSchedulerOutput:
        op = Op(key[0])
        reqs = [self.requests[r.req_id] for r in rows]
        # The gate applies to a single decoded token. A prompt chunk must run
        # every pass: later tokens attend to its KV at all depths.
        gated = [
            op == Op.CORE
            and self.layout.has_gate
            and r.exit_params.uses_gate
            and r.span_tokens == 1
            and r.span_samples
            for r in reqs
        ]
        # Async rows with no gate can be advanced right now, because their next
        # op does not depend on anything this step returns. The final op never
        # is: the span only ends when its sampled token comes back.
        final = op in (Op.CODA, Op.FULL)
        advance = [self.ASYNC and not g and not final for g in gated]
        step = LoopStep(
            op=op,
            depths=tuple(r.depth for r in rows),
            req_ids=tuple(r.req_id for r in rows),
            num_tokens=tuple(r.tokens for r in rows),
            can_exit=tuple(gated),
            advanced=tuple(advance),
        )
        out = self._make_output(step, reqs)
        for r, a in zip(reqs, advance):
            r.in_flight = not a
            if a:
                self._advance(r, exited=False)
        picked = {r.req_id for r in rows}
        for r in self.running:
            if r.in_loop and r.request_id not in picked and not r.in_flight:
                r.waited += 1
        for r in reqs:
            r.waited = 0
        return out

    def _make_output(self, step: LoopStep, reqs) -> LoopSchedulerOutput:
        out = LoopSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={r.request_id: n for r, n in zip(reqs, step.num_tokens)},
            total_num_scheduled_tokens=step.total_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
        announced = self._carry.drain_into(out)
        cached = out.scheduled_cached_reqs
        for r in reqs:
            if r.request_id in announced:
                continue
            cached.req_ids.append(r.request_id)
            cached.new_token_ids.append([])
            cached.new_block_ids.append(self._carry.take_blocks(r.request_id))
            cached.num_computed_tokens.append(r.span_start)
            cached.num_output_tokens.append(r.span_num_output)
        out.loop_step = step
        out.exit_params = {
            r.request_id: (
                r.exit_params.threshold,
                r.exit_params.min_loops,
                r.exit_params.limit(self.layout.num_passes),
            )
            for r, g in zip(reqs, step.can_exit)
            if g
        }
        return out

    def _empty_output(self) -> LoopSchedulerOutput:
        out = LoopSchedulerOutput.make_empty()
        self._carry.drain_into(out)
        return out
