"""Loop level of the loop scheduler: choosing and building the next loop step."""

from __future__ import annotations

from vllm.v1.core.sched.output import CachedRequestData

from vllm_rlt.plugin.output import LoopSchedulerOutput
from vllm_rlt.plugin.parallel import choose_op
from vllm_rlt.plugin.plan import Ready, candidates, group_key, unblocked
from vllm_rlt.plugin.program import LoopStep, Op


class EmitMixin:
    def _ready(self) -> list[Ready]:
        rows = []
        rank = lambda s: self.layout.rank(s.op, s.depth)  # noqa: E731
        for r in self.running:
            for s in unblocked(r.spans, rank):
                rows.append(Ready(r.request_id, s.op, s.depth, s.tokens, s.waited, s.sid))
        return rows

    def _choose(self, ready: list[Ready]):
        """This rank's own preferred (op, depth), or None.

        Under data parallelism the *workers* agree on the op between ranks (see
        ``parallel.py``); the scheduler only offers its candidates with the step
        (``candidates``). A rank whose preference loses runs a dummy of the
        agreed op and reports its rows skipped, so they stay where they were."""
        local = candidates(ready, self.allow_mixed_depth, self.wait_limit)
        self._candidates = local
        return choose_op([local])

    def _build_step(self, key, rows: list[Ready]) -> LoopSchedulerOutput:
        op = Op(key[0])
        reqs = [self.requests[r.req_id] for r in rows]
        spans = [q.get_span(r.sid) for q, r in zip(reqs, rows)]
        # The gate applies to a single decoded token. A prompt chunk must run
        # every pass (later tokens attend to its KV at all depths), and so must a
        # speculative round (its tokens are verified at full depth).
        gated = [
            op == Op.CORE
            and self.layout.has_gate
            and q.exit_params.uses_gate
            and s.tokens == 1
            and s.samples
            for q, s in zip(reqs, spans)
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
            spans=tuple(r.sid for r in rows),
        )
        out = self._make_output(step, reqs, rows, spans)
        for q, s, a in zip(reqs, spans, advance):
            s.in_flight = not a
            if a:
                self._advance(q, s, exited=False)
        picked = {(r.req_id, r.sid) for r in rows}
        for q in self.running:
            for s in q.spans:
                if (q.request_id, s.sid) not in picked and not s.in_flight:
                    s.waited += 1
        for s in spans:
            s.waited = 0
        return out

    def _make_output(self, step: LoopStep, reqs, rows=(), spans=()) -> LoopSchedulerOutput:
        out = LoopSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={r.request_id: n for r, n in zip(reqs, step.num_tokens)},
            total_num_scheduled_tokens=step.total_tokens,
            # Draft ids live on the worker (it wrote them); the scheduler only
            # says how many each row carries, which is all vLLM's accounting uses.
            scheduled_spec_decode_tokens={
                q.request_id: [-1] * s.draft for q, s in zip(reqs, spans) if s.draft
            },
            scheduled_encoder_inputs={},
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
        announced = self._carry.drain_into(out)
        cached = out.scheduled_cached_reqs
        for q, s in zip(reqs, spans):
            if q.request_id in announced:
                continue
            cached.req_ids.append(q.request_id)
            cached.new_token_ids.append([])
            cached.new_block_ids.append(self._carry.take_blocks(q.request_id))
            cached.num_computed_tokens.append(s.start)
            cached.num_output_tokens.append(s.num_output)
        out.loop_step = step
        # The runner proposes this many drafts after a final op (vLLM's own
        # dynamic-K field, set when it opened the spans).
        out.num_spec_tokens_to_schedule = getattr(self, "_spec_to_schedule", 0)
        out.span_starts = {q.request_id: s.start for q, s in zip(reqs, spans)}
        out.candidates = dict(getattr(self, "_candidates", {}))
        out.step_key = group_key(rows[0], self.allow_mixed_depth) if rows else None
        out.exit_params = {
            q.request_id: (
                q.exit_params.threshold,
                q.exit_params.min_loops,
                self.layout.limit(q.exit_params),
            )
            for q, g in zip(reqs, step.can_exit)
            if g
        }
        return out

    def _empty_output(self) -> LoopSchedulerOutput:
        out = LoopSchedulerOutput.make_empty()
        self._carry.drain_into(out)
        return out
