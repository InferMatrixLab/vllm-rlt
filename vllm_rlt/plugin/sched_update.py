"""Output side of the loop scheduler: advancing spans when a step returns."""

from __future__ import annotations

from dataclasses import replace

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_rlt.plugin.program import Op


class UpdateMixin:
    def _advance(self, request, span, exited: bool) -> bool:
        """Move a span to the op after its current one. Returns True when the
        span is complete (its final op was the one that just ran)."""
        last = self.layout.limit(request.exit_params) - 1
        nxt = self.layout.next_op(span.op, span.depth, last, exited)
        if nxt is None:
            return True
        if span.op == Op.CORE and exited:
            span.exit_depth = span.depth
        span.op, span.depth = nxt
        return False

    @staticmethod
    def _without(out: SchedulerOutput, req_ids) -> SchedulerOutput:
        """``out`` with some requests removed from what vLLM will process."""
        drop = set(req_ids)
        kept = {r: n for r, n in out.num_scheduled_tokens.items() if r not in drop}
        spec = {r: t for r, t in out.scheduled_spec_decode_tokens.items() if r not in drop}
        return replace(
            out,
            num_scheduled_tokens=kept,
            total_num_scheduled_tokens=sum(kept.values()),
            scheduled_spec_decode_tokens=spec,
        )

    def update_from_output(self, scheduler_output: SchedulerOutput, model_runner_output):
        step = getattr(scheduler_output, "loop_step", None)
        if step is None or step.is_dummy or not step.req_ids:
            return super().update_from_output(scheduler_output, model_runner_output)
        exit_depths = getattr(model_runner_output, "exit_depths", None) or {}
        skipped = set(getattr(model_runner_output, "skipped", None) or ())
        final = step.is_final
        for rid, sid, advanced in zip(step.req_ids, step.spans, step.advanced):
            request = self.requests.get(rid)
            if request is None or request.is_finished():
                self._carry.drop(rid)
                continue
            span = request.get_span(sid)
            if span is None:  # preempted while the step was in flight
                skipped.add(rid)
                continue
            if advanced and rid not in skipped:
                # Async: the span moved on when this step was emitted and a later
                # step for it may be in flight now; that step owns ``in_flight``.
                continue
            span.in_flight = False
            if rid in skipped:
                if advanced:
                    span.op, span.depth = step.op, step.depths[step.req_ids.index(rid)]
                continue  # not run this step: stays on the same op
            if final:
                # vLLM's update_from_output (below) consumes the sampled token.
                depth = span.exit_depth
                limit = self.layout.limit(request.exit_params) - 1
                request.exit_depths.append(limit if depth is None else depth)
                request.end_span(sid)
            elif not advanced:
                self._advance(request, span, rid in exit_depths)
        if not final:
            return {}
        if skipped:
            scheduler_output = self._without(scheduler_output, skipped)
            if not scheduler_output.num_scheduled_tokens:
                return {}
        return super().update_from_output(scheduler_output, model_runner_output)
