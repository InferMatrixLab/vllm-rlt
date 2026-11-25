"""Output side of the loop scheduler: advancing spans when a step returns."""

from __future__ import annotations

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_rlt.plugin.program import Op


class UpdateMixin:
    def _advance(self, request, exited: bool) -> bool:
        """Move a request to the op after its current one. Returns True when
        the span is complete (its final op was the one that just ran)."""
        last = request.exit_params.limit(self.layout.num_passes) - 1
        nxt = self.layout.next_op(request.loop_op, request.loop_depth, last, exited)
        if nxt is None:
            return True
        if request.loop_op == Op.CORE and exited:
            request.pending_exit_depth = request.loop_depth
        request.loop_op, request.loop_depth = nxt
        return False

    def update_from_output(self, scheduler_output: SchedulerOutput, model_runner_output):
        step = getattr(scheduler_output, "loop_step", None)
        if step is None:  # not one of ours (e.g. an empty step)
            return super().update_from_output(scheduler_output, model_runner_output)
        exit_depths = getattr(model_runner_output, "exit_depths", None) or {}
        final = step.op in (Op.CODA, Op.FULL)
        for rid, advanced in zip(step.req_ids, step.advanced):
            request = self.requests.get(rid)
            if request is None or request.is_finished():
                self._carry.drop(rid)
                continue
            request.in_flight = False
            if final:
                # vLLM's update_from_output (below) consumes the sampled token.
                depth = request.pending_exit_depth
                request.exit_depths.append(
                    request.exit_params.limit(self.layout.num_passes) - 1
                    if depth is None
                    else depth
                )
                request.end_span()
            elif not advanced:
                self._advance(request, rid in exit_depths)
        if not final:
            return {}
        return super().update_from_output(scheduler_output, model_runner_output)
