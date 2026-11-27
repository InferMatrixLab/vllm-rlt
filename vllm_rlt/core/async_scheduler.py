"""Asynchronous loop scheduling for operations with known successors."""

from vllm_rlt.core.loop import LoopOp
from vllm_rlt.core.loop_scheduler import LoopScheduler


class LoopAsyncScheduler(LoopScheduler):
    def _pending_coda(self, state):
        request = state.request
        if not self.layout.has_coda or self.self_speculation or len(state.spans) != 1:
            return None
        work = next(iter(state.spans.values()))
        if work.next_op is not None:
            return None
        if work.prefill:
            # A fixed prompt can supply its next input while CODA is in flight.
            if (
                self.async_prefill_lookahead
                and not self.wavefront_prefill
                and not self.decode_pipeline
                and not work.produces_token
                and not request.num_output_tokens
                and work.start + work.count < request.num_prompt_tokens
            ):
                return work
        elif (
            work.produces_token
            and request.num_output_placeholders == 1
            and request.num_output_tokens + request.num_output_placeholders < request.max_tokens
            and request.num_tokens + 1 < self.max_model_len
        ):
            return work
        return None

    def _on_submit(self, state, work, op, span):
        if span.sample:
            state.request.num_output_placeholders += 1
        if self.decode_pipeline:
            return super()._on_submit(state, work, op, span)
        if op is LoopOp.PRELUDE or (
            op is LoopOp.CORE
            and not span.check_exit
            and not span.last_core
            and not span.propose_draft
        ):
            self._advance(work, op, span, exited=False)
            return True
        if op is LoopOp.CORE and span.last_core and not span.check_exit and self.layout.has_coda:
            self._advance(work, op, span, exited=False)
            return True
        return super()._on_submit(state, work, op, span)

    def _resolve_output_placeholder(self, request):
        if request.num_output_placeholders:
            request.num_output_placeholders -= 1
