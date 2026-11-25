"""LoopScheduler: loop-level scheduling on top of vLLM's token-level Scheduler.

Two levels. vLLM's own ``schedule()`` stays responsible for admission, KV
allocation, preemption and token accounting; it hands out token spans. This
class expands each span into loop steps (PRELUDE, CORE depth 0..N-1, CODA) and
emits one step per ``schedule()`` call: mapping A, one EngineCore step is one
loop step. A step holds a single op kind, so it is never partly sampled.

How vLLM stays consistent:

* A request whose span is mid-loop is taken out of ``self.running`` while
  vLLM's ``schedule()`` runs, and its limits shrink by the same amount.
* ``_update_after_schedule`` is not overridden: vLLM advances
  ``num_computed_tokens`` when it opens the span, which is when the span's KV
  slots were allocated.
* ``update_from_output`` reaches vLLM only when a span's last op has returned.
* Self-speculation is vLLM's own speculative decoding (see ``selfspec.py``): a
  verify round is a span of K+1 tokens, so vLLM's draft-token placeholders,
  rollback and async accounting apply unchanged.
* WCPB (``rlt_wcpb``): a prompt with chunks left stays visible to vLLM, so the
  next chunk opens while earlier chunks are still mid-loop; the loop level then
  runs chunk c at depth d once chunk c-1 is past depth d.

Models that vLLM already runs natively (``layout.full_only``) are passed
through untouched, so with loop features off the baseline is vLLM itself.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_rlt.plugin.carry import Carry
from vllm_rlt.plugin.models import loop_layout
from vllm_rlt.plugin.output import LoopSchedulerOutput
from vllm_rlt.plugin.plan import take_rows
from vllm_rlt.plugin.sched_emit import EmitMixin
from vllm_rlt.plugin.sched_open import OpenMixin
from vllm_rlt.plugin.sched_update import UpdateMixin

logger = init_logger(__name__)
_TRACE = bool(os.environ.get("RLT_TRACE"))


class LoopScheduler(UpdateMixin, EmitMixin, OpenMixin, Scheduler):
    ASYNC = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layout = self._resolve_layout()
        extra = self.vllm_config.additional_config
        extra = extra if isinstance(extra, dict) else {}
        # A CORE step mixes rows of one depth only unless this is on. Rows carry
        # their depth; the runner and the model must support mixed depths
        # (Ouro does; Inkling's AttnRes state does not).
        self.wait_limit = int(extra.get("rlt_wait_limit", 8))
        self.max_rows = self.scheduler_config.max_num_seqs
        self.wcpb = bool(extra.get("rlt_wcpb", False)) and not self.layout.full_only
        # WCPB's gain is chunks at different depths sharing one CORE step, so it
        # turns mixed depth on wherever the model can run it (Ouro; not Inkling,
        # whose AttnRes bank differs per pass).
        self.allow_mixed_depth = self.layout.mixed_depth and (
            bool(extra.get("rlt_allow_mixed_depth", False)) or self.wcpb
        )
        if self.wcpb and self.layout.share_cache:
            raise ValueError("WCPB needs a KV cache per pass; it cannot share one (LAST_EXITED)")
        # Prelude, each core pass and coda: at most this many chunks of one prompt
        # can be in flight, one per program position.
        self.wcpb_depth = int(extra.get("rlt_wcpb_depth", self.layout.num_passes + 2))
        self._carry = Carry()
        self._spec_to_schedule = self.num_spec_tokens

    def _resolve_layout(self):
        return loop_layout(self.vllm_config)

    def schedule(self, throttle_prefills: bool = False) -> LoopSchedulerOutput:
        if self.layout.full_only:
            return super().schedule(throttle_prefills)
        if self._needs_token_level():
            out = self._open_spans_via_vllm(throttle_prefills)
            self._spec_to_schedule = out.num_spec_tokens_to_schedule
            if _TRACE:
                logger.warning(
                    "rlt open: spec_to_schedule=%s scheduled=%s drafts=%s waiting_spec=%s",
                    out.num_spec_tokens_to_schedule,
                    out.num_scheduled_tokens,
                    {k: len(v) for k, v in out.scheduled_spec_decode_tokens.items()},
                    {r.request_id: len(r.spec_token_ids) for r in self.running},
                )
            self._carry.absorb(out)
            self._adopt_spans(out)
        ready = self._ready()
        key = self._choose(ready)
        if key is None:
            return self._empty_output()
        rows = take_rows(
            ready, key, self.max_rows, self.max_num_scheduled_tokens, self.allow_mixed_depth
        )
        return self._build_step(key, rows)

    def finish_requests(self, request_ids, finished_status):
        finished = super().finish_requests(request_ids, finished_status)
        for request in finished:
            self._carry.drop(request.request_id)
            request.end_span()
        return finished


class LoopAsyncScheduler(LoopScheduler, AsyncScheduler):
    """Adds vLLM's output-placeholder accounting (asynchronous scheduling).

    Composition only, as in vllm-omni's OmniARAsyncScheduler. MRO:
    LoopAsyncScheduler, LoopScheduler, (mixins), AsyncScheduler, Scheduler.
    """

    ASYNC = True
