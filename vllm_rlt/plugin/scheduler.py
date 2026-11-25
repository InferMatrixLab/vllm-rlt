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

Models that vLLM already runs natively (``layout.full_only``) are passed
through untouched, so with loop features off the baseline is vLLM itself.
"""

from __future__ import annotations

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_rlt.plugin.carry import Carry
from vllm_rlt.plugin.models import loop_layout
from vllm_rlt.plugin.output import LoopSchedulerOutput
from vllm_rlt.plugin.plan import take_rows
from vllm_rlt.plugin.sched_emit import EmitMixin
from vllm_rlt.plugin.sched_open import OpenMixin
from vllm_rlt.plugin.sched_update import UpdateMixin


class LoopScheduler(UpdateMixin, EmitMixin, OpenMixin, Scheduler):
    ASYNC = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layout = self._resolve_layout()
        extra = self.vllm_config.additional_config
        extra = extra if isinstance(extra, dict) else {}
        # v0: a CORE step mixes rows of one depth only. Rows carry their depth
        # and the planner can mix; the runner and model reject mixed depths.
        self.allow_mixed_depth = bool(extra.get("rlt_allow_mixed_depth", False))
        self.wait_limit = int(extra.get("rlt_wait_limit", 8))
        self.max_rows = self.scheduler_config.max_num_seqs
        self._carry = Carry()

    def _resolve_layout(self):
        return loop_layout(self.vllm_config)

    def schedule(self, throttle_prefills: bool = False) -> LoopSchedulerOutput:
        if self.layout.full_only:
            return super().schedule(throttle_prefills)
        if self._needs_token_level():
            out = self._open_spans_via_vllm(throttle_prefills)
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
