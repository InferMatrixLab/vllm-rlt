"""LoopRequest: a vLLM Request that also tracks its position in the loop program.

vLLM only constructs requests in ``Request.from_engine_core_request`` and
every module imports the class by name, so the plugin replaces the class
everywhere (see ``patch.py``) instead of keeping a side table that would have
to follow the request through finish, preempt and resume.

Loop state is per span (one token for decode, a chunk for prefill). vLLM
advances ``num_computed_tokens`` when it opens the span; the span stays out of
vLLM's token-level scheduling until its last op returns.
"""

from __future__ import annotations

from vllm.v1.request import Request

from vllm_rlt.plugin.exit import ExitParams
from vllm_rlt.plugin.program import Op


class LoopRequest(Request):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        extra = self.sampling_params.extra_args if self.sampling_params is not None else None
        self.exit_params = ExitParams.from_extra_args(extra)
        self.exit_depths: list[int] = []  # depth each finished token left at
        self.loop_op: Op | None = None
        self.loop_depth = 0
        self.span_tokens = 0
        self.span_start = 0  # num_computed_tokens before the span
        self.span_num_output = 0  # output-token count the worker is told about
        self.span_samples = True  # False for a prompt chunk that does not sample
        self.waited = 0  # loop steps this request was ready but not chosen
        self.in_flight = False  # an emitted step for it has not returned
        self.pending_exit_depth: int | None = None

    @property
    def in_loop(self) -> bool:
        return self.loop_op is not None

    def begin_span(
        self, num_tokens: int, first: tuple[Op, int], start: int, num_output: int, samples: bool
    ) -> None:
        self.span_tokens = num_tokens
        self.span_start = start
        self.span_num_output = num_output
        self.span_samples = samples
        self.loop_op, self.loop_depth = first
        self.waited = 0
        self.in_flight = False
        self.pending_exit_depth = None

    def end_span(self) -> None:
        self.loop_op = None
        self.loop_depth = 0
        self.in_flight = False
