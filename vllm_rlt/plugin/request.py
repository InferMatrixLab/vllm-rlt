"""LoopRequest: a vLLM Request that also tracks its position in the loop program.

vLLM only constructs requests in ``Request.from_engine_core_request`` and
every module imports the class by name, so the plugin replaces the class
everywhere (see ``patch.py``) instead of keeping a side table that would have
to follow the request through finish, preempt and resume.

Loop state is per *span*: one token for decode, a chunk for prefill, or K+1
tokens for a self-speculative round. vLLM advances ``num_computed_tokens`` when
it opens the span. A request has one span in flight, except under WCPB where a
prompt keeps up to ``num_passes`` chunk spans in flight at different depths
(``spans``); ``Span`` is the unit the loop scheduler moves through the program.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.request import Request

from vllm_rlt.plugin.exit import ExitParams
from vllm_rlt.plugin.program import Op


@dataclass
class Span:
    """A run of tokens walking the loop program together."""

    sid: int
    tokens: int
    start: int  # num_computed_tokens before the span
    op: Op
    depth: int = 0
    samples: bool = True  # False for a prompt chunk that does not sample
    num_output: int = 0  # output-token count the worker is told about
    waited: int = 0  # loop steps it was ready but not chosen
    in_flight: bool = False  # an emitted step for it has not returned
    exit_depth: int | None = None  # depth its token left at (gate or pass count)
    draft: int = 0  # self-spec: number of draft tokens at the span's tail


class LoopRequest(Request):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        extra = self.sampling_params.extra_args if self.sampling_params is not None else None
        self.exit_params = ExitParams.from_extra_args(extra)
        self.exit_depths: list[int] = []  # depth each finished token left at
        self.spans: list[Span] = []
        self._next_sid = 0

    # -- single-span view (the common case) -------------------------------------

    @property
    def span(self) -> Span | None:
        return self.spans[0] if self.spans else None

    @property
    def in_loop(self) -> bool:
        return bool(self.spans)

    @property
    def loop_op(self) -> Op | None:
        return self.spans[0].op if self.spans else None

    @property
    def loop_depth(self) -> int:
        return self.spans[0].depth if self.spans else 0

    @property
    def span_tokens(self) -> int:
        return self.spans[0].tokens if self.spans else 0

    @property
    def in_flight(self) -> bool:
        return any(s.in_flight for s in self.spans)

    @in_flight.setter
    def in_flight(self, value: bool) -> None:
        for s in self.spans:
            s.in_flight = value

    @property
    def waited(self) -> int:
        return max((s.waited for s in self.spans), default=0)

    @waited.setter
    def waited(self, value: int) -> None:
        for s in self.spans:
            s.waited = value

    # -- spans -------------------------------------------------------------------

    def begin_span(
        self,
        num_tokens: int,
        first: tuple[Op, int],
        start: int,
        num_output: int,
        samples: bool,
        *,
        draft: int = 0,
    ) -> Span:
        span = Span(
            self._next_sid,
            num_tokens,
            start,
            first[0],
            first[1],
            samples,
            num_output,
            draft=draft,
        )
        self._next_sid += 1
        self.spans.append(span)
        return span

    def get_span(self, sid: int) -> Span | None:
        return next((s for s in self.spans if s.sid == sid), None)

    def end_span(self, sid: int | None = None) -> None:
        """Drop one span (default: the oldest), or all of them when the request
        itself ends."""
        if sid is None:
            self.spans.clear()
        else:
            self.spans = [s for s in self.spans if s.sid != sid]
