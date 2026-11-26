"""Loop model runner: vLLM's GPUModelRunner plus a loop program.

Everything that is not about the loop is inherited: persistent request state,
block tables, attention metadata, the sampler, the rejection sampler and the
speculator hook. The runner adds:

* ``execute_model`` reads ``scheduler_output.loop_step`` and tells the model
  which op and pass to run (the model's ``loop_ctx``).
* Hidden states stay on the device between ops, keyed by (request, span). Rows
  inside a step follow vLLM's *sorted* batch order, so state is split and merged
  in that order.
* With WCPB several chunks of one prompt are in flight, so each step first sets
  the row's ``num_computed_tokens`` to the span's start (``span_starts``).
* Data-parallel ranks agree on the op first; a rank whose step differs runs a
  dummy of the agreed op and reports its rows as skipped.
* After a CORE step the exit gate is evaluated for rows that may exit
  (``runner_exit.py``), and a token that exits has its skipped passes' KV filled.
* With ``rlt_graphs`` the decode steps of every op replay per-op CUDA graphs
  (``loopgraph.py``).

NOT RUN ON A GPU. Specific assumptions that a real run must confirm are listed
in ``docs/vllm-plugin.md``.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_runner import GPUModelRunner, sort_batch_req_ids

from vllm_rlt.plugin.context import LoopContext, state_cat, state_like, state_slice
from vllm_rlt.plugin.models import loop_layout
from vllm_rlt.plugin.output import LoopModelRunnerOutput
from vllm_rlt.plugin.parallel import NoAlign
from vllm_rlt.plugin.program import LoopStep, Op
from vllm_rlt.plugin.runner_exit import ExitMixin

logger = init_logger(__name__)
_TRACE = bool(os.environ.get("RLT_TRACE"))


class LoopModelRunner(ExitMixin, GPUModelRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loop_state: dict[tuple[str, int], object] = {}  # (req, span) -> state
        self.loop_remaining: dict[str, float] = {}  # req_id -> P(still running)
        self.aligner = NoAlign()  # replaced by the worker when data parallel
        self.loop_layout = loop_layout(self.vllm_config)
        extra = self.vllm_config.additional_config
        extra = extra if isinstance(extra, dict) else {}
        self.loop_graphs_requested = bool(extra.get("rlt_graphs", False))
        self.loop_graphs = None
        self._step = None
        self._skipped: list[str] = []
        self._exit_depths: dict[str, int] = {}
        if self.speculator is not None and hasattr(self.speculator, "bind"):
            self._speculator_pending = True
        else:
            self._speculator_pending = False

    # -- entry points ----------------------------------------------------------

    def execute_model(self, scheduler_output, *args, **kwargs):
        if kwargs.get("dummy_run"):
            # vLLM's own dummy run (profiling, warmup, and ours below) is a
            # nested call: it must not touch the bookkeeping of the loop step.
            return super().execute_model(scheduler_output, *args, **kwargs)
        step = getattr(scheduler_output, "loop_step", None)
        self._step, self._skipped = step, []
        if step is None:
            return super().execute_model(scheduler_output, *args, **kwargs)
        step = self._align(scheduler_output, step)
        self._step = step
        if step.is_dummy:
            return self._run_dummy(step)
        return self._run_step(scheduler_output, step, args, kwargs)

    def sample_tokens(self, grammar_output):
        step = self._step
        if step is None:
            return super().sample_tokens(grammar_output)
        if step.is_dummy:
            self.execute_model_state = None
            return LoopModelRunnerOutput(req_ids=[], skipped=self._skipped)
        if not step.is_final:
            self.execute_model_state = None  # non-final ops yield state only
            out = LoopModelRunnerOutput(req_ids=list(step.req_ids))
            out.exit_depths, self._exit_depths = self._exit_depths, {}
            return out
        out = super().sample_tokens(grammar_output)
        if out is not None:
            out.skipped = self._skipped
        return out

    def take_draft_token_ids(self):
        # The engine asks after every step; only a final op proposes drafts.
        # After any other op the handler still holds the previous round's
        # drafts, which must not be handed out again.
        step = self._step
        if step is not None and (step.is_dummy or not step.is_final):
            return None
        out = super().take_draft_token_ids()
        if _TRACE and out is not None:
            logger.warning(
                "rlt take drafts %s", dict(zip(out.req_ids, map(len, out.draft_token_ids)))
            )
        return out

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if self._speculator_pending:
            self.speculator.bind(self)
        return result

    def capture_model(self, *args, **kwargs) -> int:
        used = super().capture_model(*args, **kwargs)
        if self.loop_graphs_requested:
            from vllm_rlt.plugin.loopgraph import LoopGraphs

            self.loop_graphs = LoopGraphs(self)
            used += self.loop_graphs.capture()
        return used

    # -- data parallel ---------------------------------------------------------

    def _align(self, scheduler_output, step):
        """Agree with the other DP ranks on one op; otherwise run a dummy of it.

        A dummy step (an idle rank) was already aligned by the worker, so it must
        not align again: every rank takes part in exactly one alignment per step.
        """
        if step.is_dummy or isinstance(self.aligner, NoAlign):
            return step
        chosen = self.aligner.align(scheduler_output.candidates)
        if chosen is None or chosen == scheduler_output.step_key:
            return step
        self._skipped = list(step.req_ids)
        return LoopStep.dummy(Op(chosen[0]), max(chosen[1], 0), rows=max(1, step.total_tokens))

    # -- one loop step ---------------------------------------------------------

    def _run_dummy(self, step):
        """Run vLLM's dummy batch with the loop context set, so an idle rank
        goes through the same layers (and collectives) as the aligned op. The
        model builds whatever input state the op needs from zeros."""
        model = self.get_model()
        model.loop_ctx = LoopContext(step.op, step.depths[0], None)
        try:
            return self._dummy_run(step.dummy_rows, uniform_decode=True)
        finally:
            model.loop_ctx = None

    def _run_step(self, scheduler_output, step, args, kwargs):
        model = self.get_model()
        self._set_span_positions(scheduler_output, step)
        order = sort_batch_req_ids(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.scheduled_spec_decode_tokens,
            self.decode_query_len,
        )
        sids = dict(zip(step.req_ids, step.spans))
        ctx = self._make_context(scheduler_output, step, order, sids)
        model.loop_ctx = ctx
        graphs = self.cudagraph_manager
        use_graph = self._use_graph(ctx, step)
        if _TRACE:
            logger.warning(
                "rlt step op=%s depths=%s tokens=%s graph=%s rows=%s",
                step.op.name,
                step.depths,
                step.num_tokens,
                use_graph,
                [
                    (r[:3], sid, getattr(scheduler_output, "span_starts", {}).get(r))
                    for r, sid in zip(step.req_ids, step.spans)
                ],
            )
        captured = getattr(graphs, "_graphs_captured", None)
        if use_graph:
            self.loop_graphs.activate(ctx)
            self.loop_graphs.stage(ctx, step.total_tokens)
        elif captured is not None:
            graphs._graphs_captured = False  # eager: no graph was captured for this
        try:
            result = super().execute_model(scheduler_output, *args, **kwargs)
            if use_graph and ctx.out is None:
                # The model did not run in Python: a graph replayed, and the state
                # the op produced is in the graph's static output buffer. Read it
                # while that op's manager is still the active one.
                ctx.out = self.loop_graphs.output(ctx, step.total_tokens)
            return result
        finally:
            if use_graph:
                self.loop_graphs.deactivate()
            elif captured is not None:
                graphs._graphs_captured = captured
            model.loop_ctx = None
            self._after_step(scheduler_output, step, order, ctx, sids)

    def _use_graph(self, ctx, step) -> bool:
        """Replay the op's graph for a uniform one-token-per-row decode step (the
        shape the graphs were captured for). Prompt chunks, speculative spans and
        mixed-depth steps run eagerly."""
        graphs = self.loop_graphs
        if graphs is None or not graphs.ready or ctx.mixed:
            return False
        return all(t == 1 for t in step.num_tokens)

    def _set_span_positions(self, scheduler_output, step) -> None:
        """Point each row's ``num_computed_tokens`` at its span's start. vLLM
        advances the counter when it samples, which a mid-loop op never does, and
        with WCPB the chunks of one prompt are not run in the counter's order."""
        starts = getattr(scheduler_output, "span_starts", None)
        if not starts:
            return
        index = self.req_states.req_id_to_index
        counts = self.req_states.num_computed_tokens_np
        gpu = self.req_states.num_computed_tokens
        for rid, start in starts.items():
            slot = index.get(rid)
            if slot is None:
                continue
            counts[slot] = start
            gpu.stage_write_elem(slot, start)
        gpu.apply_write()
        np.minimum(
            counts, self.req_states.prefill_len.np, out=self.req_states.num_computed_prefill_tokens
        )

    def _make_context(self, scheduler_output, step, order, sids) -> LoopContext:
        depth_of = dict(zip(step.req_ids, step.depths))
        counts = scheduler_output.num_scheduled_tokens
        state = None
        needs_state = (
            step.op == Op.CODA
            or any(d > 0 for d in step.depths)
            or (step.op == Op.CORE and self.loop_layout.has_prelude)
        )
        if needs_state:
            have = [self.loop_state.get((r, sids[r])) for r in order]
            known = next((p for p in have if p is not None), None)
            if known is not None:
                # A row with no carried state (depth 0 in a mixed step) keeps its
                # place with zeros; the model embeds it. Skipping it would shift
                # every later row onto another row's state.
                parts = [
                    p if p is not None else state_like(known, counts[r])
                    for r, p in zip(order, have)
                ]
                state = state_cat(parts)
        if step.uniform_depth:
            return LoopContext(step.op, step.depths[0], state)
        row_depth = torch.cat(
            [torch.full((counts[r],), depth_of[r], dtype=torch.int32) for r in order]
        ).to(self.device)
        groups: dict[int, list] = {}
        offset = 0
        for rid in order:
            groups.setdefault(depth_of[rid], []).append((offset, counts[rid]))
            offset += counts[rid]
        rows = {
            d: torch.cat([torch.arange(s, s + n) for s, n in spans]).to(self.device)
            for d, spans in groups.items()
        }
        return LoopContext(step.op, -1, state, row_depth, rows)

    def _after_step(self, scheduler_output, step, order, ctx, sids) -> None:
        if step.op == Op.CODA:
            for rid in step.req_ids:
                self.loop_state.pop((rid, sids[rid]), None)
            return
        if ctx.out is None:
            return
        counts = scheduler_output.num_scheduled_tokens
        start, offset = {}, 0
        for rid in order:
            start[rid] = offset
            self.loop_state[(rid, sids[rid])] = state_slice(ctx.out, offset, counts[rid])
            offset += counts[rid]
        if step.op == Op.CORE and any(step.can_exit):
            self._evaluate_exit(scheduler_output, step, start, counts, ctx)
