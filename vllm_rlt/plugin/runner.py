"""Loop model runner: vLLM's GPUModelRunner plus a loop program.

Everything that is not about the loop is inherited: persistent request state,
block tables, attention metadata, the sampler. The runner adds:

* ``execute_model`` reads ``scheduler_output.loop_step`` and tells the model
  which op and pass to run (the model's ``loop_ctx``).
* Hidden states stay on the device between ops, keyed by request. Rows inside a
  step follow vLLM's *sorted* batch order, so state is split and merged in that
  order.
* Data-parallel ranks agree on the op first; a rank whose step differs runs a
  dummy of the agreed op and reports its rows as skipped.
* After a CORE step the exit gate is evaluated for rows that may exit, and a
  token that exits has its skipped passes' KV filled.

NOT RUN ON A GPU. Specific assumptions that a real run must confirm are listed
in ``docs/vllm-plugin.md``. A loop step never replays a CUDA graph captured for
the whole model (graphs are switched off for the call).
"""

from __future__ import annotations

import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner, sort_batch_req_ids

from vllm_rlt.plugin.context import LoopContext, state_cat, state_slice
from vllm_rlt.plugin.exit import exit_mask
from vllm_rlt.plugin.kvfill import fill_skipped_passes
from vllm_rlt.plugin.models import loop_layout
from vllm_rlt.plugin.output import LoopModelRunnerOutput
from vllm_rlt.plugin.parallel import NoAlign
from vllm_rlt.plugin.program import LoopStep, Op


class LoopModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loop_state: dict[str, object] = {}  # req_id -> state between ops
        self.loop_remaining: dict[str, float] = {}  # req_id -> P(still running)
        self.aligner = NoAlign()  # replaced by the worker when data parallel
        self.loop_layout = loop_layout(self.vllm_config)
        self._step = None
        self._skipped: list[str] = []
        self._exit_depths: dict[str, int] = {}

    # -- entry points ----------------------------------------------------------

    def execute_model(self, scheduler_output, *args, **kwargs):
        if kwargs.get("dummy_run"):
            # vLLM's own dummy run (profiling, warmup, and ours below) is a
            # nested call: it must not touch the bookkeeping of the loop step.
            return super().execute_model(scheduler_output, *args, **kwargs)
        step = getattr(scheduler_output, "loop_step", None)
        self._step, self._skipped = step, []
        if step is None or step.op == Op.FULL:
            return super().execute_model(scheduler_output, *args, **kwargs)
        step = self._align(scheduler_output, step)
        self._step = step
        if step.is_dummy:
            return self._run_dummy(step)
        return self._run_step(scheduler_output, step, args, kwargs)

    def sample_tokens(self, grammar_output):
        step = self._step
        if step is None or step.op == Op.FULL:
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
        order = sort_batch_req_ids(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.scheduled_spec_decode_tokens,
            self.decode_query_len,
        )
        ctx = self._make_context(scheduler_output, step, order)
        model.loop_ctx = ctx
        graphs = self.cudagraph_manager
        captured = getattr(graphs, "_graphs_captured", None)
        if captured is not None:
            graphs._graphs_captured = False  # eager: loop ops are not what was captured
        try:
            return super().execute_model(scheduler_output, *args, **kwargs)
        finally:
            if captured is not None:
                graphs._graphs_captured = captured
            model.loop_ctx = None
            self._after_step(scheduler_output, step, order, ctx)

    def _make_context(self, scheduler_output, step, order) -> LoopContext:
        depth_of = dict(zip(step.req_ids, step.depths))
        counts = scheduler_output.num_scheduled_tokens
        state = None
        needs_state = (
            step.op == Op.CODA
            or any(d > 0 for d in step.depths)
            or (step.op == Op.CORE and self.loop_layout.has_prelude)
        )
        if needs_state:
            parts = [self.loop_state[r] for r in order if r in self.loop_state]
            state = state_cat(parts) if parts else None
        if step.uniform_depth:
            return LoopContext(step.op, step.depths[0], state)
        row_depth = torch.cat(
            [torch.full((counts[r],), depth_of[r], dtype=torch.int32) for r in order]
        ).to(self.device)
        groups: dict[int, torch.Tensor] = {}
        offset = 0
        for rid in order:
            groups.setdefault(depth_of[rid], []).append((offset, counts[rid]))
            offset += counts[rid]
        rows = {
            d: torch.cat([torch.arange(s, s + n) for s, n in spans]).to(self.device)
            for d, spans in groups.items()
        }
        return LoopContext(step.op, -1, state, row_depth, rows)

    def _after_step(self, scheduler_output, step, order, ctx) -> None:
        if step.op == Op.CODA:
            for rid in step.req_ids:
                self.loop_state.pop(rid, None)
            return
        if ctx.out is None:
            return
        counts = scheduler_output.num_scheduled_tokens
        start, offset = {}, 0
        for rid in order:
            start[rid] = offset
            self.loop_state[rid] = state_slice(ctx.out, offset, counts[rid])
            offset += counts[rid]
        if step.op == Op.CORE and any(step.can_exit):
            self._evaluate_exit(scheduler_output, step, start, counts, ctx)

    def _evaluate_exit(self, scheduler_output, step, start, counts, ctx) -> None:
        model = self.get_model()
        depth_of = dict(zip(step.req_ids, step.depths))
        rids = [r for r, g in zip(step.req_ids, step.can_exit) if g]
        rows = [start[r] + counts[r] - 1 for r in rids]
        hidden = ctx.out if isinstance(ctx.out, torch.Tensor) else ctx.out["hidden"]
        gate = model.loop_gate(hidden[rows])
        dev = gate.device
        rem = torch.tensor([self.loop_remaining.get(r, 1.0) for r in rids], device=dev)
        passes = torch.tensor([depth_of[r] + 1 for r in rids], device=dev)
        p = [scheduler_output.exit_params[r] for r in rids]
        thr, mn, lim = (torch.tensor([x[k] for x in p], device=dev) for k in range(3))
        done = exit_mask(gate, rem, passes, thr, mn, lim)
        state = self.execute_model_state
        for rid, left, d in zip(rids, rem.tolist(), done.tolist()):
            self.loop_remaining[rid] = left
            if not d:
                continue
            self._exit_depths[rid] = depth_of[rid]
            self.loop_remaining.pop(rid, None)
            i = state.input_batch.req_ids.index(rid)
            lo = int(state.input_batch.query_start_loc_np[i])
            hi = int(state.input_batch.query_start_loc_np[i + 1])
            fill_skipped_passes(
                model.loop_attention(),
                lambda name, lo=lo, hi=hi: state.slot_mappings_by_layer[name][lo:hi].long(),
                depth_of[rid],
            )
