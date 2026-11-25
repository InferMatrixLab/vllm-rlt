"""Loop model runner: vLLM's GPUModelRunner plus a loop program.

Everything that is not about the loop is inherited: persistent request state,
block tables, attention metadata, the sampler. The runner adds:

* ``execute_model`` reads ``scheduler_output.loop_step`` and tells the model
  which op and pass to run (the model's ``loop_ctx``).
* Hidden states stay on the device between ops, keyed by request. Rows are in
  vLLM's *sorted* batch order inside a step, so they are split by that order.
* After a CORE step the exit gate is evaluated for rows that may exit; a token
  that exits has its skipped passes' KV filled.

Not verified on a GPU. v0 limits, each an error rather than a wrong answer:
CORE steps must be one depth; a loop step never replays a CUDA graph captured
for the whole model (graphs are switched off for the call).
"""

from __future__ import annotations

import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner, sort_batch_req_ids

from vllm_rlt.plugin.exit import exit_mask
from vllm_rlt.plugin.kvfill import fill_skipped_passes
from vllm_rlt.plugin.ouro import LoopContext
from vllm_rlt.plugin.output import LoopModelRunnerOutput
from vllm_rlt.plugin.program import Op


class LoopModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loop_hidden: dict[str, torch.Tensor] = {}  # req_id -> [tokens, hidden]
        self.loop_remaining: dict[str, float] = {}  # req_id -> P(still running)
        self._step = None
        self._exit_depths: dict[str, int] = {}

    def execute_model(self, scheduler_output, *args, **kwargs):
        step = getattr(scheduler_output, "loop_step", None)
        self._step = step
        if step is None or step.op == Op.FULL:
            return super().execute_model(scheduler_output, *args, **kwargs)
        model = self.get_model()
        if step.is_dummy:
            hidden = torch.zeros(
                step.dummy_rows, model.config.hidden_size, dtype=self.dtype, device=self.device
            )
            model.loop_ctx = LoopContext(step.op, step.depths[0], hidden)
            try:
                return self._dummy_run(step.dummy_rows, uniform_decode=True)
            finally:
                model.loop_ctx = None
        if not step.uniform_depth:
            raise NotImplementedError("mixed-depth CORE steps are not supported by the v0 runner")
        order = sort_batch_req_ids(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.scheduled_spec_decode_tokens,
            self.decode_query_len,
        )
        model.loop_ctx = LoopContext(step.op, step.depths[0], self._gather_hidden(step, order))
        graphs = self.cudagraph_manager
        captured = getattr(graphs, "_graphs_captured", None)
        if captured is not None:
            graphs._graphs_captured = False  # eager: loop ops are not what was captured
        try:
            return super().execute_model(scheduler_output, *args, **kwargs)
        finally:
            if captured is not None:
                graphs._graphs_captured = captured
            ctx, model.loop_ctx = model.loop_ctx, None
            self._after_step(scheduler_output, step, order, ctx)

    def sample_tokens(self, grammar_output):
        step = self._step
        if step is not None and not step.is_final and not step.is_dummy:
            self.execute_model_state = None  # non-final ops yield hidden states only
            out = LoopModelRunnerOutput(req_ids=list(step.req_ids))
            out.exit_depths, self._exit_depths = self._exit_depths, {}
            return out
        return super().sample_tokens(grammar_output)

    def _gather_hidden(self, step, order):
        if step.op == Op.CORE and step.depths[0] == 0:
            return None  # depth 0 embeds the tokens itself
        return torch.cat([self.loop_hidden[r] for r in order])

    def _after_step(self, scheduler_output, step, order, ctx) -> None:
        if step.op == Op.CODA:
            for rid in step.req_ids:
                self.loop_hidden.pop(rid, None)
            return
        if ctx is None or ctx.out is None:
            return
        counts = scheduler_output.num_scheduled_tokens
        start = {}
        offset = 0
        for rid in order:
            start[rid] = offset
            self.loop_hidden[rid] = ctx.out[offset : offset + counts[rid]]
            offset += counts[rid]
        if step.op == Op.CORE and any(step.can_exit):
            self._evaluate_exit(scheduler_output, step, start, counts, ctx)

    def _evaluate_exit(self, scheduler_output, step, start, counts, ctx) -> None:
        model, depth = self.get_model(), step.depths[0]
        rids = [r for r, g in zip(step.req_ids, step.can_exit) if g]
        rows = [start[r] + counts[r] - 1 for r in rids]
        gate = model.loop_gate(ctx.out[rows])
        dev = gate.device
        rem = torch.tensor([self.loop_remaining.get(r, 1.0) for r in rids], device=dev)
        p = [scheduler_output.exit_params[r] for r in rids]
        thr, mn, lim = (torch.tensor([x[k] for x in p], device=dev) for k in range(3))
        done = exit_mask(gate, rem, depth + 1, thr, mn, lim)
        state = self.execute_model_state
        for rid, left, d in zip(rids, rem.tolist(), done.tolist()):
            self.loop_remaining[rid] = left
            if not d:
                continue
            self._exit_depths[rid] = depth
            self.loop_remaining.pop(rid, None)
            i = state.input_batch.req_ids.index(rid)
            lo = int(state.input_batch.query_start_loc_np[i])
            hi = int(state.input_batch.query_start_loc_np[i + 1])
            fill_skipped_passes(
                model.loop_attention(),
                lambda name: state.slot_mappings_by_layer[name][lo:hi].long(),
                depth,
            )
