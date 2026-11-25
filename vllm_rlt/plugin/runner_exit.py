"""Early-exit evaluation after a CORE step (Ouro's gate), as a runner mixin.

Runs on the device: the gate logit of each gated row's last token updates the
row's ``remaining`` probability and the exit mask is computed with one
vectorised call. The only host read is the mask itself, which the scheduler
needs to choose the row's next op. Exiting rows get their skipped passes' KV
filled from the pass they left at (LAST_EXITED).

Inkling has no gate, so none of this runs for it; its "exit" is the configured
pass count, which the scheduler's program already encodes.
"""

from __future__ import annotations

import torch

from vllm_rlt.plugin.exit import exit_mask
from vllm_rlt.plugin.kvfill import fill_skipped_passes


class ExitMixin:
    def _evaluate_exit(self, scheduler_output, step, start, counts, ctx) -> None:
        model = self.get_model()
        if not self.loop_layout.has_gate:
            return
        hidden = ctx.out
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(
                "the exit gate reads one hidden-state tensor; this model carries "
                f"{type(hidden).__name__} between ops"
            )
        depth_of = dict(zip(step.req_ids, step.depths))
        rids = [r for r, g in zip(step.req_ids, step.can_exit) if g]
        rows = torch.tensor([start[r] + counts[r] - 1 for r in rids], device=hidden.device)
        gate = model.loop_gate(hidden[rows])
        dev = gate.device
        rem = torch.tensor([self.loop_remaining.get(r, 1.0) for r in rids], device=dev)
        passes = torch.tensor([depth_of[r] + 1 for r in rids], device=dev)
        params = [scheduler_output.exit_params[r] for r in rids]
        thr, mn, lim = (torch.tensor([p[k] for p in params], device=dev) for k in range(3))
        done = exit_mask(gate, rem, passes, thr, mn, lim)
        state = self.execute_model_state
        for rid, left, d in zip(rids, rem.tolist(), done.tolist()):
            self.loop_remaining[rid] = left
            if not d:
                continue
            self._exit_depths[rid] = depth_of[rid]
            self.loop_remaining.pop(rid, None)
            if depth_of[rid] >= self.loop_layout.num_passes - 1:
                continue  # ran every pass: nothing to fill
            i = state.input_batch.req_ids.index(rid)
            lo = int(state.input_batch.query_start_loc_np[i])
            hi = int(state.input_batch.query_start_loc_np[i + 1])
            fill_skipped_passes(
                model.loop_attention(),
                lambda name, lo=lo, hi=hi: state.slot_mappings_by_layer[name][lo:hi].long(),
                depth_of[rid],
            )
