"""Inkling split into loop ops, built on vLLM's own Inkling loop model.

vLLM's ``Inkling2LoopModel.forward`` runs prelude, ``num_passes`` core passes
and coda as one loop over ``visits``. This subclass runs the same visits one op
at a time, so the loop scheduler can put a different batch between ops. All
weights, layers and kernels are vLLM's; only the control flow is split, and the
state that the single loop kept in locals is carried between ops instead:

    stack           [slots, T, D]  completed AttnRes blocks and the running partial
    prelude_partial [T, D]         the prelude's partial block, restored every pass
    bank            [B, T, D]      the previous pass's blocks, the pass's "messages"
    inputs          [S, T, D]      the next visit's sublayer inputs (fp32 mixes)
    num_blocks      int            completed blocks in ``stack``

Token axes: stack 1, prelude_partial 0, bank 1, inputs 1. ``num_blocks`` is the
same for every token of a step (an op runs the same layers for all rows).

``passes`` is how many core passes the tokens of the step run. The last layer
of the last executed pass mixes its output for the CODA's first layer, not for
the next pass's, so the visit list is built for ``passes``, exactly as vLLM
builds it for ``num_loop_passes == passes``. A self-speculative draft runs with
fewer passes than a verify.

NOT RUN ON A GPU. It reuses vLLM's private ``_run_layer``; if that signature
changes the plugin fails at the first call. The equivalence with vLLM's single
forward (same hidden state out) is what a GPU run must confirm first.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.models.inkling2.nvidia.loop import (
    Inkling2LoopForConditionalGeneration,
    Inkling2LoopModel,
)

from vllm_rlt.plugin.context import LoopContext, TokenState, pad_state
from vllm_rlt.plugin.program import Op

DIMS = {"stack": 1, "prelude_partial": 0, "bank": 1, "inputs": 1, "prelude_dests": 1}


class SplitInkling2LoopModel(Inkling2LoopModel):
    """``forward_op(op, depth, positions, embeds, state, passes)``."""

    def _visits(self, passes: int | None = None) -> list[tuple[int, int]]:
        core = self.loop_core
        passes = passes or self.config.num_loop_passes
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        return visits

    def _op_range(self, op: Op, depth: int, passes: int | None = None) -> range:
        """Indices into ``_visits(passes)`` that one op executes."""
        passes = passes or self.config.num_loop_passes
        n_pre = self.loop_core.start
        n_core = len(self.loop_core)
        if op == Op.PRELUDE:
            return range(0, n_pre)
        if op == Op.CORE:
            return range(n_pre + depth * n_core, n_pre + (depth + 1) * n_core)
        coda_start = n_pre + passes * n_core
        return range(coda_start, coda_start + len(self.layers) - self.loop_core.stop)

    def _next_query(self, visits, visit):
        if visit + 1 == len(visits):
            return self.attn_res.output_query, self.norm
        nxt, nxt_pass = visits[visit + 1]
        query = self.attn_res.attn_queries[nxt - 1]
        return query, (None if nxt_pass else self.layers[nxt].attn_norm)

    def forward_op(self, op, depth, positions, inputs_embeds, state, passes=None):
        """Run one op. Returns the state after it, or the final hidden for CODA."""
        from vllm.models.inkling2.nvidia.sconv import conv_metadata

        tokens, hidden = inputs_embeds.shape
        visits = self._visits(passes)
        core = self.loop_core
        tau = self._log_scaling_tau(positions)
        plan = [(False, False)] * 2
        if (
            self.allreduce_workspace is not None
            and conv_metadata(self.layers[0].conv_state) is not None
        ):
            plan = self.allreduce_workspace.plan(tokens)
        slots = self._max_loop_blocks() + 1
        if state is None:  # first op of the token
            if op not in (Op.PRELUDE,) and depth > 0:
                raise RuntimeError(
                    f"Inkling {op.name} at depth {depth} needs the state the previous op "
                    "carried; the runner lost it (preempted request or dropped step)"
                )
            stack = inputs_embeds.new_empty((slots, tokens, hidden))
            stack[0].copy_(inputs_embeds)
            carry = dict(
                num_blocks=0,
                prelude_blocks=None,
                prelude_partial=None,
                bank=None,
                inputs=None,
                prelude_dests=None,
            )
        else:
            stack, carry = self._restore(state, slots, tokens, hidden, inputs_embeds)
        num_blocks = carry["num_blocks"]
        prelude_blocks, prelude_partial = carry["prelude_blocks"], carry["prelude_partial"]
        bank, inputs, prelude_dests = carry["bank"], carry["inputs"], carry["prelude_dests"]
        for visit in self._op_range(op, depth, passes):
            i, pass_idx = visits[visit]
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = self._prelude_dests(stack, num_blocks, prelude_partial, query)
                inputs = prelude_dests
            next_query, next_norm = self._next_query(visits, visit)
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
                plan,
            )
        if op == Op.CODA:
            return inputs[0]
        return self._pack(
            stack, num_blocks, prelude_blocks, prelude_partial, bank, inputs, prelude_dests
        )

    # -- state packing: tensors keep their token axis, ints go in ``meta`` ------

    def _prelude_dests(self, stack, num_blocks, prelude_partial, query):
        from vllm.models.inkling2.nvidia.ops.attn_res import attn_res

        return attn_res(stack[:num_blocks], prelude_partial, None, query, None, 0.0, True)

    def _pack(self, stack, num_blocks, prelude_blocks, prelude_partial, bank, inputs, dests):
        tensors = {"stack": stack, "inputs": inputs}
        for name, value in (
            ("prelude_partial", prelude_partial),
            ("bank", bank),
            ("prelude_dests", dests),
        ):
            if value is not None:
                tensors[name] = value
        meta = {"num_blocks": num_blocks}
        if prelude_blocks is not None:
            meta["prelude_blocks"] = prelude_blocks
        return TokenState(tensors, {k: DIMS[k] for k in tensors}, meta)

    def _restore(self, state: TokenState, slots, tokens, hidden, inputs_embeds):
        state = pad_state(state, tokens)
        stack = state["stack"]
        if stack.shape[0] < slots:
            fresh = inputs_embeds.new_empty((slots, tokens, hidden))
            fresh[: stack.shape[0]].copy_(stack)
            stack = fresh
        carry = dict(
            num_blocks=state.meta["num_blocks"],
            prelude_blocks=state.meta.get("prelude_blocks"),
            prelude_partial=state.tensors.get("prelude_partial"),
            bank=state.tensors.get("bank"),
            inputs=state.tensors.get("inputs"),
            prelude_dests=state.tensors.get("prelude_dests"),
        )
        return stack, carry


class Inkling2LoopOpsForCausalLM(Inkling2LoopForConditionalGeneration):
    """vLLM's Inkling loop model with the loop split into ops (the plugin's
    ``Inkling2LoopForConditionalGeneration``). With no loop context it behaves
    exactly like vLLM's, so profiling, warmup and the FULL path are unchanged."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model.__class__ = SplitInkling2LoopModel  # same weights, split forward
        self.num_passes = self.text_config.num_loop_passes
        self.loop_ctx: LoopContext | None = None

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
        ctx = self.loop_ctx
        if ctx is None or ctx.op == Op.FULL:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kw)
        if ctx.mixed:
            raise NotImplementedError(
                "Inkling mixes AttnRes state per op; a step must hold one core depth"
            )
        tokens = positions.shape[0]
        state = None if ctx.op == Op.PRELUDE else ctx.state_for(tokens)
        if inputs_embeds is None and (ctx.op == Op.PRELUDE or state is None):
            inputs_embeds = self.model.embed_input_ids(input_ids)
        elif inputs_embeds is None:
            # Later ops start from carried state; embeddings only fix shape/dtype.
            weight = self.model.embed_tokens.weight
            inputs_embeds = torch.zeros(
                tokens, self.text_config.hidden_size, dtype=weight.dtype, device=weight.device
            )
        out = self.model.forward_op(ctx.op, ctx.depth, positions, inputs_embeds, state, ctx.passes)
        ctx.emit(tokens, out)
        return out if isinstance(out, torch.Tensor) else out["inputs"][0]

    def draft_hidden(self, input_ids, positions, passes: int) -> torch.Tensor:
        """Final hidden states of the tokens after prelude, ``passes`` core passes
        and coda in one go: a self-speculative draft step (VarLoop)."""
        return self.run_ops(input_ids, positions, passes)

    def run_ops(self, input_ids, positions, passes: int) -> torch.Tensor:
        """Prelude, ``passes`` core passes and coda on one batch, inside one
        forward context (a self-speculative draft step). Returns the final
        hidden states, which the LM head reads."""
        state = None
        for op, depth in [(Op.PRELUDE, 0), *[(Op.CORE, p) for p in range(passes)]]:
            ctx = LoopContext(op, depth, state, passes=passes)
            self.loop_ctx = ctx
            self.forward(input_ids, positions)
            state = ctx.out
        ctx = LoopContext(Op.CODA, passes - 1, state, passes=passes)
        self.loop_ctx = ctx
        try:
            return self.forward(input_ids, positions)
        finally:
            self.loop_ctx = None

    def loop_attention(self):
        """Inkling's per-pass KV is already vLLM layers; early exit is not used."""
        return []
