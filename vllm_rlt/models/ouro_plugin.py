"""Ouro's recurrent layer execution on vLLM's dense model components."""

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM

from vllm_rlt.config import LoopRuntimeConfig
from vllm_rlt.worker.kv_updates import backfill_kv, mask_slot_mapping
from vllm_rlt.worker.loop_state import loop_execution


class _OuroLayer(LlamaDecoderLayer):
    def __init__(self, vllm_config, prefix):
        super().__init__(vllm_config, prefix)
        config = vllm_config.model_config.hf_config
        options = LoopRuntimeConfig.from_additional_config(vllm_config.additional_config)
        if options.silu_and_mul:
            from vllm_rlt.ops.silu_and_mul import RltSiluAndMul

            self.mlp.act_fn = RltSiluAndMul()
        self.input_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        owner = self.self_attn.attn
        self.pass_caches = nn.ModuleList(
            Attention(
                owner.num_heads,
                owner.head_size,
                self.self_attn.scaling,
                num_kv_heads=owner.num_kv_heads,
                cache_config=vllm_config.cache_config,
                per_layer_sliding_window=owner.sliding_window,
                attn_backend=owner.get_attn_backend(),
                prefix=f"{owner.layer_name}.pass{depth}",
            )
            for depth in range(1, config.total_ut_steps)
        )

    def _project(self, positions, hidden):
        attention = self.self_attn
        normalized = self.input_layernorm(hidden)
        qkv, _ = attention.qkv_proj(normalized)
        query, key, value = qkv.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
        )
        query, key = attention.rotary_emb(positions, query, key)
        return query, key, value

    def _finish(self, hidden, attention_output):
        projected, _ = self.self_attn.o_proj(attention_output)
        hidden = hidden + self.input_layernorm_2(projected)
        return hidden + self.post_attention_layernorm_2(
            self.mlp(self.post_attention_layernorm(hidden))
        )

    def _cache(self, depth):
        return self.pass_caches[depth - 1] if depth else self.self_attn.attn

    def forward(self, positions, hidden, depth):
        query, key, value = self._project(positions, hidden)
        hidden = self._finish(hidden, self._cache(depth)(query, key, value))
        return hidden, key, value

    def forward_mixed(self, positions, hidden, groups):
        query, key, value = self._project(positions, hidden)
        attention = torch.empty_like(query)
        context = get_forward_context()
        original = context.slot_mapping
        try:
            for depth, rows, slots in groups:
                context.slot_mapping = slots
                output = self._cache(depth)(query, key, value)
                attention.index_copy_(0, rows, output.index_select(0, rows))
        finally:
            context.slot_mapping = original
        return self._finish(hidden, attention), key, value


@dataclass(slots=True)
class _OuroState:
    hidden: torch.Tensor
    remaining: torch.Tensor | None = None


class LoopOuroForCausalLM(LlamaForCausalLM):
    rlt_loop_ops = True
    rlt_core_samples = True

    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        options = LoopRuntimeConfig.from_additional_config(vllm_config.additional_config)
        if options.silu_and_mul and vllm_config.quant_config is not None:
            raise ValueError("rlt_silu_and_mul requires an unquantized Ouro model")
        if getattr(config, "rope_parameters", None) is None:
            config.rope_parameters = {"rope_type": "default", "rope_theta": config.rope_theta}
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=_OuroLayer)
        self.model.early_exit_gate = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        step = loop_execution.get()
        if step is None:
            hidden = (
                self.model.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
            )
            for depth in range(self.config.total_ut_steps):
                hidden, _ = self._run_pass(positions, hidden, depth, capture_kv=False)
            return hidden

        if len(step.ordered) > 1:
            return self._forward_mixed(step, input_ids, positions)

        (span,) = step.ordered
        states = step.state.setdefault(span.request_id, {})
        state = states.pop(span.token_start, None)
        if state is None:
            state = _OuroState(self.model.embed_input_ids(input_ids))
        hidden, kv = self._run_pass(
            positions,
            state.hidden,
            span.depth,
            capture_kv=not span.prefill and span.depth + 1 < self.config.total_ut_steps,
        )
        adaptive_exit = False
        if span.exit_threshold is not None and not span.last_core:
            hazard = self.model.early_exit_gate(hidden).float().sigmoid().squeeze(-1)
            remaining = state.remaining
            state.remaining = (1.0 - hazard) if remaining is None else remaining * (1.0 - hazard)
            adaptive_exit = span.depth + 1 >= span.min_loops and bool(
                (1.0 - state.remaining >= span.exit_threshold).item()
            )
        if adaptive_exit or span.last_core:
            if span.check_exit:
                step.exit_depths[span.request_id] = span.depth
            for layer, (key, value) in zip(self.model.layers, kv):
                for cache in layer.pass_caches[span.depth :]:
                    backfill_kv(cache, key, value)
        else:
            state.hidden = hidden
            states[span.token_start] = state
        return hidden

    def _forward_mixed(self, step, input_ids, positions):
        ctx = get_forward_context()
        slots = ctx.slot_mapping
        assert isinstance(slots, dict)
        size = positions.numel()
        embedded = self.model.embed_input_ids(input_ids)
        states, ranges, by_depth = [], [], defaultdict(list)
        start = 0
        for span in step.ordered:
            end = start + span.token_count
            stash = step.state.setdefault(span.request_id, {})
            state = stash.pop(span.token_start, None) or _OuroState(embedded[start:end])
            states.append(state)
            ranges.append((start, end))
            by_depth[span.depth].extend(range(start, end))
            start = end

        def build_slots(row_ids):
            rows = torch.as_tensor(row_ids, device=positions.device)
            active = torch.zeros(size, dtype=torch.bool, device=positions.device)
            active.index_fill_(0, rows, True)
            return rows, mask_slot_mapping(slots, active)

        groups = [(depth, *build_slots(rows)) for depth, rows in by_depth.items()]
        hidden = torch.cat([state.hidden for state in states])
        kv = []
        for layer in self.model.layers:
            hidden, key, value = layer.forward_mixed(positions, hidden, groups)
            kv.append((key, value))
        hidden = self.model.norm(hidden)

        checked = [
            (span, state, start)
            for span, state, (start, _) in zip(step.ordered, states, ranges, strict=True)
            if span.check_exit and not span.last_core
        ]
        exit_ids = set()
        if checked:
            hazards = self.model.early_exit_gate(hidden).float().sigmoid().squeeze(-1)
            results = []
            for span, state, start in checked:
                remaining = 1.0 - hazards[start : start + 1]
                state.remaining = (
                    remaining if state.remaining is None else state.remaining * remaining
                )
                results.append(
                    (1.0 - state.remaining >= span.exit_threshold)
                    & (span.depth + 1 >= span.min_loops)
                )
            exit_ids.update(
                span.request_id
                for (span, _, _), reached in zip(checked, torch.cat(results).tolist(), strict=True)
                if reached
            )

        backfill = defaultdict(list)
        for span, state, (start, end) in zip(step.ordered, states, ranges, strict=True):
            if span.last_core or span.request_id in exit_ids:
                if span.check_exit:
                    step.exit_depths[span.request_id] = span.depth
                if not span.prefill:
                    for depth in range(span.depth + 1, self.config.total_ut_steps):
                        backfill[depth].extend(range(start, end))
            else:
                state.hidden = hidden[start:end].clone()
                step.state[span.request_id][span.token_start] = state
        try:
            for depth, row_ids in backfill.items():
                _, ctx.slot_mapping = build_slots(row_ids)
                for layer, (key, value) in zip(self.model.layers, kv, strict=True):
                    backfill_kv(layer._cache(depth), key, value)
        finally:
            ctx.slot_mapping = slots
        return hidden

    def _run_pass(self, positions, hidden, depth, *, capture_kv):
        kv = []
        for layer in self.model.layers:
            hidden, key, value = layer(positions, hidden, depth)
            if capture_kv:
                kv.append((key, value))
        return self.model.norm(hidden), kv
