"""Ouro layers on vLLM's building blocks.

Weights are shared across passes; each (layer, pass) pair owns an ``Attention``
instance. vLLM gives every ``Attention`` its own KV tensor but one block table
per KV group, so per-pass KV needs no new cache code (this is how vLLM's own
Inkling loop model does it).

When a step mixes depths, the QKV and MLP projections run once for all rows (the
weight reads that dominate decode), and attention runs once per depth present,
each call writing KV only for its own rows (``masked_slots``) and contributing
only its own rows to the output.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.llama import LlamaMLP

from vllm_rlt.plugin.context import LoopContext, masked_slots


class OuroAttention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, num_passes: int, prefix: str) -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        quant = vllm_config.quant_config
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.qkv_proj = QKVParallelLinear(
            cfg.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            cfg.hidden_size,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=cfg.max_position_embeddings,
            rope_parameters=getattr(cfg, "rope_parameters", None)
            or {"rope_type": "default", "rope_theta": cfg.rope_theta},
            is_neox_style=True,
        )
        tp = self.qkv_proj.tp_size
        self.q_size = self.num_heads // tp * self.head_dim
        self.kv_size = max(1, self.num_kv_heads // tp) * self.head_dim
        # One Attention per pass: distinct names give distinct KV tensors.
        self.attn = nn.ModuleList(
            Attention(
                self.num_heads // tp,
                self.head_dim,
                self.head_dim**-0.5,
                num_kv_heads=max(1, self.num_kv_heads // tp),
                cache_config=vllm_config.cache_config,
                quant_config=quant,
                prefix=f"{prefix}.attn.pass{p}",
            )
            for p in range(num_passes)
        )

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, ctx: LoopContext):
        qkv, _ = self.qkv_proj(hidden)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if not ctx.mixed:
            attended = self.attn[ctx.depth](q, k, v)
        else:
            attended = None
            for depth, rows in ctx.group_rows.items():
                attn = self.attn[depth]
                with masked_slots(attn.layer_name, ctx.row_depth, depth):
                    part = attn(q, k, v)
                if attended is None:
                    attended = torch.zeros_like(part)
                attended[rows] = part[rows]
        out, _ = self.o_proj(attended)
        return out


class OuroLayer(nn.Module):
    """Ouro's sandwich-norm decoder layer (four RMSNorms)."""

    def __init__(self, vllm_config: VllmConfig, num_passes: int, prefix: str) -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        eps = cfg.rms_norm_eps
        self.self_attn = OuroAttention(vllm_config, num_passes, f"{prefix}.self_attn")
        self.mlp = LlamaMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            hidden_act=cfg.hidden_act,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=eps)
        self.input_layernorm_2 = RMSNorm(cfg.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=eps)
        self.post_attention_layernorm_2 = RMSNorm(cfg.hidden_size, eps=eps)

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, ctx: LoopContext):
        hidden = hidden + self.input_layernorm_2(
            self.self_attn(positions, self.input_layernorm(hidden), ctx)
        )
        return hidden + self.post_attention_layernorm_2(
            self.mlp(self.post_attention_layernorm(hidden))
        )
