"""Ouro layers on vLLM's building blocks.

Weights are shared across passes; each (layer, pass) pair owns an ``Attention``
instance. vLLM gives every ``Attention`` its own KV tensor but one block table
per KV group, so per-pass KV needs no new cache code (this is how vLLM's own
Inkling loop model does it).
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

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, depth: int):
        qkv, _ = self.qkv_proj(hidden)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        out, _ = self.o_proj(self.attn[depth](q, k, v))
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
        norm = lambda: RMSNorm(cfg.hidden_size, eps=eps)  # noqa: E731
        self.input_layernorm, self.input_layernorm_2 = norm(), norm()
        self.post_attention_layernorm, self.post_attention_layernorm_2 = norm(), norm()

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, depth: int):
        hidden = hidden + self.input_layernorm_2(
            self.self_attn(positions, self.input_layernorm(hidden), depth)
        )
        return hidden + self.post_attention_layernorm_2(
            self.mlp(self.post_attention_layernorm(hidden))
        )
