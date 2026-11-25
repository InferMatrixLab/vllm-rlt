"""Ouro as a loop model for the plugin (vLLM removed its own Ouro in 0.26).

One CORE op runs all 24 layers at one pass index (or, when rows sit at different
depths, each row at its own), then the shared final norm and the exit gate.
There is no PRELUDE op (the embedding is the first thing CORE depth 0 does) and
CODA is just the LM head, which vLLM's runner applies when it samples. The
runner supplies the loop context before each step.

The step's batch can be padded by vLLM (graph or DP size), so a carried state may
have fewer rows than ``input_ids``; ``forward`` pads it to match.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_rlt.plugin.context import LoopContext, pad_rows
from vllm_rlt.plugin.ouro_layers import OuroLayer
from vllm_rlt.plugin.program import Op


class OuroModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, num_passes: int, prefix: str) -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            OuroLayer(vllm_config, num_passes, f"{prefix}.layers.{i}")
            for i in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.early_exit_gate = nn.Linear(cfg.hidden_size, 1, bias=True)


class OuroLoopForCausalLM(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.config = cfg
        self.num_passes = int(getattr(cfg, "total_ut_steps", 4))
        self.model = OuroModel(vllm_config, self.num_passes, "model")
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size, prefix="lm_head")
        self.logits_processor = LogitsProcessor(cfg.vocab_size)
        self.loop_ctx: LoopContext | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        ctx = self.loop_ctx
        if ctx is None:  # a plain vLLM call (profiling, warmup): one pass at depth 0
            ctx = LoopContext(Op.CORE, 0)
        rows = positions.shape[0]
        if ctx.op == Op.CODA:
            # An idle DP rank has no carried state: use embeddings so the same
            # tensors flow through the LM head.
            hidden = self.embed_input_ids(input_ids) if ctx.state is None else ctx.state
            ctx.out = hidden = pad_rows(hidden, rows)
            return hidden
        if ctx.state is None or (not ctx.mixed and ctx.depth == 0):
            hidden = self.embed_input_ids(input_ids)
        else:
            hidden = pad_rows(ctx.state, rows)
        if ctx.mixed and ctx.state is not None:
            # Depth-0 rows have no carried hidden state; embed them.
            embedded = self.embed_input_ids(input_ids)
            first = pad_rows(ctx.row_depth, rows, -1) == 0
            hidden = torch.where(first.unsqueeze(-1), embedded, hidden)
        for layer in self.model.layers:
            hidden = layer(positions, hidden, ctx)
        hidden = self.model.norm(hidden)
        ctx.out = hidden
        return hidden

    def loop_gate(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.early_exit_gate(hidden).squeeze(-1)

    def loop_attention(self) -> list[list[nn.Module]]:
        """Attention instances indexed [pass][layer], for the LAST_EXITED fill."""
        return [
            [layer.self_attn.attn[p] for layer in self.model.layers] for p in range(self.num_passes)
        ]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
