"""Ouro's HF config, registered so vLLM can load the checkpoint.

The checkpoint ships ``configuration_ouro.py`` (remote code) and vLLM removed its
own Ouro support in 0.26, so without this ``AutoConfig`` refuses to load it. The
fields are exactly the ones the model reads; unknown keys pass through to
``PretrainedConfig``.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class OuroConfig(PretrainedConfig):
    model_type = "ouro"

    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 128,
        max_position_embeddings: int = 65536,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        total_ut_steps: int = 4,
        early_exit_threshold: float = 1.0,
        hidden_act: str = "silu",
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.total_ut_steps = total_ut_steps
        self.early_exit_threshold = early_exit_threshold
        self.hidden_act = hidden_act
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def register_ouro_config() -> None:
    AutoConfig.register("ouro", OuroConfig, exist_ok=True)
