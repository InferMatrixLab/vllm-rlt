# SPDX-License-Identifier: Apache-2.0
# Architecture adapted from vllm/model_executor/models/dory.py (vLLM #60149),
# revision 75fc05a53a366068f534a61ab3ee6b50485419f2 (Apache-2.0).
# Original architecture Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Native Dory: normalized-GPT latent recurrent transformer with boundary KV.

Dory is one physical stack of 72 sublayers, each either an attention mixer or an
MLP mixer: 16 input, 40 recurrent and 16 output sublayers. The recurrent
sublayers run ``n_recurrent_loops`` times; before every loop the state is
combined with the *retained* input-stage output and re-normalized, so the
injection input is part of the carried hidden state.

The runner therefore carries ``[recurrent state, injection input]`` exactly like
Huginn: ``hidden_size`` is the doubled carry width and ``d_model`` is the
artifact width. Attention-bearing PRELUDE and CODA own boundary KV and use
depth zero. Recurrent attention writes at its own loop depth, so the cache
``per_loop`` mode is the per-depth plane layout and ``last_loop`` mode is the
single shared plane.
"""

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.layers import RotaryEmbedding, apply_rotary_pos_emb

if TYPE_CHECKING:
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.worker.metadata import DeviceKVMetadata, KVMetadata

# Fixed NGPT epsilon of the reviewed upstream implementation, deliberately not a
# configuration field: changing it changes the model, not a deployment choice.
NGPT_EPS = 1e-6
_LAYER_TYPES = ("sliding_attention", "full_attention", "mlp")
_DEFAULT_ROPE_PARAMETERS = {
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
    },
    "full_attention": {
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 1.0,
    },
}


def justnorm(value: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """NGPT unit-length normalization with FP32 accumulation and cast back."""
    value32 = value.float()
    return (value32 * torch.rsqrt(value32.square().sum(dim=dim, keepdim=True) + NGPT_EPS)).to(
        value.dtype
    )


def slerp_norm(residual: torch.Tensor, update: torch.Tensor, weight: torch.Tensor):
    """Normalized residual interpolation of two unit-length directions."""
    base, target = justnorm(residual).float(), justnorm(update).float()
    return justnorm(base + weight.float() * (target - base)).to(residual.dtype)


def slerp_alpha(residual, update, alpha, alpha_scale):
    return slerp_norm(residual, update, (alpha.float() * alpha_scale).abs())


@dataclass(frozen=True)
class DoryConfig:
    """Dory-1.0-4B configuration with inference-only validation.

    ``d_model`` is the artifact width (``hidden_size`` in config.json);
    ``hidden_size`` is the runner's doubled carry width because the retained
    injection input travels with the recurrent state.
    """

    vocab_size: int = 131072
    d_model: int = 2560
    intermediate_size: int = 7168
    num_hidden_layers: int = 72
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.0177
    ngpt_alpha_init: float = 0.05
    attention_output_gate_logit_scale_init: float | None = None
    mlp_bias: bool = False
    attention_bias: bool = False
    n_input_layers: int = 16
    n_recurrent_layers: int = 40
    n_output_layers: int = 16
    n_recurrent_loops: int = 2
    recurrent_kv_cache_mode: str = "per_loop"
    sliding_window: int | None = 1024
    layer_types: tuple[str, ...] | None = None
    rope_parameters: dict[str, dict[str, Any]] | None = None
    tie_word_embeddings: bool = False
    bos_token_id: int | None = 1
    eos_token_id: int | tuple[int, ...] | None = (2, 11)
    pad_token_id: int | None = 11

    def __post_init__(self) -> None:
        counts = (
            "vocab_size",
            "d_model",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "n_input_layers",
            "n_recurrent_layers",
            "n_output_layers",
            "n_recurrent_loops",
        )
        for name in counts:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_input_layers + self.n_recurrent_layers + self.n_output_layers != (
            self.num_hidden_layers
        ):
            raise ValueError(
                "n_input_layers + n_recurrent_layers + n_output_layers must equal num_hidden_layers"
            )
        if self.n_recurrent_layers < 2 or self.n_recurrent_layers % 2:
            raise ValueError("the recurrent stack must hold attention/MLP pairs")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for split-half RoPE")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        for name in ("initializer_range", "ngpt_alpha_init"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.attention_output_gate_logit_scale_init is not None and (
            not math.isfinite(self.attention_output_gate_logit_scale_init)
            or self.attention_output_gate_logit_scale_init <= 0
        ):
            raise ValueError("attention_output_gate_logit_scale_init must be finite and positive")
        if self.hidden_act != "silu":
            raise ValueError("Only Dory's silu activation is supported")
        if self.mlp_bias or self.attention_bias:
            raise ValueError("Dory-1.0-4B has no attention or MLP bias terms")
        if self.tie_word_embeddings:
            raise ValueError("Dory's embeddings and output projection are untied")
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("recurrent_kv_cache_mode must be 'per_loop' or 'last_loop'")

        layer_types = self.layer_types
        if layer_types is None:
            layer_types = tuple(
                "sliding_attention" if index % 2 == 0 else "mlp"
                for index in range(self.num_hidden_layers)
            )
        else:
            layer_types = tuple(layer_types)
            if len(layer_types) != self.num_hidden_layers:
                raise ValueError("layer_types must describe every physical sublayer")
            if any(name not in _LAYER_TYPES for name in layer_types):
                raise ValueError(f"layer_types entries must be one of {_LAYER_TYPES}")
        object.__setattr__(self, "layer_types", layer_types)
        # Every stage must contain the mixers its stage math requires; an MLP-only
        # stage would silently drop attention from a boundary that owns KV.
        stages = self.stage_slices
        for stage, (start, end) in zip(("input", "recurrent", "output"), stages):
            if not any(layer_types[index] != "mlp" for index in range(start, end)):
                raise ValueError(f"the {stage} stage needs at least one attention sublayer")
        sliding = [name for name in layer_types if name == "sliding_attention"]
        if sliding and (
            self.sliding_window is None
            or isinstance(self.sliding_window, bool)
            or not isinstance(self.sliding_window, int)
            or self.sliding_window <= 0
        ):
            raise ValueError("sliding attention needs a positive sliding_window")
        if not sliding and self.sliding_window is not None:
            raise ValueError("sliding_window is set but no sublayer uses sliding attention")

        parameters = self.rope_parameters
        if parameters is None:
            parameters = _DEFAULT_ROPE_PARAMETERS
        parameters = {name: dict(values) for name, values in parameters.items()}
        for name in set(layer_types) - {"mlp"}:
            if name not in parameters:
                raise ValueError(f"missing rope_parameters for {name}")
        for name, values in parameters.items():
            if values.get("rope_type") != "default":
                raise ValueError("Only default rotary embedding is supported")
            if values.get("partial_rotary_factor", 1.0) != 1.0:
                raise ValueError("Dory rotates the full head dimension")
            theta = values.get("rope_theta")
            if not isinstance(theta, int | float) or not math.isfinite(theta) or theta <= 0:
                raise ValueError("rope_theta must be finite and positive")
        object.__setattr__(self, "rope_parameters", parameters)

        eos = self.eos_token_id
        eos = (eos,) if isinstance(eos, int) and not isinstance(eos, bool) else tuple(eos or ())
        for token_id in (*eos, self.bos_token_id, self.pad_token_id):
            if token_id is None:
                continue
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("special token IDs must be integers")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError("special token IDs must be vocabulary indices")
        object.__setattr__(self, "eos_token_id", eos or None)

    @property
    def hidden_size(self) -> int:
        """Carried width of ``[recurrent state, injection input]``."""
        return 2 * self.d_model

    @property
    def total_ut_steps(self) -> int:
        """Loop count the scheduler must run for a full-depth request."""
        return self.n_recurrent_loops

    @property
    def stage_slices(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Physical sublayer ranges of the input, recurrent and output stages."""
        start = 0
        spans = []
        for count in (self.n_input_layers, self.n_recurrent_layers, self.n_output_layers):
            spans.append((start, start + count))
            start += count
        return tuple(spans)

    @property
    def kv_cache_layout(self) -> str:
        """KV layout that implements ``recurrent_kv_cache_mode`` in this runtime."""
        return "last_exited" if self.recurrent_kv_cache_mode == "per_loop" else "shared"

    def attention_layers(self, stage: str) -> tuple[int, ...]:
        names = ("input", "recurrent", "output")
        if stage not in names:
            raise ValueError(f"stage must be one of {names}")
        start, end = self.stage_slices[names.index(stage)]
        return tuple(index for index in range(start, end) if self.layer_types[index] != "mlp")

    def rope_theta(self, layer_type: str) -> float:
        return float(self.rope_parameters[layer_type]["rope_theta"])

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DoryConfig":
        if values.get("model_type", "dory") != "dory":
            raise ValueError("Only model_type='dory' is supported")
        if values.get("architectures", ["DoryForCausalLM"]) != ["DoryForCausalLM"]:
            raise ValueError("Only the DoryForCausalLM architecture is supported")
        names = {field.name for field in fields(cls)}
        # Artifact metadata that carries no inference behavior.
        metadata = {
            "architectures",
            "auto_map",
            "model_type",
            "torch_dtype",
            "dtype",
            "transformers_version",
            "use_cache",
            "_name_or_path",
            "rope_scaling",
        }
        unknown = values.keys() - names - metadata - {"hidden_size"}
        if unknown:
            raise ValueError(f"Unsupported Dory configuration fields: {sorted(unknown)}")
        if "hidden_size" in values and "d_model" in values:
            raise ValueError("Specify either hidden_size or d_model, not both")
        config_values = {
            key: value for key, value in values.items() if key in names and key != "d_model"
        }
        # Dory's head dimension is explicit; 2560 / 16 is a projection shape, not
        # the attention head dimension, so an absent head_dim is an error.
        if "head_dim" not in config_values:
            raise ValueError("Dory configuration must declare the explicit head_dim")
        model_width = values.get("hidden_size", values.get("d_model"))
        if model_width is not None:
            config_values["d_model"] = model_width
        return cls(**config_values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["hidden_size"] = values.pop("d_model")
        values["layer_types"] = list(self.layer_types)
        return {"model_type": "dory", "architectures": ["DoryForCausalLM"], **values}


class DoryAttention(nn.Module):
    """GQA with post-RoPE NGPT Q/K normalization, output gate and justnorm."""

    def __init__(self, config: DoryConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.d_model, self.q_size, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.kv_size, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.kv_size, bias=False)
        self.g_proj = nn.Linear(config.d_model, self.q_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.d_model, bias=False)
        # Learned per-channel scales, stored without the constant factors that the
        # NGPT parametrization applies at use time.
        self.sqk = nn.Parameter(torch.empty(self.q_size))
        self.s_gate = nn.Parameter(torch.empty(self.q_size))
        self.sqk_factor = self.head_dim**0.5 / config.initializer_range
        gate_scale = config.attention_output_gate_logit_scale_init
        if gate_scale is None:
            gate_scale = config.d_model**0.5
        self.s_gate_factor = gate_scale**0.5 / config.initializer_range
        self.layer_type = config.layer_types[layer_idx]
        self.window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim, rope_theta=config.rope_theta(self.layer_type)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        batch: "KVMetadata | DeviceKVMetadata",
        cache: "KVCacheManager",
    ) -> torch.Tensor:
        count = hidden.shape[0]
        q = self.q_proj(hidden).view(count, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(count, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(count, self.num_kv_heads, self.head_dim)
        gate = self.g_proj(hidden)
        cos, sin = self.rotary_emb(hidden, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # nGPT Q/K normalization happens after RoPE: queries carry the square of
        # the learned scale, keys only the unit-length normalization.
        sqk = (self.sqk.float() * self.sqk_factor).view(1, self.num_heads, self.head_dim)
        q = (sqk.square() * justnorm(q).float()).to(q.dtype)
        k = justnorm(k)
        cache._write_prepared(self.layer_idx, batch, k, v)
        attended = cache._attend_prepared(self.layer_idx, batch, q, window=self.window)
        # Output gate before the projection, then fngpt justnorm over the full
        # head concatenation.
        scale = (self.s_gate.float() * self.s_gate_factor).square()
        attended = (attended.reshape(count, -1).float() * torch.sigmoid(gate.float() * scale)).to(
            hidden.dtype
        )
        return self.o_proj(justnorm(attended))


class DoryMLP(nn.Module):
    """SwiGLU with NGPT gate/up scaling and justnorm before the projection."""

    def __init__(self, config: DoryConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)
        self.suv_gate = nn.Parameter(torch.empty(config.intermediate_size))
        self.suv_up = nn.Parameter(torch.empty(config.intermediate_size))
        self.suv_factor = config.d_model**0.5

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden)
        up = self.up_proj(hidden)
        gate = (gate.float() * (self.suv_gate.float() * self.suv_factor)).to(gate.dtype)
        up = (up.float() * (self.suv_up.float() * self.suv_factor)).to(up.dtype)
        mixed = (F.silu(gate.float()) * up.float()).to(gate.dtype)
        return self.down_proj(justnorm(mixed))


class DorySublayer(nn.Module):
    """One physical sublayer: a single attention or MLP mixer plus slerp residual."""

    def __init__(self, config: DoryConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_attention = self.layer_type != "mlp"
        self.alpha_scale = config.ngpt_alpha_init / config.initializer_range
        if self.is_attention:
            self.mixer = DoryAttention(config, layer_idx)
            self.attn_alpha = nn.Parameter(torch.empty(config.d_model))
        else:
            self.mixer = DoryMLP(config)
            self.mlp_alpha = nn.Parameter(torch.empty(config.d_model))

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        batch: "KVMetadata | DeviceKVMetadata",
        cache: "KVCacheManager",
    ) -> torch.Tensor:
        if self.is_attention:
            update = self.mixer(hidden, position_ids, batch, cache)
            return slerp_alpha(hidden, update, self.attn_alpha, self.alpha_scale)
        return slerp_alpha(hidden, self.mixer(hidden), self.mlp_alpha, self.alpha_scale)


class DoryBackbone(nn.Module):
    """Embedding plus the single physical 72-sublayer stack."""

    def __init__(self, config: DoryConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            config.vocab_size, config.d_model, config.pad_token_id if config.pad_token_id else None
        )
        self.layers = nn.ModuleList(
            DorySublayer(config, index) for index in range(config.num_hidden_layers)
        )
        input_slice, recurrent_slice, output_slice = config.stage_slices
        assert input_slice[0] == 0
        # Store stage index ranges only: a sliced ModuleList would register every
        # sublayer a second time under a different name.
        self.input_layers = tuple(range(*input_slice))
        self.recurrent_layers = tuple(range(*recurrent_slice))
        self.output_layers = tuple(range(*output_slice))


class DoryForCausalLM(nn.Module):
    """Inference-only Dory with eager KV boundaries and a graphable recurrent core."""

    requires_boundary_kv = True
    # No Ouro-style learned exit gate: the reviewed implementation always runs
    # the configured loop count, so requesting a shallower depth is unsupported.
    fixed_loop_count = True
    # Sliding-window masking currently needs host-planned KV metadata.
    supports_cuda_graphs = False

    def __init__(self, config: DoryConfig) -> None:
        super().__init__()
        if not isinstance(config, DoryConfig):
            raise TypeError("config must be a DoryConfig")
        self.config = config
        self.backbone = DoryBackbone(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Learned vocabulary-wise logit scale of the reviewed implementation.
        self.lm_head.logit_scale = nn.Parameter(torch.empty(config.vocab_size))
        self.prelude_kv_layers = config.attention_layers("input")
        self.recurrent_kv_layers = config.attention_layers("recurrent")
        self.coda_kv_layers = config.attention_layers("output")
        self.apply(self._initialize)
        self.requires_grad_(False)
        self.eval()

    def _initialize(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
            if isinstance(module, nn.Linear) and hasattr(module, "logit_scale"):
                nn.init.normal_(module.logit_scale, mean=0, std=std)
        elif isinstance(module, DoryAttention):
            nn.init.normal_(module.sqk, mean=0, std=std)
            nn.init.normal_(module.s_gate, mean=0, std=std)
        elif isinstance(module, DoryMLP):
            nn.init.normal_(module.suv_gate, mean=0, std=std)
            nn.init.normal_(module.suv_up, mean=0, std=std)
        elif isinstance(module, DorySublayer):
            alpha = module.attn_alpha if module.is_attention else module.mlp_alpha
            nn.init.normal_(alpha, mean=0, std=std)

    def prelude_prepared(self, tokens, batch, cache, *, generators=None) -> torch.Tensor:
        """Embed, then run the input stage; loop zero starts from its output."""
        if tokens.ndim != 1:
            raise ValueError("prelude expects packed token_ids with shape [N]")
        hidden = self.backbone.embeddings(tokens)
        for index in self.backbone.input_layers:
            hidden = self.backbone.layers[index](hidden, batch.position_ids, batch, cache)
        # The retained injection input is half of the carried state, and loop zero
        # starts from the same tensor, matching the reference ``state = input_h``.
        return torch.cat((hidden, hidden), dim=-1)

    def recurrent(
        self,
        hidden: torch.Tensor,
        request_ids,
        depths,
        positions,
        cache: "KVCacheManager",
        *,
        compute_gate: bool = True,
    ):
        batch = cache._prepare_batch(request_ids, depths, positions)
        return self.recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate: bool = True):
        """Run one loop: combine state with the injection input, then the core."""
        if hidden.ndim != 2 or hidden.shape[1] != self.config.hidden_size:
            raise ValueError("recurrent expects hidden with shape [N, hidden_size]")
        state, injection = hidden.split(self.config.d_model, dim=-1)
        current = justnorm(state + injection)
        for index in self.backbone.recurrent_layers:
            current = self.backbone.layers[index](current, batch.position_ids, batch, cache)
        state = justnorm(current)
        # Dory has no exit gate; a constant gate keeps the scheduler on the
        # configured depth instead of inventing an exit policy.
        gate = (
            torch.full((len(hidden),), -1e4, device=hidden.device, dtype=hidden.dtype)
            if compute_gate
            else None
        )
        return torch.cat((state, injection), dim=-1), gate

    def coda_prepared(self, hidden, batch, cache, *, compute_logits: bool = True):
        """Run the attention-bearing output stage and the scaled LM head."""
        state, _ = hidden.split(self.config.d_model, dim=-1)
        for index in self.backbone.output_layers:
            state = self.backbone.layers[index](state, batch.position_ids, batch, cache)
        return self.logits(state) if compute_logits else None

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(state).float()
        return logits * (self.lm_head.logit_scale.float() / self.config.initializer_range)

    def prelude(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Dory's input stage owns attention KV; use prelude_prepared")

    def coda(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Dory's output stage owns attention KV; use coda_prepared")

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        config: DoryConfig | None = None,
        loops: int | None = None,
    ) -> "DoryForCausalLM":
        """Strictly stream safetensors into a meta model; never execute Hub code."""
        from safetensors import safe_open

        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        folder = Path(path_or_repo).expanduser()
        if not folder.is_dir():
            from vllm_rlt.models import resolve_model_config

            source, revision, _ = resolve_model_config(path_or_repo, revision=revision)
            folder = Path(source)
        if config is None:
            config = DoryConfig.from_dict(json.loads((folder / "config.json").read_text()))
        elif not isinstance(config, DoryConfig):
            raise TypeError("config must be a DoryConfig")
        if loops is not None:
            if type(loops) is not int or loops <= 0:
                raise ValueError("loops must be a positive integer")
            config = replace(config, n_recurrent_loops=loops)
        index_path = folder / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())["weight_map"] if index_path.is_file() else None
        if index is None:
            shards = [folder / "model.safetensors"]
        else:
            filenames = set(index.values())
            if any(
                Path(name).name != name or not name.endswith(".safetensors") for name in filenames
            ):
                raise ValueError("Checkpoint index must reference local safetensors basenames")
            shards = [folder / name for name in sorted(filenames)]
        if any(not shard.is_file() for shard in shards):
            raise FileNotFoundError("A complete safetensors checkpoint is required")
        with torch.device("meta"):
            model = cls(config)
        expected = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
        discovered: dict[str, Path] = {}
        # Validate every key and shape before allocating model weight memory.
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    if name in discovered:
                        raise ValueError(f"Duplicate checkpoint tensor: {name}")
                    if name not in expected:
                        raise ValueError(f"Unexpected checkpoint tensor: {name}")
                    if tuple(checkpoint.get_slice(name).get_shape()) != expected[name]:
                        raise ValueError(f"Checkpoint shape mismatch for {name}")
                    if index is not None and index.get(name) != shard.name:
                        raise ValueError(f"Checkpoint index mismatch for {name}")
                    discovered[name] = shard
        missing = expected.keys() - discovered.keys()
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        if index is not None and set(index) != set(expected):
            raise ValueError("Checkpoint index does not match Dory parameter names")
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    tensor = checkpoint.get_tensor(name)
                    if not tensor.is_floating_point():
                        raise ValueError(f"Checkpoint tensor must be floating point: {name}")
                    module_path, parameter_name = name.rsplit(".", 1)
                    module = model.get_submodule(module_path)
                    setattr(
                        module,
                        parameter_name,
                        nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False),
                    )
        for module in model.modules():
            if isinstance(module, RotaryEmbedding):
                module.inv_freq = module.frequencies(device=device)
        return model.eval()


__all__ = [
    "DoryConfig",
    "DoryForCausalLM",
    "justnorm",
    "slerp_alpha",
    "slerp_norm",
]
