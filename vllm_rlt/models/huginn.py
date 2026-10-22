# SPDX-License-Identifier: Apache-2.0
"""Native Huginn execution with per-depth recurrent KV and boundary KV.

The runner carries ``[recurrent state, prelude output]`` between stages. The
second half is the fixed injection input to every recurrent step. Prelude and
coda attention use depth zero; recurrent attention uses its own depth plane.
"""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.layers import RMSNorm

if TYPE_CHECKING:
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.worker.metadata import KVMetadata


@dataclass(frozen=True)
class HuginnConfig:
    n_embd: int = 5280
    n_heads: int = 55
    n_layers: int = 8
    n_layers_in_prelude: int = 2
    n_layers_in_recurrent_block: int = 4
    n_layers_in_coda: int = 2
    intermediate_size: int = 17920
    mean_recurrence: int = 32
    block_size: int = 4096
    vocab_size: int = 65536
    padded_vocab_size: int = 65536
    norm_eps: float = 1e-6
    rope_base: float = 50000.0
    qk_bias: bool = True
    bias: bool = False
    tie_embeddings: bool = True
    injection_type: str = "linear"
    state_init: str = "like-init"
    bos_token_id: int | None = 65504
    eos_token_id: int | None = 65505
    pad_token_id: int | None = 65509

    def __post_init__(self) -> None:
        values = asdict(self)
        for name in (
            "n_embd",
            "n_heads",
            "n_layers",
            "n_layers_in_prelude",
            "n_layers_in_recurrent_block",
            "n_layers_in_coda",
            "intermediate_size",
            "mean_recurrence",
            "block_size",
            "vocab_size",
            "padded_vocab_size",
        ):
            value = values[name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_layers != (
            self.n_layers_in_prelude + self.n_layers_in_recurrent_block + self.n_layers_in_coda
        ):
            raise ValueError("n_layers must equal prelude + recurrent + coda layers")
        if self.n_embd % self.n_heads or self.head_dim % 2:
            raise ValueError("Huginn attention requires an even integral head dimension")
        if self.padded_vocab_size != self.vocab_size:
            raise ValueError("padded_vocab_size must match vocab_size")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be finite and positive")
        if self.bias or not self.qk_bias or not self.tie_embeddings:
            raise ValueError("Only the Huginn-0125 bias, QK bias and tied-head layout is supported")
        if self.injection_type != "linear" or self.state_init != "like-init":
            raise ValueError(
                "Only linear injection and like-init state initialization are supported"
            )

    @property
    def hidden_size(self) -> int:
        # The runner retains both recurrent state and its fixed injection input.
        return 2 * self.n_embd

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_heads

    @property
    def num_attention_heads(self) -> int:
        return self.n_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.n_heads

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layers

    @property
    def total_ut_steps(self) -> int:
        return self.mean_recurrence

    @property
    def max_position_embeddings(self) -> int:
        return self.block_size

    @property
    def initializer_range(self) -> float:
        return math.sqrt(2 / (5 * self.n_embd))

    @classmethod
    def from_dict(cls, values: dict) -> "HuginnConfig":
        if values.get("model_type") != "huginn_raven":
            raise ValueError("Only model_type='huginn_raven' is supported")
        if values.get("architectures") != ["RavenForCausalLM"]:
            raise ValueError("Only the RavenForCausalLM architecture is supported")
        for key, expected in (
            ("architecture_class_name", "RecurrentGPT"),
            ("block_class_name", "SandwichBlock"),
            ("norm_class_name", "RMSNorm_llama"),
            ("mlp_class_name", "GatedMLP"),
            ("nonlin_name", "SiLU"),
        ):
            if key in values and values[key] != expected:
                raise ValueError(f"Unsupported Huginn {key}: {values[key]!r}")
        if values.get("test_time_noise", 0) != 0:
            raise ValueError("Huginn test-time noise is unsupported")
        heads = values.get("n_heads", cls.n_heads)
        if values.get("num_key_value_heads", heads) != heads:
            raise ValueError("Huginn-0125 requires equal query and KV head counts")
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})

    def to_dict(self) -> dict:
        return {"model_type": "huginn_raven", "architectures": ["RavenForCausalLM"], **asdict(self)}


class HuginnAttention(nn.Module):
    def __init__(self, config: HuginnConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.Wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.qk_bias = nn.Parameter(torch.zeros(2, 1, config.n_heads, config.head_dim))
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, hidden, freqs, batch, cache):
        count = hidden.shape[0]
        q, k, v = self.Wqkv(hidden).split(self.n_heads * self.head_dim, dim=-1)
        shape = (count, self.n_heads, self.head_dim)
        q = q.view(shape) + self.qk_bias[0]
        k = k.view(shape) + self.qk_bias[1]
        v = v.view(shape)
        # Huginn rotates adjacent pairs, unlike split-half RoPE architectures.
        pair = torch.stack((q, k), dim=0).float().reshape(2, count, self.n_heads, -1, 2)
        cos, sin = freqs[..., 0].unsqueeze(0), freqs[..., 1].unsqueeze(0)
        real = pair[..., 0] * cos - pair[..., 1] * sin
        imag = pair[..., 1] * cos + pair[..., 0] * sin
        q, k = torch.stack((real, imag), dim=-1).flatten(-2).to(hidden.dtype).unbind(0)
        cache._write_prepared(self.layer_idx, batch, k, v)
        attended = cache._attend_prepared(self.layer_idx, batch, q)
        return self.proj(attended.reshape(count, -1))


class HuginnMLP(nn.Module):
    def __init__(self, config: HuginnConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 2 * config.intermediate_size, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, hidden):
        gate, up = self.fc(hidden).chunk(2, dim=-1)
        return self.proj(F.silu(gate) * up)


class HuginnBlock(nn.Module):
    def __init__(self, config: HuginnConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = HuginnAttention(config, layer_idx)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = HuginnMLP(config)
        self.norm_3 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_4 = RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, hidden, freqs, batch, cache):
        hidden = self.norm_2(self.attn(self.norm_1(hidden), freqs, batch, cache) + hidden)
        return self.norm_4(self.mlp(self.norm_3(hidden)) + hidden)


class HuginnForCausalLM(nn.Module):
    """Huginn-0125 with eager KV boundaries and a graphable recurrent core."""

    requires_boundary_kv = True

    def __init__(self, config: HuginnConfig) -> None:
        super().__init__()
        self.config = config
        self.recurrent_kv_layers = tuple(
            range(
                config.n_layers_in_prelude,
                config.n_layers_in_prelude + config.n_layers_in_recurrent_block,
            )
        )
        self.coda_kv_layers = tuple(
            range(config.n_layers_in_prelude + config.n_layers_in_recurrent_block, config.n_layers)
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                prelude=nn.ModuleList(
                    HuginnBlock(config, i) for i in range(config.n_layers_in_prelude)
                ),
                adapter=nn.Linear(2 * config.n_embd, config.n_embd, bias=False),
                core_block=nn.ModuleList(
                    HuginnBlock(config, config.n_layers_in_prelude + i)
                    for i in range(config.n_layers_in_recurrent_block)
                ),
                coda=nn.ModuleList(
                    HuginnBlock(
                        config, config.n_layers_in_prelude + config.n_layers_in_recurrent_block + i
                    )
                    for i in range(config.n_layers_in_coda)
                ),
                ln_f=RMSNorm(config.n_embd, eps=config.norm_eps),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        inv = 1.0 / (
            config.rope_base
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        angles = torch.outer(torch.arange(config.block_size, dtype=torch.float32), inv)
        freqs = torch.stack((angles.cos(), angles.sin()), dim=-1)[None, :, None]
        self.register_buffer("freqs_cis", freqs, persistent=True)

    def _freqs(self, batch):
        return self.freqs_cis[0].index_select(0, batch.position_ids)

    def prelude_prepared(
        self, tokens, batch: "KVMetadata", cache: "KVCacheManager", *, generators=None
    ):
        if generators is not None and len(generators) != len(tokens):
            raise ValueError("one request generator is required per token row")
        hidden = self.transformer.wte(tokens) * math.sqrt(self.config.n_embd)
        freqs = self._freqs(batch)
        for block in self.transformer.prelude:
            hidden = block(hidden, freqs, batch, cache)
        state = torch.empty_like(hidden)
        start = 0
        while start < len(state):
            end = start + 1
            generator = generators[start] if generators is not None else None
            while end < len(state) and (generators is None or generators[end] is generator):
                end += 1
            torch.nn.init.trunc_normal_(
                state[start:end],
                std=self.config.initializer_range,
                a=-3 * self.config.initializer_range,
                b=3 * self.config.initializer_range,
                generator=generator,
            )
            start = end
        state = state * math.sqrt(self.config.n_embd)
        return torch.cat((state, hidden), dim=-1)

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate=True):
        state, injection = hidden.split(self.config.n_embd, dim=-1)
        state = self.transformer.adapter(torch.cat((state, injection), dim=-1))
        freqs = self._freqs(batch)
        for block in self.transformer.core_block:
            state = block(state, freqs, batch, cache)
        gate = (
            torch.full((len(hidden),), -1e4, device=hidden.device, dtype=hidden.dtype)
            if compute_gate
            else None
        )
        return torch.cat((state, injection), dim=-1), gate

    def recurrent(self, hidden, request_ids, depths, positions, cache, *, compute_gate=True):
        batch = cache._prepare_batch(request_ids, depths, positions)
        return self.recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)

    def coda_prepared(
        self, hidden, batch: "KVMetadata", cache: "KVCacheManager", *, compute_logits=True
    ):
        state, _ = hidden.split(self.config.n_embd, dim=-1)
        state = self.transformer.ln_f(state)
        freqs = self._freqs(batch)
        for block in self.transformer.coda:
            state = block(state, freqs, batch, cache)
        return self.lm_head(self.transformer.ln_f(state)) if compute_logits else None

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "HuginnForCausalLM":
        """Strictly stream safetensors into a meta model without Hub code execution."""
        from safetensors import safe_open

        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        folder = Path(path_or_repo).expanduser()
        if not folder.is_dir():
            from vllm_rlt.models import resolve_model_config

            source, _, _ = resolve_model_config(path_or_repo, revision=revision)
            folder = Path(source)
        config = HuginnConfig.from_dict(json.loads((folder / "config.json").read_text()))
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
        expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
        discovered = {}
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    if name in discovered or name not in expected:
                        raise ValueError(f"Duplicate or unexpected checkpoint tensor: {name}")
                    if tuple(checkpoint.get_slice(name).get_shape()) != expected[name]:
                        raise ValueError(f"Checkpoint shape mismatch for {name}")
                    if index is not None and index.get(name) != shard.name:
                        raise ValueError(f"Checkpoint index mismatch for {name}")
                    discovered[name] = shard
        if set(discovered) != set(expected) or (index is not None and set(index) != set(expected)):
            raise ValueError(
                f"Checkpoint keys differ: missing={sorted(set(expected) - set(discovered))}"
            )
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    tensor = checkpoint.get_tensor(name)
                    if not tensor.is_floating_point():
                        raise ValueError(f"Checkpoint tensor must be floating point: {name}")
                    module_path, tensor_name = name.rsplit(".", 1) if "." in name else ("", name)
                    module = model.get_submodule(module_path)
                    value = tensor.to(
                        device=device, dtype=torch.float32 if name == "freqs_cis" else dtype
                    )
                    if name == "freqs_cis":
                        module.register_buffer(tensor_name, value, persistent=True)
                    else:
                        setattr(module, tensor_name, nn.Parameter(value, requires_grad=False))
        if not torch.equal(model.lm_head.weight, model.transformer.wte.weight):
            raise ValueError("Tied embedding and LM head checkpoint tensors differ")
        model.lm_head.weight = model.transformer.wte.weight
        return model.eval()
