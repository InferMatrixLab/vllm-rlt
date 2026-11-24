# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini text backbone with scheduler-visible loop passes."""

import json
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from vllm_rlt.kernels.inkling2mini import (
    KernelBatch,
    QuantizedExperts,
    attention_residual,
    load_ops,
    message_controller,
    packed_attention_plans,
    residual_stack,
    rms_norm,
)


@dataclass(frozen=True)
class Inkling2Config:
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    padded_vocab_size: int
    max_position_embeddings: int
    eos_token_id: int
    width: int
    core_start: int
    core_end: int
    checkpoint_passes: int
    is_varloop: bool
    local_layers: tuple[int, ...]
    window: int
    rel_extent: int
    block_layers: int
    experts_per_token: int
    route_scale: float
    gate_softcap: float
    up_softcap: float
    logit_softcap: float
    rms_norm_eps: float
    qk_clip: float
    rel_clip: float
    log_scale_floor: int
    log_scale_alpha: float
    loop_num_passes: int | None = None

    @property
    def total_ut_steps(self):
        return self.checkpoint_passes if self.loop_num_passes is None else self.loop_num_passes

    @property
    def hidden_size(self):
        # Completed and partial prelude blocks, the core's message bank and a pass flag.
        return (self.core_end // self.block_layers + 1) * self.width + 1

    @property
    def auxiliary_cache_width(self):
        # Raw projections for the four short convolutions on every layer visit.
        per_head = 2 * (self.head_dim + self.width // self.num_key_value_heads)
        return self.num_key_value_heads * (1 << (per_head - 1).bit_length())

    def with_loop_num_passes(self, passes):
        if not self.is_varloop:
            raise ValueError("loop_num_passes requires an Inkling 2 Mini variable-loop checkpoint")
        if type(passes) is not int or passes < 1:
            raise ValueError("loop_num_passes must be a positive integer")
        return replace(self, loop_num_passes=passes)

    @classmethod
    def from_dict(cls, values):
        text = values["text_config"]
        loop = text["loop_config"]
        if type(loop["is_varloop"]) is not bool:
            raise ValueError("Inkling 2 Mini loop_config.is_varloop must be a boolean")
        if text.get("loop_share_cache", False):
            raise ValueError("Inkling 2 Mini currently requires per-pass KV and convolution state")
        for dim in ("head_dim", "num_attention_heads", "num_key_value_heads"):
            if text[dim] != text[f"swa_{dim}"]:
                raise ValueError("Inkling 2 Mini requires matching local and global head shapes")
        config = cls(
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            vocab_size=text["unpadded_vocab_size"],
            padded_vocab_size=text["vocab_size"],
            max_position_embeddings=text["model_max_length"],
            eos_token_id=values["eos_token_id"],
            width=text["hidden_size"],
            core_start=loop["core_start"],
            core_end=loop["core_end"],
            checkpoint_passes=loop["num_passes"],
            is_varloop=loop["is_varloop"],
            local_layers=tuple(text["local_layer_ids"]),
            window=text["sliding_window_size"],
            rel_extent=text["rel_extent"],
            block_layers=text["n_layers_per_attn_res_block"],
            experts_per_token=text["num_experts_per_tok"],
            route_scale=text["route_scale"],
            gate_softcap=text["mlp_gate_softcap"],
            up_softcap=text["mlp_up_softcap"],
            logit_softcap=text["final_logit_softcapping"],
            rms_norm_eps=text["rms_norm_eps"],
            qk_clip=text["qk_l2_clip_threshold"],
            rel_clip=text["rel_l2_clip_threshold"],
            log_scale_floor=text["log_scaling_n_floor"],
            log_scale_alpha=text["log_scaling_alpha"],
        )
        if (
            not 0 < config.core_start < config.core_end < config.num_hidden_layers
            or config.core_start % config.block_layers
            or config.core_end % config.block_layers
            or type(config.checkpoint_passes) is not int
            or config.checkpoint_passes < (1 if config.is_varloop else 2)
        ):
            raise ValueError("invalid Inkling 2 Mini loop configuration")
        passes = text.get("loop_num_passes")
        return config if passes is None else config.with_loop_num_passes(passes)

    def to_dict(self):
        # Runtime model-family checks use this field; the checkpoint retains its JSON schema.
        return {"model_type": "inkling2_loop"}


class _OrderedCodaBatch:
    """Share independent projections while ordering each history write."""

    def __init__(self, target, draft, target_rows):
        self.segments = (
            (target, slice(0, target_rows)),
            (draft, slice(target_rows, None)),
        )

    def attend(self, layer, *projected):
        # Both scopes use their own position, mask and page-table views.
        return torch.cat(
            [
                batch.attend(
                    layer, *(part[rows] if part is not None else None for part in projected)
                )
                for batch, rows in self.segments
            ]
        )

    def conv_mix(
        self,
        layer,
        offset,
        hidden,
        weight,
        out,
        stack,
        last,
        queries,
        norm,
        *,
        residual=None,
        addend=None,
    ):
        output = []
        for batch, rows in self.segments:
            output.append(
                batch.conv_mix(
                    layer,
                    offset,
                    hidden[rows],
                    weight,
                    out[rows],
                    stack[:, rows],
                    last[rows] if last is not None else None,
                    queries,
                    norm,
                    residual=residual[rows] if residual is not None else None,
                    addend=addend[rows] if addend is not None else None,
                )
            )
        return torch.cat(output, dim=1)


class _Layer(nn.Module):
    def __init__(self, config, index, read):
        super().__init__()
        self.index = index
        self.local = index in config.local_layers
        self.boundary = index % config.block_layers == 0
        self.is_core = config.core_start <= index < config.core_end
        self.ops = load_ops()

        def param(name):
            return nn.Parameter(read(name), requires_grad=False)

        for name, key in (
            ("query", "wq_du.weight"),
            ("key", "wk_dv.weight"),
            ("value", "wv_dv.weight"),
            ("relative", "wr_du.weight"),
            ("rel_basis", "rel_logits_proj.proj"),
            ("query_scale", "q_l2_clip.scale"),
            ("rel_scale", "rel_l2_clip.scale"),
            ("sink", "attn_sink_proj.weight"),
            ("output", "wo_ud.weight"),
            ("output_gate", "gate_attn.weight"),
        ):
            setattr(self, name, param(f"attn.{key}"))
        for name, key in (
            ("conv_k", "attn.k_sconv"),
            ("conv_v", "attn.v_sconv"),
            ("conv_attn", "attn_sconv"),
            ("conv_mlp", "mlp_sconv"),
        ):
            weight = (
                read(f"{key}.static_w")
                if not self.local and name in ("conv_k", "conv_v")
                else read(f"{key}.weight").squeeze(1)
            )
            setattr(self, name, nn.Parameter(weight, requires_grad=False))
        self._pack_attention_weights()
        if not self.local:
            self.dynamic_k = param("attn.k_sconv.weight_proj")
            self.dynamic_v = param("attn.v_sconv.weight_proj")
        names = ("attn_norm",) if self.local else ("q_input_norm", "k_input_norm", "v_input_norm")
        self.input_norms = nn.ParameterList(param(f"{name}.weight") for name in names)
        self.mlp_norm = param("mlp_norm.weight")
        self.register_buffer("attn_norm_rows", torch.stack(tuple(self.input_norms)))
        self.mlp = QuantizedExperts(config, lambda name: read(f"mlp.{name}"))
        if self.is_core:
            streams = ("attn", "k", "v", "mlp")
            self.controller_down = nn.Parameter(
                torch.stack([read(f"controllers.{stream}.down_weight") for stream in streams]),
                requires_grad=False,
            )
            self.controller_up = nn.Parameter(
                torch.stack(
                    [
                        torch.cat(
                            [
                                read(f"controllers.{stream}.{gate}_weight")
                                for gate in ("gain", "forget")
                            ]
                        )
                        for stream in streams
                    ]
                ),
                requires_grad=False,
            )

    def _pack_attention_weights(self):
        weights = (self.query, self.relative, self.output_gate, self.sink, self.key, self.value)
        self.projection_sizes = tuple(weight.shape[0] for weight in weights)
        joined = torch.cat(weights)
        padding = -len(joined) % 8
        if padding:
            joined = F.pad(joined, (0, 0, 0, padding))
        self.qkvr = nn.Parameter(joined, requires_grad=False)
        self.register_buffer(
            "shifted_rel_proj", self.ops.sconv.shift_rel_proj(self.rel_basis, self.local)
        )
        del self.query, self.relative, self.output_gate, self.sink, self.key, self.value

    def _attention(self, inputs, batch):
        line = self.ops.linear.linear
        q_input, k_input, v_input = inputs
        sizes = self.projection_sizes
        if self.local:
            packed = line(q_input, self.qkvr)
            q, r, gate, sinks, k, v = packed[:, : sum(sizes)].split(sizes, dim=-1)
            k_dynamic = v_dynamic = None
        else:
            q_side = sum(sizes[:4])
            k_end = q_side + sizes[4]
            projected = line(q_input, self.qkvr[: q_side + (-q_side % 8)])
            q, r, gate, sinks = projected[:, :q_side].split(sizes[:4], dim=-1)
            k = line(k_input, self.qkvr[q_side:k_end])
            v = line(v_input, self.qkvr[k_end : k_end + sizes[5]])
            k_dynamic = F.linear(k_input, self.dynamic_k)
            v_dynamic = F.linear(v_input, self.dynamic_v)
        return line(batch.attend(self, q, r, gate, sinks, k, v, k_dynamic, v_dynamic), self.output)


class Inkling2ForCausalLM(nn.Module):
    requires_boundary_kv = True
    supports_async_scheduling = True
    # Keep the per-pass attention and convolution page mapping on the host.
    supports_device_kv_metadata = False
    supports_packed_recurrent_graphs = True
    supports_packed_prefill_graphs = True
    supports_recurrent_gate = False
    supports_mixed_prefill = True
    supports_wavefront_prefill = True
    requires_ordered_pipeline_coda = True
    supports_cuda_graphs = True
    # A VarLoop checkpoint selects its pass count when the engine starts.
    fixed_loop_count = True
    interleaved_kv = True
    state_dtype = torch.bfloat16
    batch_type = KernelBatch

    def __init__(self, config: Inkling2Config, read=None):
        super().__init__()
        if read is None:
            raise ValueError("Inkling 2 Mini requires a checkpoint")
        self.config = config
        self.embed = nn.Parameter(read("embed.weight"), requires_grad=False)
        self.embed_norm = nn.Parameter(read("embed_norm.weight"), requires_grad=False)
        self.norm = nn.Parameter(read("norm.weight"), requires_grad=False)
        self.head = nn.Parameter(read("unembed.weight"), requires_grad=False)
        self.prelude_kv_layers = tuple(range(config.core_start))
        self.recurrent_kv_layers = tuple(range(config.core_start, config.core_end))
        self.coda_kv_layers = tuple(range(config.core_end, config.num_hidden_layers))
        self.layers = nn.ModuleList(
            [
                _Layer(config, index, lambda name: read(f"layers.{index}.{name}"))
                for index in range(config.num_hidden_layers)
            ]
        )
        factor = config.width**-0.5
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                (
                    read(f"attn_res.attn_queries.{i - 1}").view(-1, config.width).float() * factor
                ).contiguous(),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        mlp_queries = torch.stack(
            [
                read(f"attn_res.mlp_queries.{i}").view(1, config.width)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.register_buffer("mlp_queries", mlp_queries.float() * factor)
        self.register_buffer(
            "output_query", read("attn_res.output_query").view(1, -1).float() * factor
        )
        main = read("middle_cycle_cross_pass_queries").view(-1, 2, config.width)
        kv = read("middle_cycle_cross_pass_kv_queries").view_as(main)
        self.message_queries = nn.Parameter(
            torch.cat((main[:, :1], kv, main[:, 1:]), dim=1), requires_grad=False
        )
        self.prelude_blocks = config.core_start // config.block_layers + 1

    def _run_layers(self, blocks, indices, batch, bank=None, has_message=None, *, final=False):
        cfg = self.config
        size = (cfg.num_hidden_layers + cfg.block_layers - 1) // cfg.block_layers + 1
        stack = blocks.new_empty(size, len(blocks), cfg.width)
        num_blocks = blocks.shape[1] - 1
        stack[: num_blocks + 1].copy_(residual_stack(blocks))
        ops = load_ops()
        inputs = None
        for index in indices:
            layer = self.layers[index]
            partial = stack[num_blocks]
            messages = (
                None
                if bank is None
                else attention_residual(
                    bank, self.message_queries[index - cfg.core_start]
                ).transpose(0, 1)
            )
            if inputs is None:
                if index == 0:
                    inputs = rms_norm(partial, layer.input_norms[0], cfg.rms_norm_eps)[None]
                else:
                    inputs = ops.attn_res.attn_res(
                        stack[:num_blocks],
                        partial,
                        None,
                        self.attn_queries[index - 1],
                        layer.attn_norm_rows if messages is None else None,
                        cfg.rms_norm_eps,
                        True,
                    )
            if layer.local:
                inputs = inputs.expand(3, -1, -1)
            if messages is not None:
                inputs = message_controller(
                    inputs,
                    messages[:3],
                    layer.controller_down[:3],
                    layer.controller_up[:3],
                    layer.attn_norm_rows.expand(3, -1),
                    cfg.rms_norm_eps,
                    has_message,
                )
            attn = layer._attention(inputs, batch)
            attn_out = torch.empty_like(partial)
            mlp_norm = layer.mlp_norm[None]
            mlp_in = batch.conv_mix(
                index,
                2 * cfg.head_dim,
                attn,
                layer.conv_attn,
                attn_out,
                stack[: num_blocks + layer.boundary],
                None if layer.boundary else partial,
                self.mlp_queries[index],
                mlp_norm if messages is None else None,
            )
            if messages is not None:
                mlp_in = message_controller(
                    mlp_in,
                    messages[3:],
                    layer.controller_down[3:],
                    layer.controller_up[3:],
                    mlp_norm,
                    cfg.rms_norm_eps,
                    has_message,
                )
            mlp = layer.mlp(mlp_in[0])
            num_blocks += layer.boundary
            if index + 1 < cfg.num_hidden_layers:
                query = self.attn_queries[index]
                next_norm = self.layers[index + 1].attn_norm_rows if bank is None else None
            else:
                query, next_norm = self.output_query, self.norm[None]
            inputs = batch.conv_mix(
                index,
                2 * cfg.head_dim + cfg.width // cfg.num_key_value_heads,
                mlp,
                layer.conv_mlp,
                stack[num_blocks],
                stack[:num_blocks],
                None,
                query,
                next_norm,
                residual=None if layer.boundary else partial,
                addend=attn_out,
            )
        return inputs[0] if final else stack[: num_blocks + 1].transpose(0, 1)

    def recurrent_graph_variant(self, batch):
        rows = getattr(batch, "rows", None)
        if rows is None:
            # A device-owned pipeline selects the first-pass path from hidden state.
            return "mixed"
        first = {depth == 0 for _, depth, _ in rows}
        if len(first) != 1:
            return "mixed"
        return "first" if True in first else "later"

    def packed_recurrent_graph_plan(self, batch):
        return packed_attention_plans(batch, self.config)

    def prelude_prepared(self, tokens, metadata, cache, *, generators=None):
        batch = self.batch_type(cache, metadata, self.config)
        embed = F.embedding(tokens, self.embed)
        initial = rms_norm(embed, self.embed_norm, self.config.rms_norm_eps)
        initial = initial.to(self.state_dtype).unsqueeze(1)
        blocks = self._run_layers(initial, self.prelude_kv_layers, batch)
        tail = blocks.new_zeros(
            (len(tokens), self.config.hidden_size - blocks.shape[1] * blocks.shape[2])
        )
        return torch.cat((blocks.flatten(1), tail), dim=-1)

    def recurrent_prepared(self, hidden, metadata, cache, *, compute_gate=True):
        batch = self.batch_type(cache, metadata, self.config)
        rows = len(batch.positions)
        packed = hidden[:rows, :-1].unflatten(-1, (-1, self.config.width))
        prelude, bank = packed[:, : self.prelude_blocks], packed[:, self.prelude_blocks :]
        has_message = hidden[:rows, -1].view(1, -1, 1).ne(0)
        variant = (
            metadata.graph_variant
            if hasattr(metadata, "graph_variant")
            else self.recurrent_graph_variant(metadata)
        )
        if variant == "first":
            bank = None
        elif variant == "later":
            has_message = None
        blocks = self._run_layers(prelude, self.recurrent_kv_layers, batch, bank, has_message)
        # The first completed core entry repeats the retained prelude partial.
        result = torch.cat(
            (
                packed[:, : self.prelude_blocks].flatten(1),
                blocks[:, self.prelude_blocks :].flatten(1),
                hidden.new_ones((rows, 1)),
            ),
            dim=-1,
        )
        if rows < len(hidden):
            result = F.pad(result, (0, 0, 0, len(hidden) - rows))
        return result, None

    def recurrent(self, hidden, request_ids, depths, positions, cache, *, compute_gate=True):
        return self.recurrent_prepared(
            hidden,
            cache._prepare_batch(request_ids, depths, positions),
            cache,
            compute_gate=compute_gate,
        )

    def _coda_inputs(self, hidden, metadata, cache):
        batch = self.batch_type(cache, metadata, self.config)
        blocks = hidden[: len(batch.positions), :-1].unflatten(-1, (-1, self.config.width))
        return batch, blocks

    def coda_hidden_prepared(self, hidden, metadata, cache):
        """Update the boundary history and return the normalized LM head input."""
        batch, blocks = self._coda_inputs(hidden, metadata, cache)
        return self._run_layers(blocks, self.coda_kv_layers, batch, final=True)

    def project_coda(self, state):
        logits = F.linear(state, self.head)[..., : self.config.vocab_size]
        cap = self.config.logit_softcap
        return (logits / cap).tanh() * cap

    def coda_prepared(self, hidden, metadata, cache, *, compute_logits=True):
        if compute_logits:
            return self.project_coda(self.coda_hidden_prepared(hidden, metadata, cache))
        batch, blocks = self._coda_inputs(hidden, metadata, cache)
        self._run_layers(blocks, self.coda_kv_layers, batch)
        return None

    def coda_pair(self, target_hidden, target_metadata, draft_hidden, draft_metadata, cache):
        target, target_blocks = self._coda_inputs(target_hidden, target_metadata, cache)
        draft, draft_blocks = self._coda_inputs(draft_hidden, draft_metadata, cache)
        # Order attention and convolution writes before the draft reads target history.
        batch = _OrderedCodaBatch(target, draft, len(target_blocks))
        blocks = torch.cat((target_blocks, draft_blocks))
        state = self._run_layers(blocks, self.coda_kv_layers, batch, final=True)
        return self.project_coda(state).argmax(-1).view(2, -1)

    @classmethod
    def from_pretrained(
        cls, path, *, revision=None, device="cpu", dtype=torch.bfloat16, config=None
    ):
        if torch.device(device).type != "cuda" or dtype != torch.bfloat16:
            raise ValueError("Inkling 2 Mini currently requires CUDA and BF16 model dtype")
        folder = Path(path).expanduser()
        if not folder.is_dir():
            from . import resolve_model_config

            source, _, _ = resolve_model_config(path, revision=revision)
            folder = Path(source)
        if config is None:
            config = Inkling2Config.from_dict(json.loads((folder / "config.json").read_text()))
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise ValueError("Inkling 2 Mini requires SM10x for the vLLM operators")
        load_ops()
        index_path = folder / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())["weight_map"]
            names = sorted({file for key, file in index.items() if key.startswith("model.llm.")})
        else:
            names = ["model.safetensors"]
        if not names or any(
            Path(name).name != name or not name.endswith(".safetensors") for name in names
        ):
            raise ValueError(
                "Inkling 2 Mini requires local safetensors entries in its checkpoint index"
            )
        with ExitStack() as stack:
            weights = {}
            for name in names:
                shard = stack.enter_context(safe_open(folder / name, framework="pt", device="cpu"))
                weights.update(
                    {name: shard for name in shard.keys() if name.startswith("model.llm.")}
                )

            def read(name):
                key = f"model.llm.{name}"
                if key not in weights:
                    raise ValueError(f"Inkling 2 Mini checkpoint is missing {key}")
                return weights[key].get_tensor(key).to(device)

            return cls(config, read).eval()
