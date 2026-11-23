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
    conv_taps: int
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
    kernel_backend: str = "rlt"

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
        if self.kernel_backend == "vllm":
            per_head = 2 * (self.head_dim + self.width // self.num_key_value_heads)
            return self.num_key_value_heads * (1 << (per_head - 1).bit_length())
        return 2 * (self.num_key_value_heads * self.head_dim + self.width)

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
            conv_taps=text["sconv_kernel_size"],
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


def _mix(blocks, queries):
    values = F.rms_norm(blocks, (blocks.shape[-1],), eps=1e-6)
    scores = F.linear(values, queries) * blocks.shape[-1] ** -0.5
    return scores.softmax(dim=1).transpose(1, 2) @ values


def _clip(x, threshold):
    x = x.float()
    scale = threshold / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-6)
    return x * scale.clamp(max=1)


def _norm(x, weight, eps):
    return F.rms_norm(x.float(), (x.shape[-1],), eps=eps).mul(weight).to(weight.dtype)


def _expert_weight(read, name):
    """Expand the checkpoint's E2M1 or E4M3 weights into BF16 for the first backend."""
    weight = read(f"{name}_weight")
    scale = read(f"{name}_weight_scale")
    fp4 = weight.dtype == torch.uint8
    experts, rows, cols = weight.shape
    output = torch.empty(
        (experts, rows, cols * (2 if fp4 else 1)), device=weight.device, dtype=torch.bfloat16
    )
    if fp4:
        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
        lut = torch.cat((lut, -lut))
        byte = torch.arange(256, device=weight.device)
        pairs = torch.stack((lut[byte & 15], lut[byte >> 4]), dim=-1)
        outer = read(f"{name}_weight_amax") / read(f"{name}_weight_outer_scale_denominator")
    for start in range(0, experts, 4):
        chunk = slice(start, start + 4)
        scales = scale[chunk].float()
        if fp4:
            decoded = pairs[weight[chunk].long()].reshape(*scales.shape, 16)
            decoded = decoded * scales.unsqueeze(-1) * outer[chunk, None, None, None]
        else:
            decoded = weight[chunk].float().reshape(*scales.shape, 32)
            decoded = decoded * torch.exp2(scales - 127).unsqueeze(-1)
        output[chunk].copy_(decoded.flatten(-2))
    return output


class _Experts(nn.Module):
    def __init__(self, config, read):
        super().__init__()
        self.config = config
        self.router = nn.Parameter(read("gate.weight").float(), requires_grad=False)
        self.bias = nn.Parameter(read("gate.bias"), requires_grad=False)
        self.up = nn.Parameter(_expert_weight(read, "experts.w13"), requires_grad=False)
        self.down = nn.Parameter(_expert_weight(read, "experts.w2"), requires_grad=False)
        self.register_buffer(
            "expert_ids", torch.arange(len(self.router), device=self.router.device)
        )

    def forward(self, x):
        config = self.config
        logits = F.linear(x.float(), self.router)
        selected = (logits.sigmoid() + self.bias).topk(config.experts_per_token, dim=-1).indices
        routing = F.logsigmoid(logits.gather(1, selected)).softmax(dim=-1) * config.route_scale
        experts, order = selected.flatten().sort()
        offsets = torch.searchsorted(experts, self.expert_ids, right=True, out_int32=True)
        tokens = order // config.experts_per_token
        # w13 stores gate and up interleaved within each output pair.
        gate_up = F.grouped_mm(x[tokens], self.up.transpose(1, 2), offs=offsets)
        gate, up = gate_up.float().unflatten(-1, (-1, 2)).unbind(-1)
        gate = gate.sigmoid() * config.gate_softcap * (gate / config.gate_softcap).tanh()
        up = config.up_softcap * (up / config.up_softcap).tanh()
        expert_input = (gate * up * routing.flatten()[order, None]).to(x.dtype)
        projected = F.grouped_mm(expert_input, self.down.transpose(1, 2), offs=offsets)
        # Each route is unique. Restore token-major ordering for a deterministic sum.
        ordered = torch.empty_like(projected)
        ordered[order] = projected
        return ordered.float().reshape(len(x), config.experts_per_token, -1).sum(dim=1).to(x.dtype)


class _Batch:
    def __init__(self, cache, metadata, config):
        self.cache = cache
        self.metadata = metadata
        self.positions = metadata.position_ids[: len(metadata.write_blocks)].contiguous()
        lag = torch.arange(config.conv_taps, device=self.positions.device)
        history = self.positions[:, None] - lag
        block_size = cache.block_size
        columns = history.clamp(min=0) // block_size
        cu = metadata.cu_seqlens_q
        if cu is None:
            pages = metadata.block_tables[: len(self.positions)].gather(1, columns)
        else:
            rows = torch.arange(len(self.positions), device=self.positions.device, dtype=cu.dtype)
            sequences = torch.searchsorted(cu[1:], rows, right=True, out_int32=True)
            pages = metadata.block_tables[sequences[:, None], columns]
        self.history_pages = pages.clamp(min=0)
        self.history_offsets = history % block_size
        self.history_valid = (history >= 0) & (pages >= 0)

    def convolve(self, layer, start, x, weights, dynamic=None):
        state = self.cache.auxiliary_cache[:, layer, :, start : start + x.shape[1]]
        batch = self.metadata
        state[batch.write_blocks, batch.write_offsets] = x
        history = state[self.history_pages, self.history_offsets].float()
        history = history.masked_fill(~self.history_valid[..., None], 0)
        taps = weights.shape[0]
        if dynamic is None:
            result = (history * weights.float()).sum(dim=1)
            scale = (taps + 1) ** -0.5
        else:
            heads = dynamic.shape[2]
            kernel = weights.float().unflatten(-1, (heads, -1)) + dynamic.float().unsqueeze(-1)
            result = (history.unflatten(-1, (heads, -1)) * kernel).sum(dim=1).flatten(1)
            scale = (2 * taps + 1) ** -0.5
        return (result + x.float()) * scale

    def attend(self, layer, q, k, v, bias, sink, window):
        from vllm_rlt.kernels.inkling2_attention import paged_attention

        self.cache._write_prepared(layer, self.metadata, k, v)
        keys, values = self.cache.key_cache[:, layer], self.cache.value_cache[:, layer]
        batch = self.metadata
        if batch.cu_seqlens_q is not None:
            from vllm_rlt.kernels.triton_attention import paged_prefill_attention

            return paged_prefill_attention(
                q,
                keys,
                values,
                batch.block_tables,
                batch.context_lengths,
                batch.cu_seqlens_q,
                batch.max_seqlen_q,
                window=window or 0,
                compensate=True,
                relative_bias=bias,
                sinks=sink,
            )
        return paged_attention(q, keys, values, bias, sink, batch, window)


class _Layer(nn.Module):
    def __init__(self, config, index, read):
        super().__init__()
        self.config, self.index = config, index
        self.local = index in config.local_layers
        self.boundary = index % config.block_layers == 0
        self.is_core = config.core_start <= index < config.core_end
        self.native = config.kernel_backend == "vllm"
        self.mix, self.norm = _mix, _norm
        experts = _Experts
        if self.native:
            from vllm_rlt.kernels.inkling2_vllm import (
                QuantizedExperts,
                attention_residual,
                load_ops,
                rms_norm,
            )

            experts = QuantizedExperts
            self.mix, self.norm = attention_residual, rms_norm
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
            if not self.native and (self.local or name not in ("conv_k", "conv_v")):
                weight = weight.flip(-1).T.contiguous()
            setattr(self, name, nn.Parameter(weight, requires_grad=False))
        if self.native:
            self._pack_attention_weights()
        if not self.local:
            self.dynamic_k = param("attn.k_sconv.weight_proj")
            self.dynamic_v = param("attn.v_sconv.weight_proj")
        names = ("attn_norm",) if self.local else ("q_input_norm", "k_input_norm", "v_input_norm")
        self.input_norms = nn.ParameterList(param(f"{name}.weight") for name in names)
        self.mlp_norm = param("mlp_norm.weight")
        if self.native:
            self.register_buffer("attn_norm_rows", torch.stack(tuple(self.input_norms)))
        self.mlp = experts(config, lambda name: read(f"mlp.{name}"))
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

    def _blend(self, dest, messages, streams, has_message):
        pair = torch.stack((dest, messages), dim=-2)
        pair = F.rms_norm(pair, (dest.shape[-1],), eps=1e-6).flatten(-2)
        pair = pair.to(self.controller_down.dtype)
        rank = torch.bmm(pair, self.controller_down[streams].transpose(1, 2))
        gains = torch.bmm(
            rank, self.controller_up[streams].transpose(1, 2), out_dtype=torch.float32
        )
        gain, forget = gains.tanh().chunk(2, dim=-1)
        return torch.where(has_message, (1 - forget) * dest + gain * messages, dest)

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

    def _native_attention(self, inputs, batch):
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

    def _attention(self, inputs, batch):
        cfg = self.config
        q_input, k_input, v_input = inputs
        dim, kv_heads = cfg.head_dim, cfg.num_key_value_heads
        k = F.linear(k_input, self.key)
        v = F.linear(v_input, self.value)
        width = k.shape[-1]
        k_taps = v_taps = None
        if not self.local:
            shape = (-1, kv_heads, cfg.conv_taps)
            k_taps = F.linear(k_input, self.dynamic_k).view(shape).transpose(1, 2)
            v_taps = F.linear(v_input, self.dynamic_v).view(shape).transpose(1, 2)
        k = batch.convolve(self.index, 0, k, self.conv_k, k_taps).view(-1, kv_heads, dim)
        v = batch.convolve(self.index, width, v, self.conv_v, v_taps).view(-1, kv_heads, dim)
        k = _clip(k.to(q_input.dtype), cfg.qk_clip).to(q_input.dtype)
        v = F.rms_norm(v.to(q_input.dtype).float(), (dim,), eps=cfg.rms_norm_eps).to(q_input.dtype)
        q = F.linear(q_input, self.query).view(-1, cfg.num_attention_heads, dim)
        rel = F.linear(q_input, self.relative).view(
            -1, cfg.num_attention_heads, self.rel_scale.shape[-1]
        )
        q = _clip(q.float() * self.query_scale, cfg.qk_clip)
        rel = _clip(rel.float() * self.rel_scale, cfg.rel_clip)
        if not self.local:
            tau = (
                1
                + cfg.log_scale_alpha
                * ((batch.positions.float() + 1) / cfg.log_scale_floor).clamp(min=1).log()
            )
            q = q * tau[:, None, None]
            rel = rel * tau[:, None, None]
        bias = rel.to(q_input.dtype) @ self.rel_basis
        sink = F.linear(q_input, self.sink)
        hidden = batch.attend(
            self.index,
            q.to(q_input.dtype),
            k,
            v,
            bias,
            sink,
            cfg.window if self.local else None,
        ).flatten(1)
        gated = hidden.float() * F.linear(q_input, self.output_gate).float().sigmoid()
        return F.linear(gated.to(q_input.dtype), self.output)

    def forward(self, blocks, attn_query, mlp_query, batch, messages=None, has_message=None):
        cfg = self.config
        partial = blocks[:, -1]
        inputs = (
            partial.unsqueeze(0)
            if attn_query is None
            else self.mix(blocks, attn_query).transpose(0, 1)
        )
        if self.local:
            inputs = inputs.expand(3, -1, -1)
        if messages is not None:
            inputs = self._blend(inputs, messages[:3], slice(0, 3), has_message)
        norms = tuple(self.input_norms) * 3 if self.local else tuple(self.input_norms)
        inputs = tuple(self.norm(x, w, cfg.rms_norm_eps) for x, w in zip(inputs, norms))
        attn = self._attention(inputs, batch)
        offset = 2 * cfg.num_key_value_heads * cfg.head_dim
        attn = batch.convolve(self.index, offset, attn, self.conv_attn)
        if self.boundary:
            blocks = torch.cat((blocks, attn[:, None]), dim=1)
            partial = attn
        else:
            partial = partial + attn
            blocks = torch.cat((blocks[:, :-1], partial[:, None]), dim=1)
        mlp_input = self.mix(blocks, mlp_query).transpose(0, 1)
        if messages is not None:
            mlp_input = self._blend(mlp_input, messages[3:], slice(3, 4), has_message)
        mlp = self.mlp(self.norm(mlp_input[0], self.mlp_norm, cfg.rms_norm_eps))
        mlp = batch.convolve(self.index, offset + cfg.width, mlp, self.conv_mlp)
        return torch.cat((blocks[:, :-1], (partial + mlp)[:, None]), dim=1)


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
    state_dtype = torch.float32

    def __init__(self, config: Inkling2Config, read=None):
        super().__init__()
        if read is None:
            raise ValueError("Inkling 2 Mini requires a checkpoint")
        self.config = config
        self.embed = nn.Parameter(read("embed.weight"), requires_grad=False)
        self.interleaved_kv = config.kernel_backend == "vllm"
        if self.interleaved_kv:
            self.state_dtype = torch.bfloat16
        self.embed_norm = nn.Parameter(read("embed_norm.weight"), requires_grad=False)
        self.norm = nn.Parameter(read("norm.weight"), requires_grad=False)
        head = read("unembed.weight")
        if not self.interleaved_kv:
            head = head[: config.vocab_size]
        self.head = nn.Parameter(head, requires_grad=False)
        self.prelude_kv_layers = tuple(range(config.core_start))
        self.recurrent_kv_layers = tuple(range(config.core_start, config.core_end))
        self.coda_kv_layers = tuple(range(config.core_end, config.num_hidden_layers))
        self.layers = nn.ModuleList(
            [
                _Layer(config, index, lambda name: read(f"layers.{index}.{name}"))
                for index in range(config.num_hidden_layers)
            ]
        )
        self.mix, self.normalize = self.layers[0].mix, self.layers[0].norm
        self.batch_type = _Batch
        if config.kernel_backend == "vllm":
            from vllm_rlt.kernels.inkling2_vllm import KernelBatch

            self.batch_type = KernelBatch
        self.attn_queries = nn.ParameterList(
            [
                nn.Parameter(
                    read(f"attn_res.attn_queries.{i - 1}").view(-1, config.width),
                    requires_grad=False,
                )
                for i in range(1, config.num_hidden_layers)
            ]
        )
        self.mlp_queries = nn.Parameter(
            torch.stack(
                [
                    read(f"attn_res.mlp_queries.{i}").view(1, config.width)
                    for i in range(config.num_hidden_layers)
                ]
            ),
            requires_grad=False,
        )
        self.output_query = nn.Parameter(
            read("attn_res.output_query").view(1, -1), requires_grad=False
        )
        main = read("middle_cycle_cross_pass_queries").view(-1, 2, config.width)
        kv = read("middle_cycle_cross_pass_kv_queries").view_as(main)
        self.message_queries = nn.Parameter(
            torch.cat((main[:, :1], kv, main[:, 1:]), dim=1), requires_grad=False
        )
        self.prelude_blocks = config.core_start // config.block_layers + 1
        if self.interleaved_kv:
            factor = config.width**-0.5
            self.native_attn_queries = nn.ParameterList(
                nn.Parameter((q.float() * factor).contiguous(), requires_grad=False)
                for q in self.attn_queries
            )
            self.register_buffer("native_mlp_queries", self.mlp_queries.float() * factor)
            self.register_buffer("native_output_query", self.output_query.float() * factor)

    def _run_layers(self, blocks, indices, batch, bank=None, has_message=None):
        if self.interleaved_kv:
            return self._run_native_layers(blocks, indices, batch, bank, has_message)
        for index in indices:
            query = self.attn_queries[index - 1] if index else None
            messages = (
                None
                if bank is None
                else self.mix(bank, self.message_queries[index - self.config.core_start]).transpose(
                    0, 1
                )
            )
            blocks = self.layers[index](
                blocks, query, self.mlp_queries[index], batch, messages, has_message
            )
        return blocks

    def _run_native_layers(self, blocks, indices, batch, bank, has_message, *, final=False):
        from vllm_rlt.kernels.inkling2_vllm import load_ops, message_controller, residual_stack

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
                else self.mix(bank, self.message_queries[index - cfg.core_start]).transpose(0, 1)
            )
            if inputs is None:
                if index == 0:
                    inputs = self.normalize(partial, layer.input_norms[0], cfg.rms_norm_eps)[None]
                else:
                    inputs = ops.attn_res.attn_res(
                        stack[:num_blocks],
                        partial,
                        None,
                        self.native_attn_queries[index - 1],
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
            attn = layer._native_attention(inputs, batch)
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
                self.native_mlp_queries[index],
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
                query = self.native_attn_queries[index]
                next_norm = self.layers[index + 1].attn_norm_rows if bank is None else None
            else:
                query, next_norm = self.native_output_query, self.norm[None]
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
        if not self.interleaved_kv:
            return None
        rows = getattr(batch, "rows", None)
        if rows is None:
            # A device-owned pipeline selects the first-pass path from hidden state.
            return "mixed"
        first = {depth == 0 for _, depth, _ in rows}
        if len(first) != 1:
            return "mixed"
        return "first" if True in first else "later"

    def packed_recurrent_graph_plan(self, batch):
        if self.interleaved_kv:
            from vllm_rlt.kernels.inkling2_vllm import packed_attention_plans

            return packed_attention_plans(batch, self.config)
        return None

    def prelude_prepared(self, tokens, metadata, cache, *, generators=None):
        batch = self.batch_type(cache, metadata, self.config)
        embed = F.embedding(tokens, self.embed)
        initial = self.normalize(embed, self.embed_norm, self.config.rms_norm_eps)
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
        if self.interleaved_kv:
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

    def coda_prepared(self, hidden, metadata, cache, *, compute_logits=True):
        batch = self.batch_type(cache, metadata, self.config)
        blocks = hidden[: len(batch.positions), :-1].unflatten(-1, (-1, self.config.width))
        if self.interleaved_kv and compute_logits:
            state = self._run_native_layers(
                blocks, self.coda_kv_layers, batch, None, None, final=True
            )
            logits = F.linear(state, self.head)[..., : self.config.vocab_size]
            cap = self.config.logit_softcap
            return (logits / cap).tanh() * cap
        blocks = self._run_layers(blocks, self.coda_kv_layers, batch)
        if compute_logits:
            state = self.normalize(
                self.mix(blocks, self.output_query)[:, 0], self.norm, self.config.rms_norm_eps
            )
            logits = torch.mm(state, self.head.T, out_dtype=torch.float32)
            cap = self.config.logit_softcap
            return (logits / cap).tanh() * cap
        return None

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
        if config.kernel_backend == "vllm":
            if torch.cuda.get_device_capability(device)[0] != 10:
                raise ValueError("vLLM Inkling kernels currently require SM10x")
        elif not hasattr(F, "grouped_mm"):
            raise ValueError("Inkling 2 Mini requires PyTorch with torch.nn.functional.grouped_mm")
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
