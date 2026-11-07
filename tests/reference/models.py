# SPDX-License-Identifier: Apache-2.0
"""Independent dense model oracles for correctness validation, not engine execution."""

import torch
from torch.nn import functional as F


@torch.inference_mode()
def dense_reference(model, token_ids, loops, *, return_kv=False):
    """Functional causal oracle: no model modules, paged cache, or recurrent API.

    Mirrors the equations of the pinned official model, including split-half
    RoPE, GQA, all four sandwich norms, and norm at each loop boundary.
    Returns every depth so intermediate hidden states and gate logits are checked.
    With return_kv=True, also return the true pre-GQA K/V planes per loop/layer.
    """
    config = model.config
    weights = model.state_dict()
    hidden = weights["model.embed_tokens.weight"][token_ids]
    length = len(token_ids)
    positions = torch.arange(length, dtype=torch.float32, device=hidden.device)
    frequencies = config.rope_theta ** (
        -torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=hidden.device)
        / config.head_dim
    )
    angles = torch.outer(positions, frequencies)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)[:, None, :].to(hidden.dtype)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)[:, None, :].to(hidden.dtype)
    future = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(diagonal=1)

    def norm(value, name):
        normalized = value.float() / torch.sqrt(
            value.float().square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
        return normalized.to(value.dtype) * weights[name + ".weight"]

    def linear(value, name):
        return value @ weights[name + ".weight"].T

    def rotary(value):
        split = config.head_dim // 2
        rotated = torch.cat([-value[..., split:], value[..., :split]], dim=-1)
        return value * cos + rotated * sin

    outputs, kv_planes = [], []
    for _ in range(loops):
        layer_kv = []
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            attention_input = norm(hidden, prefix + ".input_layernorm")
            q = rotary(
                linear(attention_input, prefix + ".self_attn.q_proj").reshape(
                    length, config.num_attention_heads, config.head_dim
                )
            )
            k = rotary(
                linear(attention_input, prefix + ".self_attn.k_proj").reshape(
                    length, config.num_key_value_heads, config.head_dim
                )
            )
            v = linear(attention_input, prefix + ".self_attn.v_proj").reshape(
                length, config.num_key_value_heads, config.head_dim
            )
            if return_kv:
                layer_kv.append((k.clone(), v.clone()))
            groups = config.num_attention_heads // config.num_key_value_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            scores = torch.einsum("thd,shd->hts", q, k) / config.head_dim**0.5
            scores.masked_fill_(future, -torch.inf)
            probabilities = scores.float().softmax(dim=-1).to(hidden.dtype)
            attention = torch.einsum("hts,shd->thd", probabilities, v).reshape(length, -1)
            attention = linear(attention, prefix + ".self_attn.o_proj")
            hidden = hidden + norm(attention, prefix + ".input_layernorm_2")
            mlp_input = norm(hidden, prefix + ".post_attention_layernorm")
            mlp = F.silu(linear(mlp_input, prefix + ".mlp.gate_proj")) * linear(
                mlp_input, prefix + ".mlp.up_proj"
            )
            hidden = hidden + norm(
                linear(mlp, prefix + ".mlp.down_proj"), prefix + ".post_attention_layernorm_2"
            )
        hidden = norm(hidden, "model.norm")
        gate = (
            linear(hidden, "model.early_exit_gate") + weights["model.early_exit_gate.bias"]
        ).squeeze(-1)
        outputs.append((hidden.clone(), gate, linear(hidden, "lm_head")))
        if return_kv:
            kv_planes.append(layer_kv)
    return (outputs, kv_planes) if return_kv else outputs


@torch.inference_mode()
def dense_nanbeige_reference(model, token_ids, loops):
    """Functional causal oracle for Nanbeige: no modules, paged cache, or recurrent API."""
    config = model.config
    weights = model.state_dict()
    hidden = weights["model.embed_tokens.weight"][token_ids]
    length = len(token_ids)
    positions = torch.arange(length, dtype=torch.float32, device=hidden.device)
    frequencies = config.rope_theta ** (
        -torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=hidden.device)
        / config.head_dim
    )
    angles = torch.outer(positions, frequencies)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)[:, None, :].to(hidden.dtype)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)[:, None, :].to(hidden.dtype)
    future = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(diagonal=1)

    def norm(value, name):
        normalized = value.float() / torch.sqrt(
            value.float().square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
        return normalized.to(value.dtype) * weights[name + ".weight"]

    def linear(value, name):
        return value @ weights[name + ".weight"].T

    def rotary(value):
        split = config.head_dim // 2
        rotated = torch.cat([-value[..., split:], value[..., :split]], dim=-1)
        return value * cos + rotated * sin

    outputs = []
    for _ in range(loops):
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            attention_input = norm(hidden, prefix + ".input_layernorm")
            q = rotary(
                linear(attention_input, prefix + ".self_attn.q_proj").reshape(
                    length, config.num_attention_heads, config.head_dim
                )
            )
            k = rotary(
                linear(attention_input, prefix + ".self_attn.k_proj").reshape(
                    length, config.num_key_value_heads, config.head_dim
                )
            )
            v = linear(attention_input, prefix + ".self_attn.v_proj").reshape(
                length, config.num_key_value_heads, config.head_dim
            )
            groups = config.num_attention_heads // config.num_key_value_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            scores = torch.einsum("thd,shd->hts", q, k) / config.head_dim**0.5
            scores.masked_fill_(future, -torch.inf)
            probabilities = scores.float().softmax(dim=-1).to(hidden.dtype)
            attention = torch.einsum("hts,shd->thd", probabilities, v).reshape(length, -1)
            attention = linear(attention, prefix + ".self_attn.o_proj")
            hidden = hidden + attention
            mlp_input = norm(hidden, prefix + ".post_attention_layernorm")
            mlp = F.silu(linear(mlp_input, prefix + ".mlp.gate_proj")) * linear(
                mlp_input, prefix + ".mlp.up_proj"
            )
            hidden = hidden + linear(mlp, prefix + ".mlp.down_proj")
        if not getattr(config, "skip_loop_final_norm", False):
            hidden = norm(hidden, "model.norm")
        outputs.append((hidden.clone(), linear(hidden, "lm_head")))
    return outputs
