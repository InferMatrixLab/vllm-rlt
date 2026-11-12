# SPDX-License-Identifier: Apache-2.0
"""Dense functional Dory oracle, independent of paged KV and model modules.

The oracle runs a strict one-token-per-step schedule, so it describes the
declared reference execution rather than any particular batching. Every layer
keeps dense K/V history: ``per_loop`` history is keyed by loop depth, while
``last_loop`` reuses a single history that the next loop overwrites.
"""

import math

import torch
import torch.nn.functional as F

EPS = 1e-6


def _justnorm(value, dim=-1):
    value32 = value.float()
    return (value32 * torch.rsqrt(value32.square().sum(dim, keepdim=True) + EPS)).to(value.dtype)


def _slerp(residual, update, alpha, alpha_scale):
    base, target = _justnorm(residual).float(), _justnorm(update).float()
    weight = (alpha.float() * alpha_scale).abs()
    return _justnorm(base + weight * (target - base)).to(residual.dtype)


def _rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rope(value, positions, theta):
    """Split-half RoPE with the runtime's duplicated-half angle layout."""
    inverse = theta ** (-torch.arange(0, value.shape[-1], 2, dtype=torch.float32) / value.shape[-1])
    angles = positions.float().unsqueeze(-1) * inverse
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    return value * angles.cos() + _rotate_half(value) * angles.sin()


def _attend(query, keys, values, query_positions, key_positions, window):
    """Dense causal GQA with optional sliding-window masking."""
    groups = query.shape[1] // keys.shape[1]
    keys = keys.repeat_interleave(groups, dim=1).float()
    values = values.repeat_interleave(groups, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", query.float(), keys) / math.sqrt(query.shape[-1])
    mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    if window is not None:
        mask = mask | (query_positions.unsqueeze(1) - key_positions.unsqueeze(0) >= window)
    scores = scores.masked_fill(mask.unsqueeze(0), -torch.inf)
    probabilities = scores.softmax(-1)
    return torch.einsum("hqk,khd->qhd", probabilities, values).to(query.dtype)


class _History:
    """Dense per-(sublayer, plane) K/V history keyed by absolute position.

    Writing a position twice overwrites it, which is exactly what a single-plane
    ``last_loop`` history does; ``per_loop`` planes never repeat a position.
    """

    def __init__(self):
        self.entries = {}

    def append(self, layer, plane, keys, values, positions):
        store = self.entries.setdefault((layer, plane), {})
        for row, position in enumerate(positions.tolist()):
            store[int(position)] = (keys[row], values[row])
        ordered = sorted(store)
        return (
            torch.stack([store[position][0] for position in ordered]),
            torch.stack([store[position][1] for position in ordered]),
            torch.tensor(ordered),
        )


def _attention_sublayer(config, weights, index, hidden, positions, history, plane):
    prefix = f"backbone.layers.{index}.mixer"
    std = config.initializer_range
    heads, kv_heads, head_dim = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    count = hidden.shape[0]
    q = F.linear(hidden, weights[f"{prefix}.q_proj.weight"]).view(count, heads, head_dim)
    k = F.linear(hidden, weights[f"{prefix}.k_proj.weight"]).view(count, kv_heads, head_dim)
    v = F.linear(hidden, weights[f"{prefix}.v_proj.weight"]).view(count, kv_heads, head_dim)
    gate = F.linear(hidden, weights[f"{prefix}.g_proj.weight"])
    layer_type = config.layer_types[index]
    theta = config.rope_theta(layer_type)
    q = _rope(q, positions, theta)
    k = _rope(k, positions, theta)
    sqk = (weights[f"{prefix}.sqk"].float() * (head_dim**0.5 / std)).view(1, heads, head_dim)
    q = (sqk.square() * _justnorm(q).float()).to(q.dtype).reshape(count, heads * head_dim)
    k = _justnorm(k)
    keys, values, key_positions = history.append(index, plane, k, v, positions)
    window = config.sliding_window if layer_type == "sliding_attention" else None
    attended = _attend(
        q.view(count, heads, head_dim), keys, values, positions, key_positions, window
    ).reshape(count, heads * head_dim)
    gate_scale = config.attention_output_gate_logit_scale_init
    if gate_scale is None:
        gate_scale = config.d_model**0.5
    scale = (weights[f"{prefix}.s_gate"].float() * (gate_scale**0.5 / std)).square()
    attended = (attended.float() * torch.sigmoid(gate.float() * scale)).to(hidden.dtype)
    update = F.linear(_justnorm(attended), weights[f"{prefix}.o_proj.weight"])
    alpha = weights[f"backbone.layers.{index}.attn_alpha"]
    return _slerp(hidden, update, alpha, config.ngpt_alpha_init / std)


def _mlp_sublayer(config, weights, index, hidden):
    prefix = f"backbone.layers.{index}.mixer"
    std = config.initializer_range
    gate = F.linear(hidden, weights[f"{prefix}.gate_proj.weight"])
    up = F.linear(hidden, weights[f"{prefix}.up_proj.weight"])
    gate = (gate.float() * (weights[f"{prefix}.suv_gate"].float() * config.d_model**0.5)).to(
        gate.dtype
    )
    up = (up.float() * (weights[f"{prefix}.suv_up"].float() * config.d_model**0.5)).to(up.dtype)
    mixed = (F.silu(gate.float()) * up.float()).to(gate.dtype)
    update = F.linear(_justnorm(mixed), weights[f"{prefix}.down_proj.weight"])
    alpha = weights[f"backbone.layers.{index}.mlp_alpha"]
    return _slerp(hidden, update, alpha, config.ngpt_alpha_init / std)


def _sublayer(config, weights, index, hidden, positions, history, plane):
    if config.layer_types[index] == "mlp":
        return _mlp_sublayer(config, weights, index, hidden)
    return _attention_sublayer(config, weights, index, hidden, positions, history, plane)


@torch.inference_mode()
def dense_dory_reference(model, tokens):
    """One-token-per-step Dory forward; returns per-loop states and logits."""
    config = model.config
    weights = model.state_dict()
    history = _History()
    input_slice, recurrent_slice, output_slice = config.stage_slices
    input_layers = range(*input_slice)
    recurrent_layers = range(*recurrent_slice)
    output_layers = range(*output_slice)
    loops = [[] for _ in range(config.n_recurrent_loops)]
    logits = []
    for token in range(len(tokens)):
        position = torch.tensor([token], dtype=torch.long)
        hidden = weights["backbone.embeddings.weight"][tokens[token : token + 1]]
        for index in input_layers:
            hidden = _sublayer(config, weights, index, hidden, position, history, 0)
        injection, state = hidden, hidden
        # Loop order is observable in last_loop mode, where each loop overwrites
        # the single recurrent plane.
        for loop in range(config.n_recurrent_loops):
            plane = loop if config.recurrent_kv_cache_mode == "per_loop" else 0
            current = _justnorm(state + injection)
            for index in recurrent_layers:
                current = _sublayer(config, weights, index, current, position, history, plane)
            state = _justnorm(current)
            loops[loop].append(state[0])
        for index in output_layers:
            state = _sublayer(config, weights, index, state, position, history, 0)
        scaled = F.linear(state, weights["lm_head.weight"]).float()
        scaled = scaled * (weights["lm_head.logit_scale"].float() / config.initializer_range)
        logits.append(scaled[0])
    return {
        "logits": torch.stack(logits),
        "loops": [torch.stack(states) for states in loops],
    }
