"""Independent Ouro continuation with retained per-depth historical K/V."""

import torch
from torch.nn import functional as F


class OuroContinuation:
    """Functional oracle; no production layers, cache, runner or guidance helper."""

    def __init__(self, model):
        self.config = model.config
        self.weights = model.state_dict()
        self.kv = {}
        self.position = 0

    @torch.inference_mode()
    def forward(self, token_ids, depth):
        c, w = self.config, self.weights
        hidden = w["model.embed_tokens.weight"][token_ids]
        length = len(token_ids)
        positions = torch.arange(self.position, self.position + length, device=hidden.device)
        frequency = c.rope_theta ** (
            -torch.arange(0, c.head_dim, 2, device=hidden.device).float() / c.head_dim
        )
        angles = positions.float()[:, None] * frequency[None, :]
        cos = torch.cat([angles.cos(), angles.cos()], -1)[:, None].to(hidden.dtype)
        sin = torch.cat([angles.sin(), angles.sin()], -1)[:, None].to(hidden.dtype)
        future = (
            torch.arange(self.position + length, device=hidden.device)[None, :]
            > (positions[:, None])
        )

        def norm(x, name):
            value = x.float() / (x.float().square().mean(-1, keepdim=True) + c.rms_norm_eps).sqrt()
            return value.to(x.dtype) * w[name + ".weight"]

        def linear(x, name):
            return x @ w[name + ".weight"].T

        def rotary(x):
            split = c.head_dim // 2
            return x * cos + torch.cat([-x[..., split:], x[..., :split]], -1) * sin

        outputs = []
        for loop in range(depth):
            for layer in range(c.num_hidden_layers):
                prefix = f"model.layers.{layer}"
                x = norm(hidden, prefix + ".input_layernorm")
                q = rotary(
                    linear(x, prefix + ".self_attn.q_proj").reshape(
                        length, c.num_attention_heads, c.head_dim
                    )
                )
                k = rotary(
                    linear(x, prefix + ".self_attn.k_proj").reshape(
                        length, c.num_key_value_heads, c.head_dim
                    )
                )
                v = linear(x, prefix + ".self_attn.v_proj").reshape(
                    length, c.num_key_value_heads, c.head_dim
                )
                previous = self.kv.get((loop, layer))
                keys, values = (
                    (k, v)
                    if previous is None
                    else (torch.cat([previous[0], k]), torch.cat([previous[1], v]))
                )
                self.kv[loop, layer] = (keys, values)
                groups = c.num_attention_heads // c.num_key_value_heads
                scores = torch.einsum("thd,shd->hts", q, keys.repeat_interleave(groups, 1))
                scores = scores / c.head_dim**0.5
                probabilities = scores.masked_fill(future, -torch.inf).float().softmax(-1)
                attended = torch.einsum(
                    "hts,shd->thd",
                    probabilities.to(hidden.dtype),
                    values.repeat_interleave(groups, 1),
                ).reshape(length, -1)
                hidden = hidden + norm(
                    linear(attended, prefix + ".self_attn.o_proj"), prefix + ".input_layernorm_2"
                )
                x = norm(hidden, prefix + ".post_attention_layernorm")
                mlp = F.silu(linear(x, prefix + ".mlp.gate_proj")) * linear(
                    x, prefix + ".mlp.up_proj"
                )
                hidden = hidden + norm(
                    linear(mlp, prefix + ".mlp.down_proj"), prefix + ".post_attention_layernorm_2"
                )
            hidden = norm(hidden, "model.norm")
            outputs.append((hidden.clone(), linear(hidden, "lm_head")))
        # Only newly computed positions are filled. Prompt P4 history survives D2/D3.
        for loop in range(depth, c.total_ut_steps):
            for layer in range(c.num_hidden_layers):
                k, v = self.kv[depth - 1, layer]
                previous = self.kv.get((loop, layer))
                self.kv[loop, layer] = (
                    (k[-length:].clone(), v[-length:].clone())
                    if previous is None
                    else (
                        torch.cat([previous[0], k[-length:]]),
                        torch.cat([previous[1], v[-length:]]),
                    )
                )
        self.position += length
        return outputs
