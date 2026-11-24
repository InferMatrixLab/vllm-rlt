# SPDX-License-Identifier: Apache-2.0
"""Functional Inkling kernels from vLLM's SM10x path, called by the RLT model."""

from functools import cache
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch
from torch import nn


@cache
def load_ops():
    try:
        from vllm.models.inkling.nvidia.ops.sconv import sconv_seq_metadata
        from vllm.models.inkling2.nvidia.ops import (
            activation,
            attention,
            attn_res,
            block_scales,
            conv_mix,
            linear,
            loop,
            moe,
            norm,
            sconv,
            trtllm_bmm,
        )
    except ImportError as error:
        raise RuntimeError(
            "Inkling's vllm kernels require a CUDA build of vLLM with Inkling 2 support "
            "(tested source revision 3146a9df4140)"
        ) from error
    return SimpleNamespace(
        activation=activation,
        attention=attention,
        attn_res=attn_res,
        block_scales=block_scales,
        conv_mix=conv_mix,
        linear=linear,
        loop=loop,
        moe=moe,
        norm=norm,
        sconv=sconv,
        sconv_seq_metadata=sconv_seq_metadata,
        bmm=trtllm_bmm,
        prefill=attention.Inkling2FA4RelAttentionKernel(),
    )


def warmup_fa4(config, cache, max_tokens, max_sequences):
    """Compile reachable local and full relative-attention kernel variants."""
    from vllm.models.inkling.nvidia.ops.fa4_rel_attention import _num_warps_bucket

    ops = load_ops()
    kernel = ops.prefill
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    warp_buckets = sorted({_num_warps_bucket(n) for n in range(1, max_sequences + 1)})
    query = 1
    while query <= ops.attention.bucket_max_seqlen_q(max_tokens):
        if not ops.attention.plan_decode_attention(
            False, 1, query, heads, kv_heads, config.rel_extent
        ):
            for local in (False, True):
                variants = [(False, None)]
                if not local:
                    variants.extend((True, warps) for warps in (None, *warp_buckets))
                for split, warps in variants:
                    kernel.compile(
                        kernel.CompileKey(
                            is_local=local,
                            num_heads=heads,
                            num_kv_heads=kv_heads,
                            head_dim=config.head_dim,
                            rel_extent=config.window if local else config.rel_extent,
                            dtype=cache.key_cache.dtype,
                            kv_dtype=cache.key_cache.dtype,
                            block_size=cache.block_size,
                            max_seqlen_q=query,
                            split_kv=split,
                            num_warps_bucket=warps,
                            large_num_reqs=max_sequences > 1024,
                        )
                    )
        query *= 2


class QuantizedExperts(nn.Module):
    """RLT-owned weights and buffers for the upstream routing/quantized GEMM kernels."""

    def __init__(self, config, read):
        super().__init__()
        self.ops = load_ops()
        block_scales, moe, bmm = self.ops.block_scales, self.ops.moe, self.ops.bmm
        self.router = nn.Parameter(read("gate.weight"), requires_grad=False)
        self.bias = nn.Parameter(read("gate.bias"), requires_grad=False)
        self.topk = config.experts_per_token
        self.route_scale = config.route_scale
        self.gate_cap = config.gate_softcap
        self.up_cap = config.up_softcap
        self.num_experts = len(self.router)
        shapes = []
        for projection in ("w13", "w2"):
            name = f"experts.{projection}_weight"
            weight = read(name)
            scale = read(f"{name}_scale")
            nvfp4 = weight.dtype == torch.uint8
            shapes.append((weight.shape[1], weight.shape[2] * (2 if nvfp4 else 1)))
            setattr(self, projection, nn.Parameter(block_scales.shuffle_rows(weight), False))
            scale = block_scales.swizzle_block_scales(block_scales.shuffle_rows(scale))
            setattr(self, f"{projection}_scale", nn.Parameter(scale, False))
            if nvfp4:
                self.register_buffer(f"{projection}_amax", read(f"{name}_amax"))
                self.register_buffer(
                    f"{projection}_denominator", read(f"{name}_outer_scale_denominator")
                )
        self.nvfp4 = self.w13.dtype == torch.uint8
        self.shapes = tuple(shapes)
        device = self.router.device
        self.register_buffer(
            "bounds",
            torch.tensor([0, self.num_experts], device=device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "counters",
            torch.zeros(moe.NUM_COUNTERS, device=device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "counts",
            torch.zeros(self.num_experts, device=device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "ones",
            torch.ones(self.num_experts, device=device, dtype=torch.float32),
            persistent=False,
        )
        bmm.load_kernels((self.w13, self.w2), (self.w13_scale, self.w2_scale), self.topk)

    def _scales(self, projection):
        if not self.nvfp4:
            return None, None
        return getattr(self, f"{projection}_amax"), getattr(self, f"{projection}_denominator")

    def forward(self, hidden):
        activation, moe, bmm = self.ops.activation, self.ops.moe, self.ops.bmm
        hidden = hidden.contiguous()
        plan = bmm.expert_plan(len(hidden), self.topk, self.num_experts, self.nvfp4, self.shapes)
        routing, quantized = moe.route(
            hidden,
            self.router,
            self.bias,
            self.counters,
            self.counts,
            self.topk,
            self.route_scale,
            self.bounds,
            plan,
            *self._scales("w13"),
        )
        inverse, gammas, rows, tiles = routing
        gate_up = bmm.bmm(*quantized, self.w13, self.w13_scale, self.ones, tiles, plan, 0)
        count = plan.max_ctas
        quantized = activation.swiglu_quantize(
            gate_up,
            gammas,
            self.gate_cap,
            self.up_cap,
            (rows, tiles[2 * count + 2 :], tiles[:count]),
            plan.tile,
            *self._scales("w2"),
        )
        down = bmm.bmm(*quantized, self.w2, self.w2_scale, self.ones, tiles, plan, 1)
        return moe.combine(down, inverse, self.topk, hidden.dtype)


def residual_stack(blocks):
    stack = blocks.transpose(0, 1).contiguous()
    _, tokens, width = stack.shape
    return stack.as_strided(stack.shape, (tokens * width, width, 1))


def attention_residual(blocks, queries):
    stack = residual_stack(blocks)
    queries = (queries.float() * stack.shape[-1] ** -0.5).contiguous()
    mixed = load_ops().attn_res.attn_res(stack[:-1], stack[-1], None, queries, None, 0.0, True)
    return mixed.transpose(0, 1)


def message_controller(dest, message, down, gain_forget, weight, eps, active=None):
    ops = load_ops()
    message = message.contiguous()
    pair = ops.loop.pair_rms_norm(dest, message)
    rank = torch.bmm(pair, down.transpose(1, 2))
    gains = torch.bmm(rank, gain_forget.transpose(1, 2), out_dtype=torch.float32)
    mixed = ops.loop.blend_rms_norm(dest, message, gains, weight, eps)
    if active is not None:
        plain = torch.stack([rms_norm(x, w, eps) for x, w in zip(dest, weight, strict=True)])
        mixed = torch.where(active, mixed, plain)
    return mixed


def rms_norm(hidden, weight, epsilon):
    return load_ops().norm.rms_norm(hidden, weight, epsilon, out_dtype=weight.dtype)


class AttentionPlan(NamedTuple):
    decode_reqs: int
    decode_rows: int
    max_decode_query: int
    decode_splits: int
    prefill: tuple[int, int, bool] | None


def packed_attention_plans(metadata, config):
    """Return the launch parameters shared by all full or local layers."""
    from vllm_rlt.worker.metadata import MixedRecurrentKVMetadata

    mixed = isinstance(metadata, MixedRecurrentKVMetadata)
    ranges = metadata.prompt.packed_ranges if mixed else getattr(metadata, "packed_ranges", None)
    if ranges is not None:
        lengths = [seq.end for seq in ranges]
        query = [seq.end - seq.start for seq in ranges]
        if mixed:
            # Decode launch planning uses only the row count. A capacity bound
            # is available if a shape instead requires all rows to use FA4.
            capacity = metadata.block_tables.shape[1] * metadata.prompt.owner.block_size
            lengths = [capacity] * metadata.decode_rows + lengths
            query = [1] * metadata.decode_rows + query
    else:
        lengths, query = [], []
        previous = None
        for allocation, depth, pos in metadata.rows:
            key = id(allocation), depth
            if key != previous:
                lengths.append(pos + 1)
                query.append(1)
                previous = key
            else:
                if pos != lengths[-1]:
                    raise ValueError("packed Inkling rows must be contiguous within a sequence")
                lengths[-1] = pos + 1
                query[-1] += 1
    leading_decode = next((i for i, count in enumerate(query) if count > 1), len(query))
    query = np.asarray(query, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    return tuple(
        _attention_plan(config, local, metadata, packed=(leading_decode, query, lengths))
        for local in (False, True)
    )


def _attention_plan(config, local, metadata, *, packed=None):
    ops = load_ops().attention
    extent = config.window if local else config.rel_extent
    tokens = len(metadata.write_blocks)
    requests = len(metadata.context_lengths)
    max_query = metadata.max_seqlen_q

    def decode_splits(reqs, query_rows):
        return ops.plan_decode_attention(
            local, reqs, query_rows, config.num_attention_heads, config.num_key_value_heads, extent
        )

    splits = decode_splits(requests, max_query)
    if splits:
        return AttentionPlan(requests, tokens, max_query, splits, None)
    decode_reqs = 0
    if packed is None:
        query = np.ones(tokens, dtype=np.int64)
        lengths = np.asarray([pos + 1 for _, _, pos in metadata.rows], dtype=np.int64)
    else:
        leading_decode, query, lengths = packed
        if 0 < leading_decode < requests:
            splits = decode_splits(leading_decode, 1)
            if splits:
                decode_reqs = leading_decode
    query = query[decode_reqs:]
    prefill_splits, use_scheduler = ops.plan_attention(
        local, query, lengths[decode_reqs:], config.num_attention_heads, config.num_key_value_heads
    )
    prefill = ops.bucket_max_seqlen_q(int(query.max())), prefill_splits, use_scheduler
    return AttentionPlan(decode_reqs, decode_reqs, 1, splits, prefill)


def _prepare_sconv_metadata(cu_seqlens, tokens):
    """Build the per-token request index and query start for native kernels."""
    seq_idx = torch.empty(tokens, dtype=torch.int32, device=cu_seqlens.device)
    query_start = torch.empty_like(seq_idx)
    if tokens:
        load_ops().sconv_seq_metadata(cu_seqlens, len(cu_seqlens) - 1, tokens, seq_idx, query_start)
    return seq_idx, query_start


class KernelBatch:
    """Translate RLT's prepared addresses into the functional kernel inputs."""

    def __init__(self, cache, metadata, config, *, fused_sconv_metadata=False):
        self.cache, self.metadata, self.config = cache, metadata, config
        self.positions = metadata.position_ids[: len(metadata.write_blocks)]
        self.slot_mapping = metadata.write_blocks * cache.block_size + metadata.write_offsets
        mask = getattr(metadata, "write_mask", None)
        if mask is not None:
            self.slot_mapping = self.slot_mapping.masked_fill(~mask, -1)
        self.seq_lens = metadata.context_lengths
        self.block_table = metadata.block_tables
        self.cu_seqlens = metadata.cu_seqlens_q
        self.attention_segments = getattr(metadata, "attention_segments", None)
        if self.cu_seqlens is None:
            self.cu_seqlens = torch.arange(
                len(self.positions) + 1, device=self.positions.device, dtype=torch.int32
            )
            self.seq_idx = self.query_start = self.cu_seqlens[:-1]
        elif fused_sconv_metadata:
            self.seq_idx, self.query_start = _prepare_sconv_metadata(
                self.cu_seqlens, len(self.positions)
            )
        else:
            rows = torch.arange(
                len(self.positions), device=self.positions.device, dtype=torch.int32
            )
            self.seq_idx = torch.searchsorted(self.cu_seqlens[1:], rows, right=True, out_int32=True)
            if (
                self.attention_segments is not None
                and getattr(metadata, "active_rows", None) is not None
            ):
                # The staged table ends in repeated offsets for inactive rows.
                self.seq_idx.clamp_max_(len(self.cu_seqlens) - 2)
            self.query_start = self.cu_seqlens[self.seq_idx]
        self.attention_plans = getattr(metadata, "packed_attention_plan", None)
        if self.attention_segments is not None:
            return
        if self.attention_plans is None and metadata.cu_seqlens_q is not None:
            self.attention_plans = packed_attention_plans(metadata, config)
        self.tau = load_ops().attention.compute_log_scaling_tau(
            self.positions, config.log_scale_floor, config.log_scale_alpha
        )

    def conv_state(self, layer):
        return (
            self.cache.auxiliary_cache[:, layer]
            .unflatten(-1, (self.config.num_key_value_heads, -1))
            .transpose(1, 2)
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
        cfg = self.config
        return load_ops().conv_mix.conv_mix(
            hidden,
            weight,
            self.conv_state(layer),
            self.positions,
            self.block_table,
            self.seq_idx,
            self.slot_mapping,
            self.query_start,
            offset,
            cfg.width // cfg.num_key_value_heads,
            out,
            residual,
            addend,
            stack,
            last,
            queries,
            norm,
            cfg.rms_norm_eps,
        )

    def attend(self, layer, q, r, gate, sinks, k, v, k_dynamic, v_dynamic):
        output = torch.empty_like(gate, memory_format=torch.contiguous_format)
        if self.attention_segments is not None:

            def attend_live(batch, *projected):
                count = len(batch.positions)
                if count == len(projected[-1]):
                    batch._attend(layer, *projected)
                else:
                    batch._attend(
                        layer,
                        *(tensor[:count] if tensor is not None else None for tensor in projected),
                    )
                    projected[-1][count:].zero_()

            self.attention_segments.attention(
                attend_live, q, r, gate, sinks, k, v, k_dynamic, v_dynamic, output
            )
        else:
            self._attend(layer, q, r, gate, sinks, k, v, k_dynamic, v_dynamic, output)
        return output

    def _attend(self, layer, q, r, gate, sinks, k, v, k_dynamic, v_dynamic, output):
        ops, cfg, md = load_ops(), self.config, self.metadata
        tokens = len(q)
        q = q.view(tokens, cfg.num_attention_heads, cfg.head_dim)
        r = r.view(tokens, cfg.num_attention_heads, -1)
        clipped_q = torch.empty_like(q, memory_format=torch.contiguous_format)
        sheared_bias = q.new_empty(
            (tokens + 128, cfg.num_attention_heads, layer.rel_basis.shape[-1] + 256)
        )
        keys, values = self.cache.key_cache[:, layer.index], self.cache.value_cache[:, layer.index]
        ops.sconv.attention_prologue(
            q,
            r,
            k,
            v,
            layer.query_scale,
            layer.rel_scale,
            None if layer.local else self.tau,
            layer.shifted_rel_proj,
            layer.conv_k,
            layer.conv_v,
            k_dynamic,
            v_dynamic,
            self.conv_state(layer.index),
            self.positions,
            self.block_table,
            self.seq_idx,
            self.slot_mapping,
            self.query_start,
            keys,
            values,
            self.slot_mapping,
            clipped_q,
            sheared_bias,
            self.cu_seqlens,
            self.seq_lens,
            clip_threshold=cfg.qk_clip,
            rel_threshold=cfg.rel_clip,
            norm_eps=cfg.rms_norm_eps,
        )
        self.cache.record_external_kv(layer.index, md)
        if self.attention_plans is None:
            plan = _attention_plan(cfg, layer.local, md)
        else:
            plan = self.attention_plans[layer.local]
        decode_reqs, decode_rows = plan.decode_reqs, plan.decode_rows
        if decode_rows:
            ops.attention.rel_decode_attention(
                clipped_q[:decode_rows],
                keys,
                values,
                sheared_bias,
                gate[:decode_rows],
                sinks[:decode_rows],
                output[:decode_rows],
                block_table=self.block_table[:decode_reqs],
                cache_seqlens=self.seq_lens[:decode_reqs],
                cu_seqlens_q=self.cu_seqlens[: decode_reqs + 1],
                max_query_len=plan.max_decode_query,
                num_actual_tokens=decode_rows,
                softmax_scale=cfg.head_dim**-0.5,
                is_local=layer.local,
                num_splits=plan.decode_splits,
            )
        if plan.prefill is not None:
            max_query, splits, use_scheduler = plan.prefill
            result = torch.empty_like(clipped_q[decode_rows:], dtype=torch.float32)
            lse = q.new_empty((cfg.num_attention_heads, tokens - decode_rows), dtype=torch.float32)
            cu = self.cu_seqlens[decode_reqs:]
            if decode_reqs:
                cu = cu - decode_rows
            ops.prefill(
                clipped_q[decode_rows:],
                keys,
                values,
                block_table=self.block_table[decode_reqs:],
                cache_seqlens=self.seq_lens[decode_reqs:],
                cu_seqlens_q=cu,
                max_seqlen_q=max_query,
                softmax_scale=cfg.head_dim**-0.5,
                window_size=(cfg.window - 1, 0) if layer.local else (-1, -1),
                sheared_bias=sheared_bias[decode_rows:],
                num_splits=splits,
                use_scheduler=use_scheduler,
                out=result,
                lse=lse,
            )
            ops.attention.gate_sink(
                result, gate[decode_rows:], lse.t(), sinks[decode_rows:], output[decode_rows:]
            )
