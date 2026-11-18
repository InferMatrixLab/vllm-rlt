"""Official FlashAttention paged kernels; no FlashInfer or KV gathering.

Decode uses independent single-query rows. FA4 prefill uses packed query
sequences with one depth-specific page table per contiguous request chunk.
"""

import importlib
import logging
from importlib.metadata import version

import torch

FLASH_BACKENDS = ("flash_attn", "flash_attn_2", "flash_attn_3", "flash_attn_4")


def select_version(capability, backend):
    major, minor = capability
    # The pinned FA4 release accepts SM12 only without paged KV. Both adapter
    # paths require page tables, so reject before import or KV capacity probing.
    if major == 12 and backend in ("flash_attn", "flash_attn_4"):
        raise ValueError(
            f"Paged FlashAttention-4 is not supported on SM{major}{minor} by "
            "the pinned flash-attn-4 release; use --attention-backend triton until a "
            "compatible paged implementation is available."
        )
    automatic = 4 if major == 10 else 3 if major == 9 else 2
    selected = automatic if backend == "flash_attn" else int(backend.rsplit("_", 1)[1])
    supported = {2: major in (8, 9), 3: major == 9, 4: major in (9, 10)}
    if not supported[selected]:
        raise ValueError(f"FlashAttention-{selected} is not supported on SM{major}{minor}")
    return selected


class FlashPagedAttention:
    def __init__(self, device, dtype, head_dim, block_size, backend="flash_attn"):
        if device.type != "cuda" or torch.version.hip:
            raise ValueError("flash_attn currently requires an NVIDIA CUDA device")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("flash_attn requires float16 or bfloat16")
        if head_dim > 256 or head_dim % 8:
            raise ValueError("flash_attn requires head_dim divisible by 8 and <= 256")
        capability = torch.cuda.get_device_capability(device)
        self.generation = select_version(capability, backend)
        if self.generation == 2 and block_size % 256:
            raise ValueError("FlashAttention-2 paged KV requires --block-size a multiple of 256")
        module, package, function = {
            2: ("flash_attn", "flash-attn", "flash_attn_with_kvcache"),
            3: ("flash_attn_3.flash_attn_interface", "flash-attn-3", "flash_attn_with_kvcache"),
            4: ("flash_attn.cute", "flash-attn-4", "flash_attn_varlen_func"),
        }[self.generation]
        try:
            implementation = importlib.import_module(module)
            self.kernel = getattr(implementation, function)
            package_version = version(package)
        except (ImportError, AttributeError) as error:
            raise ImportError(
                f"{backend} selected FA{self.generation}; install the official {package} package. "
                "No backend fallback was performed."
            ) from error
        # FA4's public scale API also identifies the release family with generic
        # hd256 paged kernels. Older releases used a restricted hd256 kernel.
        self.scale_type = getattr(implementation, "TensorwiseScales", None)
        self.supports_fp8_static = (
            self.generation == 4
            and self.scale_type is not None
            and capability[0] == 10
            and dtype == torch.bfloat16
            and head_dim in (64, 128, 256)
        )
        self.supports_sliding_window = self.generation == 4 and (
            head_dim != 256 or self.scale_type is not None
        )
        self.info = dict(
            backend="flash_attn",
            generation=self.generation,
            package=package,
            version=package_version,
        )
        logging.getLogger(__name__).warning("Attention implementation: %s", self.info)

    def _fp8_inputs(self, q, key_cache, value_cache, key_scale, value_scale, batch_size):
        """FA4 consumes one descale per sequence/KV head, never per token.

        This prototype accepts immutable scalar scales only. Q uses the K scale,
        independent of batch composition and prefill chunk boundaries.
        """
        if key_cache.dtype != torch.float8_e4m3fn:
            if key_scale is not None or value_scale is not None:
                raise ValueError("FA descales require an FP8 KV cache")
            return q, {}
        if not self.supports_fp8_static:
            raise ValueError(
                "static FP8 requires FA4 with TensorwiseScales on SM10x, BF16, "
                "head_dim 64, 128 or 256; install the flash-attn4 extra"
            )
        if value_cache.dtype != key_cache.dtype:
            raise ValueError("FA FP8 K/V must have the same dtype")
        for scale in (key_scale, value_scale):
            if (
                scale is None
                or scale.ndim != 0
                or scale.dtype != torch.float32
                or scale.device != q.device
            ):
                raise ValueError("FA4 FP8 requires fixed scalar float32 scales on the query device")
        shape = (batch_size, key_cache.shape[-2])
        q_codes = (q.float() / key_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        k_descale = key_scale.expand(shape).contiguous()
        v_descale = value_scale.expand(shape).contiguous()
        return q_codes, dict(scales=self.scale_type(q=k_descale, k=k_descale, v=v_descale))

    def prefill(
        self,
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        cu_seqlens_q,
        max_seqlen_q,
        *,
        key_scale=None,
        value_scale=None,
        window=None,
    ):
        """Attend packed chunks to their prefixes with bottom-right causal alignment.

        Each sequence's keys end at its final query position. Thus the causal
        offset is prefix_length = seqlen_k - seqlen_q, including nonzero prefixes.
        K/V have already been written by the cache manager; no gathering is needed.
        """
        if self.generation != 4:
            raise ValueError("packed paged prefill requires FlashAttention-4")
        if window is not None and not self.supports_sliding_window:
            raise NotImplementedError(
                "this FlashAttention implementation lacks sliding-window support"
            )
        output_dtype = q.dtype
        q, descales = self._fp8_inputs(
            q, key_cache, value_cache, key_scale, value_scale, block_tables.shape[0]
        )
        out = self.kernel(
            q,
            key_cache,
            value_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=context_lengths,
            page_table=block_tables,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=block_tables.shape[1] * key_cache.shape[1],
            causal=True,
            num_splits=1,
            **({"window_size": (window - 1, 0)} if window is not None else {}),
            **descales,
        )
        out = out[0] if isinstance(out, tuple) else out
        return out.to(output_dtype)

    def __call__(
        self,
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        *,
        key_scale=None,
        value_scale=None,
        window=None,
    ):
        if window is not None and not self.supports_sliding_window:
            raise NotImplementedError(
                "this FlashAttention implementation lacks sliding-window support"
            )
        if q.shape[0] == 0:
            return torch.empty_like(q)
        # The visible prefix already includes exactly the current causal position.
        # Do not append K/V here: the cache manager has written it at this depth.
        output_dtype = q.dtype
        q, descales = self._fp8_inputs(
            q, key_cache, value_cache, key_scale, value_scale, block_tables.shape[0]
        )
        query = q.unsqueeze(1)
        if self.generation == 4:
            out = self.kernel(
                query,
                key_cache,
                value_cache,
                seqused_k=context_lengths,
                page_table=block_tables,
                max_seqlen_q=1,
                max_seqlen_k=block_tables.shape[1] * key_cache.shape[1],
                causal=True,
                # Auto splitting depends on batch size and changes BF16 reductions.
                # Keep the partition fixed across scheduler batch compositions.
                num_splits=1,
                **({"window_size": (window - 1, 0)} if window is not None else {}),
                **descales,
            )
        else:
            page_argument = "block_table" if self.generation == 2 else "page_table"
            out = self.kernel(
                query,
                key_cache,
                value_cache,
                cache_seqlens=context_lengths,
                causal=True,
                **{page_argument: block_tables},
            )
        if isinstance(out, tuple):
            out = out[0]  # FA4 beta returns (output, optional LSE).
        return out.squeeze(1).to(output_dtype)

    def windowed(
        self,
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        window,
        *,
        key_scale=None,
        value_scale=None,
    ):
        # The runtime window includes the current token; FA counts preceding keys.
        return self(
            q,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale=key_scale,
            value_scale=value_scale,
            window=window,
        )
