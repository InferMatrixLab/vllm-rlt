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
            "flash-attn-4==4.0.0b30; use --attention-backend triton until a "
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
        self.generation = select_version(torch.cuda.get_device_capability(device), backend)
        if self.generation == 2 and block_size % 256:
            raise ValueError("FlashAttention-2 paged KV requires --block-size a multiple of 256")
        module, package, function = {
            2: ("flash_attn", "flash-attn", "flash_attn_with_kvcache"),
            3: ("flash_attn_3.flash_attn_interface", "flash-attn-3", "flash_attn_with_kvcache"),
            4: ("flash_attn.cute", "flash-attn-4", "flash_attn_varlen_func"),
        }[self.generation]
        try:
            self.kernel = getattr(importlib.import_module(module), function)
            package_version = version(package)
        except (ImportError, AttributeError) as error:
            raise ImportError(
                f"{backend} selected FA{self.generation}; install the official {package} package. "
                "No backend fallback was performed."
            ) from error
        self.info = dict(
            backend="flash_attn",
            generation=self.generation,
            package=package,
            version=package_version,
        )
        logging.getLogger(__name__).warning("Attention implementation: %s", self.info)

    def prefill(
        self,
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        cu_seqlens_q,
        max_seqlen_q,
    ):
        """Attend packed chunks to their prefixes with bottom-right causal alignment.

        Each sequence's keys end at its final query position. Thus the causal
        offset is prefix_length = seqlen_k - seqlen_q, including nonzero prefixes.
        K/V have already been written by the cache manager; no gathering is needed.
        """
        if self.generation != 4:
            raise ValueError("packed paged prefill requires FlashAttention-4")
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
        )
        return out[0] if isinstance(out, tuple) else out

    def __call__(self, q, key_cache, value_cache, block_tables, context_lengths, return_lse=False):
        """Decode attention; with ``return_lse`` also the [rows, heads] natural-log normalizers."""
        if q.shape[0] == 0:
            empty = torch.empty_like(q)
            return (empty, q.new_empty(q.shape[:2], dtype=torch.float32)) if return_lse else empty
        # The visible prefix already includes exactly the current causal position.
        # Do not append K/V here: the cache manager has written it at this depth.
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
                **({"return_lse": True} if return_lse else {}),
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
                **({"return_softmax_lse": True} if return_lse else {}),
            )
        lse = None
        if isinstance(out, tuple):
            out, lse = out[0], out[1]  # FA4 beta returns (output, optional LSE).
        if return_lse:
            # [rows, heads, 1] or [rows, 1, heads]; one query position per row.
            rows, heads = q.shape[:2]
            lse = (
                lse.reshape(rows, heads)
                if lse.shape[1] == heads
                else lse.transpose(1, 2).reshape(rows, heads)
            )
            return out.squeeze(1), lse.float()
        return out.squeeze(1)
