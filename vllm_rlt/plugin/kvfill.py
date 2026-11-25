"""LAST_EXITED KV semantics for early exit.

A token that leaves the loop after pass ``e`` has no KV in passes ``e+1..N-1``.
Later tokens that run those passes would attend over a hole at its position, so
before the next token runs, pass ``e``'s K/V for the token's slots is copied
into every skipped pass (Ouro's reference does the same: the last executed
pass stands in for the missing ones).

Each pass is its own ``Attention`` layer with its own cache tensor but a shared
block table, so the copy is per slot, not per block. vLLM's
``kv_cache_block_copies`` copies whole blocks and cannot express this.

Layout: vLLM hands a layer its cache as the logical view ``(B, H, N, C)``:
``B`` kernel blocks, ``H`` KV heads, ``N`` tokens per kernel block and ``C`` the
K and V of one head packed together (FlashInfer on this vLLM: (2188, 16, 64,
256) for Ouro-1.4B). The physical stride order is the backend's business and does
not matter here because a slot is addressed through the logical indices:
``block = slot // N``, ``offset = slot % N``. A manager block that is split into
several kernel blocks is a pure view of consecutive kernel blocks, so the same
division holds.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def fill_skipped_passes(
    attention: list[list], slots_for: Callable[[str], torch.Tensor], exit_depth: int
) -> int:
    """Copy pass ``exit_depth`` KV into passes ``exit_depth+1..`` for the token.

    ``attention`` is indexed [pass][layer]; ``slots_for(layer_name)`` returns the
    flat slot ids (block * block_size + offset) of the token in that layer's KV
    group. Returns the number of passes filled. A cache that is not 4-D raises
    instead of corrupting memory.
    """
    num_passes = len(attention)
    if exit_depth >= num_passes - 1:
        return 0
    for layer in range(len(attention[0])):
        src = attention[exit_depth][layer]
        src_slots = slots_for(src.layer_name)
        for p in range(exit_depth + 1, num_passes):
            dst = attention[p][layer]
            copy_slots(src.kv_cache, src_slots, dst.kv_cache, slots_for(dst.layer_name))
    return num_passes - 1 - exit_depth


def copy_slots(src: torch.Tensor, src_slots, dst: torch.Tensor, dst_slots) -> None:
    """``dst[slot] = src[slot]`` for per-token slots, over every head."""
    if src.dim() != 4 or dst.dim() != 4:
        raise NotImplementedError(
            f"LAST_EXITED fill needs a (blocks, heads, tokens, content) KV view, got "
            f"{tuple(src.shape)} and {tuple(dst.shape)}"
        )
    keep = src_slots >= 0  # padded rows carry slot -1
    src_slots, dst_slots = src_slots[keep], dst_slots[keep]
    n_src, n_dst = src.shape[2], dst.shape[2]
    dst[dst_slots // n_dst, :, dst_slots % n_dst, :] = src[
        src_slots // n_src, :, src_slots % n_src, :
    ]
