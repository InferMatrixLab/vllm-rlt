"""LAST_EXITED KV semantics for early exit.

A token that leaves the loop after pass ``e`` has no KV in passes ``e+1..N-1``.
Later tokens that run those passes would attend over a hole at its position, so
before the next token runs, pass ``e``'s K/V for the token's slots is copied
into every skipped pass (Ouro's reference does the same: the last executed
pass stands in for the missing ones).

Each pass is its own ``Attention`` layer with its own cache tensor but a shared
block table, so the copy is per slot, not per block. vLLM's
``kv_cache_block_copies`` copies whole blocks and cannot express this.
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
    group. Returns the number of passes filled. A layout that is not
    ``(2, blocks, block_size, ...)`` raises instead of corrupting the cache.
    """
    num_passes = len(attention)
    if exit_depth >= num_passes - 1:
        return 0
    for layer in range(len(attention[0])):
        src = attention[exit_depth][layer]
        src_cache = _flat(src.kv_cache)
        src_slots = slots_for(src.layer_name)
        for p in range(exit_depth + 1, num_passes):
            dst = attention[p][layer]
            _flat(dst.kv_cache)[:, slots_for(dst.layer_name)] = src_cache[:, src_slots]
    return num_passes - 1 - exit_depth


def _flat(kv_cache: torch.Tensor) -> torch.Tensor:
    """View a (2, blocks, block_size, ...) cache as (2, blocks*block_size, ...)."""
    if kv_cache.dim() < 4 or kv_cache.shape[0] != 2:
        raise NotImplementedError(
            f"LAST_EXITED fill needs a (2, blocks, block_size, ...) KV layout, got "
            f"{tuple(kv_cache.shape)}"
        )
    return kv_cache.view(2, kv_cache.shape[1] * kv_cache.shape[2], *kv_cache.shape[3:])
