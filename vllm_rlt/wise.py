"""WISE working-set decode for recurrent-depth models (Wan and Chen, arXiv 2609.27373).

Variant ``wise-cached-decode``: during decode, the query row of each token
discovers block support with unrestricted attention at the discovery depths,
the union of those supports is reused at the later depths, and Q/K/V plus the
within-support weights stay dynamic. Differences from the paper's no-cache
reference: the cached K/V of earlier tokens were computed under their own
supports, and the query block is the single decode row. Prefill stays dense.

Support is request-local: it is tagged with (request, position), rebuilt for
every token, and a reuse depth never uses support from another position or an
incomplete discovery window. When a token's discovery is not complete, its
remaining depths attend densely instead: discovery depths that ran in a packed
batch (prefill attention records no support), or a KV offload between discovery
and reuse (``free`` releases the support). ``fallbacks`` counts those rows.
"""

import math
import weakref
from dataclasses import dataclass

import torch

from vllm_rlt.kernels.wise_attention import (
    MODE_DENSE,
    MODE_DISCOVER,
    MODE_DISCOVER_UNION,
    MODE_SPARSE,
    WISE_BLOCK,
    wise_paged_attention,
    wise_scratch,
)
from vllm_rlt.kernels.wise_flash import compact_page_tables, discover_support


@dataclass(frozen=True)
class WiseConfig:
    profile: str = "ouro4"
    discovery_depths: tuple[int, ...] = (0, 1)
    reuse_depths: tuple[int, ...] = (2, 3)
    mass_threshold: float = 0.95
    max_model_len: int = 8192
    max_slots: int = 256
    # Benchmark controls only: "all" forces every block on at reuse depths while
    # discovery still runs; "dense" runs every row densely through the same split-KV
    # kernel; a float in (0, 1] skips discovery and keeps a fixed fraction of blocks
    # (always including the newest one).
    support_override: str | float | None = None
    # "auto": WISE decode on the cache's attention backend. "triton": the Triton
    # split-KV WISE kernel even when FlashAttention serves prefill and dense decode.
    kernel: str = "auto"

    def __post_init__(self):
        if set(self.discovery_depths) & set(self.reuse_depths) or not self.discovery_depths:
            raise ValueError("discovery and reuse depths must be disjoint and nonempty")
        if max(self.discovery_depths) >= min(self.reuse_depths, default=math.inf):
            raise ValueError("every reuse depth must follow the discovery window")
        if not 0 < self.mass_threshold <= 1:
            raise ValueError("mass_threshold must be in (0, 1]")
        if self.kernel not in ("auto", "triton"):
            raise ValueError("kernel must be 'auto' or 'triton'")
        if self.support_override not in (None, "all", "dense") and not (
            isinstance(self.support_override, float) and 0 < self.support_override <= 1
        ):
            raise ValueError(
                "support_override must be None, 'all', 'dense', or a fraction in (0, 1]"
            )


class WiseState:
    """Per-request support bitmasks [slot, layer, query head, block] on the device."""

    def __init__(
        self, config: WiseConfig, num_layers: int, num_heads: int, device, head_dim: int = 128
    ):
        self.config = config
        self.head_dim = head_dim
        # Power of two: the fused selection sorts one row of blocks in registers.
        self.blocks = 1 << max(
            0, math.ceil(math.log2(math.ceil(config.max_model_len / WISE_BLOCK)))
        )
        self.support = torch.zeros(
            (config.max_slots, num_layers, num_heads, self.blocks), device=device, dtype=torch.uint8
        )
        self.slots: dict[str, int] = {}
        self.free_slots = list(reversed(range(config.max_slots)))
        # request -> (position, discovery depths completed for that position); a
        # position whose discovery window was broken has depths None (dense to the end).
        self.tags: dict[str, tuple[int, frozenset | None]] = {}
        # Decode rows that attended densely because their discovery was incomplete.
        self.fallbacks = 0
        self._batch = None
        self._plan = None
        # Set when WISE only drafts for speculative decoding: target decode stays dense.
        self.draft_only = False
        # KV metadata -> per-row decode flags, set by the runner for decode traversals.
        # Rows without a flag (prefill chunks, padding) attend densely.
        self._decode = weakref.WeakKeyDictionary()
        # A FlashPagedAttention kernel when the cache uses FlashAttention 2/3/4: sparse
        # rows then attend the per-layer head union of their support (see wise_flash).
        self.flash = None
        # (selected, causal) block counts over all selections, for diagnostics.
        self.stats = torch.zeros(2, device=device, dtype=torch.int64)
        self.eta = 2.0 if config.support_override == "all" else config.mass_threshold

    def release(self, request_id: str) -> None:
        slot = self.slots.pop(request_id, None)
        if slot is not None:
            self.free_slots.append(slot)
        self.tags.pop(request_id, None)

    def _slot(self, request_id):
        if request_id not in self.slots:
            if not self.free_slots:
                raise RuntimeError("WISE support slots exhausted; raise WiseConfig.max_slots")
            self.slots[request_id] = self.free_slots.pop()
        return self.slots[request_id]

    def mark(self, metadata, decode_rows):
        """Declare which host rows of ``metadata`` are decode tokens (others stay dense)."""
        if any(decode_rows):
            self._decode[metadata] = tuple(decode_rows)

    def inherit(self, metadata, source):
        """Carry decode flags to a view of ``source`` (e.g. trimmed graph metadata)."""
        if source in self._decode:
            self._decode[metadata] = self._decode[source]

    def is_decode(self, metadata):
        return metadata in self._decode

    def plan(self, batch):
        """Host bookkeeping once per traversal; reused by every layer of the batch."""
        if self._batch is batch:
            return self._plan
        names = {id(allocation): request_id for request_id, allocation in batch.allocations}
        decode = self._decode.get(batch, ())
        modes, slots = [], []
        override = self.config.support_override
        for row, (allocation, depth, position) in enumerate(batch.rows):
            if row >= len(decode) or not decode[row]:
                modes.append(MODE_DENSE)
                slots.append(0)
                continue
            request_id = names[id(allocation)]
            if position >= self.config.max_model_len:
                raise ValueError("position exceeds WiseConfig.max_model_len")
            slots.append(self._slot(request_id))
            if override == "dense":
                modes.append(MODE_DENSE)
            elif depth in self.config.discovery_depths and isinstance(override, float):
                modes.append(MODE_DENSE)
            elif depth in self.config.discovery_depths:
                done = self.tags.get(request_id)
                previous = done[1] if done is not None and done[0] == position else frozenset()
                if depth == self.config.discovery_depths[0]:
                    # A new discovery window; a rejected speculative draft may revisit a position.
                    previous = frozenset()
                expected = frozenset(d for d in self.config.discovery_depths if d < depth)
                if previous != expected:
                    # An earlier discovery depth of this token did not record support.
                    self.tags[request_id] = (position, None)
                    self.fallbacks += 1
                    modes.append(MODE_DENSE)
                    continue
                self.tags[request_id] = (position, previous | {depth})
                first = depth == self.config.discovery_depths[0]
                modes.append(MODE_DISCOVER if first else MODE_DISCOVER_UNION)
            elif depth in self.config.reuse_depths:
                done = self.tags.get(request_id)
                if not isinstance(override, float) and (
                    done is None or done != (position, frozenset(self.config.discovery_depths))
                ):
                    self.fallbacks += 1
                    modes.append(MODE_DENSE)
                    continue
                modes.append(MODE_SPARSE)
            else:
                modes.append(MODE_DENSE)
        # Padding rows (no host row) attend densely and never touch support.
        padding = len(batch.context_lengths) - len(modes)
        modes += [MODE_DENSE] * padding
        slots += [0] * padding
        device = batch.context_lengths.device
        modes_t = torch.tensor(modes, dtype=torch.int32, pin_memory=device.type == "cuda").to(
            device, non_blocking=True
        )
        slots_t = torch.tensor(slots, dtype=torch.int32, pin_memory=device.type == "cuda").to(
            device, non_blocking=True
        )
        plan = {
            "modes": modes_t,
            "slots": slots_t,
            "scratch": self._scratch(len(modes), batch.block_tables, batch.context_lengths),
        }
        if isinstance(override, float):
            sparse = [i for i, m in enumerate(modes) if m == MODE_SPARSE]
            if sparse:
                self._fixed_support(batch, sparse, slots, override)
        self._batch, self._plan = batch, plan
        return plan

    def _fixed_support(self, batch, rows, slots, fraction):
        # Deterministic fixed-density control: every k-th block plus the newest.
        lengths = batch.context_lengths[rows].long()
        blocks = torch.arange(self.blocks, device=lengths.device)
        valid = blocks[None, :] * WISE_BLOCK < lengths[:, None]
        newest = (lengths[:, None] - 1) // WISE_BLOCK == blocks[None, :]
        stride = max(1, round(1 / fraction))
        chosen = ((blocks[None, :] % stride == 0) | newest) & valid
        index = torch.tensor([slots[i] for i in rows], device=lengths.device)
        scale = self.support.shape[1] * self.support.shape[2]
        self.stats[0] += chosen.sum() * scale
        self.stats[1] += valid.sum() * scale
        self.support[index] = (
            chosen[:, None, None, :].expand(-1, *self.support.shape[1:3], -1).to(torch.uint8)
        )

    def _scratch(self, rows, tables, lengths):
        scratch = wise_scratch(
            rows, self.support.shape[2], self.head_dim, self.blocks, self.support.device
        )
        if self.flash is not None:
            scratch["tables"], scratch["lengths"] = (
                torch.empty_like(tables),
                torch.empty_like(lengths),
            )
        return scratch

    def _run(self, layer, q, key_cache, value_cache, tables, lengths, modes, slots, scratch):
        if self.flash is None:
            return wise_paged_attention(
                q,
                key_cache,
                value_cache,
                tables,
                lengths,
                modes,
                slots,
                self.support,
                layer,
                scratch,
                self.stats,
                self.eta,
            )
        sparse_tables, sparse_lengths = compact_page_tables(
            tables,
            lengths,
            modes,
            slots,
            self.support,
            layer,
            key_cache.shape[1],
            scratch["tables"],
            scratch["lengths"],
        )
        output, lse = self.flash(
            q, key_cache, value_cache, sparse_tables, sparse_lengths, return_lse=True
        )
        discover_support(
            q,
            key_cache,
            tables,
            lengths,
            lse,
            modes,
            slots,
            self.support,
            layer,
            scratch,
            self.stats,
            self.eta,
        )
        return output

    def attend(self, layer, batch, q, key_cache, value_cache):
        plan = self.plan(batch)
        return self._run(
            layer,
            q,
            key_cache,
            value_cache,
            batch.block_tables,
            batch.context_lengths,
            plan["modes"],
            plan["slots"],
            plan["scratch"],
        )

    def static_inputs(self, rows, tables, lengths):
        """Graph-owned per-row modes, slots and scratch for one captured batch size."""
        device = self.support.device
        return {
            "modes": torch.zeros(rows, device=device, dtype=torch.int32),
            "slots": torch.zeros(rows, device=device, dtype=torch.int32),
            "scratch": self._scratch(rows, tables, lengths),
        }

    def copy_static(self, inputs, batch):
        """Host checks for this traversal, then device copies into the graph's inputs."""
        plan = self.plan(batch)
        rows = len(plan["modes"])
        inputs["modes"].zero_()
        inputs["modes"][:rows].copy_(plan["modes"])
        inputs["slots"][:rows].copy_(plan["slots"])

    def attend_static(self, layer, metadata, q, key_cache, value_cache):
        inputs = metadata.wise
        return self._run(
            layer,
            q,
            key_cache,
            value_cache,
            metadata.block_tables,
            metadata.context_lengths,
            inputs["modes"],
            inputs["slots"],
            inputs["scratch"],
        )

    def mean_density(self):
        """Selected / causal blocks of the support the reuse depths attend: the union
        over each token's discovery depths (or the fixed support), averaged over every
        (row, layer, head) selection so far."""
        selected, causal = self.stats.tolist()
        return selected / max(causal, 1)

    def density(self, request_id, layer, length):
        """Diagnostics: selected fraction of causal blocks for one request and layer."""
        blocks = math.ceil(length / WISE_BLOCK)
        row = self.support[self.slots[request_id], layer, :, :blocks]
        return float(row.float().mean())
