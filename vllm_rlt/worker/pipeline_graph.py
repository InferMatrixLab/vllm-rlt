"""Pre-captured CUDA work for native Inkling's device-owned speculative pipeline."""

import math
from typing import NamedTuple

import torch

from vllm_rlt.worker.cuda_graph import _capture, _DeviceCache
from vllm_rlt.worker.metadata import DeviceKVMetadata


class PipelineUpdate(NamedTuple):
    token: torch.Tensor
    position: torch.Tensor
    tokens: torch.Tensor
    depths: torch.Tensor
    candidates: torch.Tensor
    active: torch.Tensor
    source: torch.Tensor
    wire: torch.Tensor


def pipeline_transition(
    tokens,
    depths,
    previous_candidates,
    predictions,
    valid,
    widths,
    positions,
    output_limits,
    input_tokens,
    lanes,
    lengths,
    draft_loops,
    target_loops,
):
    new_depths = depths + valid.long() * draft_loops
    drafted = valid & (new_depths == draft_loops)
    candidates = torch.where(drafted, predictions, previous_candidates)
    emitted = valid[:, 0] & (new_depths[:, 0] == target_loops)
    accepted = emitted & (candidates[:, 0] == predictions[:, 0])
    reject = emitted & ~accepted
    committed = emitted.long()
    next_tokens = torch.where(emitted, predictions[:, 0], input_tokens)
    next_positions = positions + committed
    source = (lanes + committed[:, None]).clamp_max(tokens.shape[1] - 1)
    remaining = lengths - committed
    # Shift after target completion and discard descendants on rejection.
    surviving = (lanes < remaining[:, None]) & ~reject[:, None]
    tail = (lengths - 1).clamp_min(0)[:, None]
    append = drafted.gather(1, tail).squeeze(1) & ~reject
    restart = reject | (emitted & (remaining == 0))
    newborn = (lanes == remaining[:, None]) & append[:, None]
    newborn = torch.where(restart[:, None], lanes == 0, newborn)
    within_limit = next_positions[:, None] + lanes < output_limits[:, None]
    newborn &= (lanes < widths[:, None]) & within_limit
    candidate = candidates.gather(1, tail).squeeze(1)
    newborn_tokens = torch.where(restart, next_tokens, candidate)
    # Wire layout consumed by SpeculativeRoundTicket.collect().
    wire = torch.stack(
        (predictions[:, 0], accepted.long(), committed, next_positions, drafted.sum(1), committed),
        dim=1,
    )
    return PipelineUpdate(
        next_tokens,
        next_positions,
        torch.where(newborn, newborn_tokens[:, None], tokens.gather(1, source)),
        new_depths.gather(1, source).masked_fill_(newborn, 0),
        candidates.gather(1, source),
        (surviving | newborn) & within_limit,
        source,
        wire,
    )


def publish_pipeline(states, hidden, update, buffer):
    """Commit before the next compute submission and stage compact D2H output."""
    count = len(states)
    for name in ("token", "position"):
        torch._foreach_copy_(
            [getattr(state, name) for state in states], getattr(update, name)[:count].unbind()
        )
    for name in ("tokens", "depths", "candidates", "active", "hidden"):
        values = hidden if name == "hidden" else getattr(update, name)
        torch._foreach_copy_(
            [getattr(state.pipeline, name) for state in states], values[:count].unbind()
        )
    buffer.result[: 6 * count].view(count, 6).copy_(update.wire[:count])


class PipelineTransitionGraphs:
    """Capture integer state updates by request count on the pipeline's compute stream."""

    def __init__(self, device, sizes, capacity, config):
        self.lanes = torch.arange(capacity, device=device)
        self.entries = {}
        self.replays = 0
        stream = torch.cuda.Stream(device=device)
        pool = torch.cuda.graph_pool_handle()
        for count in sizes:
            self._prepare(device, count, capacity, config, stream, pool)

    def _prepare(self, device, count, capacity, config, stream, pool):
        rows = torch.zeros((5, count, capacity), device=device, dtype=torch.long)
        requests = torch.zeros((4, count), device=device, dtype=torch.long)

        def run():
            tokens, depths, candidates, predictions, visible = rows.unbind()
            widths, positions, limits, input_tokens = requests.unbind()
            valid = visible.bool()
            return pipeline_transition(
                tokens,
                depths,
                candidates,
                predictions,
                valid,
                widths,
                positions,
                limits,
                input_tokens,
                self.lanes,
                valid.sum(1),
                config.draft_loops,
                config.target_loops,
            )

        graph, update = _capture(rows, stream, pool, run, startup=True)
        self.entries[count] = (rows, requests, graph, update)

    def forward(self, pipeline, predictions, valid, widths, positions, limits, input_tokens):
        rows, requests, graph, update = self.entries[len(widths)]
        torch.stack(
            (pipeline.tokens, pipeline.depths, pipeline.candidates, predictions, valid), out=rows
        )
        torch.stack((widths, positions, limits, input_tokens), out=requests)
        graph.replay()
        self.replays += 1
        # The caller enqueues every output consumer on this stream before reuse.
        return update


class PipelineRoundGraph:
    """One dense decode bucket. Mixed rounds and sparse reservations use stage graphs."""

    @torch.inference_mode()
    def __init__(self, model, cache, requests, capacity, config):
        self.model = model
        self.cache = _DeviceCache(cache)
        self.requests = requests
        self.capacity = capacity
        self.config = config
        self.pages = min(512, math.ceil(model.config.max_position_embeddings / cache.block_size))
        self.replays = 0
        self._table_versions = ()
        self._table_shape = ()
        self._limits = ()
        device = cache.device
        shape = (requests, capacity)
        self.tokens = torch.zeros(shape, dtype=torch.long, device=device)
        self.depths = torch.zeros_like(self.tokens)
        self.candidates = torch.zeros_like(self.tokens)
        self.active = torch.zeros(shape, dtype=torch.bool, device=device)
        dtype = getattr(model, "state_dtype", next(model.parameters()).dtype)
        self.hidden = torch.zeros((*shape, model.config.hidden_size), dtype=dtype, device=device)
        self.positions = torch.zeros(requests, dtype=torch.long, device=device)
        self.input_tokens = torch.zeros_like(self.positions)
        self.output_limits = torch.zeros_like(self.positions)
        self.tables = torch.zeros(
            (requests, cache.max_loops, self.pages), dtype=torch.int32, device=device
        )
        self.roots = capacity * torch.arange(requests, device=device)
        self.owners = torch.arange(requests, device=device).repeat_interleave(capacity)
        self.lanes = torch.arange(capacity, device=device)
        self.widths = torch.full_like(self.positions, capacity)
        stream = torch.cuda.Stream(device=device)
        self.graph, self.result = _capture(
            self.hidden, stream, torch.cuda.graph_pool_handle(), self._run, startup=True
        )

    def supports(self, widths, tables):
        count = len(widths)
        return (
            self.requests * 3 <= count * 4 <= self.requests * 4
            and tables.shape[-1] <= self.pages
            and all(width == self.capacity for width in widths)
        )

    def _kv(self, positions, tables, active):
        block_size = self.cache.block_size
        blocks = tables.gather(1, (positions // block_size)[:, None]).squeeze(1)
        return DeviceKVMetadata(
            allocations=(),
            position_ids=positions,
            write_blocks=blocks.long(),
            write_offsets=positions % block_size,
            block_tables=tables,
            context_lengths=torch.where(active, positions + 1, 0).int(),
            write_mask=active,
        )

    def _run(self):
        draft_loops, target_loops = self.config.draft_loops, self.config.target_loops
        lengths = self.active.sum(1)
        active = self.active.flatten()
        depths = self.depths.flatten()
        positions = (self.positions[:, None] + self.lanes).flatten()
        positions = torch.where(active, positions, 0)
        tails = self.roots + (lengths - 1).clamp_min(0)
        draft_rows = active[tails] & (depths[tails] == 0)
        boundary = self.tables[:, 0].contiguous()
        draft_kv = self._kv(positions[tails], boundary, draft_rows)
        prelude = self.model.prelude_prepared(self.tokens.flatten()[tails], draft_kv, self.cache)
        hidden = self.hidden.flatten(0, 1).clone()
        hidden[tails] = torch.where(draft_rows[:, None], prelude, hidden[tails])
        for offset in range(draft_loops):
            planes = torch.where(active, depths + offset, 0)
            tables = self.tables[self.owners, planes].contiguous()
            hidden, _ = self.model.recurrent_prepared(
                hidden, self._kv(positions, tables, active), self.cache, compute_gate=False
            )
        target_rows = active[self.roots] & (depths[self.roots] + draft_loops == target_loops)
        target_kv = self._kv(positions[self.roots], boundary, target_rows)
        target, draft = self.model.coda_pair(
            hidden[self.roots], target_kv, hidden[tails], draft_kv, self.cache
        )
        predictions = torch.zeros_like(self.tokens)
        predictions.flatten()[self.roots] = target
        predictions.flatten()[tails] = torch.where(draft_rows, draft, predictions.flatten()[tails])
        update = pipeline_transition(
            self.tokens,
            self.depths,
            self.candidates,
            predictions,
            self.active,
            self.widths,
            self.positions,
            self.output_limits,
            self.input_tokens,
            self.lanes,
            lengths,
            draft_loops,
            target_loops,
        )
        hidden = hidden.reshape_as(self.hidden).gather(
            1, update.source[:, :, None].expand(-1, -1, self.hidden.shape[2])
        )
        return hidden, update

    def forward(self, states, tables, buffer):
        count = len(states)
        if count < self.requests:
            self.active[count:].zero_()
        if self._table_shape != buffer.table_shape or any(
            old is not new
            for old, new in zip(self._table_versions, buffer.table_versions, strict=True)
        ):
            self.tables[:count, :, : tables.shape[-1]].copy_(tables)
            self._table_shape = buffer.table_shape
            self._table_versions = buffer.table_versions
        pipes = [state.pipeline for state in states]
        for name in ("tokens", "depths", "candidates", "active", "hidden"):
            torch.stack([getattr(pipe, name) for pipe in pipes], out=getattr(self, name)[:count])
        torch.stack([state.position for state in states], out=self.positions[:count])
        torch.stack([state.token for state in states], out=self.input_tokens[:count])
        limits = buffer.plan_key[1]
        if self._limits != limits:
            self.output_limits[:count].copy_(buffer.plan[1])
            self._limits = limits
        self.graph.replay()
        hidden, update = self.result
        publish_pipeline(states, hidden, update, buffer)
        self.replays += 1
