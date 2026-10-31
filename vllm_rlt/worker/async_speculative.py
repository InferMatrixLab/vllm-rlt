"""Bounded cross-round greedy speculation with device-owned token positions."""

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch

from vllm_rlt.attention.sparse import DENSE, reference_mode
from vllm_rlt.core.kv_cache_manager import _Allocation
from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import Operation, Request
from vllm_rlt.worker.cuda_graph import (
    CodaGraphs,
    DeviceRecurrentGraphs,
    speculative_capture_sizes,
)
from vllm_rlt.worker.executor import ForwardBatch
from vllm_rlt.worker.output import ModelRunnerOutput, SpeculativeResult
from vllm_rlt.worker.sampler import RngRegistry
from vllm_rlt.worker.speculative import (
    PipelineState,
    PipelineToken,
    SpeculativeRunner,
    SpeculativeStats,
)


@dataclass
class DevicePipelineState:
    """Fixed-capacity, oldest-first token slots; all validity and depths live on GPU."""

    tokens: torch.Tensor
    depths: torch.Tensor
    hidden: torch.Tensor
    candidates: torch.Tensor
    active: torch.Tensor


@dataclass
class SpeculativeRequestState:
    request: Request
    allocation: _Allocation
    token: torch.Tensor
    position: torch.Tensor
    pipeline: DevicePipelineState | None = None
    submitted: int = 0
    delivered: int = 0
    event: torch.cuda.Event | None = None
    # Row in the runner's device slot table holding (token, position).
    slot: int | None = None
    # Bumped per request instance so reference store bindings reject a reused
    # request ID across generations.
    generation: int = 0


class SpeculativeRoundBuffer:
    def __init__(self, runner, scheduler):
        cache = runner.cache
        width = math.ceil(runner.model.config.max_position_embeddings / cache.block_size)
        shape = (scheduler.max_num_seqs, cache.max_loops, width)
        self.host_tables = torch.empty(math.prod(shape), dtype=torch.int32, pin_memory=True)
        self.tables = torch.empty(math.prod(shape), dtype=torch.int32, device=cache.device)
        # Result capacity in int64 elements (not bytes), for one submission.
        # Block layout: [target token IDs for all verification rows,
        #                accepted counts per request, next positions per request].
        # Its upper bound is max_num_batched_tokens + 2 * max_num_seqs.
        # Pipeline layout: six consecutive values per request:
        # [prediction, accepted, emitted, next_position, drafted_count, verified_count].
        # `emitted` says whether this step produced a target output; prediction is
        # ignored otherwise. The last two fields are statistics for this step.
        # Reserve the larger layout in both GPU and pinned host storage; submit()
        # copies only the used prefix, and collect() decodes the selected layout.
        size = max(
            scheduler.max_num_batched_tokens + 2 * scheduler.max_num_seqs,
            6 * scheduler.max_num_seqs,
        )
        self.host_result = torch.empty(size, dtype=torch.int64, pin_memory=True)
        self.result = torch.empty(size, dtype=torch.int64, device=cache.device)
        self.leased = False
        self.copy_done: torch.cuda.Event | None = None
        self.compute_done: torch.cuda.Event | None = None
        self.keepalive: list[torch.Tensor] = []


@dataclass
class SpeculativeRoundTicket:
    batch: SchedulerOutput
    buffer: SpeculativeRoundBuffer
    states: tuple[SpeculativeRequestState, ...]
    rounds: tuple[int, ...]
    rows: int
    mode: Literal["block", "pipeline"] = "block"
    stats: SpeculativeStats | None = None
    cached: ModelRunnerOutput | None = None
    retired: bool = False
    # Reference leases of this round. Released on retire, after the final
    # reader completed, so no later submission can observe or overwrite them.
    references: tuple = ()
    reference_store: object | None = None

    @property
    def has_result(self):
        return True

    def ready(self):
        assert self.buffer.copy_done is not None
        return self.buffer.copy_done.query()

    def collect(self) -> ModelRunnerOutput:
        if self.cached is None:
            if self.retired:
                raise RuntimeError("cannot collect a retired speculative ticket")
            assert self.buffer.copy_done is not None
            self.buffer.copy_done.synchronize()
            b = len(self.batch.requests)
            if self.mode == "pipeline":
                values = self.buffer.host_result[: 6 * b].view(b, 6).tolist()
                self.cached = ModelRunnerOutput(
                    speculative=[
                        SpeculativeResult([token] if emitted else [], accepted)
                        for token, accepted, emitted, _, _, _ in values
                    ],
                    frontiers=[row[3] for row in values],
                )
                if self.stats is not None:
                    self.stats.drafted_tokens += sum(row[4] for row in values)
                    self.stats.verified_rows += sum(row[5] for row in values)
                return self.cached
            values = self.buffer.host_result[: self.rows + 2 * b].tolist()
            results, start = [], 0
            for i, scheduled_request in enumerate(self.batch.requests):
                accepted = values[self.rows + i]
                results.append(
                    SpeculativeResult(
                        values[start : start + accepted + 1],
                        accepted,
                    )
                )
                start += scheduled_request.token_count
            self.cached = ModelRunnerOutput(speculative=results, frontiers=values[self.rows + b :])
        return self.cached

    def retire(self, *, discard=False):
        if self.retired:
            return
        if self.cached is None:
            if not discard:
                raise RuntimeError("collect a speculative ticket before retiring its buffer")
            # Discarding still waits for the final reader, but does not need to
            # interpret host data that may have caused the original failure.
            assert self.buffer.copy_done is not None
            self.buffer.copy_done.synchronize()
        # collect() already waited for copy_done; normal delivery adds no wait.
        self.buffer.keepalive.clear()
        self.buffer.leased = False
        if self.reference_store is not None and self.references:
            self.reference_store.release(self.references)
        self.references = ()
        self.retired = True


class AsyncSpeculativeRunner(SpeculativeRunner):
    """Two submitted rounds may share a request; only GPU state joins them.

    Every request's work runs on one compute stream. CPU token_start is only a
    reservation upper bound and must never be used as the actual write position.
    """

    # Sparse decode attention lifecycle; test doubles that skip __init__
    # inherit the disabled defaults.
    reference_store = None
    sparse_decode = None

    def __init__(
        self,
        model,
        cache,
        config,
        scheduler,
        *,
        execution,
        compute_stream,
        copy_stream,
        rng: RngRegistry | None = None,
        runtime=None,
        reference_store=None,
        sparse_decode=None,
    ):
        super().__init__(
            model,
            cache,
            config,
            replace(execution, cuda_graphs=False),
            rng=rng,
            runtime=runtime,
            compute_stream=compute_stream,
        )
        # Sparse decode attention lifecycle: the runner owns the per-token
        # reference pool (RFC #156 section 5) and drives retirement from the
        # GPU acceptance counts without a host readback.
        self.reference_store = reference_store
        self.sparse_decode = sparse_decode
        self._generations: dict[str, int] = {}
        sizes = speculative_capture_sizes(execution, scheduler, config)
        self.graphs = (
            DeviceRecurrentGraphs(model, cache, execution, False, capture_sizes=sizes)
            if execution.cuda_graphs
            else None
        )
        self.coda_graphs = (
            CodaGraphs(model, execution, bucketed=True, capture_sizes=sizes)
            if execution.cuda_graphs
            else None
        )
        if self.graphs is not None:
            self.graphs.prepare()
            self.coda_graphs.prepare()
        self.compute_stream = compute_stream
        self.copy_stream = copy_stream
        self.compute_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.buffers = [SpeculativeRoundBuffer(self, scheduler) for _ in range(2)]
        self.states: dict[str, SpeculativeRequestState] = {}
        # Device (input token, next position) per live request. A round gathers
        # and scatters these rows in one operation each instead of per request.
        self.slot_values = torch.zeros(
            (2 * scheduler.max_num_seqs, 2), dtype=torch.long, device=self.device
        )
        self.free_slots = list(reversed(range(len(self.slot_values))))
        self._retired_slot_values: list[torch.Tensor] = []
        self.submitted_before_collect = 0
        self.discarded_rounds = 0
        self.readback_waits = 0

    def _state(self, request, buffer):
        rid = request.request_id
        allocation = self.cache._get_allocation(rid)
        state = self.states.get(rid)
        if state is not None:
            if state.request is not request or state.allocation is not allocation:
                raise RuntimeError("speculative request state reused before retirement")
            return state
        # Bootstrap only: all later input IDs and positions stay on the GPU.
        host = torch.tensor([request.input_token_id, request.position], pin_memory=True)
        values = host.to(self.device, non_blocking=True)
        buffer.keepalive.extend((host, values))
        generation = self._generations.get(rid, 0) + 1
        self._generations[rid] = generation
        if self.config.mode == "pipeline":
            state = SpeculativeRequestState(
                request, allocation, values[0], values[1], generation=generation
            )
        else:
            if not self.free_slots:
                self._grow_slots()
            slot = self.free_slots.pop()
            self.slot_values[slot].copy_(values)
            state = SpeculativeRequestState(
                request,
                allocation,
                self.slot_values[slot, 0],
                self.slot_values[slot, 1],
                slot=slot,
                generation=generation,
            )
        if self.config.mode == "pipeline":
            width = min(
                self.config.num_speculative_tokens + 1,
                self.config.target_loops // self.config.draft_loops,
            )
            parameter = next(self.model.parameters())
            state.pipeline = DevicePipelineState(
                state.token.expand(width).clone(),
                torch.zeros(width, dtype=torch.long, device=self.device),
                torch.zeros(
                    (width, self.model.config.hidden_size),
                    dtype=parameter.dtype,
                    device=self.device,
                ),
                torch.zeros(width, dtype=torch.long, device=self.device),
                torch.arange(width, device=self.device) == 0,
            )
        self.states[rid] = state
        return state

    def _grow_slots(self):
        """Double the slot table on the compute stream; live states move with it."""
        old = self.slot_values
        new = torch.zeros((2 * len(old), 2), dtype=old.dtype, device=old.device)
        new[: len(old)].copy_(old)
        # Work already queued still addresses the old storage.
        self._retired_slot_values.append(old)
        self.slot_values = new
        self.free_slots = list(reversed(range(len(old), len(new)))) + self.free_slots
        for state in self.states.values():
            state.token, state.position = new[state.slot, 0], new[state.slot, 1]

    def _execute_recurrent_step(
        self, hidden, positions, tables, depth, allocations, active=None, reference_plan=None
    ):
        if isinstance(depth, torch.Tensor):
            table = tables.gather(1, depth[:, None, None].expand(-1, 1, tables.shape[2])).squeeze(1)
        else:
            table = tables[:, depth].contiguous()
        blocks = table.gather(1, (positions // self.cache.block_size)[:, None]).squeeze(1)
        metadata = self.metadata.prepare_device(
            allocations=allocations,
            position_ids=positions,
            write_blocks=blocks.long(),
            write_offsets=positions % self.cache.block_size,
            block_tables=table,
            context_lengths=(
                (positions + 1) if active is None else torch.where(active, positions + 1, 0)
            ).int(),
            write_mask=active,
            reference_plan=reference_plan,
        )
        forward = ForwardBatch(
            Operation.RECURRENT, hidden=hidden, metadata=(metadata,), graphs=self.graphs
        )
        prepared = self.runtime.prepare(forward)
        return self.runtime.forward(prepared).hidden

    def _prepare_round(self, batch):
        buffer = next((b for b in self.buffers if not b.leased), None)
        if buffer is None:
            raise RuntimeError("speculative buffers exhausted; retire an earlier round")
        scheduled_requests = batch.requests
        if not scheduled_requests or any(
            scheduled_request.token_count <= 0 for scheduled_request in scheduled_requests
        ):
            raise ValueError("speculative round requires nonempty verification rows")
        allocations = tuple(
            (
                scheduled_request.request.request_id,
                self.cache._get_allocation(scheduled_request.request.request_id),
            )
            for scheduled_request in scheduled_requests
        )
        page_counts = []
        for scheduled_request, (_, allocation) in zip(scheduled_requests, allocations):
            upper = scheduled_request.token_start + scheduled_request.token_count
            if upper > allocation.max_tokens or upper <= 0:
                raise ValueError("speculative round exceeds reserved context")
            pages = math.ceil(upper / self.cache.block_size)
            if any(len(table) < pages for table in allocation.block_tables):
                raise ValueError("speculative round has unallocated KV pages")
            page_counts.append(pages)
        # A prefix of flat reusable storage provides contiguous active views even
        # as request count/page width change. The previous copy must be retired.
        shape = (len(scheduled_requests), self.cache.max_loops, max(page_counts))
        size = math.prod(shape)
        host_tables = buffer.host_tables[:size].view(shape)
        tables = buffer.tables[:size].view(shape)
        table_view = host_tables.numpy()
        table_view.fill(0)
        for i, ((_, allocation), pages) in enumerate(zip(allocations, page_counts)):
            for depth, table in enumerate(allocation.block_tables):
                table_view[i, depth, :pages] = table[:pages]
        buffer.leased = True
        return buffer, allocations, host_tables, tables

    def _round_plan(self, items, states, buffer):
        """Host-built gather/scatter indices for a round, uploaded in one copy.

        Offset o drafts the requests whose token_count exceeds o, in item order;
        verification rows stay request-major in item order, as results expect.
        """
        counts = np.array([item.token_count for item in items], dtype=np.int64)
        n, width = len(counts), int(counts.max())
        parts, layout = [], {}

        def add(name, values):
            layout[name] = (sum(len(p) for p in parts), len(values))
            parts.append(np.asarray(values, dtype=np.int64))

        add("slots", [state.slot for state in states])
        add("drafts", counts - 1)
        ranks = np.cumsum(counts[None, :] > np.arange(width)[:, None], axis=1) - 1
        sizes = (counts[None, :] > np.arange(width)[:, None]).sum(1)
        offsets = []
        for offset in range(width):
            active = np.nonzero(counts > offset)[0]
            drafting = np.nonzero(counts[active] > offset + 1)[0]
            offsets.append((len(active), len(drafting)))
            if len(active) < n:
                add(f"active{offset}", active)
            if len(drafting):
                add(f"drafting{offset}", active[drafting])
                if len(drafting) < len(active):
                    add(f"draft_rows{offset}", drafting)
        request_rows = np.repeat(np.arange(n), counts)
        row_offsets = np.arange(len(request_rows)) - np.repeat(np.cumsum(counts) - counts, counts)
        base = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        add("request_rows", request_rows)
        add("row_offsets", row_offsets)
        add("hidden_rows", base[row_offsets] + ranks[row_offsets, request_rows])
        host = torch.from_numpy(np.concatenate(parts)).pin_memory()
        values = host.to(self.device, non_blocking=True)
        buffer.keepalive.extend((host, values))
        views = {name: values[start : start + size] for name, (start, size) in layout.items()}
        return SimpleNamespace(
            n=n, width=width, rows=len(request_rows), offsets=offsets, counts=counts, **views
        )

    def _reference_plan(self, leases, rows, depth):
        """Build the sparse decode plan of one recurrent step.

        ``rows`` holds one ``(lease_index, row)`` pair per step row in step
        order. Dense steps return ``None`` so their metadata stays untouched.
        """
        if self.reference_store is None or leases is None:
            return None
        mode = reference_mode(depth)
        if mode == DENSE:
            return None
        owners = [(leases[lease_index], row) for lease_index, row in rows]
        return self.reference_store.build_plan(
            owners, [mode] * len(owners), key_retention=self.sparse_decode.key_retention
        )

    def _draft(self, plan, tables, allocations, leases=None):
        from vllm_rlt.kernels.routing import gather_kernel

        slot_values = self.slot_values.new_empty((plan.n, 2))
        gather_kernel[(plan.n, 1)](
            self.slot_values,
            plan.slots,
            slot_values,
            plan.n,
            H=2,
            IN_STRIDE=self.slot_values.stride(0),
            OUT_STRIDE=slot_values.stride(0),
            TILE=2,
        )
        starts = slot_values[:, 1]
        tokens = torch.zeros((plan.n, plan.width), dtype=torch.long, device=self.device)
        tokens[:, 0] = slot_values[:, 0]
        hidden_blocks = []
        for offset, (active_count, drafting_count) in enumerate(plan.offsets):
            active = getattr(plan, f"active{offset}", None)
            if active is None:
                row_tokens, positions, row_tables = tokens[:, offset], starts + offset, tables
            else:
                row_tokens = tokens[active, offset]
                positions = starts[active] + offset
                row_tables = tables[active]
            hidden = self.runtime.forward(ForwardBatch(Operation.PRELUDE, tokens=row_tokens)).hidden
            # A drafting offset covers one verification row per active request:
            # the token at column `offset`, which the verifier later reuses.
            rows = range(plan.n) if active is None else [int(index) for index in active]
            for depth in range(self.config.draft_loops):
                reference = self._reference_plan(leases, [(i, offset) for i in rows], depth)
                hidden = self._execute_recurrent_step(
                    hidden,
                    positions,
                    row_tables,
                    depth,
                    allocations,
                    reference_plan=reference,
                )
            hidden_blocks.append(hidden)
            if drafting_count:
                rows = getattr(plan, f"draft_rows{offset}", None)
                drafting = hidden if rows is None else hidden[rows]
                proposed = self._coda(drafting).argmax(-1)
                tokens[getattr(plan, f"drafting{offset}"), offset + 1] = proposed
        return starts, tokens, hidden_blocks

    def _verify(self, plan, starts, hidden_blocks, tables, allocations, leases=None):
        positions = starts[plan.request_rows] + plan.row_offsets
        row_tables = tables[plan.request_rows]
        hidden = torch.cat(hidden_blocks)[plan.hidden_rows]
        # Verification rows are request-major, matching the round plan.
        verification_rows = [
            (i, j) for i, count in enumerate(plan.counts) for j in range(int(count))
        ]
        for depth in range(self.config.draft_loops, self.config.target_loops):
            reference = self._reference_plan(leases, verification_rows, depth)
            if reference is not None and depth == self.config.draft_loops:
                # One device-owned absolute position per reference slot; the
                # verification rows cover every slot of the round exactly once.
                self.reference_store.bind_positions(reference, positions)
            hidden = self._execute_recurrent_step(
                hidden,
                positions,
                row_tables,
                depth,
                allocations,
                reference_plan=reference,
            )
        return self._coda(hidden).argmax(-1)

    def _accept_and_update_state(self, plan, states, starts, tokens, target, buffer, leases=None):
        from vllm_rlt.kernels.routing import scatter_kernel

        count = len(target)
        buffer.result[:count].copy_(target)
        targets = torch.zeros((plan.n, plan.width), dtype=torch.long, device=self.device)
        targets[plan.request_rows, plan.row_offsets] = target
        # Candidate j (token column j+1) is checked against target row j; only
        # a request's own drafts count. Acceptance stops at the first mismatch.
        drafts = torch.arange(plan.width - 1, device=self.device)[None, :] < plan.drafts[:, None]
        matches = (tokens[:, 1:] == targets[:, :-1]) & drafts
        accepted = matches.long().cumprod(1).sum(1)
        if leases is not None and self.reference_store is not None:
            # GPU acceptance drives retirement: rejected candidates and their
            # dependent descendants lose their references on the device.
            self.reference_store.accept(leases, accepted)
        # All target IDs through the first rejection equal the accepted
        # candidates followed by the correction (or final bonus).
        next_tokens = targets.gather(1, accepted[:, None]).squeeze(1)
        next_positions = starts + accepted + 1
        next_values = torch.stack((next_tokens, next_positions), 1)
        scatter_kernel[(plan.n, 1)](
            next_values,
            plan.slots,
            self.slot_values,
            H=2,
            IN_STRIDE=next_values.stride(0),
            OUT_STRIDE=self.slot_values.stride(0),
            TILE=2,
        )
        buffer.result[count : count + plan.n].copy_(accepted)
        buffer.result[count + plan.n : count + 2 * plan.n].copy_(next_positions)
        for state in states:
            state.submitted += 1
        return count

    @torch.inference_mode()
    def submit(self, batch):
        # 1. Reserve a slot and prepare only the active CPU page tables.
        buffer, allocations, host_tables, tables = self._prepare_round(batch)
        scheduled_requests = batch.requests
        # A partial submission may not record events. The engine drains both
        # owned streams before releasing allocations or recycling the slot.
        # 2. Submit computation; device state joins successive rounds.
        with torch.cuda.stream(self.compute_stream):
            states = tuple(
                self._state(scheduled_request.request, buffer)
                for scheduled_request in scheduled_requests
            )
            rounds = tuple(s.submitted for s in states)
            if any(s.submitted > s.delivered for s in states):
                self.submitted_before_collect += 1
            # One reference slot per verification row: the round's live tokens.
            # Outstanding rounds keep disjoint slots until their tickets retire.
            leases = None
            if self.reference_store is not None:
                leases = tuple(
                    self.reference_store.reserve(
                        scheduled_request.request.request_id,
                        index,
                        scheduled_request.token_count,
                        state.generation,
                    )
                    for index, (scheduled_request, state) in enumerate(
                        zip(scheduled_requests, states)
                    )
                )
            try:
                tables.copy_(host_tables, non_blocking=True)
                if self.config.mode == "pipeline":
                    count = self._pipeline_step(
                        scheduled_requests, states, tables, allocations, buffer
                    )
                else:
                    plan = self._round_plan(scheduled_requests, states, buffer)
                    starts, tokens, hidden_blocks = self._draft(plan, tables, allocations, leases)
                    target = self._verify(plan, starts, hidden_blocks, tables, allocations, leases)
                    count = self._accept_and_update_state(
                        plan, states, starts, tokens, target, buffer, leases
                    )
            except BaseException:
                # The engine drains both streams before discarding, so releasing
                # here cannot race a later submission for these slots.
                if leases is not None:
                    self.reference_store.release(leases)
                raise
            buffer.compute_done = torch.cuda.Event()
            buffer.compute_done.record(self.compute_stream)
            for state in states:
                state.event = buffer.compute_done
        # 3. Transfer compact results after computation; collect waits here.
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(buffer.compute_done)
            n = (
                6 * len(scheduled_requests)
                if self.config.mode == "pipeline"
                else count + 2 * len(scheduled_requests)
            )
            buffer.host_result[:n].copy_(buffer.result[:n], non_blocking=True)
            buffer.copy_done = torch.cuda.Event()
            buffer.copy_done.record(self.copy_stream)
        self.stats.rounds += len(scheduled_requests)
        if self.config.mode != "pipeline":
            self.stats.drafted_tokens += sum(s.token_count - 1 for s in scheduled_requests)
            self.stats.verified_rows += count
        return SpeculativeRoundTicket(
            batch,
            buffer,
            states,
            rounds,
            count,
            mode=self.config.mode,
            stats=self.stats,
            references=leases or (),
            reference_store=self.reference_store,
        )

    def _pipeline_step(self, spans, states, tables, allocations, buffer):
        widths = [
            min(span.token_count, len(state.pipeline.tokens)) for span, state in zip(spans, states)
        ]
        active = torch.cat([state.pipeline.active[:w] for state, w in zip(states, widths)])
        depths = torch.cat([state.pipeline.depths[:w] for state, w in zip(states, widths)])
        positions = torch.cat(
            [
                state.position + torch.arange(w, device=self.device)
                for state, w in zip(states, widths)
            ]
        )
        # Padding never reads beyond the reserved context or writes live KV.
        positions = torch.where(active, positions, 0)
        tokens = torch.cat([state.pipeline.tokens[:w] for state, w in zip(states, widths)])
        retained = torch.cat([state.pipeline.hidden[:w] for state, w in zip(states, widths)])
        prelude = self.runtime.forward(ForwardBatch(Operation.PRELUDE, tokens=tokens)).hidden
        hidden = torch.where((depths == 0)[:, None], prelude, retained)
        row_tables = torch.cat([tables[i : i + 1].expand(w, -1, -1) for i, w in enumerate(widths)])
        for offset in range(self.config.draft_loops):
            depth = torch.where(active, depths + offset, 0)
            hidden = self._execute_recurrent_step(
                hidden, positions, row_tables, depth, allocations, active
            )
        choices = self._coda(hidden).argmax(-1)
        start = 0
        for i, (span, state, w) in enumerate(zip(spans, states, widths)):
            p = state.pipeline
            ids = torch.arange(len(p.tokens), device=self.device)
            # Budget reductions discard speculative descendants, never the root.
            valid = p.active & (ids < w)
            new_depths = p.depths + valid.long() * self.config.draft_loops
            predictions = torch.zeros_like(p.tokens)
            predictions[:w].copy_(choices[start : start + w])
            next_hidden = p.hidden.clone()
            next_hidden[:w].copy_(hidden[start : start + w])
            drafted = valid & (new_depths == self.config.draft_loops)
            candidates = torch.where(drafted, predictions, p.candidates)
            emitted = valid[0] & (new_depths[0] == self.config.target_loops)
            accepted = emitted & (candidates[0] == predictions[0])
            reject = emitted & ~accepted
            state.token.copy_(torch.where(emitted, predictions[0], state.token))
            state.position.add_(emitted.long())
            # Shift only on target completion. All decisions remain stream-ordered tensors.
            source = (ids + emitted.long()).clamp_max(len(ids) - 1)
            remaining = valid.long().sum() - emitted.long()
            surviving = (ids < remaining) & ~reject
            p.tokens.copy_(p.tokens.gather(0, source))
            p.depths.copy_(new_depths.gather(0, source))
            p.hidden.copy_(next_hidden[source])
            p.candidates.copy_(candidates.gather(0, source))
            tail = (valid.long().sum() - 1).clamp_min(0)
            append = drafted.gather(0, tail.reshape(1)).squeeze(0) & ~reject
            newborn = (ids == remaining) & append
            # A rejection (or a one-row budget) restarts from the corrected root.
            restart = reject | (emitted & (remaining == 0))
            newborn = torch.where(restart, ids == 0, newborn)
            limit = len(span.request.prompt_token_ids) + span.request.sampling_params.max_tokens - 1
            newborn &= (ids < w) & (state.position + ids < limit)
            candidate = candidates.gather(0, tail.reshape(1)).squeeze(0)
            token = torch.where(restart, state.token, candidate)
            p.tokens.copy_(torch.where(newborn, token, p.tokens))
            p.depths.masked_fill_(newborn, 0)
            p.active.copy_((surviving | newborn) & (state.position + ids < limit))
            # Per-request wire order matches SpeculativeRoundTicket.collect():
            # prediction, accepted, emitted, next_position, drafted, verified.
            # A pipeline step verifies at most one token per request, so the
            # verified count equals emitted (0 or 1).
            buffer.result[6 * i : 6 * i + 6].copy_(
                torch.stack(
                    (
                        predictions[0],
                        accepted.long(),
                        emitted.long(),
                        state.position,
                        drafted.long().sum(),
                        emitted.long(),
                    )
                )
            )
            state.submitted += 1
            start += w
        return start

    def snapshot_pipeline(self, request, device):
        state = self.states.get(request.request_id)
        if state is None:
            return None
        if state.request is not request or state.submitted != state.delivered:
            raise RuntimeError("cannot suspend a pipeline with undelivered work")
        self.synchronize()
        pipeline = state.pipeline
        # Suspension already requires CPU delivery. Reuse the synchronous snapshot
        # contract here; never convert device progress on the submission hot path.
        position = int(state.position)
        values = torch.stack(
            (pipeline.tokens, pipeline.depths, pipeline.candidates, pipeline.active.long())
        ).tolist()
        tokens, depths, candidates, active = values
        return PipelineState(
            request,
            position,
            [
                PipelineToken(
                    position + i,
                    token,
                    depths[i],
                    pipeline.hidden[i].to(device).clone() if depths[i] else None,
                    candidates[i] if depths[i] else None,
                )
                for i, token in enumerate(tokens)
                if active[i]
            ],
        )

    def restore_pipeline(self, request, snapshot):
        if snapshot is None:
            return
        if snapshot.request is not request or request.request_id in self.states:
            raise RuntimeError("invalid pipeline restoration generation")
        if snapshot.pending or snapshot.position != request.position:
            raise RuntimeError("pipeline snapshot disagrees with committed request position")
        width = min(
            self.config.num_speculative_tokens + 1,
            self.config.target_loops // self.config.draft_loops,
        )
        if len(snapshot.tokens) > width:
            raise RuntimeError("pipeline snapshot exceeds the configured window")
        with torch.cuda.stream(self.compute_stream):
            values = torch.tensor(
                [
                    [t.token for t in snapshot.tokens] + [0] * (width - len(snapshot.tokens)),
                    [t.depth for t in snapshot.tokens] + [0] * (width - len(snapshot.tokens)),
                    [t.candidate or 0 for t in snapshot.tokens]
                    + [0] * (width - len(snapshot.tokens)),
                ],
                device=self.device,
            )
            hidden = torch.zeros(
                (width, self.model.config.hidden_size),
                device=self.device,
                dtype=next(self.model.parameters()).dtype,
            )
            for i, token in enumerate(snapshot.tokens):
                if token.hidden is not None:
                    hidden[i].copy_(token.hidden)
            pipeline = DevicePipelineState(
                values[0],
                values[1],
                hidden,
                values[2],
                torch.arange(width, device=self.device) < len(snapshot.tokens),
            )
            self.states[request.request_id] = SpeculativeRequestState(
                request,
                self.cache._get_allocation(request.request_id),
                torch.tensor(request.input_token_id, device=self.device),
                torch.tensor(snapshot.position, device=self.device),
                pipeline,
            )

    def request_ready(self, request):
        state = self.states.get(request.request_id)
        return state is None or state.event is None or state.event.query()

    def release(self, request):
        state = self.states.get(request.request_id)
        if state is not None:
            if state.request is not request:
                raise RuntimeError("releasing a different speculative request generation")
            if state.event is not None:
                state.event.synchronize()
            del self.states[request.request_id]
            if state.slot is not None:
                self.free_slots.append(state.slot)

    def discard_buffers(self):
        """Reset round storage after both streams drain and all tickets retire.

        A failed submission may own a buffer without having returned a ticket.
        The engine orders draining and ticket retirement; this runner owns the
        remaining buffer cleanup. No host result needs to be decoded here.
        """
        for buffer in self.buffers:
            buffer.keepalive.clear()
            buffer.leased = False
        # Test doubles borrow this method without the runner __init__.
        reference_store = getattr(self, "reference_store", None)
        if reference_store is not None:
            # Both streams have drained and every ticket retired; any leftover
            # binding can only come from a failed submission.
            reference_store.discard()

    def synchronize(self):
        self.compute_stream.synchronize()
        self.copy_stream.synchronize()
