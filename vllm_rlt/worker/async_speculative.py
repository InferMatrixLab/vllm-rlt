"""Bounded cross-round greedy speculation with device-owned token positions."""

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import torch

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
from vllm_rlt.worker.speculative import SpeculativeRunner


@dataclass
class SpeculativeRequestState:
    request: Request
    allocation: _Allocation
    token: torch.Tensor
    position: torch.Tensor
    submitted: int = 0
    delivered: int = 0
    event: torch.cuda.Event | None = None
    # Row in the runner's device slot table holding (token, position).
    slot: int | None = None


class SpeculativeRoundBuffer:
    def __init__(self, runner, scheduler):
        cache = runner.cache
        width = math.ceil(runner.model.config.max_position_embeddings / cache.block_size)
        shape = (scheduler.max_num_seqs, cache.max_loops, width)
        self.host_tables = torch.empty(math.prod(shape), dtype=torch.int32, pin_memory=True)
        self.tables = torch.empty(math.prod(shape), dtype=torch.int32, device=cache.device)
        # One target token ID per verification row, then accepted candidate
        # counts and next token positions (one value of each per request).
        size = scheduler.max_num_batched_tokens + 2 * scheduler.max_num_seqs
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
    cached: ModelRunnerOutput | None = None
    retired: bool = False
    # (start event, end event, round features) when round timing is enabled.
    timing: tuple | None = None

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
        self.retired = True


class AsyncSpeculativeRunner(SpeculativeRunner):
    """Two submitted rounds may share a request; only GPU state joins them.

    Every request's work runs on one compute stream. CPU token_start is only a
    reservation upper bound and must never be used as the actual write position.
    """

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
        # Set by the engine for the adaptive draft policy: callable(items) -> features.
        self.round_features = None

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
        if not self.free_slots:
            self._grow_slots()
        slot = self.free_slots.pop()
        self.slot_values[slot].copy_(values)
        state = SpeculativeRequestState(
            request, allocation, self.slot_values[slot, 0], self.slot_values[slot, 1], slot=slot
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

    def _execute_recurrent_step(self, hidden, positions, tables, depth, allocations):
        table = tables[:, depth].contiguous()
        blocks = table.gather(1, (positions // self.cache.block_size)[:, None]).squeeze(1)
        metadata = self.metadata.prepare_device(
            allocations=allocations,
            position_ids=positions,
            write_blocks=blocks.long(),
            write_offsets=positions % self.cache.block_size,
            block_tables=table,
            context_lengths=(positions + 1).int(),
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
        return SimpleNamespace(n=n, width=width, rows=len(request_rows), offsets=offsets, **views)

    def _draft(self, plan, tables, allocations):
        slot_values = self.slot_values[plan.slots]
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
            for depth in range(self.config.draft_loops):
                hidden = self._execute_recurrent_step(
                    hidden, positions, row_tables, depth, allocations
                )
            hidden_blocks.append(hidden)
            if drafting_count:
                rows = getattr(plan, f"draft_rows{offset}", None)
                drafting = hidden if rows is None else hidden[rows]
                proposed = self._coda(drafting).argmax(-1)
                tokens[getattr(plan, f"drafting{offset}"), offset + 1] = proposed
        return starts, tokens, hidden_blocks

    def _verify(self, plan, starts, hidden_blocks, tables, allocations):
        positions = starts[plan.request_rows] + plan.row_offsets
        row_tables = tables[plan.request_rows]
        hidden = torch.cat(hidden_blocks)[plan.hidden_rows]
        for depth in range(self.config.draft_loops, self.config.target_loops):
            hidden = self._execute_recurrent_step(hidden, positions, row_tables, depth, allocations)
        return self._coda(hidden).argmax(-1)

    def _accept_and_update_state(self, plan, states, starts, tokens, target, buffer):
        count = len(target)
        buffer.result[:count].copy_(target)
        targets = torch.zeros((plan.n, plan.width), dtype=torch.long, device=self.device)
        targets[plan.request_rows, plan.row_offsets] = target
        # Candidate j (token column j+1) is checked against target row j; only
        # a request's own drafts count. Acceptance stops at the first mismatch.
        drafts = torch.arange(plan.width - 1, device=self.device)[None, :] < plan.drafts[:, None]
        matches = (tokens[:, 1:] == targets[:, :-1]) & drafts
        accepted = matches.long().cumprod(1).sum(1)
        # All target IDs through the first rejection equal the accepted
        # candidates followed by the correction (or final bonus).
        next_tokens = targets.gather(1, accepted[:, None]).squeeze(1)
        next_positions = starts + accepted + 1
        self.slot_values[plan.slots] = torch.stack((next_tokens, next_positions), 1)
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
            tables.copy_(host_tables, non_blocking=True)
            timing = None
            if self.round_features is not None:
                start = torch.cuda.Event(enable_timing=True)
                start.record(self.compute_stream)
            plan = self._round_plan(scheduled_requests, states, buffer)
            starts, tokens, hidden_blocks = self._draft(plan, tables, allocations)
            target = self._verify(plan, starts, hidden_blocks, tables, allocations)
            count = self._accept_and_update_state(plan, states, starts, tokens, target, buffer)
            if self.round_features is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record(self.compute_stream)
                timing = (start, end, self.round_features(scheduled_requests))
            buffer.compute_done = torch.cuda.Event()
            buffer.compute_done.record(self.compute_stream)
            for state in states:
                state.event = buffer.compute_done
        # 3. Transfer compact results after computation; collect waits here.
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(buffer.compute_done)
            n = count + 2 * len(scheduled_requests)
            buffer.host_result[:n].copy_(buffer.result[:n], non_blocking=True)
            buffer.copy_done = torch.cuda.Event()
            buffer.copy_done.record(self.copy_stream)
        self.stats.rounds += len(scheduled_requests)
        self.stats.drafted_tokens += sum(
            scheduled_request.token_count - 1 for scheduled_request in scheduled_requests
        )
        self.stats.verified_rows += count
        return SpeculativeRoundTicket(batch, buffer, states, rounds, count, timing=timing)

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

    def synchronize(self):
        self.compute_stream.synchronize()
        self.copy_stream.synchronize()
