"""Bounded cross-round greedy speculation with device-owned token positions."""

import math
from dataclasses import dataclass

import torch

from vllm_rlt.core.execution import Operation
from vllm_rlt.core.kv_cache_manager import _Allocation
from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import Request
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
        super().__init__(model, cache, config, execution, rng=rng, runtime=runtime)
        self.compute_stream = compute_stream
        self.copy_stream = copy_stream
        self.compute_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.buffers = [SpeculativeRoundBuffer(self, scheduler) for _ in range(2)]
        self.states: dict[str, SpeculativeRequestState] = {}
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
        state = SpeculativeRequestState(request, allocation, values[0], values[1])
        self.states[rid] = state
        return state

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

    def _draft(self, scheduled_requests, states, tables, allocations):
        starts = [s.position.clone() for s in states]
        candidates = [[] for _ in scheduled_requests]
        hidden_rows = [[] for _ in scheduled_requests]
        for offset in range(
            max(scheduled_request.token_count for scheduled_request in scheduled_requests)
        ):
            active = [
                i
                for i, scheduled_request in enumerate(scheduled_requests)
                if offset < scheduled_request.token_count
            ]
            tokens = torch.stack(
                [states[i].token if offset == 0 else candidates[i][-1] for i in active]
            )
            positions = torch.stack([starts[i] + offset for i in active])
            row_tables = torch.stack([tables[i] for i in active])
            hidden = self.runtime.forward(ForwardBatch(Operation.PRELUDE, tokens=tokens)).hidden
            for depth in range(self.config.draft_loops):
                hidden = self._execute_recurrent_step(
                    hidden, positions, row_tables, depth, allocations
                )
            drafting = []
            for row, i in enumerate(active):
                hidden_rows[i].append(hidden[row])
                if offset + 1 < scheduled_requests[i].token_count:
                    drafting.append((row, i))
            if drafting:
                proposed = self._coda(torch.stack([hidden[row] for row, _ in drafting])).argmax(-1)
                for row, (_, i) in enumerate(drafting):
                    candidates[i].append(proposed[row])
        return starts, candidates, hidden_rows

    def _verify(self, scheduled_requests, starts, hidden_rows, tables, allocations):
        positions = torch.cat(
            [
                start + torch.arange(scheduled_request.token_count, device=self.device)
                for start, scheduled_request in zip(starts, scheduled_requests)
            ]
        )
        row_tables = torch.cat(
            [
                tables[i : i + 1].expand(scheduled_request.token_count, -1, -1)
                for i, scheduled_request in enumerate(scheduled_requests)
            ]
        )
        hidden = torch.stack([h for rows in hidden_rows for h in rows])
        for depth in range(self.config.draft_loops, self.config.target_loops):
            hidden = self._execute_recurrent_step(hidden, positions, row_tables, depth, allocations)
        target = self._coda(hidden).argmax(-1)
        return target

    def _accept_and_update_state(
        self, scheduled_requests, states, starts, candidates, target, buffer
    ):
        count = len(target)
        buffer.result[:count].copy_(target)
        start = 0
        for i, (scheduled_request, state) in enumerate(zip(scheduled_requests, states)):
            rows = target[start : start + scheduled_request.token_count]
            if candidates[i]:
                matches = torch.stack(candidates[i]) == rows[:-1]
                accepted = matches.long().cumprod(0).sum()
            else:
                accepted = rows.new_zeros(())
            # All target IDs through the first rejection equal the accepted
            # candidates followed by the correction (or final bonus).
            state.token.copy_(rows.gather(0, accepted.reshape(1)).squeeze(0))
            state.position.copy_(starts[i] + accepted + 1)
            buffer.result[count + i].copy_(accepted)
            buffer.result[count + len(scheduled_requests) + i].copy_(state.position)
            state.submitted += 1
            start += scheduled_request.token_count
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
            starts, candidates, hidden_rows = self._draft(
                scheduled_requests, states, tables, allocations
            )
            target = self._verify(scheduled_requests, starts, hidden_rows, tables, allocations)
            count = self._accept_and_update_state(
                scheduled_requests, states, starts, candidates, target, buffer
            )
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
        return SpeculativeRoundTicket(batch, buffer, states, rounds, count)

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
