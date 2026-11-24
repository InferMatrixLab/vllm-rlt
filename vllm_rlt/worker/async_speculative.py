"""Bounded cross-round speculation with device-owned token positions.

Block mode supports greedy and random (temperature/top-k/top-p) requests in one batch. Greedy
requests verify by target argmax. Random requests draw drafts from their ``q`` distributions,
verify with speculative rejection sampling against ``p`` (see ``speculative_sampling``) and
return their committed sequence (accepted drafts, then correction or bonus) in the result rows.
A batch without random requests takes the greedy path with no extra device work.
"""

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Literal, NamedTuple

import numpy as np
import torch

from vllm_rlt.core.kv_cache_manager import _Allocation
from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import Operation, Request
from vllm_rlt.worker.cuda_graph import (
    CodaGraphs,
    CodaPairGraphs,
    DeviceRecurrentGraphs,
    PreludeGraphs,
    graph_capture_sizes,
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
from vllm_rlt.worker.speculative_sampling import (
    AcceptResult,
    batched_probabilities,
    categorical,
    speculative_accept,
)
from vllm_rlt.worker.tensors import copy_tensor, pinned_tensor

# Float plan arrays travel in the int64 upload as float64 bit patterns.
_FLOAT_UPLOADS = frozenset({"r_temperature", "r_top_p", "v_temperature", "v_top_p"})


@dataclass(frozen=True, eq=False)
class RandomSamplingPlan:
    """Host plan for the random requests of one block round (numpy only, no device access).

    Random requests are those with ``temperature != 0``; their *rank* is their position among
    random requests in item order, and it indexes ``u``/``q``/``p`` rows. ``width`` is the
    largest ``token_count`` among random requests (``q`` is ``[r, width - 1, V]``, ``p`` is
    ``[r, width, V]``).
    """

    requests: np.ndarray  # [r] item index of each random request
    temperature: np.ndarray  # [r] float64
    top_k: np.ndarray  # [r] int64, as given (-1 = full vocabulary)
    top_p: np.ndarray  # [r] float64
    max_top_k: int  # largest k with 0 < k < vocab over random requests, else 0
    apply_top_p: bool  # any random request has top_p < 1
    width: int
    # Per draft offset o < width - 1: (rows of that offset's drafting block that are random,
    # their ranks). Empty arrays when the offset drafts no random request.
    draft_rows: tuple[tuple[np.ndarray, np.ndarray], ...]
    verify_rows: np.ndarray  # flat request-major verify rows of random requests
    verify_ranks: np.ndarray  # rank owning each such row
    verify_offsets: np.ndarray  # row index within its request (p / result column)
    verify_temperature: np.ndarray  # per verify row, float64
    verify_top_k: np.ndarray
    verify_top_p: np.ndarray

    def uploads(self):
        """(name, int64 array) pairs appended to the round's single index upload."""
        parts = [
            ("r_requests", self.requests),
            ("r_temperature", self.temperature),
            ("r_top_k", self.top_k),
            ("r_top_p", self.top_p),
            ("v_rows", self.verify_rows),
            ("v_ranks", self.verify_ranks),
            ("v_offsets", self.verify_offsets),
            ("v_temperature", self.verify_temperature),
            ("v_top_k", self.verify_top_k),
            ("v_top_p", self.verify_top_p),
        ]
        for offset, (rows, ranks) in enumerate(self.draft_rows):
            if len(rows):
                parts += [(f"r_rows{offset}", rows), (f"r_ranks{offset}", ranks)]
        return [
            (
                name,
                np.asarray(values, dtype=np.float64).view(np.int64)
                if name in _FLOAT_UPLOADS
                else np.asarray(values, dtype=np.int64),
            )
            for name, values in parts
        ]


def plan_random_sampling(sampling_params, token_counts, vocab_size, num_speculative_tokens):
    """Plan the random rows of a block round, or return None when every request is greedy.

    ``sampling_params[i]`` and ``token_counts[i]`` describe item ``i`` (``token_count - 1``
    drafts). Draft ``j`` of a request is proposed at offset ``j`` and checked against its
    verify row ``j``; verify rows are request-major. Pure host code: no device access.
    """
    counts = np.asarray(token_counts, dtype=np.int64)
    is_random = np.array([params.temperature != 0 for params in sampling_params], dtype=bool)
    if not is_random.any():
        return None
    requests = np.flatnonzero(is_random)
    r = len(requests)
    rank_of = np.full(len(counts), -1, dtype=np.int64)
    rank_of[requests] = np.arange(r)
    request_counts = counts[requests]
    width = int(request_counts.max())
    if width - 1 > num_speculative_tokens:
        raise ValueError("random request has more drafts than num_speculative_tokens")
    temperature = np.array([sampling_params[i].temperature for i in requests], dtype=np.float64)
    top_k = np.array([sampling_params[i].top_k for i in requests], dtype=np.int64)
    top_p = np.array([sampling_params[i].top_p for i in requests], dtype=np.float64)
    truncating = top_k[(top_k > 0) & (top_k < vocab_size)]
    draft_rows = []
    for offset in range(width - 1):
        drafting = np.flatnonzero(counts > offset + 1)
        rows = np.flatnonzero(is_random[drafting])
        draft_rows.append((rows, rank_of[drafting[rows]]))
    starts = np.cumsum(counts) - counts
    verify_rows = np.concatenate([starts[i] + np.arange(counts[i]) for i in requests]).astype(
        np.int64
    )
    verify_ranks = np.repeat(np.arange(r), request_counts)
    verify_offsets = np.concatenate([np.arange(c) for c in request_counts]).astype(np.int64)
    return RandomSamplingPlan(
        requests=requests,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_top_k=int(truncating.max()) if len(truncating) else 0,
        apply_top_p=bool((top_p < 1).any()),
        width=width,
        draft_rows=tuple(draft_rows),
        verify_rows=verify_rows,
        verify_ranks=verify_ranks,
        verify_offsets=verify_offsets,
        verify_temperature=np.repeat(temperature, request_counts),
        verify_top_k=np.repeat(top_k, request_counts),
        verify_top_p=np.repeat(top_p, request_counts),
    )


class RandomRoundMerge(NamedTuple):
    rows: torch.Tensor  # [R] result rows (committed sequences for random requests)
    accepted: torch.Tensor  # [n]
    next_tokens: torch.Tensor  # [n] (placeholders for failed requests)
    next_positions: torch.Tensor  # [n] starts + accepted + 1 (informational for failed rows)
    next_values: torch.Tensor  # [n, 2] (token, position) to scatter into the slot table
    error: torch.Tensor  # [n] int64 0/1 flags


def merge_random_round(
    target,
    accepted,
    next_tokens,
    starts,
    input_tokens,
    result: AcceptResult,
    random_requests,
    verify_rows,
    verify_ranks,
    verify_offsets,
):
    """Overlay random-request outcomes on the greedy round tensors; no host reads.

    ``target`` ``[R]`` argmax per request-major verify row, ``accepted``/``next_tokens``
    ``[n]`` the greedy values, ``starts``/``input_tokens`` ``[n]`` the slot values read at the
    start of the round. ``result`` is the ``speculative_accept`` outcome of the random requests
    ``random_requests`` ``[r]``. Verify row ``start_i + j`` of random request ``rank`` becomes
    ``result.tokens[rank, j]`` (accepted drafts, then the final token); greedy rows are
    untouched. A failed request keeps its old ``(input_token, start)`` slot values, so a queued
    next round reads the span it already reserved; the caller must treat the round as fatal.
    """
    rows = target.clone()
    rows[verify_rows] = result.tokens[verify_ranks, verify_offsets]
    accepted = accepted.clone()
    accepted[random_requests] = result.accepted
    next_tokens = next_tokens.clone()
    next_tokens[random_requests] = result.final
    failed = torch.zeros(len(starts), dtype=torch.bool, device=starts.device)
    failed[random_requests] = result.error
    next_positions = starts + accepted + 1
    next_values = torch.where(
        failed[:, None],
        torch.stack((input_tokens, starts), 1),
        torch.stack((next_tokens, next_positions), 1),
    )
    return RandomRoundMerge(rows, accepted, next_tokens, next_positions, next_values, failed.long())


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


class SpeculativeRoundBuffer:
    def __init__(self, runner, scheduler, *, pipeline_capacity=0):
        cache = runner.cache
        width = math.ceil(runner.model.config.max_position_embeddings / cache.block_size)
        shape = (scheduler.max_num_seqs, cache.max_loops, width)
        self.host_tables = pinned_tensor((math.prod(shape),), torch.int32, cache.device)
        self.table_versions = ()
        self.table_shape = ()
        self.tables_dirty = False
        self.tables = torch.empty(math.prod(shape), dtype=torch.int32, device=cache.device)
        self.plan_key = None
        self.plan = None
        if pipeline_capacity:
            size = ((3 + 2 * pipeline_capacity) * scheduler.max_num_seqs,)
            self.host_plan = pinned_tensor(size, torch.int64, cache.device)
            self.device_plan = torch.empty(size, dtype=torch.int64, device=cache.device)
        self.inputs_ready = None
        # Result capacity in int64 elements (not bytes), for one submission.
        # Block layout: [result token IDs for all verification rows,
        #                accepted counts per request, next positions per request,
        #                error flags per request (only when the round has random requests)].
        # Result rows are target argmax IDs for greedy requests and committed sequences for
        # random ones. Its upper bound is max_num_batched_tokens + 3 * max_num_seqs.
        # Pipeline layout: six consecutive values per request:
        # [prediction, accepted, emitted, next_position, drafted_count, verified_count].
        # `emitted` says whether this step produced a target output; prediction is
        # ignored otherwise. The last two fields are statistics for this step.
        # Reserve the larger layout in both GPU and pinned host storage; submit()
        # copies only the used prefix, and collect() decodes the selected layout.
        size = max(
            scheduler.max_num_batched_tokens + 3 * scheduler.max_num_seqs,
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
    # Block rounds with random requests append one numerical-error flag per request.
    error_flags: bool = False

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
            # [rows | accepted (b) | next positions (b) | error flags (b, random rounds only)]
            # Only these ranges are read; a reused buffer's tail may hold older rounds' values.
            values = self.buffer.host_result[: self.rows + (3 if self.error_flags else 2) * b]
            values = values.tolist()
            flags = values[self.rows + 2 * b : self.rows + 3 * b] if self.error_flags else None
            results, start = [], 0
            for i, scheduled_request in enumerate(self.batch.requests):
                accepted = values[self.rows + i]
                results.append(
                    SpeculativeResult(
                        values[start : start + accepted + 1],
                        accepted,
                        numerical_error=bool(flags[i]) if flags is not None else False,
                    )
                )
                start += scheduled_request.token_count
            self.cached = ModelRunnerOutput(
                speculative=results, frontiers=values[self.rows + b : self.rows + 2 * b]
            )
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
            replace(execution, cuda_graphs=False, cuda_graph_warmup=False),
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
            if execution.cuda_graphs and not self.ordered_pipeline_coda
            else None
        )
        self.coda_pair_graphs = None
        if execution.cuda_graphs and self.ordered_pipeline_coda:
            sizes = graph_capture_sizes(execution, preferred=(scheduler.max_num_seqs,))
            self.prelude_graphs = PreludeGraphs(
                model, execution, cache, bucketed=True, capture_sizes=sizes
            )
            self.coda_pair_graphs = CodaPairGraphs(model, cache, sizes)
            self.prelude_graphs.prepare()
            self.coda_pair_graphs.prepare()
        if self.graphs is not None:
            self.graphs.prepare()
        if self.coda_graphs is not None:
            self.coda_graphs.prepare()
        self.compute_stream = compute_stream
        self.copy_stream = copy_stream
        self.input_stream = (
            torch.cuda.Stream(device=self.device) if execution.multi_stream else compute_stream
        )
        self.use_uva = execution.use_uva
        self.pipeline_capacity = (
            min(config.num_speculative_tokens + 1, config.target_loops // config.draft_loops)
            if config.mode == "pipeline"
            else 0
        )
        self.pipeline_lanes = torch.arange(self.pipeline_capacity, device=self.device)
        self.compute_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.buffers = [
            SpeculativeRoundBuffer(self, scheduler, pipeline_capacity=self.pipeline_capacity)
            for _ in range(2)
        ]
        if self.input_stream is not self.compute_stream:
            for buffer in self.buffers:
                buffer.inputs_ready = torch.cuda.Event()
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
        if self.config.mode == "pipeline":
            state = SpeculativeRequestState(request, allocation, values[0], values[1])
        else:
            if not self.free_slots:
                self._grow_slots()
            slot = self.free_slots.pop()
            self.slot_values[slot].copy_(values)
            state = SpeculativeRequestState(
                request, allocation, self.slot_values[slot, 0], self.slot_values[slot, 1], slot=slot
            )
        if self.config.mode == "pipeline":
            width = self.pipeline_capacity
            parameter = next(self.model.parameters())
            state.pipeline = DevicePipelineState(
                state.token.expand(width).clone(),
                torch.zeros(width, dtype=torch.long, device=self.device),
                torch.zeros(
                    (width, self.model.config.hidden_size),
                    dtype=getattr(self.model, "state_dtype", parameter.dtype),
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

    def _device_kv(self, positions, table, allocations, active):
        blocks = table.gather(1, (positions // self.cache.block_size)[:, None]).squeeze(1)
        return self.metadata.prepare_device(
            allocations=allocations,
            position_ids=positions,
            write_blocks=blocks.long(),
            write_offsets=positions % self.cache.block_size,
            block_tables=table,
            context_lengths=(
                (positions + 1) if active is None else torch.where(active, positions + 1, 0)
            ).int(),
            write_mask=active,
        )

    def _execute_recurrent_step(
        self, hidden, positions, tables, depth, allocations, active=None, *, prompt=None
    ):
        if isinstance(depth, torch.Tensor):
            table = tables.gather(1, depth[:, None, None].expand(-1, 1, tables.shape[2])).squeeze(1)
        else:
            table = tables[:, depth].contiguous()
        metadata = self._device_kv(positions, table, allocations, active)
        if prompt is not None:
            runner, prepared = prompt
            prompt_forward = prepared.forwards[0].forward
            decode_rows = len(hidden)
            metadata = self.metadata.join_recurrent(metadata, prompt_forward.metadata[0])
            forward = ForwardBatch(
                Operation.RECURRENT,
                hidden=torch.cat((hidden, prompt_forward.hidden)),
                metadata=(metadata,),
            )
            result = self.runtime.forward(forward).hidden
            runner.finish_mixed_recurrent(prepared, result[decode_rows:])
            return result[:decode_rows]
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
        width, versions = 0, []
        for scheduled_request, (_, allocation) in zip(scheduled_requests, allocations):
            upper = scheduled_request.token_start + scheduled_request.token_count
            if upper > allocation.max_tokens or upper <= 0:
                raise ValueError("speculative round exceeds reserved context")
            pages = math.ceil(upper / self.cache.block_size)
            version = allocation.block_tables
            if any(len(table) < pages for table in version):
                raise ValueError("speculative round has unallocated KV pages")
            width = max(width, pages)
            versions.append(version)
        # A prefix of flat reusable storage provides contiguous active views even
        # as request count/page width change. The previous copy must be retired.
        shape = (len(scheduled_requests), self.cache.max_loops, width)
        size = math.prod(shape)
        host_tables = buffer.host_tables[:size].view(shape)
        tables = buffer.tables[:size].view(shape)
        buffer.tables_dirty = (
            buffer.table_shape != shape
            or any(old is not new for old, new in zip(buffer.table_versions, versions, strict=True))
        )
        if buffer.tables_dirty:
            table_view = host_tables.numpy()
            table_view.fill(0)
            # Upload allocated pages through this batch's width. Device context
            # lengths still limit each row. Advancing within these pages is free.
            for i, version in enumerate(versions):
                for depth, table in enumerate(version):
                    pages = min(width, len(table))
                    table_view[i, depth, :pages] = table[:pages]
            # Retain immutable tuples for identity checks, including across KV growth.
            buffer.table_versions = tuple(versions)
            buffer.table_shape = shape
        buffer.leased = True
        return buffer, allocations, host_tables, tables

    def _stage_pipeline_plan(self, spans, buffer):
        capacity = self.pipeline_capacity
        widths = tuple(min(span.token_count, capacity) for span in spans)
        limits = tuple(
            len(span.request.prompt_token_ids) + span.request.sampling_params.max_tokens - 1
            for span in spans
        )
        key = (widths, limits)
        if key == buffer.plan_key:
            return False
        roots, rows, owners = [], [], []
        for index, width in enumerate(widths):
            roots.append(len(rows))
            rows.extend(range(index * capacity, index * capacity + width))
            owners.extend([index] * width)
        n, num_rows = len(widths), len(rows)
        size = 3 * n + 2 * num_rows
        host = buffer.host_plan[:size]
        host.numpy()[:] = (*widths, *limits, *roots, *rows, *owners)
        buffer.plan = buffer.device_plan[:size].split((n, n, n, num_rows, num_rows))
        buffer.plan_key = key
        return True

    def _round_plan(self, items, states, buffer):
        """Host-built gather/scatter indices for a round, uploaded in one copy.

        Offset o drafts the requests whose token_count exceeds o, in item order;
        verification rows stay request-major in item order, as results expect. When any
        request is random, ``plan.random`` is its host plan and its arrays join the same upload;
        for an all-greedy round ``plan.random`` is None and nothing extra is built.
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
        random = plan_random_sampling(
            [item.request.sampling_params for item in items],
            counts,
            self.model.config.vocab_size,
            self.config.num_speculative_tokens,
        )
        if random is not None:
            for name, values in random.uploads():
                add(name, values)
        host = torch.from_numpy(np.concatenate(parts)).pin_memory()
        values = host.to(self.device, non_blocking=True)
        buffer.keepalive.extend((host, values))
        views = {name: values[start : start + size] for name, (start, size) in layout.items()}
        for name in _FLOAT_UPLOADS & views.keys():
            views[name] = views[name].view(torch.float64).float()
        return SimpleNamespace(
            n=n, width=width, rows=len(request_rows), offsets=offsets, random=random, **views
        )

    def _uniforms(self, items, random):
        """Uniforms ``[r, 2K + 1]`` for the random requests, drawn in item order.

        Columns ``[0:K]`` sample drafts, ``[K:2K]`` decide acceptance and ``[2K]`` samples the
        correction or bonus token. Each request consumes exactly one draw per round whatever
        its draft count or acceptance outcome; greedy requests never touch the registry.
        """
        rows = []
        for i in random.requests:
            request = items[i].request
            generator = self.rng.acquire(request.request_id, request.sampling_params, self.device)
            rows.append(
                torch.rand(
                    (2 * self.config.num_speculative_tokens + 1,),
                    generator=generator,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
        return torch.stack(rows)

    def _probabilities(self, plan, logits, temperature, top_k, top_p):
        return batched_probabilities(
            logits,
            temperature,
            top_k,
            top_p,
            max_top_k=plan.random.max_top_k,
            apply_top_p=plan.random.apply_top_p,
        )

    def _draft(self, plan, tables, allocations, u=None):
        """Propose drafts; returns ``(starts, tokens, hidden_blocks, q)``.

        Greedy rows take the CODA argmax. For random rows (``plan.random``) the proposal is
        drawn from ``q = batched_probabilities(logits)`` with ``u[rank, offset]`` and ``q`` is
        stored in ``q[rank, offset]`` (``q`` is None when no random request drafts).
        """
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
        random = plan.random
        q = None
        if random is not None and random.width > 1:
            q = torch.zeros(
                (len(random.requests), random.width - 1, self.model.config.vocab_size),
                dtype=torch.float32,
                device=self.device,
            )
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
                # CUDA graphs may return a static buffer that the next _coda replay
                # overwrites: everything below is derived from `logits` immediately.
                logits = self._coda(drafting)
                proposed = logits.argmax(-1)
                random_rows = getattr(plan, f"r_rows{offset}", None)
                if random_rows is not None:
                    ranks = getattr(plan, f"r_ranks{offset}")
                    q_o = self._probabilities(
                        plan,
                        logits[random_rows],
                        plan.r_temperature[ranks],
                        plan.r_top_k[ranks],
                        plan.r_top_p[ranks],
                    )
                    # Invalid q rows give placeholder token 0; speculative_accept flags them.
                    sampled, _ = categorical(q_o, u[ranks, offset])
                    proposed[random_rows] = sampled
                    q[ranks, offset] = q_o
                tokens[getattr(plan, f"drafting{offset}"), offset + 1] = proposed
        return starts, tokens, hidden_blocks, q

    def _verify(self, plan, starts, hidden_blocks, tables, allocations):
        """Run the target loops; returns ``(target argmax per verify row, p)``.

        ``p`` ``[r, width, V]`` holds the verify distributions of the random requests (zeros
        in padding) and is None when every request is greedy.
        """
        positions = starts[plan.request_rows] + plan.row_offsets
        row_tables = tables[plan.request_rows]
        hidden = torch.cat(hidden_blocks)[plan.hidden_rows]
        for depth in range(self.config.draft_loops, self.config.target_loops):
            hidden = self._execute_recurrent_step(hidden, positions, row_tables, depth, allocations)
        logits = self._coda(hidden)
        target = logits.argmax(-1)
        random = plan.random
        if random is None:
            return target, None
        p = torch.zeros(
            (len(random.requests), random.width, self.model.config.vocab_size),
            dtype=torch.float32,
            device=self.device,
        )
        p[plan.v_ranks, plan.v_offsets] = self._probabilities(
            plan, logits[plan.v_rows], plan.v_temperature, plan.v_top_k, plan.v_top_p
        )
        return target, p

    def _accept_and_update_state(
        self, plan, states, starts, tokens, target, buffer, *, u=None, q=None, p=None
    ):
        """Commit the round on the device and fill ``buffer.result``.

        Layout: ``[rows (count) | accepted (n) | next positions (n)]`` plus ``n`` error flags
        when ``plan.random`` is set. Greedy requests keep the target-argmax rows; random
        requests write their committed sequence (accepted drafts, then correction or bonus).
        """
        from vllm_rlt.kernels.routing import scatter_kernel

        count = len(target)
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
        rows = target
        random = plan.random
        if random is None:
            next_positions = starts + accepted + 1
            next_values = torch.stack((next_tokens, next_positions), 1)
        else:
            width = random.width
            requests = plan.r_requests
            if q is None:
                q = p.new_empty((len(random.requests), 0, p.shape[-1]))
            k = self.config.num_speculative_tokens
            outcome = speculative_accept(
                tokens[requests, 1:width],
                plan.drafts[requests],
                q,
                p,
                u[:, k : k + width - 1],
                u[:, 2 * k],
            )
            merged = merge_random_round(
                target,
                accepted,
                next_tokens,
                starts,
                tokens[:, 0],
                outcome,
                requests,
                plan.v_rows,
                plan.v_ranks,
                plan.v_offsets,
            )
            rows, accepted, next_positions, next_values = (
                merged.rows,
                merged.accepted,
                merged.next_positions,
                merged.next_values,
            )
        buffer.result[:count].copy_(rows)
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
        if random is not None:
            # Written every random round: a reused buffer never exposes older flags.
            buffer.result[count + 2 * plan.n : count + 3 * plan.n].copy_(merged.error)
        for state in states:
            state.submitted += 1
        return count

    @torch.inference_mode()
    def submit(self, batch, *, prompt=None):
        # 1. Reserve a slot and prepare only the active CPU page tables.
        buffer, allocations, host_tables, tables = self._prepare_round(batch)
        scheduled_requests = batch.requests
        plan_dirty = self.config.mode == "pipeline" and self._stage_pipeline_plan(
            scheduled_requests, buffer
        )
        upload = buffer.tables_dirty or plan_dirty
        if upload:
            # The ticket keeps this bank immutable through readback. Upload can
            # overlap earlier compute and uses neither the compute nor D2H stream.
            with torch.cuda.stream(self.input_stream):
                if buffer.tables_dirty:
                    copy_tensor(host_tables, tables, use_uva=self.use_uva)
                if plan_dirty:
                    size = sum(t.numel() for t in buffer.plan)
                    copy_tensor(
                        buffer.host_plan[:size], buffer.device_plan[:size], use_uva=self.use_uva
                    )
                if buffer.inputs_ready is not None:
                    buffer.inputs_ready.record(self.input_stream)
        # A partial submission may not record events. The engine drains both
        # owned streams before releasing allocations or recycling the slot.
        # 2. Submit computation; device state joins successive rounds.
        with torch.cuda.stream(self.compute_stream):
            if upload and buffer.inputs_ready is not None:
                self.compute_stream.wait_event(buffer.inputs_ready)
            states = tuple(
                self._state(scheduled_request.request, buffer)
                for scheduled_request in scheduled_requests
            )
            rounds = tuple(s.submitted for s in states)
            if any(s.submitted > s.delivered for s in states):
                self.submitted_before_collect += 1
            error_flags = False
            if self.config.mode == "pipeline":
                args = (scheduled_requests, states, tables, allocations, buffer)
                count = (
                    self._pipeline_step(*args)
                    if prompt is None
                    else self._pipeline_step(*args, prompt=prompt)
                )
            else:
                plan = self._round_plan(scheduled_requests, states, buffer)
                u = None if plan.random is None else self._uniforms(scheduled_requests, plan.random)
                starts, tokens, hidden_blocks, q = self._draft(plan, tables, allocations, u)
                target, p = self._verify(plan, starts, hidden_blocks, tables, allocations)
                count = self._accept_and_update_state(
                    plan, states, starts, tokens, target, buffer, u=u, q=q, p=p
                )
                error_flags = plan.random is not None
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
                else count + (3 if error_flags else 2) * len(scheduled_requests)
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
            error_flags=error_flags,
        )

    def _pipeline_step(self, spans, states, tables, allocations, buffer, *, prompt=None):
        pipelines = [state.pipeline for state in states]
        pipeline = DevicePipelineState(
            torch.stack([state.tokens for state in pipelines]),
            torch.stack([state.depths for state in pipelines]),
            torch.stack([state.hidden for state in pipelines]),
            torch.stack([state.candidates for state in pipelines]),
            torch.stack([state.active for state in pipelines]),
        )
        num_requests, capacity = pipeline.tokens.shape
        widths, output_limits, roots, rows, owners = buffer.plan
        num_rows = len(rows)
        lanes = self.pipeline_lanes
        valid = pipeline.active & (lanes < widths[:, None])
        lengths = valid.sum(1)
        active = valid.flatten()[rows]
        depths = pipeline.depths.flatten()[rows]
        positions = torch.stack([state.position for state in states])
        row_positions = (positions[:, None] + lanes).flatten()[rows]
        # Padding never reads beyond the reserved context or writes live KV.
        row_positions = torch.where(active, row_positions, 0)
        tokens = pipeline.tokens.flatten()[rows]
        retained = pipeline.hidden.flatten(0, 1)[rows]
        if self.ordered_pipeline_coda:
            tails = roots + (lengths - 1).clamp_min(0)
            boundary_tables = tables[:, 0].contiguous()
            draft_rows = active[tails] & (depths[tails] == 0)
            prelude = self._prelude(
                tokens[tails],
                self._device_kv(row_positions[tails], boundary_tables, allocations, draft_rows),
            )
            hidden = retained
            hidden[tails] = torch.where(draft_rows[:, None], prelude, retained[tails])
        else:
            prelude = self._prelude(tokens)
            hidden = torch.where((depths == 0)[:, None], prelude, retained)
        row_tables = tables[owners]
        for offset in range(self.config.draft_loops):
            depth = torch.where(active, depths + offset, 0)
            hidden = self._execute_recurrent_step(
                hidden, row_positions, row_tables, depth, allocations, active, prompt=prompt
            )
        if self.ordered_pipeline_coda:
            target_rows = active[roots] & (
                depths[roots] + self.config.draft_loops == self.config.target_loops
            )
            # Write verified Coda history before the draft reads any earlier position.
            target_kv = self._device_kv(
                row_positions[roots], boundary_tables, allocations, target_rows
            )
            draft_kv = self._device_kv(
                row_positions[tails], boundary_tables, allocations, draft_rows
            )
            if self.coda_pair_graphs is None:
                target, draft = self.model.coda_pair(
                    hidden[roots], target_kv, hidden[tails], draft_kv, self.cache
                )
            else:
                target, draft = self.coda_pair_graphs.forward(
                    hidden[roots], target_kv, hidden[tails], draft_kv
                )
            choices = torch.zeros_like(tokens)
            choices[roots] = target
            choices[tails] = torch.where(draft_rows, draft, choices[tails])
        else:
            choices = self._coda(hidden).argmax(-1)
        predictions = torch.zeros_like(pipeline.tokens)
        predictions.flatten().index_copy_(0, rows, choices)
        pipeline.hidden.flatten(0, 1).index_copy_(0, rows, hidden)
        new_depths = pipeline.depths + valid.long() * self.config.draft_loops
        drafted = valid & (new_depths == self.config.draft_loops)
        candidates = torch.where(drafted, predictions, pipeline.candidates)
        emitted = valid[:, 0] & (new_depths[:, 0] == self.config.target_loops)
        accepted = emitted & (candidates[:, 0] == predictions[:, 0])
        reject = emitted & ~accepted
        committed = emitted.long()
        input_tokens = torch.stack([state.token for state in states])
        next_tokens = torch.where(emitted, predictions[:, 0], input_tokens)
        next_positions = positions + committed
        # Shift after target completion. Rejecting a candidate invalidates its
        # descendants, while shrinking the reservation masks excess slots.
        source = (lanes + committed[:, None]).clamp_max(capacity - 1)
        remaining = lengths - committed
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
        shifted_tokens = pipeline.tokens.gather(1, source)
        shifted_depths = new_depths.gather(1, source)
        shifted_hidden = pipeline.hidden.gather(
            1, source[:, :, None].expand(-1, -1, pipeline.hidden.shape[2])
        )
        # Publish the device state before the next submission on this stream.
        torch._foreach_copy_([state.token for state in states], next_tokens.unbind())
        torch._foreach_copy_([state.position for state in states], next_positions.unbind())
        torch._foreach_copy_(
            [state.tokens for state in pipelines],
            torch.where(newborn, newborn_tokens[:, None], shifted_tokens).unbind(),
        )
        torch._foreach_copy_(
            [state.depths for state in pipelines], shifted_depths.masked_fill_(newborn, 0).unbind()
        )
        torch._foreach_copy_([state.hidden for state in pipelines], shifted_hidden.unbind())
        torch._foreach_copy_(
            [state.candidates for state in pipelines], candidates.gather(1, source).unbind()
        )
        torch._foreach_copy_(
            [state.active for state in pipelines], ((surviving | newborn) & within_limit).unbind()
        )
        # Per-request wire order matches SpeculativeRoundTicket.collect().
        # Each pipeline step verifies at most one token per request.
        buffer.result[: 6 * num_requests].view(num_requests, 6).copy_(
            torch.stack(
                (
                    predictions[:, 0],
                    accepted.long(),
                    committed,
                    next_positions,
                    drafted.sum(1),
                    committed,
                ),
                dim=1,
            )
        )
        for state in states:
            state.submitted += 1
        return num_rows

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
            buffer.table_shape = ()
            buffer.plan_key = None

    def synchronize(self):
        if self.input_stream is not self.compute_stream:
            self.input_stream.synchronize()
        self.compute_stream.synchronize()
        self.copy_stream.synchronize()
