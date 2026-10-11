"""Stage execution, random lookahead signals and event-owned CUDA submissions."""

import math
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch

from vllm_rlt.config import ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import Request, Stage
from vllm_rlt.worker.buffers import Workspace
from vllm_rlt.worker.cuda_graph import RecurrentGraphs
from vllm_rlt.worker.loopcd import Reference, active, extrapolate
from vllm_rlt.worker.sampler import Sampler


@dataclass
class ReadbackSlot:
    storage: torch.Tensor
    leased: bool = False
    event: object = None


@dataclass(frozen=True)
class PreparedExecution:
    batch: SchedulerOutput
    depths: tuple[int, ...]
    positions: tuple[int, ...]
    output_indices: tuple[int, ...]
    token_tensors: tuple[torch.Tensor, ...] = ()
    kv: object = None
    routing: object = None


@dataclass
class Submission:
    batch: SchedulerOutput
    values: torch.Tensor | None
    event: object = None
    cached: list | None = None
    slot: ReadbackSlot | None = None
    device_values: torch.Tensor | None = None
    depths: tuple[int, ...] = ()
    output_indices: tuple[int, ...] = ()

    def __del__(self):
        if self.slot is not None:
            self.slot.leased = False

    def ready(self):
        return self.event is None or self.event.query()

    def collect(self):
        if self.cached is not None:
            return self.cached
        if self.event is not None:
            self.event.synchronize()
        if self.cached is None and self.values is not None:
            self.cached = self.values.tolist()
            self.values = None
            if self.slot is not None:
                self.slot.leased = False
                self.slot = None
        return self.cached


class ModelRunner:
    def __init__(
        self,
        model,
        cache_manager,
        *,
        exit_config=None,
        execution_config=None,
        scheduler_config=None,
    ):
        self.model = model.eval()
        self.cache_manager = cache_manager
        parameter = next(model.parameters())
        self.device = parameter.device
        self.sampler = Sampler(self.device)
        self.exit_config = exit_config or ExitConfig()
        self.execution_config = execution_config or ExecutionConfig()
        self.loopcd_references = {}
        self.loopcd_stats = dict(
            captures=0,
            guided_rows=0,
            head_calls=0,
            head_rows=0,
            skipped_prompt_head_calls=0,
            skipped_prompt_head_rows=0,
            reference_peak_bytes=0,
            prefill_core_rows=0,
            prefill_submitted_rows=0,
            decode_core_rows=0,
            decode_submitted_rows=0,
        )
        self.prefill_depth = self.execution_config.prefill_depth or model.config.total_ut_steps
        scheduler = scheduler_config or SchedulerConfig()
        self.graphs = (
            RecurrentGraphs(
                model,
                cache_manager,
                self.execution_config,
                self.exit_config.mode in ("ouro", "ouro_delayed"),
            )
            if self.execution_config.cuda_graphs
            else None
        )
        self._prefill_banks = None
        self._prefill_bank_index = 0
        self.lookahead_head = None
        if self.exit_config.mode == "random_lookahead":
            # Keep auxiliary random weights separate from the strict base checkpoint.
            # CPU RNG isolation also makes initialization independent of device count.
            with torch.random.fork_rng(devices=[]):
                generator = torch.Generator(device="cpu").manual_seed(self.exit_config.seed)
                head = torch.nn.Linear(model.config.hidden_size, 1, device="cpu")
                torch.nn.init.normal_(
                    head.weight, std=model.config.initializer_range, generator=generator
                )
                torch.nn.init.zeros_(head.bias)
            self.lookahead_head = head.to(device=self.device, dtype=parameter.dtype).eval()
            self.lookahead_head.requires_grad_(False)
        self.core_stream = self.boundary_stream = self.copy_stream = None
        if self.execution_config.async_scheduling and self.device.type == "cuda":
            self.core_stream = torch.cuda.Stream(device=self.device)
            self.boundary_stream = (
                torch.cuda.Stream(device=self.device)
                if self.execution_config.multi_stream
                else self.core_stream
            )
            self.copy_stream = (
                torch.cuda.Stream(device=self.device)
                if self.execution_config.multi_stream
                else self.core_stream
            )
            for stream in (self.core_stream, self.boundary_stream, self.copy_stream):
                stream.wait_stream(torch.cuda.current_stream(self.device))
        self.readback_slots = []
        if self.execution_config.async_scheduling and self.device.type == "cuda":
            # cudaHostAlloc during submission can serialize streams. Allocate all
            # readback storage before work starts, retaining leases until consumed.
            for dtype in (torch.float32, torch.int64):
                for _ in range(scheduler.max_num_seqs + 4):
                    self.readback_slots.append(
                        ReadbackSlot(
                            torch.empty(
                                scheduler.max_num_batched_tokens, dtype=dtype, pin_memory=True
                            )
                        )
                    )
        # Two core rounds plus one interleaved boundary stage can be in flight.
        # Prepare the next batch BEFORE retiring the oldest submission.
        self.submission_events = deque()
        self.events = {}
        self.workspaces = {}
        self.workspace_index = {"core": 0, "boundary": 0}
        self.state_slots = {}
        self.free_state_slots = list(reversed(range(scheduler.max_num_seqs)))
        self.states = None
        self.last_effective_size = self.last_submitted_size = 0
        if self.execution_config.static_buffers:
            rows = scheduler.max_num_batched_tokens
            if self.execution_config.pad_to_power_of_two:
                rows = 1 << (rows - 1).bit_length()
            width = math.ceil(model.config.max_position_embeddings / cache_manager.block_size)
            for group in ("core", "boundary"):
                self.workspaces[group] = [
                    Workspace(rows, width, model.config.hidden_size, self.device, parameter.dtype)
                    for _ in range(2)
                ]
            self.states = torch.empty(
                (scheduler.max_num_seqs, model.config.hidden_size),
                device=self.device,
                dtype=parameter.dtype,
            )
            # Buffers allocated after stream creation must be visible there too.
            for stream in (self.core_stream, self.boundary_stream):
                if stream is not None:
                    stream.wait_stream(torch.cuda.current_stream(self.device))

        self.async_state = None
        if self.core_stream is not None:
            from vllm_rlt.worker.async_state import AsyncState

            rows = scheduler.max_num_batched_tokens
            if self.execution_config.pad_to_power_of_two:
                rows = 1 << (rows - 1).bit_length()
            self.async_state = AsyncState(
                cache_manager, model.config, scheduler, rows, use_uva=not torch.version.hip
            )
            self.states = self.async_state.hidden
            self.state_slots = self.async_state.slots
            self.free_state_slots = self.async_state.free
            for stream in (self.core_stream, self.boundary_stream, self.copy_stream):
                stream.wait_stream(torch.cuda.current_stream(self.device))

    def _size(self, count):
        size = count
        if self.execution_config.pad_to_power_of_two:
            size = 1 << (count - 1).bit_length()
        self.last_effective_size, self.last_submitted_size = count, size
        return size

    def _workspace(self, group):
        if not self.workspaces:
            return None
        index = self.workspace_index[group]
        self.workspace_index[group] = 1 - index
        workspace = self.workspaces[group][index]
        workspace.acquire()
        return workspace

    def _save(self, request, state):
        if self.states is None or (
            self.async_state is not None and request.request_id not in self.async_state.owners
        ):
            request.hidden_state = state.clone()
        else:
            rid = request.request_id
            if rid not in self.state_slots:
                self.state_slots[rid] = self.free_state_slots.pop()
            request.hidden_state = self.states[self.state_slots[rid]]
            request.hidden_state.copy_(state)

    def _save_batch(self, requests, hidden, routing):
        if routing is None:
            for request, state in zip(requests, hidden):
                self._save(request, state)
            return
        routing.scatter(hidden)
        for request in requests:
            request.hidden_state = self.states[self.state_slots[request.request_id]]

    def _gather(self, requests, workspace, size):
        if workspace is None:
            return torch.stack([r.hidden_state for r in requests])
        hidden = workspace.hidden[:size]
        hidden.zero_()
        for row, request in enumerate(requests):
            hidden[row].copy_(request.hidden_state)
        return hidden

    def _core(self, hidden, ids, depths, positions, workspace, size):
        if workspace is None:
            return self.model.recurrent(
                hidden,
                ids,
                depths,
                positions,
                self.cache_manager,
                compute_gate=self.exit_config.mode in ("ouro", "ouro_delayed"),
            )
        batch = workspace.prepare(self.cache_manager, ids, depths, positions, size)
        return self.model.recurrent_prepared(
            hidden,
            batch,
            self.cache_manager,
            compute_gate=self.exit_config.mode in ("ouro", "ouro_delayed"),
        )

    def _prefill(self, batch):
        if self.cache_manager.layout == "shared":
            # Define chunk-invariant shared semantics: complete all loops of each
            # position before advancing that request. Parallelize across requests.
            for offset in range(max(i.token_count for i in batch.items)):
                active = [i for i in batch.items if offset < i.token_count]
                ids = [i.request.request_id for i in active]
                positions = [i.token_start + offset for i in active]
                tokens = [i.request.prompt_token_ids[p] for i, p in zip(active, positions)]
                hidden = self._prefill_tokens(ids, positions, tokens, [i.request for i in active])
                for row, item in enumerate(active):
                    self._save(item.request, hidden[row])
            return
        ids, positions, tokens, requests = [], [], [], []
        for item in batch.items:
            start, count, request = item.token_start, item.token_count, item.request
            ids.extend([request.request_id] * count)
            positions.extend(range(start, start + count))
            tokens.extend(request.prompt_token_ids[start : start + count])
            requests.extend([request] * count)
        hidden = self._prefill_tokens(ids, positions, tokens, requests)
        offset = 0
        for item in batch.items:
            offset += item.token_count
            self._save(item.request, hidden[offset - 1])

    def _capture_loopcd(self, requests, hidden, depths, positions, indices, *, prefill=False):
        if not self.execution_config.loopcd:
            return
        for row, (request, depth, position, index) in enumerate(
            zip(requests, depths, positions, indices)
        ):
            if not active(request.sampling_params):
                continue
            if prefill and row + 1 < len(requests) and requests[row + 1] is request:
                continue
            cd = request.sampling_params.loopcd
            loop = cd.prefill_loop if prefill else cd.reference_loop
            if depth + 1 == loop:
                self.loopcd_references[request.request_id] = Reference(
                    request, position, index, loop, hidden[row].clone()
                )
                self.loopcd_stats["captures"] += 1
        size = sum(
            ref.hidden.numel() * ref.hidden.element_size()
            for ref in self.loopcd_references.values()
        )
        self.loopcd_stats["reference_peak_bytes"] = max(
            self.loopcd_stats["reference_peak_bytes"], size
        )

    def _count_core(self, effective, submitted, *, prefill=False):
        if self.execution_config.loopcd:
            phase = "prefill" if prefill else "decode"
            self.loopcd_stats[phase + "_core_rows"] += effective
            self.loopcd_stats[phase + "_submitted_rows"] += submitted

    def _loopcd_readout(self, hidden, requests, positions, indices):
        if not self.execution_config.loopcd:
            return self.model.coda(hidden)
        references = []
        for request, position, index in zip(requests, positions, indices):
            ref = None
            if active(request.sampling_params):
                cd = request.sampling_params.loopcd
                loop = cd.prefill_loop if index == 0 else cd.reference_loop
                ref = self.loopcd_references.get(request.request_id)
                if ref is None:
                    raise RuntimeError("LoopCD reference missing at readout")
                ref.check(request, position, index, loop)
            references.append(ref)
        guided = hidden
        for row, ref in enumerate(references):
            if ref is None or requests[row].sampling_params.loopcd.implementation == "two_head":
                continue
            if guided is hidden:
                guided = hidden.clone()
            guided[row] = extrapolate(
                hidden[row], ref.hidden, requests[row].sampling_params.loopcd.strength
            )
        logits = self.model.coda(guided)
        self.loopcd_stats["head_calls"] += 1
        self.loopcd_stats["head_rows"] += logits.shape[0]
        weak_rows = [
            row
            for row, ref in enumerate(references)
            if ref is not None and requests[row].sampling_params.loopcd.implementation == "two_head"
        ]
        if weak_rows:
            weak = self.model.coda(torch.stack([references[row].hidden for row in weak_rows]))
            logits = logits.float()
            for row, value in zip(weak_rows, weak):
                logits[row] = extrapolate(
                    logits[row], value.float(), requests[row].sampling_params.loopcd.strength
                )
            self.loopcd_stats["head_calls"] += 1
            self.loopcd_stats["head_rows"] += len(weak_rows)
        self.loopcd_stats["guided_rows"] += sum(ref is not None for ref in references)
        return logits

    def _prefill_tokens(self, ids, positions, tokens, requests=()):
        cache = self.cache_manager
        if cache.layout == "last_exited" and cache.attention_capabilities.packed_prefill:
            # Prefill has genuinely ragged query sequences. Do not pad token rows
            # or reuse decode's per-query, model-max-width static page tables.
            bank = None
            if self.execution_config.prefill_uva:
                from vllm_rlt.worker.prefill_metadata import PrefillMetadataBank

                if self._prefill_banks is None:
                    self._prefill_banks = [PrefillMetadataBank(cache) for _ in range(3)]
                bank = self._prefill_banks[self._prefill_bank_index]
                self._prefill_bank_index = (self._prefill_bank_index + 1) % len(self._prefill_banks)
            tensor = (
                bank.prepare(ids, positions, tokens)
                if bank
                else torch.tensor(tokens, device=self.device, dtype=torch.long)
            )
            hidden = self.model.prelude(tensor)
            for depth in range(self.prefill_depth):
                metadata = (
                    bank.metadata(depth)
                    if bank
                    else cache._prepare_batch(
                        ids, [depth] * len(ids), positions, packed_prefill=True
                    )
                )
                hidden, _ = self.model.recurrent_prepared(
                    hidden, metadata, cache, compute_gate=False
                )
                self._count_core(len(ids), len(ids), prefill=True)
                self._capture_loopcd(
                    requests,
                    hidden,
                    [depth] * len(requests),
                    positions,
                    [0] * len(requests),
                    prefill=True,
                )
            if bank:
                bank.release()
            return hidden
        size = self._size(len(tokens))
        workspace = self._workspace("core")
        tensor = (
            workspace.tokens(tokens, size)
            if workspace
            else torch.tensor(tokens, device=self.device, dtype=torch.long)
        )
        hidden = self.model.prelude(tensor)
        for depth in range(self.prefill_depth):
            # Same workspace metadata can be refilled only once previous DMA is done.
            if workspace:
                workspace.acquire()
            hidden, _ = self._core(hidden, ids, [depth] * len(ids), positions, workspace, size)
            self._count_core(len(ids), size, prefill=True)
            self._capture_loopcd(
                requests,
                hidden,
                [depth] * len(requests),
                positions,
                [0] * len(requests),
                prefill=True,
            )
            if workspace:
                workspace.release()
        return hidden

    def prepare(self, batch):
        """Snapshot CPU routing/addresses while the preceding GPU work runs."""
        requests = [i.request for i in batch.items]
        depths = tuple(r.loops_done for r in requests)
        positions = tuple(r.position for r in requests)
        indices = tuple(r.num_scheduled_outputs for r in requests)
        tokens = ()
        if batch.stage == Stage.PRELUDE:
            tokens = tuple(r.input_token_tensor for r in requests)
            if any(t is None for t in tokens):
                raise RuntimeError("async prelude requires the preceding device sample")
        if self.async_state is not None:
            routing, kv = self.async_state.prepare(
                requests,
                depths,
                positions,
                self._size(len(requests)),
                recurrent=batch.stage == Stage.RECURRENT,
            )
            return PreparedExecution(batch, depths, positions, indices, tokens, kv, routing=routing)
        kv = None
        if batch.stage == Stage.RECURRENT and not self.workspaces:
            kv = self.cache_manager._prepare_batch(
                [r.request_id for r in requests], depths, positions
            )
        return PreparedExecution(batch, depths, positions, indices, tokens, kv)

    @torch.inference_mode()
    def _execute(self, batch, prepared=None):
        requests = [i.request for i in batch.items]
        if batch.stage == Stage.PREFILL:
            self._prefill(batch)
            return None
        size = self._size(len(requests))
        group = "core" if batch.stage == Stage.RECURRENT else "boundary"
        routing = prepared.routing if prepared is not None else None
        workspace = None if routing is not None else self._workspace(group)
        if batch.stage == Stage.PRELUDE:
            for request in requests:
                self.loopcd_references.pop(request.request_id, None)
            if prepared is not None:
                # Feed the sampled GPU IDs directly to embedding. No .item(),
                # .tolist(), or CPU token roundtrip on the dependency path.
                tokens = (
                    routing.gather(tokens=True)
                    if routing is not None
                    else torch.stack(prepared.token_tensors)
                )
                if tokens.is_cuda:
                    for tensor in prepared.token_tensors:
                        tensor.record_stream(torch.cuda.current_stream(self.device))
                if routing is None and size > len(requests):
                    tokens = torch.cat((tokens, tokens.new_zeros(size - len(requests))))
                for request in requests:
                    request.input_token_tensor = None
            else:
                ids = [r.input_token_id for r in requests]
                tokens = (
                    workspace.tokens(ids, size)
                    if workspace
                    else torch.tensor(ids, device=self.device, dtype=torch.long)
                )
            hidden = self.model.prelude(tokens)
            self._save_batch(requests, hidden, routing)
            result = None
        else:
            hidden = (
                routing.gather() if routing is not None else self._gather(requests, workspace, size)
            )
            if batch.stage == Stage.RECURRENT:
                if self.graphs is not None:
                    kv = prepared.kv if prepared is not None else None
                    if kv is None:
                        kv = self.cache_manager._prepare_batch(
                            [r.request_id for r in requests],
                            [r.loops_done for r in requests],
                            [r.position for r in requests],
                        )
                    hidden, logits = self.graphs.run(hidden[: len(requests)], kv)
                elif prepared is not None and prepared.kv is not None:
                    kv = prepared.kv
                    hidden, logits = self.model.recurrent_prepared(
                        hidden,
                        kv,
                        self.cache_manager,
                        compute_gate=self.exit_config.mode in ("ouro", "ouro_delayed"),
                    )
                else:
                    hidden, logits = self._core(
                        hidden,
                        [r.request_id for r in requests],
                        [r.loops_done for r in requests],
                        [r.position for r in requests],
                        workspace,
                        size,
                    )
                self._count_core(len(requests), len(hidden))
                self._capture_loopcd(
                    requests,
                    hidden,
                    prepared.depths if prepared else [r.loops_done for r in requests],
                    prepared.positions if prepared else [r.position for r in requests],
                    prepared.output_indices
                    if prepared
                    else [r.num_scheduled_outputs for r in requests],
                )
                self._save_batch(requests, hidden, routing)
                if self.lookahead_head is not None:
                    logits = self.lookahead_head(hidden).squeeze(-1)
                result = logits[: len(requests)].float().sigmoid() if logits is not None else None
            elif batch.stage == Stage.CODA:
                logits = self._loopcd_readout(
                    hidden,
                    requests,
                    prepared.positions if prepared else [r.position for r in requests],
                    prepared.output_indices
                    if prepared
                    else [r.num_scheduled_outputs for r in requests],
                )
                result = torch.stack(
                    [self._sample_tensor(row, r) for row, r in zip(logits, requests)]
                )
                if routing is not None:
                    routing.scatter(result, tokens=True)
            else:
                raise ValueError(f"unsupported execution stage {batch.stage}")
        if workspace:
            workspace.release()
        return result

    def execute(self, batch: SchedulerOutput):
        # Baseline explicitly synchronizes signals; delayed routing can be tested here.
        result = self._execute(batch)
        return result.cpu().tolist() if result is not None else None

    def _readback_slot(self, result):
        for slot in self.readback_slots:
            if slot.storage.dtype == result.dtype and not slot.leased:
                if slot.event is not None:
                    slot.event.synchronize()
                slot.leased = True
                return slot
        raise RuntimeError("readback buffers exhausted; retire completed submissions")

    def submit(self, batch: SchedulerOutput):
        prepared = self.prepare(batch)
        while self.submission_events and self.submission_events[0].query():
            self.submission_events.popleft()
        if len(self.submission_events) >= 3:
            self.submission_events.popleft().synchronize()
        stream = (
            self.core_stream
            if batch.stage in (Stage.PREFILL, Stage.RECURRENT)
            else self.boundary_stream
        )
        if prepared.routing is not None:
            if prepared.routing.imports:
                self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.copy_stream):
                kv = prepared.routing.transfer(prepared.kv)
            stream.wait_event(prepared.routing.ready_event)
            prepared = replace(prepared, kv=kv)
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            for item in batch.items:
                event = self.events.get(item.request.request_id)
                if stream is not None and event is not None:
                    stream.wait_event(event)
                hidden = item.request.hidden_state
                if stream is not None and hidden is not None:
                    hidden.record_stream(stream)
            try:
                result = self._execute(batch, prepared)
            finally:
                if prepared.routing is not None:
                    prepared.routing.record_done()
            device_values = result
            event = None
            slot = None
            if self.device.type == "cuda":
                if result is not None:
                    slot = self._readback_slot(result)
                    host = slot.storage[: result.numel()].view(result.shape)
                    host.copy_(result, non_blocking=True)
                    result = host
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.device))
                if slot is not None:
                    slot.event = event
                for item in batch.items:
                    self.events[item.request.request_id] = event
                self.submission_events.append(event)
            return Submission(
                batch,
                result,
                event,
                slot=slot,
                device_values=device_values,
                depths=prepared.depths,
                output_indices=prepared.output_indices,
            )

    def finalize_many(self, requests):
        if not requests:
            return
        if self.async_state is None or self.cache_manager.layout == "shared":
            for request in requests:
                self.finalize(request)
            return
        from vllm_rlt.kernels.routing import finalize_kernel

        cache = self.cache_manager
        copies = []
        for request in requests:
            depth = request.loops_done - 1
            if not cache.token_written(request.request_id, request.position, depth):
                raise RuntimeError("cannot finalize before every layer has written KV")
            if depth + 1 < cache.max_loops:
                copies.append(request)
        if not copies:
            return
        bank, _ = self.async_state.prepare(
            copies,
            [r.loops_done - 1 for r in copies],
            [r.position for r in copies],
            len(copies),
            finalize=True,
        )
        with torch.cuda.stream(self.boundary_stream):
            for request in copies:
                event = self.events.get(request.request_id)
                if event is not None:
                    self.boundary_stream.wait_event(event)
            bank.descriptor = (
                bank.host
                if self.async_state.use_uva
                else bank.host[: bank.count].to(self.device, non_blocking=True)
            )
            channels = cache.num_kv_heads * cache.head_dim
            finalize_kernel[
                (len(copies), cache.max_loops, math.ceil(cache.num_layers * channels / 256))
            ](
                bank.descriptor,
                self.async_state.tables,
                cache.key_cache,
                cache.value_cache,
                self.async_state.width,
                cache.max_loops,
                cache.block_size,
                cache.num_layers,
                channels,
                *cache.key_cache.stride()[:3],
                256,
            )
            bank.record_done()
            for request in copies:
                self.events[request.request_id] = bank.done
                cache.mark_finalized(request.request_id, request.position, request.loops_done - 1)

    def finalize(self, request):
        # The final core event must precede copies and coda on the boundary stream.
        stream = self.boundary_stream
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            event = self.events.get(request.request_id)
            if stream is not None and event is not None:
                stream.wait_event(event)
            self.cache_manager.finalize_token(
                request.request_id, request.position, request.loops_done - 1
            )
            if stream is not None:
                event = torch.cuda.Event()
                event.record(stream)
                self.events[request.request_id] = event

    def release(self, request_id):
        # The latest submission or KV-finalization event follows earlier work
        # through stream dependencies. Wait before recycling this request's slot.
        event = self.events.pop(request_id, None)
        if event is not None:
            event.synchronize()
        self.loopcd_references.pop(request_id, None)
        if self.async_state is not None:
            self.async_state.release(request_id)
            return
        slot = self.state_slots.pop(request_id, None)
        if slot is not None:
            self.free_state_slots.append(slot)

    def synchronize(self):
        for stream in (self.core_stream, self.boundary_stream, self.copy_stream):
            if stream is not None:
                stream.synchronize()

    def _sample_tensor(self, logits: torch.Tensor, request: Request):
        # Thin delegate kept for the async path and for tests that monkeypatch
        # this method; the algorithm lives in Sampler.
        token, generator = self.sampler.sample(logits, request.sampling_params, request.generator)
        request.generator = generator
        return token
