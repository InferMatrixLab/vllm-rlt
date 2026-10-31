"""Submitted block self-speculation with depth-specific KV/activation reuse.

Temporary tensors belong to a submission until collection. Requests expose only committed
outputs; Scheduler applies returned tokens and KV commits; Engine coordinates cleanup.
"""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import lru_cache

import torch

from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import Operation, Request
from vllm_rlt.worker.buffers import stage_metadata
from vllm_rlt.worker.cuda_graph import CodaGraphs, RecurrentGraphs
from vllm_rlt.worker.executor import ForwardBatch, ModelRuntime
from vllm_rlt.worker.output import ModelRunnerOutput, SpeculativeResult
from vllm_rlt.worker.sampler import RngRegistry, Sampler
from vllm_rlt.worker.submission_queue import SubmissionQueue
from vllm_rlt.worker.tensors import StagedInputTensor


@dataclass
class SpeculativeStats:
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    committed_tokens: int = 0
    verified_rows: int = 0


def greedy_accept(candidates, targets):
    """Return (tokens, accepted) for one request from host candidate and target IDs.

    targets[j] is the target argmax of verification row j (len(candidates) + 1 rows).
    Accepted candidates equal their targets, so the committed tokens are the target
    prefix through the first mismatch: the correction, or the bonus if all match.
    """
    accepted = 0
    for candidate, target in zip(candidates, targets):
        if candidate != target:
            break
        accepted += 1
    return list(targets[: accepted + 1]), accepted


def rejection_sample(candidate, target, proposal, generator):
    """Return (token, accepted) for normalized, actually sampled p and q.

    Comparing u*q to p avoids division by tiny q. A zero residual after a real
    rejection is a numerical error, never a reason to silently sample from p.
    """
    mass = proposal[candidate]
    if not bool(mass > 0):
        raise ValueError("candidate has zero proposal probability")
    uniform = torch.rand((), device=target.device, generator=generator)
    if bool(uniform * mass < target[candidate]):
        return candidate, True
    residual = (target - proposal).clamp_min(0)
    total = residual.sum()
    if not bool(torch.isfinite(total) & (total > 0)):
        raise RuntimeError("invalid speculative residual probability mass")
    return int(Sampler.draw(residual / total, generator).item()), False


@lru_cache(maxsize=64)
def loop_waves(token_counts, draft_loops, target_loops):
    """Return the mixed-depth rows of one loop-level round, one tuple per wave.

    Offset o of a request enters at wave o * draft_loops, after its parent's draft
    head chose its input, and runs one depth per wave up to target_loops. Rows are
    (request, offset, depth), ordered by request and offset. ``drafts`` lists the
    rows whose state after the wave feeds the draft head for the next offset.
    The schedule depends only on token counts, never on token values.
    """
    waves = []
    for wave in range(max((count - 1) * draft_loops + target_loops for count in token_counts)):
        rows, drafts = [], []
        for i, count in enumerate(token_counts):
            first = max(0, -(-(wave - target_loops + 1) // draft_loops))
            for offset in range(first, min(count, wave // draft_loops + 1)):
                depth = wave - offset * draft_loops
                if depth == draft_loops - 1 and offset + 1 < count:
                    drafts.append(len(rows))
                rows.append((i, offset, depth))
        waves.append((tuple(rows), tuple(drafts)))
    return tuple(waves)


@dataclass
class SpeculativeSubmission:
    runner: "SpeculativeRunner"
    batch: SchedulerOutput
    logits: torch.Tensor | None
    drafted: list
    proposals: list
    event: object = None
    cached: ModelRunnerOutput | None = None
    pipeline: "PipelineStep | None" = None

    @property
    def has_result(self):
        return True

    def ready(self):
        return self.event is None or self.event.query()

    def collect(self) -> ModelRunnerOutput:
        if self.cached is None:
            if self.event is not None:
                self.event.synchronize()
            stream = self.runner.compute_stream
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                self.cached = ModelRunnerOutput(
                    speculative=(
                        self.runner._collect_pipeline(self)
                        if self.pipeline is not None
                        else self.runner._collect_round(self)
                    ),
                    completion=self.event,
                )
            self.pipeline = None
            self.logits = None
            self.drafted.clear()
            self.proposals.clear()
        return self.cached


@dataclass
class PipelineToken:
    """One input token retained across submissions, with its next loop depth."""

    position: int
    token: int
    depth: int = 0
    hidden: torch.Tensor | None = None
    candidate: int | None = None


@dataclass
class PipelineState:
    request: Request
    position: int
    tokens: list[PipelineToken]
    pending: bool = False

    def to(self, device):
        """Own hidden copies for suspension; never borrow graph/runner buffers."""
        return replace(
            self,
            tokens=[
                replace(
                    token, hidden=None if token.hidden is None else token.hidden.to(device).clone()
                )
                for token in self.tokens
            ],
        )


@dataclass
class PipelineStep:
    states: list[PipelineState]
    tokens: list[PipelineToken]
    endpoints: list[int]
    decisions: torch.Tensor


class SpeculativeRunner:
    def __init__(
        self,
        model,
        cache,
        config,
        execution,
        rng: RngRegistry | None = None,
        *,
        runtime=None,
        compute_stream=None,
    ):
        self.model, self.cache, self.config = model, cache, config
        self.metadata = cache.metadata_builder
        self.runtime = runtime if runtime is not None else ModelRuntime(model, cache)
        self.pipeline_states: dict[str, PipelineState] = {}
        self.device = next(model.parameters()).device
        self.compute_stream = compute_stream
        self.submissions = SubmissionQueue()
        self.input_buffers = {
            dtype: StagedInputTensor(dtype, self.device, use_uva=execution.use_uva)
            for dtype in (torch.int64, torch.int32)
        }
        # The engine injects one registry per engine so both sampling paths share
        # each request's RNG slot.
        self.rng = rng if rng is not None else RngRegistry()
        self.stats = SpeculativeStats()
        self.graphs = (
            RecurrentGraphs(model, cache, execution, False) if execution.cuda_graphs else None
        )
        self.coda_graphs = CodaGraphs(model, execution) if execution.cuda_graphs else None

    def _packed(self, packed):
        capabilities = self.cache.attention_capabilities
        return packed and capabilities.packed_prefill and capabilities.packed_matches_decode

    def _prepare_inputs(self, batch):
        """Plan all host-known inputs, then publish one snapshot per dtype."""
        for buffer in self.input_buffers.values():
            buffer.reset()
        requests = batch.requests
        tokens = self.input_buffers[torch.int64].append(
            [span.request.input_token_id for span in requests]
        )

        def metadata(ids, positions, depths, *, packed=False):
            layouts = self.metadata.build(
                ids,
                [[depth] * len(ids) for depth in depths],
                positions,
                packed=self._packed(packed),
            )
            return stage_metadata(
                layouts, lambda values, dtype: self.input_buffers[dtype].append(values)
            )

        steps = []
        for offset in range(max(span.token_count for span in requests)):
            active = [i for i, span in enumerate(requests) if offset < span.token_count]
            ids = [requests[i].request.request_id for i in active]
            positions = [requests[i].token_start + offset for i in active]
            steps.append((active, metadata(ids, positions, range(self.config.draft_loops))))
        ids, positions = [], []
        for span in requests:
            ids.extend([span.request.request_id] * span.token_count)
            positions.extend(range(span.token_start, span.token_start + span.token_count))
        verification = metadata(
            ids, positions, range(self.config.draft_loops, self.config.target_loops), packed=True
        )
        uploaded = {dtype: buffer.upload() for dtype, buffer in self.input_buffers.items()}
        tokens = uploaded[tokens.dtype].as_strided(
            tokens.shape, tokens.stride(), tokens.storage_offset()
        )
        steps = [
            (active, tuple(self.metadata.bind(kv, uploaded) for kv in layouts))
            for active, layouts in steps
        ]
        verification = tuple(self.metadata.bind(kv, uploaded) for kv in verification)
        return tokens, steps, verification

    def _recurrent(self, hidden, metadata):
        forward = ForwardBatch(
            Operation.RECURRENT, hidden=hidden, metadata=metadata, graphs=self.graphs
        )
        return self.runtime.forward(self.runtime.prepare(forward)).hidden

    def _acquire_generator(self, request, logits):
        """Resolve this request's generator from the shared engine registry.

        The generator has to live on the distribution's device, so it follows the
        logits row rather than the device the model was loaded on.
        """
        return self.rng.acquire(request.request_id, request.sampling_params, logits.device)

    def _coda(self, hidden):
        return self.runtime.forward(
            self.runtime.prepare(
                ForwardBatch(
                    Operation.CODA,
                    hidden=hidden,
                    graphs=self.coda_graphs,
                )
            )
        ).logits

    @torch.inference_mode()
    def submit(self, batch):
        self.submissions.begin()
        try:
            if self.compute_stream is not None:
                self.compute_stream.wait_stream(torch.cuda.current_stream(self.device))
            with (
                torch.cuda.stream(self.compute_stream)
                if self.compute_stream is not None
                else nullcontext()
            ):
                if self.config.mode == "pipeline":
                    submission = self._submit_pipeline(batch)
                elif self.config.loop_level_pipelining:
                    submission = self._submit_loop_round(batch, *self._prepare_loop_inputs(batch))
                else:
                    inputs, steps, verification = self._prepare_inputs(batch)
                    submission = self._submit_round(batch, inputs, steps, verification)
                self.submissions.complete(submission.event)
                return submission
        except Exception:
            # Failed submissions may not have recorded a completion event.
            if self.device.type == "cuda":
                (self.compute_stream or torch.cuda.current_stream(self.device)).synchronize()
            self.submissions.drain()
            raise

    def release(self, request):
        """Discard only this request generation after its submission has completed."""
        state = self.pipeline_states.get(request.request_id)
        if state is not None and state.request is request:
            self.submissions.drain()
            del self.pipeline_states[request.request_id]

    def snapshot_pipeline(self, request, device):
        state = self.pipeline_states.get(request.request_id)
        if state is None or state.request is not request:
            return None
        if state.pending:
            raise RuntimeError("cannot suspend an uncollected pipeline submission")
        self.submissions.drain()
        return state.to(device)

    def restore_pipeline(self, request, state):
        if state is not None:
            if state.request is not request or request.request_id in self.pipeline_states:
                raise RuntimeError("pipeline snapshot belongs to a different request generation")
            self.pipeline_states[request.request_id] = state.to(self.device)

    def _submit_pipeline(self, batch):
        # Scheduler still owns admission, capacity reservation and committed output.
        # Retained tokens are private execution state, bounded by the reserved span.
        states, tokens, ids = [], [], []
        for span in batch.requests:
            request = span.request
            if request.sampling_params.temperature != 0:
                raise ValueError("pipeline speculation requires greedy sampling")
            state = self.pipeline_states.get(request.request_id)
            if state is None or state.request is not request or state.position != request.position:
                state = PipelineState(
                    request,
                    request.position,
                    [PipelineToken(request.position, request.input_token_id)],
                )
                self.pipeline_states[request.request_id] = state
            if state.pending:
                raise RuntimeError("collect the previous pipeline submission before resubmission")
            state.pending = True
            # A smaller scheduler budget may drop speculative descendants safely.
            state.tokens = [
                t for t in state.tokens if t.position < span.token_start + span.token_count
            ]
            states.append(state)
            tokens.extend(state.tokens)
            ids.extend([request.request_id] * len(state.tokens))
        for buffer in self.input_buffers.values():
            buffer.reset()
        new = [t for t in tokens if t.hidden is None]
        input_view = self.input_buffers[torch.int64].append([t.token for t in new])
        layouts = self.metadata.build(
            ids,
            [[t.depth + offset for t in tokens] for offset in range(self.config.draft_loops)],
            [t.position for t in tokens],
            packed=False,
        )
        layouts = stage_metadata(
            layouts, lambda values, dtype: self.input_buffers[dtype].append(values)
        )
        uploaded = {dtype: buffer.upload() for dtype, buffer in self.input_buffers.items()}
        if new:
            inputs = uploaded[input_view.dtype].as_strided(
                input_view.shape, input_view.stride(), input_view.storage_offset()
            )
            hidden = self.runtime.forward(ForwardBatch(Operation.PRELUDE, tokens=inputs)).hidden
            for token, row in zip(new, hidden):
                token.hidden = row
        hidden = self._recurrent(
            torch.stack([t.hidden for t in tokens]),
            tuple(self.metadata.bind(kv, uploaded) for kv in layouts),
        )
        endpoints = []
        for i, (token, row) in enumerate(zip(tokens, hidden)):
            token.hidden = row
            token.depth += self.config.draft_loops
            if token.depth in (self.config.draft_loops, self.config.target_loops):
                endpoints.append(i)
        # Synchronous execution reads decisions in collect(), before the next step.
        choices = (
            self._coda(hidden[endpoints]).argmax(-1)
            if endpoints
            else hidden.new_empty(0, dtype=torch.long)
        )
        decisions = choices
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        self.stats.rounds += len(states)
        return SpeculativeSubmission(
            self,
            batch,
            None,
            [],
            [],
            event,
            pipeline=PipelineStep(states, tokens, endpoints, decisions),
        )

    def _collect_pipeline(self, submission):
        step = submission.pipeline
        predictions = {
            id(step.tokens[i]): value for i, value in zip(step.endpoints, step.decisions.tolist())
        }
        results = []
        for span, state in zip(submission.batch.requests, step.states):
            if self.pipeline_states.get(span.request.request_id) is not state:
                results.append(SpeculativeResult([], 0))
                continue
            state.pending = False
            following, final = [], None
            limit = (
                len(state.request.prompt_token_ids) + state.request.sampling_params.max_tokens - 1
            )
            for token in state.tokens:
                if token.depth == self.config.target_loops:
                    final = token
                else:
                    following.append(token)
                    if token.depth == self.config.draft_loops:
                        token.candidate = predictions[id(token)]
                        self.stats.drafted_tokens += 1
                        if token.position + 1 < min(limit, span.token_start + span.token_count):
                            following.append(PipelineToken(token.position + 1, token.candidate))
            emitted, accepted = [], 0
            if final is not None:
                prediction = predictions[id(final)]
                emitted = [prediction]
                accepted = int(final.candidate == prediction)
                state.position += 1
                self.stats.verified_rows += 1
                if not accepted:
                    following = []
                if not following and state.position < limit:
                    following = [PipelineToken(state.position, prediction)]
            state.tokens = sorted(following, key=lambda t: t.position)
            frontier = max(
                [state.position] + [t.position + 1 for t in state.tokens if t.hidden is not None]
            )
            results.append(SpeculativeResult(emitted, accepted, kv_frontier=frontier))
        return results

    def _submit_round(self, batch, inputs, steps, verification):
        scheduled_requests = batch.requests
        proposals = [[] for _ in scheduled_requests]
        states = [[] for _ in scheduled_requests]
        # Draft IDs stay on the device: each offset's IDs are the next offset's
        # prelude input, and all of them are read back once after verification.
        drafted = []
        # Each step consumes the previous step's GPU-resident sampled IDs.
        for offset, (active, metadata) in enumerate(steps):
            hidden = self.runtime.forward(ForwardBatch(Operation.PRELUDE, tokens=inputs)).hidden
            hidden = self._recurrent(hidden, metadata)
            drafting = []
            for row, i in enumerate(active):
                states[i].append(hidden[row])
                if offset + 1 < scheduled_requests[i].token_count:
                    drafting.append((row, i))
            if not drafting:
                break
            logits = self._coda(torch.stack([hidden[row] for row, _ in drafting]))
            inputs = logits.argmax(-1)
            for row, (_, i) in enumerate(drafting):
                request = scheduled_requests[i].request
                if request.sampling_params.temperature != 0:
                    q = Sampler.probabilities(logits[row], request.sampling_params)
                    proposals[i].append(q)
                    # Per-request RNG consumption order is unchanged; only the read moves.
                    inputs[row] = Sampler.draw(q, self._acquire_generator(request, logits[row]))
            drafted.append(([i for _, i in drafting], inputs))
        # Verification rows are contiguous by request, matching prepared metadata.
        hidden = torch.stack([h for request_states in states for h in request_states])
        hidden = self._recurrent(hidden, verification)
        logits = self._coda(hidden)
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        self.stats.rounds += len(scheduled_requests)
        self.stats.drafted_tokens += sum(len(rows) for rows, _ in drafted)
        self.stats.verified_rows += sum(span.token_count for span in scheduled_requests)
        return SpeculativeSubmission(self, batch, logits, drafted, proposals, event)

    def _prepare_loop_inputs(self, batch):
        """Plan every wave of a loop-level round and publish one snapshot per dtype.

        Branch states live in one buffer whose rows follow verification order:
        contiguous by request, then by offset. Each wave gathers its rows, runs
        one mixed-depth recurrent call and scatters the states back.
        """
        for buffer in self.input_buffers.values():
            buffer.reset()
        requests = batch.requests
        counts = tuple(span.token_count for span in requests)
        starts = [0]
        for count in counts:
            starts.append(starts[-1] + count)
        int64 = self.input_buffers[torch.int64]
        tokens = int64.append([span.request.input_token_id for span in requests])
        firsts = int64.append(starts[:-1])
        planned = []
        for rows, drafts in loop_waves(counts, self.config.draft_loops, self.config.target_loops):
            layouts = self.metadata.build(
                [requests[i].request.request_id for i, _, _ in rows],
                [[depth for _, _, depth in rows]],
                [requests[i].token_start + offset for i, offset, _ in rows],
            )
            metadata = stage_metadata(
                layouts, lambda values, dtype: self.input_buffers[dtype].append(values)
            )
            flat = [starts[i] + offset for i, offset, _ in rows]
            draft = None
            if drafts:
                draft = (
                    [rows[row][0] for row in drafts],
                    int64.append(drafts),
                    int64.append([flat[row] + 1 for row in drafts]),
                )
            planned.append((int64.append(flat), metadata, draft))
        uploaded = {dtype: buffer.upload() for dtype, buffer in self.input_buffers.items()}

        def bind(tensor):
            return uploaded[tensor.dtype].as_strided(
                tensor.shape, tensor.stride(), tensor.storage_offset()
            )

        waves = [
            (
                bind(flat),
                tuple(self.metadata.bind(kv, uploaded) for kv in metadata),
                None if draft is None else (draft[0], bind(draft[1]), bind(draft[2])),
            )
            for flat, metadata, draft in planned
        ]
        return bind(tokens), bind(firsts), starts[-1], waves

    def _submit_loop_round(self, batch, tokens, firsts, total_rows, waves):
        """Advance all positions one depth per wave; children start at draft depth.

        A child starts draft_loops waves after its parent, so the parent's KV at
        each depth is written before the child attends to it. Only draft-ready rows
        run the head inside the loop; target logits come from one head call after
        the last wave. Draft IDs stay on the device, as in the block round.
        """
        scheduled_requests = batch.requests
        proposals = [[] for _ in scheduled_requests]
        drafted = []
        hidden = self.runtime.forward(ForwardBatch(Operation.PRELUDE, tokens=tokens)).hidden
        states = hidden.new_empty((total_rows, hidden.shape[-1]))
        states.index_copy_(0, firsts, hidden)
        for flat, metadata, draft in waves:
            hidden = self._recurrent(states.index_select(0, flat), metadata)
            states.index_copy_(0, flat, hidden)
            if draft is None:
                continue
            owners, rows, children = draft
            logits = self._coda(hidden.index_select(0, rows))
            inputs = logits.argmax(-1)
            for row, i in enumerate(owners):
                request = scheduled_requests[i].request
                if request.sampling_params.temperature != 0:
                    q = Sampler.probabilities(logits[row], request.sampling_params)
                    proposals[i].append(q)
                    inputs[row] = Sampler.draw(q, self._acquire_generator(request, logits[row]))
            drafted.append((owners, inputs))
            child = self.runtime.forward(ForwardBatch(Operation.PRELUDE, tokens=inputs)).hidden
            states.index_copy_(0, children, child)
        logits = self._coda(states)
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        self.stats.rounds += len(scheduled_requests)
        self.stats.drafted_tokens += sum(len(rows) for rows, _ in drafted)
        self.stats.verified_rows += total_rows
        return SpeculativeSubmission(self, batch, logits, drafted, proposals, event)

    @torch.inference_mode()
    def _collect_round(self, submission):
        scheduled_requests = submission.batch.requests
        logits, drafted, proposals = submission.logits, submission.drafted, submission.proposals
        # One device-to-host read returns every draft ID and, for greedy requests,
        # every verification target argmax.
        greedy = any(
            scheduled_request.request.sampling_params.temperature == 0
            for scheduled_request in scheduled_requests
        )
        pending = [draft for _, draft in drafted] + ([logits.argmax(-1)] if greedy else [])
        values = torch.cat(pending).tolist() if pending else []
        candidates = [[] for _ in scheduled_requests]
        start = 0
        for rows, _ in drafted:
            for i, candidate in zip(rows, values[start : start + len(rows)]):
                candidates[i].append(candidate)
            start += len(rows)
        targets = values[start:]
        results, start = [], 0
        for i, scheduled_request in enumerate(scheduled_requests):
            rows = logits[start : start + scheduled_request.token_count]
            request = scheduled_request.request
            if request.sampling_params.temperature == 0:
                tokens, accepted = greedy_accept(
                    candidates[i], targets[start : start + scheduled_request.token_count]
                )
                start += scheduled_request.token_count
                results.append(SpeculativeResult(tokens, accepted))
                continue
            start += scheduled_request.token_count
            # Sampling keeps sequential rejection; RNG consumption order is unchanged.
            tokens, accepted = [], 0
            for offset, candidate in enumerate(candidates[i]):
                p = Sampler.probabilities(rows[offset], request.sampling_params)
                token, accept = rejection_sample(
                    candidate,
                    p,
                    proposals[i][offset],
                    self._acquire_generator(request, rows[offset]),
                )
                tokens.append(token)
                if not accept:
                    break
                accepted += 1
            else:
                token = Sampler.draw(
                    Sampler.probabilities(rows[-1], request.sampling_params),
                    self._acquire_generator(request, rows[-1]),
                )
                tokens.append(int(token.item()))
            results.append(SpeculativeResult(tokens, accepted))
        return results
