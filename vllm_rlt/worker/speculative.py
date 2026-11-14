"""Submitted block self-speculation with depth-specific KV/activation reuse.

Temporary tensors belong to a submission until collection. Requests expose only committed
outputs; Scheduler applies returned tokens and KV commits; Engine coordinates cleanup.

Boundary-KV models (attention in PRELUDE and CODA) keep one boundary history at
depth 0. PRELUDE KV depends only on the input tokens, so a candidate's prelude
is the target's whenever the candidate is accepted. CODA KV depends on the
state it reads: a draft CODA stores draft-depth KV at its own position, which is
never below the committed frontier, and the target CODA of that position stores
over it before any target row attends it. Target rows therefore read target
history only; a rejected suffix is cut by ``truncate_suffix``.
"""

from contextlib import nullcontext
from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class DraftStep:
    """Prepared inputs of one draft offset in a block round.

    ``requests`` holds batch request indices; row ``r`` of this step's hidden
    tensor belongs to ``requests[r]``. ``coda_rows`` holds indices into those
    rows, in order, for the requests that draft another token; the draft CODA
    reads exactly these hidden rows and ``coda`` describes exactly these rows.
    ``layouts`` are the step's recurrent depths; ``layouts[0]`` is depth 0,
    which boundary-KV models also use for the PRELUDE.
    """

    requests: tuple[int, ...]
    coda_rows: tuple[int, ...]
    layouts: tuple
    coda: object | None = None


@dataclass(frozen=True)
class PreparedRound:
    """Bound inputs of one block round, borrowed from the runner's input buffers.

    Every tensor views the single upload per dtype made by ``_prepare_inputs``
    and stays valid until the next preparation resets those buffers, i.e. until
    the round's submission has been collected. Verification and target CODA rows
    are all round rows grouped by request, in position order.
    """

    tokens: torch.Tensor
    steps: tuple[DraftStep, ...]
    verification: tuple
    target_coda: object | None = None


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
        self.boundary_kv = getattr(model, "requires_boundary_kv", False)
        # Windowed attention needs one query row per token.
        self.row_queries = getattr(model, "requires_sliding_window", False)
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
        return (
            packed
            and not self.row_queries
            and capabilities.packed_prefill
            and capabilities.packed_matches_decode
        )

    def _stage(self, values, dtype):
        return self.input_buffers[dtype].append(values)

    def _view(self, source, rows, depths, *, packed=False):
        """Stage one selection of ``source``: the same rows at each depth index.

        ``rows=None`` selects every row in source order without re-planning.
        Returns ``None`` for an empty selection.
        """
        if rows is None and not packed:
            layouts = [source[d] for d in depths]
        else:
            rows = range(len(source[0].rows)) if rows is None else rows
            layouts = [self.metadata.select_rows(source[d], rows, packed=packed) for d in depths]
        if layouts[0] is None:
            return None
        return stage_metadata(layouts, self._stage)

    def _boundary(self, source, rows, depth):
        """Stage the depth-0 boundary view of ``rows``; ``None`` when empty."""
        view = self._view(source, rows, (depth,))
        return None if view is None else view[0]

    def _bind(self, metadata, uploaded):
        return None if metadata is None else self.metadata.bind(metadata, uploaded)

    def _prepare_inputs(self, batch):
        """Plan every round input from one layout source, then upload once per dtype.

        The source covers all round rows (request order, then position) at
        every target depth. Each operation stages a row/depth selection of it;
        nothing is selected, uploaded or read back inside the execution loop.
        """
        for buffer in self.input_buffers.values():
            buffer.reset()
        requests = batch.requests
        tokens = self._stage([span.request.input_token_id for span in requests], torch.int64)
        ids, positions, starts = [], [], []
        for span in requests:
            starts.append(len(ids))
            ids.extend([span.request.request_id] * span.token_count)
            positions.extend(range(span.token_start, span.token_start + span.token_count))
        draft, target = self.config.draft_loops, self.config.target_loops
        source = self.metadata.build(
            ids, [[depth] * len(ids) for depth in range(target)], positions
        )
        steps = []
        for offset in range(max(span.token_count for span in requests)):
            active = tuple(i for i, span in enumerate(requests) if offset < span.token_count)
            layouts = self._view(source, [starts[i] + offset for i in active], range(draft))
            coda_rows = tuple(
                row for row, i in enumerate(active) if offset + 1 < requests[i].token_count
            )
            coda = None
            if self.boundary_kv and coda_rows:
                # Depth 0 is the boundary history. When every row drafts, the
                # CODA selection is the step's own depth-0 view.
                coda = (
                    layouts[0]
                    if len(coda_rows) == len(active)
                    else self._boundary(
                        source, [starts[active[row]] + offset for row in coda_rows], 0
                    )
                )
            steps.append(DraftStep(active, coda_rows, layouts, coda))
        verification = self._view(source, None, range(draft, target), packed=self._packed(True))
        target_coda = self._boundary(source, None, 0) if self.boundary_kv else None
        uploaded = {dtype: buffer.upload() for dtype, buffer in self.input_buffers.items()}
        return PreparedRound(
            tokens=uploaded[tokens.dtype].as_strided(
                tokens.shape, tokens.stride(), tokens.storage_offset()
            ),
            steps=tuple(
                replace(
                    step,
                    layouts=tuple(self._bind(kv, uploaded) for kv in step.layouts),
                    coda=self._bind(step.coda, uploaded),
                )
                for step in steps
            ),
            verification=tuple(self._bind(kv, uploaded) for kv in verification),
            target_coda=self._bind(target_coda, uploaded),
        )

    def _prelude(self, tokens, boundary=None):
        return self.runtime.forward(
            ForwardBatch(Operation.PRELUDE, tokens=tokens, prelude_coda_kv_metadata=boundary)
        ).hidden

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

    def _coda(self, hidden, boundary=None):
        return self.runtime.forward(
            self.runtime.prepare(
                ForwardBatch(
                    Operation.CODA,
                    hidden=hidden,
                    prelude_coda_kv_metadata=boundary,
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
                else:
                    submission = self._submit_round(batch, self._prepare_inputs(batch))
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
        new = [i for i, t in enumerate(tokens) if t.hidden is None]
        input_view = self.input_buffers[torch.int64].append([tokens[i].token for i in new])

        positions = [t.position for t in tokens]
        draft = self.config.draft_loops
        # Depth sets per row: this step's loops, then (boundary KV) depth 0.
        depth_sets = [[t.depth + offset for t in tokens] for offset in range(draft)]
        if self.boundary_kv:
            depth_sets.append([0] * len(tokens))
        source = self.metadata.build(ids, depth_sets, positions, packed=False)
        layouts = self._view(source, None, range(draft))
        # Rows that read a draft or target distribution after this step's loops.
        endpoints = [
            i for i, t in enumerate(tokens) if t.depth + draft in (draft, self.config.target_loops)
        ]
        prelude_kv = coda_kv = None
        if self.boundary_kv:
            # Boundary history lives at depth 0. A request's rows stay ordered by
            # position and every row stores before any row attends, so a draft
            # row reads the target KV its predecessor stores in the same call.
            # PRELUDE runs for new rows only; CODA reads the endpoint rows.
            prelude_kv = self._boundary(source, new, draft)
            coda_kv = self._boundary(source, endpoints, draft)
        uploaded = {dtype: buffer.upload() for dtype, buffer in self.input_buffers.items()}
        if new:
            inputs = uploaded[input_view.dtype].as_strided(
                input_view.shape, input_view.stride(), input_view.storage_offset()
            )
            hidden = self._prelude(inputs, self._bind(prelude_kv, uploaded))
            for i, row in zip(new, hidden):
                tokens[i].hidden = row
        hidden = self._recurrent(
            torch.stack([t.hidden for t in tokens]),
            tuple(self.metadata.bind(kv, uploaded) for kv in layouts),
        )
        for token, row in zip(tokens, hidden):
            token.hidden = row
            token.depth += self.config.draft_loops
        # Synchronous execution reads decisions in collect(), before the next step.
        choices = (
            self._coda(hidden[endpoints], self._bind(coda_kv, uploaded)).argmax(-1)
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

    def _submit_round(self, batch, prepared):
        scheduled_requests = batch.requests
        proposals = [[] for _ in scheduled_requests]
        states = [[] for _ in scheduled_requests]
        # Draft IDs stay on the device: each offset's IDs are the next offset's
        # prelude input, and all of them are read back once after verification.
        drafted = []
        inputs = prepared.tokens
        # Each step consumes the previous step's GPU-resident sampled IDs.
        for step in prepared.steps:
            hidden = self._prelude(inputs, step.layouts[0] if self.boundary_kv else None)
            hidden = self._recurrent(hidden, step.layouts)
            for row, i in enumerate(step.requests):
                states[i].append(hidden[row])
            if not step.coda_rows:
                break
            logits = self._coda(torch.stack([hidden[row] for row in step.coda_rows]), step.coda)
            inputs = logits.argmax(-1)
            drafting = [step.requests[row] for row in step.coda_rows]
            for row, i in enumerate(drafting):
                request = scheduled_requests[i].request
                if request.sampling_params.temperature != 0:
                    q = Sampler.probabilities(logits[row], request.sampling_params)
                    proposals[i].append(q)
                    # Per-request RNG consumption order is unchanged; only the read moves.
                    inputs[row] = Sampler.draw(q, self._acquire_generator(request, logits[row]))
            drafted.append((drafting, inputs))
        # Verification rows are contiguous by request, matching prepared metadata.
        hidden = torch.stack([h for request_states in states for h in request_states])
        hidden = self._recurrent(hidden, prepared.verification)
        logits = self._coda(hidden, prepared.target_coda)
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        self.stats.rounds += len(scheduled_requests)
        self.stats.drafted_tokens += sum(len(rows) for rows, _ in drafted)
        self.stats.verified_rows += sum(span.token_count for span in scheduled_requests)
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
