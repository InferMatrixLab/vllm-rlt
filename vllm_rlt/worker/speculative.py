"""Submitted block self-speculation with depth-specific KV/activation reuse.

Temporary tensors belong to a submission until collection. Requests expose only committed
outputs; Scheduler applies returned tokens and KV commits; Engine coordinates cleanup.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import Operation
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
                    speculative=self.runner._collect_round(self),
                    completion=self.event,
                )
            self.logits = None
            self.drafted.clear()
            self.proposals.clear()
        return self.cached


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
                if request.sampling_params.logprobs is not None:
                    results[-1].log_probs.extend(
                        Sampler.selected_logprob(row, token, request.sampling_params).detach().cpu()
                        for row, token in zip(rows, tokens)
                    )
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
            if request.sampling_params.logprobs is not None:
                results[-1].log_probs.extend(
                    Sampler.selected_logprob(row, token, request.sampling_params).detach().cpu()
                    for row, token in zip(rows, tokens)
                )
        return results
