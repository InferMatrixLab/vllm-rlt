"""Synchronous block self-speculation with depth-specific KV/activation reuse.

Temporary tensors belong to one execute call. Requests expose only committed
outputs; the engine applies returned tokens and owns KV commit/truncation.
"""

from dataclasses import dataclass

import torch

from vllm_rlt.worker.cuda_graph import CodaGraphs, RecurrentGraphs
from vllm_rlt.worker.sampler import RngRegistry, Sampler


@dataclass(frozen=True)
class SpeculativeResult:
    token_ids: list[int]
    accepted_count: int
    draft_count: int


@dataclass
class SpeculativeStats:
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    committed_tokens: int = 0
    verified_rows: int = 0


@dataclass
class _PipelineBranch:
    """One speculative position and its current recurrent state."""

    item_index: int
    offset: int
    hidden: torch.Tensor
    depth: int = 0


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


class SpeculativeRunner:
    def __init__(self, model, cache, config, execution, rng: RngRegistry | None = None):
        self.model, self.cache, self.config = model, cache, config
        self.device = next(model.parameters()).device
        # The engine injects one registry per engine so both sampling paths share
        # each request's RNG slot.
        self.rng = rng if rng is not None else RngRegistry()
        self.stats = SpeculativeStats()
        self.graphs = (
            RecurrentGraphs(model, cache, execution, False) if execution.cuda_graphs else None
        )
        self.coda_graphs = CodaGraphs(model, execution) if execution.cuda_graphs else None

    def _packed(self, packed):
        return packed and self.cache.attention_capabilities.packed_prefill

    def _cores(self, hidden, ids, positions, depths, *, packed=False):
        # Depth-independent metadata is prepared and copied once for all depths.
        batches = self.cache._prepare_batches(
            ids,
            [[depth] * len(ids) for depth in depths],
            positions,
            packed_prefill=self._packed(packed),
        )
        for depth, batch in zip(depths, batches):
            hidden = self._core(hidden, ids, positions, depth, packed=packed, batch=batch)
        return hidden

    def _core(self, hidden, ids, positions, depth, *, packed=False, batch=None):
        if batch is None:
            batch = self.cache._prepare_batch(
                ids, [depth] * len(ids), positions, packed_prefill=self._packed(packed)
            )
        if self.graphs is not None:
            hidden, _ = self.graphs.run(hidden, batch)
        else:
            hidden, _ = self.model.recurrent_prepared(hidden, batch, self.cache, compute_gate=False)
        return hidden

    def _acquire_generator(self, request, logits):
        """Resolve this request's generator from the shared engine registry.

        The generator has to live on the distribution's device, so it follows the
        logits row rather than the device the model was loaded on.
        """
        return self.rng.acquire(request.request_id, request.sampling_params, logits.device)

    def _core_mixed(self, hidden, ids, positions, depths):
        """Advance active pipeline branches with one depth per row."""
        batch = self.cache._prepare_batch(ids, depths, positions)
        return self._core(hidden, ids, positions, None, batch=batch)

    def _coda(self, hidden):
        return (
            self.coda_graphs.run(hidden)
            if self.coda_graphs is not None
            else self.model.coda(hidden)
        )

    def _greedy_pruning_enabled(self, items):
        return bool(getattr(self.config, "early_rejection_pruning", False)) and all(
            item.request.sampling_params.temperature == 0 for item in items
        )

    @torch.inference_mode()
    def _execute_loop_level_greedy_pruned(self, batch):
        """Run loop-level speculation while pruning rejected greedy suffixes.

        A target row is compared as soon as its coda output is available.  A
        rejected request is removed before the next mixed-depth core wave, so
        shallow work already in flight is kept but no later suffix work is
        started.  This path is greedy-only; sampling retains the common
        batched verification path so its RNG ordering is unchanged.
        """
        items = batch.items
        accepted = [0] * len(items)
        draft_counts = [0] * len(items)
        results = [[] for _ in items]
        candidates = [{} for _ in items]
        stopped = [False] * len(items)

        initial = self.cache._stage([item.request.input_token_id for item in items], torch.long)
        hidden = self.model.prelude(initial)
        active = [_PipelineBranch(i, 0, hidden[i]) for i in range(len(items))]
        verified_rows = 0

        while active:
            ids = [items[branch.item_index].request.request_id for branch in active]
            positions = [items[branch.item_index].token_start + branch.offset for branch in active]
            depths = [branch.depth for branch in active]
            hidden = torch.stack([branch.hidden for branch in active])
            hidden = self._core_mixed(hidden, ids, positions, depths)
            for row, branch in enumerate(active):
                branch.hidden = hidden[row]
                branch.depth += 1

            draft_ready = [
                branch
                for branch in active
                if branch.depth == self.config.draft_loops
                and branch.offset + 1 < items[branch.item_index].token_count
                and not stopped[branch.item_index]
            ]
            target_ready = [branch for branch in active if branch.depth == self.config.target_loops]
            ready = draft_ready + target_ready
            ready_logits = None
            ready_ids = None
            ready_ids_host = None
            if ready:
                ready_logits = self._coda(torch.stack([branch.hidden for branch in ready]))
                # One device-to-host transfer per wave supplies both draft
                # and target IDs.  Per-row `.item()` here would synchronize
                # the GPU once for every target and draft branch.
                ready_ids = ready_logits.argmax(-1)
                ready_ids_host = ready_ids.tolist()

            # Target rows are handled first.  If a parent is rejected, a draft
            # row for the same request in this coda batch must not create its
            # child position.
            target_start = len(draft_ready)
            for row, branch in enumerate(target_ready):
                item_index = branch.item_index
                item = items[item_index]
                target = ready_ids_host[target_start + row]
                verified_rows += 1
                if branch.offset == item.token_count - 1:
                    # The final row has no candidate: it is the bonus token.
                    results[item_index].append(target)
                    stopped[item_index] = True
                    continue
                candidate = candidates[item_index].get(branch.offset)
                if candidate is None:
                    raise RuntimeError(
                        f"missing greedy candidate for item {item_index}, offset {branch.offset}"
                    )
                results[item_index].append(target)
                accepted[item_index] += candidate == target
                if candidate != target:
                    stopped[item_index] = True

            # Generate children only for requests that survived target
            # validation in this wave.  Candidate IDs are selected on-device;
            # their host values come from the single wave-level readback above.
            draft_rows = [
                (row, branch)
                for row, branch in enumerate(draft_ready)
                if not stopped[branch.item_index]
            ]
            children = []
            if draft_rows:
                row_ids = torch.tensor([row for row, _ in draft_rows], device=self.device)
                child_tokens = ready_ids[: len(draft_ready)].index_select(0, row_ids)
                child_hidden = self.model.prelude(child_tokens)
                for child_row, (draft_row, branch) in enumerate(draft_rows):
                    item_index = branch.item_index
                    candidate = ready_ids_host[draft_row]
                    candidates[item_index][branch.offset] = candidate
                    draft_counts[item_index] += 1
                    children.append(
                        _PipelineBranch(item_index, branch.offset + 1, child_hidden[child_row])
                    )

            active = [
                branch
                for branch in active
                if branch.depth < self.config.target_loops
                and not stopped[branch.item_index]
            ]
            active.extend(children)

        self.stats.rounds += len(items)
        self.stats.drafted_tokens += sum(draft_counts)
        self.stats.verified_rows += verified_rows
        return [
            SpeculativeResult(results[i], accepted[i], draft_counts[i]) for i in range(len(items))
        ]

    @torch.inference_mode()
    def _execute_loop_level(self, batch):
        """Run a synchronous single-proposal loop-level pipeline.

        Every active branch advances one recurrent depth per step.  Once a
        branch reaches the draft depth it starts its child position while the
        parent continues toward the full target depth.  The finite window is
        fully verified before returning, so the engine keeps its existing
        atomic commit and KV rollback boundary.
        """
        items = batch.items
        drafted = []
        proposals = [[] for _ in items]
        target_logits = [{} for _ in items]

        initial = self.cache._stage([item.request.input_token_id for item in items], torch.long)
        hidden = self.model.prelude(initial)
        active = [_PipelineBranch(i, 0, hidden[i]) for i in range(len(items))]

        while active:
            # Children start draft_loops steps behind their parents. Thus the
            # earlier position's KV at this row's depth is already complete.
            # Use independent query rows: adjacent positions here have different
            # depth planes and must not form one packed prefill sequence.
            ids = [items[branch.item_index].request.request_id for branch in active]
            positions = [items[branch.item_index].token_start + branch.offset for branch in active]
            depths = [branch.depth for branch in active]
            hidden = torch.stack([branch.hidden for branch in active])
            hidden = self._core_mixed(hidden, ids, positions, depths)
            for row, branch in enumerate(active):
                branch.hidden = hidden[row]
                branch.depth += 1

            draft_ready = [
                branch
                for branch in active
                if branch.depth == self.config.draft_loops
                and branch.offset + 1 < items[branch.item_index].token_count
            ]
            target_ready = [branch for branch in active if branch.depth == self.config.target_loops]
            ready = draft_ready + target_ready
            ready_logits = None
            if ready:
                ready_logits = self._coda(torch.stack([branch.hidden for branch in ready]))

            children = []
            if draft_ready:
                draft_logits = ready_logits[: len(draft_ready)]
                child_tokens = draft_logits.argmax(-1)
                for row, branch in enumerate(draft_ready):
                    request = items[branch.item_index].request
                    if request.sampling_params.temperature != 0:
                        q = Sampler.probabilities(draft_logits[row], request.sampling_params)
                        proposals[branch.item_index].append(q)
                        child_tokens[row] = Sampler.draw(
                            q, self._acquire_generator(request, draft_logits[row])
                        )
                drafted.append(([branch.item_index for branch in draft_ready], child_tokens))
                child_hidden = self.model.prelude(child_tokens)
                children = [
                    _PipelineBranch(
                        branch.item_index,
                        branch.offset + 1,
                        child_hidden[row],
                    )
                    for row, branch in enumerate(draft_ready)
                ]

            if target_ready:
                target_start = len(draft_ready)
                for row, branch in enumerate(target_ready):
                    target_logits[branch.item_index][branch.offset] = ready_logits[
                        target_start + row
                    ].clone()

            active = [branch for branch in active if branch.depth < self.config.target_loops]
            active.extend(children)

        logits = torch.stack(
            [
                target_logits[i][offset]
                for i, item in enumerate(items)
                for offset in range(item.token_count)
            ]
        )
        return self._verify(items, drafted, proposals, logits)

    @torch.inference_mode()
    def execute(self, batch):
        if self.config.loop_level_pipelining:
            if self._greedy_pruning_enabled(batch.items):
                return self._execute_loop_level_greedy_pruned(batch)
            return self._execute_loop_level(batch)
        items = batch.items
        proposals = [[] for _ in items]
        states = [[] for _ in items]
        # Draft IDs stay on the device: each offset's IDs are the next offset's
        # prelude input, and all of them are read back once after verification.
        drafted = []
        inputs = self.cache._stage([item.request.input_token_id for item in items], torch.long)
        # Each item includes the final shallow input needed for the bonus row.
        # Drafting batches independent requests at every autoregressive step.
        for offset in range(max(item.token_count for item in items)):
            # The rows still drafting at the previous offset are exactly this
            # offset's active rows, in the same order, so their IDs are the inputs.
            active = [i for i, item in enumerate(items) if offset < item.token_count]
            ids = [items[i].request.request_id for i in active]
            positions = [items[i].token_start + offset for i in active]
            hidden = self.model.prelude(inputs)
            hidden = self._cores(hidden, ids, positions, range(self.config.draft_loops))
            drafting = []
            for row, i in enumerate(active):
                states[i].append(hidden[row])
                if offset + 1 < items[i].token_count:
                    drafting.append((row, i))
            if not drafting:
                break
            logits = self._coda(torch.stack([hidden[row] for row, _ in drafting]))
            inputs = logits.argmax(-1)
            for row, (_, i) in enumerate(drafting):
                request = items[i].request
                if request.sampling_params.temperature != 0:
                    q = Sampler.probabilities(logits[row], request.sampling_params)
                    proposals[i].append(q)
                    # Per-request RNG consumption order is unchanged; only the read moves.
                    inputs[row] = Sampler.draw(q, self._acquire_generator(request, logits[row]))
            drafted.append(([i for _, i in drafting], inputs))
        # Group contiguous positions per request for causal ragged attention.
        ids, positions = [], []
        for item in items:
            ids.extend([item.request.request_id] * item.token_count)
            positions.extend(range(item.token_start, item.token_start + item.token_count))
        hidden = torch.stack([h for request_states in states for h in request_states])
        hidden = self._cores(
            hidden,
            ids,
            positions,
            range(self.config.draft_loops, self.config.target_loops),
            packed=True,
        )
        logits = self._coda(hidden)
        return self._verify(items, drafted, proposals, logits)

    def _verify(self, items, drafted, proposals, logits):
        """Collect the finite window and apply the common acceptance rule."""
        # One device-to-host read returns every draft ID and, for greedy requests,
        # every verification target argmax.
        greedy = any(item.request.sampling_params.temperature == 0 for item in items)
        pending = [draft for _, draft in drafted] + ([logits.argmax(-1)] if greedy else [])
        values = torch.cat(pending).tolist() if pending else []
        candidates = [[] for _ in items]
        start = 0
        for rows, _ in drafted:
            for i, candidate in zip(rows, values[start : start + len(rows)]):
                candidates[i].append(candidate)
            start += len(rows)
        targets = values[start:]
        results, start = [], 0
        for i, item in enumerate(items):
            rows = logits[start : start + item.token_count]
            request = item.request
            if request.sampling_params.temperature == 0:
                tokens, accepted = greedy_accept(
                    candidates[i], targets[start : start + item.token_count]
                )
                start += item.token_count
                results.append(SpeculativeResult(tokens, accepted, len(candidates[i])))
                continue
            start += item.token_count
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
            results.append(SpeculativeResult(tokens, accepted, len(candidates[i])))
        self.stats.rounds += len(items)
        self.stats.drafted_tokens += sum(len(c) for c in candidates)
        self.stats.verified_rows += len(logits)
        return results
