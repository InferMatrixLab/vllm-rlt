"""Greedy, top-k and top-p sampling for a single logits row.

The arithmetic was moved out of ``ModelRunner``'s per-row sampling helper so the
sampling contract can be pinned by unit tests without a runner. ``probabilities``
and ``draw`` were unified here from ``worker/sampling.py``, so the CODA and
speculative paths keep one copy of the distribution construction.

Top-k keeps exactly ``k`` candidates by index, so tokens tied with the k-th logit
no longer widen the support. That is a deliberate arithmetic change from the
earlier threshold mask, which kept every tied token; the remaining order is
unchanged (``float()/temperature`` -> top-k -> top-p -> ``multinomial``).

``Sampler`` holds no state and never creates a generator: it consumes the one the
caller obtained from ``RngRegistry``. The registry is the single owner of RNG
creation, lookup and lifetime, and the engine shares one instance between its
runners.
"""

import torch

from vllm_rlt.sampling_params import SamplingParams


class Sampler:
    """Greedy / top-k / top-p sampling for one logits row."""

    @staticmethod
    def probabilities(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Return the temperature/top-k/top-p distribution for one logits row.

        ``params`` is assumed to be validated by ``SamplingParams.__post_init__``.
        Greedy decoding has no temperature-scaled distribution, so it is a caller
        error rather than a silent argmax. Top-k keeps exactly ``k`` candidates
        (or every token when ``k`` reaches the vocabulary); top-p may trim that
        support further.
        """
        if params.temperature == 0:
            raise ValueError("greedy decoding has no temperature-scaled distribution")
        logits = logits.float() / params.temperature
        if params.top_k > 0:
            # Select by index, not by the k-th logit value: a threshold mask also
            # keeps every token tied with it, so the support could exceed k. Which
            # index survives a tie is not specified across devices, so callers
            # must not rely on a particular one. ``scatter_`` writes into the
            # buffer created here, so the caller's row is never modified and the
            # candidates come from topk instead of a second gather.
            values, indices = logits.topk(min(params.top_k, logits.numel()))
            logits = torch.full_like(logits, -torch.inf).scatter_(0, indices, values)
        if params.top_p < 1:
            sorted_logits, indices = logits.sort(descending=True)
            remove = sorted_logits.softmax(-1).cumsum(-1) > params.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits = logits.scatter(0, indices, sorted_logits.masked_fill(remove, -torch.inf))
        return logits.softmax(-1)

    @staticmethod
    def selected_logprob(
        logits: torch.Tensor, token: int | torch.Tensor, params: SamplingParams
    ) -> torch.Tensor:
        """Score the selected token using the same filters as token sampling."""
        if params.logprobs_mode == "processed" and params.temperature > 0:
            return Sampler.probabilities(logits, params)[token].log()
        # Integer speculative tokens otherwise retain a view of the vocabulary row.
        return logits.float().log_softmax(-1)[token].clone()

    @staticmethod
    def draw(probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Draw one token from a normalized distribution as a 0-dim long tensor."""
        return torch.multinomial(probs, 1, generator=generator).squeeze(0)

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Return a 0-dim long device tensor drawn from a caller-resolved generator.

        Greedy decoding needs no RNG and returns ``argmax`` whatever the caller
        passed. Non-greedy decoding requires a generator: ``None`` would silently
        draw from the process-wide default generator, so it is rejected instead.

        No host synchronization happens here: the returned tensor stays on the
        logits device so the async CODA path can feed it to the next prelude.
        """
        if params.temperature == 0:
            return logits.argmax()
        if generator is None:
            raise ValueError("non-greedy sampling requires a generator")
        return self.draw(self.probabilities(logits, params), generator)


class RngRegistry:
    """Per-request sampling generators for one engine.

    The engine owns one registry and injects the same instance into every runner,
    so the CODA and speculative paths share a request's RNG slot. This registry is
    the single creator of generators. Greedy token sampling needs no slot, but
    a model's random recurrent-state initialization can acquire one. Suspension
    keeps a slot, and termination drops it so a reused request ID starts from a
    fresh stream.
    """

    def __init__(self):
        self.generators: dict[str, torch.Generator] = {}

    def peek(self, request_id: str) -> torch.Generator | None:
        """Return the request's generator, or None before its first random draw."""
        return self.generators.get(request_id)

    def acquire(
        self, request_id: str, params: SamplingParams, device: torch.device
    ) -> torch.Generator:
        """Return the request's generator, creating it on ``device`` if needed.

        The caller passes the device of the distribution it will sample from, so
        ``multinomial`` and its generator agree. An existing generator is returned
        as-is and never migrated.
        """
        generator = self.generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(params.seed)
            self.generators[request_id] = generator
        return generator

    def drop(self, request_id: str) -> None:
        """Forget the request's generator; a suspended request must not reach here."""
        self.generators.pop(request_id, None)
