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

Rollout logprobs (``worker/logprobs.py``) score sampled tokens with
``Sampler.raw_logprobs``, on ``Sampler.processed_logits`` in processed mode. That
module imports this one, never the reverse.
"""

import torch

from vllm_rlt.sampling_params import SamplingParams


class Sampler:
    """Greedy / top-k / top-p sampling for one logits row."""

    @staticmethod
    def processed_logits(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Return the FP32 temperature/top-k/top-p logits for one logits row.

        ``params`` is assumed to be validated by ``SamplingParams.__post_init__``.
        Greedy decoding has no temperature-scaled distribution, so it is a caller
        error rather than a silent argmax. Excluded tokens are ``-inf``. Top-k
        keeps exactly ``k`` candidates (or every token when ``k`` reaches the
        vocabulary); top-p may trim that support further. ``probabilities`` (CODA
        and speculative sampling) and processed logprobs share this arithmetic.
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
        return logits

    @staticmethod
    def probabilities(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Return the temperature/top-k/top-p distribution for one logits row.

        It is the softmax of ``processed_logits``, which rejects greedy decoding.
        """
        return Sampler.processed_logits(logits, params).softmax(-1)

    @staticmethod
    def draw(probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Draw one token from a normalized distribution as a 0-dim long tensor."""
        return torch.multinomial(probs, 1, generator=generator).squeeze(0)

    @staticmethod
    def raw_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """FP32 [rows] log-probabilities of ``token_ids`` [rows] under ``logits`` [rows, vocab].

        Scoring consumes no RNG and keeps the result on the logits device.
        """
        logprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        return logprobs.gather(-1, token_ids[:, None]).squeeze(-1)

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

    def sample_with_logprob(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``sample()`` plus the 0-dim FP32 processed logprob of the drawn token.

        The logprob is ``log_softmax`` of the same processed logits the token is
        drawn from, so it equals ``token_logprob(..., "processed_logprobs")``
        without rebuilding them; greedy rows report the unscaled distribution.
        Tokens, generator requirements and RNG consumption are those of ``sample()``.
        """
        if params.temperature == 0:
            token = logits.argmax()
            return token, self.raw_logprobs(logits[None], token.view(1))[0]
        if generator is None:
            raise ValueError("non-greedy sampling requires a generator")
        processed = self.processed_logits(logits, params)
        token = self.draw(processed.softmax(-1), generator)
        return token, self.raw_logprobs(processed[None], token.view(1))[0]


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
        ``multinomial`` and its generator agree. A new generator is seeded with
        ``params.seed``, which must be engine-resolved, so its ``initial_seed()``
        is the seed the request's outputs report. An existing generator is
        returned as-is and never reseeded or migrated.
        """
        generator = self.generators.get(request_id)
        if generator is None:
            if params.seed is None:  # Engines resolve seed=None at admission.
                raise ValueError("seed=None must be resolved by the engine before sampling")
            generator = torch.Generator(device=device).manual_seed(params.seed)
            self.generators[request_id] = generator
        return generator

    def drop(self, request_id: str) -> None:
        """Forget the request's generator; a suspended request must not reach here."""
        self.generators.pop(request_id, None)
