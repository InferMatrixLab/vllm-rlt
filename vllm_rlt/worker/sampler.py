"""Greedy, top-k and top-p sampling for a single logits row.

The arithmetic is a verbatim move out of ``ModelRunner._sample_tensor`` so the
sampling contract can be pinned by unit tests without a runner. This class owns
no per-request state: the caller stores and passes in the RNG generator.
"""

import torch

from vllm_rlt.sampling_params import SamplingParams


def probabilities(
    logits: torch.Tensor,
    params: SamplingParams,
    temperature: float | None = None,
) -> torch.Tensor:
    temp = params.temperature if temperature is None else temperature
    if temp == 0:
        raise ValueError("greedy decoding has no temperature-scaled distribution")
    logits = logits.float() / temp
    if params.top_k > 0:
        threshold = logits.topk(min(params.top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if params.top_p < 1:
        sorted_logits, indices = logits.sort(descending=True)
        remove = sorted_logits.softmax(-1).cumsum(-1) > params.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits = logits.scatter(0, indices, sorted_logits.masked_fill(remove, -torch.inf))
    return logits.softmax(-1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    repetition_penalty: float,
    token_ids: list[int],
) -> torch.Tensor:
    if repetition_penalty == 1.0 or not token_ids:
        return logits

    prev_tokens = torch.tensor(
        list(set(token_ids)),
        device=logits.device,
        dtype=torch.long,
    )
    prev_logits = logits[prev_tokens]
    logits[prev_tokens] = torch.where(
        prev_logits > 0,
        prev_logits / repetition_penalty,
        prev_logits * repetition_penalty,
    )
    return logits


class Sampler:
    """Greedy / top-k / top-p sampling for one logits row."""

    def __init__(self, device: torch.device):
        self.device = device

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None,
        *,
        token_ids: list[int] | None = None,
        loops_done: int = 0,
    ) -> tuple[torch.Tensor, torch.Generator | None]:
        """Return a 0-dim long device tensor and the possibly created generator.

        ``params`` is assumed to be validated by ``SamplingParams.__post_init__``.
        No host synchronization happens here: the returned tensor stays on the
        logits device so the async CODA path can feed it to the next prelude.
        """
        logits = logits.float().clone()
        if params.repetition_penalty != 1.0:
            logits = apply_repetition_penalty(
                logits,
                params.repetition_penalty,
                token_ids or [],
            )

        temperature = params.temperature
        if params.dynamic_depth_temp and temperature > 0:
            if 0 < loops_done <= 2:
                temperature = max(temperature * 0.7, 0.1)
            elif loops_done >= 4:
                temperature = temperature * 1.2

        if temperature == 0:
            return logits.argmax(), generator

        probs = probabilities(logits, params, temperature=temperature)
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)
        return torch.multinomial(probs, 1, generator=generator).squeeze(0), generator
