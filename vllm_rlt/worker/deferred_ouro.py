# SPDX-License-Identifier: Apache-2.0
"""Experimental fixed-interval Ouro decoding with exact per-loop KV histories.

This inference-only runner groups loops three and four across response positions.
It groups genuine deep history without LAST_EXITED propagation copies.
Weights must remain unchanged until the runner is closed.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.ouro import OuroForCausalLM
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.sampler import RngRegistry, Sampler


@dataclass
class _Request:
    request_id: str
    capacity: int
    position: int = 0
    response_inputs: int = 0
    pending_hidden: list[torch.Tensor] = field(default_factory=list)
    pending_positions: list[int] = field(default_factory=list)
    live: bool = False


@dataclass
class DeferredOutput:
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    readout_depths: list[int] = field(default_factory=list)
    finish_reason: str = "length"


class DeferredOuroRunner:
    """One fixed-weight batch; shallow readout is two and deep readout is four.

    ``defer=False`` is the same-readout eager reference, useful for comparisons.
    Positions passed to attention are original request positions, including when
    several requests and response blocks are concatenated into one deep batch.
    """

    def __init__(
        self,
        model: OuroForCausalLM,
        capacities: Sequence[int],
        *,
        interval: int = 4,
        backend: str = "auto",
        defer: bool = True,
    ):
        if model.config.total_ut_steps != 4:
            raise ValueError("Deferred Ouro decoding requires a four-loop model")
        if type(interval) is not int or interval not in (1, 2, 4, 8):
            raise ValueError("interval must be one of 1, 2, 4, 8")
        if not capacities or any(
            type(n) is not int or not 0 < n <= model.config.max_position_embeddings
            for n in capacities
        ):
            raise ValueError("Provide positive capacities within the model's position limit")
        self.model, self.interval, self.defer = model, interval, defer
        parameter = next(model.parameters())
        self.device = parameter.device
        self.prefilled = False
        block_size = 16
        self.cache = KVCacheManager(
            num_layers=model.config.num_hidden_layers,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            num_blocks=sum(math.ceil(n / block_size) * 4 for n in capacities),
            block_size=block_size,
            max_loops=4,
            device=self.device,
            dtype=parameter.dtype,
            backend=backend,
            layout="last_exited",
        )
        self.requests = [_Request(f"deferred-{i}", n) for i, n in enumerate(capacities)]
        for request in self.requests:
            if not self.cache.allocate(request.request_id, request.capacity):
                self.close()
                raise RuntimeError("Insufficient KV pages for the request batch")
            request.live = True

    def _run_loops(self, hidden, request_ids, positions, depths):
        try:
            for depth in depths:
                hidden, _ = self.model.recurrent(
                    hidden,
                    request_ids,
                    [depth] * len(request_ids),
                    positions,
                    self.cache,
                    compute_gate=False,
                )
            return hidden
        except Exception:
            self.close()
            raise

    @torch.inference_mode()
    def prefill(self, prompts: Sequence[Sequence[int]]) -> torch.Tensor:
        if self.prefilled or not all(request.live for request in self.requests):
            raise ValueError("A runner accepts one prefill batch")
        if len(prompts) != len(self.requests) or any(
            not prompt or len(prompt) > request.capacity
            for prompt, request in zip(prompts, self.requests, strict=True)
        ):
            raise ValueError("Provide one nonempty prompt within each request capacity")
        tokens, request_ids, positions, ends = [], [], [], []
        for prompt, request in zip(prompts, self.requests, strict=True):
            tokens.extend(prompt)
            request_ids.extend([request.request_id] * len(prompt))
            positions.extend(range(len(prompt)))
            ends.append(len(tokens) - 1)
            request.position = len(prompt)
        hidden = self.model.prelude(torch.tensor(tokens, device=self.device, dtype=torch.long))
        hidden = self._run_loops(hidden, request_ids, positions, range(4))
        self.prefilled = True
        return hidden[ends]

    @torch.inference_mode()
    def step(
        self, active: Sequence[int], token_ids: Sequence[int]
    ) -> tuple[torch.Tensor, list[int]]:
        if (
            not self.prefilled
            or not active
            or len(active) != len(token_ids)
            or len(set(active)) != len(active)
        ):
            raise ValueError(
                "Provide distinct active requests and one token per request after prefill"
            )
        if any(type(i) is not int or not 0 <= i < len(self.requests) for i in active):
            raise ValueError("Invalid active request index")
        requests = [self.requests[i] for i in active]
        if any(not request.live or request.position >= request.capacity for request in requests):
            raise ValueError("Request is closed or has exhausted its KV capacity")
        ids = [request.request_id for request in requests]
        positions = [request.position for request in requests]
        hidden = self.model.prelude(torch.tensor(token_ids, device=self.device, dtype=torch.long))
        hidden = self._run_loops(hidden, ids, positions, range(2))
        selected = hidden.clone()
        readouts, deep_hidden, deep_ids, deep_positions, ends = [], [], [], [], []
        for row, request in enumerate(requests):
            request.response_inputs += 1
            depth = 4 if request.response_inputs % self.interval == 0 else 2
            readouts.append(depth)
            if self.defer:
                request.pending_hidden.append(hidden[row : row + 1].clone())
                request.pending_positions.append(request.position)
                if depth == 4:
                    deep_hidden.extend(request.pending_hidden)
                    deep_ids.extend([request.request_id] * len(request.pending_positions))
                    deep_positions.extend(request.pending_positions)
                    ends.append((row, len(deep_positions) - 1))
                    request.pending_hidden.clear()
                    request.pending_positions.clear()
            else:
                deep_hidden.append(hidden[row : row + 1])
                deep_ids.append(request.request_id)
                deep_positions.append(request.position)
                if depth == 4:
                    ends.append((row, len(deep_positions) - 1))
            request.position += 1
        if deep_hidden:
            deep = self._run_loops(torch.cat(deep_hidden), deep_ids, deep_positions, range(2, 4))
            for row, end in ends:
                selected[row] = deep[end]
        return selected, readouts

    def release(self, active: Sequence[int]) -> None:
        """Discard unread deep tails; never copy shallow KV into deep planes."""
        if any(type(index) is not int or not 0 <= index < len(self.requests) for index in active):
            raise ValueError("Invalid request index for release")
        for index in active:
            request = self.requests[index]
            if request.live:
                request.pending_hidden.clear()
                request.pending_positions.clear()
                self.cache.free(request.request_id)
                request.live = False

    def close(self) -> None:
        self.release(range(len(self.requests)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@torch.inference_mode()
def generate_deferred(
    model: OuroForCausalLM,
    prompts: Sequence[Sequence[int]],
    seeds: Sequence[int],
    *,
    interval: int = 4,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    stop_token_ids: Sequence[int] = (0, 2),
    backend: str = "auto",
    defer: bool = True,
) -> list[DeferredOutput]:
    """Full-vocabulary sampling with independent per-answer RNG and processed scores.

    The first output uses full four-loop prompt prefill. This is a mixed-readout
    inference policy, not the native four-loop policy and not a training adapter.
    """
    if (
        not prompts
        or len(prompts) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Provide one nonnegative seed per nonempty prompt batch")
    if (
        type(max_tokens) is not int
        or max_tokens <= 0
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("max_tokens and temperature must be positive")
    params = [
        SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed) for seed in seeds
    ]
    outputs = [DeferredOutput() for _ in prompts]
    registry, sampler = RngRegistry(), Sampler()
    with DeferredOuroRunner(
        model,
        [len(prompt) + max_tokens - 1 for prompt in prompts],
        interval=interval,
        backend=backend,
        defer=defer,
    ) as runner:
        generators = [
            registry.acquire(request.request_id, parameter, runner.device)
            for request, parameter in zip(runner.requests, params, strict=True)
        ]
        hidden = runner.prefill(prompts)
        active, depths = list(range(len(prompts))), [4] * len(prompts)
        for step in range(max_tokens):
            logits = model.coda(hidden).float()
            survivors, tokens, finished = [], [], []
            for row, index in enumerate(active):
                probabilities = sampler.probabilities(logits[row], params[index])
                token = int(sampler.draw(probabilities, generators[index]))
                output = outputs[index]
                output.token_ids.append(token)
                output.logprobs.append(float(probabilities[token].log()))
                output.readout_depths.append(depths[row])
                if token in stop_token_ids:
                    output.finish_reason = "stop"
                    finished.append(index)
                elif step + 1 < max_tokens:
                    survivors.append(index)
                    tokens.append(token)
            runner.release(finished)
            if not survivors:
                break
            hidden, depths = runner.step(survivors, tokens)
            active = survivors
    return outputs
