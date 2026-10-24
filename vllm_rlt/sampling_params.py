import math
import secrets
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    # None lets the engine draw a per-request seed; outputs report the seed used.
    seed: int | None = 0
    min_loops: int = 2
    max_loops: int | None = None
    exit_threshold: float = 1.0
    ignore_eos: bool = False
    priority: int = 0
    # vLLM convention: 0 returns only the sampled token's log-probability.
    logprobs: int | None = None
    # Stop after emitting any of these IDs, even with ignore_eos (vLLM semantics).
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")
        for name in ("max_tokens", "min_loops", "max_loops"):
            value = getattr(self, name)
            if name == "max_loops" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_loops is not None and self.max_loops < self.min_loops:
            raise ValueError("max_loops must be at least min_loops")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or (self.top_k != -1 and self.top_k < 1):
            raise ValueError("top_k must be -1 or positive")
        if not 0 <= self.exit_threshold <= 1:
            raise ValueError("exit_threshold must be in [0, 1]")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("seed must be None or an integer in [0, 2**63)")
        if self.logprobs is not None:
            if type(self.logprobs) is not int or self.logprobs < 0:
                raise ValueError("logprobs must be None or 0")
            if self.logprobs > 0:
                raise ValueError(
                    "logprobs > 0 (top-k alternatives) is not supported yet; "
                    "use 0 for the sampled token's log-probability"
                )
        if not isinstance(self.stop_token_ids, (list, tuple)) or any(
            type(token) is not int or token < 0 for token in self.stop_token_ids
        ):
            raise ValueError("stop_token_ids must be a list of nonnegative integers")
        object.__setattr__(self, "stop_token_ids", tuple(self.stop_token_ids))


def resolve_seed(params: SamplingParams) -> SamplingParams:
    """Return ``params`` with ``seed=None`` replaced by a fresh 63-bit seed.

    The seed comes from OS entropy, so torch and Python global RNG streams are
    not consumed. Engines call this before creating a Request, so outputs and
    every PD worker observe the same effective seed.
    """
    if params.seed is not None:
        return params
    return replace(params, seed=secrets.randbits(63))
