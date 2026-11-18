import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = 0
    logprobs: int | None = None
    logprobs_mode: str = "raw"
    stop_token_ids: tuple[int, ...] = ()
    min_loops: int = 2
    max_loops: int | None = None
    exit_threshold: float = 1.0
    ignore_eos: bool = False
    priority: int = 0

    def __post_init__(self):
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")
        for name, value in (
            ("max_tokens", self.max_tokens),
            ("min_loops", self.min_loops),
            ("max_loops", self.max_loops),
        ):
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
            raise ValueError("seed must be an integer in [0, 2**63)")
        if self.logprobs is not None and (type(self.logprobs) is not int or self.logprobs != 0):
            raise ValueError("logprobs=0 returns the selected-token probability")
        if self.logprobs_mode not in ("raw", "processed"):
            raise ValueError("logprobs_mode must be raw or processed")
        if any(type(token) is not int or token < 0 for token in self.stop_token_ids):
            raise ValueError("stop_token_ids must contain nonnegative integer IDs")
