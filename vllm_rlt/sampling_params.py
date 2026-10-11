import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopCDParams:
    """Fixed guidance after complete one-based recurrent updates."""

    mode: str = "logits"
    reference_loop: int = 1
    strength: float = 0.3
    implementation: str = "two_head"
    strength_mode: str = "fixed"
    prefill_reference_loop: int | None = None

    def __post_init__(self):
        if self.mode != "logits":
            raise ValueError("Ouro LoopCD supports logits mode only")
        if self.strength_mode != "fixed":
            raise ValueError("LoopCD supports fixed strength only")
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("LoopCD strength must be finite and nonnegative")
        for value in (self.reference_loop, self.prefill_reference_loop):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("LoopCD reference_loop must be a positive integer")
        if self.reference_loop is None:
            raise ValueError("LoopCD reference_loop is required")
        allowed = ("two_head", "linear_fused")
        if self.implementation not in allowed:
            raise ValueError("LoopCD implementation does not match guidance mode")

    @property
    def prefill_loop(self):
        return self.prefill_reference_loop or self.reference_loop


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 0
    min_loops: int = 2
    max_loops: int | None = None
    exit_threshold: float = 1.0
    ignore_eos: bool = False
    priority: int = 0
    loopcd: LoopCDParams | None = None

    def __post_init__(self):
        if self.loopcd is not None and not isinstance(self.loopcd, LoopCDParams):
            raise ValueError("loopcd must be LoopCDParams or None")
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
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
