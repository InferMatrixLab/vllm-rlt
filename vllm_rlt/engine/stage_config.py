"""Deployment capability, independent of batching and speculation policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StageConfig:
    stage_id: str = "local"
    role: str = "prefill_decode"

    @property
    def accepts_prompt(self):
        return self.role != "decode"

    @property
    def produces_tokens(self):
        return self.role != "prefill"

    def __post_init__(self):
        if self.role not in ("prefill", "decode", "prefill_decode"):
            raise ValueError("stage role must be prefill, decode or prefill_decode")
        if not self.stage_id:
            raise ValueError("stage_id must not be empty")
