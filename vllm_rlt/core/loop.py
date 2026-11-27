"""Work units exchanged between the loop scheduler and the model runner."""

from dataclasses import dataclass
from enum import StrEnum


class LoopOp(StrEnum):
    PRELUDE = "prelude"
    CORE = "core"
    CODA = "coda"


@dataclass(frozen=True, slots=True)
class LoopLayout:
    num_loops: int
    has_prelude: bool
    has_coda: bool
    variable_depth: bool
    early_exit: bool = False

    @property
    def first_op(self) -> LoopOp:
        return LoopOp.PRELUDE if self.has_prelude else LoopOp.CORE

    @classmethod
    def from_model_config(cls, model_config) -> "LoopLayout":
        text = model_config.hf_text_config
        if text.model_type == "ouro":
            return cls(text.total_ut_steps, False, False, True, early_exit=True)
        loop = getattr(text, "loop_config", None)
        if loop and hasattr(text, "num_loop_passes"):
            return cls(text.num_loop_passes, True, True, loop["is_varloop"])
        raise ValueError(f"the rlt scheduler does not support model type {text.model_type!r}")


@dataclass(frozen=True, slots=True)
class LoopSpan:
    """A contiguous range of token positions at one loop depth."""

    request_id: str
    token_start: int
    token_count: int
    depth: int
    prefill: bool = False
    last_core: bool = False
    sample: bool = False
    exit_threshold: float | None = None
    min_loops: int = 1
    propose_draft: bool = False
    draft_coda: bool = False
    input_token: int | None = None

    def __post_init__(self) -> None:
        if self.token_start < 0 or self.token_count < 1 or self.depth < 0:
            raise ValueError("loop span needs a nonnegative start and depth and positive count")

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_count

    @property
    def check_exit(self) -> bool:
        return self.exit_threshold is not None
