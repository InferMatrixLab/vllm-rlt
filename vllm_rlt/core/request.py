"""Per-request loop options, carried by the upstream request lifecycle."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class LoopRequestParams:
    target_loops: int | None = None
    exit_threshold: float | None = None
    min_loops: int | None = None
    max_loops: int | None = None

    @classmethod
    def from_extra_args(cls, extra_args: Mapping[str, object] | None) -> "LoopRequestParams":
        if not extra_args:
            return cls()

        target = _read_loop_count(extra_args, "rlt_target_loops")
        minimum = _read_loop_count(extra_args, "rlt_min_loops")
        maximum = _read_loop_count(extra_args, "rlt_max_loops")
        threshold = extra_args.get("rlt_exit_threshold")
        if threshold is not None:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not 0 <= threshold <= 1
            ):
                raise VLLMValidationError(
                    "rlt_exit_threshold must be a number in [0, 1]",
                    parameter="rlt_exit_threshold",
                    value=threshold,
                )

        if minimum is not None and maximum is not None and minimum > maximum:
            raise VLLMValidationError(
                "rlt_min_loops must be at most rlt_max_loops",
                parameter="rlt_min_loops",
                value=minimum,
            )
        if target is not None and any(value is not None for value in (minimum, maximum, threshold)):
            raise VLLMValidationError(
                "rlt_target_loops cannot be combined with adaptive exit options",
                parameter="rlt_target_loops",
                value=target,
            )

        return cls(target, float(threshold) if threshold is not None else None, minimum, maximum)


def _read_loop_count(extra_args: Mapping[str, object], key: str) -> int | None:
    value = extra_args.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise VLLMValidationError(f"{key} must be a positive integer", parameter=key, value=value)
    return value


class LoopRequest(Request):
    """An upstream request with standalone, validated loop options."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        sampling = self.sampling_params
        self.loop_params = LoopRequestParams.from_extra_args(
            sampling.extra_args if sampling is not None else None
        )
