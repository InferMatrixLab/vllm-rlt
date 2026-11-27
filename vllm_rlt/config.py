"""Validate RLT settings carried in vLLM's additional_config."""

from collections.abc import Mapping
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class LoopRuntimeConfig:
    trace_steps: bool = False
    wavefront_prefill: bool = False
    prefill_chunk_size: int | None = None
    self_speculation: bool = False
    mixed_core: bool = False
    compact_mixed_core: bool = False
    packed_prefill: bool = False
    decode_pipeline: bool = False
    async_prefill_lookahead: bool = False
    merge_decode_core: bool = False
    merge_prefill_core: bool = False
    decode_state_slots: bool = False
    batch_decode_coda: bool = False
    finish_decode_token: bool = False
    silu_and_mul: bool = False

    @classmethod
    def from_additional_config(cls, config: Mapping[str, object] | None) -> "LoopRuntimeConfig":
        values = {}
        names = {f"rlt_{field.name}": field.name for field in fields(cls)}
        for key, value in (config or {}).items():
            if not key.startswith("rlt_"):
                continue
            name = names.get(key)
            if name is None:
                raise ValueError(f"Unknown RLT runtime option {key!r}")
            if name == "prefill_chunk_size":
                if type(value) is not int or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
            elif type(value) is not bool:
                raise ValueError(f"{key} must be a boolean")
            values[name] = value
        return cls(**values)
