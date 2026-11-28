"""Validate RLT settings carried in vLLM's additional_config."""

from collections.abc import Mapping
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class LoopRuntimeConfig:
    trace_steps: bool = False
    prefill_chunk_size: int | None = None
    prefill_window: int | None = None
    self_speculation: bool = False
    mixed_core: bool = False
    compact_mixed_core: bool = False
    packed_prefill: bool = False
    decode_pipeline: bool = False
    async_prefill_lookahead: bool = False
    merge_decode_core: bool = False
    merge_prefill_core: bool = False
    decode_state_slots: bool = False
    unified_coda: bool = True
    cross_depth_decode: bool = True
    split_prefill_coda: bool = False
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
            if name in ("prefill_chunk_size", "prefill_window"):
                if type(value) is not int or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
            elif type(value) is not bool:
                raise ValueError(f"{key} must be a boolean")
            values[name] = value
        return cls(**values)


def supports_prefill_pipeline(config, options):
    model = config.model_config
    text = model.hf_text_config
    parallel = config.parallel_config
    return (
        hasattr(text, "num_loop_passes")
        and not getattr(text, "loop_share_cache", False)
        and not (options.self_speculation or options.decode_pipeline)
        and config.kv_transfer_config is None
        and config.lora_config is None
        and config.speculative_config is None
        and parallel.tensor_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and not parallel.use_ubatching
        and not getattr(model, "enable_return_routed_experts", False)
    )


def supports_cross_depth_decode(config, options):
    return (
        options.cross_depth_decode and options.mixed_core
        and supports_prefill_pipeline(config, options)
    )
