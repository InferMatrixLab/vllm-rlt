"""Entry-owned model configuration shared by deployment clients and workers."""

from dataclasses import dataclass
from pathlib import Path

from torch import nn

from .huginn import HuginnConfig
from .nanbeige import NanbeigeConfig
from .ouro import OuroConfig

_CONFIG_TYPES = {
    "ouro": OuroConfig,
    "nanbeige": NanbeigeConfig,
    "huginn_raven": HuginnConfig,
}


@dataclass(frozen=True)
class ModelConfig:
    """A concrete model config, treated as read-only after initialization.

    Spawned workers receive copies, not shared mutable configuration. Derived
    limits remain properties of the concrete config rather than duplicated fields.
    """

    model_specific_config: OuroConfig | NanbeigeConfig | HuginnConfig

    def __post_init__(self):
        if type(self.model_specific_config) not in _CONFIG_TYPES.values():
            raise ValueError("Unsupported model configuration type")

    @classmethod
    def from_dict(cls, values):
        model_type = values.get("model_type", "").lower()
        if model_type not in _CONFIG_TYPES:
            raise ValueError(f"Unsupported model_type {model_type!r}")
        return cls(_CONFIG_TYPES[model_type].from_dict(values))

    @property
    def model_type(self):
        return self.model_specific_config.to_dict()["model_type"]

    @property
    def vocab_size(self):
        return self.model_specific_config.vocab_size

    @property
    def total_ut_steps(self):
        return self.model_specific_config.total_ut_steps

    @property
    def max_position_embeddings(self):
        return self.model_specific_config.max_position_embeddings


def prepare_model(model, *, revision=None, model_config=None):
    """Resolve a source once, before spawning; preserve its weight-source semantics."""
    if model_config is not None:
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be a ModelConfig")
        concrete = model.config if isinstance(model, nn.Module) else model
        if not isinstance(model, (str, Path)) and concrete != model_config.model_specific_config:
            raise ValueError("Model source and model_config do not match")
        return model, revision, model_config
    if isinstance(model, (str, Path)):
        # The resolver also owns download approval and snapshot pinning.
        from . import resolve_model_config

        model, revision, values = resolve_model_config(model, revision=revision)
        model_config = ModelConfig.from_dict(values)
    else:
        model_config = ModelConfig(model.config if isinstance(model, nn.Module) else model)
    return model, revision, model_config
