# SPDX-License-Identifier: Apache-2.0
"""Shared small models, workload-specific engine factories and protocol doubles."""

from typing import Any

import pytest
import torch

from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.entrypoints.llm import LLM
from vllm_rlt.models import NanbeigeConfig, OuroConfig, OuroForCausalLM


def tiny_ouro_config(**overrides: Any) -> OuroConfig:
    """Small randomly initialized Ouro architecture for tests."""
    values = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        total_ut_steps=4,
        bos_token_id=0,
        eos_token_id=0,
    )
    values.update(overrides)
    return OuroConfig(**values)


def tiny_nanbeige_config(**overrides: Any) -> NanbeigeConfig:
    """Small randomly initialized Nanbeige architecture for tests."""
    values = dict(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        num_loops=2,
        total_ut_steps=2,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    values.update(overrides)
    return NanbeigeConfig(**values)


class LocalLLM(LLM):
    """Runner-test harness; production LLM always launches a stage process."""

    def __init__(self, model, **options):
        self.tokenizer = None
        self._tokenizer_source = None
        self._next_request_id = 0
        self.engine = StageEngine(model, **options)


def tiny_ouro_model(device="cpu", dtype=torch.float32):
    torch.manual_seed(123)
    return OuroForCausalLM(tiny_ouro_config()).to(device=device, dtype=dtype)


def make_cpu_cache(config, *, blocks=64):
    return KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=blocks,
        block_size=2,
        max_loops=config.total_ut_steps,
        device="cpu",
        dtype=torch.float32,
    )


def make_prefill_engine(*, atomic=False, **kwargs):
    kwargs.setdefault("cache_config", CacheConfig(64, 2))
    kwargs.setdefault("execution_config", ExecutionConfig(atomic_prefill=atomic))
    kwargs.setdefault(
        "scheduler_config",
        SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    return StageEngine(tiny_ouro_model(), **kwargs)


def make_stage_engine(*, role="prefill_decode", atomic=False, asynchronous=False, mixed=False):
    torch.manual_seed(123)
    return StageEngine(
        OuroForCausalLM(tiny_ouro_config()),
        stage_config=StageConfig(role=role),
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(
            mode="mixed" if mixed else "refill",
            max_num_seqs=3,
            max_num_batched_tokens=4,
            prefill_chunk_size=3,
        ),
        execution_config=ExecutionConfig(
            atomic_prefill=atomic,
            async_scheduling=asynchronous,
            static_buffers=True,
        ),
        exit_config=ExitConfig("ouro_delayed" if asynchronous else "ouro"),
    )


class RecordingConnector:
    def __init__(self):
        self.blocked = set()
        self.submissions = []

    def exports(self, request):
        return True

    def can_schedule(self, request):
        return request.request_id not in self.blocked

    def prefill_submitted(self, request, **submission):
        self.submissions.append((request, submission))


def make_speculative_engine(
    model, *, asynchronous=True, k=3, budget=16, prefix=False, multi_stream=True
):
    return StageEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(256, 2, enable_prefix_caching=prefix, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=budget, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(async_scheduling=asynchronous, multi_stream=multi_stream),
        speculative_config=SpeculativeConfig(k, draft_policy="fixed"),
    )


def bootstrap_speculation(engine, rid="r", max_tokens=24, ignore_eos=True):
    engine.add_request(rid, [2, 3, 4], SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos))
    for _ in range(30):
        if engine.step():
            return engine.scheduler.requests[rid]
    pytest.fail("bootstrap did not produce a token")
