# SPDX-License-Identifier: Apache-2.0
"""Dory numerical, configuration, checkpoint and scheduling regressions."""

import json
import math

import pytest
import torch

from tests.reference.dory import dense_dory_reference
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.attention import BackendCapabilities
from vllm_rlt.attention.backend import AttentionBackend
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.models import AutoModelForCausalLM, DoryConfig, DoryForCausalLM

ARTIFACT_PAIRS = 36


def artifact_layer_types():
    """Dory-1.0-4B layout: 27 sliding and 9 full attention sublayers."""
    names = []
    for pair in range(ARTIFACT_PAIRS):
        names.append("full_attention" if pair % 4 == 3 else "sliding_attention")
        names.append("mlp")
    return names


def artifact_config_dict():
    return {
        "architectures": ["DoryForCausalLM"],
        "model_type": "dory",
        "vocab_size": 131072,
        "hidden_size": 2560,
        "num_hidden_layers": 72,
        "intermediate_size": 7168,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "ngpt_alpha_init": 0.05,
        "attention_output_gate_logit_scale_init": None,
        "mlp_bias": False,
        "attention_bias": False,
        "n_input_layers": 16,
        "n_recurrent_layers": 40,
        "n_output_layers": 16,
        "n_recurrent_loops": 2,
        "recurrent_kv_cache_mode": "per_loop",
        "head_dim": 256,
        "hidden_act": "silu",
        "initializer_range": 0.0177,
        "dtype": "bfloat16",
        "sliding_window": 1024,
        "rope_parameters": {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
            },
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 1.0,
            },
        },
        "layer_types": artifact_layer_types(),
    }


def tiny_dory_config(**overrides):
    values = dict(
        vocab_size=64,
        d_model=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        initializer_range=0.02,
        ngpt_alpha_init=0.05,
        n_input_layers=2,
        n_recurrent_layers=4,
        n_output_layers=2,
        n_recurrent_loops=2,
        sliding_window=4,
        layer_types=(
            "sliding_attention",
            "mlp",
            "full_attention",
            "mlp",
            "sliding_attention",
            "mlp",
            "sliding_attention",
            "mlp",
        ),
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    values.update(overrides)
    return DoryConfig(**values)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return DoryForCausalLM(tiny_dory_config()).eval()


def make_cache(model, layout=None, block_size=2):
    config = model.config
    return KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=256,
        block_size=block_size,
        max_loops=config.total_ut_steps,
        device="cpu",
        dtype=next(model.parameters()).dtype,
        backend="torch",
        layout=layout or config.kv_cache_layout,
        recurrent_layers=model.recurrent_kv_layers,
    )


def run_one_token_per_step(model, tokens):
    """Execute the declared reference schedule: one token row at a time."""
    config = model.config
    cache = make_cache(model)
    assert cache.allocate("request", len(tokens))
    loops = [[] for _ in range(config.total_ut_steps)]
    logits = []
    for position in range(len(tokens)):
        boundary = cache._prepare_batch(["request"], [0], [position])
        hidden = model.prelude_prepared(tokens[position : position + 1], boundary, cache)
        for depth in range(config.total_ut_steps):
            core = cache._prepare_batch(["request"], [depth], [position])
            hidden, gate = model.recurrent_prepared(hidden, core, cache)
            assert gate is not None and bool((gate < -1e3).all())
            loops[depth].append(hidden[0, : config.d_model])
        logits.append(model.coda_prepared(hidden, boundary, cache)[0])
        cache.finalize_token("request", position, config.total_ut_steps - 1)
    return {"logits": torch.stack(logits), "loops": [torch.stack(item) for item in loops]}, cache


def run_all_positions_at_once(model, tokens):
    """Same schedule, but every stage call carries all causal positions."""
    config = model.config
    cache = make_cache(model)
    assert cache.allocate("request", len(tokens))
    positions = list(range(len(tokens)))
    ids = ["request"] * len(tokens)
    boundary = cache._prepare_batch(ids, [0] * len(positions), positions)
    hidden = model.prelude_prepared(tokens, boundary, cache)
    loops = []
    for depth in range(config.total_ut_steps):
        core = cache._prepare_batch(ids, [depth] * len(positions), positions)
        hidden, _ = model.recurrent_prepared(hidden, core, cache)
        loops.append(hidden[:, : config.d_model])
    logits = model.coda_prepared(hidden, boundary, cache)
    return {"logits": logits, "loops": loops}, cache


def assert_matches_reference(actual, expected, atol=3e-6, rtol=3e-5):
    torch.testing.assert_close(actual["logits"], expected["logits"], atol=atol, rtol=rtol)
    for got, want in zip(actual["loops"], expected["loops"]):
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)


def test_artifact_config_maps_widths_stages_and_kv_layout():
    config = DoryConfig.from_dict(artifact_config_dict())
    assert config.d_model == 2560
    # The runner carries the retained injection input beside the state.
    assert config.hidden_size == 5120
    assert config.total_ut_steps == 2
    assert config.kv_cache_layout == "last_exited"
    assert [len(config.attention_layers(stage)) for stage in ("input", "recurrent", "output")] == [
        8,
        20,
        8,
    ]
    assert config.layer_types.count("full_attention") == 9
    assert config.layer_types.count("sliding_attention") == 27
    assert config.rope_theta("sliding_attention") == 10000
    assert config.rope_theta("full_attention") == 10000000
    assert DoryConfig.from_dict(config.to_dict()) == config


def test_last_loop_mode_uses_the_shared_plane():
    config = DoryConfig.from_dict(
        {**artifact_config_dict(), "recurrent_kv_cache_mode": "last_loop"}
    )
    assert config.kv_cache_layout == "shared"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"head_dim": 7}, "even"),
        ({"num_key_value_heads": 5}, "divisible"),
        ({"n_output_layers": 17}, "num_hidden_layers"),
        ({"mlp_bias": True}, "bias"),
        ({"tie_word_embeddings": True}, "untied"),
        ({"recurrent_kv_cache_mode": "every_second"}, "recurrent_kv_cache_mode"),
        ({"sliding_window": None}, "positive sliding_window"),
        ({"hidden_act": "gelu"}, "silu"),
        ({"ngpt_alpha_init": 0.0}, "finite and positive"),
        (
            {
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "yarn", "rope_theta": 1e4},
                    "full_attention": {"rope_type": "default", "rope_theta": 1e7},
                }
            },
            "default rotary",
        ),
        (
            {
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
                    "full_attention": {"rope_type": "default", "rope_theta": 1e7},
                }
            },
            None,
        ),
    ],
)
def test_config_rejects_unqualified_layouts(overrides, error):
    values = {**artifact_config_dict(), **overrides}
    if error is None:
        assert DoryConfig.from_dict(values).n_recurrent_loops == 2
        return
    with pytest.raises(ValueError, match=error):
        DoryConfig.from_dict(values)


def test_config_requires_explicit_head_dim_and_rejects_unknown_fields():
    values = artifact_config_dict()
    values.pop("head_dim")
    with pytest.raises(ValueError, match="explicit head_dim"):
        DoryConfig.from_dict(values)
    with pytest.raises(ValueError, match="Unsupported Dory configuration fields"):
        DoryConfig.from_dict({**artifact_config_dict(), "sink_tokens": 4})


@pytest.mark.parametrize("kv_mode", ["per_loop", "last_loop"])
def test_staged_execution_matches_independent_dense_reference(model, kv_mode):
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode=kv_mode)).eval()
    tokens = torch.tensor([4, 7, 9, 3, 12, 5])
    expected = dense_dory_reference(model, tokens)
    actual, cache = run_one_token_per_step(model, tokens)
    assert_matches_reference(actual, expected)
    cache.free("request")
    assert cache.num_used_blocks == 0


def test_batched_positions_match_reference_for_per_loop_kv(model):
    tokens = torch.tensor([4, 7, 9, 3, 12, 5])
    expected = dense_dory_reference(model, tokens)
    actual, cache = run_all_positions_at_once(model, tokens)
    assert_matches_reference(actual, expected)
    cache.free("request")
    assert cache.num_used_blocks == 0


def test_recurrent_kv_is_laid_out_per_loop_plane(model):
    tokens = torch.tensor([4, 7, 9, 3, 12, 5])
    _, cache = run_one_token_per_step(model, tokens)
    allocation = cache._get_allocation("request")
    recurrent = model.recurrent_kv_layers[0]
    boundary = model.prelude_kv_layers[0]
    for depth in range(model.config.total_ut_steps):
        assert all(0 in allocation.written[depth][recurrent] for _ in (0,))
    # Boundary stages own one history at plane 0, never one copy per loop.
    assert 0 in allocation.written[0][boundary]
    assert all(0 not in allocation.written[depth][boundary] for depth in range(1, 2))


def test_last_loop_kv_shares_one_physical_plane():
    model = DoryForCausalLM(tiny_dory_config(recurrent_kv_cache_mode="last_loop")).eval()
    tokens = torch.tensor([4, 7, 9])
    _, cache = run_one_token_per_step(model, tokens)
    allocation = cache._get_allocation("request")
    assert len(allocation.block_tables) == 1
    torch.testing.assert_close(
        dense_dory_reference(model, tokens)["logits"],
        run_one_token_per_step(model, tokens)[0]["logits"],
        atol=3e-6,
        rtol=3e-5,
    )


def test_windowed_attention_masks_keys_outside_the_window(model):
    cache = make_cache(model)
    assert cache.allocate("request", 8)
    layer = model.recurrent_kv_layers[0]
    torch.manual_seed(7)
    for position in range(8):
        cache.write(
            layer,
            ["request"],
            [0],
            [position],
            torch.randn(1, 2, 8),
            torch.randn(1, 2, 8),
        )
    query = torch.randn(1, 4, 8)
    full = cache.attend(layer, ["request"], [0], [7], query)
    windowed = cache.attend(layer, ["request"], [0], [7], query, window=3)
    assert not torch.allclose(full, windowed)
    keys, values = cache.read(layer, "request", 0, 8)
    groups = 4 // 2
    scores = torch.einsum(
        "hd,thd->ht", query[0].float(), keys[-3:].float().repeat_interleave(groups, dim=1)
    ) / math.sqrt(8)
    expected = torch.einsum(
        "ht,thd->hd",
        scores.softmax(-1),
        values[-3:].float().repeat_interleave(groups, dim=1),
    )
    torch.testing.assert_close(windowed[0], expected, atol=3e-6, rtol=1e-5)


def test_windowed_attention_refuses_backends_without_masking(model):
    stub = AttentionBackend(
        torch_paged_attention, BackendCapabilities(), {"selected_backend": "stub"}
    )
    config = model.config
    cache = KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=16,
        block_size=2,
        max_loops=config.total_ut_steps,
        device="cpu",
        dtype=next(model.parameters()).dtype,
        backend="torch",
        attention=stub,
        layout=config.kv_cache_layout,
        recurrent_layers=model.recurrent_kv_layers,
    )
    layer = model.recurrent_kv_layers[0]
    assert cache.allocate("request", 2)
    cache.write(layer, ["request"], [0], [0], torch.randn(1, 2, 8), torch.randn(1, 2, 8))
    with pytest.raises(NotImplementedError, match="sliding-window"):
        cache.attend(layer, ["request"], [0], [0], torch.randn(1, 4, 8), window=2)


@pytest.mark.parametrize("sharded", [False, True])
def test_strict_checkpoint_and_auto_dispatch(tmp_path, model, sharded):
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    weights = {name: value.clone().contiguous() for name, value in model.state_dict().items()}
    names = list(weights)
    groups = [names[: len(names) // 2], names[len(names) // 2 :]] if sharded else [names]
    mapping = {}
    for index, group in enumerate(groups):
        filename = f"model-{index}.safetensors" if sharded else "model.safetensors"
        save_file({name: weights[name] for name in group}, tmp_path / filename)
        mapping.update(dict.fromkeys(group, filename))
    if sharded:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert isinstance(loaded, DoryForCausalLM)
    assert loaded.config == model.config
    for name, expected in weights.items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)


def test_loader_rejects_unexpected_tensors(tmp_path, model):
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    weights = {name: value.clone().contiguous() for name, value in model.state_dict().items()}
    weights["backbone.layers.0.router.weight"] = torch.zeros(2, 2)
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="Unexpected checkpoint tensor"):
        DoryForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)


def test_engine_serves_fixed_loops_and_matches_greedy_reference(model):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=256, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
    )
    prompt = [4, 7, 3]
    params = SamplingParams(max_tokens=3, min_loops=2, max_loops=2, ignore_eos=True)
    engine.add_request("request", prompt, params)
    finished = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output
    assert set(finished) == {"request"}
    result = finished["request"]
    assert result.exit_depths == [model.config.total_ut_steps] * 3
    assert engine.cache_manager.num_used_blocks == 0

    tokens = list(prompt)
    for _ in range(3):
        logits = dense_dory_reference(model, torch.tensor(tokens))["logits"]
        tokens.append(int(logits[-1].argmax()))
    assert result.token_ids == tokens[len(prompt) :]


def test_engine_rejects_unqualified_dory_configuration(model):
    with pytest.raises(ValueError, match="max_loops"):
        LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=256, block_size=2),
            scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
        ).add_request("request", [4, 7, 3], SamplingParams(max_tokens=1, min_loops=1, max_loops=1))
    with pytest.raises(ValueError, match="last_exited"):
        LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=256, block_size=2, layout="shared"),
            scheduler_config=SchedulerConfig(),
        )
    with pytest.raises(ValueError, match="CUDA graphs"):
        LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=256, block_size=2),
            scheduler_config=SchedulerConfig(),
            execution_config=ExecutionConfig(cuda_graphs=True),
        )
