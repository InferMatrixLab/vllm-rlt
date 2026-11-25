"""Ouro's model code built and run on CPU with vLLM's CPU attention backend.

This is the nearest thing to a GPU run that is available without a free GPU. It
checks what does not depend on CUDA: layer construction, parameter names against
the checkpoint, one KV layer per (layer, pass), the loop context, hidden-state
shapes, and that a mixed-depth step writes each token's KV into its own pass
only. It does not check kernels, graphs, the runner or the worker.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

from vllm.config import (  # noqa: E402
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)

from vllm_rlt.plugin.context import LoopContext, TokenState, masked_slots, state_cat, state_slice  # noqa: E402  # noqa: E402
from vllm_rlt.plugin.ouro_config import OuroConfig, register_ouro_config  # noqa: E402
from vllm_rlt.plugin.program import Op  # noqa: E402

TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=8,
    max_position_embeddings=128,
    total_ut_steps=3,
)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("ouro")
    cfg = OuroConfig(**TINY)
    cfg.architectures = ["OuroForCausalLM"]
    cfg.save_pretrained(path)
    (path / "config.json").write_text(
        json.dumps({**json.loads((path / "config.json").read_text()), "model_type": "ouro"})
    )
    return path


@pytest.fixture(scope="module")
def vllm_config(tiny_dir, tmp_path_factory):
    """A one-rank CPU vLLM config with parallel groups initialised, as vLLM's own
    tests set them up (gloo, because the platform's NCCL needs a GPU)."""
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_rlt.plugin import register

    register()  # what vLLM's plugin loader does before it builds a ModelConfig
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tiny_dir),
            tokenizer=str(tiny_dir),
            skip_tokenizer_init=True,
            dtype="float32",
            seed=0,
            trust_remote_code=False,
            enforce_eager=True,
        ),
        cache_config=CacheConfig(block_size=16),
        device_config=DeviceConfig("cpu"),
    )
    store = tmp_path_factory.mktemp("dist") / "store"
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{store}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
    yield config
    cleanup_dist_env_and_memory()


def test_config_registers_and_roundtrips(tiny_dir):
    register_ouro_config()
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(tiny_dir)
    assert isinstance(cfg, OuroConfig) and cfg.total_ut_steps == 3


@pytest.fixture(scope="module")
def model(vllm_config):
    from vllm_rlt.plugin.ouro import OuroLoopForCausalLM

    with set_current_vllm_config(vllm_config):
        return OuroLoopForCausalLM(vllm_config=vllm_config)


def test_model_builds_with_one_attention_per_layer_and_pass(model):
    grid = model.loop_attention()
    assert len(grid) == 3 and all(len(row) == 2 for row in grid)
    names = {a.layer_name for row in grid for a in row}
    assert len(names) == 6, "every (layer, pass) needs its own KV layer name"
    assert all(".attn.pass" in n for n in names)


def test_parameter_names_match_ouro_checkpoint_after_stacking(model):
    params = dict(model.named_parameters())
    for name in (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.early_exit_gate.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm_2.weight",
        "model.layers.1.post_attention_layernorm_2.weight",
        "lm_head.weight",
    ):
        assert name in params, name
    # passes share weights: no per-pass parameters
    assert not any("pass" in n for n in params)


def test_checkpoint_names_map_onto_model_parameters(model):
    cfg = model.config
    ckpt = {
        "model.embed_tokens.weight": (cfg.vocab_size, cfg.hidden_size),
        "model.norm.weight": (cfg.hidden_size,),
        "model.early_exit_gate.weight": (1, cfg.hidden_size),
        "model.early_exit_gate.bias": (1,),
        "lm_head.weight": (cfg.vocab_size, cfg.hidden_size),
    }
    q = cfg.num_attention_heads * cfg.head_dim
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        for proj, n in (("q", q), ("k", q), ("v", q)):
            ckpt[f"{p}.self_attn.{proj}_proj.weight"] = (n, cfg.hidden_size)
        ckpt[f"{p}.self_attn.o_proj.weight"] = (cfg.hidden_size, q)
        for proj in ("gate", "up"):
            ckpt[f"{p}.mlp.{proj}_proj.weight"] = (cfg.intermediate_size, cfg.hidden_size)
        ckpt[f"{p}.mlp.down_proj.weight"] = (cfg.hidden_size, cfg.intermediate_size)
        for norm in (
            "input_layernorm",
            "input_layernorm_2",
            "post_attention_layernorm",
            "post_attention_layernorm_2",
        ):
            ckpt[f"{p}.{norm}.weight"] = (cfg.hidden_size,)
    torch.manual_seed(0)
    weights = [(k, torch.randn(*shape)) for k, shape in ckpt.items()]
    loaded = model.load_weights(weights)
    missing = set(dict(model.named_parameters())) - loaded
    assert not missing, sorted(missing)[:5]


def test_state_helpers_roundtrip_both_state_kinds():
    t = torch.arange(10.0).reshape(5, 2)
    assert torch.equal(state_cat([state_slice(t, 0, 2), state_slice(t, 2, 3)]), t)
    ts = TokenState(
        {"a": torch.arange(6.0).reshape(3, 2), "b": torch.arange(12.0).reshape(2, 3, 2)},
        {"a": 0, "b": 1},
    )
    joined = state_cat([state_slice(ts, 0, 1), state_slice(ts, 1, 2)])
    assert torch.equal(joined["a"], ts["a"]) and torch.equal(joined["b"], ts["b"])


def test_masked_slots_only_writes_the_rows_at_that_depth():
    from vllm.forward_context import ForwardContext, override_forward_context

    slots = torch.arange(100, 106)
    row_depth = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.int32)
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={"L": slots})
    with override_forward_context(ctx):
        with masked_slots("L", row_depth, 1):
            assert ctx.slot_mapping["L"].tolist() == [-1, 101, 102, -1, -1, -1]
        with masked_slots("L", row_depth, 2):
            assert ctx.slot_mapping["L"].tolist() == [-1, -1, -1, 103, -1, 105]
        assert torch.equal(ctx.slot_mapping["L"], slots), "restored afterwards"
        with masked_slots("missing", row_depth, 0):
            pass  # unknown layer: no-op


def test_loop_context_flags():
    assert LoopContext(Op.CORE, -1).mixed and not LoopContext(Op.CORE, 2).mixed
