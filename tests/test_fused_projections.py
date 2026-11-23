# SPDX-License-Identifier: Apache-2.0
"""Exercise Ouro's packed weights through checkpoint and GPU execution paths."""

import pickle

import pytest
import torch
from torch.nn import functional as F

from tests.helpers import tiny_ouro_config
from vllm_rlt.models.ouro import OuroForCausalLM, OuroMLP


def test_reload_legacy_weights_updates_a_previously_used_mlp():
    model = OuroForCausalLM(tiny_ouro_config())
    mlp = model.model.layers[0].mlp
    hidden = torch.ones(4, model.config.hidden_size)
    before = mlp(hidden)
    assert torch.count_nonzero(before)

    original = mlp.state_dict()["up_proj.weight"].clone()
    missing, unexpected = mlp.load_state_dict(
        {"gate_proj.weight": torch.zeros_like(original)}, strict=False
    )
    assert set(missing) == {"up_proj.weight", "down_proj.weight"}
    assert not unexpected
    assert torch.equal(mlp.state_dict()["up_proj.weight"], original)
    torch.testing.assert_close(mlp(hidden), torch.zeros_like(before), atol=0, rtol=0)


@pytest.mark.parametrize("roundtrip", ["convert", "pickle", "load", "assign"])
def test_checkpoint_views_share_one_weight_after_roundtrip(roundtrip):
    config = tiny_ouro_config(num_key_value_heads=1)
    original = OuroForCausalLM(config)
    checkpoint = original.state_dict()
    if roundtrip == "convert":
        model = original.bfloat16()
    elif roundtrip == "pickle":
        model = pickle.loads(pickle.dumps(original))
    elif roundtrip == "load":
        model = OuroForCausalLM(config)
        model.load_state_dict(checkpoint)
    else:
        with torch.device("meta"):
            model = OuroForCausalLM(config)
        model.load_state_dict(checkpoint, assign=True)

    weights = model.state_dict()
    assert weights.keys() == checkpoint.keys()
    for name, value in weights.items():
        assert torch.equal(value, checkpoint[name].to(dtype=value.dtype))
    for layer in model.model.layers:
        for module, names in (
            (layer.self_attn, ("q_proj", "k_proj", "v_proj")),
            (layer.mlp, ("gate_proj", "up_proj")),
        ):
            expected_offset = 0
            for name in names:
                part = module.state_dict()[f"{name}.weight"]
                assert part.untyped_storage().data_ptr() == module.input_weight.data_ptr()
                assert part.storage_offset() == expected_offset
                expected_offset += part.numel()
            assert expected_offset == module.input_weight.numel()
            assert module.input_weight.untyped_storage().nbytes() == (
                expected_offset * module.input_weight.element_size()
            )


@pytest.mark.parametrize("corruption", ["missing", "shape", "internal"])
def test_strict_load_validates_legacy_fused_keys(corruption):
    model = OuroForCausalLM(tiny_ouro_config())
    checkpoint = model.state_dict()
    key = "model.layers.0.self_attn.k_proj.weight"
    if corruption == "missing":
        del checkpoint[key]
    elif corruption == "shape":
        checkpoint[key] = checkpoint[key][:-1]
    else:
        key = "model.layers.0.self_attn.input_weight"
        checkpoint[key] = model.model.layers[0].self_attn.input_weight
    with pytest.raises(RuntimeError, match=key):
        model.load_state_dict(checkpoint)


@pytest.mark.gpu
def test_mlp_output_is_bitwise_identical_to_the_unfused_expression():
    from vllm_rlt.kernels._extension import load_extension

    load_extension()
    torch.manual_seed(1)
    config = tiny_ouro_config(hidden_size=64, intermediate_size=128)
    mlp = OuroMLP(config).to("cuda", torch.bfloat16).eval()
    hidden = torch.randn(4, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    params = mlp.state_dict()
    with torch.no_grad():
        expected = mlp.down_proj(
            F.silu(F.linear(hidden, params["gate_proj.weight"]))
            * F.linear(hidden, params["up_proj.weight"])
        )
        got = mlp(hidden)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.gpu
def test_attention_split_matches_separate_projections_within_bf16_rounding():
    torch.manual_seed(2)
    config = tiny_ouro_config(hidden_size=64, num_key_value_heads=1)
    model = OuroForCausalLM(config).to("cuda", torch.bfloat16)
    attn = model.model.layers[0].self_attn
    hidden = torch.randn(4, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    params = attn.state_dict()
    with torch.no_grad():
        parts = attn.project(hidden)
        separate = tuple(F.linear(hidden, params[f"{name}_proj.weight"]) for name in "qkv")
    for got, exp in zip(parts, separate):
        diff = (got.float() - exp.float()).abs().max().item()
        scale = max(exp.float().abs().max().item(), 1e-9)
        assert diff / scale < 0.0078, f"relative diff {diff / scale:.5f} exceeds 1 BF16 ULP"
