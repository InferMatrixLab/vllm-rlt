# SPDX-License-Identifier: Apache-2.0
"""Fusing projections that share an input row must not change what they compute.

The fused weight is a concatenation of the existing parameters, so an exact comparison
against the unfused call is the right gate for the values: any real bug shows up as a
nonzero difference, not as a rounding-level one.

The remaining tests exist because each one caught a real defect in an earlier version of
the cache: a cyclic module graph that broke ``state_dict``, a weakref that broke pickling,
and a cached tensor registered as a parameter that broke checkpoint round-trips.
"""

import pickle

import pytest
import torch

from vllm_rlt.layers.fused_projections import FusedWeight
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM, OuroMLP

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


class _Triple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 32, bias=False)
        self.b = torch.nn.Linear(64, 16, bias=False)
        self.c = torch.nn.Linear(64, 16, bias=False)
        self.fused = FusedWeight(self, ("a", "b", "c"))


def test_fused_weight_is_the_concatenation_in_source_order():
    layer = _Triple()
    fused = layer.fused.weight()
    assert fused.shape == (64, 64)
    assert torch.equal(fused[:32], layer.a.weight)
    assert torch.equal(fused[32:48], layer.b.weight)
    assert torch.equal(fused[48:], layer.c.weight)


def test_split_matches_separate_projections_exactly():
    torch.manual_seed(0)
    layer = _Triple()
    hidden = torch.randn(3, 64)
    expected = (layer.a(hidden), layer.b(hidden), layer.c(hidden))
    fused = layer.fused(hidden)
    got = (fused[:, :32], fused[:, 32:48], fused[:, 48:])
    for g, e in zip(got, expected):
        assert torch.equal(g, e)


def test_fused_weight_rebuilds_after_parameter_replacement():
    """Checkpoint loading replaces parameters; a stale cache would silently keep the old
    weights, so the signature has to notice."""
    layer = _Triple()
    first = layer.fused.weight()
    with torch.no_grad():
        layer.a.weight = torch.nn.Parameter(torch.zeros_like(layer.a.weight), requires_grad=False)
    second = layer.fused.weight()
    assert second.data_ptr() != first.data_ptr()
    assert torch.equal(second[:32], layer.a.weight)


def test_fused_weight_follows_dtype_conversion():
    layer = _Triple()
    assert layer.fused.weight().dtype == torch.float32
    layer.half()
    fused = layer.fused.weight()
    assert fused.dtype == torch.float16
    assert torch.equal(fused[32:48], layer.b.weight)


def test_fused_weight_is_not_registered_in_state_dict():
    """A registered Parameter would add an unexpected key and break checkpoint round-trips."""
    layer = _Triple()
    layer.fused.weight()  # populate the cache
    keys = set(layer.state_dict())
    assert keys == {"a.weight", "b.weight", "c.weight"}, sorted(keys)
    assert all(id(p) != id(layer.fused.weight()) for p in layer.parameters())


def test_model_with_fused_projections_is_picklable():
    """The engine ships the model to a worker process."""
    layer = _Triple()
    layer.fused.weight()
    restored = pickle.loads(pickle.dumps(layer))
    assert torch.equal(restored.fused.weight(), layer.fused.weight())


def test_state_dict_is_unchanged_by_fusion():
    """The fused matrices are derived, so they must not appear as checkpoint keys."""
    config = OuroConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=128,
        total_ut_steps=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = OuroForCausalLM(config)
    keys = set(model.state_dict())
    assert not any("qkv_proj" in k or "gate_up_proj" in k for k in keys), sorted(keys)
    expected = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    }
    assert expected <= keys


@requires_cuda
def test_mlp_output_is_bitwise_identical_to_the_unfused_expression():
    torch.manual_seed(1)
    config = OuroConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=128,
        total_ut_steps=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    mlp = OuroMLP(config).to("cuda", torch.bfloat16).eval()
    hidden = torch.randn(4, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = mlp.down_proj(
            torch.nn.functional.silu(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
        )
        got = mlp(hidden)
    assert torch.equal(got, expected), (
        f"max diff {(got.float() - expected.float()).abs().max().item()}"
    )


@requires_cuda
def test_attention_split_matches_separate_projections_within_bf16_rounding():
    """cuBLAS picks a different kernel for N=6144 than for N=2048, so the fused qkv is not
    bitwise identical. The bound below is fraction-of-a-BF16-ULP, measured at 0.41% max
    over all 72 Ouro projections; anything larger would mean a real ordering bug."""
    torch.manual_seed(2)
    config = OuroConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=128,
        total_ut_steps=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = OuroForCausalLM(config).to("cuda", torch.bfloat16).eval()
    attn = model.model.layers[0].self_attn
    hidden = torch.randn(4, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        fused = attn.qkv_proj(hidden)
        widths = [attn.q_proj.out_features, attn.k_proj.out_features, attn.v_proj.out_features]
        parts = fused.split(widths, dim=-1)
        separate = (attn.q_proj(hidden), attn.k_proj(hidden), attn.v_proj(hidden))
    for got, exp in zip(parts, separate):
        diff = (got.float() - exp.float()).abs().max().item()
        scale = max(exp.float().abs().max().item(), 1e-9)
        assert diff / scale < 0.0078, f"relative diff {diff / scale:.5f} exceeds 1 BF16 ULP"
