"""CPU numerical checks against an independent, dense Ouro implementation."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.helpers import make_cpu_cache, tiny_ouro_config
from tests.reference import dense_reference
from vllm_rlt.models import OuroConfig, OuroForCausalLM


@pytest.fixture
def model():
    torch.manual_seed(71)
    return OuroForCausalLM(tiny_ouro_config())


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_packed_prefill_matches_dense_at_every_depth(kv_heads):
    torch.manual_seed(8)
    config = tiny_ouro_config(num_key_value_heads=kv_heads)
    model = OuroForCausalLM(config)
    tokens = torch.tensor([5, 7, 2, 11, 3])
    expected = dense_reference(model, tokens, config.total_ut_steps)
    cache = make_cpu_cache(config)
    assert cache.allocate("prompt", len(tokens))
    hidden = model.prelude(tokens)
    for depth, (expected_hidden, expected_gate, expected_logits) in enumerate(expected):
        hidden, gate = model.recurrent(
            hidden, ["prompt"] * len(tokens), [depth] * len(tokens), list(range(len(tokens))), cache
        )
        torch.testing.assert_close(hidden, expected_hidden, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(gate, expected_gate, atol=2e-6, rtol=3e-5)
        torch.testing.assert_close(model.coda(hidden), expected_logits, atol=2e-6, rtol=3e-5)


def test_incremental_tokens_match_dense_fixed_depth(model):
    tokens = torch.tensor([4, 15, 6, 2, 9, 12, 3])
    expected = dense_reference(model, tokens, model.config.total_ut_steps)
    cache = make_cpu_cache(model.config)
    assert cache.allocate("sequence", len(tokens))
    for position, token in enumerate(tokens):
        hidden = model.prelude(token.unsqueeze(0))
        for depth in range(model.config.total_ut_steps):
            hidden, gate = model.recurrent(hidden, ["sequence"], [depth], [position], cache)
            expected_hidden, expected_gate, expected_logits = expected[depth]
            torch.testing.assert_close(hidden[0], expected_hidden[position], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(gate[0], expected_gate[position], atol=2e-6, rtol=3e-5)
            torch.testing.assert_close(
                model.coda(hidden)[0], expected_logits[position], atol=2e-6, rtol=3e-5
            )


def test_mixed_request_depth_batch_matches_separate_dense(model):
    first = torch.tensor([5, 8, 12])
    second = torch.tensor([6, 13, 2, 4])
    expected_first = dense_reference(model, first, 3)[2]
    expected_second = dense_reference(model, second, 1)[0]
    cache = make_cpu_cache(model.config)
    assert cache.allocate("first", len(first))
    assert cache.allocate("second", len(second))
    first_hidden = model.prelude(first)
    for depth in range(2):
        first_hidden, _ = model.recurrent(
            first_hidden, ["first"] * 3, [depth] * 3, [0, 1, 2], cache
        )
    hidden, gate = model.recurrent(
        torch.cat([first_hidden, model.prelude(second)]),
        ["first"] * 3 + ["second"] * 4,
        [2] * 3 + [0] * 4,
        [0, 1, 2, 0, 1, 2, 3],
        cache,
    )
    torch.testing.assert_close(
        hidden, torch.cat([expected_first[0], expected_second[0]]), atol=3e-6, rtol=3e-5
    )
    torch.testing.assert_close(
        gate, torch.cat([expected_first[1], expected_second[1]]), atol=2e-6, rtol=3e-5
    )


@pytest.mark.parametrize("sharded", [False, True])
def test_native_safetensors_roundtrip(tmp_path, model, sharded):
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    weights = model.state_dict()
    if sharded:
        names = list(weights)
        halves = [names[::2], names[1::2]]
        weight_map = {}
        for index, half in enumerate(halves):
            filename = f"model-{index:05d}-of-00002.safetensors"
            save_file({name: weights[name] for name in half}, tmp_path / filename)
            weight_map.update({name: filename for name in half})
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    else:
        save_file(weights, tmp_path / "model.safetensors")
    loaded = OuroForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert not loaded.training
    assert not any(
        parameter.is_meta or parameter.requires_grad for parameter in loaded.parameters()
    )
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, weights[name], atol=0, rtol=0)
    torch.testing.assert_close(loaded.model.rotary_emb.inv_freq, model.model.rotary_emb.inv_freq)
    tokens = torch.tensor([5, 8, 12])
    cache = make_cpu_cache(model.config)
    cache.allocate("loaded", len(tokens))
    hidden = loaded.prelude(tokens)
    for depth in range(2):
        hidden, _ = loaded.recurrent(hidden, ["loaded"] * 3, [depth] * 3, [0, 1, 2], cache)
    torch.testing.assert_close(
        loaded.coda(hidden), dense_reference(model, tokens, 2)[1][2], atol=2e-6, rtol=3e-5
    )


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_checkpoint_mismatch_is_rejected(tmp_path, model, corruption):
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    weights = model.state_dict()
    if corruption == "missing":
        weights.pop("model.early_exit_gate.bias")
    elif corruption == "unexpected":
        weights["model.unexpected.weight"] = torch.ones(1)
    else:
        weights["model.norm.weight"] = torch.ones(1)
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(
        ValueError,
        match={"missing": "Missing", "unexpected": "Unexpected", "shape": "shape mismatch"}[
            corruption
        ],
    ):
        OuroForCausalLM.from_pretrained(tmp_path)


def test_official_repo_is_pinned_and_remote_code_is_not_requested(tmp_path, model, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    called = {}

    def snapshot_download(**kwargs):
        called.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(tmp_path / "config.json")
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    loaded = OuroForCausalLM.from_pretrained("test-org/checkpoint", revision="test-revision")
    for name, parameter in loaded.named_parameters():
        torch.testing.assert_close(parameter, model.state_dict()[name].bfloat16(), atol=0, rtol=0)
    assert loaded.model.rotary_emb.inv_freq.dtype == torch.float32
    assert called["revision"] == "test-revision"
    assert called["repo_id"] == "test-org/checkpoint"
    assert not any(".py" in pattern for pattern in called["allow_patterns"])


def test_caller_revision_is_consistently_used_without_override(tmp_path, model, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    called = {}

    def snapshot_download(**kwargs):
        called.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(tmp_path / "config.json")
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    OuroForCausalLM.from_pretrained("test-org/checkpoint", revision=None)
    assert called["revision"] is None
    assert called["repo_id"] == "test-org/checkpoint"

    OuroForCausalLM.from_pretrained("test-org/checkpoint", revision="custom-branch")
    assert called["revision"] == "custom-branch"


@pytest.mark.parametrize(
    "overrides",
    [
        {"rope_scaling": {"rope_type": "linear", "factor": 2}},
        {"use_sliding_window": True},
        {"hidden_act": "gelu"},
        {"layer_types": ["full_attention", "sliding_attention"]},
        {"num_key_value_heads": 3},
        {"head_dim": 7},
        {"tie_word_embeddings": True},
        {"eos_token_id": 1000},
    ],
)
def test_unsupported_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        tiny_ouro_config(**overrides)


def test_config_roundtrip_and_unrecognized_architecture():
    config = tiny_ouro_config()
    assert OuroConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="model_type"):
        OuroConfig.from_dict({"model_type": "llama"})
    with pytest.raises(ValueError, match="Unsupported"):
        OuroConfig.from_dict({"attention_bias": True})


def test_dtype_conversion_preserves_float32_rotary_frequencies(model):
    expected_frequencies = model.model.rotary_emb.inv_freq.clone()
    model.bfloat16()
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    torch.testing.assert_close(
        model.model.rotary_emb.inv_freq, expected_frequencies, atol=0, rtol=0
    )
    positions = torch.tensor([32767, 32768])
    hidden = model.prelude(torch.tensor([3, 4]))
    cos, sin = model.model.rotary_emb(hidden, positions)
    angles = positions.float()[:, None] * expected_frequencies
    angles = torch.cat([angles, angles], dim=-1)
    torch.testing.assert_close(cos[:, 0], angles.cos().bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(sin[:, 0], angles.sin().bfloat16(), atol=0, rtol=0)
