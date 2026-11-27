"""CPU checks for the RLT service adapter and the release exit hooks."""

import json
import subprocess
import sys
from io import StringIO
from types import SimpleNamespace

import pytest
import torch

from benchmarks.gsm8k import fixed_exit

ADAPTIVE_EXIT = {
    "mode": "ouro",
    "threshold": 0.5,
    "min_loops": 1,
    "max_loops": 4,
    "async_scheduling": False,
}


def test_backend_helpers_import_without_registering_unrelated_lm_eval_models():
    pytest.importorskip("lm_eval")
    script = """
import sys
from pathlib import Path
import lm_eval
from benchmarks import gsm8k_backends as backends
assert 'lm_eval.models' not in sys.modules
expected = Path(lm_eval.__file__).parent / 'models' / 'utils.py'
for helper in (backends.stop_sequences_criteria, backends.postprocess_generated_text):
    assert Path(helper.__code__.co_filename) == expected
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, timeout=60)


@pytest.fixture
def generator(monkeypatch):
    pytest.importorskip("lm_eval")
    from benchmarks import gsm8k_backends as backends

    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(backends.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(backends, "stop_sequences_criteria", lambda *args: None)
    generator = backends.Generator.__new__(backends.Generator)
    generator.backend = "transformers"
    generator.tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda ids, **kw: "answer STOP")
    return generator


@pytest.mark.parametrize(
    "policy,depths,valid",
    [
        (fixed_exit(4), [4, 4], True),
        (fixed_exit(4), [4, 2], False),
        (ADAPTIVE_EXIT, [4, 2], True),
        (ADAPTIVE_EXIT, [4], False),
        (ADAPTIVE_EXIT, [1, 2], False),
    ],
)
def test_plugin_matches_committed_depths(monkeypatch, tmp_path, policy, depths, valid):
    pytest.importorskip("lm_eval")
    from benchmarks import gsm8k_backends as backends

    log = tmp_path / "server.log"
    log.write_text("rlt commit request=cmpl-own-0-stale position=2 depth=1\n")

    def complete(request, *, timeout):
        assert request.full_url == "http://localhost:8000/v1/completions"
        assert json.loads(request.data) == {
            "model": "ouro",
            "prompt": [1, 2],
            "max_tokens": 8,
            "temperature": 0,
            "return_token_ids": True,
            "include_stop_str_in_output": True,
            "stop": ["STOP"],
            "vllm_xargs": {
                "rlt_min_loops": policy["min_loops"],
                "rlt_max_loops": 4,
                "rlt_exit_threshold": policy["threshold"],
            },
        }
        with log.open("a") as trace:
            trace.write("rlt commit request=cmpl-other-0 position=2 depth=2\n")
            for position, step in enumerate(depths, 2):
                trace.write(
                    f"rlt commit request=cmpl-own-0-suffix position={position} depth={step}\n"
                )
        return StringIO(json.dumps({"id": "cmpl-own", "choices": [{"token_ids": [10, 11]}]}))

    monkeypatch.setattr(backends, "urlopen", complete)
    plugin = backends.PluginCompletions("http://localhost:8000", "ouro", log)
    if valid:
        assert plugin.generate([1, 2], 8, ["STOP", "<EOS>"], policy, "<EOS>") == ([10, 11], depths)
    else:
        with pytest.raises(RuntimeError, match="Committed RLT depths|full-depth"):
            plugin.generate([1, 2], 8, ["STOP", "<EOS>"], policy, "<EOS>")


def test_model_depth_must_match_the_protocol():
    pytest.importorskip("lm_eval")
    from benchmarks.gsm8k_backends import check_depth

    check_depth(SimpleNamespace(total_ut_steps=4), 4)
    with pytest.raises(ValueError, match="6 loops but the protocol expects 4"):
        check_depth(SimpleNamespace(total_ut_steps=6), 4)


class ReleaseModel:
    config = SimpleNamespace(num_hidden_layers=1, total_ut_steps=4)

    def __init__(self, depths=None, recorded=(4, 2)):
        self.kwargs = None
        self.depths = depths
        self.recorded = recorded

    def generate(self, inputs, **kwargs):
        self.kwargs = kwargs
        if self.depths is not None:
            # Stands in for ReleaseExitHooks, which records one depth per forward.
            self.depths.depths.extend(self.recorded)
        return torch.cat([inputs.cpu(), torch.tensor([[10, 11]])], dim=1)


@pytest.mark.parametrize("loops", [4, 6])
@pytest.mark.parametrize("adaptive", [False, True])
def test_transformers_exit_arguments(generator, monkeypatch, adaptive, loops):
    from benchmarks import gsm8k_backends as backends

    class Cache:
        def __init__(self):
            self.layers = [None]

        def append_new_layers(self, index):
            self.layers.extend([None] * index)

    monkeypatch.setattr(backends, "DynamicCache", Cache)
    generator.backend = "transformers"
    generator.exit_hooks = SimpleNamespace(depths=["stale"]) if adaptive else None
    generator.model = ReleaseModel(generator.exit_hooks)
    generator.cache_slots = 4
    generator.exit = {**ADAPTIVE_EXIT, "max_loops": loops} if adaptive else fixed_exit(loops)
    result = generator.generate([1, 2], 8, ["STOP"])
    kwargs = generator.model.kwargs
    if adaptive:
        assert kwargs["exit_threshold"] == 0.5 and "exit_at_step" not in kwargs
    else:
        # The last loop of the protocol's depth: 3 for the four-loop checkpoint.
        assert kwargs["exit_at_step"] == loops - 1 and "exit_threshold" not in kwargs
    assert result["token_ids"] == [10, 11]
    if adaptive:
        # Depths from the previous question are cleared before generation.
        assert result["exit_depths"] == [4, 2]
    else:
        assert "exit_depths" not in result


def test_transformers_depth_count_must_match_tokens(generator, monkeypatch):
    from benchmarks import gsm8k_backends as backends

    class Cache:
        layers = [None] * 4

        def append_new_layers(self, index):
            pass

    monkeypatch.setattr(backends, "DynamicCache", Cache)
    generator.backend = "transformers"
    generator.exit_hooks = SimpleNamespace(depths=[])
    generator.model = ReleaseModel(generator.exit_hooks, recorded=(4,))
    generator.cache_slots = 4
    generator.exit = ADAPTIVE_EXIT
    with pytest.raises(RuntimeError, match="exit depths"):
        generator.generate([1, 2], 8, ["STOP"])


class GateModel(torch.nn.Module):
    """Minimal stand-in for the release: the inner model returns (outputs, hidden, gates)."""

    config = SimpleNamespace(total_ut_steps=4)

    def __init__(self, hazards):
        super().__init__()
        logits = torch.logit(torch.tensor(hazards, dtype=torch.float32))

        class Inner(torch.nn.Module):
            def forward(self, length):
                return None, None, [value.expand(1, length, 1) for value in logits]

        self.model = Inner()
        self.exit_at_step = []

    def forward(self, length, cache_position=None, exit_at_step=None):
        self.exit_at_step.append(exit_at_step)
        return self.model(length)


@pytest.mark.parametrize("threshold,depth", [(0.05, 1), (0.2, 2), (0.5, 3), (0.9, 4), (1.0, 4)])
def test_release_exit_hooks(threshold, depth):
    pytest.importorskip("lm_eval")
    from benchmarks.gsm8k_backends import ReleaseExitHooks

    # Hazards 0.1, 0.3, 0.5 give cumulative exit probabilities 0.1, 0.37, 0.685, then 1.
    model = GateModel([0.1, 0.3, 0.5, 0.9])
    hooks = ReleaseExitHooks(model, threshold)
    model(3, cache_position=torch.tensor([0, 1, 2]))
    model(1, cache_position=torch.tensor([3]))
    # Prefill is forced to full depth like native; decode follows the release's rule.
    assert model.exit_at_step == [3, None]
    assert hooks.depths == [4, depth]
