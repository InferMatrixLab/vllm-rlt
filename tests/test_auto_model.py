# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AutoModelForCausalLM loading flow, approval logic, and dispatch."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
from vllm_rlt.models import (
    AutoModelForCausalLM,
    NanbeigeForCausalLM,
    OuroForCausalLM,
    resolve_local_model_path,
)


@pytest.fixture
def confirm_download(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")


def test_local_dispatch_ouro(tmp_path):
    config = tiny_ouro_config()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    model = OuroForCausalLM(config)
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert isinstance(loaded, OuroForCausalLM)


def test_local_dispatch_nanbeige(tmp_path):
    config = tiny_nanbeige_config()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    model = NanbeigeForCausalLM(config)
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert isinstance(loaded, NanbeigeForCausalLM)


def test_unsupported_model_type_raises_directly(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))

    with pytest.raises(ValueError, match="Unsupported model_type 'llama'"):
        AutoModelForCausalLM.from_pretrained(tmp_path)


def test_remote_model_requires_approval_by_default(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(RuntimeError, match="Download requires approval"):
        AutoModelForCausalLM.from_pretrained("remote-org/unapproved-model")


def test_remote_approved_download_selects_by_config(tmp_path, monkeypatch, confirm_download):
    config = tiny_nanbeige_config()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    model = NanbeigeForCausalLM(config)
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    # Mock hf_hub_download returning config from tmp_path
    def mock_hf_download(repo_id, filename, revision=None):
        return str(tmp_path / filename)

    # Mock snapshot_download returning the full dir
    def mock_snapshot(repo_id, revision=None, allow_patterns=None):
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_hf_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", mock_snapshot)

    loaded = AutoModelForCausalLM.from_pretrained("custom-org/custom-model", dtype=torch.float32)
    assert isinstance(loaded, NanbeigeForCausalLM)


def test_remote_download_surfaces_errors_directly(monkeypatch, confirm_download):
    def mock_failing_download(repo_id, filename, revision=None):
        raise ConnectionError("Simulated network failure")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_failing_download)

    with pytest.raises(ConnectionError, match="Simulated network failure"):
        AutoModelForCausalLM.from_pretrained("some/remote-model")


def test_remote_download_preserves_source(tmp_path, monkeypatch, confirm_download):
    config = tiny_nanbeige_config()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    model = NanbeigeForCausalLM(config)
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    downloaded_repo = []

    def mock_hf_download(repo_id, filename, revision=None):
        downloaded_repo.append((repo_id, revision))
        return str(tmp_path / filename)

    def mock_snapshot(repo_id, revision=None, allow_patterns=None):
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_hf_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", mock_snapshot)

    loaded = AutoModelForCausalLM.from_pretrained("test-org/custom-model", dtype=torch.float32)
    assert isinstance(loaded, NanbeigeForCausalLM)
    assert downloaded_repo[0][0] == "test-org/custom-model"


def test_artifacts_basename_does_not_override_repository(tmp_path, monkeypatch):
    from vllm_rlt.models import resolve_local_model_path

    dummy = tmp_path / "artifacts/models/same-name"
    dummy.mkdir(parents=True)
    (dummy / "config.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    assert resolve_local_model_path("other-org/same-name") is None
    assert resolve_local_model_path(dummy) == dummy


@pytest.fixture
def checkpoint(tmp_path):
    model = NanbeigeForCausalLM(tiny_nanbeige_config())
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    return tmp_path


@pytest.mark.parametrize("revision", [None, "my-tag"])
@pytest.mark.parametrize("source", ["custom-org/model", "ouro", "nanbeige"])
def test_caller_revision_reaches_hub_unchanged(
    monkeypatch, checkpoint, revision, source, confirm_download
):
    fetch = Mock(return_value=str(checkpoint / "config.json"))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    snapshot = Mock(return_value=str(checkpoint))
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    load = Mock(return_value=object())
    monkeypatch.setattr(NanbeigeForCausalLM, "from_pretrained", load)
    AutoModelForCausalLM.from_pretrained(source, revision=revision)
    assert fetch.call_args.args == (source, "config.json")
    assert fetch.call_args.kwargs["revision"] == revision
    assert load.call_args.kwargs["revision"] == revision
    assert snapshot.call_args.kwargs["repo_id"] == source
    assert load.call_args.args == (str(checkpoint),)


def test_revision_specific_cached_checkpoint_needs_no_network(monkeypatch, checkpoint):
    lookup = Mock(return_value=str(checkpoint / "config.json"))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lookup)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", Mock(side_effect=AssertionError))
    monkeypatch.setattr("huggingface_hub.snapshot_download", Mock(side_effect=AssertionError))
    model = AutoModelForCausalLM.from_pretrained("my-org/model", revision="pinned")
    assert isinstance(model, NanbeigeForCausalLM)
    assert lookup.call_args.kwargs["revision"] == "pinned"
    assert lookup.call_args.args[0] == "my-org/model"


def test_partial_cache_does_not_authorize_download(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    (tmp_path / "config.json").write_text('{"model_type": "nanbeige"}')
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache", lambda *a, **k: str(tmp_path / "config.json")
    )
    assert resolve_local_model_path("my-org/model") is None
    with pytest.raises(RuntimeError, match="Download requires approval"):
        AutoModelForCausalLM.from_pretrained("my-org/model")


def test_text_generation_uses_caller_source_and_revision(monkeypatch, checkpoint, confirm_download):
    from vllm_rlt import LLM, SamplingParams

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(checkpoint / "config.json")
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: str(checkpoint))
    tokenizer = SimpleNamespace(encode=lambda text: [2, 3], decode=lambda ids, **kw: "text")
    token_load = Mock(return_value=tokenizer)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", token_load)
    llm = LLM("custom-org/checkpoint", revision="custom", dtype=torch.float32)
    try:
        outputs = llm.generate("hello", SamplingParams(max_tokens=1, ignore_eos=True))
        assert outputs[0].text == "text"
        assert token_load.call_args.args == ("custom-org/checkpoint",)
        assert token_load.call_args.kwargs["revision"] == "custom"
    finally:
        llm.close()


@pytest.mark.parametrize("answer", ["y", "yes", " YES ", "n", "", "true"])
def test_interactive_download(monkeypatch, checkpoint, answer):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    prompt = Mock(return_value=answer)
    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    fetch = Mock(return_value=str(checkpoint / "config.json"))
    snapshot = Mock(return_value=str(checkpoint))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    if answer.strip().lower() in ("y", "yes"):
        assert isinstance(AutoModelForCausalLM.from_pretrained("org/model"), NanbeigeForCausalLM)
        snapshot.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="not approved"):
            AutoModelForCausalLM.from_pretrained("org/model")
        fetch.assert_not_called()
        snapshot.assert_not_called()
    prompt.assert_called_once()


@pytest.mark.parametrize("loader", [AutoModelForCausalLM, OuroForCausalLM, NanbeigeForCausalLM])
def test_noninteractive_download_fails_without_network(monkeypatch, loader):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    fetch = Mock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    with pytest.raises(RuntimeError, match="Run interactively"):
        loader.from_pretrained("org/model")
    fetch.assert_not_called()


def test_pd_rejects_unapproved_download_before_spawning(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    from vllm_rlt.pd.engine import PDEngine

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    spawn = Mock(side_effect=AssertionError("must reject before spawning"))
    monkeypatch.setattr("vllm_rlt.engine.worker.mp.get_context", spawn)
    with pytest.raises(RuntimeError, match="Download requires approval"):
        PDEngine("my-org/model")
    spawn.assert_not_called()


def test_serving_custom_repository_dispatches_from_config(
    monkeypatch, checkpoint, confirm_download
):
    from tokenizers.decoders import WordPiece

    from vllm_rlt.entrypoints.serve import load_engine

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(checkpoint / "config.json")
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: str(checkpoint))
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(decoder=WordPiece()))
    token_load = Mock(return_value=tokenizer)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", token_load)
    args = SimpleNamespace(
        model="custom-org/custom-name",
        revision="tag",
        tokenizer=None,
        tokenizer_revision=None,
        device="cpu",
        dtype="float32",
        num_blocks=16,
        block_size=2,
        max_num_seqs=1,
        max_num_batched_tokens=3,
        mode="refill",
        attention_backend="torch",
    )
    engine, actual = load_engine(args)
    try:
        assert engine.model_config.model_type == "nanbeige"
        assert actual is tokenizer
        assert token_load.call_args.args == (args.model,)
        assert token_load.call_args.kwargs["revision"] == "tag"
    finally:
        engine.close()


def test_pd_downloads_in_parent_before_spawning(monkeypatch, checkpoint, confirm_download):
    from vllm_rlt.pd.engine import PDEngine

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(checkpoint / "config.json")
    )
    snapshot = Mock(return_value=str(checkpoint))
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)

    def spawn(method):
        snapshot.assert_called_once()
        raise RuntimeError("stop before workers")

    monkeypatch.setattr("vllm_rlt.engine.worker.mp.get_context", spawn)
    with pytest.raises(RuntimeError, match="stop before workers"):
        PDEngine("custom-org/model")


def test_resolved_hub_snapshot_freezes_model_and_tokenizer_revision(
    monkeypatch, tmp_path, confirm_download
):
    from vllm_rlt.models import resolve_model_config

    config = tmp_path / "snapshots" / ("a" * 40) / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"model_type": "nanbeige"}')
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    fetch = Mock(return_value=str(config))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    snapshot = Mock(return_value=str(config.parent))
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    source, revision, _ = resolve_model_config("custom-org/checkpoint")
    assert source == str(config.parent)
    assert snapshot.call_args.kwargs["repo_id"] == "custom-org/checkpoint"
    assert fetch.call_args.kwargs["revision"] is None
    assert revision == "a" * 40


def test_model_loaders_require_explicit_source():
    from vllm_rlt.models import OuroForCausalLM

    for loader in (AutoModelForCausalLM, OuroForCausalLM, NanbeigeForCausalLM):
        with pytest.raises(TypeError):
            loader.from_pretrained()


def test_cli_requires_source_for_real_model(monkeypatch):
    from vllm_rlt.entrypoints.cli import main

    monkeypatch.setattr("sys.argv", ["vllm-rlt", "--prompt", "hello"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_missing_tokenizer_prompts_before_download(monkeypatch, answer):
    from vllm_rlt.models import load_tokenizer

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    prompt = Mock(return_value=answer)
    monkeypatch.setattr("builtins.input", prompt)
    tokenizer = object()
    load = Mock(side_effect=[OSError("not cached"), tokenizer])
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", load)
    if answer == "yes":
        assert load_tokenizer("org/tokenizer", revision="tag") is tokenizer
        assert load.call_count == 2
        assert load.call_args.kwargs == {"revision": "tag", "trust_remote_code": False}
    else:
        with pytest.raises(RuntimeError, match="not approved"):
            load_tokenizer("org/tokenizer", revision="tag")
        assert load.call_count == 1
    assert load.call_args_list[0].kwargs["local_files_only"] is True
    prompt.assert_called_once()


def test_local_checkpoint_never_prompts(monkeypatch, checkpoint):
    prompt = Mock(side_effect=AssertionError("local loading must not ask"))
    monkeypatch.setattr("builtins.input", prompt)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    assert isinstance(model, NanbeigeForCausalLM)
    prompt.assert_not_called()
