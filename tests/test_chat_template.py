"""CPU-only chat preparation; real tokenizer checks are explicitly opt-in."""

import os
from copy import deepcopy

import pytest
from jinja2 import TemplateError

from benchmarks.chat_template import render_chat_prompt


class RecordingTokenizer:
    def __init__(self, result="rendered", error=None):
        self.result = result
        self.error = error
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize("enabled", [None, False, True])
def test_official_template_options_and_no_message_mutation(enabled):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "分析🙂"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": ""},
    ]
    before = deepcopy(messages)
    tokenizer = RecordingTokenizer()
    assert render_chat_prompt(tokenizer, messages, enable_thinking=enabled) == "rendered"
    options = {"tokenize": False, "add_generation_prompt": True}
    if enabled is not None:
        options["enable_thinking"] = enabled
    assert tokenizer.calls == [(messages, options)]
    assert messages == before


@pytest.mark.parametrize("enabled", [0, 1, "true", "false", [], {}, 1.0])
def test_reject_non_boolean_thinking_option_before_render(enabled):
    tokenizer = RecordingTokenizer()
    with pytest.raises(ValueError, match="enable_thinking"):
        render_chat_prompt(tokenizer, [{"role": "user", "content": "Hi"}], enable_thinking=enabled)
    assert not tokenizer.calls


@pytest.mark.parametrize(
    "messages",
    [
        None,
        [],
        "Hi",
        {"role": "user", "content": "Hi"},
        [[{"role": "user", "content": "Hi"}]],
        [None],
        [{}],
        [{"role": "user"}],
        [{"content": "Hi"}],
        [{"role": "user", "content": None}],
        [{"role": "user", "content": 1}],
        [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        [{"role": [], "content": "Hi"}],
        [{"role": "", "content": "Hi"}],
        [{"role": "tool", "content": "Hi"}],
        [{"role": "assistant", "content": "", "tool_calls": []}],
        [{"role": "user", "content": "Hi", "name": "Alice"}],
        [{"role": "user", "content": "\ud800"}],
    ],
)
def test_reject_unsupported_messages_before_render(messages):
    tokenizer = RecordingTokenizer()
    with pytest.raises(ValueError):
        render_chat_prompt(tokenizer, messages)
    assert not tokenizer.calls


@pytest.mark.parametrize("error", [TemplateError("missing field"), TypeError("content type")])
def test_template_input_errors_are_value_errors(error):
    with pytest.raises(ValueError, match="chat template") as result:
        render_chat_prompt(RecordingTokenizer(error=error), [{"role": "user", "content": "Hi"}])
    assert result.value.__cause__ is error


def test_non_input_failure_is_not_hidden():
    with pytest.raises(RuntimeError, match="broken tokenizer"):
        render_chat_prompt(
            RecordingTokenizer(error=RuntimeError("broken tokenizer")),
            [{"role": "user", "content": "Hi"}],
        )


@pytest.mark.parametrize("result", [["prompt"], {"input_ids": [1]}, None])
def test_template_must_return_one_string(result):
    with pytest.raises(ValueError, match="one string"):
        render_chat_prompt(RecordingTokenizer(result=result), [{"role": "user", "content": "Hi"}])


@pytest.mark.parametrize(
    "env_name,supports_thinking",
    [
        ("OURO_BASE_TOKENIZER", False),
        ("OURO_14B_THINKING_TOKENIZER", True),
        ("OURO_26B_THINKING_TOKENIZER", True),
    ],
)
def test_prepared_checkpoint_template_matches_official(env_name, supports_thinking):
    source = os.environ.get(env_name)
    if source is None:
        pytest.skip(f"set {env_name} to prepared local tokenizer assets; no automatic downloads")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "分析🙂：2+2?"},
    ]
    for enabled in [None, False, True]:
        options = {} if enabled is None else {"enable_thinking": enabled}
        actual = render_chat_prompt(tokenizer, messages, enable_thinking=enabled)
        expected = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **options
        )
        assert actual == expected
        assert tokenizer.encode(actual) == tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True, **options
        )
        assert actual.endswith("<think>\n") == (supports_thinking and enabled is True)
    assert render_chat_prompt(tokenizer, messages) == render_chat_prompt(
        tokenizer, messages, enable_thinking=False
    )
