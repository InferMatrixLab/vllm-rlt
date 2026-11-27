"""Prepare text-only chat prompts without changing the raw generation contract."""


def render_chat_prompt(tokenizer, messages, *, enable_thinking=None) -> str:
    """Render one conversation using the checkpoint's official chat template.

    Always starts a new assistant turn. Tools, multimodal content, batched
    conversations and assistant continuation are not supported. ``None`` omits
    the Thinking option and leaves the checkpoint's template default unchanged.
    The option only affects prompting, not sampling or recurrent depth.
    """
    if enable_thinking is not None and type(enable_thinking) is not bool:
        raise ValueError("enable_thinking must be a boolean or None")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be one nonempty conversation list")
    for message in messages:
        if not isinstance(message, dict) or message.keys() != {"role", "content"}:
            raise ValueError("each message must contain only role and content")
        role = message["role"]
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise ValueError("message role must be system, user or assistant")
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        try:
            content.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("message content must contain valid Unicode scalar values") from exc

    from jinja2 import TemplateError

    options = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **options
        )
    except (TemplateError, TypeError) as exc:
        raise ValueError("chat template could not render the messages") from exc
    if not isinstance(prompt, str):
        raise ValueError("chat template must render one string")
    return prompt


def special_token_ids(tokenizer):
    """Include added tokens whose special flag is absent from all_special_ids."""
    ids = set(tokenizer.all_special_ids)
    ids.update(
        token_id
        for token_id, token in getattr(tokenizer, "added_tokens_decoder", {}).items()
        if token.special
    )
    return ids
