"""The deliberately small completions contract, independent of HTTP and device code."""

import codecs
import json
import math
from dataclasses import dataclass, fields

from vllm_rlt.sampling_params import SamplingParams


@dataclass(frozen=True)
class ServingLimits:
    max_requests: int = 64
    output_buffer: int = 32
    max_body_bytes: int = 1024 * 1024
    request_timeout: float = 300.0
    write_timeout: float = 10.0
    shutdown_timeout: float = 30.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is int and type(value) is not int:
                raise ValueError(f"{field.name} must be a positive integer")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be finite and positive")


class ServingError(Exception):
    def __init__(self, message, status=500, code="server_error"):
        super().__init__(message)
        self.status = status
        self.code = code

    def as_dict(self):
        return {"error": {"message": str(self), "type": self.code, "code": self.code}}

    @classmethod
    def invalid_request(cls, message):
        return cls(message, 400, "invalid_request_error")

    @classmethod
    def not_ready(cls):
        return cls("engine is not ready", 503, "not_ready")

    @classmethod
    def overloaded(cls):
        return cls("request capacity exhausted", 429, "overloaded")


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    params: SamplingParams
    stream: bool = False
    include_usage: bool = False
    trace_id: str | None = None

    @classmethod
    def parse(cls, body, model):
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        params_keys = {field.name for field in fields(SamplingParams)}
        neutral = {
            "n": 1,
            "best_of": 1,
            "echo": False,
            "logprobs": None,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": None,
            "suffix": None,
            "logit_bias": None,
        }
        allowed = (
            params_keys
            | neutral.keys()
            | {"model", "prompt", "stream", "stream_options", "trace_id"}
        )
        if unknown := body.keys() - allowed:
            raise ValueError(f"unsupported fields: {', '.join(sorted(unknown))}")
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise ValueError("model must be a nonempty string")
        if body["model"] != model:
            raise ServingError(f"model must be {model!r}", 404, "model_not_found")
        if not isinstance(body.get("prompt"), str):
            raise ValueError("prompt must be one string")
        try:
            body["prompt"].encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("prompt must contain valid Unicode scalar values") from exc
        for key, default in neutral.items():
            value = body.get(key, default)
            valid_type = type(value) is type(default)
            if type(default) is float:
                valid_type = type(value) in (int, float)
            if not valid_type or value != default:
                raise ValueError(f"only {key}={default!r} is supported")
        stream = body.get("stream", False)
        if type(stream) is not bool:
            raise ValueError("stream must be a boolean")
        options = body.get("stream_options")
        include_usage = False
        if options is not None:
            if not stream or not isinstance(options, dict):
                raise ValueError("stream_options requires stream=true and an object")
            if options.keys() - {"include_usage"}:
                raise ValueError("only stream_options.include_usage is supported")
            include_usage = options.get("include_usage", False)
            if type(include_usage) is not bool:
                raise ValueError("include_usage must be a boolean")
        params = {key: body[key] for key in params_keys & body.keys()}
        for key in ("temperature", "top_p", "exit_threshold", "repetition_penalty"):
            if key in params and (
                type(params[key]) not in (int, float) or not math.isfinite(params[key])
            ):
                raise ValueError(f"{key} must be a finite number")
        if "ignore_eos" in params and type(params["ignore_eos"]) is not bool:
            raise ValueError("ignore_eos must be a boolean")
        trace_id = body.get("trace_id")
        if "trace_id" in body and (not isinstance(trace_id, str) or not trace_id):
            raise ValueError("trace_id must be a nonempty string")
        return cls(body["prompt"], SamplingParams(**params), stream, include_usage, trace_id)


class IncrementalText:
    """Decode each sampled ID once, retaining only incomplete UTF-8 bytes.

    Cleanup is disabled, matching the released tokenizer. Incomplete UTF-8
    prefixes are withheld until a later token completes them (or final flush).
    Each sampled ID is decoded once; the worker may combine several IDs into one event.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(tokenizer.all_special_ids)
        # ByteLevel's byte alphabet is stable; avoid Transformers' private helper.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        self.byte_decoder = {chr(byte): byte for byte in visible}
        for offset, byte in enumerate(b for b in range(256) if b not in visible):
            self.byte_decoder[chr(256 + offset)] = byte
        self.utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def decode(self, token_id, finished):
        data = b""
        if token_id not in self.special_ids:
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if token is None:
                raise ValueError(f"tokenizer has no token for sampled ID {token_id}")
            try:
                data = bytes(self.byte_decoder[char] for char in token)
            except KeyError:
                # Handle SentencePiece leading space (\u2581) if present
                if token.startswith("\u2581"):
                    token = " " + token[1:]
                data = token.encode("utf-8")
        return self.utf8.decode(data, final=finished)


def usage(prompt_tokens, completion_tokens):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def completion(request_id, model, created, text, finish_reason, token_usage=None, *, summary=False):
    result = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": []
        if summary
        else [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}],
    }
    if token_usage is not None:
        result["usage"] = token_usage
    return result


def sse(data):
    return ("data: " + json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n\n").encode()
