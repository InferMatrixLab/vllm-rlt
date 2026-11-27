"""Optional end-to-end checks against running Inkling plugin and native servers."""

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

_PROMPT = [100, 200, 300, 400] * 32


def _serve(endpoint, model, max_tokens, depth=None):
    payload = {
        "model": model,
        "prompt": _PROMPT,
        "temperature": 0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    if depth is not None:
        payload["vllm_xargs"] = {"rlt_target_loops": depth}
    request = Request(
        endpoint + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


@pytest.fixture
def inkling_service():
    endpoint = os.environ.get("RLT_SMOKE_PLUGIN_URL")
    if not endpoint:
        pytest.skip("set RLT_SMOKE_PLUGIN_URL to a running Inkling RLT service")
    return endpoint.rstrip("/"), os.environ.get("RLT_SMOKE_MODEL", "inkling")


def test_w1_fixed_depth_matches_native(inkling_service):
    endpoint, model = inkling_service
    native = os.environ.get("RLT_SMOKE_NATIVE_URL")
    if not native:
        pytest.skip("set RLT_SMOKE_NATIVE_URL to matching native vLLM")
    reference = _serve(native.rstrip("/"), model, 128)["choices"][0]["token_ids"]
    candidate = _serve(endpoint, model, 128, depth=3)["choices"][0]["token_ids"]
    assert len(candidate) == 128
    assert candidate == reference


def test_mixed_request_depths_match_trace(inkling_service):
    endpoint, model = inkling_service
    step_log = os.environ.get("RLT_SMOKE_STEP_LOG")
    if not step_log:
        pytest.skip("set RLT_SMOKE_STEP_LOG to the RLT log with step tracing enabled")
    depths = (1, 2, 3, None)
    with Path(step_log).open() as trace:
        trace.seek(0, 2)
        with ThreadPoolExecutor(max_workers=len(depths)) as pool:
            requests = [pool.submit(_serve, endpoint, model, 8, depth) for depth in depths]
            results = [request.result() for request in requests]
        trace_lines = [line for line in trace if "rlt step core " in line]

    outputs = [result["choices"][0]["token_ids"] for result in results]
    assert all(len(tokens) == 8 for tokens in outputs)
    assert len({tokens[0] for tokens in outputs}) == 1
    assert outputs[2] == outputs[3]
    for result, depth in zip(results, depths, strict=True):
        pattern = re.compile(
            rf"request_id='{re.escape(result['id'])}-0(?:-[^']+)?', "
            r"token_start=(\d+), token_count=(\d+), depth=(\d+), prefill=(True|False)"
        )
        events = Counter(
            (pos, int(loop), prompt == "True")
            for line in trace_lines
            for position, count, loop, prompt in pattern.findall(line)
            for pos in range(int(position), int(position) + int(count))
        )
        expected = Counter(
            (position, loop, True) for position in range(len(_PROMPT)) for loop in range(3)
        )
        expected.update(
            (position, loop, False)
            for position in range(len(_PROMPT), len(_PROMPT) + 7)
            for loop in range(depth or 3)
        )
        assert events == expected
