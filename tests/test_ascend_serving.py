"""Real socket + spawned NPU worker smoke, opt-in with pinned local weights."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("backend", ["torch", "ascend_eager"])
def test_checkpoint_http_completions_stream_and_disconnect(tmp_path, backend):
    import aiohttp

    source = os.environ.get("VLLM_RLT_OURO_CHECKPOINT")
    if not source:
        pytest.skip("set VLLM_RLT_OURO_CHECKPOINT to the pinned local Ouro-1.4B checkpoint")
    folder = Path(os.environ.get("VLLM_RLT_ASCEND_RESULTS", tmp_path)) / backend
    folder.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "vllm_rlt.entrypoints.serve",
        "--model",
        source,
        "--served-model-name",
        "ouro-ascend",
        "--device",
        "npu",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        backend,
        "--num-blocks",
        "64",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "32",
        "--port",
        str(port),
    ]
    with (folder / "service.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )

        async def exercise():
            url = f"http://127.0.0.1:{port}"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as client:
                deadline = time.monotonic() + 120
                while True:
                    assert process.poll() is None, (folder / "service.log").read_text()
                    try:
                        async with client.get(url + "/health") as response:
                            if response.status == 200:
                                break
                    except aiohttp.ClientConnectionError:
                        pass
                    assert time.monotonic() < deadline, "NPU service startup timed out"
                    await asyncio.sleep(0.25)

                def body(prompt, **extra):
                    return dict(
                        model="ouro-ascend",
                        prompt=prompt,
                        max_tokens=4,
                        temperature=0,
                        min_loops=4,
                        max_loops=4,
                        exit_threshold=1,
                        ignore_eos=True,
                        **extra,
                    )

                async def complete(prompt):
                    async with client.post(url + "/v1/completions", json=body(prompt)) as response:
                        result = await response.json()
                        assert response.status == 200, result
                        assert result["usage"]["completion_tokens"] == 4
                        return result

                prompts = ["The capital of France is", "One plus one equals"]
                normal = await asyncio.gather(*(complete(prompt) for prompt in prompts))
                events = []
                async with client.post(
                    url + "/v1/completions",
                    json=body(prompts[0], stream=True, stream_options={"include_usage": True}),
                ) as response:
                    assert response.status == 200
                    async for line in response.content:
                        if line.startswith(b"data: "):
                            data = line[6:].strip().decode()
                            events.append(data if data == "[DONE]" else json.loads(data))
                assert events[-1] == "[DONE]"
                text = "".join(
                    choice["text"] for event in events[:-1] for choice in event.get("choices", [])
                )
                assert text == normal[0]["choices"][0]["text"]
                assert any(
                    isinstance(event, dict)
                    and (event.get("usage") or {}).get("completion_tokens") == 4
                    for event in events
                )
                # Close a real streaming connection early, then verify the worker
                # remains usable. Direct engine tests check page reclamation.
                async with client.post(
                    url + "/v1/completions", json=body(prompts[1], stream=True)
                ) as response:
                    assert response.status == 200
                    await response.content.readline()
                after_disconnect = await complete(prompts[1])
                assert after_disconnect["choices"][0]["text"] == normal[1]["choices"][0]["text"]
                return dict(normal=normal, stream=events, after_disconnect=after_disconnect)

        try:
            result = asyncio.run(exercise())
            (folder / "service-results.json").write_text(json.dumps(result, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        assert process.returncode == 0, (folder / "service.log").read_text()
