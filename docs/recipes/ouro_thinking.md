# Ouro-Thinking: shared basic-integration recipe

One native Ouro backend serves both checkpoint sizes. This recipe targets BF16,
fixed four-loop, synchronous eager inference with LAST_EXITED KV. It adds no chat
endpoint, structured reasoning parser, alternate precision profile or PD path.
See [RFC #69](https://github.com/InferMatrixLab/vllm-rlt/issues/69) for model scope
and [RFC #32](https://github.com/InferMatrixLab/vllm-rlt/issues/32) for ownership:
message preparation belongs to M1; model, sampling and attention remain unchanged.

## 1. Prepare an environment and checkpoint

Follow the [installation guide](../launching.md). Run from the repository root
with the project's Python environment activated. Download assets separately;
the commands below use a prepared local directory and do not download weights.

| Checkpoint | Validated checkpoint revision |
| --- | --- |
| `ByteDance/Ouro-1.4B-Thinking` | `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` |
| `ByteDance/Ouro-2.6B-Thinking` | `f1edd81e7ac41355db670500ceaf204e0f73af68` |

Use each checkpoint's own config and tokenizer. Match client and server assets.
Choose either model; all subsequent commands are shared:

```bash
export MODEL=/path/to/prepared/Ouro-1.4B-Thinking
# Alternatively: export MODEL=/path/to/prepared/Ouro-2.6B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

This recipe uses the native runtime, not the official Hugging Face model code.
Reference-model execution requires its own verified dependency and cache setup;
it is not a prerequisite for the commands below.

## 2. Start the server

Use an allocated device with sufficient free memory. Set its visible index
explicitly; do not copy a device number without checking availability.

```bash
CUDA_VISIBLE_DEVICES=<allocated-device> OMP_NUM_THREADS=2 \
python -m vllm_rlt.entrypoints.serve \
  --model "$MODEL" --tokenizer "$MODEL" \
  --served-model-name ouro-thinking \
  --device cuda --dtype bfloat16 --attention-backend triton \
  --num-blocks 160 --block-size 16 --max-num-seqs 2 \
  --max-num-batched-tokens 128 --cpu-threads 2 \
  --max-requests 4 --request-timeout 180 \
  --host 127.0.0.1 --port 18080
```

Explicit KV blocks do not impose a total CUDA allocator limit. Allow space for weights,
activations, scratch and CUDA context in addition to KV. Async/graphs/speculation
are not enabled. The request below explicitly selects four loops.

For CPU operation, replace `--device cuda --attention-backend triton` with
`--device cpu --attention-backend torch` and set `CUDA_VISIBLE_DEVICES=''`.
Real-checkpoint CPU execution can be slow; do not automatically extend budgets.

In another terminal, with the same MODEL environment variable:

```bash
curl --fail --silent --show-error http://127.0.0.1:18080/health
```

Wait for HTTP 200; 503 during initialization is not readiness. This is a local
benchmark server, not authenticated/TLS-hardened production deployment.

## 3. Send JSON and SSE requests

The following complete example makes exactly two requests, each capped at eight
new tokens. It loads tokenizer assets only, renders once, and checks termination,
usage and complete SSE transport. It requires only the installed project and
Python standard-library HTTP client.

```python
import json
import os
from urllib.request import Request, urlopen

from transformers import AutoTokenizer
from vllm_rlt.entrypoints.chat_template import render_chat_prompt

model = os.environ["MODEL"]
tokenizer = AutoTokenizer.from_pretrained(
    model, local_files_only=True, trust_remote_code=False
)
prompt = render_chat_prompt(
    tokenizer,
    [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
    enable_thinking=True,
)
prompt_tokens = len(tokenizer.encode(prompt))
for stream in (False, True):
    body = {
        "model": "ouro-thinking", "prompt": prompt,
        "max_tokens": 8, "min_loops": 4, "max_loops": 4,
        "exit_threshold": 1, "temperature": 0, "seed": 0,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    request = Request(
        "http://127.0.0.1:18080/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        if not stream:
            payload = json.load(response)
            text = payload["choices"][0]["text"]
            reason = payload["choices"][0]["finish_reason"]
            usage = payload["usage"]
        else:
            parts, reason, usage, done, token_events = [], None, None, False, 0
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                assert "error" not in event, event
                if event["choices"]:
                    choice = event["choices"][0]
                    parts.append(choice["text"])
                    token_events += 1
                    if choice["finish_reason"] is not None:
                        reason = choice["finish_reason"]
                if event.get("usage") is not None:
                    usage = event["usage"]
            assert done and usage is not None and reason is not None
            assert token_events == usage["completion_tokens"]
            text = "".join(parts)
    assert reason in ("stop", "length")
    assert usage["prompt_tokens"] == prompt_tokens
    assert 0 < usage["completion_tokens"] <= 8
    assert usage["total_tokens"] == prompt_tokens + usage["completion_tokens"]
    print(json.dumps({"stream": stream, "text": text,
                      "finish_reason": reason, "usage": usage}))
```

Eight tokens are a format/transport smoke, **not** a complete Thinking answer.
No retries, continuation prompts or synthesized endings are used. For a bounded
reliability load, a separate client can submit eight such requests at concurrency 2
(maximum 64 generated tokens), retaining every failure. Do not infer throughput
qualification from this tiny load; use the
[serving benchmark guide](../serving.md#reuse-the-upstream-benchmark)
for separately budgeted performance work.

## 4. Prompt and output semantics

The prepared `prompt` can also be passed to the existing
`llm.generate([prompt], sampling_params)` API, with four-loop sampling parameters.
No `LLM.chat` method or HTTP `messages`/`enable_thinking` field is introduced.
`max_tokens` is a **request** field; only the offline CLI has `--max-tokens`.

The helper accepts only `role`/`content` messages with system/user/assistant
roles and string content (including empty strings). Tools, multimodal inputs,
implicit batching and assistant continuation are not supported.

- `enable_thinking` must be a boolean or `None`; `None` omits the option and
  follows the official template default. It does not control loop depth.
- `False` only omits a prefilled opening marker; it does not prohibit
  autonomous reasoning or turn Thinking weights into base weights.
- Default decoding hides Thinking/control tokens but retains reasoning text.
  An opening in the prompt is not echoed; a closing is not EOS. Do not treat
  `</think>` as a stop signal or append one after length truncation.
- `length` does not mean reasoning completed. Never append a synthetic closing.
- Usage counts actual token IDs, including hidden EOS/control tokens. SSE empty
  text events are valid and still count as sampled tokens.
- Offline token-ID inspection can use `skip_special_tokens=False`, which exposes
  all special tokens, not just Thinking markers.
- Keep the client tokenizer/template revision aligned with the deployed weights
  and render each conversation once. Short smoke tests do not qualify model
  quality, full Thinking completion or official HF equivalence.

## 5. Capacity and lightweight checks

For four loops, LAST_EXITED and block size 16:

```text
positions = prompt_tokens + max_tokens - 1
physical_blocks = 4 * ceil(positions / 16)
```

A 33-token prompt with eight output tokens needs 12 blocks; two such requests fit
in 160. A 512-token output budget needs 136 blocks per request: 160 is not sufficient
for two simultaneous reservations of that size. Longer prompts and concurrency
require recalculation. Budgets never guarantee natural completion.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -m pytest -q tests/test_chat_template.py tests/test_serving.py
```

Install the project's dev dependencies for tests. Template unit tests require no
weights/device/download. Set `OURO_BASE_TOKENIZER`, `OURO_14B_THINKING_TOKENIZER`
and `OURO_26B_THINKING_TOKENIZER` to prepared local assets for the three optional
official-template checks; otherwise those cases explicitly skip.

## 6. Evidence and limitations

Validation at commit `53a6ec5` on NVIDIA H100 (BF16, Triton attention, fixed
four-loop synchronous eager execution, greedy):

- The JSON/SSE example above completed for both pinned Thinking checkpoints
  with complete usage and termination metadata; both servers shut down cleanly.
  A tiny BF16 model also passed with the documented CPU/Torch substitution.
- The [GSM8K Thinking evaluation](../accuracy.md#thinking-checkpoints) ran the
  87 fixed question IDs from the Base fixture under zero-shot Thinking prompts
  (3072 new tokens / 4096 total context) against each checkpoint's official
  Hugging Face implementation. Native versus HF: 1.4B 77/87 vs 76/87, 2.6B
  63/87 vs 62/87; both passed the at-most-1pp regression gate. The earlier
  1024-token run had incomplete finals from length truncation and was retained
  as a failed attempt, not as evidence. This selected cohort is a regression
  screen, not a held-out or full-test accuracy estimate.

This recipe does not qualify general accuracy, performance, strict official-HF
or cross-backend numerical equivalence. BF16 backend decision differences remain
possible; no precision, kernel or sampler change is part of this integration.
Base Ouro-2.6B shares the implementation, but the Thinking checks here do not
establish Base checkpoint quality or performance.

Adaptive exit, CUDA Graphs, asynchronous execution, PD, speculation and
alternative backend/layout combinations require their own checkpoint-specific
validation. Shared runtime features in
[PR #30](https://github.com/InferMatrixLab/vllm-rlt/pull/30) and
[PR #31](https://github.com/InferMatrixLab/vllm-rlt/pull/31) retain their own
validation scope; [RFC #43](https://github.com/InferMatrixLab/vllm-rlt/issues/43)
tracks speculative decoding separately. Shared implementation does not imply
that every combination has been validated on both Thinking checkpoints.
