# OpenAI-compatible Ouro serving

Install `python -m pip install -e '.[serve,triton]'` in a prepared PyTorch
environment. The `vllm-rlt-serve` entrypoint (or `python -m
vllm_rlt.entrypoints.serve`) serves one resident model on one device. Bind defaults
to `127.0.0.1:8000`; this benchmark frontend has no authentication or TLS.

Prepare the Ouro checkpoint and tokenizer separately. The official model ID
pins revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; local paths are supported.
Use a GPU scheduler reservation on shared hosts:

```bash
gpu status
gpu run --gpu-ids <available-id> --timeout 30m --note "Ouro HTTP serving" -- \
  python -m vllm_rlt.entrypoints.serve \
  --model /path/to/prepared/ouro --tokenizer /path/to/prepared/ouro \
  --device cuda --dtype bfloat16 --attention-backend triton \
  --num-blocks 512 --max-num-seqs 8
```

BF16 weights, ordinary activations and KV are this frontend's primary target,
as described in the [precision policy](precision-policy.md).
The current model uses FP32 RMSNorm/RoPE intermediates and gate/sampling
probabilities; Triton attention uses FP32 internal accumulation. Record the
matrix multiplication backend/reduction flags separately. `--dtype float32`
is available for diagnostics. CPU execution requires `--attention-backend torch`.

`GET /health` returns 503 during initialization, shutdown or engine failure, and
200 only after model, tokenizer and engine initialization finish. Tokenization
and inference are not deferred to a dummy readiness response. Loading failure
leaves health unavailable and emits the cause in the server log. The first
actual inference may include lazy library/kernel setup; validate it explicitly.
`GET /v1/models` lists the configured `--served-model-name` (default
`ByteDance/Ouro-1.4B`).

## Stage deployment and batching

Serving workers use the common stage engine. A local `prefill_decode` stage
handles prompt and generation work; PD deploys prefill and decode stages and
uses a connector for state transfer. Request submission and output delivery
remain separate from the stage-local model execution contract.

Mixed scheduling means batching compatible prompt/decode recurrent work within
one stage. It does not mean a worker role, and PD does not require it. For
supported models, select it with `--mode mixed --no-atomic-prefill` and
LAST_EXITED KV. PD can configure speculative decoding on the decode stage, but
mixed scheduling and speculative rounds are not a supported combination.
See [runtime configuration](runtime.md#execution-batches-and-mixed-scheduling)
for constraints and [the architecture](unified_execution.md) for ownership.

## Completions contract

`POST /v1/completions` requires `Content-Type: application/json`, `model` and one
string `prompt`. Requests with an `Origin` header are rejected with 403: this
frontend supports command-line clients, not browser callers. Other content
types return 415. All routes reject unrecognized Host headers; accepted hosts
are the connection's local IP address, `localhost`, and the configured `--host`.
Streaming and ordinary JSON responses use the same generation and text contract.

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ByteDance/Ouro-1.4B","prompt":"The capital of France is","max_tokens":16,"stream":true,"stream_options":{"include_usage":true}}'
```

| Field | Default / supported behavior |
| --- | --- |
| `max_tokens` | 16; positive integer |
| `temperature`, `top_p`, `top_k` | 0, 1, -1; passed to `SamplingParams` |
| `seed`, `ignore_eos` | 0, false; request-local RNG and EOS policy |
| `min_loops`, `max_loops`, `exit_threshold` | 2, model depth, 1; Ouro extensions |
| `stream` | false; true enables SSE |
| `stream_options.include_usage` | false; valid with streaming |
| `n`, `best_of`, `echo` | only 1, 1, false |
| `repetition_penalty`, `presence_penalty`, `frequency_penalty` | only neutral 1, 0, 0 |
| `logprobs`, `stop`, `suffix`, `logit_bias` | only null |

Unsupported fields/behaviors, invalid values, empty tokenized prompts, impossible
KV reservations and context overflow return 400. An unknown model returns 404.
Missing or invalid model identifiers return 400. Omit fields to use defaults;
explicit null is accepted only for `max_loops`, `stream_options` and the four
null-only fields above. Structured errors use the same value for `error.type`
and `error.code`; admission messages include diagnostic context/KV limits.
There is no chat endpoint, prompt-list batching, token-ID prompt API, stop-string
matching, beam search or multi-completion support. Independent HTTP requests
share continuous engine batching. Prompt prefill always runs full depth;
loop extensions control subsequent decode, as in the offline engine.

The official byte-level tokenizer is required. Prompt encoding uses its normal
special-token behavior. Decoding skips special tokens and disables whitespace
cleanup, matching the released tokenizer. Incomplete UTF-8 suffixes are held
until complete or flushed at the final token. Every sampled token emits one
choice event, including tokens whose current text delta is empty. The final
token carries `finish_reason` (`stop` or `length`); no extra finish-only choice
is appended. When requested, one `choices: []` usage event follows, then
`data: [DONE]`. Ordinary responses always include usage. Counts come from
actual prompt/generated IDs, including generated EOS or other hidden special
tokens, rather than from re-tokenizing the displayed text.
The decoder processes each sampled token once, retaining only incomplete UTF-8
bytes between events. Completed text is never decoded again.

## Ouro-Thinking prompts

For checkpoint setup, executable JSON/SSE examples, loop configuration,
KV sizing and validation scope, see the
[Ouro-Thinking recipe](recipes/ouro_thinking.md).

The completions API contract here is unchanged for Thinking checkpoints: the
server accepts a single pre-rendered `prompt` string, not `messages` or an
`enable_thinking` field, and does not apply a chat template itself. Prepare the
prompt on the client with the shared helper described in the recipe. Responses
remain cleaned completion text with no separate reasoning field; decoding hides
special tokens while retaining reasoning prose.

## Ownership, limits and errors

One worker thread owns initialization, tokenization, engine admission, every
step, detokenization, cancellation and teardown. The HTTP event loop awaits
that worker without executing device work. New arrivals/cancellations are
processed between steps. At most one prefill batch can postpone an existing
recurrent loop; the same progress guard applies to refill and no-refill modes.

- `--max-requests` (64) bounds concurrent HTTP completion handlers and accepted
  request channels, including waiting requests and completed responses still
  being consumed. Saturation returns 429 with `Retry-After: 1`.
- `--max-body-bytes` (1 MiB) bounds each request body; excess returns 413.
- `--output-buffer` (32) bounds queued token events per request. Overflow aborts
  that request without blocking the engine or other clients.
- `--request-timeout` (300 seconds) covers body reading through response
  completion. `--write-timeout` (10 seconds) bounds individual stream writes.
- `--shutdown-timeout` (30 seconds) bounds the cooperative shutdown wait.
  Exceeding it logs a warning and allows HTTP teardown to continue. Owner-thread
  cleanup still runs if the call returns. A call that never returns requires the
  outer process supervisor; Python cannot terminate a running thread safely.

Disconnects and deadlines enqueue cancellation through the owner. Request slots
remain reserved until the owner acknowledges cancellation. Invalid requests,
slow consumers, decode validation errors and cancellation preserve unrelated
engine work. Unexpected engine failures conservatively make the service unready, fail outstanding
requests and clean up all request state; automatic engine restart is not
implemented. The engine itself invalidates a failed execution batch.

Before streaming starts, errors use structured JSON and an HTTP status. After
headers/token output, the server aborts the chunked HTTP transport without
`[DONE]` or a terminating HTTP chunk. This is deliberate: the pinned benchmark
client treats an in-band SSE error, or clean EOF after a token, as success.
The real-socket CPU tests verify transport-error behavior and cleanup. Server
logs retain generation counts, errors and final request/KV cleanup.

The default 512-page cache holds eight requests of 128 input and 64 output
tokens at four loops: each reserves `ceil(191 / 16) * 4 = 48` pages. That workload
needs at least 384 pages. The offline default of 256 pages fits only five.
HTTP concurrency and engine-admitted concurrency are separate controls.

## Reuse the upstream benchmark

Use the unmodified **vLLM 0.28.0** client, source tag commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`, in its own prepared environment.
The server does not depend on vLLM. Keep model/tokenizer revisions fixed on both
sides. Disable optional external vLLM plugins for the benchmark client.

```bash
VLLM_PLUGINS='' vllm bench serve \
  --backend openai --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions --model ByteDance/Ouro-1.4B \
  --tokenizer /path/to/prepared/ouro --dataset-name random \
  --random-input-len 128 --random-output-len 64 --random-range-ratio 0 \
  --num-prompts 100 --seed 20260912 --max-concurrency 8 --request-rate inf \
  --ignore-eos \
  --extra-body '{"temperature":0,"seed":0,"min_loops":2,"max_loops":4,"exit_threshold":1}' \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed --result-dir artifacts/http \
  --result-filename concurrent.json
```

For a serial workload use `--max-concurrency 1 --request-rate inf`; for a finite
arrival-rate example use `--max-concurrency 4 --request-rate 2`. Save each result
under a separate filename. These are compatibility examples, not a tail-latency
qualification. Inspect failed requests and detailed errors as well as successful
request/token throughput, TTFT, TPOT, ITL and end-to-end latency. The client may
probe optional `/tokenize` or `/metrics` endpoints; they are outside this server's
contract and not required for these completions workloads.

The pinned parser timestamps choice events, including empty text. Usage and
`[DONE]` are not token samples. ITL therefore measures receipt of token events;
Unicode text can become visible only with a later event, and network buffering
can deliver several events together. Its latency starts after the client
concurrency semaphore and ends at the last choice event, excluding client-side
queueing and the usage/`[DONE]` tail. The historical offline M1 harness instead timestamps
engine enqueue/steps and excludes HTTP/tokenization; the values are not
interchangeable. The M1 harness has been removed; see the
[archived baseline report](https://github.com/hsliuustc0106/vllm-rlt/blob/5bee22950357d93e3f7a3c6c87b8fbed004c9296/docs/benchmarks/m1-20260910.md)
for its measurement scope.

## Validation

CPU tests use small models and real loopback sockets:

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_serving.py tests/test_engine.py tests/test_chat_template.py
```

Chat-template tests cover input validation and forwarding of template options.
Set `OURO_BASE_TOKENIZER`, `OURO_14B_THINKING_TOKENIZER` and
`OURO_26B_THINKING_TOKENIZER` to prepared local checkpoint directories to also
check their official templates; these opt-in tests never download assets.

For real-model checks, validate the first inference after readiness and
overlapping requests against direct engine execution for text, usage and finish
reasons, including fixed/adaptive loop policies. Keep settings fixed; BF16
arithmetic can differ across configurations. Then run the serial, concurrent
and finite-rate client commands above against the same resident server.

Keep commands, source/model revisions, environment details and raw results
under ignored `artifacts/`. Record validation outcomes in the PR description.
Separate preparation and startup from client timings, and preserve failed runs.
