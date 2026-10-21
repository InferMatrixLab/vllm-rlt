# User Guide: From Installation to the First Request

Follow steps 1–8 to serve Ouro-1.4B on one NVIDIA GPU and receive a generated
response. Commands work in Bash and fish on Linux; use the activation command
for your shell. Run them in the same terminal unless a step explicitly asks
you to open another one.

## 1. Check the machine

Use Linux with Git, curl, [uv](https://docs.astral.sh/uv/getting-started/installation/),
Python 3.10+, and an NVIDIA GPU
that supports BF16. Run `nvidia-smi` to confirm that the driver recognizes your
GPU. Installation and model download require access to GitHub, Python package
indexes, and Hugging Face. The BF16 model weights alone need roughly 3 GB of
GPU memory; leave additional room for the KV cache and runtime buffers.

## 2. Create and activate a uv environment

```bash
uv venv --python 3.10 .venv  # Replace .venv with your preferred environment name
source .venv/bin/activate   # Bash; use the same environment name here
# fish: source .venv/bin/activate.fish
```

Python 3.10 is an example; you can select another supported version (3.10+).
Keep this environment active for the following steps.

## 3. Clone the repository

```bash
git clone https://github.com/hsliuustc0106/vllm-rlt.git
cd vllm-rlt
```

## 4. Install the project

```bash
uv pip install -e .
python -m vllm_rlt.entrypoints.serve --help
```

The help command should display the server options, confirming that the
current environment can import the server entrypoint. It does not start the server.

This installs PyTorch, Transformers, the HTTP server dependencies, and Triton.
The project requires **PyTorch 2.5 or newer**; it does not pin an exact version.
uv keeps an existing compatible installation, or resolves a compatible version
from your configured package index in a fresh environment.

Use the same installation and startup commands for GPUs such as the RTX 4090,
A100, and H100. There is no CUDA version to select based on the GPU model;
your NVIDIA driver must support the CUDA runtime supplied with PyTorch.
FlashAttention and NIXL are optional and are not needed for this guide.

Select your GPU and confirm that PyTorch can use it:

```bash
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.is_bf16_supported(), 'BF16 required'"
```

Use your allocated GPU ID in place of `0`. If your job scheduler already sets
`CUDA_VISIBLE_DEVICES`, keep its value and skip the export.

## 5. Download the model and tokenizer

Download the pinned checkpoint once, including tokenizer files. This separates
network/download failures from server startup failures.

```bash
python -c "
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='ByteDance/Ouro-1.4B',
    revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1',
    local_dir='./artifacts/models/Ouro-1.4B',
    allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'],
)
"
```

Wait for the download to finish. Keep running commands from the repository
root so `./artifacts/models/Ouro-1.4B` resolves to the same directory. For an existing
local checkpoint, replace this path in the commands below; it must contain
both the model configuration/weights and the tokenizer files. Loading uses
native code and safetensors, without `trust_remote_code`.

## 6. Start the server

In the same terminal, run:

```bash
python -m vllm_rlt.entrypoints.serve \
  --model ./artifacts/models/Ouro-1.4B \
  --served-model-name ouro \
  --device cuda \
  --dtype bfloat16 \
  --attention-backend triton \
  --num-blocks 64 \
  --max-num-seqs 1 \
  --host 127.0.0.1 \
  --port 8000
```

Leave this process running. The equivalent installed command is
`vllm-rlt-serve` with the same arguments. Startup loads the model and tokenizer;
the first generation can also incur kernel compilation overhead.

This first-run configuration uses a bounded KV pool and one active sequence
for short prompts. It is not a performance-benchmark configuration. For longer
prompts or more concurrency, adjust cache capacity and scheduling as described
in the [runtime guide](runtime.md). Omitting `--num-blocks` enables automatic
CUDA KV sizing.

## 7. Check readiness and send a request

Open a **second terminal on the same machine**. These commands only require
curl; the server remains running in the first terminal.

```bash
curl -i http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```

Wait until `/health` returns **HTTP 200**. HTTP 503 means it is not ready;
inspect the first terminal for initialization progress or errors. The model
list must include `ouro`, matching `--served-model-name`.

Send a non-streaming request first:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ouro",
    "prompt": "The capital of France is",
    "max_tokens": 32,
    "temperature": 0,
    "exit_threshold": 0.7,
    "stream": false
  }'
```

A successful response contains `choices[0].text`, a `finish_reason`, and `usage`
with prompt/completion token counts. The exact generated text can vary; checking
readiness alone does not verify generation.

To receive tokens as a stream:

```bash
curl --fail-with-body -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ouro","prompt":"The capital of France is","max_tokens":32,"temperature":0,"exit_threshold":0.7,"stream":true}'
```

Expect SSE `data:` events ending with `data: [DONE]`. This is a completions
endpoint, not `/v1/chat/completions`. See the [HTTP API reference](serving.md)
for supported fields, streaming behavior, and limits.

## 8. Stop and restart

Press **Ctrl+C in the server terminal** to stop it and release its GPU resources.
For a later session, return to the repository, reactivate the environment,
select your allocated GPU, and rerun step 6. The model download does not need
to be repeated.

```bash
cd /path/to/vllm-rlt
source ../.venv/bin/activate  # Bash; use the environment name from step 2
# fish: source ../.venv/bin/activate.fish
export CUDA_VISIBLE_DEVICES=0
```

Replace `/path/to/vllm-rlt` with your checkout path and preserve scheduler-set
GPU visibility when applicable. This completes the first-run path:
installation, model loading, readiness, a real completion, and shutdown.

## Command-line inference

For an offline batch, stop the HTTP server first if you will reuse the same
GPU, then run from the activated environment and repository root:

```bash
python -m vllm_rlt.entrypoints.cli \
  --model ./artifacts/models/Ouro-1.4B \
  --device cuda \
  --attention-backend triton \
  --num-blocks 64 \
  --max-num-seqs 1 \
  --prompt 'The capital of France is' \
  --prompt '2 + 2 =' \
  --max-tokens 32 \
  --exit-threshold 0.7
```

The equivalent installed command is `vllm-rlt`. Repeat `--prompt` for a batch;
the CLI prints one JSON result per prompt, including text, token IDs, exit
depths, and the finish reason.

### Attention backend selection

Use `--attention-backend auto` with the inference CLI or the local HTTP server
to select a compatible implementation at startup. Auto tries FA4, FA3, FA2,
Triton, then Torch, checking hardware, dtype, head dimension, block size, installed
packages, and execution requirements. CPU uses Torch. For example, on an RTX
4090, the default 16-token block size rules out FA2's 256-token page requirement;
auto selects Triton if it is installed.

GPU async scheduling and CUDA Graphs require a device backend. UVA packed
prefill requires declared packed-prefill support and LAST_EXITED KV. Auto reports
an error if no implementation satisfies these requirements. Selection occurs
before memory profiling, and the startup log reports the selected backend and
why earlier candidates were skipped.

Explicit choices (`torch`, `triton`, `flash_attn`, or `flash_attn_2/3/4`) retain
their validation errors and never fall back. Existing defaults are unchanged:
the inference CLI defaults to Torch and the local server defaults to Triton.
The Python equivalent is `attention_backend="auto"`. Read the worker startup
log for the selection decision. `LLM.engine` is a `StageClient`; worker-local
cache objects are not exposed through that client.

## Python API

Save the following as `example.py` in the repository root, then run
`python example.py` from the activated environment. Stop the HTTP server first
if it uses the same GPU.

`LLM` starts a spawned worker process. In Python scripts, create it inside
`main()` and use the `if __name__ == "__main__":` guard shown below. The context
manager closes the worker when inference finishes or raises.

```python
from vllm_rlt import CacheConfig, LLM, SamplingParams


def main():
    with LLM(
        "./artifacts/models/Ouro-1.4B",
        device="cuda",
        attention_backend="triton",
        cache_config=CacheConfig(num_blocks=64),
    ) as llm:
        outputs = llm.generate(
            ["The capital of France is", "2 + 2 ="],
            SamplingParams(max_tokens=32, min_loops=2, max_loops=4, exit_threshold=0.7),
        )
        for output in outputs:
            print(output.text)
            print("Exit depths:", output.exit_depths)


if __name__ == "__main__":
    main()
```

Results preserve input order. Prompt prefill runs at full depth; the exit
threshold controls subsequent decode. Set `exit_threshold=1.0` for fixed-depth
execution. The first generated token comes from full-depth prefill, reflected
in `exit_depths`. For dynamic arrivals, stepwise output, and cancellation, see
the [architecture overview](unified_execution.md).

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `vllm-rlt-serve: command not found` | Activate the environment from step 2 and rerun the editable install. The module entrypoint `python -m vllm_rlt.entrypoints.serve` uses the same server. |
| `No module named vllm_rlt`, `aiohttp`, or `transformers` | Activate the environment from step 2, verify `sys.executable`, and rerun `uv pip install -e .`. |
| CUDA check fails or reports a driver error | Check your GPU allocation and `CUDA_VISIBLE_DEVICES`. If the driver is too old for the installed PyTorch CUDA build, update the driver or choose a compatible build using the [PyTorch installer](https://pytorch.org/get-started/locally/). The installer does not select builds based on your installed driver. See [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). |
| Model download fails | Check access to Hugging Face and available disk space, then rerun step 5. An offline deployment needs a complete downloaded checkpoint and tokenizer. |
| CUDA out of memory | Stop duplicate model processes you started, inspect free memory with `nvidia-smi`, and use the bounded short-prompt configuration in step 6. The weights and runtime still need sufficient free memory. |
| `/health` gives 503 or connection refused | Check the server terminal. It may still be initializing or may have exited with an error. |
| Port 8000 is already in use | Stop your old server or choose another `--port` and use it in all curl URLs. |
| Request returns an unknown-model error | The JSON `model` must match `--served-model-name`: `ouro` in this guide. |
| Request is rejected for KV capacity/context limits | Shorten the prompt/output or increase the KV budget for your workload. The first-run cache is intentionally small. |
| Chat endpoint returns 404 | Use `/v1/completions` with a string `prompt`. |

When reporting a problem, include the failing command, server error, GPU model,
PyTorch/CUDA versions from step 4, and your Git revision (`git rev-parse HEAD`).

## Choosing scheduling and deployment separately

The default atomic prefill path and loop-yielding prefill use the same execution
architecture. On supported models with LAST_EXITED KV, the CLI/server flags
`--mode mixed --no-atomic-prefill` allow compatible prompt and decode recurrence
to share a batch. This is a batching choice, not PD disaggregation.

Stage roles are `prefill`, `decode` and `prefill_decode`; `mixed` is a scheduling
mode, not a stage role. Speculation requires refill scheduling and cannot join
mixed recurrent batches. Model/backend restrictions still apply. See the
[runtime guide](runtime.md#execution-batches-and-mixed-scheduling) before
combining optional features.
