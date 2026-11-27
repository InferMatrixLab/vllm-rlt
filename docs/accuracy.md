# Ouro GSM8K-87 accuracy regression

The default regression uses 87 fixed questions from GSM8K `main`/`test`.
The [fixture](../benchmarks/fixtures/gsm8k-87.json) pins the dataset revision,
question IDs and a recipe fingerprint. The recorded Transformers baseline was
59/87 correct. The standalone engine also scored 59/87 at the
[pre-plugin revision](https://github.com/InferMatrixLab/vllm-rlt/blob/4821d47cec92aa1d7d2ec4be5f589d19b2794ae3/docs/accuracy.md).

The default uses the Ouro-1.4B release, BF16, greedy four-loop decoding, one
request at a time, three demonstrations, a 2,048-token context and a
1,024-token generation limit. The default 1 percentage point tolerance
requires at least 59 correct. Adaptive runs report accuracy and actual decode
depth for each generated token.

## Serve Ouro

Use the vLLM environment from the [installation guide](../README.md).

```bash
VLLM_PLUGINS=rlt vllm serve /path/to/ouro \
  --served-model-name ouro --trust-remote-code --dtype bfloat16 \
  --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 128 \
  --no-enable-prefix-caching --enforce-eager --attention-backend TRITON_ATTN \
  --generation-config vllm --no-async-scheduling \
  --scheduler-cls vllm_rlt.core.loop_scheduler.LoopScheduler \
  --worker-cls vllm_rlt.worker.loop_worker.LoopWorker \
  --additional-config '{"rlt_trace_steps":true}' > /path/to/ouro.log 2>&1
```

## Evaluate

Use a separate Python environment for the evaluation dependencies in
`benchmarks/requirements-accuracy.txt`. Reproducing the stored recipe also
requires the package versions recorded in the fixture. Preparation verifies
the local checkpoint files against the specified Hub revision and freezes the
question IDs, tokenized prompts and generation settings.

```bash
python -m pip install -r benchmarks/requirements-accuracy.txt

python -m benchmarks.gsm8k prepare --model /path/to/ouro \
  --model-repo ByteDance/Ouro-1.4B \
  --model-revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --output /path/to/gsm8k-87-protocol.json

python -m benchmarks.gsm8k run --backend native \
  --endpoint http://localhost:8000 --served-model ouro --step-log /path/to/ouro.log \
  --protocol /path/to/gsm8k-87-protocol.json --output /path/to/ouro-fixed
```

The evaluator submits each protocol's token IDs to `/v1/completions` and reads
the committed depths from the server log. To evaluate adaptive decoding, add
`--exit-threshold 0.5 --min-loops 1` and use a separate output directory.
Ouro always runs full-depth prefill. The mean decode depth excludes the first
output token.

Use `run --backend transformers` in the evaluation environment with one
visible CUDA device to run the released reference; then compare the samples
with `compare --reference /path/to/reference --candidate /path/to/ouro-fixed
--output /path/to/comparison.json`. For a short protocol check, prepare two
disjoint train examples with `--split train --limit 2`.
