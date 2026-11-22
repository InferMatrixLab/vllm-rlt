# Ouro-1.4B on RTX 5090 with FlashInfer XQA

## Setup and launch

Use one NVIDIA RTX 5090 (SM120), the BF16 Ouro-1.4B checkpoint and page size
16. FlashInfer `0.7.5.post1` supports SM120 with CUDA 12.8 or newer.
Set `CUDA_HOME` to the toolkit that will compile XQA. See the
[toolkit and JIT requirements](../flashinfer.md#installation-and-runtime-requirements).

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL=/path/to/Ouro-1.4B
export RESULTS=/path/to/results
mkdir -p "$RESULTS"
python -m pip install -e '.[flashinfer]' 'flashinfer-python==0.7.5.post1'
hf download ByteDance/Ouro-1.4B --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
    --local-dir "$MODEL"
vllm-rlt --model "$MODEL" --device cuda --dtype bfloat16 --attention-backend flashinfer \
    --block-size 16 --no-packed-prefill --min-loops 4 --max-loops 4 --max-tokens 16 \
    --prompt "What is 2 + 2?"
```

The startup record identifies the selected kernel as `xqa`, its FlashInfer
version and page size. To exercise the calibrated E4M3 KV cache, add
`--kv-cache-dtype fp8_e4m3`. Q and output stay BF16. The accuracy and
throughput runs below use BF16 KV.

## Accuracy

Use the [frozen GSM8K-87 protocol](../accuracy.md#prepare-and-run-the-default-case)
and the [matched native commands](../accuracy.md#attention-backend-runs).
Both backends use four loops, greedy generation, per-row prefill and the same
strict scorer. A failed gate writes the run summary and returns a nonzero exit
status. The [per-question results](../flashinfer.md#accuracy-note-gsm8k-87-single-greedy-pass)
record the numerical and answer-format losses.

## Matched throughput

```bash
python -m benchmarks.ouro_xqa_throughput --model "$MODEL" \
    --output "$RESULTS/throughput" --concurrency 1 8 32 --pairs 3
```

Each trial processes twice the concurrency in closed-loop replacement, with
512 prompt tokens and 128 generated tokens per request, four loops, eager
execution and per-row prefill. One full wave at the measured concurrency warms
each backend before each timed trial.
Timing covers initial admission, prefill, replacement and drain. Backend order
alternates across pairs. The result directory contains the checkpoint hashes
and package versions in `metadata.json`, the individual timings and token counts
in `trials.jsonl` and median output tokens per second in `medians.json`.

On 2026-11-22, the RTX 5090 ran PyTorch `2.13.0+cu130`,
Triton `3.7.1` and FlashInfer `0.7.5.post1` with NVIDIA driver
`580.76.05`. Values are median output tokens/s with the three-trial
range in parentheses. All 18 timed trials completed with the expected count
and exit depths, producing 62,976 output tokens in total.

| Concurrent requests | Triton | FlashInfer XQA | XQA vs Triton |
| ---: | ---: | ---: | ---: |
| 1 | 15.9 (15.8–16.2) | 16.9 (16.7–16.9) | +6.2% |
| 8 | 108.2 (108.1–108.2) | 115.7 (112.2–116.1) | +6.9% |
| 32 | 317.3 (313.9–320.0) | 334.0 (330.6–334.3) | +5.3% |
