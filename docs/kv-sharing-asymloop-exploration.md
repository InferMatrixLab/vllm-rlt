# Not qualified: BeNice! KV sharing and AsymLoop on Ouro-1.4B

**Status: not qualified (both explorations).** Neither is proposed for merge. This
page records what was measured, so that the two routes are not repeated without
a new reason, and how to reproduce the measurements.

| Exploration | Paper | What would have to hold | Outcome |
| --- | --- | --- | --- |
| BeNice! KV sharing | [arXiv 2610.31633](https://arxiv.org/abs/2610.31633) | Shared final-recurrence KV raises high-load throughput and lowers TTFT without hurting low load or quality | Not qualified: overload throughput +28–34%, but TPOT 1.7–3.4× worse, low-load parity and prefix-hit criteria failed, GSM8K-87 accuracy 60/87 → 0/87 |
| AsymLoop | [arXiv 2610.28489](https://arxiv.org/abs/2610.28489) | A one-pass encoder plus a looped decoder serves more load than Ouro's full loop | Not qualified: same loop count gives 1.33× KV capacity but only 1.08–1.10× throughput at 16K and 0.99× on a Mooncake trace; the paper's depth-matched setting gives 0.48–0.63× |

Both are **serving proxies on Ouro-1.4B weights**. Neither paper releases weights
for these architectures, so quality numbers say nothing about trained BeNice! or
AsymLoop models. BeNice!'s own Table 1 reports that Ouro-1.4B is not shareable
without post-training.

## Setup shared by both

- One NVIDIA L40S (48 GB) per run; Ouro-1.4B at revision
  `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, bf16; torch 2.13.0+cu130, Triton 3.7.1.
- KV pool pinned to 30 GiB in every arm, so capacity differences come only from
  the layout. Triton attention, block size 16, refill scheduling,
  `max_num_seqs` 64, 2048 batched tokens, 2048-token prefill chunks.
- `ouro_delayed` exit with threshold 1.0, so every token runs the full depth.
- Workloads:
  - **W1**: bursts of C fresh random prompts of 8K or 16K tokens, C ∈ {1, 8, 32}.
  - **W2**: open-loop replay of the Mooncake FAST'25 `toolagent_trace.jsonl`, with
    512-token hash blocks mapped to deterministic token blocks, so the trace's
    prefix reuse is real. Prompts are capped at 16K and arrivals rescaled to a
    fixed offered rate.
- Hypotheses and thresholds were written before any GPU result. Every cell ran
  once unless a replicate is listed.
- Code: branch
  [`research/benice-asymloop-20261026`](https://github.com/0z5a/vllm-rlt/tree/research/benice-asymloop-20261026)
  on a fork, three commits on `8fbf4cd`:
  - `6ae0277`: loop-order shared prefill;
  - `8cfd342`: AsymLoop;
  - `4f23cd6`: benchmark drivers.

## BeNice! KV sharing

BeNice! lets every recurrence read the final recurrence's KV, so the cache keeps
one plane per token instead of one per loop. vllm-rlt's existing `shared` layout
has this semantics, but its prompt runs position by position, and the layout
rejected prefix caching. The research branch adds two things:
- `CacheConfig.shared_prefill="loop"`: a shared-KV prompt chunk advances one loop
  at a time through packed prefill.
  - Inside a chunk, loop d reads the chunk's own loop-d KV; earlier chunks are
    read at their final KV.
  - The result therefore depends on chunk boundaries. A single chunk is bitwise
    equal to `last_exited`.
- Prefix caching for `shared` KV.

**Arms.** Both use async scheduling, CUDA graphs, static buffers and packed
prefill (`prefill_uva`), with prefix caching on.
- **A0**: `last_exited`, 40,960 tokens in 30 GiB.
- **B1**: `shared` with `shared_prefill="loop"`, 163,840 tokens (4.00×).

| Criterion (pre-registered) | Cell | Result | Verdict |
| --- | --- | --- | --- |
| Capacity 4× | 30 GiB | 40,960 → 163,840 tokens | pass |
| Low-load parity: TTFT and TPOT ratio in [0.9, 1.1] | W1 8K C=1 / W1 16K C=1 | 1.03 and 1.02 / 0.98 and 0.99 | pass / pass |
| Low-load parity | W2 0.1 req/s | TTFT p50 0.89, TPOT p50 1.74 | fail |
| High load: throughput ≥ 1.15× and p90 TTFT ≤ 0.67× | W1 8K C=32 | 1.31×, 0.58× | pass |
| High load | W1 16K C=32 | 1.32×, 0.72× | fail |
| High load | W2 0.6 req/s | 1.28×, 0.42× (replicate on a second host: 1.30×, 0.39×) | pass |
| Prefix hit +5 points | W2 0.6 req/s | 22.8% → 25.0% (replicate 24.0% → 24.4%) | fail |
| Quality (expected to fail on Ouro) | GSM8K-87, greedy | 60/87 → 0/87, −69.0 pp (paired SE 5.0) | fail |

TPOT at high load went from 0.078 to 0.244 s (W1 8K C=32), from 0.071 to 0.233 s
(W1 16K C=32) and from 0.095 to 0.319 s (W2 0.6). Raising the CUDA-graph cache cap
from 16 to 64 removed B1's graph fallbacks but changed TPOT by less than 3%.

In the GSM8K run, 86 of 87 first tokens matched A0. Decoding then degenerates into
repetition within a few tokens: the prefill change is not the cause; reading
final-loop KV during decode is.

**Why it does not qualify.** Sharing saves KV memory but not KV reads.
- ncu on the decode attention kernel (W1 16K, C=8):
  - A0, 2 rows: 422 µs per launch;
  - B1, 8 rows: 1.70 ms per launch;
  - both at ~85% of DRAM peak, SM throughput ~7.5%, L2 hit rate ~0.3%.
- The extra capacity therefore becomes larger decode batches whose step time
  grows with rows × context.
- Throughput gains appear only past saturation, and they cost a 1.7–3.4× TPOT
  regression.
- On Ouro the quality loss is total.

## AsymLoop

AsymLoop runs the first E layers once per token as an encoder. Each decoder layer
projects the encoder output once to a fixed global KV. The D decoder layers loop
K times, and each one attends to the global KV and the current loop's KV under
one causal softmax. The proxy uses E = D = 12 on Ouro's 24 layers.

**Implementation** (`8cfd342`):
- **KV planes.** `KVCacheManager(aux_planes=2)` adds an encoder plane and a
  global plane next to the per-depth planes.
  - The pool is cut into page groups: plane p of group g is block p·G + g.
  - Any depth's block table therefore maps to the same page's encoder and global
    blocks by arithmetic, and mixed-depth batches need no new metadata.
- **Attention.** Paged attention (the Torch reference, Triton decode and packed
  prefill) gains `sources=2`: two tables with the same causal length under one
  online softmax. `sources=1` is unchanged.
- **Model.** `AsymLoopOuroForCausalLM` uses the KV-bearing prelude path, which
  this branch opens to packed prefill:
  - prelude = embedding, encoder, and global KV writes;
  - recurrent = the decoder;
  - coda = the LM head.
- **Global projections.** Ouro has none, so each starts as a copy of its decoder
  layer's K/V projections, applied after that layer's input norm.
- **Not wired.** Asynchronous scheduling and CUDA graphs are not supported for
  this model.
- **Correctness.** CPU and GPU tests compare the engine (chunked, mixed-depth,
  with and without `prefill_uva`) token for token with an unpaged AsymLoop
  reference.

**Arms.** All three use synchronous scheduling, eager execution, packed prefill
and no prefix cache. The baseline was re-measured in this configuration, so
these numbers are not comparable with the BeNice! table.

| Arm | Structure | KV slots per token | Layer passes per token | Decode KV read per token (× context) | Tokens in 30 GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| A0s | Ouro full loop, K = 4 | 96 | 96 | 96 | 40,960 |
| AL4 | AsymLoop, K = 4 (same loop count) | 72 | 60 | 108 | 54,608 |
| AL7 | AsymLoop, K = 7 (same effective depth 96, the paper's matching rule) | 108 | 96 | 180 | 36,400 |

The workloads were W1 16K (C = 1, 8 and 32, 128 output tokens) and W2 at 0.6 req/s
(60 requests, at most 128 output tokens). Ratios are against A0s.

| Criterion (pre-registered) | Arm | Cell | Result | Verdict |
| --- | --- | --- | --- | --- |
| Capacity matches the table | all | 30 GiB | exact | pass |
| Single-request TPOT ratio in [0.80, 1.00] / [1.30, 1.70] | AL4 / AL7 | W1 16K C=1 | 0.78 / 0.87 | fail / fail (not interpretable, see below) |
| Single-request TTFT ratio in [0.75, 0.95] / [1.20, 1.50] | AL4 / AL7 | W1 16K C=1 | 0.87 / 1.49 | pass / pass |
| Throughput ≥ 1.15× | AL4 | W1 16K C=32 | 1.10× (replicate 1.08×) | fail |
| Throughput ≥ 1.15× | AL4 | W2 0.6 | 0.75× (warm replicate 0.99×) | fail |
| Throughput ≤ 0.90× | AL7 | W1 16K C=32 / W2 0.6 | 0.63× / 0.61× (0.48× against the warm A0s replicate) | pass / pass |

AL7 passing means the predicted loss was confirmed. In the setting where the
paper claims its quality advantage, serving is clearly more expensive.

**Where the time goes.** These are post-hoc diagnostics and are not scored.
- **ncu.** On W1 16K C=4, every decode attention launch costs ~210 µs per row and
  per KV source: A0s 2 rows 423 µs, AL4 decoder 3 rows × 2 sources 1.25 ms, AL4
  encoder 1 row 213 µs. SM throughput is 7–8%.
  - Per token at 16K this is 20.3 ms for A0s and 22.8 ms for AL4, exactly 108/96.
  - The two-source kernel adds nothing beyond the extra bytes.
- **Per-step timing, W1 16K C=32 replicate.**
  - Prefill: AL4 spends 55 s, A0s 69 s.
  - Decode loops: about equal, 121 s with 2.86 rows per step against 153 s with
    1.98 rows.
  - AL4 also runs the encoder as a separate prelude step for every new token:
    28 s in total, 8.8 ms per step, 1.27 rows per step. In A0s the prelude is an
    embedding lookup (0.7 s).
  - The capacity gain is spent on two-source reads and on this step.
- **Single request.** AL4 is faster at C=1 (warm TPOT 0.048 s against 0.087 s)
  only because the eager path pays a fixed host cost per layer, and AL4 runs 36
  fewer layers per token.
- **cProfile.** On W2 the synchronous eager engine is host-bound: only 29 of
  169 s wait on the GPU; the rest goes to per-row, per-layer written-position
  and prefix bookkeeping and eager RMSNorm.

**Confound found after scoring.** `_paged_attention_kernel` declares
`table_stride` (the block-table width) as `tl.constexpr`.
- Every new table width compiles a new kernel variant, and `sources=2` adds its
  own variants.
- The per-cell warm-up uses 512-token prompts and does not reach 16K widths, so
  the first cell at those lengths compiled inside the timed window. This was
  seen in the cold run (`cc1plus` busy) and as 300–490 ms steps in a W2 replicate.
- The single-request TPOT ratios and the first W2 run are cold-versus-warm
  confounded across arms. The pre-registered verdicts above are kept as scored.
- TTFT (the prefill kernel's stride is not a constant) and W1 C=32 (its widths
  were compiled by the earlier cells; the replicate agreed within 3%) are not
  affected.

## Notes for related engine work

- KV sharing or extra KV sources move work from capacity to KV bandwidth. On
  L40S, decode attention is DRAM-bound at ~210 µs per 16K row and per source.
  Any proposal should show decode-step cost at the target batch size, not only
  capacity.
- A model whose prelude carries KV adds a per-token engine step in decode that
  batches poorly. AsymLoop-style designs need that step fused into the
  recurrent step, or graph-captured, before they can pay off.
- In eager runs, warm up at the measured context lengths, or make the Triton
  decode `table_stride` a runtime argument; otherwise each new width recompiles
  during measurement.

## Reproducing

Use the fork branch above. Install the repository as usual, download Ouro-1.4B at
the pinned revision, and download the trace from
`https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl`.
Run on a single GPU, with `CUDA_VISIBLE_DEVICES` set to one device. Every command
writes one JSON record with a `CAPACITY` line, per-cell latency and throughput,
and optional per-step timing (`--step-trace`).

Engine flags for the BeNice! arms (async, CUDA graphs):

```bash
COMMON="--model /path/to/Ouro-1.4B --attention triton --dtype bfloat16 --block-size 16 \
  --kv-bytes 30 --max-num-seqs 64 --max-batched-tokens 2048 --chunk 2048 --mode refill \
  --async --cuda-graphs --graph-max 64 --static-buffers --pad-pow2 --prefill-uva \
  --exit-mode ouro_delayed --exit-threshold 1.0 --prefix-cache"
A0="--kv-layout last_exited --shared-prefill position"
B1="--kv-layout shared --shared-prefill loop"
```

Engine flags for the AsymLoop arms (synchronous, eager, no prefix cache):

```bash
COMMON="--model /path/to/Ouro-1.4B --attention triton --dtype bfloat16 --block-size 16 \
  --kv-bytes 30 --max-num-seqs 64 --max-batched-tokens 2048 --chunk 2048 --mode refill \
  --prefill-uva --exit-mode ouro_delayed --exit-threshold 1.0 --kv-layout last_exited"
A0s=""
AL4="--asymloop-encoder 12 --asymloop-loops 4"
AL7="--asymloop-encoder 12 --asymloop-loops 7"
```

Workloads (the BeNice! run used decode 256, 30/60/90 requests for W2 at
0.1/0.3/0.6 req/s, and also W1 8K; the AsymLoop run used decode 128 and 60
requests):

```bash
python -m benchmarks.kv_sharing_asymloop $COMMON $AL4 --label AL4-w1 --out AL4-w1.json \
  --workload burst --prompt-len 16384 --decode 128 --concurrency 1,8,32 --warmup 1
python -m benchmarks.kv_sharing_asymloop $COMMON $AL4 --label AL4-w2 --out AL4-w2.json \
  --workload mooncake --traces toolagent_trace.jsonl --tokens-per-block 512 \
  --max-prompt 16384 --max-decode 128 --requests 60 --concurrency 0.6 --warmup 1
```

Run every arm on the same GPU. Before an eager measurement, run one cell at the
target lengths to keep Triton compiles out of the timed window.

For the BeNice! quality gate, prepare the GSM8K-87 protocol as described in
[accuracy evaluation](accuracy.md), then run both layouts and compare them:

```bash
python -m benchmarks.kv_sharing_asymloop_gsm8k --protocol protocol.json --out A0.json \
  --label A0 --max-num-seqs 32 --kv-layout last_exited --shared-prefill position
python -m benchmarks.kv_sharing_asymloop_gsm8k --protocol protocol.json --out B1.json \
  --label B1 --max-num-seqs 32 --kv-layout shared --shared-prefill loop
python -m benchmarks.kv_sharing_asymloop_gsm8k --compare A0.json B1.json
```

For the decode attention profile, run ncu on one cell with
`--kernel-name regex:_paged_attention_kernel` and a small `--launch-count`, about
24. Kernel replay backs up the 30 GiB KV pool for every profiled launch.
