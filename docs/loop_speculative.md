# Loop-level speculative execution

This opt-in round implements the mixed-depth batching part of
[Issue #43](https://github.com/ThinkFlowLab/vllm-rlt/issues/43): shallow work for
later token positions runs in the same recurrent call as deeper work for earlier
positions. It keeps the finite speculative window, the shared verifier and the
atomic round commit. The default remains the block round.

## Enable

```python
SpeculativeConfig(
    num_speculative_tokens=4,
    draft_loops=2,
    target_loops=4,
    loop_level_pipelining=True,
)
```

The offline and serving CLI accept `--speculative-tokens 4 --loop-level-pipelining`.
The restrictions of fixed-depth self-speculation apply unchanged. The loop round
is a drop-in replacement for the block round inside `SpeculativeRunner.submit`, so
it composes with deferred collection, several in-flight submissions, preemption
between rounds and CUDA graphs.

## Schedule

For K=2, d=2 (`draft_loops`) and D=4 (`target_loops`), A is the last committed
input and B and C are proposals. Numbers are zero-based depths; columns are
successive recurrent calls ("waves").

| Input position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 1 | 2 | 3 | | | | |
| B | | | 0 | 1 | 2 | 3 | | |
| C | | | | | 0 | 1 | 2 | 3 |

Offset `o` enters at wave `o*d`, after its parent's draft head chose its input, and
advances one depth per wave. A child therefore starts `d` waves after its parent,
so at every layer the same-depth KV of earlier positions is written before the
child attends to it. Rows use single-query paged attention with their own depth
plane and context length; adjacent positions are never packed into one prefill
sequence. Each position and depth is computed exactly once.

`loop_waves(token_counts, d, D)` returns the schedule. It depends only on the
token counts of the round, never on token values, so the whole round is planned
before any kernel runs.

## Execution

- `_prepare_loop_inputs` builds every wave's KV metadata and gather/scatter
  indices on the host and uploads them in one snapshot per dtype, like the block
  round's `_prepare_inputs`.
- Branch states live in one buffer whose rows follow verification order
  (contiguous by request, then offset). Each wave gathers its rows, runs one
  mixed-depth recurrent call and scatters the states back.
- After a wave, only rows that reached draft depth run the head. Their IDs (and,
  for sampling, the filtered proposal `q` and the draw) stay on the device and
  feed the children's prelude.
- After the last wave, one head call produces all `K+1` verification rows per
  request. Collection is the block round's `_collect_round`, so greedy acceptance,
  rejection sampling and the single device-to-host read are shared.

Both rounds consume each request's generator in the same order: drafts by
offset, then verification. On CPU in FP32 the two rounds produce identical tokens
for the same seeds. On GPU in BF16 the two rounds run GEMMs with different row
counts, which can select different kernels and round differently, so near-tied
argmax or sampling decisions can diverge (see the measurements). This is the
same class of difference as between batch sizes in ordinary decoding.

Greedy early-rejection pruning is not supported: deciding per wave whether to
start a child needs a host readback every wave, which the asynchronous
submission path does not allow.

## Cost model

For a uniform round (every request has `K+1` rows), both rounds launch the same
number of recurrent calls, `d*K + D`, over the same `(K+1)*D*C` rows for `C`
requests. Only the row counts per call differ. With K=4, d=2, D=4:

| Round | Recurrent calls by row count |
| --- | --- |
| Block | 10 calls of `C` rows (drafts), 2 calls of `5C` rows (verification) |
| Loop | 4 calls of `C` rows, 8 calls of `2C` rows |

With `c(M)` the cost of one recurrent call over `M` rows, the loop round is faster
only if `c(5C) - c(C) > 4 * (c(2C) - c(C))`. If `c` is affine in `M` — flat when
weight-bandwidth bound, linear when compute bound — both sides are equal, so the
schedule has no intrinsic advantage. Gains and losses come from where GEMM kernel
selection changes `c(M)` between `C`, `2C` and `5C`, which depends on the GPU,
the BLAS library and the model shape. Treat the mode as hardware-specific and
measure it on the target deployment.

Two configuration effects can masquerade as schedule gains:

- `cuda_graph_max_batch_size`: the block round's verification calls have `5C`
  rows and fall back to eager execution once `5C` exceeds the limit, while the
  loop round's `2C`-row waves still replay graphs.
- `cuda_graph_max_graphs`: ragged rounds near the end of generation produce more
  distinct wave sizes in the loop round and can exhaust the graph cache.

The benchmark reports graph fallbacks for both rounds; compare runs where both
are zero.

## Measurements (RTX 5090)

Ouro-1.4B (`574fa66c`), BF16, Triton attention, CUDA graphs with
`--cuda-graph-max-batch-size 1024 --cuda-graph-max-graphs 128` (no fallbacks in
either round), K=4, d=2, D=4, 16 prompts cycled over C requests, 128 new tokens,
7 alternating paired repeats; torch 2.13.0+cu130, Triton 3.7.1, driver 580.

Per-round time compares the schedules directly. Committed throughput also
depends on acceptance, which differs once BF16 outputs diverge.

| C | Block ms/round | Loop ms/round | Round-time ratio | `c(M)` prediction | Throughput ratio | Acceptance block / loop | Diverged request trials |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.80 | 52.97 | 0.846 | 0.846 | 0.846 | 0.884 / 0.884 | 0 / 7 |
| 4 | 56.24 | 58.98 | 0.953 | 0.955 | 0.926 | 0.839 / 0.800 | 21 / 28 |
| 8 | 59.26 | 61.28 | 0.967 | 0.980 | 1.024 | 0.804 / 0.828 | 28 / 56 |
| 16 | 63.39 | 59.97 | 1.057 | 1.227 | 1.028 | 0.824 / 0.822 | 63 / 112 |
| 32 | 63.13 | 63.14 | 1.000 | 1.067 | 1.088 | 0.800 / 0.836 | 126 / 224 |
| 64 | 75.45 | 80.56 | 0.937 | 0.922 | 0.910 | 0.823 / 0.829 | 280 / 448 |

Ratios are block time over loop time (above 1 means the loop round is faster).
Bootstrap 95% intervals of the paired throughput ratio are narrower than ±0.01
at every point. Every repeat replays the same prompts and seeds, so these
intervals reflect timing jitter only, not prompt or seed variance. C=128 did not fit with two resident engines on the 32 GB card.

The prediction uses a linear-layer-only proxy: graph-replayed time of all
linear layers of one recurrent pass for M rows (`c(1)=1.90`, `c(2..16)≈2.66`,
`c(20..32)≈2.0`, `c(64)=2.14`, `c(80..192)≈3.0-3.3`, `c(256)=4.46`,
`c(320)=4.77`, `c(640)=7.94` ms). It excludes attention, norms and the head,
so it matches best at small C. Nsight Compute shows the steps come from kernel
selection: M=1 uses a GEMV kernel, M=2-16 a 16x16 WMMA GEMM that is about 48%
slower, M=20-32 a split-K 32x32 WMMA GEMM that is faster than the GEMV, and M=80
switches to a larger tile again.

Top-p (temperature 0.8, top-p 0.8, fixed per-request seeds):

| C | Block ms/round | Loop ms/round | Round-time ratio | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 47.72 | 56.43 | 0.846 | 0.846 |
| 16 | 73.52 | 71.97 | 1.021 | 0.985 |
| 64 | 123.66 | 128.16 | 0.965 | 0.965 |

With the graph batch limit at 128 and C=32, the block round's 160-row
verification calls fall back to eager (448 recurrent fallbacks) and its round
time rises from 63.13 to 66.0 ms, while the loop round stays at 63.01 ms: a 4.7%
"gain" that is entirely the graph limit.

## Validation

```bash
OMP_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
python -m ruff format --check vllm_rlt tests benchmarks

# On an allocated CUDA device with Triton:
python -m pytest -q --run-gpu -m gpu -k 'not flash_attn_4' \
  tests/test_speculative.py tests/test_loop_speculative.py
```

`tests/test_speculative.py` runs the existing speculative contracts in both modes:
native greedy agreement with ragged rounds and budgets, the serial hidden/logit/KV
oracle, every rejection boundary and bonus, EOS, prefix reuse and incremental
allocation, seeded sampling, preemption between rounds, split prefill, deferred
collection, several in-flight submissions, and CUDA graph/eager parity.
`tests/test_loop_speculative.py` checks the schedule itself, per-round equivalence
with the block round (results, verification logits, KV and stats) for d=1/2/3 and
sampling, the head calls inside the loop, CLI/config handling, failure cleanup and
rollback.

Paired benchmark:

```bash
python -m benchmarks.loop_speculative \
  --model /path/to/local/Ouro-1.4B --prompts prompts.json \
  --cuda-graphs --cuda-graph-max-batch-size 1024 --cuda-graph-max-graphs 128 \
  --k 4 --concurrency 16 --new-tokens 128 --repeats 7 \
  --output /tmp/loop-speculative-c16.json
```

Both engines share one resident model. Prefill is excluded, trial order
alternates and every request has a fixed seed, so sampled arms draw the same
random numbers. The report contains the median of per-repeat paired ratios with a
bootstrap interval, quartiles, per-round time, acceptance, CUDA graph counters,
first divergent output positions and raw trials. Peak memory includes both
engines; the checkpoint config hash is not a weight hash.

This mode does not reduce recurrent work or the round's critical path, and does
not implement cross-round hidden-state retention or the continuous LoopSpec
algorithm from Issue #43.
