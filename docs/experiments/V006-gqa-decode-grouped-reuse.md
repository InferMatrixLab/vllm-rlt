# GQA decode: reusing the in-repo paged-prefill structure

Version V006. Branch `perf/fuse-norm-rope`. Host `lyx`, 8x A100-SXM4-80GB (108 SMs, 40 MB
L2), torch `2.14.0+cu130`, triton `3.8.0`, vLLM `0.33.0` wheel, page size 16, head_dim 128.

This entry closes the one operator gap V005 identified: GQA decode was 1.45x / 3.02x / 3.47x
slower than vLLM at 8 / 32 / 128 packed rows. The fix was not a new kernel. It reuses
`_paged_prefill_kernel`, which already solves the same problem shape, and drops a split-count
constant that had never been measured.

## What was measured before changing anything

V005 recorded the gap but not its cause. Profiling it first (ncu, 48/8 geometry, 128 rows,
2048 context, identical 1.11 GB DRAM traffic on both sides) gave:

| metric | rlt (per-head kernel) | vLLM (grouped) |
| --- | --- | --- |
| instructions executed | 778.6 M | 115.5 M |
| DRAM bytes | 1.11 GB | 1.10 GB |
| DRAM throughput | 23.1 % | 78.4 % |

So the kernels moved the same bytes; rlt executed 6.7x the instructions. The launch shape
explains it exactly: rlt ran one program per `(row, query_head, split)`, so the 6 query heads
of a KV group each walked the same KV tile. The work is redundant, not the memory.

## Reference implementation and what was adapted

Two in-repo pieces were referenced, and no kernel was written from scratch.

### 1. `_paged_prefill_kernel` (same file, `vllm_rlt/kernels/triton_attention.py`)

It already does "many query rows share one KV tile": a per-row online-softmax state
(`maximum`/`normalizer`/`accumulator`), a `[BLOCK_M, BLOCK_N]` score tile, and two `tl.dot`s
per KV tile. `_paged_attention_group_kernel` keeps that structure verbatim, including the
accumulator/correction rescale order and the FP8 dequantize-then-narrow path, and changes
only what a "row" is:

| | `_paged_prefill_kernel` | `_paged_attention_group_kernel` |
| --- | --- | --- |
| query rows | `BLOCK_M` tokens of one sequence | the query heads of one KV group |
| program id 1 | query head | **KV head** |
| mask | causal + window | none (decode's query position is the context end) |
| output | stored directly | split partials, merged by the existing merge kernel |

Two defects had to be fixed before it was correct, both caught by
`tests/test_gqa_decode.py` rather than by a benchmark:

- **KV-head mapping.** The first version took the program id as a *block of query heads* and
  derived `kv_head = first_head // HEAD_GROUPS`. With `HEAD_GROUPS=6` and a padded
  `BLOCK_H`, one block spans heads 0..15 while `kv_head 0` only covers heads 0..5, so heads
  6..15 were scored against the wrong KV head, and two programs could claim the same output
  head. vLLM avoids this by deriving the heads from the KV head
  (`cur_head = cur_head_id * VALID_BLOCK_H + arange`); the kernel now does the same
  (`heads = kv_head * HEAD_GROUPS + lane`) and masks every store by `in_group`, which makes
  each output head have exactly one owner.
- **Double normalisation.** The prefill kernel's `accumulator` is unnormalised because the
  merge kernel weights parts by their normalizers; the first version stored
  `accumulator / normalizer`, so the merge divided by the normalizer a second time. Measured
  as an output scaled by `1/4.878` for a part whose normalizer was `4.878`.

### 2. vLLM `triton_decode_attention.py` (read, not copied)

Read to confirm the contracts, not to lift code: it confirms `BLOCK_H = 16`, one program per
KV head with `VALID_BLOCK_H = min(BLOCK_H, kv_group_num)`, `BLOCK_N = 32`,
`num_warps = 4`, `num_stages = 2`, and that stage 2 is a separate merge pass. rlt's `BLOCK_T`,
`num_warps` and `num_stages` were then swept independently (16/32/64/128 x 4/8 warps x 2/3
stages) and the shipped values were already optimal, so they were left alone.

### 3. The split-count constant

`NUM_SPLITS = 16` was never measured. It is now derived from the batch shape
(`_split_count`), which keeps both properties the constant was there for: a power of two for
`tl.arange`, and a value that depends only on tensor shapes so a captured CUDA graph stays
valid. Sweeping splits 1/2/4/8/16 at 108 SMs shows 16 is never the best and the optimum tracks
a program count of roughly 512 (`rows x head_axis x splits`):

| rows | best splits (MHA) | best splits (GQA 48/8) |
| --- | --- | --- |
| 1 | 16 | 16 |
| 8 | 8 | 4-8 |
| 32 | 2-8 | 2 |
| 128 | 2-4 | 1 |

The two kernels need different bounds because their second grid axis carries different work:
the grouped kernel spends one program per KV head (covering the whole query group), the
per-head kernel one per query head. Hence `_GROUPED_SPLITS = (1, 512)` and
`_PER_HEAD_SPLITS = (4, 2048)`: the per-head kernel needs real splits even at 128 rows
(measured 1 vs 4 splits: 1.37 vs 1.27 ms at length 2048), while the grouped kernel prefers a
single part there (4 splits cost 1.11x).

## Results

### Correctness gate (runs before any timing)

`tests/test_gqa_decode.py` - 4 shipped geometries (16/16 MHA, 32/8, 48/8, 16/4) x rows
1/3/8/32/128 x lengths 1/33/512/2048, plus batch-invariance, windowing, and zero-length rows,
all against the FP32 `torch_paged_attention` oracle: **87 passed**. Full suite:
**1447 passed, 914 skipped, 0 failed**. This is the gate the first attempt lacked: it was
3.5x faster and numerically wrong (0.21 vs reference, where vLLM was 0.001).

### Operator-level, GQA 48/8, context 2048, page 16 (device ms)

| rows | before (V005) | after | vs before | vLLM unified | after / vLLM |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.173 | **0.080** | **2.16x** | 0.119 | **0.67x (rlt faster)** |
| 32 | 0.544 | **0.172** | **3.17x** | 0.180 | **0.95x (rlt faster)** |
| 128 | 2.158 | **0.653** | **3.31x** | 0.621 | 1.05x |

GQA 32/8 behaves the same (0.079 / 0.171 / 0.650), and MHA was not regressed
(0.101 / 0.357 / 1.269 at rows 8/32/128; 1.05x faster than before at rows 128).

### Where the residual difference comes from (measured, not inferred)

Timing both implementations' two phases separately at context 2048:

| rows | vLLM stage1 | vLLM merge | rlt stage1 | rlt merge | stage1 | total |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.0567 | 0.0767 | 0.0554 | 0.0153 | **0.98x** | 0.53x |
| 32 | 0.1824 | 0.0807 | 0.1804 | 0.0155 | **0.99x** | 0.74x |
| 128 | 0.7184 | 0.1227 | 0.7088 | 0.0379 | **0.99x** | 0.89x |

The attention stage is at parity (0.98-0.99x). The residual 1.05x at 128 rows is vLLM's
`unified_attention` not splitting KV at all, while rlt still launches a merge pass.

### Short context is a known, bounded gap

At context 512 the operator is 1.23-1.37x slower than vLLM's single-pass kernel, because both
rlt and vLLM sit at kernel-launch floor (~0.057 ms device, ~0.080 ms measured) and rlt pays
one extra launch. Two candidate fixes were measured and rejected on evidence:

- Inlining the merge for `splits == 1` (divide `part_o` and copy, no second kernel): correct,
  but slightly *slower* (0.1800 -> 0.1876 ms at 512/128), so it was reverted.
- Dispatching to the existing `paged_prefill_attention`, which is genuinely faster here
  (0.0448 vs 0.0791 ms at 128/8 - faster than vLLM's 0.0559) and needs no merge pass. It
  cannot be selected safely yet: which path wins depends on `rows x context_length`, and
  context lengths live on device (the CUDA-graph path fills `context_lengths` into a
  preallocated buffer), so a host-side branch on the length would either force a device sync
  or be wrong under graph capture. The decision needs a measured
  host-length-at-graph-capture contract before it can ship.

### End to end

`bench_e2e.py --prompt 512 --decode 128 --batch 8` (Ouro-1.4B, CUDA graphs on): 14.587 s
after vs 14.553 s before, 70.2 vs 70.36 tok/s - **inside run-to-run noise (0.2%)**. That is
expected: V002's per-step budget attributes 0.85 ms of 13.86 ms to attention, so a 2-3x
attention improvement moves the loop by roughly 1%, and the Ouro path is MHA, which this
change does not alter.

## What this says about the operator layer

- The GQA gap was a launch-shape bug with a redundant-work cause, not a missing optimization:
  the same bytes were moved 6.7x more often than necessary.
- The reusable implementation was already in the repository. Adapting
  `_paged_prefill_kernel` needed ~40 lines of structural change (row meaning, mask, store
  target) rather than a new kernel.
- An unmeasured constant (`NUM_SPLITS = 16`) cost up to 1.6x depending on shape. Deriving it
  from shapes kept the numerical contract (bitwise-identical to the old split alignment at the
  same split count, verified by the batch-invariance test) and removed the waste.

## Reproduce

```bash
# correctness gate
/root/kun/vllm-omni/.venv/bin/python -m pytest tests/test_gqa_decode.py -q -p no:randomly

# operator-level comparison (rlt vs vLLM decode vs vLLM unified), packed rows
/root/kun/vllm-omni/.venv/bin/python /tmp/pd2.py --rows 8,32,128 --length 512,2048 \
    --geom ouro-mha,nanbeige-gqa6,gqa4 --output out/pd_final6.json

# phase attribution against vLLM
/root/kun/vllm-omni/.venv/bin/python /root/kun/rlt-perf/phase_attr.py

# splits sweep (the data behind _split_count)
/root/kun/vllm-omni/.venv/bin/python /root/kun/rlt-perf/split_len.py

# end to end
/root/kun/vllm-omni/.venv/bin/python bench/bench_e2e.py --prompt 512 --decode 128 --batch 8
```

Scripts are kept at `C:\Users\kunku\.pi-desktop\scratch\<session>\` locally and mirrored under
`/root/kun/rlt-perf/` on `lyx`.
