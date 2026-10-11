# Ouro LoopCD scale qualification

This continuation starts at PR #86 head
`d700ecca664da2e8f3122f66ad9d45de4295e45a`. Production model, engine and
feature guards are unchanged. Nonzero guidance still requires synchronous
LAST_EXITED execution. Running the experiment as a background job does not
enable asynchronous engine scheduling.

## Independent continuation

`tests/reference/continuation.py` retains independent K/V tensors for each
depth and layer. The prompt runs at P4; eight subsequent teacher-forced inputs
run at D2 or D3. Skipped depths receive only the newly computed positions from
the last executed depth. The P4 prompt history is retained. The final sampled
output is not forwarded and therefore does not create a KV position.

The 12 FP32 CPU cases cover prompt lengths 3/4/5 around the block/chunk size4,
both decode depths, and guidance strengths0/.3. They compare recurrent state,
every historical K/V plane, vocabulary logits and selected log probability
with fixed atol=rtol=1e-4. They use a tiny initialized model, not official
checkpoint weights. There is no new BF16 tolerance or quality claim.

## Closed-loop driver

`python -m benchmarks.loopcd_scale --help` lists the implemented interface.
Each trial submits C outstanding requests, immediately refills completed slots
until 4C requests have completed, then drains. Request IDs are reused after
consuming the full output batch. Fixed output count, actual P/D depths, finish
reason, repeat token identity, empty references and full KV reclamation are
checked. Every request and trial is retained, including a failing trial.

The driver reports scheduler B, configured/outstanding C, observed resident A,
and actual decode core rows S separately. A CPU capacity regression tests
B16/C32, B32/C64 and B64/C128 with both full B residency and an eight-request
physical KV limit. It asserts actual A and S reach the expected residency,
alongside queued/refill behavior and exact repeat outputs.

Each process uses one model/engine, five warmups and five measured trials.
TTFT, TPOT, ITL, admission delay, completion latency, TPS/RPS, native work,
physical KV pool/peak, reference high water, CUDA allocated/reserved peaks and
Graph capture/replay/fallback deltas are recorded. Percentiles use nearest
rank. A cold capture during measured work invalidates the case; a fallback is
charged to elapsed time. Model loading, tokenization and diagnostics are
outside the native timing window. Prompts are deterministic tokenized repeated
text: these are fixed-work load tests, not task-quality or HTTP SLO estimates.

Example, after resource admission and official input verification:

```bash
python -m benchmarks.loopcd_scale \
  --model /absolute/path/to/pinned-ouro \
  --batch 32 --concurrency 64 --input-lengths 512 2048 \
  --output-tokens 128 --depth 4 --strength 0.3 --kv-gib 60 \
  --output /absolute/path/to/new-result
```

The prepared H20 queue covers B/C=1/1,32/32,32/64,64/64,64/128 and a32/32
input2048 case. It compares P4D4 off/two_head in eager/Graph mode with two
fresh process starts and reversed configuration order. The fixed60GiB KV pool
targets the observed96GB H20 and is not suitable for the currently occupied
Thor. It is a planned pool, not an observed capacity or OOM result.

No reduced-depth quality candidate has been selected. P4D2/P4D3 CLI options
support later separate qualification; the prepared timing queue does not
claim quality-preserving depth reduction. The earlier BF16 linear_fused
failure is unchanged; two_head remains the guidance baseline.

## Actual validation status, 2026-10-06

- Local: independent continuation12 passed, capacity/refill6 passed, existing
  LoopCD regression62 passed. Earlier three refill cases also passed before
  tightening the capacity coverage and are not added to the final case count.
- Thor: original12 continuation plus3 refill cases passed in12.27s; CUDA was
  uninitialized before/after. The stronger6-case capacity run also passed in18.20s with CUDA hidden.
  Both controllers and test children exited naturally0.
- Official-weight high-concurrency GPU execution: **NOT_RUN**. The H20 lacks
  the public loader/tokenizer dependencies; isolated-environment authorization
  is pending. The inherited Thor Qwen job retains its original GPU lock.
- These additions are preparation/correctness evidence. They do not close
  public loader qualification, official mixed-depth correctness, quality,
  high-concurrency GPU performance or Task4.
