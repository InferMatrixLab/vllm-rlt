# Local preemption benchmarks

`benchmarks.preemption` compares preemption off/on with the same seeded arrivals,
token prompts, fixed output limits, KV budget and priority policy. It covers all
eight combinations of equal/mixed priority, uniform/mixed short-long requests,
and low/high offered load. Higher-priority arrivals use priority 0; lower-priority
requests use 10. The first active wave has lower priority in mixed-priority cells.

These are controlled scheduling workloads on a real local Ouro checkpoint,
not a language-quality evaluation. Uniform requests use 128 input/64 output
tokens. Mixed lengths use 256/128 and 32/16. Greedy decoding ignores EOS so
both arms do the same requested work. Paired output token IDs and exit depths
must match; all partial records and failures remain in `raw.jsonl`.

Choose and freeze the two request rates using a separate pilot on the actual
platform. Low/high are offered-load labels: inspect the measured waiting queue
and lateness to confirm that they exercise the intended regimes. Do not adjust
rates after looking at the paired improvement. Run only inside the existing
resource reservation, using an already verified local checkpoint and runtime.

```bash
python -m benchmarks.preemption --model /path/to/Ouro-1.4B \
  --output results/preemption-start0 --low-rps 0.5 --high-rps 4 \
  --pairs 5 --seed 1729 --profile-case mixed_mixed_high
python -m benchmarks.preemption --model /path/to/Ouro-1.4B \
  --output results/preemption-start1 --low-rps 0.5 --high-rps 4 \
  --pairs 5 --seed 2718 --profile-case mixed_mixed_high
```

The example rates are placeholders, not measured platform capacity. Each command
is a fresh process. Use `--decode native` to check ordinary decoding separately,
and `--cuda-graphs` for a separate graph-enabled comparison. A `--plan-only`
invocation imports no Torch or model. `--case` can select a diagnostic subset;
its summary records incomplete matrix coverage.

Each arm constructs its own engine and warms the same shapes before measurement.
Pair order alternates off/on and on/off. Timed runs retain per-request arrival,
enqueue, first output and completion times; whole-run throughput; peak PyTorch
memory; and preemption/resumption counts. TTFT and E2E include enqueue lateness
relative to the frozen arrival schedule. Enqueue-relative values are reported
separately. Mean, median and p95 metrics are broken down by priority and length;
`summary.json` retains each pair's on/off ratio without discarding slow values.
Model loading, checkpoint hashing and warmup are outside timing.

Optional profiling runs occur after the timing matrix. They use the repository's
[PyTorch profiling interface](profiling.md), with shapes, stacks and memory
enabled for a finite step window. Capture and artifact-worker completion are
joined before exit. Inspect each profile status and its recorded window: a short
window may miss the actual preemption. Instrumented timing is excluded from the
performance summary. `--profile-only --profile-case mixed_mixed_high` captures
that case without rerunning a completed timing matrix. Profiling artifacts retain host identity and must stay
private when the platform's disclosure policy requires that.

`SchedulerConfig.enable_preemption` remains `False`. The harness never changes
that default. Enabling it by default needs consistent TTFT benefits with a small
E2E penalty across this matrix and both process starts, including the affected
priority/length groups. CPU regressions and unexecuted benchmark plans provide
no evidence for that change.
