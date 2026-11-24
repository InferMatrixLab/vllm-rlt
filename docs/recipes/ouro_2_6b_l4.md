# Ouro-2.6B on NVIDIA L4: bounded CUDA Graphs validation

Base `ByteDance/Ouro-2.6B` shares the native Ouro implementation with
`Ouro-1.4B` and the [Ouro-Thinking checkpoints](ouro_thinking.md). This note
records a bounded BF16, fixed-four-loop validation on one 24 GiB NVIDIA L4
(SM89) comparing synchronous eager execution against CUDA Graphs at
concurrency 8.

## 1. Environment and pinned assets

- Repository commit: `ce200f0d2085fb041c66691245e29aef2a9ed714` (clean tree).
- Checkpoint: `ByteDance/Ouro-2.6B` at revision
  `1ed04250da1a9936042725d302e81c8fa2ab5abd` (BF16, `total_ut_steps = 4`).
- Dataset: `openai/gsm8k` (`main`/`test`) at revision
  `740312add88f781978c0658806c59bc2815b9866`.
- Hardware: one NVIDIA L4 (22,563 MiB reported by CUDA, 58 SMs, driver
  580.82.07).
- Software: PyTorch `2.14.1+cu130` (CUDA 13.0), Triton `3.8.0`, Transformers
  `4.55.0`, `lm-eval` `0.4.9.2`, `datasets` `3.6.0`, `tokenizers` `0.21.4`.

## 2. Configuration

The comparison holds every runtime setting fixed except `cuda_graphs`:

```python
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
)
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_path, device="cuda", dtype=torch.bfloat16
).eval()

engine = StageEngine(
    model,
    attention_backend="triton",
    cache_config=CacheConfig(num_blocks=1248, block_size=16, layout="last_exited"),
    scheduler_config=SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=512,
        prefill_chunk_size=512,
        mode="refill",
    ),
    exit_config=ExitConfig("ouro"),
    execution_config=ExecutionConfig(
        async_scheduling=False,
        cuda_graphs=True,  # False for the eager baseline
    ),
)
```

- KV sizing: `8 * 4 * ceil((360 + 256) / 16) = 1248` blocks for 8 resident
  requests, 4 loops, max prompt length 360 tokens, and 256-token output cap.
- Resolved defaults: `packed_prefill=True`, `use_uva=True`,
  `cuda_graph_max_batch_size=128`, `cuda_graph_max_graphs=16`,
  `cuda_graph_memory_reserve_bytes=1 GiB`.
- Decoding: greedy (`temperature=0`), fixed four loops
  (`min_loops=4, max_loops=4, exit_threshold=1.0`), no adaptive exit, prefix
  caching, async scheduling, or speculation.

## 3. Protocol and measured results

The workload uses 32 fixed GSM8K `test` questions (the first 32 sorted IDs from
the seed-0 100-question selection), formatted with the 3-shot CoT raw prompt
from [accuracy evaluation](../accuracy.md) (no chat template, natural EOS plus
harness stop sequences, 256-token output cap, 2,048-token context limit,
`lm_eval_strict` scoring). Requests are submitted in four waves of eight; each
arm warms up on the full 32-question cohort before measurement.

All four measured passes below ran on the same L4 allocation in bracketed order
(`B1 -> C (x2) -> B2`) to check for time drift:

| Pass | Mode | Wall time (s) | Throughput (tok/s) | Req/s | p50 req (s) | p95 req (s) | Strict correct | Peak alloc / reserved (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1` | C=8 eager | 201.00 | 16.55 | 0.159 | 28.91 | 48.48 | 22/32 | 12.32 / 12.74 |
| `C` repeat 1 | C=8 graphs | 95.66 | 34.78 | 0.335 | 14.95 | 23.11 | 22/32 | 12.33 / 12.76 |
| `C` repeat 2 | C=8 graphs | 95.60 | 34.80 | 0.335 | 14.94 | 23.08 | 22/32 | 12.33 / 12.76 |
| `B2` | C=8 eager | 198.75 | 16.74 | 0.161 | 28.73 | 48.13 | 22/32 | 12.32 / 12.74 |

Observations on this fixed cohort:

- **Exact output parity:** `B1`, `C` repeat 1, `C` repeat 2, and `B2` generated
  3,327 total output tokens each with **32/32 identical token IDs, decoded
  text, and exit depths** (zero token divergences, zero answer flips, zero
  length-cap truncations; 8/32 unparseable under strict matching in every pass).
- **Graph coverage:** warmup captured 8 graphs; each measured `C` pass recorded
  2,809 replays and 0 fallbacks.
- **Wall time and memory:** wall-time ranges do not overlap (198.75–201.00 s
  eager vs 95.60–95.66 s graphs, a 2.08–2.10x wall-time ratio on this C=8 wave
  load). Peak allocated memory increased by 0.01 GiB (12.32 to 12.33 GiB).
- **Separate-session C=1 reference:** on the same commit, protocol, and GPU
  class in earlier L4 sessions, native C=1 eager scored 20/32 (810.50–823.98 s
  across two repeats) and official HF Transformers C=1 scored 20/32 (899.92 s).
  Between native C=1 and native C=8, 8/32 questions had BF16 sequence
  divergences (0 strict regressions). Because C=1 and C=8 ran on separate L4
  allocations, those numbers are context only, not a same-allocation paired
  comparison.

## 4. Qualification limits

- This is a bounded 32-question, 256-token-cap, C=8 wave check on one L4 with
  Triton attention. It does not qualify full-dataset accuracy, general serving
  capacity, online arrival latency, or other GPUs/backends.
- Per-request p50/p95 latencies above are full-completion times within an
  8-request wave, not single-request latency or TTFT.
- Exact token agreement between C=8 eager and C=8 CUDA Graphs is observed on
  this 32-question cohort and does not guarantee general numerical equivalence
  across other batch sizes, sampling settings, or prompt lengths.
- Adaptive exit, asynchronous scheduling, self-speculation, prefix caching,
  preemption, and PD remain unvalidated for Base `Ouro-2.6B` here.
