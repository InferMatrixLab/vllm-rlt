# Dory baseline

This page documents the native Dory implementation added for
[NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
and the exact scope that is qualified today. It describes a **baseline
increment**, not full support: the plan for continuous serving, cache
management, execution and speculation is tracked in
[RFC #176](https://github.com/ThinkFlowLab/vllm-rlt/issues/176).

## Model layout

Dory is one physical stack of 72 sublayers. Each sublayer holds a single mixer,
either attention or MLP, plus an NGPT slerp residual:

| Stage | Sublayers | KV |
| --- | --- | --- |
| Input (`n_input_layers`) | 16 (8 attention, 8 MLP) | boundary KV at depth 0 |
| Recurrent (`n_recurrent_layers`) | 40 (20 attention, 20 MLP) | per loop depth, or one shared plane |
| Output (`n_output_layers`) | 16 (8 attention, 8 MLP) | boundary KV at depth 0 |

The recurrent stage runs `n_recurrent_loops` times. Before every loop the
retained input-stage output is combined with the state and re-normalized, so
the runner carries `[recurrent state, injection input]` and
`DoryConfig.hidden_size` is the doubled carry width while
`DoryConfig.d_model` is the artifact width. `config.json`'s `hidden_size` maps
to `d_model`.

NGPT arithmetic is preserved explicitly: FP32 `justnorm` accumulation, learned
per-channel scales stored without their constant factors, post-RoPE Q/K
normalization, the output gate before the projection, the MLP gate/up scaling,
and slerp residuals with `ngpt_alpha_init`. Logits are scaled by the learned
vocabulary-wise `lm_head.logit_scale`.

## KV modes

`recurrent_kv_cache_mode` selects between two different semantics, and each maps
to one cache layout:

| Mode | Runtime layout | Recurrent attention history |
| --- | --- | --- |
| `per_loop` (default) | `last_exited` | Independent KV per loop depth |
| `last_loop` | `shared` | One plane per layer, overwritten by each loop |

Boundary stages always write depth 0, so input and output KV are never
replicated per loop. In `last_loop` mode a token's current-step KV is overwritten
by the next loop, so results depend on the chunk schedule; the qualified
schedule is one token per step. `per_loop` gives the same result for
token-at-a-time and all-positions-at-once execution.

## Supported today

- Native strict safetensors loading, including packed projection key mapping and
  the learned scale tensors.
- Synchronous eager execution: full-prompt prefill followed by ordinary token
  decoding, with the configured fixed loop count.
- `loops=` at load time to fix 1, 2 or 4 loops.
- Greedy and sampled decoding, request seeds, EOS (`2`, `11`) and length
  stopping through the existing engine.

## Rejected today

These fail fast rather than degrade silently:

- WCPB, mixed or staged prefill; asynchronous execution; PD.
- CUDA Graphs (the sliding-window path needs host-planned KV metadata).
- Prefix caching, snapshots and preemption of Dory requests.
- Self-speculation.
- Per-request loop counts other than the configured full depth: Dory has no
  trained exit gate, so the runtime refuses to invent one.

Sliding-window masking is implemented in the **Torch reference attention
backend** only (`window=` on `KVCacheManager.attend`). Attention backends
without masking raise instead of silently attending the full history; adding
windowed Triton/Flash kernels and SWA storage reclamation is follow-up work.

## Validation status

`tests/test_dory.py` covers configuration validation, strict loading with
single-file and sharded checkpoints, stage/KV contracts, sliding-window masking,
engine scheduling, and numerical agreement with an independent dense oracle
(`tests/reference/dory.py`) for both KV modes — including a greedy generation
comparison against that oracle.

Not yet performed, and required before claiming support:

- Real-checkpoint BF16 comparison with the pinned upstream implementation
  (logits, generated outputs, stopping) at 1, 2 and 4 loops.
- GPU execution; every validated path today is CPU `torch` attention.
- Long-context behaviour beyond the tested lengths; the artifact's 256K context
  is a configuration limit, not a qualified deployment length.
- Performance measurement. No throughput or memory claim is made here.
