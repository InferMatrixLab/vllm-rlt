# Training rollout and full-weight publication

`LLM`, `StageClient`, and `LLMEngine` (the local `StageEngine` alias) expose the same full-update operations. `LLM` also accepts
an existing `LLMEngine` or `PDEngine`, so framework adapters use the same generation
and publication calls with either backend.

```python
llm.start_weight_update(version=1)
for chunk in converted_hf_weights:
    llm.update_weights(chunk)  # Iterable[(physical_parameter_name, torch.Tensor)]
llm.finish_weight_update()
outputs = llm.generate(prompts, SamplingParams(
    max_tokens=128, temperature=1.0, seed=42, logprobs=0,
))
```

Each output contains aligned selected-token `log_probs`, `exit_depths`, the
committed `weight_version`, and its effective `sampling_params`. `logprobs=None`
keeps the ordinary inference path. Reported parameters resolve the model's
default `max_loops` and cap `top_k` at its vocabulary size. `logprobs=0` requests only the emitted token's
probability; top-N alternatives are not implemented.

`logprobs_mode="raw"` is the default: FP32 log-softmax of unscaled model logits.
`"processed"` returns probabilities after positive temperature and top-k/top-p
filtering. Greedy requests return the raw score in either mode. Speculative
outputs use the verified **target** policy's score, including correction/bonus
tokens; proposal or rejection-residual probabilities are never exposed as the
policy score. Explicit `stop_token_ids` apply even with `ignore_eos=True`.
`seed=None` resolves an independent seed in `[0, 2**63)` when the request is
created. The effective seed is returned in `sampling_params`, including for an
abort before sampling. Reusing those parameters replays the same request on the
same policy and execution configuration. The default remains `seed=0`.

Publication requires an idle engine. Each chunk is checked for duplicate names,
unknown names and mismatched shapes before that chunk is copied. `finish_weight_update`
requires every physical parameter exactly once. Copies preserve parameter addresses
and captured graphs; a fresh transaction invalidates all cached prefix KV. If a
transfer fails, generation stays blocked until a fresh full transaction succeeds.
Starting again discards an unfinished transaction. `pause_generation` stops
admission while existing requests drain; `resume_generation` requires publication
to be complete. `abort_request` remains available to cancel requests before an update.

PDEngine waits for all transfers to release, broadcasts CPU-staged weight chunks
to every P/D worker, and commits a version only after every peer acknowledges it.
Recovery uses a version newer than **all attempted** PD versions because some
workers may already have committed. With `terminate_workers_on_failure=False`, fatal workers retain their registered
memory rather than receive forced process signals; normal `close()` remains
cooperative. No CUDA tensors change ownership over IPC;
NIXL's existing GPU KV path and registrations remain intact. Prefix invalidation
does not replace registered KV storage. PD worker control errors return replies
without terminating the worker.

Existing exit, async and speculative configuration rules still apply. Self-speculation
uses a full-depth target. Selected-token scores are supported by synchronous
self-speculation and ordinary asynchronous decoding; cross-round asynchronous
self-speculation rejects logprobs requests at admission because its compact result
buffers contain token IDs only. PD follows the same synchronous and asynchronous
self-speculation limits on its decode workers. Publication
adds no such restriction to ordinary early-exit or async rollouts. Trainers must
replay full-depth prompt prefill and the returned decode depths rather than assume
the entire sequence used K.
