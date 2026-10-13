"""Single-request BF16 generation with identical token inputs and stop criteria."""

import torch
from lm_eval.models.utils import postprocess_generated_text, stop_sequences_criteria
from transformers import AutoModelForCausalLM, DynamicCache, LogitsProcessor

from benchmarks.gsm8k import is_fixed
from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
)


def check_logits(logits):
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("Non-finite generation logits")


class FiniteLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        check_logits(scores)
        return scores


def check_depth(config, loops):
    if config.total_ut_steps != loops:
        raise ValueError(
            f"Model has {config.total_ut_steps} loops but the protocol expects {loops}"
        )


class ReleaseExitHooks:
    """Align the release's threshold exit with native semantics and record its exit depths.

    The release applies ``exit_threshold`` on every forward, including prefill, so its first
    output token could come from an early loop. Native prefill always runs full depth, so the
    pre-hook forces ``exit_at_step`` (checked before ``exit_threshold``) on the prefill call.
    The model hook repeats the release's exit rule on the last position, in the release's
    dtype, and records the selected loop count per generated token.
    """

    def __init__(self, model, threshold):
        self.threshold = threshold
        self.full_depth = model.config.total_ut_steps
        self.depths = []
        self._forced = False
        model.register_forward_pre_hook(self._before_forward, with_kwargs=True)
        model.model.register_forward_hook(self._after_model)

    def _before_forward(self, module, args, kwargs):
        position = kwargs.get("cache_position")
        self._forced = position is not None and position[0].item() == 0
        if self._forced:
            kwargs["exit_at_step"] = self.full_depth - 1
        return args, kwargs

    def _after_model(self, module, inputs, output):
        if self._forced:
            self.depths.append(self.full_depth)
            return
        # Same operations and dtype as the release's exit_threshold branch, on the one
        # position whose logits are kept (logits_to_keep=1).
        gates = [gate[:, -1:].squeeze(-1) for gate in output[2]]
        pdf, remaining = [], torch.ones_like(gates[0])
        for index, gate in enumerate(gates):
            hazard = torch.sigmoid(gate)
            if index < len(gates) - 1:
                pdf.append(hazard * remaining)
                remaining = remaining * (1.0 - hazard)
            else:
                pdf.append(remaining)
        reached = torch.cumsum(torch.stack(pdf, dim=2), dim=2) >= self.threshold
        step = torch.argmax(reached.float(), dim=2)
        step[~reached.any(dim=2)] = len(gates) - 1
        self.depths.append(int(step.item()) + 1)


class Generator:
    def __init__(
        self, backend, model_path, tokenizer, max_length, exit_policy, attention_backend="triton"
    ):
        self.backend, self.tokenizer, self.exit = backend, tokenizer, exit_policy
        # Full depth, taken from the protocol (the checkpoint's total_ut_steps).
        loops = exit_policy["max_loops"]
        if backend == "transformers":
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                )
                .eval()
                .to("cuda")
            )
            # Ouro indexes KV by depth * physical_layers + layer. Allocate all
            # slots in the standard HF cache so the release need not replace it
            # with its cache class, whose interface predates Transformers 4.55.
            self.cache_slots = (
                self.model.config.num_hidden_layers * self.model.config.total_ut_steps
            )
            check_depth(self.model.config, loops)
            self.exit_hooks = (
                None
                if is_fixed(exit_policy)
                else ReleaseExitHooks(self.model, exit_policy["threshold"])
            )
        else:
            self.llm = LLM(
                model_path,
                device="cuda",
                dtype=torch.bfloat16,
                attention_backend=attention_backend,
                # last_exited KV keeps one plane per loop.
                cache_config=CacheConfig(num_blocks=loops * ((max_length + 15) // 16)),
                scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=128),
                exit_config=ExitConfig(exit_policy["mode"]),
                execution_config=ExecutionConfig(async_scheduling=exit_policy["async_scheduling"]),
            )
            # Observe the existing sampling boundary without changing arithmetic.
            self.model = self.llm.engine.model
            check_depth(self.model.config, loops)
            self._finite_hook = self.model.lm_head.register_forward_hook(
                lambda module, inputs, output: check_logits(output)
            )

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens, stops):
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        criteria = stop_sequences_criteria(self.tokenizer, stops, len(prompt_ids), 1)
        if self.backend == "transformers":
            if self.exit_hooks is not None:
                self.exit_hooks.depths = []
            cache = DynamicCache()
            cache.append_new_layers(self.cache_slots - 1)
            ids = self.model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=criteria,
                logits_processor=[FiniteLogits()],
                use_cache=True,
                past_key_values=cache,
                # The release selects the exited loop's hidden state after running
                # every loop, so its KV stays full-depth even when exiting early.
                **(
                    {"exit_at_step": self.exit["max_loops"] - 1}
                    if is_fixed(self.exit)
                    else {"exit_threshold": self.exit["threshold"]}
                ),
                use_weighted_exit=False,
                logits_to_keep=1,
            )[0, len(prompt_ids) :].tolist()
            if len(cache.layers) != self.cache_slots:
                raise RuntimeError("Official cache does not cover every loop/layer")
            depths = self.exit_hooks.depths if self.exit_hooks is not None else None
            if depths is not None and len(depths) != len(ids):
                raise RuntimeError("Recorded exit depths do not match the generated tokens")
        else:
            engine = self.llm.engine
            engine.add_request(
                "gsm8k",
                prompt_ids,
                SamplingParams(
                    max_tokens=max_new_tokens,
                    min_loops=self.exit["min_loops"],
                    max_loops=self.exit["max_loops"],
                    exit_threshold=self.exit["threshold"],
                    temperature=0,
                ),
            )
            ids = []
            depths = []
            try:
                while engine.has_unfinished_requests():
                    for output in engine.step():
                        ids = output.token_ids
                        depths = output.exit_depths
                        full = self.exit["max_loops"]
                        if is_fixed(self.exit) and any(depth != full for depth in depths):
                            raise RuntimeError("Expected fixed full-depth generation")
                        sequence = torch.tensor([prompt_ids + ids], device="cuda")
                        if not output.finished and criteria(sequence, None).all().item():
                            engine.abort_request("gsm8k")
            finally:
                if engine.has_unfinished_requests():
                    engine.abort_request("gsm8k")
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        text = postprocess_generated_text(raw, stop=stops, think_end_token=None)
        eos = bool(ids and ids[-1] == self.tokenizer.eos_token_id)
        reason = "eos" if eos else "stop" if any(s in raw for s in stops) else "length"
        result = {"token_ids": ids, "raw_text": raw, "text": text, "finish_reason": reason}
        if depths is not None:
            # The first output comes from full-depth prefill on both backends; later entries are
            # decode depths. The release still computes every loop for every token.
            result["exit_depths"] = list(depths)
        return result
