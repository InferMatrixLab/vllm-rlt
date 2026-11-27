"""Single-request BF16 generation with identical token inputs and stop criteria."""

import importlib.util
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import lm_eval
import torch
from transformers import AutoModelForCausalLM, DynamicCache, LogitsProcessor

from benchmarks.gsm8k import is_fixed


def _load_lm_eval_utils():
    # Importing lm_eval.models in the pinned harness eagerly imports every model
    # backend, including multimodal classes removed from Transformers 5. Load
    # its standalone helpers without importing the unrelated model registry;
    # retain the pinned harness's exact stopping and postprocessing behavior.
    path = Path(lm_eval.__file__).parent / "models" / "utils.py"
    spec = importlib.util.spec_from_file_location("_gsm8k_lm_eval_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lm_eval_utils = _load_lm_eval_utils()
postprocess_generated_text = _lm_eval_utils.postprocess_generated_text
stop_sequences_criteria = _lm_eval_utils.stop_sequences_criteria


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


class PluginCompletions:
    """Run the same token prompt on an RLT server and read committed Ouro depths."""

    def __init__(self, endpoint, served_model, step_log):
        self.endpoint = endpoint.rstrip("/")
        self.served_model = served_model
        self.step_log = Path(step_log)

    def check_model(self, model, max_length):
        with urlopen(self.endpoint + "/v1/models", timeout=30) as response:
            data = json.load(response)["data"]
        card = next((card for card in data if card["id"] == self.served_model), None)
        if card is None or not card.get("root") or Path(card["root"]).resolve() != model.resolve():
            raise ValueError("The RLT server must serve the protocol's local checkpoint")
        if card.get("max_model_len", 0) < max_length:
            raise ValueError("The RLT server context is shorter than the protocol requires")
        if not self.step_log.is_file():
            raise ValueError("The RLT server step log is missing")

    def generate(self, prompt_ids, max_new_tokens, stops, exit_policy, eos_token):
        payload = {
            "model": self.served_model,
            "prompt": prompt_ids,
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "return_token_ids": True,
            "include_stop_str_in_output": True,
            "stop": [stop for stop in stops if stop != eos_token],
            "vllm_xargs": {
                "rlt_min_loops": exit_policy["min_loops"],
                "rlt_max_loops": exit_policy["max_loops"],
                "rlt_exit_threshold": exit_policy["threshold"],
            },
        }
        request = Request(
            self.endpoint + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self.step_log.open() as trace:
            trace.seek(0, 2)
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
            # vLLM appends the completion choice index and an internal request suffix.
            pattern = re.compile(
                rf"rlt commit request={re.escape(result['id'])}-0(?:-[^ ]+)? "
                r"position=(\d+) depth=(\d+)"
            )
            events = [
                (int(match.group(1)), int(match.group(2)))
                for line in trace
                if (match := pattern.search(line))
            ]

        ids = result["choices"][0]["token_ids"]
        if [position for position, _ in events] != list(
            range(len(prompt_ids), len(prompt_ids) + len(ids))
        ):
            raise RuntimeError("Committed RLT depths do not match the generated token positions")
        depths = [depth for _, depth in events]
        maximum = exit_policy["max_loops"]
        if depths and depths[0] != maximum:
            raise RuntimeError("Expected full-depth prefill")
        if is_fixed(exit_policy) and any(depth != maximum for depth in depths):
            raise RuntimeError("Expected fixed full-depth generation")
        return ids, depths


class Generator:
    def __init__(self, backend, model_path, tokenizer, exit_policy, *, plugin=None):
        self.backend, self.tokenizer, self.exit = backend, tokenizer, exit_policy
        self.plugin = plugin
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
            # The release indexes KV by depth * physical_layers + layer.
            self.cache_slots = (
                self.model.config.num_hidden_layers * self.model.config.total_ut_steps
            )
            check_depth(self.model.config, exit_policy["max_loops"])
            self.exit_hooks = (
                None
                if is_fixed(exit_policy)
                else ReleaseExitHooks(self.model, exit_policy["threshold"])
            )

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens, stops):
        if self.backend == "transformers":
            ids, depths = self._generate_transformers(prompt_ids, max_new_tokens, stops)
        else:
            ids, depths = self.plugin.generate(
                prompt_ids, max_new_tokens, stops, self.exit, self.tokenizer.eos_token
            )
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        text = postprocess_generated_text(raw, stop=stops, think_end_token=None)
        eos = bool(ids and ids[-1] == self.tokenizer.eos_token_id)
        reason = "eos" if eos else "stop" if any(s in raw for s in stops) else "length"
        result = {"token_ids": ids, "raw_text": raw, "text": text, "finish_reason": reason}
        if depths is not None:
            result["exit_depths"] = list(depths)
        return result

    def _generate_transformers(self, prompt_ids, max_new_tokens, stops):
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        criteria = stop_sequences_criteria(self.tokenizer, stops, len(prompt_ids), 1)
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
        return ids, depths
