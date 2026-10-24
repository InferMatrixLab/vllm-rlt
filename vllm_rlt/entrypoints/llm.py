from collections.abc import Iterable
from dataclasses import replace

import torch

from vllm_rlt.engine.llm_engine import LLMEngine, StageClient
from vllm_rlt.models import (
    load_tokenizer,
    resolve_model_config,
    resolve_model_source,
)
from vllm_rlt.pd.engine import PDEngine
from vllm_rlt.sampling_params import SamplingParams


class LLM:
    """vLLM-style offline facade; token ID input does not require Transformers."""

    def __init__(
        self,
        model,
        *,
        tokenizer=None,
        revision=None,
        device=None,
        dtype=None,
        cache_config=None,
        scheduler_config=None,
        attention_backend="torch",
        exit_config=None,
        execution_config=None,
        speculative_config=None,
    ):
        self._tokenizer_source = None
        if isinstance(model, (LLMEngine, StageClient, PDEngine)):
            self.tokenizer = tokenizer
            self.engine = model
            self._next_request_id = 0
            return
        if isinstance(model, str):
            tokenizer_source = resolve_model_source(model)
            model, revision, _ = resolve_model_config(model, revision=revision)
            if tokenizer is None:
                self._tokenizer_source = (tokenizer_source, revision)
        self.tokenizer = tokenizer
        self.engine = StageClient(
            model,
            device=device,
            dtype=dtype,
            revision=revision,
            seed=torch.initial_seed(),
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            attention_backend=attention_backend,
            exit_config=exit_config,
            execution_config=execution_config,
            speculative_config=speculative_config,
        )
        self._next_request_id = 0

    def generate(self, prompts, sampling_params=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        params = sampling_params or SamplingParams()
        all_params = params if isinstance(params, list) else [params] * len(prompts)
        if len(all_params) != len(prompts):
            raise ValueError("provide one SamplingParams per prompt")
        if self.engine.has_unfinished_requests():
            raise RuntimeError("finish existing engine requests before calling generate")
        request_ids = []
        try:
            for prompt, request_params in zip(prompts, all_params):
                if isinstance(prompt, str):
                    if self.tokenizer is None and self._tokenizer_source is not None:
                        source, revision = self._tokenizer_source
                        self.tokenizer = load_tokenizer(
                            source,
                            revision=revision,
                        )
                    if self.tokenizer is None:
                        raise ValueError(
                            "text prompts require a tokenizer; pass token ID lists instead"
                        )
                    prompt = self.tokenizer.encode(prompt)
                request_id = str(self._next_request_id)
                self._next_request_id += 1
                self.engine.add_request(request_id, list(prompt), request_params)
                request_ids.append(request_id)
            results = {}
            while self.engine.has_unfinished_requests():
                for output in self.engine.step():
                    if output.finished:
                        if self.tokenizer is not None:
                            output = replace(
                                output,
                                text=self.tokenizer.decode(
                                    output.token_ids, skip_special_tokens=True
                                ),
                            )
                        results[output.request_id] = output
            return [results[request_id] for request_id in request_ids]
        except Exception:
            for request_id in request_ids:
                if self.engine.has_request(request_id):
                    self.engine.abort_request(request_id)
            raise

    def start_profile(self, config, *, scheduled=False):
        return self.engine.start_profile(config, scheduled=scheduled)

    def start_weight_update(self, version: int | None = None):
        self.engine.start_weight_update(version)

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        self.engine.update_weights(weights)

    def finish_weight_update(self):
        self.engine.finish_weight_update()

    def get_weight_version(self) -> int:
        return self.engine.get_weight_version()

    def reset_prefix_cache(self):
        self.engine.reset_prefix_cache()

    def pause_generation(self):
        self.engine.pause_generation()

    def resume_generation(self):
        self.engine.resume_generation()

    def stop_profile(self):
        return self.engine.stop_profile()

    def profile_status(self):
        return self.engine.profile_status()

    def wait_for_profile_artifacts(self, timeout=None):
        return self.engine.wait_for_profile_artifacts(timeout)

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
