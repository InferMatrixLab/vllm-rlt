"""vLLM plugin entry point (``vllm.general_plugins``).

vLLM loads general plugins in every process (API server, EngineCore, workers),
so this runs once per process and is idempotent. It does three things:

1. patch ``Request`` (the only patch, see ``patch.py``);
2. register the plugin's models (Ouro, which vLLM removed in 0.26);
3. nothing else. The scheduler and worker are chosen by the user with
   ``scheduler_cls`` / ``worker_cls``, so a vLLM run without those options is
   untouched by this package.

No vLLM version check. The development target is vLLM
0.34.0rc1.dev1149+g3146a9df4.
"""

from __future__ import annotations

SCHEDULER_CLS = "vllm_rlt.plugin.scheduler.LoopScheduler"
ASYNC_SCHEDULER_CLS = "vllm_rlt.plugin.scheduler.LoopAsyncScheduler"
WORKER_CLS = "vllm_rlt.plugin.worker.LoopWorker"

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    from vllm import ModelRegistry

    from vllm_rlt.plugin.ouro_config import register_ouro_config
    from vllm_rlt.plugin.patch import apply_request_patch
    from vllm_rlt.plugin.selfspec import register_self_spec

    apply_request_patch()
    register_ouro_config()
    register_self_spec()
    ModelRegistry.register_model("OuroForCausalLM", "vllm_rlt.plugin.ouro:OuroLoopForCausalLM")
    # Inkling: vLLM's own model, subclassed to run one loop op at a time. Its
    # single forward is only used by vLLM's profiling and warmup runs.
    ModelRegistry.register_model(
        "Inkling2LoopForConditionalGeneration",
        "vllm_rlt.plugin.inkling:Inkling2LoopOpsForCausalLM",
    )
