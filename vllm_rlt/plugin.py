"""vLLM general plugin entry point."""


def register() -> None:
    from vllm.model_executor.models import ModelRegistry

    from vllm_rlt.compat.request_patch import replace_request

    replace_request()
    ModelRegistry.register_model(
        "Inkling2LoopForConditionalGeneration",
        "vllm_rlt.models.inkling_plugin:LoopInklingForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "OuroForCausalLM", "vllm_rlt.models.ouro_plugin:LoopOuroForCausalLM"
    )
