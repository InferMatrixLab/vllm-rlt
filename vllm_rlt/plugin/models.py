"""Which loop shape a model has. Resolved from the model config, or overridden
with ``additional_config={"rlt_loop": {...}}`` (used by tests and by models
that vLLM already runs natively).

* Inkling (2 Mini): vLLM runs the whole loop in one forward. v0 keeps that as
  the baseline path (``full_only``), so loop features off == vLLM.
* Ouro: no boundary layers. The embedding is the first thing CORE depth 0 does
  and CODA is only the LM head, so there is no PRELUDE op. It has an exit gate.
"""

from __future__ import annotations

from vllm_rlt.plugin.program import LoopLayout


def _additional(vllm_config) -> dict:
    extra = getattr(vllm_config, "additional_config", None)
    return extra if isinstance(extra, dict) else {}


def _inkling(hf, split: bool = False) -> LoopLayout:
    """``full_only`` runs vLLM's own forward as one op (the baseline path, equal
    to vLLM); ``split`` runs prelude, each core pass and coda as separate ops."""
    text = getattr(hf, "text_config", hf)
    loop = getattr(text, "loop_config", None) or {}
    passes = getattr(text, "num_loop_passes", None) or loop.get("num_passes") or 1
    return LoopLayout(num_passes=int(passes), has_prelude=True, full_only=not split)


def _ouro(hf) -> LoopLayout:
    return LoopLayout(
        num_passes=int(getattr(hf, "total_ut_steps", 4)),
        has_prelude=False,
        has_gate=True,
    )


_BY_ARCH = {
    "Inkling2LoopForConditionalGeneration": _inkling,
    "Inkling2ForConditionalGeneration": _inkling,
    "OuroForCausalLM": _ouro,
}


def loop_layout(vllm_config) -> LoopLayout:
    extra = _additional(vllm_config)
    override = extra.get("rlt_loop")
    if override is not None:
        return LoopLayout(**override)
    hf = vllm_config.model_config.hf_config
    for arch in getattr(hf, "architectures", None) or ():
        if arch in _BY_ARCH:
            if _BY_ARCH[arch] is _inkling:
                return _inkling(hf, split=bool(extra.get("rlt_split_ops", False)))
            return _BY_ARCH[arch](hf)
    raise ValueError(
        "vllm-rlt: no loop layout for this model; add it to plugin/models.py or "
        'pass additional_config={"rlt_loop": {"num_passes": N, ...}}'
    )
