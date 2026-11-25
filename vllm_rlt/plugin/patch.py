"""The one place where the plugin patches vLLM.

``Request`` is constructed in a single spot (``EngineCore.add_request``) but
imported by name in dozens of modules, so replacing the class means replacing
every reference, as vllm-omni's ``patch.py`` does. Nothing else is patched:
the scheduler, worker and runner are selected through ``scheduler_cls`` and
``worker_cls``.

The patch checks itself at import: if vLLM changes in a way that leaves a
reference pointing at the original class, importing the plugin raises instead
of silently serving with a half-patched request type. There is no vLLM version
guard.
"""

from __future__ import annotations

import sys

from vllm.v1.request import Request as _OriginalRequest

from vllm_rlt.plugin.request import LoopRequest

_applied = False


def apply_request_patch() -> int:
    """Point every ``Request`` reference in vLLM at ``LoopRequest``.

    Returns the number of modules rewritten. Idempotent.
    """
    global _applied
    if _applied:
        return 0
    patched = 0
    # Snapshot: touching attributes can trigger lazy imports that mutate
    # sys.modules while we iterate.
    for name, module in list(sys.modules.items()):
        if module is None or "vllm" not in name or name.startswith("vllm_rlt"):
            continue
        if getattr(module, "Request", None) is _OriginalRequest:
            setattr(module, "Request", LoopRequest)
            patched += 1
    _verify()
    _applied = True
    return patched


def _verify() -> None:
    import vllm.v1.request as request_module

    if request_module.Request is not LoopRequest:
        raise RuntimeError(
            "vllm-rlt: Request patch did not take effect on vllm.v1.request; "
            "vLLM's module layout changed"
        )
    if not issubclass(LoopRequest, _OriginalRequest):
        raise RuntimeError("vllm-rlt: LoopRequest must subclass vLLM's Request")
