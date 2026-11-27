"""Install the loop request at vLLM's request construction boundary."""

import sys
from types import ModuleType

from vllm.v1 import request as vllm_request

from vllm_rlt.core.request import LoopRequest


def replace_request() -> None:
    """Replace the request class and any aliases imported before plugin loading.

    vLLM g3146a9df4 binds Request at module scope in vllm/v1/engine/core.py.
    Request.from_engine_core_request(request, block_hasher, block_hashes) calls
    cls(...), so replacing the class preserves the engine's conversion path.
    """
    original_request = LoopRequest.__base__
    if vllm_request.Request not in (original_request, LoopRequest):
        raise RuntimeError("rlt found an unexpected vLLM Request binding")

    # Later imports must see the new class even if EngineCore is not loaded yet.
    vllm_request.Request = LoopRequest
    for module in tuple(sys.modules.values()):
        if isinstance(module, ModuleType):
            namespace = vars(module)
            for name, value in tuple(namespace.items()):
                if value is original_request:
                    namespace[name] = LoopRequest

    engine_core = sys.modules.get("vllm.v1.engine.core")
    if vllm_request.Request is not LoopRequest or (
        engine_core is not None and vars(engine_core).get("Request") is not LoopRequest
    ):
        raise RuntimeError("rlt could not replace the vLLM EngineCore Request binding")
