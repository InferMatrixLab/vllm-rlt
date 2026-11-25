"""Scheduler tests run against the real vLLM Scheduler, on CPU.

vLLM's own ``tests/v1/core/utils.py`` builds the scheduler; it lives in vLLM's
source tree, so point ``VLLM_SRC`` at a checkout of the pinned commit.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

pytest.importorskip("vllm")
VLLM_SRC = os.environ.get("VLLM_SRC")
if not VLLM_SRC:
    pytest.skip("set VLLM_SRC to a vLLM source checkout", allow_module_level=True)

from vllm.sampling_params import SamplingParams  # noqa: E402

from vllm_rlt.plugin.output import LoopModelRunnerOutput  # noqa: E402
from vllm_rlt.plugin.program import LoopLayout  # noqa: E402
from vllm_rlt.plugin.request import LoopRequest  # noqa: E402
from vllm_rlt.plugin.scheduler import LoopAsyncScheduler, LoopScheduler  # noqa: E402


def _vllm_core_utils():
    """Load vLLM's scheduler test helpers without clashing with our ``tests``."""
    stubs = [
        "tests.v1",
        "tests.v1.kv_connector",
        "tests.v1.kv_connector.unit",
        "tests.v1.kv_connector.unit.utils",
    ]
    added = [name for name in stubs if name not in sys.modules]
    for name in added:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["tests.v1.kv_connector.unit.utils"].MockKVConfig = type("MockKVConfig", (), {})
    try:
        path = os.path.join(VLLM_SRC, "tests/v1/core/utils.py")
        spec = importlib.util.spec_from_file_location("_vllm_core_utils", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in added:
            sys.modules.pop(name, None)


UTILS = _vllm_core_utils()
EOS = UTILS.EOS_TOKEN_ID


def make_scheduler(layout: LoopLayout, *, asynchronous: bool = False, **kwargs):
    class Sched(LoopAsyncScheduler if asynchronous else LoopScheduler):
        def _resolve_layout(self):
            return layout

    return UTILS.create_scheduler(scheduler_cls=Sched, async_scheduling=asynchronous, **kwargs)


def make_request(rid: str, prompt_len: int = 4, max_tokens: int = 3, **extra) -> LoopRequest:
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, extra_args=extra or None)
    params.update_from_generation_config({}, EOS)
    return LoopRequest(rid, list(range(prompt_len)), params, None)


def run_step(sched, out, token: int = 7, exit_depths: dict | None = None):
    """Hand back what a runner would: a token on final ops, exit depths on CORE."""
    step = out.loop_step
    ids = list(out.num_scheduled_tokens)
    sampled = [[token] for _ in ids] if step.is_final else []
    return sched.update_from_output(
        out,
        LoopModelRunnerOutput(
            req_ids=ids, sampled_token_ids=sampled, exit_depths=exit_depths or {}
        ),
    )
