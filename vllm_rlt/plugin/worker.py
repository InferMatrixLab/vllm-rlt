"""LoopWorker: vLLM's GPU worker with the loop runner and loop-aware dummy runs.

Selected with ``worker_cls``. Two overrides:

* ``_make_model_runner`` returns ``LoopModelRunner``.
* ``execute_dummy_batch`` is what an idle DP rank runs so the others' MoE
  collectives line up. vLLM's version always builds a decode-shaped batch; a
  loop rank must run the op the other ranks chose, so it takes it from the
  last aligned step (set by ``set_dummy_step``) and falls back to vLLM's.
"""

from __future__ import annotations

from vllm.v1.worker.gpu_worker import Worker

from vllm_rlt.plugin.program import LoopStep, Op


class LoopWorker(Worker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dummy_step: LoopStep | None = None

    def _make_model_runner(self):
        from vllm_rlt.plugin.runner import LoopModelRunner

        return LoopModelRunner(self.vllm_config, self.device)

    def set_dummy_step(self, op: int, depth: int = 0, rows: int = 1) -> None:
        self._dummy_step = LoopStep.dummy(Op(op), depth, rows)

    def execute_dummy_batch(self) -> None:
        step, self._dummy_step = self._dummy_step, None
        if step is None:
            return super().execute_dummy_batch()
        from vllm.v1.core.sched.output import SchedulerOutput

        out = SchedulerOutput.make_empty()
        out.loop_step = step  # the runner runs a dummy op for it
        self.model_runner.execute_model(out, dummy_run=False)
