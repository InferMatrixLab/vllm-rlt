"""LoopWorker: vLLM's GPU worker with the loop runner and DP op alignment.

Selected with ``worker_cls``. Overrides:

* ``_make_model_runner`` returns ``LoopModelRunner``.
* ``init_device`` / ``load_model`` finish by giving the runner a DP aligner when
  data parallelism is on. The aligner all-reduces each rank's candidate table
  over vLLM's DP CPU group, the same group vLLM already uses to agree on the
  CUDA-graph shape every step.
* ``execute_dummy_batch`` is what an idle DP rank runs so the other ranks' MoE
  collectives line up. vLLM's version builds a decode-shaped batch; a loop rank
  must run the op the other ranks chose, so an idle rank also takes part in the
  alignment (with an empty table) and then runs a dummy of the agreed op.
"""

from __future__ import annotations

from vllm.v1.worker.gpu_worker import Worker

from vllm_rlt.plugin.parallel import DPOpAlign, NoAlign
from vllm_rlt.plugin.program import LoopStep, Op


class LoopWorker(Worker):
    def _make_model_runner(self):
        from vllm_rlt.plugin.runner import LoopModelRunner

        return LoopModelRunner(self.vllm_config, self.device)

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self._install_aligner()
        return result

    def _install_aligner(self) -> None:
        parallel = self.vllm_config.parallel_config
        runner = self.model_runner
        if parallel.data_parallel_size <= 1:
            runner.aligner = NoAlign()
            return
        from vllm.distributed.parallel_state import get_dp_group

        group = get_dp_group()
        passes = getattr(runner.get_model(), "num_passes", 1)
        runner.aligner = DPOpAlign(
            rank=parallel.data_parallel_rank,
            size=parallel.data_parallel_size,
            num_passes=passes,
            all_reduce=group.cpu_all_reduce_,
        )

    def execute_dummy_batch(self) -> None:
        runner = self.model_runner
        layout = getattr(runner, "loop_layout", None)
        if layout is None or isinstance(runner.aligner, NoAlign):
            return super().execute_dummy_batch()
        # An idle rank offers nothing and runs whatever the busy ranks choose.
        chosen = runner.aligner.align({})
        if chosen is None:
            return super().execute_dummy_batch()
        from vllm.v1.core.sched.output import SchedulerOutput

        out = SchedulerOutput.make_empty()
        out.loop_step = LoopStep.dummy(Op(chosen[0]), max(chosen[1], 0), rows=1)
        runner.execute_model(out)
        runner.sample_tokens(None)
