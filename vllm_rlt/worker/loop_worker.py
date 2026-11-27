"""Use the upstream worker lifecycle with the loop model runner."""

from copy import copy

from vllm.config import set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu_worker import Worker


class LoopWorker(Worker):
    def _make_model_runner(self):
        from vllm_rlt.worker.loop_model_runner import LoopModelRunner

        parallel = self.parallel_config
        if parallel.data_parallel_size > 1 or parallel.pipeline_parallel_size > 1:
            raise ValueError("rlt loop execution currently requires DP=1 and PP=1")

        # UniProc shares the engine config with its worker. Keep the selected
        # loop graph mode available to the scheduler when disabling native graphs.
        worker_config = copy(self.vllm_config)
        graph_config = copy(worker_config.compilation_config)
        graph_mode = graph_config.cudagraph_mode
        graph_config.cudagraph_mode = CUDAGraphMode.NONE
        worker_config.compilation_config = graph_config
        self.vllm_config = worker_config
        self.compilation_config = graph_config
        with set_current_vllm_config(worker_config):
            return LoopModelRunner(worker_config, self.device, graph_mode)
