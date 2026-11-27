"""Use the upstream worker lifecycle with the loop model runner."""

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu_worker import Worker


class LoopWorker(Worker):
    def _make_model_runner(self):
        from vllm_rlt.worker.loop_model_runner import LoopModelRunner

        graph_config = self.vllm_config.compilation_config
        graph_mode = graph_config.cudagraph_mode
        # RLT forwards run a single op. The upstream whole-model graph cannot replay them.
        graph_config.cudagraph_mode = CUDAGraphMode.NONE
        parallel = self.vllm_config.parallel_config
        if parallel.data_parallel_size > 1 or parallel.pipeline_parallel_size > 1:
            raise ValueError("rlt loop execution currently requires DP=1 and PP=1")
        return LoopModelRunner(self.vllm_config, self.device, graph_mode)
