import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import torch

from vllm_rlt.attention import BackendCapabilities, create_backend
from vllm_rlt.config import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SchedulerConfig,
    resolve_cuda_graph_rows,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import calibrate_fp8_scale, plan_cache
from vllm_rlt.core.scheduler import Scheduler, SchedulerOutput
from vllm_rlt.core.scheduling_policy import requires_staged_prefill
from vllm_rlt.engine.connector import StageConnector
from vllm_rlt.engine.packed_prefill import packed_prefill_eligible
from vllm_rlt.engine.preemption import PreemptionManager
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.kernels.paged_attention import PackedWindowShape
from vllm_rlt.profiling import Profiler
from vllm_rlt.request import FinishReason, Request, RequestOutput, RequestPhase
from vllm_rlt.sampling_params import DEFAULT_MIN_LOOPS, SamplingParams
from vllm_rlt.worker.executor import Executor
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.output import ExitSignal, ModelRunnerOutput
from vllm_rlt.worker.sampler import RngRegistry
from vllm_rlt.worker.speculative import SpeculativeRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm_rlt.worker.async_speculative import AsyncSpeculativeRunner


@dataclass(frozen=True)
class SignalHandle:
    ticket: object
    row: int


class StageEngine:
    """Single-device loop-level engine with synchronous or pipelined scheduling."""

    def __init__(
        self,
        model,
        *,
        cache_config=None,
        scheduler_config=None,
        attention_backend="torch",
        exit_config=None,
        execution_config=None,
        speculative_config=None,
        stage_config=None,
        connector: StageConnector | None = None,
    ):
        self.model = model
        self.stage_config = stage_config or StageConfig()
        self.profiling = Profiler(next(model.parameters()).device)
        cache_config = cache_config or CacheConfig()
        scheduler_config = scheduler_config or SchedulerConfig()
        parameter = next(model.parameters())
        config = model.config
        auxiliary_width = getattr(config, "auxiliary_cache_width", 0)
        if getattr(model, "requires_boundary_kv", False):
            required_layout = getattr(model, "kv_cache_layout", "last_exited")
            if cache_config.layout != required_layout:
                raise ValueError(f"boundary-KV models require {required_layout} KV")
            # Only models that verified hit/miss parity opt in: the cache must
            # cover boundary KV at depth 0 and recurrent KV at every depth.
            if cache_config.enable_prefix_caching and not getattr(
                model, "supports_boundary_prefix_caching", False
            ):
                raise ValueError("boundary-KV prefix caching is not yet supported")
        self.exit_config = exit_config or ExitConfig()
        self.execution_config = resolve_cuda_graph_rows(
            execution_config or ExecutionConfig(), scheduler_config, speculative_config
        )
        if self.execution_config.cuda_graph_warmup and not getattr(model, "interleaved_kv", False):
            raise ValueError("CUDA Graph warmup requires native Inkling kernels")
        if auxiliary_width:
            if attention_backend not in ("auto", "torch", "triton"):
                raise ValueError("Inkling 2 Mini requires the Triton KV backend")
            # Inkling uses Triton for KV layout and vLLM for model attention.
            attention_backend = "triton"
            if (
                cache_config.kv_cache_dtype != "auto"
                or scheduler_config.enable_preemption
                or self.stage_config.role != "prefill_decode"
            ):
                raise ValueError(
                    "Inkling 2 Mini requires a single stage with BF16 KV, without preemption"
                )
            if self.exit_config.mode != "ouro":
                raise ValueError("Inkling 2 Mini runs the configured number of passes")
        if parameter.device.type == "npu":
            if config.to_dict()["model_type"] != "ouro" or config.total_ut_steps != 4:
                raise ValueError("NPU currently supports Ouro with four loops")
            if parameter.dtype != torch.bfloat16:
                raise ValueError("NPU Ouro requires BF16 weights")
            if cache_config.kv_cache_dtype != "auto":
                raise ValueError("NPU Ouro requires BF16 KV cache")
            if (
                self.execution_config.cuda_graphs
                or (self.execution_config.async_scheduling and self.execution_config.static_buffers)
                or speculative_config is not None
                or self.stage_config.role != "prefill_decode"
                or scheduler_config.enable_preemption
                or cache_config.enable_prefix_caching
                or cache_config.layout != "last_exited"
                or self.exit_config.mode not in ("ouro", "ouro_delayed")
            ):
                raise ValueError(
                    "NPU Ouro requires local execution, ouro/ouro_delayed exit mode and "
                    "last_exited KV without speculation, graphs, preemption or prefix caching"
                )
        staged_prefill = requires_staged_prefill(
            scheduler_config, produces_tokens=self.stage_config.produces_tokens
        )
        sliding_window = getattr(model, "requires_sliding_window", False)
        # Automatic packed prefill wins where supported; it needs all-layer
        # blocks, so packed_prefill=False keeps grouped storage and reclamation.
        prefer_packed = packed_prefill_eligible(
            self.execution_config,
            device_type=parameter.device.type,
            layout=cache_config.layout,
            staged_prefill=staged_prefill,
            sliding_window=sliding_window,
            stage_role=self.stage_config.role,
            speculative=speculative_config is not None,
            preemption=scheduler_config.enable_preemption,
        )
        ordered_pipeline = (
            speculative_config is not None
            and speculative_config.mode == "pipeline"
            and getattr(model, "requires_ordered_pipeline_coda", False)
        )
        device_boundary_pipeline = ordered_pipeline and getattr(model, "interleaved_kv", False)
        mixed_speculative = scheduler_config.mode == "mixed" and speculative_config is not None
        if mixed_speculative:
            if not (
                device_boundary_pipeline
                and getattr(model, "fixed_loop_count", False)
                and self.execution_config.async_scheduling
                and speculative_config.draft_loops == 1
                and self.stage_config.role == "prefill_decode"
            ):
                raise ValueError(
                    "mixed speculative decoding requires the async native Inkling pipeline"
                )
            if self.execution_config.packed_prefill is False:
                raise ValueError("mixed speculative decoding requires packed prefill")
            self.execution_config = replace(self.execution_config, packed_prefill=True)
            prefer_packed = True
        if getattr(model, "requires_boundary_kv", False):
            prefill_capability = (
                "supports_wavefront_prefill"
                if scheduler_config.wavefront_chunked_prefill
                else "supports_mixed_prefill"
            )
            if staged_prefill and not getattr(model, prefill_capability, False):
                raise ValueError(
                    "boundary-KV models do not support staged prefill required by this scheduler"
                )
            if self.execution_config.async_scheduling and not getattr(
                model, "supports_async_scheduling", False
            ):
                raise ValueError("boundary-KV models require synchronous scheduling")
            if speculative_config is not None:
                if not ordered_pipeline and not getattr(model, "supports_self_speculation", False):
                    raise ValueError("boundary-KV speculative decoding is not yet supported")
                if ordered_pipeline:
                    if self.execution_config.async_scheduling and not device_boundary_pipeline:
                        raise ValueError("async boundary pipeline requires interleaved KV")
                    if scheduler_config.enable_preemption:
                        raise ValueError("Inkling pipeline requires preemption to be disabled")
                elif (
                    self.execution_config.async_scheduling
                    or self.execution_config.cuda_graphs
                    or scheduler_config.enable_preemption
                ):
                    raise ValueError(
                        "boundary-KV speculative decoding requires synchronous eager "
                        "execution without CUDA graphs or preemption"
                    )
        if cache_config.enable_prefix_caching and cache_config.layout == "shared":
            # The shared runner finishes every loop of a token before the next,
            # so cached final-loop KV is the cold prefill state.
            if not getattr(model, "supports_boundary_prefix_caching", False):
                raise ValueError("this model does not support shared-KV prefix caching")
            if self.execution_config.async_scheduling or self.execution_config.cuda_graphs:
                raise ValueError("shared-KV prefix caching requires synchronous eager execution")
        if not getattr(model, "supports_cuda_graphs", True) and self.execution_config.cuda_graphs:
            raise ValueError("this model does not support CUDA graphs yet")
        self.speculative_config = speculative_config
        self.pipeline_speculative = bool(
            speculative_config and speculative_config.mode == "pipeline"
        )
        self.async_speculative = bool(speculative_config and self.execution_config.async_scheduling)
        if speculative_config is not None:
            if speculative_config.pipeline_graph != "off" and not (
                device_boundary_pipeline
                and self.async_speculative
                and self.execution_config.cuda_graphs
            ):
                raise ValueError(
                    "pipeline_graph requires native Inkling, async scheduling and CUDA Graphs"
                )
            if cache_config.layout != "last_exited":
                raise ValueError("speculative decoding requires last_exited KV")
            if speculative_config.target_loops != config.total_ut_steps:
                raise ValueError("speculative target_loops must equal the model full depth")
            if self.exit_config.mode != "ouro":
                raise ValueError("speculative decoding requires fixed-depth ouro exit mode")
            if self.async_speculative and parameter.device.type != "cuda":
                raise ValueError("async speculative decoding requires CUDA")
            if (scheduler_config.mode != "refill" and not mixed_speculative) or (
                self.async_speculative
                and not self.pipeline_speculative
                and scheduler_config.enable_preemption
            ):
                raise ValueError(
                    "speculative decoding requires refill scheduling without preemption"
                )
            # Retained q distributions and sampling scratch coexist with target
            # logits. Reserve beyond ordinary prefill/core profiling, even when
            # requests later choose sampling rather than greedy.
            scratch = scheduler_config.max_num_batched_tokens * (
                config.vocab_size * 24 + config.hidden_size * parameter.element_size() * 2
            )
            if self.async_speculative:
                # Two in-flight rounds, plus device and pinned page-table buffers.
                width = (config.max_position_embeddings + cache_config.block_size - 1) // (
                    cache_config.block_size
                )
                scratch = 2 * scratch + 8 * scheduler_config.max_num_seqs * (
                    config.total_ut_steps * width
                )
            cache_config = replace(
                cache_config, memory_reserve_bytes=cache_config.memory_reserve_bytes + scratch
            )
        if self.execution_config.cuda_graphs and parameter.device.type != "cuda":
            raise ValueError("CUDA graphs require CUDA with Triton or FlashAttention")
        if self.execution_config.cuda_graphs and speculative_config is not None:
            rows = min(
                scheduler_config.max_num_seqs * (speculative_config.num_speculative_tokens + 1),
                scheduler_config.max_num_batched_tokens,
            )
            if rows > self.execution_config.cuda_graph_max_batch_size:
                logger.warning(
                    "CUDA Graph row limit %d cannot cover %d verification rows "
                    "(max_num_seqs=%d, K=%d); larger batches run eager. "
                    "Raise cuda_graph_max_batch_size, or leave it unset to cover them, "
                    "and budget additional graph memory.",
                    self.execution_config.cuda_graph_max_batch_size,
                    rows,
                    scheduler_config.max_num_seqs,
                    speculative_config.num_speculative_tokens,
                )
        if self.execution_config.async_scheduling:
            # A fixed-loop model has no exit decision to delay or look ahead to.
            if (
                not self.async_speculative
                and not getattr(model, "fixed_loop_count", False)
                and self.exit_config.mode
                not in (
                    "ouro_delayed",
                    "random_lookahead",
                    "trace",
                )
            ):
                raise ValueError(
                    "async scheduling requires ouro_delayed, random_lookahead or trace exit mode"
                )
        if staged_prefill and cache_config.layout != "last_exited":
            raise ValueError("staged prefill requires last_exited KV planes")
        require_packed = self.execution_config.packed_prefill is True
        kv_layers = getattr(config, "num_kv_layers", config.num_hidden_layers)
        window = getattr(config, "sliding_window", None)
        packed_window_shape = (
            PackedWindowShape(
                config.num_attention_heads,
                config.num_key_value_heads,
                kv_layers,
                window,
                fp8_kv=cache_config.kv_cache_dtype == "fp8_e4m3",
            )
            if prefer_packed and sliding_window
            else None
        )
        attention = create_backend(
            attention_backend,
            parameter.device,
            parameter.dtype,
            config.head_dim,
            cache_config.block_size,
            required_capabilities=BackendCapabilities(
                packed_prefill=require_packed,
                packed_sliding_window=require_packed and sliding_window,
                cuda_graphs=self.execution_config.cuda_graphs,
                async_scheduling=(
                    parameter.device.type in ("cuda", "npu")
                    and self.execution_config.async_scheduling
                ),
                device_context_lengths=self.async_speculative,
                fp8_kv=cache_config.kv_cache_dtype == "fp8_e4m3",
                sliding_window=sliding_window,
            ),
            prefer_packed_prefill=prefer_packed and not sliding_window,
            packed_window_shape=packed_window_shape,
            allow_fp8_calibration=True,
        )
        fp8_scale = None
        if cache_config.kv_cache_dtype == "fp8_e4m3" and not attention.capabilities.fp8_kv:
            fp8_scale = calibrate_fp8_scale(model, cache_config, attention)
            logger.info(
                "Calibrated FP8 scale %g for %s", fp8_scale, attention.info["selected_backend"]
            )
        if self.execution_config.packed_prefill is None:
            supported = attention.capabilities.packed_prefill and (
                not sliding_window or attention.capabilities.packed_sliding_window
            )
            self.execution_config = replace(
                self.execution_config, packed_prefill=prefer_packed and supported
            )
        layer_windows = getattr(model, "kv_layer_windows", None)
        # Graph, async, packed, speculative, FP8 and preemption paths keep
        # all-layer blocks: their metadata, rollback, scales or CPU offload have
        # no grouped support yet.
        if layer_windows is not None and (
            self.execution_config.cuda_graphs
            or self.execution_config.async_scheduling
            or self.execution_config.packed_prefill
            or speculative_config is not None
            or cache_config.kv_cache_dtype != "auto"
            or scheduler_config.enable_preemption
        ):
            layer_windows = None
        # Grouped pages expire only when every token runs all loops in ouro order.
        reclaim_swa = (
            layer_windows is not None
            and any(window is not None for window in layer_windows.values())
            and getattr(model, "fixed_loop_count", False)
            and self.exit_config.mode == "ouro"
        )
        num_blocks, self.memory_plan = plan_cache(
            model,
            cache_config,
            scheduler_config,
            self.execution_config,
            attention.info["selected_backend"],
            speculative=speculative_config,
            staged_prefill=staged_prefill,
            fp8_scale=fp8_scale,
            attention=attention,
            layer_windows=layer_windows,
            reclaim_swa=reclaim_swa,
        )
        self.cache_manager = KVCacheManager(
            num_layers=kv_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_loops=config.total_ut_steps,
            num_blocks=num_blocks,
            layout=cache_config.layout,
            block_size=cache_config.block_size,
            device=parameter.device,
            dtype=parameter.dtype,
            backend=attention_backend,
            attention=attention,
            enable_prefix_caching=cache_config.enable_prefix_caching,
            incremental_allocation=cache_config.incremental_allocation,
            watermark_ratio=cache_config.watermark_ratio,
            recurrent_layers=getattr(model, "recurrent_kv_layers", None),
            kv_cache_dtype=cache_config.kv_cache_dtype,
            fp8_scale=fp8_scale,
            auxiliary_width=auxiliary_width,
            interleaved_kv=getattr(model, "interleaved_kv", False),
            layer_windows=layer_windows,
            reclaim_swa=reclaim_swa,
            max_submission_tokens=min(
                scheduler_config.prefill_chunk_size, scheduler_config.max_num_batched_tokens
            ),
        )
        self.scheduler = Scheduler(
            scheduler_config,
            self.cache_manager,
            speculative_config,
            produces_tokens=self.stage_config.produces_tokens,
            total_ut_steps=config.total_ut_steps,
            eos_token_id=config.eos_token_id,
            exit_mode=self.exit_config.mode,
            mixed_speculative=mixed_speculative,
            prioritize_prefill_continuations=(
                getattr(model, "interleaved_kv", False)
                and getattr(model, "fixed_loop_count", False)
                and connector is None
                and not scheduler_config.enable_preemption
            ),
        )
        self.connector = connector
        # One RNG registry per engine, shared by both runners: the CODA and
        # speculative paths must advance the same per-request slot, and only the
        # engine knows when a request terminates.
        self.rng = RngRegistry()
        runner_type = ModelRunner
        if parameter.device.type == "npu":
            from vllm_rlt.worker.ascend_model_runner import AscendModelRunner

            runner_type = AscendModelRunner
        self.model_runner = runner_type(
            model,
            self.cache_manager,
            exit_config=self.exit_config,
            execution_config=self.execution_config,
            scheduler_config=scheduler_config,
            staged_prefill=self.scheduler.staged_prefill,
            rng=self.rng,
        )
        if device_boundary_pipeline and self.async_speculative:
            # Speculative steps execute the boundary layers as well as Core.
            # Their Inkling MoE scratch is shared with new-request prefill.
            # Keep model work ordered and retain separate readback streams.
            self.model_runner.boundary_stream = self.model_runner.core_stream
        if mixed_speculative and self.model_runner.packed_graphs is not None:
            self.model_runner.packed_graphs.enable_speculative_core()
        self.graph_warmup = None
        if self.execution_config.cuda_graph_warmup:
            from vllm_rlt.worker.inkling_graph_warmup import warmup_inkling_graphs

            self.graph_warmup = warmup_inkling_graphs(self.model_runner, scheduler_config)
        spec_execution = self.execution_config
        if ordered_pipeline and not self.async_speculative:
            # Boundary calls share the normal model's graph owners, including
            # their keys, input buffers, and replay completion events.
            spec_execution = replace(spec_execution, cuda_graphs=False, cuda_graph_warmup=False)
        self.speculative_runner = (
            SpeculativeRunner(
                model,
                self.cache_manager,
                speculative_config,
                spec_execution,
                rng=self.rng,
                runtime=self.model_runner.runtime,
                compute_stream=self.model_runner.core_stream,
            )
            if speculative_config is not None and not self.async_speculative
            else None
        )
        if ordered_pipeline and self.speculative_runner is not None:
            self.speculative_runner.prelude_graphs = self.model_runner.prelude_graphs
            self.speculative_runner.graphs = self.model_runner.graphs
            self.speculative_runner.coda_graphs = self.model_runner.coda_graphs
            if device_boundary_pipeline and self.execution_config.cuda_graphs:
                assert speculative_config is not None
                from vllm_rlt.worker.cuda_graph import (
                    DeviceRecurrentGraphs,
                    speculative_capture_sizes,
                )

                graphs = DeviceRecurrentGraphs(
                    model,
                    self.cache_manager,
                    self.execution_config,
                    False,
                    capture_sizes=speculative_capture_sizes(
                        self.execution_config, scheduler_config, speculative_config
                    ),
                )
                graphs.prepare()
                self.speculative_runner.graphs = graphs
        if self.async_speculative:
            from vllm_rlt.worker.async_speculative import AsyncSpeculativeRunner

            self.speculative_runner = AsyncSpeculativeRunner(
                model,
                self.cache_manager,
                speculative_config,
                scheduler_config,
                execution=self.execution_config,
                compute_stream=self.model_runner.core_stream,
                copy_stream=self.model_runner.copy_stream,
                rng=self.rng,
                runtime=self.model_runner.runtime,
            )
        self.executor = Executor(self.model_runner, speculation=self.speculative_runner)
        self._pending_speculative = []
        self._retiring_speculative = {}
        self._exit_traces = {
            key: tuple(values) for key, values in (self.exit_config.depths_by_request or {}).items()
        }
        # Unconsumed exit scores: request ID -> SignalHandle.
        # Delayed policies consume the preceding loop's score after submitting the next.
        self._pending_exit_signals = {}
        self._pending_coda = []
        self._inflight = []
        self._overlap_boundary = False
        self.last_schedule = None
        self.preemption = PreemptionManager(self)
        if scheduler_config.enable_preemption:
            self.scheduler.suspend_callback = self.preemption.suspend
            self.scheduler.resume_callback = self.preemption.resume

    @property
    def connector(self):
        return self._connector

    @connector.setter
    def connector(self, connector):
        self._connector = connector
        self._connector_blocked = set()
        self.scheduler.execution_ready = (
            self._execution_ready
            if connector is not None or not self.stage_config.produces_tokens
            else None
        )
        self.scheduler.exports_request = self._exports_request if connector is not None else None

    def _exports_request(self, request):
        return self._connector.exports(request)

    def _execution_ready(self, request):
        if not self.stage_config.produces_tokens:
            if self.connector is None:
                raise RuntimeError("a prefill stage requires an outgoing connector")
            if not self.connector.exports(request):
                raise ValueError("prefill-only stage requires an outgoing route per request")
        ready = self.connector.can_schedule(request)
        if not ready:
            self._connector_blocked.add(request.request_id)
        return ready

    def _resolve_loop_bounds(self, params):
        model_depth = self.model.config.total_ut_steps
        fixed_loop_count = getattr(self.model, "fixed_loop_count", False)
        trace_mode = self.exit_config.mode == "trace"

        if trace_mode:
            if params.target_loops is not None and params.target_loops > model_depth:
                raise ValueError("target_loops cannot exceed total_ut_steps")
            max_loops = params.max_loops or model_depth
            if max_loops > model_depth or (not fixed_loop_count and params.min_loops > max_loops):
                raise ValueError("requested loop bounds exceed the model's supported depth")
            return (None if fixed_loop_count else model_depth), max_loops

        if params.target_loops is not None and not fixed_loop_count:
            raise ValueError("target_loops is only supported for fixed-loop models")
        if params.target_loops is not None and params.target_loops > model_depth:
            raise ValueError("target_loops cannot exceed total_ut_steps")

        if fixed_loop_count:
            target_loops = params.target_loops or model_depth
            if params.max_loops is not None and params.max_loops != target_loops:
                raise ValueError("max_loops must match the fixed target_loops")
            if params.min_loops not in (DEFAULT_MIN_LOOPS, target_loops):
                raise ValueError(
                    "min_loops and max_loops must match the fixed target_loops when specified, "
                    "except the min_loops default"
                )
            if target_loops != model_depth:
                raise ValueError(
                    "target_loops below total_ut_steps requires per-request prefill support"
                )
            return target_loops, target_loops

        max_loops = params.max_loops or model_depth
        if max_loops > model_depth or params.min_loops > max_loops:
            raise ValueError("requested loop bounds exceed the model's supported depth")
        return model_depth, max_loops

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        *,
        trace_id: str | None = None,
    ):
        if not self.stage_config.accepts_prompt:
            raise ValueError("decode stage requests must enter through reserve_transfer")
        self._add_request(request_id, prompt_token_ids, sampling_params, trace_id=trace_id)

    def _add_request(self, request_id, prompt_token_ids, sampling_params, *, trace_id=None):
        """Validate and register a stage-local request from either ingress path."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        params = sampling_params or SamplingParams()
        if (
            self.speculative_config and self.speculative_config.mode == "pipeline"
        ) and params.temperature != 0:
            raise ValueError("pipeline speculative decoding currently requires greedy sampling")
        if request_id in self._retiring_speculative:
            self._retire_speculative(wait=True, request_id=request_id)
        config = self.model.config
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if any(type(t) is not int or not 0 <= t < config.vocab_size for t in prompt_token_ids):
            raise ValueError("prompt token IDs must be integers within the model vocabulary")
        fixed_loop_count = getattr(self.model, "fixed_loop_count", False)
        target_loops, max_loops = self._resolve_loop_bounds(params)
        if next(self.model.parameters()).device.type == "npu" and (
            max_loops != 4 or params.exit_threshold != 1.0 or params.temperature != 0
        ):
            raise ValueError("NPU Ouro requests require four loops, exit_threshold=1 and greedy")
        if self.speculative_config is not None and (
            max_loops != self.speculative_config.target_loops or params.exit_threshold != 1.0
        ):
            raise ValueError("speculative requests require fixed target depth and exit_threshold=1")
        if trace_id is not None:
            if not isinstance(trace_id, str) or not trace_id:
                raise ValueError("trace_id must be a nonempty string")
            if self.exit_config.mode != "trace":
                raise ValueError("trace_id requires trace exit mode")
        trace = ()
        if self.exit_config.mode == "trace":
            key = request_id if trace_id is None else trace_id
            if key not in self._exit_traces:
                raise ValueError(f"unknown exit trace: {key!r}; specify a configured trace_id")
            trace = self._exit_traces[key]
            if (
                len(trace) < params.max_tokens
                or trace[0] != config.total_ut_steps
                or any(
                    type(d) is not int
                    or not (1 if fixed_loop_count else params.min_loops) <= d <= max_loops
                    for d in trace[1 : params.max_tokens]
                )
            ):
                raise ValueError(
                    "exit trace must cover every output and obey prefill/decode bounds"
                )
        capacity = len(prompt_token_ids) + params.max_tokens - 1
        if capacity > config.max_position_embeddings:
            raise ValueError("prompt plus decode positions exceed the model context length")
        cache = self.cache_manager
        required = cache.required_blocks(capacity)
        if required > cache.num_blocks:
            raise ValueError(
                f"request requires {required} KV blocks but cache has {cache.num_blocks}; "
                "increase num_blocks or reduce prompt/max_tokens"
            )
        self.scheduler.add_request(
            Request(
                request_id,
                list(prompt_token_ids),
                params,
                exit_trace=trace,
                # Prefill always runs the model's full depth, whatever decode
                # loop bound the request asks for.
                prefill_depth_ranges=Request.prefill_boundaries(
                    len(prompt_token_ids), config.total_ut_steps
                ),
                _target_loops=(
                    target_loops if fixed_loop_count and self.exit_config.mode != "trace" else None
                ),
            )
        )

    def request_ids(self):
        return tuple(self.scheduler.requests)

    def has_request(self, request_id):
        request = self.scheduler.requests.get(request_id)
        return request is not None and request.phase != RequestPhase.FINISHED

    def stats(self):
        return dict(
            requests=len(self.scheduler.requests), used_blocks=self.cache_manager.num_used_blocks
        )

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests or bool(self._pending_speculative)

    def close(self):
        """Drain owned GPU work before serving teardown drops tensor owners."""
        self.profiling.close()
        self.model_runner.synchronize()
        if self.execution_config.cuda_graphs and getattr(
            self.model.config, "auxiliary_cache_width", 0
        ):
            names = ("prelude_graphs", "graphs", "mixed_graphs", "packed_graphs", "coda_graphs")
            for name in names:
                graph = getattr(self.model_runner, name)
                if graph is None:
                    continue
                logger.info(
                    "Inkling %s CUDA Graph captures=%d replays=%d fallbacks=%d",
                    name,
                    graph.captures,
                    graph.replays,
                    graph.fallbacks,
                )
        if self.async_speculative:
            cast("AsyncSpeculativeRunner", self.speculative_runner).synchronize()
        for rid in list(self.scheduler.requests):
            self.abort_request(rid)
        if self.async_speculative:
            self._discard_speculative()

    def _discard_speculative(self):
        """Release speculative owners after the caller has drained both streams."""
        self._retire_speculative(wait=True)
        for ticket in self._pending_speculative:
            ticket.retire(discard=True)
        self._pending_speculative.clear()
        cast("AsyncSpeculativeRunner", self.speculative_runner).discard_buffers()

    def abort_request(self, request_id: str) -> RequestOutput:
        request = self.scheduler.requests[request_id]
        if request.phase == RequestPhase.FINISHED:
            return RequestOutput.from_request(request)
        if self.async_speculative:
            request = self.scheduler.requests[request_id]
            self._finish(request, FinishReason.ABORT)
            return RequestOutput.from_request(request)
        self.preemption.discard_snapshot(request_id)
        if self.speculative_runner is not None:
            self.speculative_runner.release(request)
        self.model_runner.release(request_id)
        self._pending_exit_signals.pop(request_id, None)
        return RequestOutput.from_request(self.scheduler.abort(request_id))

    def start_profile(self, config, *, scheduled=False, **identity):
        return self.profiling.start(config, scheduled=scheduled, **identity)

    def stop_profile(self):
        return self.profiling.stop()

    def profile_status(self):
        return self.profiling.status()

    def wait_for_profile_artifacts(self, timeout=None):
        return self.profiling.wait(timeout)

    def step(self) -> list[RequestOutput]:
        if not self.profiling.recording:
            return self._step()
        try:
            outputs = self._step()
            self.profiling.step()
            return outputs
        except BaseException:
            self.profiling.stop()
            raise

    def _step(self) -> list[RequestOutput]:
        if self.execution_config.async_scheduling:
            try:
                if self.async_speculative:
                    return self._step_async_speculative()
                return self._step_async()
            except Exception:
                if self.connector is not None:
                    # The deployment owner must terminate remote writers before
                    # pages can be reclaimed after a fatal transport/compute error.
                    raise
                # A submission can fail before recording an event. Drain owned
                # streams before invalidating requests or recycling any memory.
                self.model_runner.synchronize()
                if self.async_speculative:
                    cast("AsyncSpeculativeRunner", self.speculative_runner).synchronize()
                for rid in list(self.scheduler.requests):
                    self.abort_request(rid)
                if self.async_speculative:
                    self._discard_speculative()
                self._pending_exit_signals.clear()
                self._pending_coda.clear()
                self._inflight.clear()
                raise
        batch = self._schedule()
        self.last_schedule = batch
        if batch is None:
            # PD imports wait for external KV completion; yield to the IPC loop.
            if not self._waiting_for_connector():
                raise RuntimeError("scheduler made no progress")
            return []
        try:
            ticket = self.executor.submit(batch)
            # Only host-dependent decisions collect now. Prompt/PRELUDE submissions
            # advance logically under their event without a readback wait.
            result = ticket.collect() if ticket.has_result else None
            outputs = self._update(batch, result)
            # Publish this submission's prompt pages before SWA retires them.
            self.cache_manager.finish_submission(ticket.event)
            return outputs
        except Exception:
            if self.connector is not None:
                raise  # Transfer leases remain quarantined until peer shutdown.
            # A failed execution may have partially written KV; invalidate the affected requests.
            for scheduled_request in batch.requests:
                if scheduled_request.request.request_id in self.scheduler.requests:
                    self.abort_request(scheduled_request.request.request_id)
            raise

    def _schedule(self, *, prefer_recurrent=False):
        self._connector_blocked.clear()
        return self.scheduler.schedule(prefer_recurrent=prefer_recurrent)

    def _waiting_for_connector(self):
        # Capture this scheduling attempt's readiness observation. Re-querying
        # an event here can misclassify progress that occurred after selection.
        transfer_waiters = {
            r.request_id
            for r in self.scheduler.requests.values()
            if r.phase in (RequestPhase.RECEIVING, RequestPhase.TRANSFERRING)
            or (
                r.request_id in self._connector_blocked
                and r.request_id in self.scheduler.queues[r.phase]
            )
        }
        return all(
            r.request_id in transfer_waiters
            # An admitted export may retain a compute slot while backpressured,
            # even before it reaches TRANSFERRING. Queued arrivals must yield too.
            # A WAITING request absent from its queue remains an invariant error.
            or (transfer_waiters and r.request_id in self.scheduler.queues[RequestPhase.WAITING])
            for r in self.scheduler.requests.values()
        )

    def _notify_connector_prefill(self, scheduled_request, phase, progress):
        request = scheduled_request.request
        if self.connector is not None and self.connector.exports(request):
            self.connector.prefill_submitted(
                request,
                phase=phase,
                start=(
                    scheduled_request.token_start
                    if phase == RequestPhase.PREFILL
                    else progress.transfer_start
                    if progress.transfer_start is not None
                    else scheduled_request.chunk_start
                ),
                progress=progress,
                event=self.model_runner.events.get(request.request_id),
                stream=self.model_runner.core_stream,
            )

    def reserve_transfer(
        self, request_id, prompt_token_ids, sampling_params, *, receiving, trace_id=None
    ):
        """Create one stage-local Request and reserve/pin its transfer allocation.

        The wire ID includes the transfer generation. Subsequent chunk messages
        reuse this Request; a repeated reservation is rejected by add_request.
        None means admission failed without retaining a request or KV allocation.
        """
        if receiving:
            self._add_request(request_id, prompt_token_ids, sampling_params, trace_id=trace_id)
        else:
            self.add_request(request_id, prompt_token_ids, sampling_params, trace_id=trace_id)
        request = self.scheduler.requests[request_id]
        plan = self.scheduler.reserve_transfer(request)
        if plan is None:
            self.scheduler.abort(request_id)
            return None
        count = len(prompt_token_ids)
        hit = plan.cached_tokens
        request.num_prefilled_tokens = request.num_preluded_tokens = hit
        request.prefill_window_origin = hit
        request.prefill_reclaimable_tokens = hit
        request.prefill_depth_ranges[:] = Request.prefill_boundaries(
            count, self.model.config.total_ut_steps, start=hit
        )
        # Reservation owns transport resources, not a compute slot. Outgoing
        # requests compete with local arrivals in the scheduler's WAITING queue.
        if receiving:
            self.scheduler.queues[RequestPhase.WAITING].remove(request_id)
            request.phase = RequestPhase.RECEIVING
        return request

    def suspend_transfer(self, request):
        """Cancel future computation, retaining in-flight state and transfer leases."""
        return self.scheduler.suspend_for_transfer(request)

    def release_transfer_compute(self, request, *, ready_event):
        """Release compute state after the connector's final staging read completes.

        KV and transfer staging remain leased until delivery or safe cancellation.
        Query is nonblocking; a stale Request can never release its replacement.
        """
        if self.scheduler.requests.get(request.request_id) is not request:
            return False
        if request.phase != RequestPhase.TRANSFERRING:
            raise ValueError("request is not awaiting transfer completion")
        if not ready_event.query():
            return False
        self.model_runner.release(request.request_id)
        request.hidden_state = None
        request.compute_released = True
        return True

    def release_transfer(self, request):
        """Called only after remote accesses are known to have ended."""
        if self.scheduler.requests.get(request.request_id) is not request:
            return False
        rid = request.request_id
        if self.async_speculative:
            if request.phase != RequestPhase.FINISHED:
                self._finish(request, FinishReason.ABORT)
            self._retire_speculative()
            if self.scheduler.requests.get(rid) is request:
                return False
        else:
            self.model_runner.release(rid)
            self.scheduler.abort(rid)
        if self.cache_manager.has_allocation(rid):
            self.cache_manager.unpin_transfer(rid, rid)
        return True

    def accept_import(self, request, hidden, *, cached_tokens=0):
        """Queue a valid imported Request for ordinary scheduler admission.

        The transfer generation is checked by the connector before this call;
        object identity additionally rejects local cancellation/ID reuse.
        """
        if self.scheduler.requests.get(request.request_id) is not request:
            return False
        if (
            request.phase != RequestPhase.RECEIVING
            or request.request_id in self.scheduler.queues[RequestPhase.WAITING]
        ):
            raise ValueError("request is not awaiting an import")
        count = len(request.prompt_token_ids)
        self.cache_manager.mark_imported_prefix(request.request_id, count, start=cached_tokens)
        self.cache_manager.publish_prefix(request.request_id, request.prompt_token_ids, count)
        request.num_prefilled_tokens = request.num_preluded_tokens = count
        request.prefill_window_origin = count
        request.prefill_reclaimable_tokens = count
        request.loops_done = request.target_loops or self.model.config.total_ut_steps
        request.hidden_state = hidden
        # Keep RECEIVING until the scheduler grants a compute slot. The lease
        # protects the imported allocation while it waits alongside local work.
        self.scheduler.queues[RequestPhase.WAITING].append(request.request_id)
        return True

    def _rewind_split_prefill(self, request):
        """Rewind an in-flight chunk; the caller rebuilds its queue membership."""
        if request.phase not in (
            RequestPhase.PREFILLING_PRELUDE,
            RequestPhase.PREFILLING_RECURRENT,
        ):
            return
        frontier = request.num_prefilled_tokens
        request.num_preluded_tokens = frontier
        request.prefill_window_origin = frontier
        request.prefill_reclaimable_tokens = frontier
        request.prefill_depth_ranges[:] = Request.prefill_boundaries(
            len(request.prompt_token_ids),
            self.model.config.total_ut_steps,
            start=frontier,
        )
        request.phase = RequestPhase.PREFILLING_PRELUDE

    def _apply_update(self, update) -> list[RequestOutput]:
        for scheduled_request, phase, progress in update.prefill:
            self._notify_connector_prefill(scheduled_request, phase, progress)
        if update.exited:
            self.model_runner.finalize_many(update.exited)
        if self.speculative_runner is not None:
            self.speculative_runner.stats.committed_tokens += update.committed_tokens
            self.speculative_runner.stats.accepted_tokens += update.accepted_tokens
        for request in update.finished:
            self._finish(request, request.finish_reason)
        return [RequestOutput.from_request(request) for request in update.outputs]

    def _update(
        self, batch, result: ModelRunnerOutput | None, *, delivered: bool = False
    ) -> list[RequestOutput]:
        return self._apply_update(
            self.scheduler.update_from_output(
                batch,
                result,
                request_events=self.model_runner.events,
                delivered=delivered,
            )
        )

    def _finish(self, request, reason: FinishReason):
        if self.async_speculative:
            # Logical finish is immediate; allocation ownership survives until
            # the final submitted round has stopped accessing the request.
            request.phase = RequestPhase.FINISHED
            request.finish_reason = reason
            for queue in self.scheduler.queues.values():
                while request.request_id in queue:
                    queue.remove(request.request_id)
            self._retiring_speculative[request.request_id] = request
            return
        if self.speculative_runner is not None:
            self.speculative_runner.release(request)
        self.model_runner.release(request.request_id)
        self.cache_manager.poll_prefixes()
        self._pending_exit_signals.pop(request.request_id, None)
        self.scheduler.finish(request, reason)

    def _retire_speculative(self, *, wait=False, request_id=None):
        runner = cast("AsyncSpeculativeRunner", self.speculative_runner)
        for rid, request in list(self._retiring_speculative.items()):
            if request_id is not None and rid != request_id:
                continue
            if not wait and not runner.request_ready(request):
                continue
            runner.release(request)
            self.model_runner.release(rid)
            self.cache_manager.poll_prefixes()
            self.scheduler.finish(request, request.finish_reason)
            del self._retiring_speculative[rid]

    def _collect_speculative(self):
        assert self.speculative_config is not None
        ticket = self._pending_speculative[0]
        runner = cast("AsyncSpeculativeRunner", self.speculative_runner)
        if not ticket.ready():
            runner.readback_waits += 1
        result = ticket.collect()
        for row, (scheduled_request, state, round_id) in enumerate(
            zip(ticket.batch.requests, ticket.states, ticket.rounds)
        ):
            request = scheduled_request.request
            if (
                self.scheduler.requests.get(request.request_id) is not request
                or request.phase == RequestPhase.FINISHED
            ):
                runner.discarded_rounds += 1
                continue
            if state.delivered != round_id:
                raise RuntimeError("out-of-order speculative round delivery")
            # Stale rows were skipped above, so only live requests can fail the round. The
            # check precedes _update (and the scheduler's EOS truncation): failure is atomic.
            if result.speculative[row].numerical_error:
                raise RuntimeError(
                    f"invalid speculative sampling distribution for request {request.request_id!r}"
                )
            state.delivered += 1
        outputs = self._update(ticket.batch, result)
        ticket.retire()
        self._pending_speculative.pop(0)
        self._retire_speculative()
        return outputs

    def _step_async_speculative(self):
        self._retire_speculative()
        outputs = []
        # Waiting admissions need a quiescent point before the scheduler can
        # select a victim. Victim selection remains entirely in Scheduler.
        if (
            self.pipeline_speculative
            and self.scheduler.config.enable_preemption
            and self.scheduler.queues[RequestPhase.WAITING]
        ):
            while self._pending_speculative:
                outputs.extend(self._collect_speculative())
        # Submit ahead before collecting, even when a small round already
        # completed. Two leased buffers provide a hard global queue bound.
        if len(self._pending_speculative) == 2:
            outputs.extend(self._collect_speculative())
        batch = self._schedule()
        self.last_schedule = batch
        if batch is not None:
            if batch.is_mixed:
                speculative = SchedulerOutput(
                    RequestPhase.SPECULATIVE,
                    [item for item in batch.requests if item.phase == RequestPhase.SPECULATIVE],
                    loop_count=0,
                )
                prompt = SchedulerOutput(
                    RequestPhase.PREFILLING_RECURRENT,
                    [
                        item
                        for item in batch.requests
                        if item.phase == RequestPhase.PREFILLING_RECURRENT
                    ],
                    loop_count=batch.loop_count,
                )
                ticket = self.executor.submit_mixed_speculative(speculative, prompt)
                self._pending_speculative.append(ticket)
                outputs.extend(self._update(prompt, None))
                self._apply_update(self.scheduler.update_after_submission(speculative))
            elif batch.is_speculative:
                ticket = self.executor.submit(batch)
                self._pending_speculative.append(ticket)
                self._apply_update(self.scheduler.update_after_submission(batch))
            else:
                # Bootstrap uses the same submit contract; only actual host
                # results (the first sampled token) require collection here.
                ticket = self.executor.submit(batch)
                result = ticket.collect() if ticket.has_result else None
                outputs.extend(self._update(batch, result))
        elif self._pending_speculative:
            outputs.extend(self._collect_speculative())
        elif self._retiring_speculative:
            self._retire_speculative(wait=True)
        elif self.scheduler.has_unfinished_requests and not self._waiting_for_connector():
            raise RuntimeError("speculative scheduler made no progress")
        return outputs

    def _deliver_coda(self, ticket):
        return self._update(ticket.batch, ticket.collect(), delivered=True)

    def _collect_coda(self, wait=False):
        outputs = []
        pending = []
        for ticket in self._pending_coda:
            if ticket.ready() or wait:
                outputs.extend(self._deliver_coda(ticket))
                wait = False
            else:
                pending.append(ticket)
        self._pending_coda = pending
        return outputs

    def _step_async(self):
        # Hold readback buffers until their DMA completes, even for discarded scores.
        self._inflight = [t for t in self._inflight if not t.ready()]
        outputs = self._collect_coda()
        batch = self._schedule(
            # If the preceding core is still running, boundary work can overlap
            # the independent next core. Once it has completed, refill first:
            # splitting on a briefly pending coda readback fragments the batch.
            prefer_recurrent=self._overlap_boundary
            and any(
                t.batch.phase == RequestPhase.RECURRENT and not t.ready() for t in self._inflight
            )
        )
        self._overlap_boundary = False
        self.last_schedule = batch
        if batch is None:
            # PD imports wait for external KV completion; yield to the IPC loop.
            if self._pending_coda:
                outputs.extend(self._collect_coda(wait=True))
            elif not self._waiting_for_connector():
                raise RuntimeError("scheduler made no progress")
            return outputs
        if batch.phase == RequestPhase.CODA:
            # Bound speculation to one output per request. Its next prelude and
            # core may run before delivery (including a possible EOS), but never
            # sample another token until that output's stop decision is known.
            while any(i.request.num_output_placeholders for i in batch.requests):
                outputs.extend(self._collect_coda(wait=True))
            batch = replace(
                batch,
                requests=[
                    i
                    for i in batch.requests
                    if self.scheduler.requests.get(i.request.request_id) is i.request
                ],
            )
            self.last_schedule = batch
            if not batch.requests:
                return outputs
        ticket = self.executor.submit(batch)
        self._inflight.append(ticket)
        if batch.phase == RequestPhase.CODA:
            self._pending_coda.append(ticket)
            runner = self.model_runner
            self._overlap_boundary = (
                runner.boundary_stream is not None
                and runner.boundary_stream is not runner.core_stream
            )
        signals = {}
        for row, scheduled_request in enumerate(batch.requests):
            phase = batch.phase_for(scheduled_request)
            request = scheduled_request.request
            if phase == RequestPhase.PRELUDE:
                self._pending_exit_signals.pop(request.request_id, None)
            elif phase == RequestPhase.RECURRENT:
                # Submission is already enqueued. Only now may the preceding score
                # block CPU delivery; max-depth/trace decisions require no readback.
                previous = self._pending_exit_signals.pop(request.request_id, None)
                if previous is None or not self.scheduler.needs_exit_signal(request):
                    continue
                old_ticket, old_row = previous.ticket, previous.row
                if old_ticket.batch.requests[old_row].request is not request:
                    continue
                signals[row] = ExitSignal(
                    request=request,
                    position=old_ticket.positions[old_row],
                    depth=old_ticket.depths[old_row] + 1,
                    score=old_ticket.collect().gate_scores[old_row],
                )
        update = self.scheduler.update_after_submission(
            batch,
            completion=ticket.event,
            request_events=self.model_runner.events,
            signals=signals,
        )
        self.model_runner.bind_sampled_tokens(batch, ticket.device_values, update.next_prelude)
        for row in update.retain_signal:
            self._pending_exit_signals[batch.requests[row].request.request_id] = SignalHandle(
                ticket, row
            )
        outputs.extend(self._apply_update(update))
        return outputs
