from argparse import ArgumentParser

from vllm_rlt.entrypoints.runtime_args import add_runtime_args, apply_runtime_args


def test_cli_explicit_flags_overlay_additional_config():
    parser = ArgumentParser()
    parser.set_defaults(
        additional_config={"rlt_self_speculation": True, "rlt_trace_steps": True},
        async_scheduling=False,
        scheduler_cls=None,
        worker_cls="auto",
    )
    add_runtime_args(parser)
    args = parser.parse_args(
        [
            "--no-rlt-self-speculation",
            "--rlt-wavefront-prefill",
            "--rlt-prefill-chunk-size",
            "16",
            "--rlt-merge-decode-core",
            "--rlt-decode-state-slots",
            "--rlt-batch-decode-coda",
        ]
    )
    apply_runtime_args(args)

    assert args.additional_config == {
        "rlt_self_speculation": False,
        "rlt_trace_steps": True,
        "rlt_wavefront_prefill": True,
        "rlt_prefill_chunk_size": 16,
        "rlt_merge_decode_core": True,
        "rlt_decode_state_slots": True,
        "rlt_batch_decode_coda": True,
    }
    assert args.scheduler_cls == "vllm_rlt.core.loop_scheduler.LoopScheduler"
    assert args.worker_cls == "vllm_rlt.worker.loop_worker.LoopWorker"


def test_cli_leaves_explicit_upstream_classes_and_config():
    parser = ArgumentParser()
    parser.set_defaults(
        additional_config={"rlt_silu_and_mul": False},
        async_scheduling=True,
        scheduler_cls="custom.Scheduler",
        worker_cls="custom.Worker",
    )
    add_runtime_args(parser)
    args = parser.parse_args(["--rlt-silu-and-mul"])
    apply_runtime_args(args)

    assert args.additional_config == {"rlt_silu_and_mul": True}
    assert args.scheduler_cls == "custom.Scheduler"
    assert args.worker_cls == "custom.Worker"
