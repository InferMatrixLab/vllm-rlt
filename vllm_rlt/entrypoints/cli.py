"""Serve the RLT models through vLLM's CLI and OpenAI frontend."""

import os

from vllm_rlt.entrypoints.runtime_args import add_runtime_args, apply_runtime_args


def main() -> None:
    allowlist = os.environ.get("VLLM_PLUGINS")
    if allowlist is not None and "rlt" not in allowlist.split(","):
        os.environ["VLLM_PLUGINS"] = ",".join(filter(None, (allowlist, "rlt")))

    from vllm.config.compilation import CUDAGraphMode
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.entrypoints.common.startup import cli_env_setup
    from vllm.logger import configure_logging_from_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    parser = FlexibleArgumentParser(description="vLLM-RLT CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    command = ServeSubcommand()
    serve_parser = command.subparser_init(subparsers)
    serve_parser.usage = "vllm-rlt serve [model_tag] [options]"
    serve_parser.set_defaults(async_scheduling=True, enforce_eager=None, generation_config="vllm")
    add_runtime_args(serve_parser)

    args = parser.parse_args()
    if args.enforce_eager is None:
        args.enforce_eager = args.compilation_config.cudagraph_mode in (None, CUDAGraphMode.NONE)
    try:
        apply_runtime_args(args)
    except ValueError as exc:
        serve_parser.error(str(exc))
    configure_logging_from_args(args)
    command.validate(args)
    command.cmd(args)


if __name__ == "__main__":
    main()
