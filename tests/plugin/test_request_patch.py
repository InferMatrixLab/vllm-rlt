import subprocess
import sys
from textwrap import dedent

import pytest


def test_plugin_import_is_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_rlt.plugin; "
            "assert 'vllm' not in sys.modules and 'torch' not in sys.modules",
        ],
        check=True,
        timeout=60,
    )


def test_registration_updates_already_imported_request_and_preserves_wire_parameters():
    pytest.importorskip("vllm")
    # The patch deliberately affects every reference in this process. Isolate it
    # from the rest of the suite and exercise the upstream preprocessing call.
    subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""\
                import sys
                from types import ModuleType, SimpleNamespace

                from vllm.entrypoints.openai.completion.protocol import CompletionRequest
                from vllm.exceptions import VLLMValidationError
                from vllm.v1.engine import EngineCoreRequest, core
                from vllm.v1 import request as request_module

                from vllm_rlt.core.request import LoopRequest
                from vllm_rlt.plugin import register

                aliases = ModuleType('rlt_request_test_aliases')
                aliases.UpstreamRequest = request_module.Request
                sys.modules[aliases.__name__] = aliases

                register()
                register()
                assert core.Request is LoopRequest
                assert request_module.Request is aliases.UpstreamRequest is LoopRequest

                engine = object.__new__(core.EngineCore)
                engine.mm_receiver_cache = None
                engine.prefix_hash_config = None
                engine.request_block_hasher = None
                engine.vllm_config = SimpleNamespace(watermark_config=None)
                for index, depth in enumerate((1, 3, None)):
                    completion = CompletionRequest(
                        model='inkling', prompt=[1, 2], max_tokens=8, temperature=0,
                        vllm_xargs={'rlt_target_loops': depth} if depth is not None else None,
                    )
                    wire = EngineCoreRequest(
                        request_id=str(index), prompt_token_ids=[1, 2], mm_features=None,
                        sampling_params=completion.to_sampling_params(max_tokens=8),
                        pooling_params=None, arrival_time=0.0, lora_request=None,
                        cache_salt=None, data_parallel_rank=None, current_wave=2,
                    )
                    req, wave = engine.preprocess_add_request(wire)
                    assert type(req) is LoopRequest and wave == 2
                    assert req.loop_params.target_loops == depth
                    assert req.max_tokens == 8 and req.num_prompt_tokens == 2

                wire.sampling_params.extra_args = {'rlt_target_loops': 0}
                try:
                    engine.preprocess_add_request(wire)
                except VLLMValidationError as error:
                    assert error.parameter == 'rlt_target_loops'
                else:
                    raise AssertionError('invalid rlt_target_loops was accepted')
                """),
        ],
        check=True,
        timeout=90,
    )


@pytest.mark.parametrize(
    "args,expected_error",
    [
        ({"rlt_target_loops": 0}, "rlt_target_loops"),
        ({"rlt_target_loops": True}, "rlt_target_loops"),
        ({"rlt_exit_threshold": float("nan")}, "rlt_exit_threshold"),
        ({"rlt_min_loops": 3, "rlt_max_loops": 2}, "rlt_min_loops"),
        ({"rlt_target_loops": 2, "rlt_exit_threshold": 0.5}, "adaptive exit"),
    ],
)
def test_loop_params_reject_invalid_combinations(args, expected_error):
    pytest.importorskip("vllm")
    from vllm.exceptions import VLLMValidationError

    from vllm_rlt.core.request import LoopRequestParams

    with pytest.raises(VLLMValidationError, match=expected_error):
        LoopRequestParams.from_extra_args(args)
