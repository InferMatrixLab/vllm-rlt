import pytest

from vllm_rlt.config import LoopRuntimeConfig


def test_runtime_options_share_vllm_config_without_consuming_other_keys():
    options = LoopRuntimeConfig.from_additional_config(
        {"rlt_wavefront_prefill": True, "rlt_prefill_chunk_size": 16, "other_setting": "enabled"}
    )
    assert options == LoopRuntimeConfig(wavefront_prefill=True, prefill_chunk_size=16)


@pytest.mark.parametrize(
    "key,value",
    [
        ("rlt_silu_and_mul", "false"),
        ("rlt_wavefront_prefill", 1),
        ("rlt_prefill_chunk_size", 0),
        ("rlt_prefill_chunk_size", True),
        ("rlt_wavefront_prefil", True),
    ],
)
def test_runtime_config_rejects_invalid_values_and_unknown_flags(key, value):
    with pytest.raises(ValueError, match=key):
        LoopRuntimeConfig.from_additional_config({key: value})
