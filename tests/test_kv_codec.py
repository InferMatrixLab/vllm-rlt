from dataclasses import dataclass, fields

import pytest
import torch

from vllm_rlt.kernels.kv_quantization import (
    KV_CODECS,
    decode_residual,
    encode_residual,
    pack_kv,
    unpack_kv,
)

_FORMATS = [("int4", 64), ("int4", 16), ("nvfp4", 16)]


@pytest.mark.parametrize("format,group_size", _FORMATS)
@pytest.mark.parametrize("kind,axis", [("key", 0), ("value", 2)])
@pytest.mark.parametrize("shape", [(64, 2, 64), (5, 2, 7), (0, 2, 7)])
def test_codec_metadata_and_actual_storage(format, group_size, kind, axis, shape, record_property):
    codec = KV_CODECS[format]
    values = torch.ones(shape, dtype=torch.bfloat16)
    packed = pack_kv(values, kind=kind, format=format, group_size=group_size)
    assert packed.format == codec.format == format
    assert codec.storage_dtype == packed.payload.dtype
    assert packed.group_axis == codec.group_axis(kind) == axis
    expected_shape = list(shape)
    expected_shape[axis] = (shape[axis] + group_size - 1) // group_size
    assert (
        packed.scale_shape
        == codec.scale_shape(shape, group_size, kind=kind)
        == tuple(expected_shape)
    )
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for field in fields(packed)
        if isinstance(tensor := getattr(packed, field.name), torch.Tensor)
    )
    assert packed.storage_bytes == codec.storage_bytes(packed) == tensor_bytes
    if values.numel():
        bits_per_element = tensor_bytes * 8 / values.numel()
        record_property("bits_per_element", bits_per_element)
        if shape == (64, 2, 64):
            expected = 4 + 32 / group_size if format == "int4" else 4.5 + 32 / values.numel()
            assert bits_per_element == expected


def test_registered_codec_works_without_format_dispatch_changes(monkeypatch):
    @dataclass(frozen=True)
    class Packed:
        payload: torch.Tensor
        original_dtype: torch.dtype
        format = "test_bf16"

        @property
        def original_shape(self):
            return tuple(self.payload.shape)

    class Codec:
        default_group_size = 1

        @staticmethod
        def pack(values, *, kind, group_size):
            assert kind == "key" and group_size == 1
            return Packed(values.clone(), values.dtype)

        @staticmethod
        def unpack(packed):
            return packed.payload.clone()

    monkeypatch.setitem(KV_CODECS, "test_bf16", Codec)
    values = torch.arange(12, dtype=torch.bfloat16).reshape(3, 1, 4)
    previous = unpack_kv(pack_kv(values, kind="key", format="test_bf16"))
    torch.testing.assert_close(previous, values, atol=0, rtol=0)
    positions = torch.tensor([0, 2])
    current = previous[positions] + 0.5
    residual = encode_residual(
        current, previous, kind="key", format="test_bf16", positions=positions
    )
    expected = previous.clone()
    expected[positions] = current
    torch.testing.assert_close(decode_residual(residual, previous), expected, atol=0, rtol=0)
