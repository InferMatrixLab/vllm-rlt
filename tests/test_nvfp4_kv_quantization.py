import numpy as np
import pytest
import torch

from vllm_rlt.kernels.kv_quantization import (
    KVResidual,
    PackedInt4KV,
    PackedNVFP4KV,
    decode_residual,
    encode_residual,
    pack_kv,
    unpack_kv,
)
from vllm_rlt.kernels.kv_quantization.int4 import PackedInt4KV as BackendPackedInt4KV

_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP8_VALUES = tuple(
    fraction / 512 if exponent == 0 else (1 + fraction / 8) * 2 ** (exponent - 7)
    for exponent in range(16)
    for fraction in range(8 if exponent < 15 else 7)
)


def _nearest_code(value, table):
    return min(range(len(table)), key=lambda code: (abs(value - table[code]), code % 2))


def scalar_nvfp4(values, kind):
    """Enumerate finite E2M1/E4M3 values independently of Torch's casts/kernels."""
    source = values.float().cpu().numpy()
    tokens, heads, channels = source.shape
    maximum = np.max(np.abs(source)) if source.size else np.float32(0)
    with np.errstate(over="ignore"):
        global_scale = (
            np.float32(np.float32(np.float32(1) / maximum) * np.float32(2688))
            if maximum
            else np.float32(1)
        )
    global_scale = min(global_scale, np.float32(np.finfo(np.float32).max))
    reciprocal_global_scale = np.float32(np.float32(1) / global_scale)
    payload = np.zeros((tokens, heads, (channels + 1) // 2), dtype=np.uint8)
    scale_shape = (
        ((tokens + 15) // 16, heads, channels)
        if kind == "key"
        else (tokens, heads, (channels + 15) // 16)
    )
    scales = np.zeros(scale_shape, dtype=np.uint8)
    decoded = np.empty(source.shape, dtype=np.float32)
    for head in range(heads):
        for fixed in range(channels if kind == "key" else tokens):
            for start in range(0, tokens if kind == "key" else channels, 16):
                if kind == "key":
                    indices = [(row, head, fixed) for row in range(start, min(start + 16, tokens))]
                    scale_index = (start // 16, head, fixed)
                else:
                    indices = [
                        (fixed, head, col) for col in range(start, min(start + 16, channels))
                    ]
                    scale_index = (fixed, head, start // 16)
                group = np.array([source[index] for index in indices])
                block_maximum = np.max(np.abs(group))
                scale = np.float32(global_scale * np.float32(block_maximum * np.float32(1 / 6)))
                scale_code = _nearest_code(min(float(scale), 448), _FP8_VALUES)
                scales[scale_index] = scale_code
                stored_scale = np.float32(_FP8_VALUES[scale_code])
                effective_scale = np.float32(stored_scale * reciprocal_global_scale)
                decode_scale = np.float32(stored_scale / global_scale)
                with np.errstate(over="ignore"):
                    multiplier = (
                        np.float32(np.float32(1) / effective_scale)
                        if effective_scale
                        else np.float32(0)
                    )
                for index, value in zip(indices, group):
                    # Robustness extension: a positive scale can have an
                    # unrepresentable reciprocal while value / scale is finite.
                    normalized = np.float32(
                        value / effective_scale
                        if not np.isfinite(multiplier)
                        else value * multiplier
                    )
                    code = _nearest_code(abs(float(normalized)), _FP4_VALUES)
                    negative = np.signbit(normalized)
                    signed_code = code | (8 if negative else 0)
                    token, _, channel = index
                    payload[token, head, channel // 2] |= signed_code << (4 * (channel % 2))
                    quantized = np.float32(-_FP4_VALUES[code] if negative else _FP4_VALUES[code])
                    decoded[token, head, channel] = np.float32(quantized * decode_scale)
    return payload, scales, global_scale, torch.from_numpy(decoded).to(values.dtype)


def assert_matches_scalar(values, kind):
    expected_payload, expected_scale, expected_global_scale, expected = scalar_nvfp4(values, kind)
    packed = pack_kv(values, kind=kind, format="nvfp4")
    assert isinstance(packed, PackedNVFP4KV)
    assert packed.format == "nvfp4" and packed.axis == (0 if kind == "key" else 2)
    assert packed.group_size == 16
    assert packed.original_shape == tuple(values.shape) and packed.original_dtype == values.dtype
    assert packed.payload.dtype == torch.uint8
    assert packed.scale.dtype == torch.float8_e4m3fn
    assert packed.global_scale.dtype == torch.float32 and packed.global_scale.ndim == 0
    torch.testing.assert_close(
        packed.payload.cpu(), torch.from_numpy(expected_payload), rtol=0, atol=0
    )
    torch.testing.assert_close(
        packed.scale.cpu().view(torch.uint8), torch.from_numpy(expected_scale), rtol=0, atol=0
    )
    assert packed.global_scale.item() == expected_global_scale
    torch.testing.assert_close(unpack_kv(packed).cpu(), expected, rtol=0, atol=0)
    tokens, heads, channels = values.shape
    scale_bytes = (
        ((tokens + 15) // 16) * heads * channels
        if kind == "key"
        else tokens * heads * ((channels + 15) // 16)
    )
    assert packed.storage_bytes == tokens * heads * ((channels + 1) // 2) + scale_bytes + 4
    return packed


@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "shape",
    [
        (5, 2, 17),
        (2, 1, 16),
        (1, 2, 31),
        (2, 1, 1),
        (0, 2, 17),
        (3, 0, 17),
        (16, 1, 3),
        (17, 2, 3),
        (33, 2, 17),
    ],
)
def test_nvfp4_layout_partial_empty_and_noncontiguous(shape, dtype, kind):
    values = torch.randn(
        (*shape[:2], shape[2] * 2), generator=torch.Generator().manual_seed(831)
    ).to(dtype)[..., ::2]
    original = values.clone()
    packed = assert_matches_scalar(values, kind)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    if shape[2] % 2:
        assert torch.count_nonzero(packed.payload[..., -1] >> 4) == 0


@pytest.mark.parametrize("kind", ["key", "value"])
def test_nvfp4_group_axis_and_partial_tail(kind):
    values = torch.zeros(33, 1, 2)
    values[:16, 0, 0] = 6
    values[16:32, 0, 0] = 12
    values[32, 0, 0] = 24
    values[:, 0, 1] = 2688
    if kind == "value":
        values = values.transpose(0, 2)
    packed = assert_matches_scalar(values, kind)
    scales = packed.scale[:, 0, 0] if kind == "key" else packed.scale[0, 0, :]
    assert packed.global_scale.item() == 1
    torch.testing.assert_close(scales.float(), torch.tensor([1.0, 2.0, 4.0]), rtol=0, atol=0)
    torch.testing.assert_close(unpack_kv(packed), values, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_nvfp4_zero_groups_and_negative_zero(dtype):
    values = torch.zeros(2, 1, 17, dtype=dtype)
    values[..., 1::2] = -0.0
    values[..., -1] = -0.0
    packed = assert_matches_scalar(values, "key")
    assert packed.global_scale.item() == 1
    assert torch.count_nonzero(packed.scale.float()) == 0
    assert packed.payload[0, 0, 0].item() == 0x80
    assert packed.payload[0, 0, -1].item() == 0x08
    assert torch.equal(torch.signbit(unpack_kv(packed)), torch.signbit(values))


def test_nvfp4_tiny_inputs_keep_scales_and_reconstruction_finite():
    values = torch.tensor([1e-42, -1e-42, 0.0, -0.0]).repeat(4).reshape(1, 1, 16)
    packed = pack_kv(values, kind="key", format="nvfp4")
    decoded = unpack_kv(packed)
    assert torch.isfinite(packed.global_scale) and torch.isfinite(packed.scale.float()).all()
    assert torch.isfinite(decoded).all() and torch.count_nonzero(decoded) == 0
    assert torch.equal(torch.signbit(decoded), torch.signbit(values))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("anchor", [False, True])
def test_nvfp4_tiny_values_survive_reciprocal_overflow(dtype, anchor, device):
    tiny = 2**-24 if dtype == torch.float16 else 1e-38
    values = torch.tensor([tiny, -tiny, 0.0, -0.0], dtype=dtype).repeat(4).reshape(1, 1, 16)
    if anchor:
        anchored = torch.zeros_like(values)
        anchored[..., 0] = tiny * (16 if dtype == torch.float16 else 800)
        values = torch.cat((values, anchored), dim=2)
    packed = pack_kv(values.to(device), kind="key", format="nvfp4")
    decoded = unpack_kv(packed).cpu()
    if device == "cpu":
        expected_payload, expected_scale, _, expected = scalar_nvfp4(values, "key")
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            packed.payload.cpu(), torch.from_numpy(expected_payload), rtol=0, atol=0
        )
        torch.testing.assert_close(
            packed.scale.cpu().view(torch.uint8), torch.from_numpy(expected_scale), rtol=0, atol=0
        )
    assert torch.isfinite(decoded).all()
    nonzero = values != 0
    assert (decoded[nonzero] != 0).all()
    assert ((decoded[nonzero].float() / values[nonzero].float() - 1).abs() < 0.1).all()
    assert torch.equal(torch.signbit(decoded), torch.signbit(values))


@pytest.mark.parametrize("kind", ["key", "value"])
def test_nvfp4_fp4_midpoints_nextafter_and_even_codes(kind):
    midpoints = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
    probes = np.stack(
        (np.nextafter(midpoints, -np.inf), midpoints, np.nextafter(midpoints, np.inf)), axis=1
    ).reshape(-1)
    values = torch.zeros(len(probes) + 1, 1, 16)
    values[:-1, 0, 0] = torch.from_numpy(probes)
    values[:-1, 0, 1] = -torch.from_numpy(probes)
    values[:-1, 0, 2] = 6
    values[-1, 0, 0] = 2688  # Fix g=1 and each probe's effective scale=1.
    if kind == "key":
        values = values.transpose(0, 2)
    packed = assert_matches_scalar(values, kind)
    assert packed.global_scale.item() == 1
    if kind == "key":
        ties = torch.stack(
            [
                (packed.payload[0, 0, channel // 2] >> (4 * (channel % 2))) & 0x0F
                for channel in range(1, len(probes), 3)
            ]
        )
    else:
        ties = packed.payload[1:-1:3, 0, 0] & 0x0F
    assert ties.tolist() == [0, 2, 2, 4, 4, 6, 6]


@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_all_low_precision_magnitudes_choose_nearest_even_code(source_dtype):
    bits = torch.arange(32768, dtype=torch.int32).to(torch.int16)
    probes = bits.view(source_dtype).float()
    probes = probes[torch.isfinite(probes) & (probes <= 6)]
    values = torch.zeros(len(probes) + 1, 1, 16)
    values[:-1, 0, 0] = probes
    values[:-1, 0, 1] = -probes
    values[:-1, 0, 2] = 6
    values[-1, 0, 0] = 2688  # Keep global and probe block scales exactly one.
    expected = torch.tensor(
        [_nearest_code(value, _FP4_VALUES) for value in probes.tolist()], dtype=torch.uint8
    )
    packed = pack_kv(values, kind="value", format="nvfp4")
    assert packed.global_scale.item() == 1
    assert torch.equal(packed.payload[:-1, 0, 0], expected | ((expected | 8) << 4))


@pytest.mark.parametrize("kind", ["key", "value"])
def test_nvfp4_fp8_scale_rounding_and_finite_maximum(kind):
    targets = np.array([0.0001, 0.0009765625, 1.0625, 1.1875, 432, 448], dtype=np.float32)
    probes = np.stack(
        (np.nextafter(targets, -np.inf), targets, np.nextafter(targets, np.inf)), axis=1
    ).reshape(-1)
    values = torch.zeros(len(probes) + 1, 1, 16)
    values[:-1, 0, 0] = torch.from_numpy(probes) * 6
    values[-1, 0, 0] = 2688
    if kind == "key":
        values = values.transpose(0, 2)
    packed = assert_matches_scalar(values, kind)
    anchor = (0, 0, -1) if kind == "key" else (-1, 0, 0)
    assert packed.scale[anchor].float().item() == 448
    assert torch.isfinite(packed.scale.float()).all()
    assert 0 in packed.scale.view(torch.uint8)


@pytest.mark.parametrize("group_size", [1, 8, 32, 64])
@pytest.mark.parametrize("kind", ["key", "value"])
def test_nvfp4_requires_sixteen_element_groups(group_size, kind):
    with pytest.raises(ValueError):
        pack_kv(torch.ones(1, 1, 16), kind=kind, format="nvfp4", group_size=group_size)


@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_nvfp4_dense_and_nested_sparse_residuals(kind, dtype):
    generator = torch.Generator().manual_seed(712)
    raw_previous = torch.randn(65, 2, 17, generator=generator).to(dtype)
    reconstructed = unpack_kv(pack_kv(raw_previous, kind=kind, format="nvfp4"))
    _, _, _, expected = scalar_nvfp4(raw_previous, kind)
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
    active = torch.arange(0, 65, 3)
    for positions in (None, active, active[::2], active[:0]):
        raw_rows = raw_previous if positions is None else raw_previous[positions]
        current = raw_rows + torch.randn(raw_rows.shape, generator=generator).to(dtype) * 0.3
        previous_rows = expected if positions is None else expected[positions]
        _, _, _, expected_delta = scalar_nvfp4(current - previous_rows, kind)
        saved_current, saved_previous = current.clone(), reconstructed.clone()
        residual = encode_residual(
            current, reconstructed, kind=kind, format="nvfp4", positions=positions
        )
        assert isinstance(residual, KVResidual) and isinstance(residual.data, PackedNVFP4KV)
        previous_input = reconstructed
        reconstructed = decode_residual(residual, previous_input)
        torch.testing.assert_close(current, saved_current, rtol=0, atol=0)
        torch.testing.assert_close(previous_input, saved_previous, rtol=0, atol=0)
        mapping_bytes = 0 if positions is None else 4 * positions.numel()
        assert residual.storage_bytes == residual.data.storage_bytes + mapping_bytes
        if positions is None:
            expected = expected + expected_delta
            raw_previous = current
        else:
            inactive = torch.ones(raw_previous.shape[0], dtype=torch.bool)
            inactive[positions] = False
            torch.testing.assert_close(
                reconstructed[inactive], saved_previous[inactive], rtol=0, atol=0
            )
            expected[positions] = expected[positions] + expected_delta
            raw_previous[positions] = current
        torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [7, 65])
def test_nvfp4_cuda_matches_scalar_and_sparse_reconstruction(dtype, kind, tokens):
    previous = torch.randn(tokens, 2, 17, generator=torch.Generator().manual_seed(68)).to(dtype)
    packed = assert_matches_scalar(previous.cuda(), kind)
    previous = unpack_kv(packed).cpu()
    positions = torch.arange(0, tokens, 3)
    current = previous[positions] + 0.2
    _, _, _, delta = scalar_nvfp4(current - previous[positions], kind)
    expected = previous.clone()
    expected[positions] += delta
    residual = encode_residual(
        current.cuda(), previous.cuda(), kind=kind, format="nvfp4", positions=positions.cuda()
    )
    torch.testing.assert_close(
        decode_residual(residual, previous.cuda()).cpu(), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("kind,axis", [("key", 0), ("value", 2)])
def test_default_int4_selection_and_exports(kind, axis):
    assert PackedInt4KV is BackendPackedInt4KV
    values = torch.randn(65, 1, 17, generator=torch.Generator().manual_seed(412))
    legacy = pack_kv(values, kind=kind)
    explicit = pack_kv(values, kind=kind, format="int4")
    assert isinstance(legacy, PackedInt4KV)
    assert legacy.format == explicit.format == "int4"
    assert legacy.axis == explicit.axis == axis
    assert legacy.group_size == explicit.group_size == 64
    for name in ("payload", "minimum", "scale"):
        torch.testing.assert_close(getattr(legacy, name), getattr(explicit, name), rtol=0, atol=0)
    torch.testing.assert_close(unpack_kv(legacy), unpack_kv(explicit), rtol=0, atol=0)
    residual = encode_residual(values, unpack_kv(legacy), kind=kind)
    assert isinstance(residual, KVResidual) and isinstance(residual.data, PackedInt4KV)


def test_bfloat16_mixed_format_residual_chain():
    generator = torch.Generator().manual_seed(614)
    raw_previous = torch.randn(9, 2, 17, generator=generator).bfloat16()
    base = pack_kv(raw_previous, kind="key", group_size=None)
    reconstructed = unpack_kv(base)
    expected = reconstructed.clone()
    torch.testing.assert_close(reconstructed, raw_previous, rtol=0, atol=0.25)
    formats = [base.format]
    for format, positions in (("nvfp4", None), ("int4", torch.tensor([0, 3, 7]))):
        raw_rows = raw_previous if positions is None else raw_previous[positions]
        current = raw_rows + torch.randn(raw_rows.shape, generator=generator).bfloat16() * 0.3
        previous_rows = expected if positions is None else expected[positions]
        delta = current - previous_rows
        expected_delta = unpack_kv(pack_kv(delta, kind="key", format=format, group_size=None))
        residual = encode_residual(
            current, reconstructed, kind="key", format=format, group_size=None, positions=positions
        )
        formats.append(residual.data.format)
        assert residual.data.group_size == (16 if format == "nvfp4" else 64)
        previous_input = reconstructed
        saved_previous = previous_input.clone()
        reconstructed = decode_residual(residual, previous_input)
        torch.testing.assert_close(previous_input, saved_previous, rtol=0, atol=0)
        if positions is None:
            expected = expected + expected_delta
            raw_previous = current
        else:
            expected[positions] = expected[positions] + expected_delta
            inactive = torch.ones(9, dtype=torch.bool)
            inactive[positions] = False
            torch.testing.assert_close(
                reconstructed[inactive], saved_previous[inactive], rtol=0, atol=0
            )
        assert reconstructed.shape == raw_previous.shape
        assert reconstructed.dtype == torch.bfloat16
        torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
        active_rows = reconstructed if positions is None else reconstructed[positions]
        torch.testing.assert_close(active_rows, current, rtol=0, atol=0.25)
    assert formats == ["int4", "nvfp4", "int4"]


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="format"):
        pack_kv(torch.ones(1, 1, 16), kind="key", format="unknown")
