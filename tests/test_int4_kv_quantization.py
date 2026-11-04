import numpy as np
import pytest
import torch

from vllm_rlt.kernels.kv_quantization import (
    decode_residual,
    encode_residual,
    pack_kv,
    unpack_kv,
)

_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def scalar_reference(values, kind, group_size):
    """Scalar affine equations, independent of the codec's tensor layout operations."""
    source = values.float().cpu().numpy()
    tokens, heads, dimensions = source.shape
    codes = np.zeros(source.shape, dtype=np.uint8)
    decoded = np.zeros(source.shape, dtype=np.float32)
    extent = tokens if kind == "key" else dimensions
    for head in range(heads):
        for fixed in range(dimensions if kind == "key" else tokens):
            for start in range(0, extent, group_size):
                indices = [
                    (index, head, fixed) if kind == "key" else (fixed, head, index)
                    for index in range(start, min(start + group_size, extent))
                ]
                low = min(source[index] for index in indices)
                high = max(source[index] for index in indices)
                # CPU/CUDA can round division by 15 differently at code boundaries.
                step = (
                    np.float32((torch.tensor(high - low, device=values.device) / 15).item())
                    if high != low
                    else np.float32(1)
                )
                stored_low, stored_step = np.float32(np.float16(low)), np.float32(np.float16(step))
                for index in indices:
                    code = min(15, max(0, round(float(np.float32((source[index] - low) / step)))))
                    codes[index] = code
                    decoded[index] = np.float32(np.float32(code) * stored_step + stored_low)
    payload = np.zeros((tokens, heads, (dimensions + 1) // 2), dtype=np.uint8)
    for token in range(tokens):
        for head in range(heads):
            for dimension in range(dimensions):
                payload[token, head, dimension // 2] |= int(codes[token, head, dimension]) << (
                    4 * (dimension % 2)
                )
    return torch.from_numpy(payload), torch.from_numpy(decoded).to(values.dtype)


def test_known_nibble_order_and_ties_to_even():
    values = torch.tensor([0, 0.5, 1.5, 2.5, 14.5, 15]).reshape(1, 1, 6)
    packed = pack_kv(values, kind="value", group_size=6)
    assert packed.payload.tolist() == [[[0x00, 0x22, 0xFE]]]
    torch.testing.assert_close(
        unpack_kv(packed), torch.tensor([0, 0, 2, 2, 14, 15.0]).reshape(1, 1, 6)
    )
    odd = pack_kv(torch.tensor([[[0.0, 1.0, 15.0]]]), kind="value", group_size=3)
    assert odd.payload.tolist() == [[[0x10, 0x0F]]]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("group_size", [1, 3, 64])
def test_groups_partial_groups_and_noncontiguous_inputs(kind, dtype, group_size, device):
    values = torch.randn(5, 2, 14, generator=torch.Generator().manual_seed(714)).to(
        device=device, dtype=dtype
    )[..., ::2]
    assert not values.is_contiguous()
    original = values.clone()
    expected_payload, expected = scalar_reference(values, kind, group_size)
    packed = pack_kv(values, kind=kind, group_size=group_size)
    assert packed.original_shape == (5, 2, 7)
    assert packed.original_dtype == dtype
    assert packed.payload.dtype == torch.uint8
    assert packed.minimum.dtype == packed.scale.dtype == torch.float16
    groups = (
        (5 + group_size - 1) // group_size if kind == "key" else (7 + group_size - 1) // group_size
    )
    assert (
        packed.minimum.shape
        == packed.scale.shape
        == ((groups, 2, 7) if kind == "key" else (5, 2, groups))
    )
    torch.testing.assert_close(packed.payload.cpu(), expected_payload, rtol=0, atol=0)
    torch.testing.assert_close(unpack_kv(packed).cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("extent", [63, 64, 65, 128, 129])
def test_default_group_boundaries(kind, dtype, extent, device):
    tokens, channels = (extent, 7) if kind == "key" else (5, extent)
    values = torch.randn(tokens, 1, channels * 2, generator=torch.Generator().manual_seed(517)).to(
        dtype
    )[..., ::2]
    values = values.to(device)
    expected_payload, expected = scalar_reference(values, kind, 64)
    packed = pack_kv(values, kind=kind)
    torch.testing.assert_close(packed.payload.cpu(), expected_payload, rtol=0, atol=0)
    torch.testing.assert_close(unpack_kv(packed).cpu(), expected, rtol=0, atol=0)
    metadata_elements = (
        ((tokens + 63) // 64) * channels if kind == "key" else tokens * ((channels + 63) // 64)
    )
    assert packed.storage_bytes == tokens * ((channels + 1) // 2) + 4 * metadata_elements
    if channels % 2:
        assert torch.count_nonzero(packed.payload[..., -1] >> 4) == 0


def test_codes_use_parameters_before_fp16_metadata_rounding():
    values = torch.tensor([[[0.10001, 0.10005, 0.10009, 0.10021]]])
    packed = pack_kv(values, kind="value", group_size=4)
    assert packed.payload.tolist() == [[[0x30, 0xF6]]]
    assert packed.minimum.item() == torch.tensor(0.10001).half().item()
    _, expected = scalar_reference(values, "value", 4)
    torch.testing.assert_close(unpack_kv(packed), expected, rtol=0, atol=0)
    assert unpack_kv(packed)[0, 0, 0] != values[0, 0, 0]


@pytest.mark.parametrize("kind", ["key", "value"])
def test_constant_groups_and_scale_underflow(kind):
    packed = pack_kv(torch.full((3, 2, 5), 1.0001), kind=kind, group_size=2)
    assert torch.count_nonzero(packed.payload) == 0
    assert torch.all(packed.scale == 1)
    torch.testing.assert_close(unpack_kv(packed), torch.ones(3, 2, 5), rtol=0, atol=0)
    values = torch.tensor([0.0, 1e-8]).reshape((2, 1, 1) if kind == "key" else (1, 1, 2))
    packed = pack_kv(values, kind=kind, group_size=2)
    assert torch.count_nonzero(packed.payload) > 0
    assert torch.count_nonzero(packed.scale) == 0
    torch.testing.assert_close(unpack_kv(packed), torch.zeros_like(values), rtol=0, atol=0)


@pytest.mark.parametrize("kind,metadata_elements", [("key", 2 * 2 * 7), ("value", 5 * 2 * 3)])
def test_storage_bytes_include_metadata_and_channel_padding(kind, metadata_elements):
    packed = pack_kv(torch.ones(5, 2, 7), kind=kind, group_size=3)
    assert packed.storage_bytes == 5 * 2 * 4 + 4 * metadata_elements
    assert packed.storage_bytes == sum(
        tensor.numel() * tensor.element_size()
        for tensor in (packed.payload, packed.minimum, packed.scale)
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("shape", [(0, 2, 7), (5, 0, 7)])
def test_empty_token_or_head_dimension(kind, device, shape):
    values = torch.empty(shape, dtype=torch.bfloat16, device=device)
    packed = pack_kv(values, kind=kind)
    assert packed.payload.shape == (*shape[:2], 4)
    assert unpack_kv(packed).shape == values.shape
    assert unpack_kv(packed).dtype == values.dtype
    assert packed.storage_bytes == 0


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["key", "value"])
def test_oversized_group_does_not_expand_decode_scratch(kind):
    values = torch.randn(5, 2, 7, device="cuda", dtype=torch.bfloat16)
    peaks = []
    for group_size in (7, 65536):
        packed = pack_kv(values, kind=kind, group_size=group_size)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        decoded = unpack_kv(packed)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - allocated)
        assert decoded.shape == values.shape
    assert peaks[1] <= peaks[0]


@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_residual_chain_uses_reconstructed_predecessors(kind, dtype):
    generator = torch.Generator().manual_seed(429)
    raw_previous = torch.randn(5, 2, 7, generator=generator).to(dtype)
    reconstructed = unpack_kv(pack_kv(raw_previous, kind=kind, group_size=3))
    _, expected = scalar_reference(raw_previous, kind, 3)
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
    distinguishes_raw_predecessor = False
    for _ in range(3):
        current = (raw_previous + torch.randn(5, 2, 7, generator=generator).to(dtype) * 0.3).to(
            dtype
        )
        saved_current, saved_previous = current.clone(), reconstructed.clone()
        residual = encode_residual(current, reconstructed, kind=kind, group_size=3)
        expected_payload, decoded_delta = scalar_reference((current - expected).to(dtype), kind, 3)
        expected = (expected + decoded_delta).to(dtype)
        _, wrong_delta = scalar_reference((current - raw_previous).to(dtype), kind, 3)
        wrong = (reconstructed + wrong_delta).to(dtype)
        distinguishes_raw_predecessor |= not torch.equal(expected, wrong)
        torch.testing.assert_close(residual.data.payload, expected_payload, rtol=0, atol=0)
        previous_input = reconstructed
        reconstructed = decode_residual(residual, previous_input)
        torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
        torch.testing.assert_close(current, saved_current, rtol=0, atol=0)
        torch.testing.assert_close(previous_input, saved_previous, rtol=0, atol=0)
        assert residual.positions is None
        assert residual.num_tokens == 5
        assert residual.storage_bytes == residual.data.storage_bytes
        raw_previous = current
    assert distinguishes_raw_predecessor


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_quantized_base_nested_sparse_residual_chain(kind, dtype, device):
    generator = torch.Generator().manual_seed(937)
    raw_previous = torch.randn(129, 1, 65, generator=generator).to(dtype)
    reconstructed = unpack_kv(pack_kv(raw_previous.to(device), kind=kind))
    _, expected = scalar_reference(raw_previous.to(device), kind, 64)
    torch.testing.assert_close(reconstructed.cpu(), expected, rtol=0, atol=0)
    active = torch.arange(0, 129, 2, dtype=torch.int32)  # 65 compact K rows cross a group boundary.
    for positions in (active, active[::3]):
        current = (
            raw_previous[positions]
            + torch.randn(len(positions), 1, 65, generator=generator).to(dtype) * 0.3
        )
        expected_payload, decoded_delta = scalar_reference(
            (current - expected[positions]).to(device), kind, 64
        )
        saved_previous = reconstructed.clone()
        residual = encode_residual(
            current.to(device),
            reconstructed,
            kind=kind,
            positions=positions.to(device),
        )
        expected[positions] += decoded_delta
        raw_previous[positions] = current
        previous_input = reconstructed
        reconstructed = decode_residual(residual, previous_input)
        torch.testing.assert_close(previous_input, saved_previous, rtol=0, atol=0)
        torch.testing.assert_close(residual.data.payload.cpu(), expected_payload, rtol=0, atol=0)
        torch.testing.assert_close(reconstructed.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "index_dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32]
)
@pytest.mark.parametrize("contiguous", [False, True])
def test_sparse_rows_use_compact_key_groups_and_clone_position_map(index_dtype, contiguous):
    previous = torch.full((8, 1, 2), 10.0)
    positions = torch.tensor([1, 0, 6, 0, 7, 0], dtype=index_dtype)[::2]
    if contiguous:
        positions = positions.contiguous()
    current = torch.tensor([10.0, 25.0, 13.0]).reshape(3, 1, 1).expand(3, 1, 2)
    residual = encode_residual(current, previous, kind="key", positions=positions, group_size=2)
    assert residual.data.payload.tolist() == [[[0x00]], [[0xFF]], [[0x00]]]
    assert residual.num_tokens == 8
    assert residual.positions.dtype == torch.int32
    assert residual.positions.device == current.device
    assert residual.positions.is_contiguous()
    assert residual.storage_bytes == residual.data.storage_bytes + 3 * 4
    assert residual.positions.untyped_storage().nbytes() == 3 * 4
    positions.fill_(0)
    expected = previous.clone()
    expected[[1, 6, 7]] = current
    actual = decode_residual(residual, previous)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(previous, torch.full_like(previous, 10), rtol=0, atol=0)


def test_sparse_absence_differs_from_present_zero_codes():
    previous = torch.arange(24.0).reshape(6, 1, 4)
    positions = torch.tensor([1, 4])
    current = previous[positions] + 2
    residual = encode_residual(current, previous, kind="value", positions=positions, group_size=4)
    assert torch.count_nonzero(residual.data.payload) == 0
    expected = previous.clone()
    expected[positions] += 2
    torch.testing.assert_close(decode_residual(residual, previous), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_sparse_reconstruction_preserves_subnormal_values(device, index_dtype):
    tiny = torch.finfo(torch.float32).tiny / 2
    previous = torch.tensor([tiny, -tiny, 1, -1, -0.0, 0.0, tiny, -tiny], device=device).reshape(
        4, 1, 2
    )
    saved_previous = previous.clone()
    positions = torch.tensor([0, 2], device=device, dtype=index_dtype)
    residual = encode_residual(previous[positions], previous, kind="value", positions=positions)
    expected = previous.clone()
    expected[2].zero_()
    assert torch.equal(
        decode_residual(residual, previous).view(torch.int32), expected.view(torch.int32)
    )
    assert torch.equal(previous.view(torch.int32), saved_previous.view(torch.int32))


@pytest.mark.parametrize("kind", ["key", "value"])
def test_empty_residual_inherits_all_previous_rows(kind):
    previous = torch.arange(24.0).reshape(6, 1, 4)
    residual = encode_residual(
        previous[:0], previous, kind=kind, positions=torch.empty(0, dtype=torch.long)
    )
    torch.testing.assert_close(decode_residual(residual, previous), previous, rtol=0, atol=0)
    assert residual.num_tokens == 6
    assert residual.storage_bytes == 0


@pytest.mark.parametrize(
    "values",
    [
        torch.ones(2, 3),
        torch.ones(2, 1, 0),
        torch.ones(2, 1, 3, dtype=torch.int32),
    ],
)
def test_invalid_values_shape_and_dtype(values):
    with pytest.raises((ValueError, TypeError)):
        pack_kv(values, kind="value")


@pytest.mark.parametrize(
    "options",
    [
        {"group_size": 0},
        {"group_size": -2},
        {"kind": "other"},
    ],
)
def test_invalid_codec_options(options):
    with pytest.raises((ValueError, TypeError)):
        pack_kv(torch.ones(2, 1, 4), **{"kind": "value", **options})


@pytest.mark.parametrize("group_size", [True, 2.5])
def test_group_size_conversion_matches_flashloop(group_size):
    values = torch.tensor([[[0.0, 0.5, 1.0, 15.0]]])
    packed = pack_kv(values, kind="value", group_size=group_size)
    expected_payload, expected = scalar_reference(values, "value", int(group_size))
    assert packed.group_size == int(group_size)
    torch.testing.assert_close(packed.payload, expected_payload, rtol=0, atol=0)
    torch.testing.assert_close(unpack_kv(packed), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([[1, 2]]),
        torch.tensor([-1, 2]),
        torch.tensor([1, 6]),
        torch.tensor([2, 1]),
        torch.tensor([2, 2]),
        torch.tensor([1]),
    ],
)
def test_invalid_sparse_positions(positions):
    with pytest.raises((ValueError, TypeError)):
        encode_residual(torch.ones(2, 1, 4), torch.ones(6, 1, 4), kind="key", positions=positions)


def test_residual_shape_and_dtype_mismatches():
    previous = torch.ones(3, 1, 4)
    for current in (torch.ones(2, 1, 4), torch.ones(3, 2, 4), previous.half()):
        with pytest.raises((ValueError, TypeError)):
            encode_residual(current, previous, kind="value")
    residual = encode_residual(previous + 1, previous, kind="value")
    for wrong in (previous[:2], previous.half(), torch.tensor(1.0)):
        with pytest.raises((ValueError, TypeError)):
            encode_residual(previous, wrong, kind="value")
        with pytest.raises((ValueError, TypeError)):
            encode_residual(previous, wrong, kind="value", positions=torch.arange(3))
        with pytest.raises((ValueError, TypeError)):
            decode_residual(residual, wrong)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_codec_and_sparse_residual_match_cpu_equations(kind, dtype):
    previous = torch.randn(7, 2, 5, generator=torch.Generator().manual_seed(12)).to(dtype)
    positions = torch.tensor([0, 3, 6])
    current = previous[positions] + 0.2
    expected_payload, decoded_delta = scalar_reference(
        (current - previous[positions]).cuda(), kind, 2
    )
    expected = previous.clone()
    expected[positions] += decoded_delta
    residual = encode_residual(
        current.cuda(), previous.cuda(), kind=kind, positions=positions.cuda(), group_size=2
    )
    assert residual.data.payload.is_cuda and residual.positions.is_cuda
    torch.testing.assert_close(residual.data.payload.cpu(), expected_payload, rtol=0, atol=0)
    torch.testing.assert_close(
        decode_residual(residual, previous.cuda()).cpu(), expected, rtol=0, atol=0
    )
    transferred = encode_residual(
        current.cuda(), previous.cuda(), kind=kind, positions=positions, group_size=2
    )
    assert transferred.positions.is_cuda and transferred.positions.dtype == torch.int32
    torch.testing.assert_close(transferred.data.payload, residual.data.payload, rtol=0, atol=0)
    torch.testing.assert_close(
        decode_residual(transferred, previous.cuda()).cpu(), expected, rtol=0, atol=0
    )
