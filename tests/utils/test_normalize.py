"""Correctness tests for UCF_VIT.utils.normalize (zscore_normalize/zscore_denormalize).

Shared by every dataset's own loader (FileReader, CatsDogsDataset) and by
save_inference_batch's denormalization -- see that module's own docstring
for the full contract.
"""

import numpy as np
import pytest

from UCF_VIT.utils.normalize import zscore_normalize, zscore_denormalize


def test_zscore_normalize_channel_first_array():
    data = np.random.RandomState(0).rand(3, 4, 4).astype(np.float32) * 100
    variables = ["r", "g", "b"]
    stats = {"r": {"mean": 50.0, "std": 10.0}, "g": {"mean": 40.0, "std": 5.0}}

    norm = zscore_normalize(data, variables, stats, channel_axis=0)

    np.testing.assert_allclose(norm[0], (data[0] - 50.0) / 10.0)
    np.testing.assert_allclose(norm[1], (data[1] - 40.0) / 5.0)
    np.testing.assert_allclose(norm[2], data[2])  # "b" has no stats -- passthrough


def test_zscore_normalize_denormalize_round_trip_channel_first():
    data = np.random.RandomState(0).rand(3, 4, 4).astype(np.float32) * 100
    variables = ["r", "g", "b"]
    stats = {"r": {"mean": 50.0, "std": 10.0}, "g": {"mean": 40.0, "std": 5.0}, "b": {"mean": 1.0, "std": 2.0}}

    norm = zscore_normalize(data, variables, stats, channel_axis=0)
    denorm = zscore_denormalize(norm, variables, stats, channel_axis=0)

    np.testing.assert_allclose(denorm, data, atol=1e-4)


def test_zscore_normalize_channel_last_array():
    data = np.random.RandomState(0).rand(4, 4, 3).astype(np.float32) * 100
    variables = ["r", "g", "b"]
    stats = {"r": {"mean": 50.0, "std": 10.0}}

    norm = zscore_normalize(data, variables, stats, channel_axis=-1)

    np.testing.assert_allclose(norm[..., 0], (data[..., 0] - 50.0) / 10.0)
    np.testing.assert_allclose(norm[..., 1], data[..., 1])


def test_zscore_normalize_list_of_arrays_sst_convention():
    data_list = [np.random.RandomState(i).rand(4, 4).astype(np.float32) * 10 for i in range(2)]
    variables = [("u", -1), ("p", 0)]  # (name, offset) tuples -- offset dropped for stats lookup
    stats = {"u": {"mean": 1.0, "std": 2.0}, "p": {"mean": 0.0, "std": 1.0}}

    norm = zscore_normalize(data_list, variables, stats)

    np.testing.assert_allclose(norm[0], (data_list[0] - 1.0) / 2.0)
    np.testing.assert_allclose(norm[1], (data_list[1] - 0.0) / 1.0)


def test_zscore_denormalize_list_of_arrays_round_trip():
    data_list = [np.random.RandomState(i).rand(4, 4).astype(np.float32) * 10 for i in range(2)]
    variables = [("u", -1), ("p", 0)]
    stats = {"u": {"mean": 1.0, "std": 2.0}, "p": {"mean": 0.0, "std": 1.0}}

    norm = zscore_normalize(data_list, variables, stats)
    denorm = zscore_denormalize(norm, variables, stats)

    for a, b in zip(denorm, data_list):
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_zscore_normalize_empty_stats_is_passthrough():
    data = np.random.RandomState(0).rand(3, 4, 4).astype(np.float32) * 100
    norm = zscore_normalize(data, ["r", "g", "b"], {}, channel_axis=0)
    np.testing.assert_allclose(norm, data)


def test_zscore_normalize_none_stats_is_passthrough():
    data = np.random.RandomState(0).rand(3, 4, 4).astype(np.float32) * 100
    norm = zscore_normalize(data, ["r", "g", "b"], None, channel_axis=0)
    np.testing.assert_allclose(norm, data)


def test_zscore_normalize_always_returns_float32():
    data = np.random.RandomState(0).randint(0, 256, size=(3, 4, 4)).astype(np.uint8)
    norm = zscore_normalize(data, ["r", "g", "b"], {}, channel_axis=0)
    assert norm.dtype == np.float32


def test_zscore_normalize_guards_zero_std():
    data = np.full((1, 4, 4), 5.0, dtype=np.float32)
    stats = {"r": {"mean": 5.0, "std": 0.0}}
    norm = zscore_normalize(data, ["r"], stats, channel_axis=0)
    assert np.isfinite(norm).all()


def test_zscore_normalize_channel_count_mismatch_raises():
    data = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(AssertionError):
        zscore_normalize(data, ["r", "g"], {}, channel_axis=0)  # 3 channels, 2 variable names
