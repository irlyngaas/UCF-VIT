"""Tests for UCF_VIT.parse's train/val/test dataset-split resolution.

_resolve_dataset_splits (called from parse_config's DATA section) and
get_split_conf are what let val.py/test.py exist as separate scripts from
train.py: for each dataset key in dict_root_dirs, either a separate
dict_val_root_dirs/dict_test_root_dirs entry is used as-is (with its own
start/end idx, defaulting to the full [0,1) range), or -- since no shipped
dataset actually has one -- val/test are auto-split out of train's own
already-configured [dict_start_idx, dict_end_idx) window via
val_split_ratio/test_split_ratio (default 0.1/0.1 each), narrowing train's own
range in place. get_split_conf then lets val.py/test.py feed a remapped conf
into the exact same calculate_load_balancing_on_the_fly/NativePytorchDataModule
construction train.py already uses, unchanged.
"""

import pytest

from UCF_VIT.parse import _resolve_dataset_splits, get_split_conf


def _resolve(dict_root_dirs, dict_start_idx=None, dict_end_idx=None,
             dict_val_root_dirs=None, dict_val_start_idx=None, dict_val_end_idx=None,
             dict_test_root_dirs=None, dict_test_start_idx=None, dict_test_end_idx=None,
             val_split_ratio=0.1, test_split_ratio=0.1):
    return _resolve_dataset_splits(
        dict_root_dirs=dict_root_dirs,
        dict_start_idx=dict_start_idx or {},
        dict_end_idx=dict_end_idx or {},
        dict_val_root_dirs=dict_val_root_dirs or {},
        dict_val_start_idx=dict_val_start_idx or {},
        dict_val_end_idx=dict_val_end_idx or {},
        dict_test_root_dirs=dict_test_root_dirs or {},
        dict_test_start_idx=dict_test_start_idx or {},
        dict_test_end_idx=dict_test_end_idx or {},
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio,
    )


def test_default_80_10_10_narrows_train_within_full_0_1_window():
    train_start, train_end, val_root, val_start, val_end, test_root, test_start, test_end = _resolve(
        dict_root_dirs={"k": "/data/k"},
        dict_start_idx={"k": 0.0},
        dict_end_idx={"k": 1.0},
    )
    assert train_start == {"k": 0.0}
    assert train_end == {"k": pytest.approx(0.8)}
    assert val_root == {"k": "/data/k"}
    assert val_start == {"k": pytest.approx(0.8)}
    assert val_end == {"k": pytest.approx(0.9)}
    assert test_root == {"k": "/data/k"}
    assert test_start == {"k": pytest.approx(0.9)}
    assert test_end == {"k": pytest.approx(1.0)}


def test_auto_split_respects_an_already_narrowed_train_window():
    # Train already only claims [0.0, 0.5) of the directory (e.g. a smoke-test
    # narrowing) -- auto-split must carve val/test out of *that* window, not
    # assume [0, 1).
    train_start, train_end, val_root, val_start, val_end, test_root, test_start, test_end = _resolve(
        dict_root_dirs={"k": "/data/k"},
        dict_start_idx={"k": 0.0},
        dict_end_idx={"k": 0.5},
        val_split_ratio=0.2,
        test_split_ratio=0.2,
    )
    assert train_start == {"k": 0.0}
    assert train_end == {"k": pytest.approx(0.3)}  # 0.5 * (1 - 0.2 - 0.2)
    assert val_start == {"k": pytest.approx(0.3)}
    assert val_end == {"k": pytest.approx(0.4)}  # 0.3 + 0.5*0.2
    assert test_start == {"k": pytest.approx(0.4)}
    assert test_end == {"k": pytest.approx(0.5)}


def test_zero_ratios_reproduce_original_100_percent_train_behavior():
    train_start, train_end, val_root, val_start, val_end, test_root, test_start, test_end = _resolve(
        dict_root_dirs={"k": "/data/k"},
        dict_start_idx={"k": 0.0},
        dict_end_idx={"k": 1.0},
        val_split_ratio=0.0,
        test_split_ratio=0.0,
    )
    assert train_start == {"k": 0.0}
    assert train_end == {"k": 1.0}
    # No separate root and ratio 0.0 -- that split has no data for this key at
    # all, not a degenerate zero-width range.
    assert val_root == {}
    assert test_root == {}


def test_explicit_val_root_leaves_train_range_untouched():
    train_start, train_end, val_root, val_start, val_end, test_root, test_start, test_end = _resolve(
        dict_root_dirs={"k": "/data/k"},
        dict_start_idx={"k": 0.0},
        dict_end_idx={"k": 1.0},
        dict_val_root_dirs={"k": "/data/k_val"},
        test_split_ratio=0.1,
    )
    # Train keeps its full range -- only test's 0.1 was reserved, val had its
    # own separate root so nothing was carved out of train for it.
    assert train_end == {"k": pytest.approx(0.9)}
    assert val_root == {"k": "/data/k_val"}
    assert val_start == {"k": 0.0}
    assert val_end == {"k": 1.0}
    assert test_root == {"k": "/data/k"}
    assert test_start == {"k": pytest.approx(0.9)}
    assert test_end == {"k": pytest.approx(1.0)}


def test_explicit_val_root_with_its_own_start_end_idx():
    train_start, train_end, val_root, val_start, val_end, test_root, test_start, test_end = _resolve(
        dict_root_dirs={"k": "/data/k"},
        dict_val_root_dirs={"k": "/data/k_val"},
        dict_val_start_idx={"k": 0.25},
        dict_val_end_idx={"k": 0.75},
        test_split_ratio=0.0,
    )
    assert val_root == {"k": "/data/k_val"}
    assert val_start == {"k": 0.25}
    assert val_end == {"k": 0.75}


def test_mixed_per_key_behavior_across_two_dataset_keys():
    # One key has a real, separate val root; the other has none and gets
    # auto-split -- resolved independently, per key.
    train_start, train_end, val_root, val_start, val_end, test_root, test_start, test_end = _resolve(
        dict_root_dirs={"a": "/data/a", "b": "/data/b"},
        dict_start_idx={"a": 0.0, "b": 0.0},
        dict_end_idx={"a": 1.0, "b": 1.0},
        dict_val_root_dirs={"a": "/data/a_val"},
        val_split_ratio=0.1,
        test_split_ratio=0.0,
    )
    assert val_root == {"a": "/data/a_val", "b": "/data/b"}
    assert val_start["a"] == 0.0 and val_end["a"] == 1.0
    assert val_start["b"] == pytest.approx(0.9) and val_end["b"] == pytest.approx(1.0)
    assert train_end["a"] == 1.0  # untouched -- a's val was separate
    assert train_end["b"] == pytest.approx(0.9)  # narrowed -- b's val was auto-split
    assert test_root == {}  # test_split_ratio:0.0, no separate root for either key


def test_ratio_sum_too_large_raises():
    with pytest.raises(AssertionError):
        _resolve(
            dict_root_dirs={"k": "/data/k"},
            val_split_ratio=0.6,
            test_split_ratio=0.5,
        )


def test_get_split_conf_swaps_only_root_dirs_and_start_end_idx():
    conf = {
        "data": {
            "dict_root_dirs": {"k": "/data/k"},
            "dict_val_root_dirs": {"k": "/data/k"},
            "dict_test_root_dirs": {"k": "/data/k"},
            "num_channels": {"k": 3},
        },
        "dataloader": {
            "dict_start_idx": {"k": 0.0},
            "dict_end_idx": {"k": 0.8},
            "dict_val_start_idx": {"k": 0.8},
            "dict_val_end_idx": {"k": 0.9},
            "dict_test_start_idx": {"k": 0.9},
            "dict_test_end_idx": {"k": 1.0},
            "batch_size": 32,
        },
        "model": {"type": "MAE"},
    }

    val_conf = get_split_conf(conf, "val")
    assert val_conf["data"]["dict_root_dirs"] == {"k": "/data/k"}
    assert val_conf["dataloader"]["dict_start_idx"] == {"k": 0.8}
    assert val_conf["dataloader"]["dict_end_idx"] == {"k": 0.9}
    # Everything else passes through unchanged.
    assert val_conf["data"]["num_channels"] == {"k": 3}
    assert val_conf["dataloader"]["batch_size"] == 32
    assert val_conf["model"] is conf["model"]

    # Original conf untouched (shallow copy, not mutated in place).
    assert conf["dataloader"]["dict_start_idx"] == {"k": 0.0}

    test_conf = get_split_conf(conf, "test")
    assert test_conf["dataloader"]["dict_start_idx"] == {"k": 0.9}
    assert test_conf["dataloader"]["dict_end_idx"] == {"k": 1.0}


def test_get_split_conf_rejects_invalid_split_name():
    with pytest.raises(AssertionError):
        get_split_conf({}, "train")


def test_get_split_conf_raises_clearly_when_split_resolved_empty():
    conf = {
        "data": {"dict_val_root_dirs": {}},
        "dataloader": {"dict_val_start_idx": {}, "dict_val_end_idx": {}},
    }
    with pytest.raises(AssertionError, match="No val data available"):
        get_split_conf(conf, "val")
