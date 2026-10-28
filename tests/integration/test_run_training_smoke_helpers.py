"""Correctness tests for run_training_smoke.py's real-data-narrowing and
config-override helpers.

run_training_smoke.py itself isn't a pytest file (no test_ prefix, and it
has its own if __name__ == "__main__" entry point -- see its module
docstring for why it's a plain script, not pytest), but its narrowing
helpers are plain, unit-testable functions, and are reused directly by
tests/distributed/test_dataloader_real_pipeline.py and
tests/dataloaders/test_dataset_speed_real_data.py, not just Tier 3 itself.
deep_merge_config_overrides is reused by
tests/integration/run_feature_matrix_smoke.py (Tier 3b).
"""

import math
import os
import tempfile

import pytest
import yaml

from run_training_smoke import (
    NoRealDataFoundError,
    compute_narrow_dict_idx,
    deep_merge_config_overrides,
    inflate_min_files_for_train_split,
    make_smoke_config,
)


def _base_conf(dict_root_dirs, dataset="basic_ct"):
    return {
        "dataloader": {"type": "iterative_dataloader"},
        "data": {"dataset": dataset, "dict_root_dirs": dict_root_dirs},
        "parallelism": {"fsdp_size": 1, "simple_ddp_size": 8},
    }


def test_compute_narrow_dict_idx_real_data_found(tmp_path):
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    for i in range(20):
        (images_dir / f"image{i}.nii").write_text("")

    conf = _base_conf({"ct1": str(tmp_path)})
    result = compute_narrow_dict_idx(conf, min_files=5)
    assert result["ct1"] == pytest.approx(5 / 20)


def test_compute_narrow_dict_idx_empty_but_existing_dir_raises(tmp_path):
    (tmp_path / "imagesTr").mkdir()  # exists, but no files in it

    conf = _base_conf({"ct1": str(tmp_path)})
    with pytest.raises(NoRealDataFoundError):
        compute_narrow_dict_idx(conf, min_files=5)


def test_compute_narrow_dict_idx_nonexistent_dir_raises():
    """Regression test: a dict_root_dirs path that doesn't exist at all
    (as opposed to existing but empty, the case above) used to propagate a
    raw FileNotFoundError/UnboundLocalError from process_root_dirs instead
    of the same NoRealDataFoundError callers already handle -- surfaced by
    tests/dataloaders/test_dataset_speed_real_data.py failing outright
    (instead of skipping) when run somewhere without the real Frontier
    mounts. compute_narrow_dict_idx now normalizes both cases to the same
    exception.
    """
    with tempfile.TemporaryDirectory() as tmp:
        nonexistent = os.path.join(tmp, "does-not-exist")

    conf = _base_conf({"ct1": nonexistent})
    with pytest.raises(NoRealDataFoundError):
        compute_narrow_dict_idx(conf, min_files=5)


def test_compute_narrow_dict_idx_non_iterative_dataloader_is_noop():
    conf = {"dataloader": {"type": "dataloader"}}
    assert compute_narrow_dict_idx(conf, min_files=5) is None


# ---------------------------------------------------------------------------
# make_smoke_config's parallelism.simple_ddp_size:"auto" resolution
# ---------------------------------------------------------------------------


def _write_base_config(path, dict_root_dirs, simple_ddp_size, fsdp_size=1, tensor_par_size=1, batch_size=2):
    conf = {
        "trainer": {"checkpoint_path": "/unused", "save_frequency": 1, "max_epochs": 1, "resume_from_checkpoint": False},
        "parallelism": {"fsdp_size": fsdp_size, "simple_ddp_size": simple_ddp_size, "tensor_par_size": tensor_par_size},
        "model": {"type": "VIT"},
        "dataloader": {"type": "iterative_dataloader", "batch_size": batch_size},
        "data": {"dataset": "basic_ct", "dict_root_dirs": dict_root_dirs},
    }
    with open(path, "w") as f:
        yaml.dump(conf, f)


def test_make_smoke_config_resolves_auto_simple_ddp_size(tmp_path):
    """Regression test: a real Frontier crash (job 5500491, sst-mae) --
    conf["parallelism"]["simple_ddp_size"]:"auto" (a raw, un-parsed YAML
    dict; parse_config, which would normally resolve "auto", never runs
    here) used to reach both the min_files cap's own arithmetic and
    compute_narrow_dict_idx's identical, independent read of the same key
    still as the literal string "auto", crashing with
    "TypeError: '<' not supported between instances of 'str' and 'int'"
    the first time, and would have hit the exact same crash a second time
    (inside compute_narrow_dict_idx) had the first one been patched without
    also fixing this. Resolved once, in place on conf, before either reads
    it.
    """
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    for i in range(20):
        (images_dir / f"image{i}.nii").write_text("")

    base_config = tmp_path / "base_config.yaml"
    _write_base_config(base_config, {"ct1": str(tmp_path)}, simple_ddp_size="auto", fsdp_size=2, tensor_par_size=1, batch_size=4)

    scratch_dir = tmp_path / "scratch"
    smoke_config_path = make_smoke_config(str(base_config), str(scratch_dir), min_files=1000, ntasks=8)

    with open(smoke_config_path) as f:
        written = yaml.load(f, Loader=yaml.FullLoader)

    # ntasks(8) // (fsdp_size(2) * tensor_par_size(1)) == 4, the same
    # resolution parse.py itself does from a live process group's
    # world_size -- baked in as a real int, not left as "auto".
    assert written["parallelism"]["simple_ddp_size"] == 4


def test_make_smoke_config_leaves_explicit_int_simple_ddp_size_untouched(tmp_path):
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    for i in range(20):
        (images_dir / f"image{i}.nii").write_text("")

    base_config = tmp_path / "base_config.yaml"
    _write_base_config(base_config, {"ct1": str(tmp_path)}, simple_ddp_size=8)

    scratch_dir = tmp_path / "scratch"
    smoke_config_path = make_smoke_config(str(base_config), str(scratch_dir), min_files=1000, ntasks=8)

    with open(smoke_config_path) as f:
        written = yaml.load(f, Loader=yaml.FullLoader)

    assert written["parallelism"]["simple_ddp_size"] == 8


# ---------------------------------------------------------------------------
# inflate_min_files_for_train_split
# ---------------------------------------------------------------------------


def test_inflate_min_files_for_train_split_default_ratios():
    # No val_split_ratio/test_split_ratio given -> parse.py's own defaults
    # (0.1/0.1 each) -> 80% train share -> scale by 1/0.8.
    conf = {"dataloader": {}}
    assert inflate_min_files_for_train_split(conf, 32) == math.ceil(32 / 0.8)


def test_inflate_min_files_for_train_split_explicit_ratios():
    conf = {"dataloader": {"val_split_ratio": 0.2, "test_split_ratio": 0.3}}
    assert inflate_min_files_for_train_split(conf, 10) == math.ceil(10 / 0.5)


def test_inflate_min_files_for_train_split_zero_ratios_is_noop():
    conf = {"dataloader": {"val_split_ratio": 0.0, "test_split_ratio": 0.0}}
    assert inflate_min_files_for_train_split(conf, 32) == 32


# ---------------------------------------------------------------------------
# deep_merge_config_overrides
# ---------------------------------------------------------------------------


def test_deep_merge_config_overrides_nested_key():
    conf = {"ap": {"do_ap": False, "fixed_length": 196}}
    result = deep_merge_config_overrides(conf, {"ap": {"do_ap": True}})

    assert result is conf  # returns conf for chaining, doesn't copy
    assert conf["ap"]["do_ap"] is True
    assert conf["ap"]["fixed_length"] == 196  # untouched


def test_deep_merge_config_overrides_replaces_non_dict_wholesale():
    conf = {"tiling": {"tile_overlap": [1, 2, 3]}}
    deep_merge_config_overrides(conf, {"tiling": {"tile_overlap": [0, 0]}})

    assert conf["tiling"]["tile_overlap"] == [0, 0]  # replaced, not merged/extended


def test_deep_merge_config_overrides_adds_new_key():
    conf = {"parallelism": {"tensor_par_size": 1}}
    deep_merge_config_overrides(conf, {"parallelism": {"fsdp_size": 1, "simple_ddp_size": 4}})

    assert conf["parallelism"] == {"tensor_par_size": 1, "fsdp_size": 1, "simple_ddp_size": 4}


def test_deep_merge_config_overrides_multiple_sections_no_cross_talk():
    conf = {"tiling": {"do_tiling": False}, "data": {"twoD": False}}
    deep_merge_config_overrides(conf, {"tiling": {"do_tiling": True}, "data": {"twoD": True}})

    assert conf == {"tiling": {"do_tiling": True}, "data": {"twoD": True}}
