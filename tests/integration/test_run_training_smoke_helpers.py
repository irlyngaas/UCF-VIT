"""Correctness tests for run_training_smoke.py's real-data-narrowing helpers.

run_training_smoke.py itself isn't a pytest file (no test_ prefix, and it
has its own if __name__ == "__main__" entry point -- see its module
docstring for why it's a plain script, not pytest), but its narrowing
helpers are plain, unit-testable functions, and are reused directly by
tests/distributed/test_dataloader_real_pipeline.py and
tests/dataloaders/test_dataset_speed_real_data.py, not just Tier 3 itself.
"""

import os
import tempfile

import pytest

from run_training_smoke import NoRealDataFoundError, compute_narrow_dict_idx


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
