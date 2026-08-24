"""Sanity checks for run_feature_matrix_smoke.py's FEATURE_MATRIX itself.

Pure, no-GPU, no-real-data checks -- catchable in seconds without ever
touching Frontier. run_feature_matrix_smoke.py itself isn't a pytest file,
same reason as run_training_smoke.py (see that file's module docstring).

Includes a real-code sanity check, not just structural checks on the matrix:
test_feature_matrix_cells_parse builds each cell's tiny-model config exactly
the way make_smoke_config would (real base config + TINY_MODEL_OVERRIDES +
the cell's own overrides) and runs it through UCF_VIT.parse.parse_config
with load_balance_offline=True (the same flag utils/validate_config.py
uses, which skips the data_par_size*tensor_par_size==world_size assertion --
so even the tensor_par_size:2 cells validate with zero real GPUs). This is
the same kind of check that would have caught, entirely locally, the
tiling.tile_overlap-as-a-list bug documented in
deep_merge_config_overrides's docstring, an invalid ap.fixed_length, a
tile_size not divisible by patch_size, or a non-whole sqrt_len/cube-root for
UNETR/SAP -- all before ever spending real Frontier GPU allocation time.
"""

import argparse
import os
import tempfile

import pytest
import yaml

from run_feature_matrix_smoke import FEATURE_MATRIX, resolve_base_config
from run_training_smoke import TINY_MODEL_OVERRIDES, deep_merge_config_overrides

from UCF_VIT.parse import parse_config


def test_feature_matrix_labels_are_unique():
    labels = [cell.label for cell in FEATURE_MATRIX]
    assert len(labels) == len(set(labels))


def test_feature_matrix_base_configs_exist():
    for cell in FEATURE_MATRIX:
        resolve_base_config(cell.base_config_relpath)  # raises FileNotFoundError if missing


def test_feature_matrix_overrides_are_nonempty_dicts():
    for cell in FEATURE_MATRIX:
        assert isinstance(cell.overrides, dict) and cell.overrides


def test_feature_matrix_tile_overlap_overrides_are_not_lists():
    """Regression test for the deep_merge_config_overrides tile_overlap-as-a-
    list gotcha (see its docstring): every cell touching
    tiling.tile_overlap must use a bare int, not a list/tuple, or
    parse_config silently mishandles it.
    """
    for cell in FEATURE_MATRIX:
        tile_overlap = cell.overrides.get("tiling", {}).get("tile_overlap")
        if tile_overlap is not None:
            assert isinstance(tile_overlap, int), (
                f"{cell.label}: tiling.tile_overlap override must be a bare int, "
                f"got {type(tile_overlap).__name__}: {tile_overlap!r}"
            )


def test_feature_matrix_min_files_override_type():
    for cell in FEATURE_MATRIX:
        if cell.min_files_override is not None:
            assert isinstance(cell.min_files_override, int) and cell.min_files_override > 0


@pytest.mark.parametrize("cell", FEATURE_MATRIX, ids=lambda c: c.label)
def test_feature_matrix_cells_parse(cell):
    """Builds each cell's tiny-model config the same way make_smoke_config
    would (minus real-data narrowing, which needs real Frontier mounts) and
    confirms it survives parse_config -- no invalid ap.fixed_length,
    non-power-of-two tile_size, non-divisible patch_size, or a reintroduced
    tile_overlap-list bug.
    """
    base_config = resolve_base_config(cell.base_config_relpath)
    with open(base_config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    conf["model"].update(TINY_MODEL_OVERRIDES)
    deep_merge_config_overrides(conf, cell.overrides)

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        parse_config(args, load_balance_offline=True)
    finally:
        os.remove(path)
