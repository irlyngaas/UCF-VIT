"""Correctness tests for utils/compute_normalization_stats.py's core accumulation logic.

`compute_normalization_stats.py` lives under the repo-level `utils/`
directory (a standalone script, like `utils/validate_config.py`/
`utils/visualize_adaptive.py`, not part of the installed `UCF_VIT` package)
-- imported here via a direct `sys.path` insertion, the same way the script
itself locally imports `validate_config`.

Scoped to `_RunningStats`/`_accumulate_array` (pure numpy math, fast, no
real data files needed) rather than the full `compute_stats` entry point --
that one was verified manually end-to-end against a synthetic basic_ct
fixture (confirmed correct mean/std against ground truth, and confirmed the
train split correctly excludes val/test-reserved files), and a real 2-rank
`gloo` distributed run (launched as two real separate processes, `SLURM_
NTASKS=2`/`SLURM_PROCID` set) confirmed to produce the exact same combined
result as the single-process run over all the same data -- but exercising
either the real `NativePytorchDataModule`/dataloader machinery or a real
multi-process `torch.distributed` group here would make this suite slow
and heavy for what's genuinely simple accumulation logic. `_RunningStats.
state()`/`merge_state()` (the pure-Python core of the cross-rank
combination) *are* covered here, directly.
"""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils"))

from compute_normalization_stats import _RunningStats, _accumulate_array, _default_output_path, _update_config_normalize_stats_path


def test_running_stats_matches_numpy_mean_std_single_update():
    values = np.random.RandomState(0).rand(1000).astype(np.float32) * 50 + 10
    stats = _RunningStats()
    stats.update("ct1", "ct_res1", values)

    result = stats.finalize()
    np.testing.assert_allclose(result["ct1"]["ct_res1"]["mean"], values.mean(), rtol=1e-5)
    np.testing.assert_allclose(result["ct1"]["ct_res1"]["std"], values.std(), rtol=1e-5)


def test_running_stats_combines_multiple_updates_correctly():
    rng = np.random.RandomState(0)
    chunks = [rng.rand(200).astype(np.float32) * 100 for _ in range(5)]
    all_values = np.concatenate(chunks)

    stats = _RunningStats()
    for chunk in chunks:
        stats.update("ct1", "ct_res1", chunk)

    result = stats.finalize()
    np.testing.assert_allclose(result["ct1"]["ct_res1"]["mean"], all_values.mean(), rtol=1e-5)
    np.testing.assert_allclose(result["ct1"]["ct_res1"]["std"], all_values.std(), rtol=1e-5)


def test_running_stats_keeps_dataset_keys_and_variables_independent():
    stats = _RunningStats()
    stats.update("ct1", "ct_res1", np.full(10, 5.0))
    stats.update("ct1", "other_var", np.full(10, 50.0))
    stats.update("ct2", "ct_res1", np.full(10, 500.0))

    result = stats.finalize()
    assert set(result.keys()) == {"ct1", "ct2"}
    assert set(result["ct1"].keys()) == {"ct_res1", "other_var"}
    assert result["ct1"]["ct_res1"]["mean"] == 5.0
    assert result["ct1"]["other_var"]["mean"] == 50.0
    assert result["ct2"]["ct_res1"]["mean"] == 500.0


def test_running_stats_constant_channel_gives_zero_std_not_nan():
    stats = _RunningStats()
    stats.update("ct1", "flat", np.full(100, 7.0))

    result = stats.finalize()
    assert result["ct1"]["flat"]["mean"] == 7.0
    assert result["ct1"]["flat"]["std"] == 0.0  # not NaN/negative from float roundoff in sumsq/count - mean**2


def test_accumulate_array_channel_first_ndarray():
    stats = _RunningStats()
    # [B, C, H, W]: channel 0 is a constant 10, channel 1 a constant 20
    array = np.stack([np.full((3, 4, 4), 10.0), np.full((3, 4, 4), 20.0)], axis=1)
    _accumulate_array(stats, "ct1", ("v0", "v1"), array, max_per_channel=None)

    result = stats.finalize()
    assert result["ct1"]["v0"]["mean"] == 10.0
    assert result["ct1"]["v1"]["mean"] == 20.0
    assert stats.sample_counts()[("ct1", "v0")] == 3 * 4 * 4


def test_accumulate_array_list_of_arrays_sst_convention():
    stats = _RunningStats()
    # sst's own convention: list of [B, ...] arrays, one per variable, entries
    # may be (name, offset) tuples.
    array = [np.full((2, 4, 4), 1.0), np.full((2, 4, 4), 2.0)]
    _accumulate_array(stats, "sst1", (("u", -1), ("p", 0)), array, max_per_channel=None)

    result = stats.finalize()
    assert result["sst1"]["u"]["mean"] == 1.0
    assert result["sst1"]["p"]["mean"] == 2.0


def test_accumulate_array_respects_max_per_channel_cap():
    stats = _RunningStats()
    array = np.full((1, 1, 100), 5.0)  # 100 values in one call
    _accumulate_array(stats, "ct1", ("v0",), array, max_per_channel=10)

    # already >= cap after the first call -- a second call must be skipped entirely
    _accumulate_array(stats, "ct1", ("v0",), np.full((1, 1, 100), 999.0), max_per_channel=10)

    assert stats.sample_counts()[("ct1", "v0")] == 100  # not 200 -- second call skipped
    assert stats.finalize()["ct1"]["v0"]["mean"] == 5.0  # unpolluted by the 999.0 call


def test_running_stats_merge_state_matches_single_accumulator():
    """The real cross-rank combination path: merging N ranks' own state()
    must give the exact same result as a single accumulator that saw all
    the same values -- not an approximation, since sum-of-sums/sum-of-
    sums-of-squares/counts are all exact regardless of partitioning.
    """
    rng = np.random.RandomState(0)
    chunks = [rng.rand(200).astype(np.float32) * 100 for _ in range(4)]
    all_values = np.concatenate(chunks)

    single = _RunningStats()
    single.update("ct1", "ct_res1", all_values)

    # Simulate 4 ranks, each seeing only its own chunk, then merged as if
    # gathered via dist.all_gather_object.
    per_rank = []
    for chunk in chunks:
        rank_stats = _RunningStats()
        rank_stats.update("ct1", "ct_res1", chunk)
        per_rank.append(rank_stats.state())

    merged = _RunningStats()
    for state in per_rank:
        merged.merge_state(state)

    expected = single.finalize()
    got = merged.finalize()
    assert got["ct1"]["ct_res1"]["mean"] == pytest.approx(expected["ct1"]["ct_res1"]["mean"], rel=1e-9)
    assert got["ct1"]["ct_res1"]["std"] == pytest.approx(expected["ct1"]["ct_res1"]["std"], rel=1e-9)
    assert merged.sample_counts()[("ct1", "ct_res1")] == single.sample_counts()[("ct1", "ct_res1")]


def test_running_stats_merge_state_combines_disjoint_keys():
    a = _RunningStats()
    a.update("ct1", "v0", np.full(10, 5.0))
    b = _RunningStats()
    b.update("ct2", "v0", np.full(10, 50.0))  # a different dataset key entirely, never seen by a

    merged = _RunningStats()
    merged.merge_state(a.state())
    merged.merge_state(b.state())

    result = merged.finalize()
    assert result["ct1"]["v0"]["mean"] == 5.0
    assert result["ct2"]["v0"]["mean"] == 50.0


_FAKE_CONFIG = """\
# a real user comment above the model section
model:
  type: "SAP"

# ---------------------------- DATA ----------------------------------------------
data:
  dataset: "basic_ct"
  img_size: [256, 256, 256] # a real inline comment
  twoD: False
"""


def _patch_repo_root(monkeypatch, repo_root):
    import compute_normalization_stats
    monkeypatch.setattr(compute_normalization_stats, "find_repo_root", lambda: str(repo_root))


def test_update_config_inserts_new_line_preserving_comments(tmp_path, monkeypatch):
    _patch_repo_root(monkeypatch, tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_FAKE_CONFIG)

    output_path = tmp_path / "stats" / "basic_ct_stats.yaml"
    _update_config_normalize_stats_path(str(config_path), str(output_path))

    updated = config_path.read_text()
    for line in _FAKE_CONFIG.splitlines():
        assert line in updated.splitlines()  # every original line untouched
    assert 'normalize_stats_path: "stats/basic_ct_stats.yaml"' in updated

    parsed = yaml.safe_load(updated)
    assert parsed["data"]["normalize_stats_path"] == "stats/basic_ct_stats.yaml"


def test_update_config_replaces_existing_line_not_duplicated(tmp_path, monkeypatch):
    _patch_repo_root(monkeypatch, tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_FAKE_CONFIG)

    _update_config_normalize_stats_path(str(config_path), str(tmp_path / "stats" / "first.yaml"))
    first_line_count = len(config_path.read_text().splitlines())

    _update_config_normalize_stats_path(str(config_path), str(tmp_path / "stats" / "second.yaml"))
    updated = config_path.read_text()

    assert updated.count("normalize_stats_path") == 1
    assert len(updated.splitlines()) == first_line_count  # replaced in place, no new line added
    assert 'normalize_stats_path: "stats/second.yaml"' in updated
    for line in _FAKE_CONFIG.splitlines():
        assert line in updated.splitlines()


def test_update_config_raises_clearly_with_no_data_section(tmp_path, monkeypatch):
    _patch_repo_root(monkeypatch, tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  type: \"SAP\"\n")

    with pytest.raises(RuntimeError, match="data:"):
        _update_config_normalize_stats_path(str(config_path), str(tmp_path / "stats.yaml"))


def test_default_output_path_mirrors_configs_directory_structure(tmp_path, monkeypatch):
    _patch_repo_root(monkeypatch, tmp_path)
    (tmp_path / "configs" / "basic_ct" / "sap").mkdir(parents=True)
    config_path = tmp_path / "configs" / "basic_ct" / "sap" / "base_config.yaml"
    config_path.write_text(_FAKE_CONFIG)

    result = _default_output_path(str(config_path))

    assert result == str(tmp_path / "stats" / "basic_ct" / "sap" / "base_config.yaml")


def test_default_output_path_disambiguates_same_basename_different_dirs(tmp_path, monkeypatch):
    """Real motivation: this repo ships many "base_config.yaml" files, one
    per dataset/model directory -- a flat "<basename>_stats.yaml" default
    would collide across all of them.
    """
    _patch_repo_root(monkeypatch, tmp_path)
    (tmp_path / "configs" / "basic_ct" / "sap").mkdir(parents=True)
    (tmp_path / "configs" / "basic_ct" / "unetr").mkdir(parents=True)

    sap_path = _default_output_path(str(tmp_path / "configs" / "basic_ct" / "sap" / "base_config.yaml"))
    unetr_path = _default_output_path(str(tmp_path / "configs" / "basic_ct" / "unetr" / "base_config.yaml"))

    assert sap_path != unetr_path
    assert os.path.basename(sap_path) == os.path.basename(unetr_path) == "base_config.yaml"


def test_default_output_path_falls_back_to_flat_basename_outside_configs(tmp_path, monkeypatch):
    _patch_repo_root(monkeypatch, tmp_path)
    config_path = tmp_path / "somewhere_else" / "my_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(_FAKE_CONFIG)

    result = _default_output_path(str(config_path))

    assert result == str(tmp_path / "stats" / "my_config.yaml")
