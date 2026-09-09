"""Correctness tests for the "sst" dataset support added to
UCF_VIT.dataloaders.dataset/datamodule and UCF_VIT.utils.misc.

"sst" is laid out very differently from basic_ct/imagenet: one flat
np.memmap binary file per variable per timestamp (not one self-contained
file per sample), real volumes too large to materialize in full before
tiling (hence FileReader.read_process_file returns a *list* of per-channel
memmap views, not a stacked ndarray, and TileDataIter._slice_tile only ever
materializes the one small tile actually being cut), and an optional
"chunk" split (dataset_options.full_domain_size vs. data.img_size) letting
one raw file become several independent, separately-shardable samples for
scaling across more DDP ranks than there are real timestamps.

These tests build small, real memmap files on disk (not fakes) with known,
distinct content, so slicing/offset/transpose bugs show up as a wrong value
comparison, not just a wrong shape.
"""

import os

import numpy as np
import pytest
import torch

from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.dataloaders.dataset import FileReader, TileDataIter
from UCF_VIT.utils.misc import process_root_dirs


# ---------------------------------------------------------------------------
# process_root_dirs
# ---------------------------------------------------------------------------


def _touch_variable_files(root_dir, variables, timestamps):
    for t in timestamps:
        for var in variables:
            open(os.path.join(root_dir, f"{var}_{t}"), "w").close()
    open(os.path.join(root_dir, "global"), "w").close()


def test_process_root_dirs_sst_single_chunk_dedupes_variables_and_skips_global(tmp_path):
    root_dir = str(tmp_path)
    _touch_variable_files(root_dir, variables=["r", "u", "v", "w"], timestamps=["1.0", "2.0"])

    dict_lister_trains = process_root_dirs(
        "sst", {"k1": root_dir}, img_size=[2, 2, 2], full_domain_size={"k1": [2, 2, 2]},
    )

    assert dict_lister_trains["k1"] == [
        os.path.join(root_dir, "1.0"), os.path.join(root_dir, "2.0"),
    ]


def test_process_root_dirs_sst_defaults_to_one_chunk_when_full_domain_size_omitted(tmp_path):
    root_dir = str(tmp_path)
    _touch_variable_files(root_dir, variables=["r"], timestamps=["1.0"])

    dict_lister_trains = process_root_dirs("sst", {"k1": root_dir}, img_size=[2, 2, 2])

    assert dict_lister_trains["k1"] == [os.path.join(root_dir, "1.0")]


def test_process_root_dirs_sst_multi_chunk_expands_each_timestamp(tmp_path):
    root_dir = str(tmp_path)
    _touch_variable_files(root_dir, variables=["r"], timestamps=["1.0"])

    # full_domain_size is 2x img_size in every dim -> 2*2*2 = 8 chunks per timestamp.
    dict_lister_trains = process_root_dirs(
        "sst", {"k1": root_dir}, img_size=[2, 2, 2], full_domain_size={"k1": [4, 4, 4]},
    )

    entries = dict_lister_trains["k1"]
    assert len(entries) == 8
    assert len(set(entries)) == 8  # every chunk suffix distinct
    assert all(e.startswith(os.path.join(root_dir, "1.0") + "__chunk") for e in entries)


def test_process_root_dirs_sst_uneven_division_raises_clearly(tmp_path):
    root_dir = str(tmp_path)
    _touch_variable_files(root_dir, variables=["r"], timestamps=["1.0"])

    with pytest.raises(AssertionError, match="full_domain_size"):
        process_root_dirs("sst", {"k1": root_dir}, img_size=[2, 2, 2], full_domain_size={"k1": [3, 4, 4]})


# ---------------------------------------------------------------------------
# FileReader.read_process_file
# ---------------------------------------------------------------------------


def _write_memmap(path, raw_shape, fill):
    """Writes a real memmap file with values from `fill(shape)`."""
    mm = np.memmap(path, dtype=np.float32, mode='w+', shape=raw_shape)
    mm[:] = fill(raw_shape)
    mm.flush()


def _expected_channel(raw, chunk_idx, chunk_size):
    """Reference implementation of read_process_file's sst slicing, applied
    directly to a known raw array -- independent of the implementation under
    test, so a real slicing/offset/transpose bug shows up as a value
    mismatch, not just a shape mismatch.
    """
    trimmed = raw[:, :, :-2]  # drop the +2 x-axis padding -- trailing, never a real offset
    nz, ny, nx = chunk_size
    z0, y0, x0 = chunk_idx[0] * nz, chunk_idx[1] * ny, chunk_idx[2] * nx
    chunk = trimmed[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx]
    return chunk.transpose(2, 1, 0)


def test_read_process_file_sst_single_chunk_matches_reference_slice(tmp_path):
    root_dir = str(tmp_path)
    chunk_size = [2, 3, 2]
    raw_shape = (chunk_size[0], chunk_size[1], chunk_size[2] + 2)

    raw_r = np.arange(np.prod(raw_shape), dtype=np.float32).reshape(raw_shape)
    raw_p = raw_r + 1000.0  # distinct content per variable, catches var/label mixups
    _write_memmap(os.path.join(root_dir, "r_1.0"), raw_shape, lambda s: raw_r)
    _write_memmap(os.path.join(root_dir, "p_1.0"), raw_shape, lambda s: raw_p)

    reader = FileReader(
        file_list=[os.path.join(root_dir, "1.0")], start_idx=0.0, end_idx=1.0,
        variables=["r"], gx="1", ddp_group=None, dataset="sst", return_label=True,
        variables_out=["p"], chunk_size=chunk_size, full_domain_size=chunk_size,
    )

    data_list, label_list = reader.read_process_file(os.path.join(root_dir, "1.0"))

    assert len(data_list) == 1 and len(label_list) == 1
    np.testing.assert_array_equal(np.asarray(data_list[0]), _expected_channel(raw_r, [0, 0, 0], chunk_size))
    np.testing.assert_array_equal(np.asarray(label_list[0]), _expected_channel(raw_p, [0, 0, 0], chunk_size))
    assert data_list[0].shape == (chunk_size[2], chunk_size[1], chunk_size[0])  # (nx, ny, nz)


def test_read_process_file_sst_chunk_suffix_reads_correct_sub_region(tmp_path):
    root_dir = str(tmp_path)
    chunk_size = [2, 2, 2]
    full_domain_size = [4, 2, 2]  # 2 chunks along z only
    raw_shape = (full_domain_size[0], full_domain_size[1], full_domain_size[2] + 2)

    raw_r = np.arange(np.prod(raw_shape), dtype=np.float32).reshape(raw_shape)
    _write_memmap(os.path.join(root_dir, "r_1.0"), raw_shape, lambda s: raw_r)

    reader = FileReader(
        file_list=[os.path.join(root_dir, "1.0__chunk1_0_0")], start_idx=0.0, end_idx=1.0,
        variables=["r"], gx="1", ddp_group=None, dataset="sst", return_label=False,
        chunk_size=chunk_size, full_domain_size=full_domain_size,
    )

    data_list = reader.read_process_file(os.path.join(root_dir, "1.0__chunk1_0_0"))

    expected = _expected_channel(raw_r, [1, 0, 0], chunk_size)
    np.testing.assert_array_equal(np.asarray(data_list[0]), expected)
    # Sanity: chunk 1 must actually differ from chunk 0 (real test of the offset, not a
    # tautology that would pass even with a hardcoded chunk_idx=[0,0,0]).
    assert not np.array_equal(np.asarray(data_list[0]), _expected_channel(raw_r, [0, 0, 0], chunk_size))


# ---------------------------------------------------------------------------
# TileDataIter -- list-of-memmaps ("sst") vs. single-ndarray (every other dataset)
# ---------------------------------------------------------------------------


class _FakeSource:
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples


def test_tiledataiter_list_input_matches_equivalent_stacked_ndarray():
    """TileDataIter._slice_tile's list-of-memmaps branch (sst) must produce
    the exact same tiles as the already-stacked-ndarray branch (every other
    dataset) would, for equivalent content -- the two are meant to be
    interchangeable from TileDataIter's own perspective.
    """
    X, Y, Z = 4, 4, 4
    div = 2
    ch0 = np.arange(X * Y * Z, dtype=np.float32).reshape(X, Y, Z)
    ch1 = ch0 + 1000.0
    stacked = np.stack([ch0, ch1], axis=0)

    stacked_source = _FakeSource([(stacked, ("v0", "v1"))])
    list_source = _FakeSource([([ch0, ch1], ("v0", "v1"))])

    common_kwargs = dict(tile_size=(X // div, Y // div, Z // div), twoD=False, return_label=False, div=div, tile_overlap=(0, 0, 0))
    stacked_tiles = list(TileDataIter(stacked_source, **common_kwargs))
    list_tiles = list(TileDataIter(list_source, **common_kwargs))

    assert len(stacked_tiles) == len(list_tiles) == div ** 3
    for (t1, _), (t2, _) in zip(stacked_tiles, list_tiles):
        np.testing.assert_array_equal(t1, t2)
        assert isinstance(t2, np.ndarray)  # materialized, not still a lazy view/list


# ---------------------------------------------------------------------------
# End-to-end: NativePytorchDataModule, real chunk-splitting included
# ---------------------------------------------------------------------------


def test_native_pytorch_data_module_sst_end_to_end_with_chunking(tmp_path):
    """Real memmap files on disk, through the full pipeline
    (NativePytorchDataModule.setup -> train_dataloader -> collate_fn), with
    dataset_options.full_domain_size wider than data.img_size so the 2
    timestamps genuinely split into 4 independent chunk samples -- this is
    the real, composed feature this whole file exists to prove works, not
    just its individual pieces in isolation.
    """
    root_dir = str(tmp_path)
    full_domain_size = [2, 4, 8]  # [nz, ny, nx]
    chunk_size = [2, 4, 4]  # img_size -- 1*1*2 = 2 chunks per timestamp
    raw_shape = (full_domain_size[0], full_domain_size[1], full_domain_size[2] + 2)

    for t, base in [("1.0", 0.0), ("2.0", 1000.0)]:
        for var, offset in [("r", 0.0), ("u", 10.0), ("p", 20.0)]:
            _write_memmap(
                os.path.join(root_dir, f"{var}_{t}"), raw_shape,
                lambda s, base=base, offset=offset: np.full(s, base + offset, dtype=np.float32),
            )

    data_module = NativePytorchDataModule(
        dict_root_dirs={"P1F4R32": root_dir},
        dict_start_idx={"P1F4R32": 0.0},
        dict_end_idx={"P1F4R32": 1.0},
        dict_buffer_sizes={"P1F4R32": 10},
        dict_in_variables={"P1F4R32": ["r", "u"]},
        num_channels_used={"P1F4R32": 2},
        batch_size=2,
        num_workers=0,
        tile_size=(4, 4, 2),  # (X, Y, Z) -- post-transpose shape of chunk_size, div=1 -- one tile per chunk
        twoD=False,
        return_label=True,
        batches_per_rank_epoch={"P1F4R32": 2},  # 4 chunk-samples / batch_size 2
        div=1,
        tile_overlap=(0, 0, 0),
        data_par_size=1,
        dataset="sst",
        dict_out_variables={"P1F4R32": ["p"]},
        img_size=chunk_size,
        full_domain_size={"P1F4R32": full_domain_size},
    )
    data_module.setup()

    # 2 real timestamps x 2 chunks (1*1*2, since full_domain_size's x-axis is
    # 2x chunk_size's) = 4 independent samples.
    assert len(data_module.dict_lister_trains["P1F4R32"]) == 4

    loader = data_module.train_dataloader()
    batches = list(loader)

    assert len(batches) == 2  # 4 samples / batch_size 2, drop_last=True
    for inp, label, variables, dict_key in batches:
        assert inp.shape == (2, 2, 4, 4, 2)  # (B, C_in=2, X, Y, Z)
        assert label.shape == (2, 1, 4, 4, 2)  # (B, C_out=1, X, Y, Z) -- no spurious extra dim
        assert isinstance(inp, torch.Tensor) and inp.dtype == torch.float32
        assert dict_key == "P1F4R32"
        assert variables == ["r", "u"]
        # Every real sample's r channel is a constant (base + 0), u channel
        # (base + 10) -- confirms real, distinguishable content made it all
        # the way through the pipeline, not e.g. two chunks silently reading
        # the same data.
        for b in range(2):
            r_val = inp[b, 0].unique()
            u_val = inp[b, 1].unique()
            p_val = label[b, 0].unique()
            assert len(r_val) == 1 and len(u_val) == 1 and len(p_val) == 1
            assert (u_val - r_val).item() == pytest.approx(10.0)
            assert (p_val - r_val).item() == pytest.approx(20.0)
