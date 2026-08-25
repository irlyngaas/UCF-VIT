"""Correctness tests for UCF_VIT.dataloaders.datamodule.collate_fn.

collate_fn is the most branch-heavy, least directly-tested part of the
iterative-dataloader stack (adaptive_patching x return_label x dataset type
x separate_channels x return_qdt), and it's only ever exercised for
"iterative_dataloader" datasets -- basic_ct and imagenet ("dataloader"-type
datasets like catsdogs use a completely separate collate function,
UCF_VIT.datasets.catsdogs.CatsDogsCollate, not this one).

Each test builds its `batch` argument from a *real* ProcessChannels instance
(not hand-fabricated tuples) so the shapes/dtypes collate_fn receives are
exactly what production actually produces -- see tests/dataloaders/
test_dataset.py for ProcessChannels' own correctness coverage.

Writing these tests surfaced four real bugs in ProcessChannels/collate_fn,
all fixed in this session -- three in the return_label=False +
adaptive_patching=True corner specifically, which had drifted out of sync
with its (correct) return_label=True sibling:
  1. ProcessChannels.__iter__ raised UnboundLocalError for
     separate_channels=True: the per-channel patchify loop discarded the
     quadtree object into `_` instead of `qdt`, then referenced the
     never-assigned `qdt` name. See test_dataset.py's
     test_processchannels_separate_channels_does_not_crash.
  2. The same discard-into-`_` mistake, in the separate_channels=False
     sibling, but only triggered when return_qdt=True (unreachable in
     production today -- return_qdt defaults to False and no config or
     caller sets it True). See test_collate_fn_return_qdt_includes_qdt_list.
  3. collate_fn's `seq` computation for dataset="basic_ct" +
     separate_channels=True produced a wrong, spurious extra dimension
     (return_label=True branch didn't check separate_channels before
     applying an expand_dims meant only for the separate_channels=False
     case). See test_collate_fn_separate_channels_true_with_label_basic_ct.
  4. collate_fn's return_label=False branch never added the channel
     dimension that basic_ct's (typically single-channel, un-separated)
     `seq` needs at all -- unlike its return_label=True sibling, which
     handles this correctly. Would produce a 3D tensor where the model's
     `rearrange(x, 'b c s p -> b s (p c)')` needs 4D. See
     test_collate_fn_adaptive_patching_basic_ct_no_label.
None of these are hit by any shipped config today: #1/#3/#4 need
separate_channels: True or (for #4) a basic_ct MAE/DiffusionVIT config with
do_ap: True, and every shipped config uses separate_channels: False; #2
additionally needs return_qdt: True, which nothing sets. Both #1's and #3's
branches even carry "TODO: Finish and Test separate_channels implementation"
comments -- this is exactly that testing.
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset

from UCF_VIT.dataloaders.dataset import ProcessChannels
from UCF_VIT.dataloaders.datamodule import collate_fn


class _FakeSource(IterableDataset):
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples


PATCH_SIZE = 4
FIXED_LENGTH = 16


def _basic_ct_batch(batch_size, num_channels, return_label, num_classes=4, separate_channels=False):
    samples = []
    for i in range(batch_size):
        img = np.random.RandomState(i).uniform(0, 1, size=(num_channels, 32, 32)).astype(np.float32)
        if return_label:
            label = np.random.RandomState(100 + i).randint(0, num_classes, size=(32, 32)).astype(np.int64)
            samples.append((img, label, tuple(f"ct_res{c}" for c in range(num_channels))))
        else:
            samples.append((img, tuple(f"ct_res{c}" for c in range(num_channels))))
    pc = ProcessChannels(
        _FakeSource(samples), num_channels=num_channels, batch_size=batch_size, return_label=return_label,
        adaptive_patching=True, separate_channels=separate_channels, interp_size=PATCH_SIZE,
        fixed_length=FIXED_LENGTH, twoD=True, _dataset="basic_ct", return_qdt=False,
    )
    return list(pc)


def _imagenet_batch(batch_size, return_label):
    samples = []
    for i in range(batch_size):
        img = np.random.RandomState(i).randint(0, 256, size=(3, 32, 32)).astype(np.uint8)
        if return_label:
            samples.append((img, i, ("r", "g", "b")))
        else:
            samples.append((img, ("r", "g", "b")))
    pc = ProcessChannels(
        _FakeSource(samples), num_channels=3, batch_size=batch_size, return_label=return_label,
        adaptive_patching=True, separate_channels=False, interp_size=PATCH_SIZE,
        fixed_length=FIXED_LENGTH, twoD=True, _dataset="imagenet", return_qdt=False,
    )
    return list(pc)


# ---------------------------------------------------------------------------
# adaptive_patching=True, return_label=True
# ---------------------------------------------------------------------------


def test_collate_fn_adaptive_patching_basic_ct_with_label():
    batch_size, num_classes = 3, 4
    batch = _basic_ct_batch(batch_size, num_channels=1, return_label=True, num_classes=num_classes)
    inp, seq, size, pos, label, seq_label, variables, dict_key = collate_fn(
        batch, return_label=True, adaptive_patching=True, separate_channels=False,
        dataset="basic_ct", num_classes=num_classes, num_labels=1, return_qdt=False, dict_key="ct1",
    )
    assert inp.shape == (batch_size, 1, 32, 32)
    assert seq.shape == (batch_size, 1, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
    assert size.shape == (batch_size, 1, FIXED_LENGTH)
    assert pos.shape == (batch_size, 1, FIXED_LENGTH, 2)
    assert label.shape == (batch_size, 1, 32, 32)
    assert seq_label.shape == (batch_size, num_classes, PATCH_SIZE * PATCH_SIZE, FIXED_LENGTH)
    # seq_label is one-hot over the class axis (dim=1) -- every position sums to exactly 1
    torch.testing.assert_close(seq_label.sum(dim=1), torch.ones(batch_size, PATCH_SIZE * PATCH_SIZE, FIXED_LENGTH))
    assert variables == ("ct_res0",)
    assert dict_key == "ct1"
    # every sample's full-image label is preserved through stacking, in order
    for i in range(batch_size):
        np.testing.assert_array_equal(label[i, 0].numpy(), batch[i][4][0])


def test_collate_fn_adaptive_patching_imagenet_with_label():
    batch_size = 3
    batch = _imagenet_batch(batch_size, return_label=True)
    inp, seq, size, pos, label, variables, dict_key = collate_fn(
        batch, return_label=True, adaptive_patching=True, separate_channels=False,
        dataset="imagenet", num_classes=0, num_labels=1, return_qdt=False, dict_key="imagenet",
    )
    assert inp.shape == (batch_size, 3, 32, 32)
    assert seq.shape == (batch_size, 3, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
    assert size.shape == (batch_size, 1, FIXED_LENGTH)
    assert pos.shape == (batch_size, 1, FIXED_LENGTH, 2)
    # class indices preserved through stacking -- ProcessChannels drains its
    # internal batch buffer LIFO (list.pop()), so order is reversed relative
    # to input, not identity (see test_dataset.py's
    # test_processchannels_no_ap_no_label_passes_tiles_through)
    assert label.tolist() == list(reversed(range(batch_size)))
    assert variables == ("r", "g", "b")
    assert dict_key == "imagenet"


# ---------------------------------------------------------------------------
# adaptive_patching=True, return_label=False
# ---------------------------------------------------------------------------


def test_collate_fn_adaptive_patching_basic_ct_no_label():
    batch_size = 2
    batch = _basic_ct_batch(batch_size, num_channels=1, return_label=False)
    inp, seq, size, pos, variables, dict_key = collate_fn(
        batch, return_label=False, adaptive_patching=True, separate_channels=False,
        dataset="basic_ct", num_classes=0, num_labels=1, return_qdt=False, dict_key="ct1",
    )
    assert inp.shape == (batch_size, 1, 32, 32)
    assert seq.shape == (batch_size, 1, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
    assert size.shape == (batch_size, 1, FIXED_LENGTH)
    assert pos.shape == (batch_size, 1, FIXED_LENGTH, 2)
    assert dict_key == "ct1"


def test_collate_fn_return_qdt_includes_qdt_list():
    """Also a regression test for bug #2 in the module docstring:
    separate_channels=False + return_qdt=True used to raise UnboundLocalError
    in ProcessChannels.__iter__ (discarded the quadtree object into `_`, then
    referenced the never-assigned `qdt` name).
    """
    batch_size = 2
    samples = [
        (np.random.RandomState(i).uniform(0, 1, size=(1, 32, 32)).astype(np.float32), ("ct_res1",))
        for i in range(batch_size)
    ]
    pc = ProcessChannels(
        _FakeSource(samples), num_channels=1, batch_size=batch_size, return_label=False,
        adaptive_patching=True, separate_channels=False, interp_size=PATCH_SIZE,
        fixed_length=FIXED_LENGTH, twoD=True, _dataset="basic_ct", return_qdt=True,
    )
    batch = list(pc)
    inp, seq, size, pos, variables, qdt_list, dict_key = collate_fn(
        batch, return_label=False, adaptive_patching=True, separate_channels=False,
        dataset="basic_ct", num_classes=0, num_labels=1, return_qdt=True, dict_key="ct1",
    )
    assert len(qdt_list) == batch_size


# ---------------------------------------------------------------------------
# separate_channels=True -- see module docstring for the two bugs this
# surfaced and fixed
# ---------------------------------------------------------------------------


def test_collate_fn_separate_channels_true_no_label():
    batch_size, num_channels = 2, 3
    batch = _basic_ct_batch(batch_size, num_channels=num_channels, return_label=False, separate_channels=True)
    inp, seq, size, pos, variables, dict_key = collate_fn(
        batch, return_label=False, adaptive_patching=True, separate_channels=True,
        dataset="basic_ct", num_classes=0, num_labels=1, return_qdt=False, dict_key="ct1",
    )
    assert inp.shape == (batch_size, num_channels, 32, 32)
    assert seq.shape == (batch_size, num_channels, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
    assert size.shape == (batch_size, num_channels, FIXED_LENGTH)
    assert pos.shape == (batch_size, num_channels, FIXED_LENGTH, 2)


def test_collate_fn_separate_channels_true_with_label_basic_ct():
    """Regression test for the seq expand_dims bug (see module docstring):
    seq must come out 4D (B, num_channels, fixed_length, interp_size**2), not
    5D with a spurious extra axis.
    """
    batch_size, num_channels, num_classes = 2, 2, 3
    batch = _basic_ct_batch(
        batch_size, num_channels=num_channels, return_label=True,
        num_classes=num_classes, separate_channels=True,
    )
    inp, seq, size, pos, label, seq_label, variables, dict_key = collate_fn(
        batch, return_label=True, adaptive_patching=True, separate_channels=True,
        dataset="basic_ct", num_classes=num_classes, num_labels=1, return_qdt=False, dict_key="ct1",
    )
    assert seq.shape == (batch_size, num_channels, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
    assert size.shape == (batch_size, num_channels, FIXED_LENGTH)
    assert pos.shape == (batch_size, num_channels, FIXED_LENGTH, 2)


# ---------------------------------------------------------------------------
# adaptive_patching=False
# ---------------------------------------------------------------------------


def test_collate_fn_non_adaptive_with_label_basic_ct():
    batch_size = 3
    samples = [
        (
            np.full((1, 8, 8), float(i), dtype=np.float32),
            np.full((8, 8), i + 10, dtype=np.int64),
            ("ct_res1",),
        )
        for i in range(batch_size)
    ]
    pc = ProcessChannels(
        _FakeSource(samples), num_channels=1, batch_size=batch_size, return_label=True,
        adaptive_patching=False, separate_channels=False, interp_size=PATCH_SIZE,
        fixed_length=FIXED_LENGTH, twoD=True, _dataset="basic_ct", return_qdt=False,
    )
    batch = list(pc)
    inp, label, variables, dict_key = collate_fn(
        batch, return_label=True, adaptive_patching=False, separate_channels=False,
        dataset="basic_ct", num_classes=0, num_labels=1, return_qdt=False, dict_key="ct1",
    )
    assert inp.shape == (batch_size, 1, 8, 8)
    assert label.shape == (batch_size, 1, 8, 8)
    # ProcessChannels drains its internal batch buffer LIFO (list.pop()), so
    # sample i (built with value i) ends up at collated index
    # batch_size-1-i, not index i
    for i in range(batch_size):
        j = batch_size - 1 - i
        assert inp[j, 0, 0, 0].item() == float(i)
        assert label[j, 0, 0, 0].item() == i + 10
    assert dict_key == "ct1"


def test_collate_fn_non_adaptive_no_label_imagenet():
    batch_size = 2
    samples = [
        (np.random.RandomState(i).uniform(0, 255, size=(3, 8, 8)).astype(np.float32), ("r", "g", "b"))
        for i in range(batch_size)
    ]
    pc = ProcessChannels(
        _FakeSource(samples), num_channels=3, batch_size=batch_size, return_label=False,
        adaptive_patching=False, separate_channels=False, interp_size=PATCH_SIZE,
        fixed_length=FIXED_LENGTH, twoD=True, _dataset="imagenet", return_qdt=False,
    )
    batch = list(pc)
    inp, variables, dict_key = collate_fn(
        batch, return_label=False, adaptive_patching=False, separate_channels=False,
        dataset="imagenet", num_classes=0, num_labels=1, return_qdt=False, dict_key="imagenet",
    )
    assert inp.shape == (batch_size, 3, 8, 8)
    assert variables == ("r", "g", "b")
    assert dict_key == "imagenet"
