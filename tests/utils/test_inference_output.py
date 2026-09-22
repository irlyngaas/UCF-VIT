import os

import nibabel as nib
import numpy as np
import torch

from UCF_VIT.utils.inference_output import save_inference_batch


def test_save_inference_batch_writes_expected_files(tmp_path):
    batch = {
        "data": torch.rand(2, 1, 4, 4, 4),
        "label": torch.randint(0, 4, (2, 1, 4, 4, 4)),
    }
    output = torch.randn(2, 4, 4, 4, 4)  # (B, num_classes, H, W, D) logits

    save_inference_batch(str(tmp_path), batch, output, batch_idx=3, rank=1, regression=False)

    for sample in (0, 1):
        for suffix in ("input", "label", "pred_label"):
            assert os.path.exists(tmp_path / f"rank1_batch3_sample{sample}_{suffix}.nii.gz")


def test_save_inference_batch_pred_matches_argmax(tmp_path):
    batch = {
        "data": torch.zeros(1, 1, 2, 2, 2),
        "label": torch.zeros(1, 1, 2, 2, 2, dtype=torch.long),
    }
    output = torch.zeros(1, 3, 2, 2, 2)
    output[:, 2] = 10.0  # class 2 should win argmax everywhere

    save_inference_batch(str(tmp_path), batch, output, batch_idx=0, rank=0, regression=False)

    pred = np.array(nib.load(str(tmp_path / "rank0_batch0_sample0_pred_label.nii.gz")).dataobj)
    assert np.all(pred == 2)


def test_save_inference_batch_regression_saves_raw_prediction_not_argmax(tmp_path):
    """model.loss_fn:"MSE" (e.g. "sst"'s pred task) -- pred must be the raw
    continuous output, not an argmaxed (trivially always 0) class index,
    and must NOT get the "_label" filename suffix (that's specifically for
    viewers to auto-load a discrete Labelmap, wrong for a continuous
    field).
    """
    batch = {
        "data": torch.rand(1, 1, 2, 2, 2),
        "label": torch.rand(1, 1, 2, 2, 2),
    }
    output = torch.full((1, 1, 2, 2, 2), 3.5)

    save_inference_batch(str(tmp_path), batch, output, batch_idx=0, rank=0, regression=True)

    assert os.path.exists(tmp_path / "rank0_batch0_sample0_pred.nii.gz")
    assert not os.path.exists(tmp_path / "rank0_batch0_sample0_pred_label.nii.gz")
    pred = np.array(nib.load(str(tmp_path / "rank0_batch0_sample0_pred.nii.gz")).dataobj)
    np.testing.assert_allclose(pred, 3.5, atol=1e-5)


def test_save_inference_batch_multi_channel_regression_not_argmaxed(tmp_path):
    """Regression test for a real bug: dispatching classification-vs-
    regression off `output.shape[1] == 1` wrongly argmaxed a multi-channel
    regression prediction (e.g. "sst" time-stepping's u,v,w,r,p-at-once
    output, num_classes:5) as if it were a 5-class segmentation. `regression`
    is now an explicit caller-supplied flag, so a >1-channel MSE output
    still saves its raw continuous first channel, not an argmaxed index.
    """
    batch = {
        "data": torch.rand(1, 5, 2, 2, 2),
        "label": torch.rand(1, 5, 2, 2, 2),
    }
    output = torch.randn(1, 5, 2, 2, 2)
    output[:, 0] = 7.0  # first channel -- the one save_inference_batch dumps

    save_inference_batch(str(tmp_path), batch, output, batch_idx=0, rank=0, regression=True)

    assert os.path.exists(tmp_path / "rank0_batch0_sample0_pred.nii.gz")
    assert not os.path.exists(tmp_path / "rank0_batch0_sample0_pred_label.nii.gz")
    pred = np.array(nib.load(str(tmp_path / "rank0_batch0_sample0_pred.nii.gz")).dataobj)
    np.testing.assert_allclose(pred, 7.0, atol=1e-5)


def _minimal_conf(dict_key, in_var, stats, out_var=None):
    return {
        "data": {
            "dict_in_variables": {dict_key: [in_var]},
            "dict_out_variables": {dict_key: [out_var]} if out_var else None,
            "normalize_stats": {dict_key: stats},
        },
    }


def test_save_inference_batch_denormalizes_input_for_classification(tmp_path):
    # regression=False (e.g. basic_ct segmentation) -- data is still
    # normalized at load time regardless of task, so it must still be
    # denormalized, even though label/pred (discrete class indices) aren't.
    batch = {
        "data": torch.zeros(1, 1, 2, 2, 2),  # normalized value 0.0
        "label": torch.zeros(1, 1, 2, 2, 2, dtype=torch.long),
        "dict_key": "ct1",
    }
    output = torch.zeros(1, 2, 2, 2, 2)
    conf = _minimal_conf("ct1", "ct_res1", {"ct_res1": {"mean": 100.0, "std": 20.0}})

    save_inference_batch(str(tmp_path), batch, output, batch_idx=0, rank=0, regression=False, conf=conf)

    data = np.array(nib.load(str(tmp_path / "rank0_batch0_sample0_input.nii.gz")).dataobj)
    np.testing.assert_allclose(data, 100.0, atol=1e-4)  # 0.0 * 20 + 100 == 100


def test_save_inference_batch_denormalizes_regression(tmp_path):
    batch = {
        "data": torch.zeros(1, 1, 2, 2, 2),
        "label": torch.full((1, 1, 2, 2, 2), 2.0),
        "dict_key": "sst1",
    }
    output = torch.full((1, 1, 2, 2, 2), 3.0)
    conf = _minimal_conf(
        "sst1", "u", {"u": {"mean": 100.0, "std": 20.0}, "p": {"mean": 5.0, "std": 1.0}}, out_var="p",
    )

    save_inference_batch(str(tmp_path), batch, output, batch_idx=0, rank=0, regression=True, conf=conf)

    data = np.array(nib.load(str(tmp_path / "rank0_batch0_sample0_input.nii.gz")).dataobj)
    label = np.array(nib.load(str(tmp_path / "rank0_batch0_sample0_label.nii.gz")).dataobj)
    pred = np.array(nib.load(str(tmp_path / "rank0_batch0_sample0_pred.nii.gz")).dataobj)
    np.testing.assert_allclose(data, 100.0, atol=1e-4)   # "u" stats: 0*20+100
    np.testing.assert_allclose(label, 7.0, atol=1e-4)    # "p" stats: 2*1+5
    np.testing.assert_allclose(pred, 8.0, atol=1e-4)     # "p" stats: 3*1+5


def test_save_inference_batch_creates_output_dir(tmp_path):
    nested = tmp_path / "nested" / "dir"
    batch = {
        "data": torch.zeros(1, 1, 2, 2, 2),
        "label": torch.zeros(1, 1, 2, 2, 2, dtype=torch.long),
    }
    output = torch.zeros(1, 2, 2, 2, 2)

    save_inference_batch(str(nested), batch, output, batch_idx=0, rank=0, regression=False)

    assert nested.is_dir()
    assert os.path.exists(nested / "rank0_batch0_sample0_input.nii.gz")
