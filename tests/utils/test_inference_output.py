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

    save_inference_batch(str(tmp_path), batch, output, batch_idx=3, rank=1)

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

    save_inference_batch(str(tmp_path), batch, output, batch_idx=0, rank=0)

    pred = np.array(nib.load(str(tmp_path / "rank0_batch0_sample0_pred_label.nii.gz")).dataobj)
    assert np.all(pred == 2)


def test_save_inference_batch_regression_saves_raw_prediction_not_argmax(tmp_path):
    """num_classes:1 (e.g. "sst"'s pred task, model.loss_fn:"MSE") -- pred
    must be the raw continuous output, not an argmaxed (trivially always 0)
    class index, and must NOT get the "_label" filename suffix (that's
    specifically for viewers to auto-load a discrete Labelmap, wrong for a
    continuous field).
    """
    batch = {
        "data": torch.rand(1, 1, 2, 2, 2),
        "label": torch.rand(1, 1, 2, 2, 2),
    }
    output = torch.full((1, 1, 2, 2, 2), 3.5)

    save_inference_batch(str(tmp_path), batch, output, batch_idx=0, rank=0)

    assert os.path.exists(tmp_path / "rank0_batch0_sample0_pred.nii.gz")
    assert not os.path.exists(tmp_path / "rank0_batch0_sample0_pred_label.nii.gz")
    pred = np.array(nib.load(str(tmp_path / "rank0_batch0_sample0_pred.nii.gz")).dataobj)
    np.testing.assert_allclose(pred, 3.5, atol=1e-5)


def test_save_inference_batch_creates_output_dir(tmp_path):
    nested = tmp_path / "nested" / "dir"
    batch = {
        "data": torch.zeros(1, 1, 2, 2, 2),
        "label": torch.zeros(1, 1, 2, 2, 2, dtype=torch.long),
    }
    output = torch.zeros(1, 2, 2, 2, 2)

    save_inference_batch(str(nested), batch, output, batch_idx=0, rank=0)

    assert nested.is_dir()
    assert os.path.exists(nested / "rank0_batch0_sample0_input.nii.gz")
