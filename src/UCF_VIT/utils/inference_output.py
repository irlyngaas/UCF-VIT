import os

import numpy as np
import nibabel as nib


def save_inference_batch(output_dir, batch, output, batch_idx, rank):
    """Saves one UNETR batch's input/label/prediction volumes as NIfTI files.

    Intended for visual inspection (e.g. in 3D Slicer/ITK-SNAP) of a UNETR
    checkpoint's real predictions, not for any downstream computation --
    written with an identity affine (the data pipeline itself discards each
    file's original NIfTI affine on load, keeping only the voxel array, so
    there's no real affine available to preserve here).

    Handles both of UNETR's two tasks (see `training.forward_step`'s own
    `model.loss_fn` dispatch): classification/segmentation (`num_classes >=
    2`, discrete `DiceCELoss`) and regression (`num_classes == 1`,
    `loss_fn:"MSE"`, e.g. the "sst" pred task) -- `output.shape[1]` alone
    distinguishes them, since a real segmentation task never has exactly 1
    class. Classification argmaxes to a discrete class-index volume (as
    before); regression saves the raw continuous prediction directly, with
    no `_label` filename suffix (that suffix exists specifically to make
    viewers like Slicer auto-load a *discrete* Labelmap -- exactly wrong for
    a continuous field, which should render as an ordinary Scalar Volume).

    Args:
        output_dir: Directory to write into; created if it doesn't exist.
        batch: Dict as returned by `training.process_batch` -- uses
            `batch["data"]` (raw input volume, shape (B, C, H, W[, D])) and
            `batch["label"]` (ground-truth class-index labels for
            classification, or the real continuous target for regression;
            shape (B, 1, H, W[, D]) either way). Only `batch["data"][:, 0]`
            (the first channel) is saved -- every real config this is used
            against today has `num_channels:1` for classification; for
            multi-channel regression inputs (e.g. "sst"'s r/u/v/w), only the
            first input channel is currently dumped.
        output: Model's raw output, shape (B, num_classes, H, W[, D]) --
            per-class logits for classification, or the raw continuous
            prediction (`num_classes == 1`) for regression.
        batch_idx: This batch's index within the current rank's local
            iteration (`eval_epoch`'s own `counter`), used in filenames.
        rank: This process's global rank (`dist.get_rank()`), used in
            filenames so concurrent ranks writing to the same `output_dir`
            don't collide -- each rank only ever saves its own local shard
            of batches, never another rank's.
    """
    os.makedirs(output_dir, exist_ok=True)

    regression = output.shape[1] == 1
    data = batch["data"][:, 0].cpu().numpy().astype(np.float32)  # (B, H, W[, D])
    if regression:
        pred = output[:, 0].cpu().numpy().astype(np.float32)  # (B, H, W[, D])
        label = batch["label"][:, 0].cpu().numpy().astype(np.float32)  # (B, H, W[, D])
    else:
        pred = output.argmax(dim=1).cpu().numpy().astype(np.int16)  # (B, H, W[, D])
        label = batch["label"][:, 0].cpu().numpy().astype(np.int16)  # (B, H, W[, D])

    for i in range(pred.shape[0]):
        prefix = os.path.join(output_dir, f"rank{rank}_batch{batch_idx}_sample{i}")
        _write_nifti(data[i], f"{prefix}_input.nii.gz")
        _write_nifti(label[i], f"{prefix}_label.nii.gz")
        if regression:
            _write_nifti(pred[i], f"{prefix}_pred.nii.gz")
        else:
            # "_label" suffix (not just "_pred") so viewers that auto-detect
            # Scalar Volume vs. Labelmap by filename (e.g. 3D Slicer's "Add
            # Data") treat this the same as the ground truth instead of
            # loading it as a continuous grayscale volume.
            _write_nifti(pred[i], f"{prefix}_pred_label.nii.gz")


def _write_nifti(array, path):
    """Writes a single 3D array to `path` as a NIfTI file, with an identity affine."""
    nib.save(nib.Nifti1Image(array, affine=np.eye(4)), path)
