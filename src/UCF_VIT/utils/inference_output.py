import os

import numpy as np
import nibabel as nib

from UCF_VIT.utils.normalize import zscore_denormalize


def save_inference_batch(output_dir, batch, output, batch_idx, rank, regression, conf=None):
    """Saves one UNETR batch's input/label/prediction volumes as NIfTI files.

    Intended for visual inspection (e.g. in 3D Slicer/ITK-SNAP) of a UNETR
    checkpoint's real predictions, not for any downstream computation --
    written with an identity affine (the data pipeline itself discards each
    file's original NIfTI affine on load, keeping only the voxel array, so
    there's no real affine available to preserve here).

    Handles both of UNETR's two tasks (see `training.forward_step`'s own
    `model.loss_fn` dispatch): classification/segmentation (discrete
    `DiceCELoss`) and regression (`loss_fn:"MSE"`, e.g. the "sst" pred task).
    Which one is the caller's own responsibility (`regression`), not
    inferred from `output.shape[1]` -- a real regression task can have more
    than 1 output channel (e.g. predicting multiple "sst" variables at once,
    not just pressure), which `output.shape[1] == 1` would have wrongly
    treated as classification and argmaxed. Classification argmaxes to a
    discrete class-index volume (as before); regression saves the raw
    continuous prediction directly, with no `_label` filename suffix (that
    suffix exists specifically to make viewers like Slicer auto-load a
    *discrete* Labelmap -- exactly wrong for a continuous field, which
    should render as an ordinary Scalar Volume).

    Training loss is computed entirely in normalized space (see `UCF_VIT.
    utils.normalize`'s own module docstring), so `batch["data"]` (always)
    and `output`/`batch["label"]` (only for `regression`, since
    classification labels/predictions are discrete class indices, never
    normalized in the first place) arrive here still normalized --
    denormalized (via `conf["data"]["normalize_stats"]`) right before
    writing, so the saved NIfTI files are in real physical units, not
    z-scores. `conf=None` (e.g. no caller has it handy, or normalization
    isn't configured for this run) skips denormalization entirely rather
    than raising -- the saved files are then in whatever space `batch`/
    `output` already are.

    Args:
        output_dir: Directory to write into; created if it doesn't exist.
        batch: Dict as returned by `training.process_batch` -- uses
            `batch["data"]` (raw input volume, shape (B, C, H, W[, D])),
            `batch["label"]` (ground-truth class-index labels for
            classification, or the real continuous target for regression;
            shape (B, 1, H, W[, D]) either way), and (when `regression` is
            True) `batch["dict_key"]` (which dataset key this batch came
            from, to resolve the right variable/stats). Only `batch["data"]
            [:, 0]`/`batch["label"][:, 0]` (the first channel) is saved --
            for multi-channel regression input/output (e.g. "sst"'s
            r/u/v/w/p), only the first channel is currently dumped.
        output: Model's raw output, shape (B, num_classes, H, W[, D]) --
            per-class logits for classification, or the raw continuous
            prediction for regression.
        batch_idx: This batch's index within the current rank's local
            iteration (`eval_epoch`'s own `counter`), used in filenames.
        rank: This process's global rank (`dist.get_rank()`), used in
            filenames so concurrent ranks writing to the same `output_dir`
            don't collide -- each rank only ever saves its own local shard
            of batches, never another rank's.
        regression: Whether `output`/`batch["label"]` are a continuous
            regression target (True, e.g. `conf["model"]["loss_fn"] ==
            "MSE"`) or discrete class labels (False).
        conf: The full parsed config, used to resolve `dict_in_variables`/
            `dict_out_variables`/`normalize_stats` for `batch["dict_key"]`
            and denormalize before writing. `None` skips denormalization.
    """
    os.makedirs(output_dir, exist_ok=True)

    dict_key = batch["dict_key"] if conf else None
    stats = conf["data"]["normalize_stats"].get(dict_key, {}) if conf else {}
    in_var = [conf["data"]["dict_in_variables"][dict_key][0]] if conf else None

    # batch["data"] is z-score normalized at load time regardless of task
    # (UCF_VIT.utils.normalize's own module docstring) -- denormalized here
    # either way, not just for regression, so the saved "_input" NIfTI is in
    # real physical units.
    data = batch["data"][:, 0].cpu().numpy().astype(np.float32)  # (B, H, W[, D])
    if conf:
        data = zscore_denormalize(data[:, None], in_var, stats, channel_axis=1)[:, 0]

    if regression:
        pred = output[:, 0].cpu().numpy().astype(np.float32)  # (B, H, W[, D])
        label = batch["label"][:, 0].cpu().numpy().astype(np.float32)  # (B, H, W[, D])

        if conf:
            # dict_out_variables can be unset for a dataset with no distinct
            # output-variable concept (see parse.py's own docstring comment
            # on it) -- falls back to the input variable's own stats, same
            # variable either way for a task like MAE reconstruction.
            out_vars = conf["data"]["dict_out_variables"]
            out_var = [out_vars[dict_key][0]] if out_vars and dict_key in out_vars else in_var
            pred = zscore_denormalize(pred[:, None], out_var, stats, channel_axis=1)[:, 0]
            label = zscore_denormalize(label[:, None], out_var, stats, channel_axis=1)[:, 0]
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
