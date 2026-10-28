"""Correlates UNETR adaptive-patching token size with prediction accuracy, per voxel.

Diagnostic for the hypothesis that UNETR's decoder bottleneck (`proj_feat`
reconstructing a fixed `decoder_grid_size**ndims`-cell canvas) destroys fine
detail specifically in regions adaptive patching covered with many small
tokens -- by directly measuring whether accuracy actually degrades as token
size shrinks, rather than just reasoning about it.

Works entirely from already-saved `utils.inference_output.save_inference_
batch` NIfTI triples (`*_input.nii.gz`/`*_label.nii.gz`/`*_pred_label.nii.gz`)
-- no model forward pass needed. For each sample:

1. Rebuilds the real adaptive-patching leaf boxes (`UCF_VIT.dataloaders.
   transform.Patchify_3D`, the same CPU path `train.py`/`test.py` use for
   `do_gpu_ap:False`) directly from the saved `_input.nii.gz` array.
2. Paints a per-voxel "owning leaf size" map from those boxes -- pure
   geometry, no model involved.
3. Bins voxels by that size (powers of two by default) and computes
   `pred_label == label` accuracy within each bucket.

Two things worth knowing before trusting the result:

- `Patchify_3D.serialize` (the octree build itself, given a fixed edge map)
  is deterministic, but `Patchify_3D.forward` randomly draws its Canny
  smoothing `variance` from `sths` every call (unless `canny_sigma` is set
  in the config) -- so this script's reconstructed leaves are a *faithful,
  config-consistent reconstruction* of what adaptive patching would do on
  this input, not a bit-exact replay of the one specific random draw used
  at the actual instant that sample was saved. Pass `--canny-sigma` to pin
  it explicitly (recommended for this script's own run-to-run
  reproducibility); omitted, it defaults to a fixed representative value
  rather than reintroducing randomness into the analysis itself.
- Canny edge detection here independently min-max normalizes each channel
  to [0, 1] before scoring (see `Patchify_3D`'s own docstring) -- that
  erases any difference a prior affine rescaling (e.g. z-score
  normalization vs. the denormalized real-physical-units array
  `save_inference_batch` actually writes) would otherwise cause, so
  feeding this script the saved (denormalized) `_input.nii.gz` directly is
  equivalent to feeding it whatever normalization was actually used at
  training/inference time.

Stratifies per sample, not just pooled across all samples -- a single
catastrophically-failing sample (confirmed to exist in this experiment: see
tests/README.md-adjacent investigation) could otherwise dominate or mask a
real per-sample trend in a pooled histogram.

Usage:
    python utils/token_size_accuracy_correlation.py \\
        ~/adaptive_area_weighted_alpha_1p0_normalize_epoch198 \\
        --config configs/unetr_token_selection_experiment/adaptive_area_weighted_alpha_1p0.yaml
"""

import argparse
import glob
import os
import re

import nibabel as nib
import numpy as np
import yaml

from UCF_VIT.dataloaders.transform import Patchify_3D

DEFAULT_CANNY_SIGMA = 1.0  # midpoint of Patchify_3D's own default sths=[0.5,1.0,2.0]
DEFAULT_BUCKET_EDGES = (2, 4, 8, 16, 32, 64, 128, 257)  # [2,4) [4,8) ... [128,257)


def _find_samples(results_dir):
    """Finds every `*_input.nii.gz`/`*_label.nii.gz`/`*_pred_label.nii.gz` triple under `results_dir`."""
    samples = []
    for input_path in sorted(glob.glob(os.path.join(results_dir, "*_input.nii.gz"))):
        prefix = input_path[: -len("_input.nii.gz")]
        label_path = prefix + "_label.nii.gz"
        pred_path = prefix + "_pred_label.nii.gz"
        if os.path.exists(label_path) and os.path.exists(pred_path):
            samples.append((os.path.basename(prefix), input_path, label_path, pred_path))
    return samples


def build_token_size_map(img_3d, fixed_length, min_size, score_fn, canny_low_threshold,
                          canny_high_threshold, canny_sigma):
    """Rebuilds the real adaptive-patching leaves for `img_3d` and paints a per-voxel size map.

    Args:
        img_3d: Input volume, shape (Z, Y, X) -- a single channel, no
            trailing channel axis (added internally).
        fixed_length: Same meaning as `ap.fixed_length` in the real config.
        min_size: Same meaning as `ap.min_size`.
        score_fn: Same meaning as `ap.score_fn`.
        canny_low_threshold: Same meaning as `ap.canny_low_threshold` (`None`
            resolves to `Patchify_3D`'s own default).
        canny_high_threshold: Same meaning as `ap.canny_high_threshold`.
        canny_sigma: Fixed Canny smoothing variance (see this module's own
            docstring on why this is pinned rather than left random).

    Returns:
        A tuple `(size_map, num_leaves)`: `size_map` is a `(Z, Y, X)` float
        array, each voxel holding the side length of the real leaf that
        covers it (every voxel gets exactly one value -- see this module's
        docstring on why that's geometrically guaranteed); `num_leaves` is
        how many non-padding leaves the tree actually produced.
    """
    patchify = Patchify_3D(
        fixed_length=fixed_length,
        canny_low_threshold=canny_low_threshold,
        canny_high_threshold=canny_high_threshold,
        interp_size=8,  # irrelevant here -- only seq_size/seq_pos are used, never seq_img
        num_channels=1,
        dataset="basic_ct",
        score_fn=score_fn,
        min_size=min_size,
        canny_sigma=canny_sigma,
    )
    img_4d = img_3d[..., None].astype(np.float32)  # (Z, Y, X, 1) -- Patchify_3D's own expected shape
    _, seq_size, seq_pos, _ = patchify.forward(img_4d)

    size_map = np.zeros(img_3d.shape, dtype=np.float32)
    num_leaves = 0
    for size, pos in zip(seq_size, seq_pos):
        if size <= 0:
            continue  # padding leaf (fixed_length exceeded real leaf count) -- never a real sample point
        num_leaves += 1
        x_c, y_c, z_c = pos  # Cube's own (x, y, z) = (axis2, axis1, axis0) convention -- see octree.py
        half = size / 2
        z1, z2 = int(round(z_c - half)), int(round(z_c + half))
        y1, y2 = int(round(y_c - half)), int(round(y_c + half))
        x1, x2 = int(round(x_c - half)), int(round(x_c + half))
        # Clip -- _pad_to_power_of_two is a no-op for basic_ct's 256^3 (already
        # a power of two), but clip defensively in case of a non-power-of-two
        # input elsewhere.
        z1, y1, x1 = max(z1, 0), max(y1, 0), max(x1, 0)
        z2, y2, x2 = min(z2, img_3d.shape[0]), min(y2, img_3d.shape[1]), min(x2, img_3d.shape[2])
        size_map[z1:z2, y1:y2, x1:x2] = size
    return size_map, num_leaves


def bucket_counts(size_map, correct, bucket_edges):
    """Computes per-`[lo, hi)`-bucket `(voxel_count, correct_count)` for one sample.

    Returns raw counts (not an accuracy ratio) specifically so callers can
    accumulate them across samples for a pooled total without ever holding
    more than one sample's full-resolution arrays in memory at once -- a
    real constraint here, not a style preference (`bucket_edges` cardinality
    is tiny; `size_map`/`correct` are each one real 256^3 array).

    Args:
        size_map: Per-voxel leaf size, from `build_token_size_map`.
        correct: Per-voxel `pred == label` boolean array, same shape.
        bucket_edges: Increasing sequence of bucket boundaries.

    Returns:
        List of `(lo, hi, voxel_count, correct_count)` tuples, one per bucket.
    """
    rows = []
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (size_map >= lo) & (size_map < hi)
        count = int(mask.sum())
        correct_count = int(correct[mask].sum()) if count > 0 else 0
        rows.append((lo, hi, count, correct_count))
    return rows


def _print_bucket_table(rows):
    print(f"{'size range':<14}{'voxels':>12}{'accuracy':>12}")
    for lo, hi, count, correct_count in rows:
        acc_str = f"{correct_count / count:.4f}" if count > 0 else "n/a"
        print(f"[{lo:>4},{hi:>4}){count:>12}{acc_str:>12}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="Directory of save_inference_batch NIfTI output (e.g. ap.inference_output.output_dir).")
    parser.add_argument("--config", required=True, help="The real training config path, to read ap.fixed_length/min_size/score_fn/canny thresholds from.")
    parser.add_argument("--canny-sigma", type=float, default=None, help=f"Fixed Canny smoothing variance (default: {DEFAULT_CANNY_SIGMA}, Patchify_3D's own sths midpoint -- see module docstring on why this is pinned).")
    parser.add_argument("--buckets", default=",".join(str(b) for b in DEFAULT_BUCKET_EDGES), help="Comma-separated increasing bucket edges (default: powers of two).")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    assert not conf["data"]["twoD"], "This script is 3D-only (Patchify_3D) -- basic_ct's do_ap path; see Patchify for the 2D analog if ever needed."

    fixed_length = conf["ap"]["fixed_length"]
    min_size = conf["ap"].get("min_size", 2)
    score_fn = conf["ap"].get("score_fn", "canny")
    canny_low_threshold = conf["ap"].get("canny_low_threshold")
    canny_high_threshold = conf["ap"].get("canny_high_threshold")
    canny_sigma = args.canny_sigma if args.canny_sigma is not None else conf["ap"].get("canny_sigma", DEFAULT_CANNY_SIGMA)
    bucket_edges = [int(b) for b in args.buckets.split(",")]

    samples = _find_samples(args.results_dir)
    assert samples, f"No *_input/_label/_pred_label.nii.gz triples found under {args.results_dir}"
    print(f"Found {len(samples)} sample(s). fixed_length={fixed_length} min_size={min_size} score_fn={score_fn!r} canny_sigma={canny_sigma}")

    # Running (voxel_count, correct_count) per bucket, summed across samples
    # -- not a list of every sample's full 256^3 arrays (this environment
    # has ~15GB RAM, most of it already in use elsewhere; holding every
    # sample's arrays until the end OOM-killed this script in practice, at
    # the exact same sample both times it was tried).
    pooled_counts = [0] * (len(bucket_edges) - 1)
    pooled_correct_counts = [0] * (len(bucket_edges) - 1)

    for name, input_path, label_path, pred_path in samples:
        img = np.asarray(nib.load(input_path).dataobj).astype(np.float32)
        label = np.asarray(nib.load(label_path).dataobj)
        pred = np.asarray(nib.load(pred_path).dataobj)
        assert img.shape == label.shape == pred.shape, f"{name}: shape mismatch {img.shape} vs {label.shape} vs {pred.shape}"

        size_map, num_leaves = build_token_size_map(
            img, fixed_length, min_size, score_fn, canny_low_threshold, canny_high_threshold, canny_sigma,
        )
        correct = (pred == label)

        print(f"\n=== {name} ({num_leaves} real leaves, overall accuracy {correct.mean():.4f}) ===")
        rows = bucket_counts(size_map, correct, bucket_edges)
        _print_bucket_table(rows)
        for i, (_, _, count, correct_count) in enumerate(rows):
            pooled_counts[i] += count
            pooled_correct_counts[i] += correct_count
        del img, label, pred, size_map, correct

    pooled_rows = [
        (lo, hi, count, correct_count)
        for (lo, hi), count, correct_count in zip(zip(bucket_edges[:-1], bucket_edges[1:]), pooled_counts, pooled_correct_counts)
    ]
    print(f"\n=== pooled across all {len(samples)} sample(s) -- see module docstring: a single outlier sample can dominate this ===")
    _print_bucket_table(pooled_rows)


if __name__ == "__main__":
    main()
