import os
import time

import numpy as np
import torch
import random
import SimpleITK as sitk
from skimage.feature import canny as skimage_canny
from torch.utils.data import get_worker_info
from .quadtree import FixedQuadTree
from .octree import FixedOctTree
from UCF_VIT.utils.misc import next_power_of_two


def _pad_to_power_of_two(img, ndim):
    """Edge-replication-pads `img`'s leading `ndim` (spatial) axes up to the next power of two.

    `FixedQuadTree`/`FixedOctTree`'s split (an integer midpoint, used for
    both children) only ever produces equal-sized children when repeatedly
    halving stays clean all the way down -- true for a power-of-two size,
    not guaranteed for any other. Padding up to the next power of two per
    axis (independently -- unlike `UCF_VIT.model.gpu_adaptive_patching.
    GPUPatchify2D`/`GPUPatchify3D`, this class's root is always a single
    node covering the whole image, not a fixed grid, so there's no cross-
    axis block-count-ratio constraint to satisfy) removes the restriction
    that `data.tile_size` be a power of two for this (CPU) path -- the same
    edge-replication `F.pad(..., mode="replicate")` `GPUPatchify2D`'s own
    `_pad_tensor` already uses for its GPU-side equivalent, via `numpy`'s
    `mode="edge"` (identical semantics, `img` here is a real `numpy` array,
    not a `torch.Tensor`). A no-op (returns `img` unchanged, not a copy)
    when every axis is already a power of two.

    Args:
        img: Array whose first `ndim` axes are spatial, e.g. `(H, W[, C])`
            or `(D, H, W[, C])`.
        ndim: Number of leading spatial axes (`2` for `Patchify`, `3` for
            `Patchify_3D`).

    Returns:
        The (possibly) padded array, same number of dims as `img`.
    """
    pad_width = [(0, next_power_of_two(img.shape[i]) - img.shape[i]) for i in range(ndim)]
    pad_width += [(0, 0)] * (img.ndim - ndim)
    if not any(p[1] for p in pad_width):
        return img
    return np.pad(img, pad_width, mode="edge")

class Patchify(torch.nn.Module):
    """Adaptive (quadtree-based) patchification transform for 2D images.

    Detects edges with a randomly smoothed Canny filter, builds a `FixedQuadTree`
    over the edge map, and serializes the image into a fixed-length sequence of
    variable-sized patches concentrated around detected edges.

    Uses two different Canny implementations depending on `dataset`:
    `imagenet`/`catsdogs` (real, possibly multi-channel photos) run
    `SimpleITK.CannyEdgeDetection` once per channel and sum the results into
    a per-pixel edge count -- the same "weight a pixel by how many channels
    independently flag it as an edge" convention `Patchify_3D` already uses,
    since `SimpleITK.CannyEdgeDetection` doesn't support multi-channel/vector
    images directly either. This deliberately replaces an earlier `cv2.Canny`
    call: `cv2.Canny`'s own multi-channel handling is a per-pixel "winner
    take all" (each pixel's edge decision comes from whichever single
    channel has the largest gradient magnitude there, discarding the other
    channels' gradients entirely at that pixel) -- a real, coarse
    simplification that doesn't generalize well to channels with different
    physical scales/meanings (fine-ish for RGB's similarly-scaled channels,
    not for arbitrary multi-channel data). Every other dataset
    (arbitrary-range float data, e.g. `basic_ct`) uses `skimage.feature.canny`
    instead, which operates on the real float values directly via quantile
    (not absolute) thresholds, dataset-scale-independent by construction;
    its own tradeoff is accepting only single-channel 2D input.

    Measured directly (not assumed): the per-channel `SimpleITK.
    CannyEdgeDetection` loop is roughly 25x slower per call than the old
    `cv2.Canny` call on a synthetic 256x256x3 image (~15.5ms vs ~0.6ms) --
    real, worth knowing if this ever becomes a dataloader-throughput
    bottleneck (`cv2` stays a real, imported dependency elsewhere in this
    codebase regardless, for `cv.resize`). If that ever becomes a genuine
    problem in practice, `cv2.Canny` remains a faster option worth
    reaching for specifically as a last resort -- its own per-pixel
    "winner take all" multi-channel simplification (see above) is the
    reason it isn't the default here: fine-ish for RGB, not something that
    generalizes correctly to arbitrary multi-channel data with differently
    scaled channels.
    """

    def __init__(self, sths=[0,1,3,5], fixed_length=196, canny_thresholds=(0.05, 0.15), canny_quantiles=(0.7, 0.9), interp_size=16, num_channels=3, dataset="imagenet", return_edges=False, score_fn="canny", min_size=2, canny_sigma=None, canny_low_threshold=None, canny_high_threshold=None) -> None:
        """Initializes the randomization ranges and patch parameters for the transform.

        Args:
            sths: Candidate Gaussian smoothing strengths to randomly choose from
                before edge detection (0 = no smoothing, use uniform random noise
                as the edge map instead). For `imagenet`/`catsdogs`
                (`SimpleITK.CannyEdgeDetection`'s own `variance` parameter,
                replicated across both axes), not numerically equivalent to
                a standard deviation -- same caveat as `Patchify_3D`'s own
                `sths`. For every other dataset (`skimage.feature.canny`'s
                own `sigma`), a real standard deviation (float). Unused when
                `score_fn="variance"` or `canny_sigma` is given (see below).
            fixed_length: Fixed number of patches the image is serialized into.
            canny_thresholds: `imagenet`/`catsdogs` only
                (`SimpleITK.CannyEdgeDetection`): `(low, high)` hysteresis
                thresholds -- absolute values on the (smoothed) gradient-
                magnitude scale, not quantiles, matching `Patchify_3D`'s own
                `canny_thresholds`. Meaningful across any dataset's real
                intensity range because each channel is independently
                min-max normalized to `[0, 1]` before Canny runs (see
                `forward`'s own comment) -- starting values, not empirically
                tuned. Unused when `score_fn="variance"` or `canny_low_
                threshold`/`canny_high_threshold` are given (see below).
            canny_quantiles: Every other dataset only (`skimage.feature.canny`,
                `use_quantiles=True`): `(low, high)` hysteresis thresholds, as
                quantiles of the edge-magnitude distribution in `[0, 1]` --
                dataset-scale-independent by construction, unlike
                `canny_thresholds`' absolute values. Starting values, not
                empirically tuned. Unused when `score_fn="variance"`. Not
                overridable by `canny_low_threshold`/`canny_high_threshold`
                (see below) -- a genuinely different kind of threshold
                (quantile, not absolute), no shared knob makes sense here.
            interp_size: Side length each (square) leaf patch is interpolated to.
            num_channels: Number of image channels.
            dataset: Dataset name; controls how edges are computed/normalized
                ("imagenet"/"catsdogs" vs. other datasets). Unused when
                `score_fn="variance"`.
            return_edges: If True, also return the computed edge map from
                `forward`. When `score_fn="variance"`, the "edge map" is
                `img` itself (see `forward`'s own docstring).
            score_fn: `"canny"` (default -- randomly-smoothed Canny edge
                density, as described above) or `"variance"` (`img` itself
                is used as `FixedQuadTree`'s domain, deterministically, no
                smoothing/thresholds -- see `UCF_VIT.dataloaders.quadtree.
                Rect.contains`'s own docstring for what that scores).
            min_size: Smallest leaf side length `FixedQuadTree` will ever
                produce -- forwarded to its own `min_size`, the same shared
                floor `UCF_VIT.model.gpu_adaptive_patching.GPUPatchify2D`
                uses for its GPU-side equivalent.
            canny_sigma: If given, overrides `sths` with this single fixed
                value (no per-call randomization) -- the same `ap.
                canny_sigma` config knob `UCF_VIT.model.gpu_adaptive_
                patching.GPUPatchify2D`'s own Canny scoring already exposes,
                shared here rather than being GPU-only. For `imagenet`/
                `catsdogs`, converted to `SimpleITK.CannyEdgeDetection`'s own
                `variance` parameter as `canny_sigma ** 2` (ITK's own
                Gaussian variance-vs-sigma convention, confirmed against
                `DiscreteGaussianImageFilter`'s own docs -- `CannyEdge
                DetectionImageFilter` reuses that same filter for its
                smoothing) -- for every other dataset, used directly as
                `skimage.feature.canny`'s own `sigma` (already the same
                meaning as `GPUPatchify2D`'s `canny_sigma`, no conversion
                needed). `None` (default) leaves `sths` -- and its per-call
                randomization -- untouched.
            canny_low_threshold: If given together with `canny_high_
                threshold`, overrides `canny_thresholds` with `(canny_low_
                threshold, canny_high_threshold)` -- the same shared `ap.
                canny_low_threshold` config knob as `GPUPatchify2D`'s own
                Canny scoring. Only affects `imagenet`/`catsdogs`
                (`canny_thresholds`' own consumer) -- `None` (default)
                leaves `canny_thresholds` untouched. Not numerically
                guaranteed equivalent across the GPU and CPU Canny
                implementations even at the same threshold value (different
                gradient-computation algorithms can produce gradient
                magnitudes on different absolute scales for the same real
                edge strength) -- shared as one convenient knob, not a
                claim of identical sensitivity.
            canny_high_threshold: See `canny_low_threshold` -- both must be
                given together to take effect.
        """
        super().__init__()

        self.fixed_length = fixed_length
        self.canny_quantiles = canny_quantiles
        self.interp_size = interp_size
        self.num_channels = num_channels
        self.dataset = dataset
        self.return_edges = return_edges
        self.score_fn = score_fn
        self.min_size = min_size

        if canny_sigma is not None:
            squared = canny_sigma ** 2 if dataset in ("imagenet", "catsdogs") else canny_sigma
            self.sths = [squared]
        else:
            self.sths = sths

        if canny_low_threshold is not None and canny_high_threshold is not None:
            self.canny_thresholds = (canny_low_threshold, canny_high_threshold)
        else:
            self.canny_thresholds = canny_thresholds

    def forward(self, img):  # we assume inputs are always structured like this
        """Computes an edge map (or, in variance mode, uses `img` directly) and adaptively patchifies it via a quadtree.

        Args:
            img: Input 2D image array, shape (H, W[, C]) -- any `H`/`W`, not
                just a power of two (see `_pad_to_power_of_two`; a no-op
                when both already are). For `score_fn="canny"` and any
                `dataset` other than `imagenet`/`catsdogs`, `C` (if present)
                must be 1 -- `skimage.feature.canny` only accepts single-
                channel input (see this class's own docstring).
                `score_fn="variance"` has no such restriction (any `C`, any
                `dataset`).

        Returns:
            If `self.return_edges` is False: `(seq_img, seq_size, seq_pos, qdt)`.
            If True: `(seq_img, seq_size, seq_pos, qdt, edges)`. `seq_img` is the
            flattened patch sequence, `seq_size` the per-patch side length,
            `seq_pos` the per-patch center position, `qdt` the `FixedQuadTree`
            instance, and `edges` the computed edge map (`img` itself, under
            `score_fn="variance"`) -- both already reflect any padding above,
            so a leaf near the padded edge may cover some replicated (not
            real) content, exactly like `GPUPatchify2D`'s own equivalent
            tradeoff.
        """
        img = _pad_to_power_of_two(img, ndim=2)

        if self.score_fn == "variance":
            edges = img
            qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length, score_fn=self.score_fn, min_size=self.min_size)
            return self._serialize(img, qdt, edges)

        # Do some transformations. Here, we're just passing though the input

        self.smooth_factor = random.choice(self.sths)
        if self.smooth_factor == 0:
            if self.dataset == "imagenet" or self.dataset == "catsdogs":
                edges = np.random.uniform(low=0,high=1,size=(img.shape[0],img.shape[1]))
            else:
                edges = np.random.uniform(low=np.min(img),high=np.max(img),size=(img.shape[0],img.shape[1]))
        else:
            if self.dataset == "imagenet" or self.dataset == "catsdogs":
                # One real 2D Canny call per channel (SimpleITK.CannyEdgeDetection
                # doesn't support multi-channel/vector images directly), summed
                # into a per-pixel count of how many channels independently flag
                # it as an edge -- Patchify_3D's own convention, replacing an
                # earlier cv2.Canny call whose own multi-channel handling was a
                # per-pixel "winner take all" across channels instead (see this
                # class's own docstring for why that's a real simplification,
                # not just a style difference).
                variance = [float(self.smooth_factor)] * 2
                edges_combined_counter = np.zeros(img.shape[:2], dtype=np.uint8)
                for j in range(self.num_channels):
                    channel = img[:, :, j].astype(np.float32)
                    # Per-channel min-max normalized to [0,1] before Canny --
                    # keeps canny_thresholds meaningful regardless of this
                    # image's real intensity range (imagenet/catsdogs are
                    # uint8 [0,255], not the ~[0,1] scale canny_thresholds
                    # assumes) -- edge-detection input only, real patch
                    # content (img, fed to self._serialize below) is
                    # untouched.
                    lo, hi = channel.min(), channel.max()
                    if hi > lo:
                        channel = (channel - lo) / (hi - lo)
                    channel_img = sitk.GetImageFromArray(channel)
                    channel_edges = sitk.CannyEdgeDetection(
                        channel_img,
                        lowerThreshold=self.canny_thresholds[0], upperThreshold=self.canny_thresholds[1],
                        variance=variance,
                    )
                    edges_combined_counter += sitk.GetArrayFromImage(channel_edges).astype(np.uint8)
                edges = edges_combined_counter
            else:
                if img.ndim == 3:
                    if img.shape[-1] != 1:
                        raise NotImplementedError(
                            f"Patchify's skimage.feature.canny path (dataset={self.dataset!r}) only "
                            f"supports single-channel input, got {img.shape[-1]} channels -- "
                            "the imagenet/catsdogs path (per-channel SimpleITK.CannyEdgeDetection, "
                            "summed into an edge count) supports any channel count, "
                            "skimage.feature.canny doesn't."
                        )
                    img_2d = img[:, :, 0]
                else:
                    img_2d = img
                edges = skimage_canny(
                    img_2d, sigma=self.smooth_factor,
                    low_threshold=self.canny_quantiles[0], high_threshold=self.canny_quantiles[1],
                    use_quantiles=True,
                )
                # Not rescaled -- Rect.contains (quadtree.py) no longer
                # divides by a fixed 255, it's an unnormalized sum only ever
                # used for relative (max-based) comparison, so any
                # consistent scale works; boolean-as-uint8 is fine as-is.
                edges = edges.astype(np.uint8)

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length, score_fn=self.score_fn, min_size=self.min_size)
        return self._serialize(img, qdt, edges)

    def _serialize(self, img, qdt, edges):
        """Serializes `img` through an already-built `qdt`, and packages the return value.

        Shared tail of `forward`'s canny and variance branches -- everything
        after the domain/tree is decided is identical either way.

        Args:
            img: Input image, shape (H, W[, C]), as passed to `forward`.
            qdt: `FixedQuadTree` already built over this call's domain.
            edges: The domain array used to build `qdt` -- only returned
                when `self.return_edges` is True.

        Returns:
            Same as `forward`'s own return value.
        """
        seq_img, seq_size, seq_pos = qdt.serialize(img, size=(self.interp_size,self.interp_size,self.num_channels))
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img, dtype=np.float32)

        if self.num_channels > 1:
            seq_img = np.moveaxis(seq_img, -1, 0)
            seq_img = np.reshape(seq_img, [self.num_channels, -1, self.interp_size*self.interp_size])
        else:
            seq_img = np.reshape(seq_img, [-1, self.interp_size*self.interp_size])

        seq_pos = np.asarray(seq_pos)
        if self.return_edges:
            return seq_img, seq_size, seq_pos, qdt, edges
        else:
            return seq_img, seq_size, seq_pos, qdt

class Patchify_3D(torch.nn.Module):
    """Adaptive (octree-based) patchification transform for 3D volumes.

    Detects edges per-channel with a genuinely 3D Canny filter
    (`SimpleITK.CannyEdgeDetection`), weights each voxel by how many channels
    independently flag it as an edge (a voxel edge on N channels scores Nx a
    voxel edge on 1 channel), builds a `FixedOctTree` over the result, and
    serializes the volume into a fixed-length sequence of variable-sized
    patches concentrated around detected edges.

    `SimpleITK.CannyEdgeDetection` doesn't support multi-channel/vector images
    directly, so edge detection runs per channel and the results are combined
    by the weighting above rather than in a single multi-channel call. Each
    channel is independently min-max normalized to `[0, 1]` before Canny runs
    (edge-detection input only, real patch content untouched) -- keeps
    `canny_thresholds` meaningful regardless of this volume's real intensity
    range, for any dataset.
    """

    def __init__(self, sths=[0.5,1.0,2.0], fixed_length=196, canny_thresholds=(0.05, 0.15), interp_size=16, num_channels=3, dataset="basic_ct", return_edges=False, profile=False, score_fn="canny", min_size=2, canny_sigma=None, canny_low_threshold=None, canny_high_threshold=None) -> None:
        """Initializes the randomization ranges and patch parameters for the transform.

        Args:
            sths: Candidate Gaussian smoothing strengths to randomly choose
                from before edge detection -- passed directly as
                `SimpleITK.CannyEdgeDetection`'s own `variance` parameter
                (replicated across all 3 axes), which handles smoothing
                internally (no separate blur step needed, unlike the old
                pipeline). Note: `variance`, not standard deviation (sigma)
                -- not numerically equivalent to this parameter's old
                meaning, needs its own tuning regardless. Unused when
                `score_fn="variance"` or `canny_sigma` is given (see below).
            fixed_length: Fixed number of patches the volume is serialized into.
            canny_thresholds: `(low, high)` hysteresis thresholds for
                `SimpleITK.CannyEdgeDetection` -- absolute values on the
                (smoothed) gradient-magnitude scale, not quantiles (unlike
                `Patchify`'s `canny_quantiles`) -- SimpleITK has no
                quantile-threshold option. Meaningful across any dataset's
                real intensity range because each channel is independently
                min-max normalized to `[0, 1]` before Canny runs (see
                `forward`'s own comment) -- starting values, not empirically
                tuned. Unused when `score_fn="variance"` or `canny_low_
                threshold`/`canny_high_threshold` are given (see below).
            interp_size: Side length each (cubic) leaf patch is interpolated to.
            num_channels: Number of volume channels.
            dataset: Dataset name -- kept for interface parity with
                `Patchify`'s own constructor, but no longer read anywhere in
                this class: edge-detection input normalization (see
                `canny_thresholds` above) now runs unconditionally for every
                dataset, not just `"sst"` (its own previous special case).
            return_edges: If True, also return the computed edge volume from
                `forward`. When `score_fn="variance"`, the "edge volume" is
                `img` itself (see `forward`'s own docstring).
            profile: Diagnostic only, off by default -- see `UCF_VIT.training.
                train_epoch`'s own `profile_dataloader` handling, which this
                mirrors from inside the DataLoader worker process (this class
                runs inside `num_workers`'s forked worker(s), not the main
                training process, so `train_epoch`'s own timing can't see
                this call's *raw* cost -- only whether the worker keeps up).
                When True, times the per-channel Canny loop, the `FixedOctTree`
                build, and `serialize` separately and prints them every call.
            score_fn: `"canny"` (default -- per-channel 3D Canny, as
                described above) or `"variance"` (`img` itself is used as
                `FixedOctTree`'s domain, deterministically -- skips the
                per-channel `SimpleITK.CannyEdgeDetection` loop entirely,
                since `Cube.contains`'s own per-channel-variance-sum handles
                multi-channel input at query time instead; see `UCF_VIT.
                dataloaders.octree.Cube.contains`'s own docstring for what
                that scores).
            min_size: Smallest leaf side length `FixedOctTree` will ever
                produce -- forwarded to its own `min_size`, the same shared
                floor `UCF_VIT.model.gpu_adaptive_patching.GPUPatchify3D`
                uses for its GPU-side equivalent.
            canny_sigma: If given, overrides `sths` with this single fixed
                value (no per-call randomization), converted to `SimpleITK.
                CannyEdgeDetection`'s own `variance` parameter as `canny_
                sigma ** 2` (ITK's own Gaussian variance-vs-sigma
                convention) -- the same `ap.canny_sigma` config knob
                `UCF_VIT.model.gpu_adaptive_patching.GPUPatchify3D`'s own
                Canny scoring already exposes, shared here rather than
                being GPU-only. `None` (default) leaves `sths` -- and its
                per-call randomization -- untouched.
            canny_low_threshold: If given together with `canny_high_
                threshold`, overrides `canny_thresholds` with `(canny_low_
                threshold, canny_high_threshold)` -- the same shared `ap.
                canny_low_threshold` config knob as `GPUPatchify3D`'s own
                Canny scoring. `None` (default) leaves `canny_thresholds`
                untouched. Not numerically guaranteed equivalent across the
                GPU and CPU Canny implementations even at the same
                threshold value (different gradient-computation algorithms
                can produce gradient magnitudes on different absolute
                scales for the same real edge strength) -- shared as one
                convenient knob, not a claim of identical sensitivity.
            canny_high_threshold: See `canny_low_threshold` -- both must be
                given together to take effect.
        """
        super().__init__()

        self.fixed_length = fixed_length
        self.interp_size = interp_size
        self.num_channels = num_channels
        self.dataset = dataset
        self.return_edges = return_edges
        self.profile = profile
        self.score_fn = score_fn
        self.min_size = min_size

        self.sths = [canny_sigma ** 2] if canny_sigma is not None else sths
        if canny_low_threshold is not None and canny_high_threshold is not None:
            self.canny_thresholds = (canny_low_threshold, canny_high_threshold)
        else:
            self.canny_thresholds = canny_thresholds

    def forward(self, img):  # we assume inputs are always structured like this
        """Computes a 3D edge volume for `img` (or, in variance mode, uses `img` directly) and adaptively patchifies it via an octree.

        Args:
            img: Input 3D volume array, shape (D, H, W, C) -- any D/H/W, not
                just a power of two (see `_pad_to_power_of_two`; a no-op
                when all three already are).

        Returns:
            If `self.return_edges` is False: `(seq_img, seq_size, seq_pos,
            octtree)`. If True: `(seq_img, seq_size, seq_pos, octtree, edges)`.
            `seq_img` is the flattened patch sequence, `seq_size` the per-patch side
            length, `seq_pos` the per-patch center position, `octtree` the
            `FixedOctTree` instance, and `edges` the computed edge volume
            (`img` itself, under `score_fn="variance"`) -- both already
            reflect any padding above, so a leaf near the padded edge may
            cover some replicated (not real) content, exactly like
            `GPUPatchify3D`'s own equivalent tradeoff.
        """
        img = _pad_to_power_of_two(img, ndim=3)

        if self.score_fn == "variance":
            edges = img
            if self.profile:
                t_octree_start = time.time()
            octtree = FixedOctTree(domain=edges, fixed_length=self.fixed_length, score_fn=self.score_fn, min_size=self.min_size)
            if self.profile:
                t_serialize_start = time.time()
                octree_time = t_serialize_start - t_octree_start
            seq_img, seq_size, seq_pos = octtree.serialize(img, size=(self.interp_size,self.interp_size,self.interp_size, self.num_channels))
            if self.profile:
                serialize_time = time.time() - t_serialize_start
                worker_info = get_worker_info()
                print(
                    "patchify_3d worker_pid", os.getpid(), "worker_id", worker_info.id if worker_info is not None else None,
                    "edge_time", 0.0, "octree_time", octree_time, "serialize_time", serialize_time,
                    "patchify_time", octree_time + serialize_time,
                    flush=True,
                )
            return self._serialize_return(seq_img, seq_size, seq_pos, octtree, edges)

        self.smooth_factor = random.choice(self.sths)
        variance = [float(self.smooth_factor)] * 3

        if self.profile:
            t_edge_start = time.time()

        # One real 3D Canny call per channel (SimpleITK.CannyEdgeDetection
        # doesn't support multi-channel/vector images directly), summed
        # into a per-voxel count of how many channels independently flag it
        # as an edge -- the actual design intent, preserved exactly (see
        # this class's own docstring).
        edges_combined_counter = np.zeros(img.shape[:3], dtype=np.uint8)
        for j in range(self.num_channels):
            channel = img[:, :, :, j].astype(np.float32)
            # Per-channel min-max normalized to [0,1] before Canny -- keeps
            # canny_thresholds meaningful regardless of this dataset's real
            # intensity scale (matters most for "sst"'s arbitrary physical
            # units, a near-no-op for "basic_ct", already close to [0,1]
            # from its own file-read-time normalization). Edge-detection
            # input only (octree.serialize below still gets `img` unmodified,
            # so the model trains on real values, not a renormalized proxy)
            # -- unconditional across every dataset, not scoped by name.
            lo, hi = channel.min(), channel.max()
            if hi > lo:
                channel = (channel - lo) / (hi - lo)
            channel_img = sitk.GetImageFromArray(channel)
            channel_edges = sitk.CannyEdgeDetection(
                channel_img,
                lowerThreshold=self.canny_thresholds[0], upperThreshold=self.canny_thresholds[1],
                variance=variance,
            )
            edges_combined_counter += sitk.GetArrayFromImage(channel_edges).astype(np.uint8)

        edges = edges_combined_counter

        if self.profile:
            t_octree_start = time.time()
            edge_time = t_octree_start - t_edge_start

        octtree = FixedOctTree(domain=edges, fixed_length=self.fixed_length, score_fn=self.score_fn, min_size=self.min_size)

        if self.profile:
            t_serialize_start = time.time()
            octree_time = t_serialize_start - t_octree_start

        seq_img, seq_size, seq_pos = octtree.serialize(img, size=(self.interp_size,self.interp_size,self.interp_size, self.num_channels))

        if self.profile:
            serialize_time = time.time() - t_serialize_start
            worker_info = get_worker_info()
            print(
                "patchify_3d worker_pid", os.getpid(), "worker_id", worker_info.id if worker_info is not None else None,
                "edge_time", edge_time, "octree_time", octree_time, "serialize_time", serialize_time,
                "patchify_time", edge_time + octree_time + serialize_time,
                flush=True,
            )

        return self._serialize_return(seq_img, seq_size, seq_pos, octtree, edges)

    def _serialize_return(self, seq_img, seq_size, seq_pos, octtree, edges):
        """Reshapes `octtree.serialize`'s raw output and packages `forward`'s return value.

        Shared tail of `forward`'s canny and variance branches -- everything
        after `octtree.serialize` is called is identical either way.

        Args:
            seq_img: Raw patch sequence from `octtree.serialize`.
            seq_size: Raw per-patch side lengths from `octtree.serialize`.
            seq_pos: Raw per-patch center positions from `octtree.serialize`.
            octtree: `FixedOctTree` used to produce the above.
            edges: The domain array used to build `octtree` -- only returned
                when `self.return_edges` is True.

        Returns:
            Same as `forward`'s own return value.
        """
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img, dtype=np.float32)
        if self.num_channels > 1:
            seq_img = np.moveaxis(seq_img, -1, 0)
            seq_img = np.reshape(seq_img, [self.num_channels, -1, self.interp_size*self.interp_size*self.interp_size])
        else:
            seq_img = np.reshape(seq_img, [-1, self.interp_size*self.interp_size*self.interp_size])

        seq_pos = np.asarray(seq_pos)
        if self.return_edges:
            return seq_img, seq_size, seq_pos, octtree, edges
        else:
            return seq_img, seq_size, seq_pos, octtree
