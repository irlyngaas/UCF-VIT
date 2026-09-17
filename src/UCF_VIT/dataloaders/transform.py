import os
import time

import numpy as np
import cv2 as cv
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
    `imagenet`/`catsdogs` (real, already-uint8 `[0,255]` photos, possibly
    multi-channel) use `cv2.Canny` directly. Every other dataset (arbitrary-range
    float data, e.g. `basic_ct`) uses `skimage.feature.canny` instead, which
    operates on the real float values directly with no `[0,1]`-normalized
    scaling/casting to `cv2.Canny`'s required 8-bit range needed. The tradeoff:
    `skimage.feature.canny` only accepts single-channel 2D input (unlike
    `cv2.Canny`, which combines multi-channel gradients), so `imagenet`/`catsdogs`
    (real multi-channel photos) keep `cv2.Canny`.
    """

    def __init__(self, sths=[0,1,3,5], fixed_length=196, cannys=[50, 100], canny_quantiles=(0.7, 0.9), interp_size=16, num_channels=3, dataset="imagenet", return_edges=False, score_fn="canny", min_size=2) -> None:
        """Initializes the randomization ranges and patch parameters for the transform.

        Args:
            sths: Candidate Gaussian smoothing strengths to randomly choose from
                before edge detection (0 = no smoothing, use uniform random noise
                as the edge map instead). For `imagenet`/`catsdogs`
                (`cv2.GaussianBlur`), a kernel size (odd integer); for every
                other dataset (`skimage.feature.canny`'s own `sigma`), a
                standard deviation (float) -- these aren't numerically
                equivalent, so pass different values for a `dataset` that isn't
                `imagenet`/`catsdogs`. Unused when `score_fn="variance"`.
            fixed_length: Fixed number of patches the image is serialized into.
            cannys: `imagenet`/`catsdogs` only (`cv2.Canny`): `[low, high)`
                range of absolute lower thresholds to randomly choose from; the
                corresponding upper threshold is `low + 50`. Unused when
                `score_fn="variance"`.
            canny_quantiles: Every other dataset only (`skimage.feature.canny`,
                `use_quantiles=True`): `(low, high)` hysteresis thresholds, as
                quantiles of the edge-magnitude distribution in `[0, 1]` --
                dataset-scale-independent by construction, unlike `cannys`'
                absolute values. Starting values, not empirically tuned.
                Unused when `score_fn="variance"`.
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
        """
        super().__init__()

        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.canny_quantiles = canny_quantiles
        self.interp_size = interp_size
        self.num_channels = num_channels
        self.dataset = dataset
        self.return_edges = return_edges
        self.score_fn = score_fn
        self.min_size = min_size

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
        c = random.choice(self.cannys)
        self.canny = [c, c+50]
        if self.smooth_factor == 0:
            if self.dataset == "imagenet" or self.dataset == "catsdogs":
                edges = np.random.uniform(low=0,high=1,size=(img.shape[0],img.shape[1]))
            else:
                edges = np.random.uniform(low=np.min(img),high=np.max(img),size=(img.shape[0],img.shape[1]))
        else:
            if self.dataset == "imagenet" or self.dataset == "catsdogs":
                grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
                edges = cv.Canny(grey_img, self.canny[0], self.canny[1])
            else:
                if img.ndim == 3:
                    if img.shape[-1] != 1:
                        raise NotImplementedError(
                            f"Patchify's skimage.feature.canny path (dataset={self.dataset!r}) only "
                            f"supports single-channel input, got {img.shape[-1]} channels -- "
                            "cv2.Canny (used for imagenet/catsdogs) combines multi-channel gradients "
                            "internally, skimage.feature.canny doesn't."
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
    by the weighting above rather than in a single multi-channel call.
    """

    def __init__(self, sths=[0.5,1.0,2.0], fixed_length=196, canny_thresholds=(0.05, 0.15), interp_size=16, num_channels=3, dataset="basic_ct", return_edges=False, profile=False, score_fn="canny", min_size=2) -> None:
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
                `score_fn="variance"`.
            fixed_length: Fixed number of patches the volume is serialized into.
            canny_thresholds: `(low, high)` hysteresis thresholds for
                `SimpleITK.CannyEdgeDetection` -- absolute values on the
                (smoothed) gradient-magnitude scale, not quantiles (unlike
                `Patchify`'s `canny_quantiles`) -- SimpleITK has no
                quantile-threshold option. Starting values, not empirically
                tuned; assumes roughly `[0,1]`-scale input intensities
                (matches this repo's own min-max-normalized `basic_ct`
                loading). Unused when `score_fn="variance"`.
            interp_size: Side length each (cubic) leaf patch is interpolated to.
            num_channels: Number of volume channels.
            dataset: Dataset name. Mostly kept for interface compatibility
                with `Patchify`, except for `"sst"`: its raw CFD fields are
                in arbitrary physical units, not the ~[0,1] scale
                `canny_thresholds` assumes (true for `basic_ct` only
                because it's min-max normalized once at file-read time) --
                `"sst"` gets its edge-detection input (only -- not the real
                patch content) locally, per-channel min-max normalized
                instead. Every other dataset's behavior is unaffected.
                Unused when `score_fn="variance"`.
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
        """
        super().__init__()

        self.sths = sths
        self.fixed_length = fixed_length
        self.canny_thresholds = canny_thresholds
        self.interp_size = interp_size
        self.num_channels = num_channels
        self.dataset = dataset
        self.return_edges = return_edges
        self.profile = profile
        self.score_fn = score_fn
        self.min_size = min_size

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
            if self.dataset == "sst":
                # "sst"'s raw CFD fields (density/velocity/pressure) are in
                # arbitrary physical units, not the ~[0,1] scale
                # canny_thresholds assumes (see this class's own docstring
                # -- true for basic_ct because it's min-max normalized once
                # at file-read time, not true here at all). Normalizing
                # *only* this edge-detection input, per channel, keeps the
                # existing thresholds meaningful without touching the real
                # patch content below (octree.serialize still gets `img`
                # unmodified, so the model trains on real physical values,
                # not a renormalized proxy). Scoped to "sst" specifically --
                # every other dataset's edge-detection input is unchanged.
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
