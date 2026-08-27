import numpy as np
import cv2 as cv
import torch
import random
import SimpleITK as sitk
from skimage.feature import canny as skimage_canny
from .quadtree import FixedQuadTree
from .octree import FixedOctTree

class Patchify(torch.nn.Module):
    """Adaptive (quadtree-based) patchification transform for 2D images.

    Detects edges with a randomly smoothed Canny filter, builds a `FixedQuadTree`
    over the edge map, and serializes the image into a fixed-length sequence of
    variable-sized patches concentrated around detected edges.

    Uses two different Canny implementations depending on `dataset`:
    `imagenet`/`catsdogs` (real, already-uint8 `[0,255]` photos, possibly
    multi-channel) use `cv2.Canny` directly, unchanged from before. Every other
    dataset (arbitrary-range float data, e.g. `basic_ct`) uses
    `skimage.feature.canny` instead of `cv2.Canny((img*255).astype(np.uint8),
    ...)` -- the old code assumed the float image was already normalized to
    exactly `[0,1]` before scaling to `cv2.Canny`'s required 8-bit range, which
    silently wastes dynamic range (or clips) whenever that assumption doesn't
    hold; `skimage.feature.canny` operates on the real float values directly,
    with no scaling/casting needed. The tradeoff: `skimage.feature.canny` only
    accepts single-channel 2D input (unlike `cv2.Canny`, which silently
    combines multi-channel gradients) -- this is why `imagenet`/`catsdogs`
    (real multi-channel photos) deliberately keep `cv2.Canny` rather than
    switching over too.
    """

    def __init__(self, sths=[0,1,3,5], fixed_length=196, cannys=[50, 100], canny_quantiles=(0.7, 0.9), interp_size=16, num_channels=3, dataset="imagenet", return_edges=False) -> None:
        """Initializes the randomization ranges and patch parameters for the transform.

        Args:
            sths: Candidate Gaussian smoothing strengths to randomly choose from
                before edge detection (0 = no smoothing, use uniform random noise
                as the edge map instead). For `imagenet`/`catsdogs`
                (`cv2.GaussianBlur`), a kernel size (odd integer); for every
                other dataset (`skimage.feature.canny`'s own `sigma`), a
                standard deviation (float) -- these aren't numerically
                equivalent, so pass different values for a `dataset` that isn't
                `imagenet`/`catsdogs`.
            fixed_length: Fixed number of patches the image is serialized into.
            cannys: `imagenet`/`catsdogs` only (`cv2.Canny`): `[low, high)`
                range of absolute lower thresholds to randomly choose from; the
                corresponding upper threshold is `low + 50`.
            canny_quantiles: Every other dataset only (`skimage.feature.canny`,
                `use_quantiles=True`): `(low, high)` hysteresis thresholds, as
                quantiles of the edge-magnitude distribution in `[0, 1]` --
                dataset-scale-independent by construction, unlike `cannys`'
                absolute values. Starting values, not empirically tuned.
            interp_size: Side length each (square) leaf patch is interpolated to.
            num_channels: Number of image channels.
            dataset: Dataset name; controls how edges are computed/normalized
                ("imagenet"/"catsdogs" vs. other datasets).
            return_edges: If True, also return the computed edge map from `forward`.
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

    def forward(self, img):  # we assume inputs are always structured like this
        """Computes an edge map for `img` and adaptively patchifies it via a quadtree.

        Args:
            img: Input 2D image array, shape (H, W[, C]). For any `dataset`
                other than `imagenet`/`catsdogs`, `C` (if present) must be 1 --
                `skimage.feature.canny` only accepts single-channel input (see
                this class's own docstring).

        Returns:
            If `self.return_edges` is False: `(seq_img, seq_size, seq_pos, qdt)`.
            If True: `(seq_img, seq_size, seq_pos, qdt, edges)`. `seq_img` is the
            flattened patch sequence, `seq_size` the per-patch side length,
            `seq_pos` the per-patch center position, `qdt` the `FixedQuadTree`
            instance, and `edges` the computed edge map.
        """
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

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img, seq_size, seq_pos = qdt.serialize(img, size=(self.interp_size,self.interp_size,self.num_channels))
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img, dtype=np.float32)

        if self.num_channels > 1:
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
    (`SimpleITK.CannyEdgeDetection`), weights each voxel by how many
    channels independently flag it as an edge, builds a `FixedOctTree` over
    the result, and serializes the volume into a fixed-length sequence of
    variable-sized patches concentrated around detected edges.

    Replaced an earlier per-slice, per-channel `cv2.Sobel`/`cv2.Canny`
    pipeline (archived at
    `../UCF-VIT-claude-archive/src/UCF_VIT/dataloaders/transform.py`) for
    two real, verified reasons:

    1. That pipeline only ever computed 2D (in-plane) gradients/edges, one
       slice at a time -- no derivative along the depth axis was ever
       computed. Verified directly: a synthetic volume that's a step
       function purely along depth (uniform within every slice, only a hard
       transition *across* slices) produced zero detected edges there,
       while `SimpleITK.CannyEdgeDetection` (a genuine 3D Canny -- one call
       over the whole volume per channel, not a per-slice loop) correctly
       marks the transition plane. A real blind spot for volumes where
       depth-direction structure matters, not just a style difference.
    2. Its per-slice `cv2.Canny` calls needed `(slice*255).astype(np.uint8)`
       to satisfy `cv2.Canny`'s 8-bit input requirement, silently assuming
       the float volume was already normalized to exactly `[0,1]` --
       wrong (wastes dynamic range, or clips) whenever that assumption
       doesn't hold. `SimpleITK.CannyEdgeDetection` operates on the real
       float values directly.

    The old pipeline's separate Sobel-based `gradient_direction`
    computation and the direction-consistency gate built on it
    (`edge_data_normalized > threshold`) were dropped entirely, not
    reimplemented in 3D: that gate was built on a strictly 2D angle
    (`arctan2(sobely, sobelx)`) with no natural 3D generalization, and its
    likely purpose -- suppressing noisy/spurious edges from a per-slice,
    single-"best-channel" accumulation -- is largely redundant with what a
    real per-channel 3D Canny already provides (its own
    non-max-suppression + hysteresis thresholding already produces a
    clean, thinned edge mask, not a noisy raw threshold). `FixedOctTree`'s
    consumed score is only ever used for relative (`max()`-based)
    comparison during tree-building (see `Cube.contains`'s own docstring),
    never read for its precise value elsewhere, so the simpler score below
    doesn't need to also be a "clean" signal, just a meaningfully-ordered
    one.

    The per-channel-edge-count weighting itself (give more weight to a
    voxel flagged as an edge on more channels) is preserved exactly --
    verified directly that a per-channel `SimpleITK.CannyEdgeDetection`-
    then-accumulate loop produces the identical counter semantics (a voxel
    edge on 2 channels scores 2x a voxel edge on 1 channel) as the old
    pipeline's `edges_combined_counter`, just fed by genuinely 3D
    per-channel edge masks instead of stacked-per-slice 2D ones.
    `SimpleITK.CannyEdgeDetection` doesn't support multi-channel/vector
    images directly (verified) -- confirmed this is why the per-channel
    loop is still needed at all, not something the 3D switch removes.
    """

    def __init__(self, sths=[0.5,1.0,2.0], fixed_length=196, canny_thresholds=(0.05, 0.15), interp_size=16, num_channels=3, dataset="basic_ct", return_edges=False) -> None:
        """Initializes the randomization ranges and patch parameters for the transform.

        Args:
            sths: Candidate Gaussian smoothing strengths to randomly choose
                from before edge detection -- passed directly as
                `SimpleITK.CannyEdgeDetection`'s own `variance` parameter
                (replicated across all 3 axes), which handles smoothing
                internally (no separate blur step needed, unlike the old
                pipeline). Note: `variance`, not standard deviation (sigma)
                -- not numerically equivalent to this parameter's old
                meaning, needs its own tuning regardless.
            fixed_length: Fixed number of patches the volume is serialized into.
            canny_thresholds: `(low, high)` hysteresis thresholds for
                `SimpleITK.CannyEdgeDetection` -- absolute values on the
                (smoothed) gradient-magnitude scale, not quantiles (unlike
                `Patchify`'s `canny_quantiles`) -- SimpleITK has no
                quantile-threshold option. Starting values, not empirically
                tuned; assumes roughly `[0,1]`-scale input intensities
                (matches this repo's own min-max-normalized `basic_ct`
                loading).
            interp_size: Side length each (cubic) leaf patch is interpolated to.
            num_channels: Number of volume channels.
            dataset: Dataset name; kept for interface compatibility with `Patchify`.
            return_edges: If True, also return the computed edge volume from
                `forward`.
        """
        super().__init__()

        self.sths = sths
        self.fixed_length = fixed_length
        self.canny_thresholds = canny_thresholds
        self.interp_size = interp_size
        self.num_channels = num_channels
        self.dataset = dataset
        self.return_edges = return_edges

    def forward(self, img):  # we assume inputs are always structured like this
        """Computes a 3D edge volume for `img` and adaptively patchifies it via an octree.

        Args:
            img: Input 3D volume array, shape (D, H, W, C).

        Returns:
            If `self.return_edges` is False: `(seq_img, seq_size, seq_pos,
            octtree)`. If True: `(seq_img, seq_size, seq_pos, octtree, edges)`.
            `seq_img` is the flattened patch sequence, `seq_size` the per-patch side
            length, `seq_pos` the per-patch center position, `octtree` the
            `FixedOctTree` instance, and `edges` the computed edge volume.
        """
        self.smooth_factor = random.choice(self.sths)
        variance = [float(self.smooth_factor)] * 3

        # One real 3D Canny call per channel (SimpleITK.CannyEdgeDetection
        # doesn't support multi-channel/vector images directly), summed
        # into a per-voxel count of how many channels independently flag it
        # as an edge -- the actual design intent, preserved exactly (see
        # this class's own docstring).
        edges_combined_counter = np.zeros(img.shape[:3], dtype=np.uint8)
        for j in range(self.num_channels):
            channel_img = sitk.GetImageFromArray(img[:, :, :, j].astype(np.float32))
            channel_edges = sitk.CannyEdgeDetection(
                channel_img,
                lowerThreshold=self.canny_thresholds[0], upperThreshold=self.canny_thresholds[1],
                variance=variance,
            )
            edges_combined_counter += sitk.GetArrayFromImage(channel_edges).astype(np.uint8)

        edges = edges_combined_counter

        octtree = FixedOctTree(domain=edges, fixed_length=self.fixed_length)

        seq_img, seq_size, seq_pos = octtree.serialize(img, size=(self.interp_size,self.interp_size,self.interp_size, self.num_channels))
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img, dtype=np.float32)
        if self.num_channels > 1:
            seq_img = np.reshape(seq_img, [self.num_channels, -1, self.interp_size*self.interp_size*self.interp_size])
        else:
            seq_img = np.reshape(seq_img, [-1, self.interp_size*self.interp_size*self.interp_size])

        seq_pos = np.asarray(seq_pos)
        if self.return_edges:
            return seq_img, seq_size, seq_pos, octtree, edges
        else:
            return seq_img, seq_size, seq_pos, octtree
