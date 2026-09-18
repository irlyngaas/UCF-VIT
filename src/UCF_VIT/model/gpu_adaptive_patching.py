import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label


def _label_regions(border_mask, region_backend="scipy"):
    """Connected components of `~border_mask` -> `(labeled, num_features)`.

    Shared by `GPUPatchify2D`/`GPUPatchify3D._detect_regions` -- dimension-
    agnostic, since both `scipy.ndimage.label` and `cupyx.scipy.ndimage.
    label` already work on N-D arrays unchanged.

    Args:
        border_mask: Bool tensor, any number of dims, `True` on block
            borders/faces (see `GPUPatchify2D._border_mask`/`GPUPatchify3D.
            _face_mask`).
        region_backend: `"scipy"` (default) -- CPU round-trip via `scipy.
            ndimage.label`, today's exact behavior. `"cupyx"` -- GPU-native
            via `cupyx.scipy.ndimage.label`, using the DLPack protocol
            (`cupy`/`torch` both implement `__dlpack__` directly, no
            manual `to_dlpack`/`toDlpack` needed) to move `border_mask`
            to/from `cupy` with no CPU round-trip. Opt-in and explicit,
            not auto-detected: raises clearly if `cupy` isn't importable
            or `border_mask` isn't on a CUDA device, rather than silently
            falling back to `"scipy"`. Not verified on real GPU hardware
            in this session (no CUDA/ROCm device, `cupy` not installed
            here) -- trusts `cupyx`'s documented `scipy.ndimage.label`
            API compatibility for the real numerics.

    Returns:
        `(labeled, num_features)`: `labeled` is an integer-dtype tensor on
        `border_mask`'s own device, same shape, `0` = background;
        `num_features` is a plain `int`.
    """
    assert region_backend in ("scipy", "cupyx"), f"region_backend must be 'scipy' or 'cupyx', got {region_backend!r}"

    if region_backend == "cupyx":
        try:
            import cupy
            from cupyx.scipy.ndimage import label as cupyx_label
        except ImportError as e:
            raise ImportError(
                "region_backend='cupyx' requires the cupy package (with a CUDA or "
                "ROCm build matching this environment's GPU), which isn't installed "
                "here."
            ) from e
        assert border_mask.is_cuda, (
            "region_backend='cupyx' requires border_mask to be on a CUDA device -- "
            f"got device {border_mask.device}. Use region_backend='scipy' for CPU tensors."
        )
        interior_cp = cupy.from_dlpack(~border_mask)
        labeled_cp, num_features = cupyx_label(interior_cp)
        labeled = torch.from_dlpack(labeled_cp)
        return labeled, int(num_features)

    device = border_mask.device
    interior = ~border_mask.cpu().numpy()
    labeled_np, num_features = scipy_label(interior)
    return torch.from_numpy(labeled_np).to(device), int(num_features)


class GPUPatchify2D(torch.nn.Module):
    """GPU-native, level-parallel adaptive patchification for 2D images.

    Runs entirely on-device (no per-sample CPU work), inside a model's
    `forward()` rather than in the dataloader -- the CPU/dataloader-side
    alternative (`UCF_VIT.dataloaders.transform.Patchify`) stays available
    and unchanged for datasets/configs that prefer it.

    Structurally the opposite of `Patchify`/`FixedQuadTree` (`quadtree.py`):
    that class starts from one root node and greedily *splits* the
    highest-scoring node into 4 children, growing 1 -> `fixed_length`. This
    class starts from the *finest* possible grid (every `min_size` block
    alive) and greedily *merges* the 4 children with the lowest merge cost
    into their parent, shrinking down to exactly `fixed_length` leaves --
    "merge cost" is variance/SSE-based (how much block-uniformity
    information would be lost by treating 4 children as one flat block),
    not edge-density. Every iteration considers every mergeable candidate
    across every level *and* every image in the batch at once, and applies
    the globally cheapest ones via a single `torch.topk` plus a vectorized
    per-image budget check -- no Python loop over individual nodes or
    images in the hot path (only over `max_level`, a small constant).

    Ported from a reference GPU implementation the user maintains
    elsewhere (`bayes-cast`'s `adaptive_patching_vectorized.Patchify`), 2D
    only for now. Merge-cost scoring is pluggable (`score_fn`, see `_compute_
    all_levels_batch`): `"variance"` (the ported default, above) or
    `"canny"` (edge-density, via `_canny_edge_map_batch`) -- see this
    module's own tests/README.md entry for the scope still left out (3D,
    and a fully-GPU connected-components backend) and why.

    Neither `score_fn` is numerically identical to its CPU-side
    (`UCF_VIT.dataloaders.transform.Patchify`) counterpart -- expect
    similar, not identical, scores/splits from the same input:

    - `"canny"`: a separate, from-scratch torch/GPU reimplementation of the
      Canny pipeline (blur -> Sobel gradients -> non-max suppression ->
      double threshold -> hysteresis), not a call into `Patchify`'s own 2D
      Canny (`cv2.Canny` for `imagenet`/`catsdogs`, `skimage.feature.canny`
      for every other dataset) -- both CPU-only, non-batched library calls
      that can't run inside this on-device `forward()`. Differs in
      smoothing (an explicit Gaussian sigma here vs. each library's own
      internal smoothing) and, most significantly, in hysteresis: this
      class approximates it with `canny_hysteresis_iters` binary-dilation
      passes rather than the true flood-fill connectivity both `cv2.Canny`
      and `skimage.feature.canny` use (see `_canny_edge_map_batch`'s own
      docstring) -- an approximation, not an exact match, by design.
    - `"variance"`: same SSE formula in spirit as `Rect.contains`'s own
      `"variance"` scoring (`quadtree.py`) -- per-channel variance, summed
      across channels, scaled by block area -- but computed via
      `torch.var`, whose default (`correction=1`, Bessel's correction, a
      `n-1` denominator) differs from `np.var`'s default (`ddof=0`, a
      plain `n` denominator) used on the CPU side. The gap is largest for
      small blocks (near `min_size`) and shrinks as block size grows; it's
      also the source of this class's own well-verified "a small, detailed
      block's own SSE can come out below its children's summed SSE" merge-
      cost property (see this module's tests/README.md entries for both
      the 2D and 3D real bugs this property was mistaken for and then
      root-caused). Structurally different from the CPU side regardless of
      this: `Patchify`/`FixedQuadTree` use this SSE as a single node's
      top-down split-priority score, while this class uses `errors[level]
      - sum_children` (the between-group variance component) to decide
      whether merging is cheap -- a different consumer of a similar
      quantity, not the same computation.
    """

    # Bounding-box + scalar fields describing each detected leaf region,
    # vectorized over all N regions in one image at once (no Python loop
    # over regions).
    RegionTensors = namedtuple(
        "RegionTensors", ["x0s", "x1s", "y0s", "y1s", "phs", "pws", "cxs", "cys", "N"]
    )

    def __init__(
            self, img_size, fixed_length=196, interp_size=16, min_size=2,
            score_fn="variance", canny_sigma=1.0, canny_low_threshold=0.1,
            canny_high_threshold=0.2, canny_hysteresis_iters=2,
            region_backend="scipy",
    ):
        """Precomputes the level structure this image size implies.

        Args:
            img_size: `(H, W)` of the (unpadded) input. If `H != W`,
                requires `H < W` (matches the reference implementation's
                own limitation -- `H > W` raises clearly rather than
                silently computing a wrong grid).
            fixed_length: Target number of leaf regions to merge down to.
                Must satisfy `(max_blocks**2 - fixed_length) % 3 == 0`
                (every merge removes exactly 3 leaves -- 4 children into 1
                parent -- starting from `max_blocks**2` level-0 blocks;
                see `_serialize_batch`'s own docstring for why every image
                in a batch must land on the exact same leaf count).
            interp_size: Side length each (square) leaf region is resized
                to.
            min_size: Finest block side length (level 0), along the
                shorter image axis.
            score_fn: `"variance"` (default -- see `_compute_all_levels_
                batch`'s own docstring) or `"canny"` (edge-density scoring,
                via `_canny_edge_map_batch` -- a merge cost is that block's
                own edge-pixel count directly, not the variance formula's
                `errors[level] - sum_children`; see `_compute_all_levels_
                batch` for why those two formulas can't be the same).
            canny_sigma: Gaussian smoothing sigma applied before computing
                gradients. Only used when `score_fn == "canny"`.
            canny_low_threshold: Lower of the two gradient-magnitude
                thresholds ("weak" edges) -- unnormalized, on whatever
                scale `img` itself is in (same "starting values, not
                empirically tuned" caveat as `UCF_VIT.dataloaders.
                transform.Patchify_3D`'s own `canny_low_threshold`). Only
                used when `score_fn == "canny"`.
            canny_high_threshold: Upper of the two gradient-magnitude
                thresholds ("strong" edges). Only used when `score_fn ==
                "canny"`.
            canny_hysteresis_iters: Number of binary-dilation passes used
                to approximate hysteresis edge-linking (grow the strong-
                edge mask and absorb any weak edges it touches, repeated
                this many times) -- not exact flood-fill connectivity, see
                `_canny_edge_map_batch`'s own docstring. Only used when
                `score_fn == "canny"`.
            region_backend: `"scipy"` (default) or `"cupyx"` -- see
                `_label_regions`'s own docstring for what each means.
        """
        super().__init__()
        self.interp_size = interp_size
        self.fixed_length = fixed_length

        assert score_fn in ("variance", "canny"), f"score_fn must be 'variance' or 'canny', got {score_fn!r}"
        self.score_fn = score_fn
        self.canny_sigma = canny_sigma
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.canny_hysteresis_iters = canny_hysteresis_iters
        self.region_backend = region_backend

        if img_size[0] == img_size[1]:
            max_blocks = img_size[0] // min_size
            self.min_size = [min_size, min_size]
        elif img_size[0] < img_size[1]:
            img_size_ratio = int(img_size[1] / img_size[0])
            max_blocks = img_size[0] // min_size
            self.min_size = [min_size, min_size * img_size_ratio]
        else:
            raise NotImplementedError(
                f"GPUPatchify2D requires img_size[0] <= img_size[1], got {img_size} "
                "-- the reference implementation this was ported from has the same "
                "limitation (its own min_size-ratio trick, which keeps both axes at "
                "the same block count per level, only derives a ratio for H <= W)."
            )

        self.max_level = int(math.floor(math.log2(max_blocks))) if max_blocks >= 1 else 0

        # initial_leaves must reflect the *padded* grid run_merge_batch actually
        # operates on, not the raw img_size//min_size above -- whenever img_size
        # isn't already a multiple of _pad_size(), _pad_tensor grows H/W, which
        # grows the real level-0 block count too (e.g. img_size=(12,12),
        # min_size=2 pads to (16,16): 8x8=64 real level-0 blocks, not 6x6=36).
        pad_bh = self.min_size[0] * (2 ** self.max_level)
        padded_h = -(-img_size[0] // pad_bh) * pad_bh  # ceil to a multiple of pad_bh
        real_max_blocks = padded_h // self.min_size[0]

        initial_leaves = real_max_blocks ** 2
        assert (initial_leaves - fixed_length) % 3 == 0, (
            f"fixed_length ({fixed_length}) must satisfy (real_max_blocks**2 - fixed_length) % 3 == 0 "
            f"-- real_max_blocks**2 is {initial_leaves} here (img_size {img_size} padded to a multiple "
            f"of {pad_bh}, min_size {min_size}), so every image's merge loop can land on exactly "
            "fixed_length leaves (each merge removes exactly 3 leaves). Off-target leaf counts would "
            "either silently break _serialize_batch's whole-batch grid_sample call (which assumes "
            "every image in the batch has the same region count), or -- if unreachable exactly -- "
            "make run_merge_batch's per-image budget hit 0 forever, hanging."
        )
        self.initial_leaves = initial_leaves

    def _pad_size(self):
        """The block size `(bh, bw)` the padded image's H/W must be divisible by."""
        return (self.min_size[0] * (2 ** self.max_level), self.min_size[1] * (2 ** self.max_level))

    def _compute_padding(self, H, W):
        bh, bw = self._pad_size()
        return (bh - (H % bh)) % bh, (bw - (W % bw)) % bw

    def _pad_tensor(self, t):
        """Edge-replication-pads a `[..., H, W]` tensor so H/W divide evenly by the coarsest block size."""
        H, W = t.shape[-2], t.shape[-1]
        pad_h, pad_w = self._compute_padding(H, W)
        if pad_h == 0 and pad_w == 0:
            return t, (H, W)
        return F.pad(t, (0, pad_w, 0, pad_h), mode="replicate"), (H, W)

    def _compute_level_stats_batch(self, block_size, imgs):
        """Per-block variance-based error (SSE) at one level, for the whole batch.

        Args:
            block_size: `(bsh, bsw)` block side lengths at this level.
            imgs: `[B, C, H, W]`.

        Returns:
            `[B, gh, gw]` sum of per-channel block variance, scaled by
            block area -- "how much this block's own detail would be lost
            if it were treated as one flat value."
        """
        B, C, Hp, Wp = imgs.shape
        bsh, bsw = block_size
        gh, gw = Hp // bsh, Wp // bsw

        reshaped = imgs.reshape(B, C, gh, bsh, gw, bsw).permute(0, 1, 2, 4, 3, 5)  # [B,C,gh,gw,bsh,bsw]
        vars_ = reshaped.var(dim=(-2, -1))  # [B,C,gh,gw]
        return vars_.sum(dim=1) * (bsh * bsw)  # [B,gh,gw]

    def _box_sum_batch(self, block_size, base_map):
        """Sum of `base_map` within each non-overlapping `block_size` block, for the whole batch.

        The `.sum()` analog of `_compute_level_stats_batch`'s `.var()*area`
        -- used for `score_fn == "canny"`, where the per-pixel edge map is
        already additive (unlike variance, which genuinely needs
        recomputing from raw pixels at each level -- see `_compute_all_
        levels_batch`'s own docstring).

        Args:
            block_size: `(bsh, bsw)` block side lengths at this level.
            base_map: `[B, H, W]`.

        Returns:
            `[B, gh, gw]`.
        """
        B, Hp, Wp = base_map.shape
        bsh, bsw = block_size
        gh, gw = Hp // bsh, Wp // bsw

        reshaped = base_map.reshape(B, gh, bsh, gw, bsw).permute(0, 1, 3, 2, 4)  # [B,gh,gw,bsh,bsw]
        return reshaped.sum(dim=(-2, -1))  # [B,gh,gw]

    def _gaussian_kernel2d(self, sigma, device, dtype):
        """Builds a fixed, normalized 2D Gaussian kernel for pre-gradient smoothing.

        Args:
            sigma: Gaussian standard deviation, in pixels.
            device: Device the kernel should live on.
            dtype: Dtype the kernel should be.

        Returns:
            `[1, 1, K, K]` kernel, `K = 2 * round(3 * sigma) + 1`.
        """
        radius = max(1, int(round(3 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)  # [K, K]
        return kernel2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

    def _sobel_gradients_batch(self, imgs):
        """Horizontal/vertical Sobel gradients, for the whole batch.

        Args:
            imgs: `[B, 1, H, W]`.

        Returns:
            `(gx, gy)`, each `[B, 1, H, W]`.
        """
        device, dtype = imgs.device, imgs.dtype
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
        padded = F.pad(imgs, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(padded, sobel_x)
        gy = F.conv2d(padded, sobel_y)
        return gx, gy

    def _non_max_suppression_batch(self, mag, angle_deg):
        """Suppresses every gradient-magnitude pixel that isn't a local max along its own gradient direction.

        Direction is discretized into 4 buckets (0/45/90/135 degrees, mod
        180 -- edge orientation is the same for `theta` and `theta+180`),
        each compared against its own pair of neighbor pixels via a
        shifted-tensor comparison (vectorized over the whole batch, no
        pixel-level Python loop).

        Args:
            mag: Gradient magnitude, `[B, 1, H, W]`.
            angle_deg: Gradient direction in degrees, already mod 180,
                same shape as `mag`.

        Returns:
            `[B, 1, H, W]`, `mag` with every non-local-max pixel zeroed.
        """
        H, W = mag.shape[-2:]

        def shift(t, dr, dc):
            padded = F.pad(t, (1, 1, 1, 1), mode="replicate")
            return padded[..., 1 + dr:1 + dr + H, 1 + dc:1 + dc + W]

        neighbor_shifts = {
            0: ((0, -1), (0, 1)),
            45: ((-1, 1), (1, -1)),
            90: ((-1, 0), (1, 0)),
            135: ((-1, -1), (1, 1)),
        }
        bucket_order = (0, 45, 90, 135)
        bucket = torch.round(angle_deg / 45.0).long() % 4

        n1 = torch.zeros_like(mag)
        n2 = torch.zeros_like(mag)
        for i, direction in enumerate(bucket_order):
            (dr1, dc1), (dr2, dc2) = neighbor_shifts[direction]
            is_bucket = (bucket == i)
            n1 = torch.where(is_bucket, shift(mag, dr1, dc1), n1)
            n2 = torch.where(is_bucket, shift(mag, dr2, dc2), n2)

        is_local_max = (mag >= n1) & (mag >= n2)
        return mag * is_local_max.to(mag.dtype)

    def _canny_edge_map_batch(self, imgs):
        """Per-channel Canny edge detection, summed into a per-pixel edge count.

        Runs the standard Canny pipeline (Gaussian blur -> Sobel gradients
        -> non-max suppression -> double threshold -> hysteresis) once per
        channel (each call already batched over `B`), then sums the
        resulting binary edge masks into one `[B, H, W]` count -- mirrors
        `UCF_VIT.dataloaders.transform.Patchify_3D`'s own "weight a voxel
        by how many channels independently flag it as an edge" convention,
        rather than requiring `C == 1` the way the 2D CPU `Patchify.forward`
        path does.

        Hysteresis here is an *approximation*: real Canny links weak edges
        to strong ones via flood-fill connectivity, a graph problem that
        doesn't vectorize cleanly. This instead grows the strong-edge mask
        by one pixel (binary dilation via `F.max_pool2d`) and absorbs any
        weak edges it touches, repeated `self.canny_hysteresis_iters`
        times -- close to real hysteresis for a small number of iterations,
        not an exact match.

        Args:
            imgs: `[B, C, H, W]`.

        Returns:
            `[B, H, W]` float tensor, values in `[0, C]`.
        """
        B, C, H, W = imgs.shape
        device, dtype = imgs.device, imgs.dtype
        kernel = self._gaussian_kernel2d(self.canny_sigma, device, dtype)
        pad = kernel.shape[-1] // 2

        edge_count = torch.zeros(B, H, W, device=device, dtype=dtype)
        for c in range(C):
            channel = imgs[:, c:c + 1]  # [B, 1, H, W]
            blurred = F.conv2d(F.pad(channel, (pad, pad, pad, pad), mode="reflect"), kernel)

            gx, gy = self._sobel_gradients_batch(blurred)
            mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
            angle = torch.atan2(gy, gx) * (180.0 / math.pi)
            angle = angle % 180.0

            nms = self._non_max_suppression_batch(mag, angle)
            strong = nms > self.canny_high_threshold
            weak = (nms > self.canny_low_threshold) & (~strong)

            edge = strong.clone()
            for _ in range(self.canny_hysteresis_iters):
                dilated = F.max_pool2d(edge.to(dtype), kernel_size=3, stride=1, padding=1) > 0
                edge = edge | (weak & dilated)

            edge_count = edge_count + edge.to(dtype).squeeze(1)

        return edge_count

    def _compute_all_levels_batch(self, imgs):
        """Per-level block errors and the merge cost of collapsing each level's blocks into their level-below parent.

        `score_fn == "variance"`: `merge_costs[level] = errors[level] -
        sum_children`, the *between-group* component of variance (law of
        total variance: a coarse block's own SSE-from-its-mean minus its
        children's own internal SSEs is exactly the part caused by the
        children genuinely differing from each other) -- meaningful for a
        variance measure.

        `score_fn == "canny"`: `merge_costs[level] = errors[level]`
        directly, no child subtraction -- an edge-pixel count is already
        purely additive (`errors[level]` would always exactly equal
        `sum_children`, making the variance-style subtraction trivially
        zero everywhere). A block's own edge density, used directly, gives
        the same greedy intent as variance's formula: flat/edge-free
        blocks (any level) get near-zero cost (cheap to merge), edge-rich
        blocks resist merging.

        Args:
            imgs: `[B, C, H, W]`.

        Returns:
            `(merge_costs, level_shapes)`: `merge_costs[level]` is `[B, gh,
            gw]` (`None` at level 0, which is never merged into anything),
            `level_shapes[level]` is `(gh, gw)`.
        """
        errors = []
        level_shapes = []
        merge_costs = [None] * (self.max_level + 1)

        if self.score_fn == "canny":
            edge_map = self._canny_edge_map_batch(imgs)  # [B, H, W]
            for l in range(self.max_level + 1):
                bs = (self.min_size[0] * (2 ** l), self.min_size[1] * (2 ** l))
                err_l = self._box_sum_batch(bs, edge_map)
                errors.append(err_l)
                level_shapes.append(tuple(err_l.shape[1:]))

            for level in range(1, self.max_level + 1):
                merge_costs[level] = errors[level]

            return merge_costs, level_shapes

        for l in range(self.max_level + 1):
            bs = (self.min_size[0] * (2 ** l), self.min_size[1] * (2 ** l))
            sse_l = self._compute_level_stats_batch(bs, imgs)
            errors.append(sse_l)
            level_shapes.append(tuple(sse_l.shape[1:]))

        for level in range(1, self.max_level + 1):
            child_err = errors[level - 1]  # [B, Hp2, Wp2]
            Hp, Wp = child_err.shape[1] // 2, child_err.shape[2] // 2
            sum_children = child_err.reshape(-1, Hp, 2, Wp, 2).sum(dim=(2, 4))  # [B, Hp, Wp]
            merge_costs[level] = errors[level] - sum_children

        return merge_costs, level_shapes

    def _border_mask(self, alive, img):
        """Marks the pixel-grid borders around every currently-alive block.

        Args:
            alive: List (one per level) of `[Hp, Wp]` bool tensors, one
                image's worth.
            img: `[Hp, Wp, C]`, only used for its shape.

        Returns:
            `[H, W]` bool tensor, `True` on the border between (or at the
            outer edge of) alive blocks -- every alive block independently
            stamps its own 4 edges, so two adjacent alive blocks (even
            same-size ones) always get a border line between them.
        """
        H, W = img.shape[0], img.shape[1]
        device = img.device
        out = torch.zeros(H + 1, W + 1, dtype=torch.bool, device=device)

        for level in reversed(range(len(alive))):
            bsh = self.min_size[0] * (2 ** level)
            bsw = self.min_size[1] * (2 ** level)
            if bsh < self.min_size[0] or bsw < self.min_size[1]:
                continue

            coords = alive[level].nonzero(as_tuple=False)  # [K, 2]
            if coords.shape[0] == 0:
                continue

            r, c = coords[:, 0], coords[:, 1]
            x0, y0 = r * bsh, c * bsw
            x1, y1 = x0 + bsh, y0 + bsw

            col_offsets = torch.arange(bsw, device=device)
            top_row_idx = x0.unsqueeze(1).expand(-1, bsw)
            top_col_idx = (y0.unsqueeze(1) + col_offsets.unsqueeze(0)).clamp(0, W)
            out[top_row_idx.reshape(-1), top_col_idx.reshape(-1)] = True

            bot_row_idx = x1.unsqueeze(1).expand(-1, bsw)
            out[bot_row_idx.reshape(-1), top_col_idx.reshape(-1)] = True

            row_offsets = torch.arange(bsh, device=device)
            lft_col_idx = y0.unsqueeze(1).expand(-1, bsh)
            lft_row_idx = (x0.unsqueeze(1) + row_offsets.unsqueeze(0)).clamp(0, H)
            out[lft_row_idx.reshape(-1), lft_col_idx.reshape(-1)] = True

            rgt_col_idx = y1.unsqueeze(1).expand(-1, bsh)
            out[lft_row_idx.reshape(-1), rgt_col_idx.reshape(-1)] = True

        return out[:H, :W]

    def _labeled_to_region_tensors(self, labeled, num_features):
        """Vectorized bounding-box extraction for every labeled region at once.

        Per-label min/max coordinates via `scatter_reduce_` (segmented
        reduction, `O(H*W + num_features)` memory) -- not a `[num_
        features, H*W]` one-hot membership matrix (`GPUPatchify3D`'s own
        3D analog hit a real `torch.OutOfMemoryError` from exactly this
        pattern, `O(num_features * D*H*W)` memory exploding at real image
        scale -- silently fine on the small synthetic images every prior
        test used; this 2D version has the identical latent issue, fixed
        here too even though it hadn't yet been observed to OOM for real).

        Args:
            labeled: `[H, W]` long tensor, 0 = background.
            num_features: Number of non-zero labels.

        Returns:
            `RegionTensors`, all `[N]` fields on `labeled`'s device.
        """
        device = labeled.device
        H, W = labeled.shape

        if num_features == 0:
            empty = torch.zeros(0, device=device, dtype=torch.float32)
            return self.RegionTensors(
                empty, empty, empty, empty,
                torch.zeros(0, device=device, dtype=torch.long),
                torch.zeros(0, device=device, dtype=torch.long),
                empty, empty, 0,
            )

        flat = labeled.reshape(-1)

        row_idx = torch.arange(H, device=device).repeat_interleave(W)
        col_idx = torch.arange(W, device=device).repeat(H)
        INF = max(H, W) + 1

        def bounds(idx):
            idx_f = idx.float()
            lo = torch.full((num_features + 1,), float(INF), device=device, dtype=torch.float32)
            hi = torch.zeros(num_features + 1, device=device, dtype=torch.float32)
            lo.scatter_reduce_(0, flat, idx_f, reduce="amin", include_self=True)
            hi.scatter_reduce_(0, flat, idx_f, reduce="amax", include_self=True)
            return lo[1:], hi[1:]  # drop index 0 (background)

        rows_min, rows_max = bounds(row_idx)
        cols_min, cols_max = bounds(col_idx)

        x0s = torch.where(cols_min != 0, cols_min - 1, cols_min).float()
        x1s = cols_max.float()
        y0s = torch.where(rows_min != 0, rows_min - 1, rows_min).float()
        y1s = rows_max.float()

        phs = (y1s - y0s + 1).long()
        pws = (x1s - x0s + 1).long()
        cxs = (x0s + x1s) / 2.0
        cys = (y0s + y1s) / 2.0

        return self.RegionTensors(x0s, x1s, y0s, y1s, phs, pws, cxs, cys, num_features)

    def _detect_regions(self, border_mask):
        """Connected components of `border_mask`'s interior -> `RegionTensors`.

        Delegates to the shared `_label_regions` (module-level) -- `"scipy"`
        (default) does a real CPU round-trip; `"cupyx"` (opt-in via `self.
        region_backend`, deliberately not `kornia.contrib.connected_
        components`, which has no 3D equivalent) stays fully on-device.
        See `_label_regions`'s own docstring for what each means and what's
        verified.
        """
        labeled, num_features = _label_regions(border_mask, self.region_backend)
        return self._labeled_to_region_tensors(labeled, num_features)

    def run_merge_batch(self, batch_merge_costs, level_shapes, padded_hwc):
        """Merges every image in the batch down to exactly `fixed_length` leaves, in lockstep.

        Instead of B independent while-loops, maintains `alive` state for
        all B images in one set of `[B, Hp, Wp]` boolean tensors and runs a
        single global `topk` per iteration -- see this class's own
        docstring for why this is "level-parallel."

        Args:
            batch_merge_costs: `merge_costs` from `_compute_all_levels_batch`.
            level_shapes: `level_shapes` from `_compute_all_levels_batch`.
            padded_hwc: `[B, H', W', C]`, only used for per-image shape in
                the final region-detection step.

        Returns:
            `list[RegionTensors]`, length B.
        """
        B = padded_hwc.shape[0]
        device = padded_hwc.device

        alive = [torch.zeros(B, *shape, dtype=torch.bool, device=device) for shape in level_shapes]
        alive[0][:] = True

        leaves_remaining = torch.full((B,), alive[0][0].numel(), dtype=torch.long, device=device)
        end_leaves = self.fixed_length

        batch_ratio = 0.15
        min_batch = 4096
        max_batch = 500_000

        while (leaves_remaining > end_leaves).any():
            cand_costs, cand_img, cand_level, cand_row, cand_col = [], [], [], [], []

            for level in range(1, self.max_level + 1):
                Hp, Wp = level_shapes[level]
                child_view = alive[level - 1].reshape(B, Hp, 2, Wp, 2)
                can_merge = child_view.all(dim=(2, 4))  # [B, Hp, Wp]

                needs_merge = (leaves_remaining > end_leaves)
                can_merge = can_merge & needs_merge[:, None, None]
                if not can_merge.any():
                    continue

                b_idx, r_idx, c_idx = can_merge.nonzero(as_tuple=True)
                costs = batch_merge_costs[level][b_idx, r_idx, c_idx]

                cand_costs.append(costs)
                cand_img.append(b_idx)
                cand_level.append(torch.full_like(b_idx, level))
                cand_row.append(r_idx)
                cand_col.append(c_idx)

            if not cand_costs:
                break

            all_costs = torch.cat(cand_costs)
            all_img = torch.cat(cand_img)
            all_level = torch.cat(cand_level)
            all_row = torch.cat(cand_row)
            all_col = torch.cat(cand_col)

            max_needed = int((leaves_remaining - end_leaves).clamp(min=0).sum() // 3) + B
            k = int(min(max(min_batch, min(int(all_costs.shape[0] * batch_ratio), max_batch)), all_costs.shape[0]))
            k = max(k, max_needed)
            k = min(k, all_costs.shape[0])

            top_idx = torch.topk(all_costs, k, largest=False).indices
            sorted_idx = top_idx[all_costs[top_idx].argsort()]

            sel_img = all_img[sorted_idx]
            sel_level = all_level[sorted_idx]
            sel_row = all_row[sorted_idx]
            sel_col = all_col[sorted_idx]

            indicator = (sel_img.unsqueeze(0) == torch.arange(B, device=device).unsqueeze(1))  # [B, k]
            cum_merges = indicator.long().cumsum(dim=1)
            budget = ((leaves_remaining - end_leaves) // 3).clamp(min=0).unsqueeze(1)  # [B, 1]
            valid_mask = (cum_merges <= budget) & indicator  # [B, k]

            apply_mask = valid_mask.any(dim=0)  # [k]

            for lvl in range(1, self.max_level + 1):
                lvl_mask = apply_mask & (sel_level == lvl)
                if not lvl_mask.any():
                    continue

                b_sel, r_sel, c_sel = sel_img[lvl_mask], sel_row[lvl_mask], sel_col[lvl_mask]
                alive[lvl][b_sel, r_sel, c_sel] = True

                r2, c2 = r_sel * 2, c_sel * 2
                tgt = alive[lvl - 1]
                tgt[b_sel, r2, c2] = False
                tgt[b_sel, r2 + 1, c2] = False
                tgt[b_sel, r2, c2 + 1] = False
                tgt[b_sel, r2 + 1, c2 + 1] = False

            merges_per_img = valid_mask.sum(dim=1).long()
            leaves_remaining -= 3 * merges_per_img

        all_regions = []
        for b in range(B):
            alive_b = [alive[l][b] for l in range(len(alive))]
            border_mask = self._border_mask(alive_b, padded_hwc[b])
            all_regions.append(self._detect_regions(border_mask))
        return all_regions

    def _serialize_batch(self, all_regions, imgs_batch):
        """Extracts and resizes every region across the whole batch in one `grid_sample` call.

        Assumes every image in the batch has the same region count `N`
        (`all_regions[0].N`) -- guaranteed by `__init__`'s congruence
        assertion given every image shares `img_size`.

        Args:
            all_regions: `list[RegionTensors]`, length B.
            imgs_batch: `[B, C, H, W]` (padded).

        Returns:
            `[B, C, N, interp_size**2]` -- always keeps the channel axis
            (even when `C == 1`), since this is consumed directly by
            `forward()`'s caller (`VIT`/`MAE`/`UNETR.forward`'s non-varemb
            `rearrange('b c s p -> b s (p c)')` path expects a real `C`
            axis) with no later collation step to add it back, unlike
            `UCF_VIT.dataloaders.transform.Patchify.forward`'s own
            per-*sample* (no batch dim yet) `C == 1` squeeze.
        """
        B, C, H, W = imgs_batch.shape
        N = all_regions[0].N
        P = self.interp_size
        device = imgs_batch.device

        # x0s/x1s are column (W-axis) bounds, y0s/y1s are row (H-axis) bounds
        # (see _labeled_to_region_tensors's own row_idx/col_idx derivation) --
        # each must be normalized by its own axis's real extent and fed to
        # grid_sample's matching grid channel (grid[...,0]=x/width,
        # grid[...,1]=y/height), or this silently extracts the row/column
        # *transpose* of the real detected region on any non-square image
        # (caught directly by this module's own multi-quadrant tests).
        x0 = torch.stack([rt.x0s for rt in all_regions]).reshape(B * N)
        x1 = torch.stack([rt.x1s for rt in all_regions]).reshape(B * N)
        y0 = torch.stack([rt.y0s for rt in all_regions]).reshape(B * N)
        y1 = torch.stack([rt.y1s for rt in all_regions]).reshape(B * N)

        x0n, x1n = 2 * x0 / (W - 1) - 1, 2 * x1 / (W - 1) - 1
        y0n, y1n = 2 * y0 / (H - 1) - 1, 2 * y1 / (H - 1) - 1

        t = torch.linspace(0, 1, P, device=device).unsqueeze(0)  # [1, P]
        xs = x0n.unsqueeze(1) + (x1n - x0n).unsqueeze(1) * t  # [B*N, P]
        ys = y0n.unsqueeze(1) + (y1n - y0n).unsqueeze(1) * t  # [B*N, P]

        grid_y = ys.unsqueeze(2).expand(-1, -1, P)
        grid_x = xs.unsqueeze(1).expand(-1, P, -1)
        grids = torch.stack([grid_x, grid_y], dim=-1).float()  # [B*N, P, P, 2]

        imgs_rep = imgs_batch.float().repeat_interleave(N, dim=0)  # [B*N, C, H, W]
        patches = F.grid_sample(imgs_rep, grids, mode="bilinear", padding_mode="border", align_corners=True)
        patches = patches.reshape(B, N, C, P * P)

        return patches.permute(0, 2, 1, 3)

    def forward(self, img):
        """Adaptively patchifies `img` into a fixed-length sequence, entirely on `img`'s own device.

        Args:
            img: `[B, C, H, W]`.

        Returns:
            `(seq_img, seq_size, seq_pos)`: `seq_img` is `[B, C,
            fixed_length, interp_size**2]` -- see `_serialize_batch`'s own
            docstring for why the channel axis is always kept, unlike
            `UCF_VIT.dataloaders.transform.Patchify.forward`'s per-sample
            `C == 1` squeeze. `seq_size` is `[B, fixed_length]`
            (each region's side length along the taller axis -- regions
            aren't always square when `min_size` isn't, unlike
            `FixedQuadTree`'s leaves). `seq_pos` is `[B, fixed_length, 2]`
            (`x, y` center coordinates, matching `Rect.get_center`'s own
            `(x, y)` convention in `quadtree.py`).
        """
        padded_img, _ = self._pad_tensor(img)
        merge_costs, level_shapes = self._compute_all_levels_batch(padded_img.float())
        padded_hwc = padded_img.permute(0, 2, 3, 1)

        all_regions = self.run_merge_batch(merge_costs, level_shapes, padded_hwc)
        seq_img = self._serialize_batch(all_regions, padded_img.float()).to(img.dtype)

        seq_size = torch.stack([rt.phs for rt in all_regions]).float()  # [B, N]
        seq_pos = torch.stack([torch.stack([rt.cxs, rt.cys], dim=-1) for rt in all_regions])  # [B, N, 2]

        return seq_img, seq_size, seq_pos


class GPUPatchify3D(torch.nn.Module):
    """GPU-native, level-parallel adaptive patchification for 3D volumes.

    3D generalization of `GPUPatchify2D` -- mirrors `UCF_VIT.dataloaders.
    transform`'s own `Patchify`/`Patchify_3D` convention (a separate class
    per dimensionality, not a unified N-D class), so `GPUPatchify2D` itself
    is untouched. Every design decision `GPUPatchify2D`'s own docstring and
    `tests/README.md` entries describe (level-parallel bottom-up merge,
    pluggable `score_fn`, why the merge-cost formula differs between
    `"variance"` and `"canny"`) applies unchanged here; only the mechanics
    generalize: 8 children merge into 1 parent (not 4), block "borders"
    become block "faces" (2D faces on a 3D boundary volume, not 1D lines),
    Canny's non-max suppression discretizes gradient direction into 13
    canonical directions (the antipodal neighbor-direction pairs of a
    3x3x3 voxel neighborhood, not 2D's 4), and `_serialize_batch` uses a
    5D (volumetric) `grid_sample` call instead of 4D.

    Not yet wired into any model (`ap.do_gpu_ap` currently asserts `twoD`
    by construction in `arch.py`) -- see this module's own tests/README.md
    entry for what's deferred.

    Same "expect similar, not identical" caveat `GPUPatchify2D`'s own
    docstring gives against `Patchify`, here against `UCF_VIT.dataloaders.
    transform.Patchify_3D`: `"canny"` is a from-scratch torch/GPU
    reimplementation, not a call into `Patchify_3D`'s own
    `SimpleITK.CannyEdgeDetection` (a CPU-only, non-batched library call
    that can't run inside this on-device `forward()`) -- differs in
    smoothing (an explicit Gaussian sigma here vs. SimpleITK's internal
    `variance` parameter, not numerically equivalent) and in hysteresis
    (`canny_hysteresis_iters` binary-dilation passes here vs. SimpleITK's
    own true connectivity-based hysteresis). `"variance"` uses the same
    `torch.var` (`n-1`/Bessel-corrected) vs. `np.var` (`n`, the CPU side's
    default) denominator difference `GPUPatchify2D`'s docstring describes,
    largest for small (near-`min_size`) blocks.
    """

    RegionTensors3D = namedtuple(
        "RegionTensors3D",
        ["x0s", "x1s", "y0s", "y1s", "z0s", "z1s", "phs", "pws", "pds", "cxs", "cys", "czs", "N"],
    )

    def __init__(
            self, img_size, fixed_length=344, interp_size=16, min_size=2,
            score_fn="variance", canny_sigma=1.0, canny_low_threshold=0.1,
            canny_high_threshold=0.2, canny_hysteresis_iters=2,
            region_backend="scipy",
    ):
        """Precomputes the level structure this volume size implies.

        Args:
            img_size: `(D, H, W)` of the (unpadded) input. Requires
                `D <= H <= W` (extends `GPUPatchify2D`'s own `H <= W`
                requirement -- the reference implementation's min_size-
                ratio trick, generalized to two ratios, `H/D` and `W/D`,
                each truncated to `int` exactly like `GPUPatchify2D`'s own
                single ratio already is).
            fixed_length: Target number of leaf regions to merge down to.
                Must satisfy `(real_max_blocks**3 - fixed_length) % 7 == 0`
                (every merge removes exactly 7 leaves -- 8 children into 1
                parent -- matching `FixedOctTree`'s own modulus; see
                `GPUPatchify2D.__init__`'s identical `% 3` assertion for
                the analogous 2D reasoning, and `_serialize_batch`'s own
                docstring for why every image in a batch must land on the
                same leaf count).
            interp_size: Side length each (cubic) leaf region is resized to.
            min_size: Finest block side length (level 0), along the
                shortest volume axis (`D`).
            score_fn: `"variance"` (default) or `"canny"` -- see
                `GPUPatchify2D`'s own `score_fn` docstring entry; identical
                reasoning, generalized to 3 axes.
            canny_sigma: Gaussian smoothing sigma applied before computing
                gradients. Only used when `score_fn == "canny"`.
            canny_low_threshold: Lower ("weak" edge) gradient-magnitude
                threshold. Only used when `score_fn == "canny"`.
            canny_high_threshold: Upper ("strong" edge) gradient-magnitude
                threshold. Only used when `score_fn == "canny"`.
            canny_hysteresis_iters: Number of binary-dilation passes
                approximating hysteresis edge-linking. Only used when
                `score_fn == "canny"`.
            region_backend: `"scipy"` (default) or `"cupyx"` -- see
                `_label_regions`'s own docstring for what each means.
        """
        super().__init__()
        self.interp_size = interp_size
        self.fixed_length = fixed_length

        assert score_fn in ("variance", "canny"), f"score_fn must be 'variance' or 'canny', got {score_fn!r}"
        self.score_fn = score_fn
        self.canny_sigma = canny_sigma
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.canny_hysteresis_iters = canny_hysteresis_iters
        self.region_backend = region_backend

        D, H, W = img_size
        if not (D <= H <= W):
            raise NotImplementedError(
                f"GPUPatchify3D requires img_size sorted D <= H <= W, got {img_size} "
                "-- extends GPUPatchify2D's own H <= W requirement (its min_size-ratio "
                "trick, which keeps every axis at the same block count per level, only "
                "derives an integer ratio from the shortest axis)."
            )
        ratio_h = int(H / D)
        ratio_w = int(W / D)
        max_blocks = D // min_size
        self.min_size = [min_size, min_size * ratio_h, min_size * ratio_w]

        self.max_level = int(math.floor(math.log2(max_blocks))) if max_blocks >= 1 else 0

        # See GPUPatchify2D.__init__'s identical comment: initial_leaves
        # must reflect the *padded* grid run_merge_batch actually operates
        # on, not the raw img_size // min_size above.
        pad_bd = self.min_size[0] * (2 ** self.max_level)
        padded_d = -(-img_size[0] // pad_bd) * pad_bd
        real_max_blocks = padded_d // self.min_size[0]

        initial_leaves = real_max_blocks ** 3
        assert (initial_leaves - fixed_length) % 7 == 0, (
            f"fixed_length ({fixed_length}) must satisfy (real_max_blocks**3 - fixed_length) % 7 == 0 "
            f"-- real_max_blocks**3 is {initial_leaves} here (img_size {img_size} padded to a multiple "
            f"of {pad_bd}, min_size {min_size}), so every image's merge loop can land on exactly "
            "fixed_length leaves (each merge removes exactly 7 leaves). Off-target leaf counts would "
            "either silently break _serialize_batch's whole-batch grid_sample call (which assumes "
            "every image in the batch has the same region count), or -- if unreachable exactly -- "
            "make run_merge_batch's per-image budget hit 0 forever, hanging."
        )
        self.initial_leaves = initial_leaves

    def _pad_size(self):
        """The block size `(bd, bh, bw)` the padded volume's D/H/W must be divisible by."""
        return tuple(s * (2 ** self.max_level) for s in self.min_size)

    def _compute_padding(self, D, H, W):
        bd, bh, bw = self._pad_size()
        return (bd - (D % bd)) % bd, (bh - (H % bh)) % bh, (bw - (W % bw)) % bw

    def _pad_tensor(self, t):
        """Edge-replication-pads a `[..., D, H, W]` tensor so D/H/W divide evenly by the coarsest block size."""
        D, H, W = t.shape[-3], t.shape[-2], t.shape[-1]
        pad_d, pad_h, pad_w = self._compute_padding(D, H, W)
        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return t, (D, H, W)
        return F.pad(t, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate"), (D, H, W)

    def _compute_level_stats_batch(self, block_size, imgs):
        """Per-block variance-based error (SSE) at one level, for the whole batch.

        3D analog of `GPUPatchify2D`'s own method of the same name --
        identical reasoning, one more spatial axis.

        Args:
            block_size: `(bsd, bsh, bsw)` block side lengths at this level.
            imgs: `[B, C, D, H, W]`.

        Returns:
            `[B, gd, gh, gw]`.
        """
        B, C, Dp, Hp, Wp = imgs.shape
        bsd, bsh, bsw = block_size
        gd, gh, gw = Dp // bsd, Hp // bsh, Wp // bsw

        reshaped = imgs.reshape(B, C, gd, bsd, gh, bsh, gw, bsw).permute(0, 1, 2, 4, 6, 3, 5, 7)  # [B,C,gd,gh,gw,bsd,bsh,bsw]
        vars_ = reshaped.var(dim=(-3, -2, -1))  # [B,C,gd,gh,gw]
        return vars_.sum(dim=1) * (bsd * bsh * bsw)  # [B,gd,gh,gw]

    def _box_sum_batch(self, block_size, base_map):
        """Sum of `base_map` within each non-overlapping `block_size` block, for the whole batch.

        3D analog of `GPUPatchify2D`'s own method of the same name.

        Args:
            block_size: `(bsd, bsh, bsw)` block side lengths at this level.
            base_map: `[B, D, H, W]`.

        Returns:
            `[B, gd, gh, gw]`.
        """
        B, Dp, Hp, Wp = base_map.shape
        bsd, bsh, bsw = block_size
        gd, gh, gw = Dp // bsd, Hp // bsh, Wp // bsw

        reshaped = base_map.reshape(B, gd, bsd, gh, bsh, gw, bsw).permute(0, 1, 3, 5, 2, 4, 6)  # [B,gd,gh,gw,bsd,bsh,bsw]
        return reshaped.sum(dim=(-3, -2, -1))  # [B,gd,gh,gw]

    def _gaussian_kernel3d(self, sigma, device, dtype):
        """Builds a fixed, normalized 3D Gaussian kernel for pre-gradient smoothing.

        Args:
            sigma: Gaussian standard deviation, in voxels.
            device: Device the kernel should live on.
            dtype: Dtype the kernel should be.

        Returns:
            `[1, 1, K, K, K]` kernel, `K = 2 * round(3 * sigma) + 1`.
        """
        radius = max(1, int(round(3 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel3d = g1d.view(-1, 1, 1) * g1d.view(1, -1, 1) * g1d.view(1, 1, -1)  # [K, K, K]
        return kernel3d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K, K]

    def _sobel_gradients_batch(self, imgs):
        """Depth/height/width Sobel gradients, for the whole batch.

        Each 3x3x3 kernel is separable: a `[-1,0,1]` derivative along its
        own axis, `[1,2,1]` smoothing along the other two -- the direct 3D
        generalization of `GPUPatchify2D`'s own 2D Sobel kernels (each an
        outer product of a `[-1,0,1]` derivative and a `[1,2,1]` smoothing
        vector).

        Args:
            imgs: `[B, 1, D, H, W]`.

        Returns:
            `(gd, gh, gw)`, each `[B, 1, D, H, W]`.
        """
        device, dtype = imgs.device, imgs.dtype
        deriv = torch.tensor([-1., 0., 1.], device=device, dtype=dtype)
        smooth = torch.tensor([1., 2., 1.], device=device, dtype=dtype)

        kd = (deriv.view(3, 1, 1) * smooth.view(1, 3, 1) * smooth.view(1, 1, 3)).view(1, 1, 3, 3, 3)
        kh = (smooth.view(3, 1, 1) * deriv.view(1, 3, 1) * smooth.view(1, 1, 3)).view(1, 1, 3, 3, 3)
        kw = (smooth.view(3, 1, 1) * smooth.view(1, 3, 1) * deriv.view(1, 1, 3)).view(1, 1, 3, 3, 3)

        padded = F.pad(imgs, (1, 1, 1, 1, 1, 1), mode="reflect")
        gd = F.conv3d(padded, kd)
        gh = F.conv3d(padded, kh)
        gw = F.conv3d(padded, kw)
        return gd, gh, gw

    def _non_max_suppression_batch(self, mag, gvec):
        """Suppresses every gradient-magnitude voxel that isn't a local max along its own gradient direction.

        3D generalization of `GPUPatchify2D`'s own method: direction is
        discretized into the 13 canonical directions of a 3x3x3 voxel
        neighborhood (the 13 antipodal pairs among its 26 non-zero
        integer offsets), picked via cosine-similarity argmax against the
        gradient vector (`abs()` of the cosine, since a direction and its
        negation are the same axis -- the 3D analog of 2D's `angle_deg %
        180` removing sign ambiguity before bucketing), then compared
        against that direction's own antipodal voxel-shift pair -- direct
        generalization of 2D's 4-entry `neighbor_shifts` dict to 13
        entries.

        Args:
            mag: Gradient magnitude, `[B, 1, D, H, W]`.
            gvec: `(gd, gh, gw)`, the raw (unnormalized) gradient
                components, each `[B, 1, D, H, W]`.

        Returns:
            `[B, 1, D, H, W]`, `mag` with every non-local-max voxel zeroed.
        """
        D, H, W = mag.shape[-3:]

        def shift(t, dd, dh, dw):
            padded = F.pad(t, (1, 1, 1, 1, 1, 1), mode="replicate")
            return padded[..., 1 + dd:1 + dd + D, 1 + dh:1 + dh + H, 1 + dw:1 + dw + W]

        raw_dirs = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    if (a, b, c) == (0, 0, 0):
                        continue
                    if a > 0 or (a == 0 and b > 0) or (a == 0 and b == 0 and c > 0):
                        raw_dirs.append((a, b, c))
        assert len(raw_dirs) == 13

        gd, gh, gw = gvec
        norm = torch.sqrt(gd ** 2 + gh ** 2 + gw ** 2 + 1e-12)
        gd_n, gh_n, gw_n = gd / norm, gh / norm, gw / norm

        dots = []
        for (a, b, c) in raw_dirs:
            dnorm = math.sqrt(a * a + b * b + c * c)
            dots.append(((gd_n * a + gh_n * b + gw_n * c) / dnorm).abs())
        bucket = torch.stack(dots, dim=0).argmax(dim=0)  # [B,1,D,H,W]

        n1 = torch.zeros_like(mag)
        n2 = torch.zeros_like(mag)
        for i, (a, b, c) in enumerate(raw_dirs):
            is_bucket = (bucket == i)
            n1 = torch.where(is_bucket, shift(mag, a, b, c), n1)
            n2 = torch.where(is_bucket, shift(mag, -a, -b, -c), n2)

        is_local_max = (mag >= n1) & (mag >= n2)
        return mag * is_local_max.to(mag.dtype)

    def _canny_edge_map_batch(self, imgs):
        """Per-channel Canny edge detection, summed into a per-voxel edge count.

        3D analog of `GPUPatchify2D`'s own method -- identical pipeline
        and hysteresis-by-dilation approximation (via `F.max_pool3d`
        instead of `F.max_pool2d`), one more spatial axis throughout.

        Args:
            imgs: `[B, C, D, H, W]`.

        Returns:
            `[B, D, H, W]` float tensor, values in `[0, C]`.
        """
        B, C, D, H, W = imgs.shape
        device, dtype = imgs.device, imgs.dtype
        kernel = self._gaussian_kernel3d(self.canny_sigma, device, dtype)
        pad = kernel.shape[-1] // 2

        edge_count = torch.zeros(B, D, H, W, device=device, dtype=dtype)
        for c in range(C):
            channel = imgs[:, c:c + 1]  # [B, 1, D, H, W]
            blurred = F.conv3d(F.pad(channel, (pad, pad, pad, pad, pad, pad), mode="reflect"), kernel)

            gd, gh, gw = self._sobel_gradients_batch(blurred)
            mag = torch.sqrt(gd ** 2 + gh ** 2 + gw ** 2 + 1e-12)

            nms = self._non_max_suppression_batch(mag, (gd, gh, gw))
            strong = nms > self.canny_high_threshold
            weak = (nms > self.canny_low_threshold) & (~strong)

            edge = strong.clone()
            for _ in range(self.canny_hysteresis_iters):
                dilated = F.max_pool3d(edge.to(dtype), kernel_size=3, stride=1, padding=1) > 0
                edge = edge | (weak & dilated)

            edge_count = edge_count + edge.to(dtype).squeeze(1)

        return edge_count

    def _compute_all_levels_batch(self, imgs):
        """Per-level block errors and the merge cost of collapsing each level's blocks into their level-below parent.

        3D analog of `GPUPatchify2D`'s own method -- identical `score_fn`
        dispatch and merge-cost formulas (see that method's own docstring
        for the full reasoning), 3-axis block sizes, and an 8-way children
        sum (not 4-way) for variance's between-group subtraction.

        Args:
            imgs: `[B, C, D, H, W]`.

        Returns:
            `(merge_costs, level_shapes)`: `merge_costs[level]` is
            `[B, gd, gh, gw]` (`None` at level 0), `level_shapes[level]` is
            `(gd, gh, gw)`.
        """
        errors = []
        level_shapes = []
        merge_costs = [None] * (self.max_level + 1)

        if self.score_fn == "canny":
            edge_map = self._canny_edge_map_batch(imgs)  # [B, D, H, W]
            for l in range(self.max_level + 1):
                bs = tuple(s * (2 ** l) for s in self.min_size)
                err_l = self._box_sum_batch(bs, edge_map)
                errors.append(err_l)
                level_shapes.append(tuple(err_l.shape[1:]))

            for level in range(1, self.max_level + 1):
                merge_costs[level] = errors[level]

            return merge_costs, level_shapes

        for l in range(self.max_level + 1):
            bs = tuple(s * (2 ** l) for s in self.min_size)
            sse_l = self._compute_level_stats_batch(bs, imgs)
            errors.append(sse_l)
            level_shapes.append(tuple(sse_l.shape[1:]))

        for level in range(1, self.max_level + 1):
            child_err = errors[level - 1]  # [B, Dp2, Hp2, Wp2]
            Dp, Hp, Wp = child_err.shape[1] // 2, child_err.shape[2] // 2, child_err.shape[3] // 2
            sum_children = child_err.reshape(-1, Dp, 2, Hp, 2, Wp, 2).sum(dim=(2, 4, 6))  # [B,Dp,Hp,Wp]
            merge_costs[level] = errors[level] - sum_children

        return merge_costs, level_shapes

    def _face_mask(self, alive, img):
        """Marks the voxel-grid faces around every currently-alive block.

        3D analog of `GPUPatchify2D`'s own `_border_mask`: every alive
        block independently stamps all 6 of its own faces (2D planes, not
        1D lines) via the same vectorized-indexing style (`arange` +
        `expand` + advanced indexing, no Python loop over blocks), so two
        adjacent alive blocks (even same-size ones) always get a face
        stamped between them.

        Args:
            alive: List (one per level) of `[Dp, Hp, Wp]` bool tensors,
                one volume's worth.
            img: `[Dp, Hp, Wp, C]`, only used for its shape.

        Returns:
            `[D, H, W]` bool tensor, `True` on the face between (or at the
            outer face of) alive blocks.
        """
        D, H, W = img.shape[0], img.shape[1], img.shape[2]
        device = img.device
        out = torch.zeros(D + 1, H + 1, W + 1, dtype=torch.bool, device=device)

        for level in reversed(range(len(alive))):
            bsd = self.min_size[0] * (2 ** level)
            bsh = self.min_size[1] * (2 ** level)
            bsw = self.min_size[2] * (2 ** level)
            if bsd < self.min_size[0] or bsh < self.min_size[1] or bsw < self.min_size[2]:
                continue

            coords = alive[level].nonzero(as_tuple=False)  # [K, 3]
            if coords.shape[0] == 0:
                continue

            di, hi, wi = coords[:, 0], coords[:, 1], coords[:, 2]
            d0, h0, w0 = di * bsd, hi * bsh, wi * bsw
            d1, h1, w1 = d0 + bsd, h0 + bsh, w0 + bsw

            d_off = torch.arange(bsd, device=device)
            h_off = torch.arange(bsh, device=device)
            w_off = torch.arange(bsw, device=device)

            dd = (d0.unsqueeze(1) + d_off.unsqueeze(0)).clamp(0, D)  # [K, bsd]
            hh = (h0.unsqueeze(1) + h_off.unsqueeze(0)).clamp(0, H)  # [K, bsh]
            ww = (w0.unsqueeze(1) + w_off.unsqueeze(0)).clamp(0, W)  # [K, bsw]

            # front/back faces (perpendicular to D): fixed d, spans (h,w)
            hh_hw = hh.unsqueeze(2).expand(-1, -1, bsw).reshape(-1)
            ww_hw = ww.unsqueeze(1).expand(-1, bsh, -1).reshape(-1)
            d0_hw = d0.view(-1, 1, 1).expand(-1, bsh, bsw).reshape(-1)
            d1_hw = d1.view(-1, 1, 1).expand(-1, bsh, bsw).reshape(-1)
            out[d0_hw, hh_hw, ww_hw] = True
            out[d1_hw, hh_hw, ww_hw] = True

            # top/bottom faces (perpendicular to H): fixed h, spans (d,w)
            dd_dw = dd.unsqueeze(2).expand(-1, -1, bsw).reshape(-1)
            ww_dw = ww.unsqueeze(1).expand(-1, bsd, -1).reshape(-1)
            h0_dw = h0.view(-1, 1, 1).expand(-1, bsd, bsw).reshape(-1)
            h1_dw = h1.view(-1, 1, 1).expand(-1, bsd, bsw).reshape(-1)
            out[dd_dw, h0_dw, ww_dw] = True
            out[dd_dw, h1_dw, ww_dw] = True

            # left/right faces (perpendicular to W): fixed w, spans (d,h)
            dd_dh = dd.unsqueeze(2).expand(-1, -1, bsh).reshape(-1)
            hh_dh = hh.unsqueeze(1).expand(-1, bsd, -1).reshape(-1)
            w0_dh = w0.view(-1, 1, 1).expand(-1, bsd, bsh).reshape(-1)
            w1_dh = w1.view(-1, 1, 1).expand(-1, bsd, bsh).reshape(-1)
            out[dd_dh, hh_dh, w0_dh] = True
            out[dd_dh, hh_dh, w1_dh] = True

        return out[:D, :H, :W]

    def _labeled_to_region_tensors(self, labeled, num_features):
        """Vectorized bounding-box extraction for every labeled region at once.

        3D analog of `GPUPatchify2D`'s own method -- `x`/`y`/`z` map to
        `W`/`H`/`D` respectively (extending `GPUPatchify2D`'s own `x=W,
        y=H` convention), matching `F.grid_sample`'s own confirmed 5D grid
        axis order (`(x,y,z)` -> `(W,H,D)`) so `_serialize_batch` can use
        these bounds directly with no axis reordering.

        Per-label min/max coordinates via `scatter_reduce_` (segmented
        reduction, `O(D*H*W + num_features)` memory) -- not a `[num_
        features, D*H*W]` one-hot membership matrix (a real bug this
        replaced: `torch.OutOfMemoryError: HIP out of memory. Tried to
        allocate 64.00 GiB`, hit on a real Frontier run of `basic_ct/
        unetr`'s do_gpu_ap:True cell, `O(num_features * D*H*W)` memory
        exploding at real image scale -- silently fine on the small
        synthetic volumes every prior test used, catastrophic on a real
        256^3 volume's finest level, where both factors are large).

        Args:
            labeled: `[D, H, W]` long tensor, 0 = background.
            num_features: Number of non-zero labels.

        Returns:
            `RegionTensors3D`, all `[N]` fields on `labeled`'s device.
        """
        device = labeled.device
        D, H, W = labeled.shape

        if num_features == 0:
            empty = torch.zeros(0, device=device, dtype=torch.float32)
            empty_long = torch.zeros(0, device=device, dtype=torch.long)
            return self.RegionTensors3D(
                empty, empty, empty, empty, empty, empty,
                empty_long, empty_long, empty_long,
                empty, empty, empty, 0,
            )

        flat = labeled.reshape(-1)

        depth_idx = torch.arange(D, device=device).repeat_interleave(H * W)
        row_idx = torch.arange(H, device=device).repeat_interleave(W).repeat(D)
        col_idx = torch.arange(W, device=device).repeat(D * H)
        INF = max(D, H, W) + 1

        def bounds(idx):
            idx_f = idx.float()
            lo = torch.full((num_features + 1,), float(INF), device=device, dtype=torch.float32)
            hi = torch.zeros(num_features + 1, device=device, dtype=torch.float32)
            lo.scatter_reduce_(0, flat, idx_f, reduce="amin", include_self=True)
            hi.scatter_reduce_(0, flat, idx_f, reduce="amax", include_self=True)
            return lo[1:], hi[1:]  # drop index 0 (background)

        depth_min, depth_max = bounds(depth_idx)
        rows_min, rows_max = bounds(row_idx)
        cols_min, cols_max = bounds(col_idx)

        x0s = torch.where(cols_min != 0, cols_min - 1, cols_min).float()
        x1s = cols_max.float()
        y0s = torch.where(rows_min != 0, rows_min - 1, rows_min).float()
        y1s = rows_max.float()
        z0s = torch.where(depth_min != 0, depth_min - 1, depth_min).float()
        z1s = depth_max.float()

        phs = (y1s - y0s + 1).long()
        pws = (x1s - x0s + 1).long()
        pds = (z1s - z0s + 1).long()
        cxs = (x0s + x1s) / 2.0
        cys = (y0s + y1s) / 2.0
        czs = (z0s + z1s) / 2.0

        return self.RegionTensors3D(x0s, x1s, y0s, y1s, z0s, z1s, phs, pws, pds, cxs, cys, czs, num_features)

    def _detect_regions(self, face_mask):
        """Connected components of `face_mask`'s interior -> `RegionTensors3D`.

        Delegates to the shared `_label_regions` (module-level) -- see
        `GPUPatchify2D._detect_regions`'s identical docstring.
        """
        labeled, num_features = _label_regions(face_mask, self.region_backend)
        return self._labeled_to_region_tensors(labeled, num_features)

    def run_merge_batch(self, batch_merge_costs, level_shapes, padded_dhwc):
        """Merges every volume in the batch down to exactly `fixed_length` leaves, in lockstep.

        3D analog of `GPUPatchify2D`'s own method -- identical level-
        parallel design (see that method's own docstring), 8-way children
        (not 4-way): each merge removes exactly 7 leaves, so budget/
        leaves-remaining bookkeeping divides by 7 (not 3).

        Args:
            batch_merge_costs: `merge_costs` from `_compute_all_levels_batch`.
            level_shapes: `level_shapes` from `_compute_all_levels_batch`.
            padded_dhwc: `[B, D', H', W', C]`, only used for per-image shape
                in the final region-detection step.

        Returns:
            `list[RegionTensors3D]`, length B.
        """
        B = padded_dhwc.shape[0]
        device = padded_dhwc.device

        alive = [torch.zeros(B, *shape, dtype=torch.bool, device=device) for shape in level_shapes]
        alive[0][:] = True

        leaves_remaining = torch.full((B,), alive[0][0].numel(), dtype=torch.long, device=device)
        end_leaves = self.fixed_length

        batch_ratio = 0.15
        min_batch = 4096
        max_batch = 500_000

        while (leaves_remaining > end_leaves).any():
            cand_costs, cand_img, cand_level, cand_d, cand_h, cand_w = [], [], [], [], [], []

            for level in range(1, self.max_level + 1):
                Dp, Hp, Wp = level_shapes[level]
                child_view = alive[level - 1].reshape(B, Dp, 2, Hp, 2, Wp, 2)
                can_merge = child_view.all(dim=(2, 4, 6))  # [B, Dp, Hp, Wp]

                needs_merge = (leaves_remaining > end_leaves)
                can_merge = can_merge & needs_merge[:, None, None, None]
                if not can_merge.any():
                    continue

                b_idx, d_idx, h_idx, w_idx = can_merge.nonzero(as_tuple=True)
                costs = batch_merge_costs[level][b_idx, d_idx, h_idx, w_idx]

                cand_costs.append(costs)
                cand_img.append(b_idx)
                cand_level.append(torch.full_like(b_idx, level))
                cand_d.append(d_idx)
                cand_h.append(h_idx)
                cand_w.append(w_idx)

            if not cand_costs:
                break

            all_costs = torch.cat(cand_costs)
            all_img = torch.cat(cand_img)
            all_level = torch.cat(cand_level)
            all_d = torch.cat(cand_d)
            all_h = torch.cat(cand_h)
            all_w = torch.cat(cand_w)

            max_needed = int((leaves_remaining - end_leaves).clamp(min=0).sum() // 7) + B
            k = int(min(max(min_batch, min(int(all_costs.shape[0] * batch_ratio), max_batch)), all_costs.shape[0]))
            k = max(k, max_needed)
            k = min(k, all_costs.shape[0])

            top_idx = torch.topk(all_costs, k, largest=False).indices
            sorted_idx = top_idx[all_costs[top_idx].argsort()]

            sel_img = all_img[sorted_idx]
            sel_level = all_level[sorted_idx]
            sel_d = all_d[sorted_idx]
            sel_h = all_h[sorted_idx]
            sel_w = all_w[sorted_idx]

            indicator = (sel_img.unsqueeze(0) == torch.arange(B, device=device).unsqueeze(1))  # [B, k]
            cum_merges = indicator.long().cumsum(dim=1)
            budget = ((leaves_remaining - end_leaves) // 7).clamp(min=0).unsqueeze(1)  # [B, 1]
            valid_mask = (cum_merges <= budget) & indicator  # [B, k]

            apply_mask = valid_mask.any(dim=0)  # [k]

            for lvl in range(1, self.max_level + 1):
                lvl_mask = apply_mask & (sel_level == lvl)
                if not lvl_mask.any():
                    continue

                b_sel, d_sel, h_sel, w_sel = sel_img[lvl_mask], sel_d[lvl_mask], sel_h[lvl_mask], sel_w[lvl_mask]
                alive[lvl][b_sel, d_sel, h_sel, w_sel] = True

                d2, h2, w2 = d_sel * 2, h_sel * 2, w_sel * 2
                tgt = alive[lvl - 1]
                for dd_ in (0, 1):
                    for dh_ in (0, 1):
                        for dw_ in (0, 1):
                            tgt[b_sel, d2 + dd_, h2 + dh_, w2 + dw_] = False

            merges_per_img = valid_mask.sum(dim=1).long()
            leaves_remaining -= 7 * merges_per_img

        all_regions = []
        for b in range(B):
            alive_b = [alive[l][b] for l in range(len(alive))]
            face_mask = self._face_mask(alive_b, padded_dhwc[b])
            all_regions.append(self._detect_regions(face_mask))
        return all_regions

    def _serialize_batch(self, all_regions, imgs_batch):
        """Extracts and resizes every region across the whole batch in one 5D `grid_sample` call.

        3D analog of `GPUPatchify2D`'s own method -- assumes every volume
        in the batch has the same region count `N` (see that method's own
        docstring for why).

        Args:
            all_regions: `list[RegionTensors3D]`, length B.
            imgs_batch: `[B, C, D, H, W]` (padded).

        Returns:
            `[B, C, N, interp_size**3]`.
        """
        B, C, D, H, W = imgs_batch.shape
        N = all_regions[0].N
        P = self.interp_size
        device = imgs_batch.device

        # x0s/x1s: W-axis bounds; y0s/y1s: H-axis bounds; z0s/z1s: D-axis
        # bounds -- matches F.grid_sample's own confirmed 5D grid axis
        # order ((x,y,z) -> (W,H,D)), see _labeled_to_region_tensors's own
        # docstring. Getting this wrong is exactly the axis-transpose bug
        # GPUPatchify2D's own _serialize_batch already found and fixed once.
        x0 = torch.stack([rt.x0s for rt in all_regions]).reshape(B * N)
        x1 = torch.stack([rt.x1s for rt in all_regions]).reshape(B * N)
        y0 = torch.stack([rt.y0s for rt in all_regions]).reshape(B * N)
        y1 = torch.stack([rt.y1s for rt in all_regions]).reshape(B * N)
        z0 = torch.stack([rt.z0s for rt in all_regions]).reshape(B * N)
        z1 = torch.stack([rt.z1s for rt in all_regions]).reshape(B * N)

        x0n, x1n = 2 * x0 / (W - 1) - 1, 2 * x1 / (W - 1) - 1
        y0n, y1n = 2 * y0 / (H - 1) - 1, 2 * y1 / (H - 1) - 1
        z0n, z1n = 2 * z0 / (D - 1) - 1, 2 * z1 / (D - 1) - 1

        t = torch.linspace(0, 1, P, device=device).unsqueeze(0)  # [1, P]
        xs = x0n.unsqueeze(1) + (x1n - x0n).unsqueeze(1) * t  # [B*N, P]
        ys = y0n.unsqueeze(1) + (y1n - y0n).unsqueeze(1) * t
        zs = z0n.unsqueeze(1) + (z1n - z0n).unsqueeze(1) * t

        grid_x = xs.view(-1, 1, 1, P).expand(-1, P, P, P)
        grid_y = ys.view(-1, 1, P, 1).expand(-1, P, P, P)
        grid_z = zs.view(-1, P, 1, 1).expand(-1, P, P, P)
        grids = torch.stack([grid_x, grid_y, grid_z], dim=-1).float()  # [B*N, P, P, P, 3]

        imgs_rep = imgs_batch.float().repeat_interleave(N, dim=0)  # [B*N, C, D, H, W]
        patches = F.grid_sample(imgs_rep, grids, mode="bilinear", padding_mode="border", align_corners=True)
        patches = patches.reshape(B, N, C, P * P * P)

        return patches.permute(0, 2, 1, 3)

    def forward(self, img):
        """Adaptively patchifies `img` into a fixed-length sequence, entirely on `img`'s own device.

        Args:
            img: `[B, C, D, H, W]`.

        Returns:
            `(seq_img, seq_size, seq_pos)`: `seq_img` is `[B, C,
            fixed_length, interp_size**3]`. `seq_size` is `[B,
            fixed_length]` (each region's side length along the `D` axis
            -- the shortest, unscaled reference axis, matching
            `GPUPatchify2D`'s own choice of the analogous unscaled axis).
            `seq_pos` is `[B, fixed_length, 3]` (`x, y, z` center
            coordinates, matching `GPUPatchify2D`'s own `(x, y)`
            convention extended with `z` = depth center).
        """
        padded_img, _ = self._pad_tensor(img)
        merge_costs, level_shapes = self._compute_all_levels_batch(padded_img.float())
        padded_dhwc = padded_img.permute(0, 2, 3, 4, 1)

        all_regions = self.run_merge_batch(merge_costs, level_shapes, padded_dhwc)
        seq_img = self._serialize_batch(all_regions, padded_img.float()).to(img.dtype)

        seq_size = torch.stack([rt.pds for rt in all_regions]).float()  # [B, N]
        seq_pos = torch.stack([torch.stack([rt.cxs, rt.cys, rt.czs], dim=-1) for rt in all_regions])  # [B, N, 3]

        return seq_img, seq_size, seq_pos
