import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label


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
    only for now -- see this module's own tests/README.md entry for the
    scope this port deliberately left out (3D, a pluggable Canny-based
    scoring alternative, and a fully-GPU connected-components backend) and
    why.
    """

    # Bounding-box + scalar fields describing each detected leaf region,
    # vectorized over all N regions in one image at once (no Python loop
    # over regions).
    RegionTensors = namedtuple(
        "RegionTensors", ["x0s", "x1s", "y0s", "y1s", "phs", "pws", "cxs", "cys", "N"]
    )

    def __init__(self, img_size, fixed_length=196, interp_size=16, min_size=2):
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
        """
        super().__init__()
        self.interp_size = interp_size
        self.fixed_length = fixed_length

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

        initial_leaves = max_blocks ** 2
        assert (initial_leaves - fixed_length) % 3 == 0, (
            f"fixed_length ({fixed_length}) must satisfy (max_blocks**2 - fixed_length) % 3 == 0 "
            f"-- max_blocks**2 is {initial_leaves} here (img_size {img_size}, min_size {min_size}), "
            "so every image's merge loop can land on exactly fixed_length leaves (each merge "
            "removes exactly 3 leaves). Off-target leaf counts would silently break "
            "_serialize_batch's whole-batch grid_sample call, which assumes every image in "
            "the batch has the same region count."
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

    def _compute_all_levels_batch(self, imgs):
        """Per-level block errors and the merge cost of collapsing each level's blocks into their level-below parent.

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
        label_ids = torch.arange(1, num_features + 1, dtype=torch.long, device=device)
        membership = (flat.unsqueeze(0) == label_ids.unsqueeze(1))  # [K, H*W]

        row_idx = torch.arange(H, device=device).repeat_interleave(W)
        col_idx = torch.arange(W, device=device).repeat(H)
        INF = max(H, W) + 1

        rows_min = torch.where(membership, row_idx.unsqueeze(0), torch.full_like(row_idx.unsqueeze(0), INF)).min(dim=1).values
        rows_max = torch.where(membership, row_idx.unsqueeze(0), torch.zeros_like(row_idx.unsqueeze(0))).max(dim=1).values
        cols_min = torch.where(membership, col_idx.unsqueeze(0), torch.full_like(col_idx.unsqueeze(0), INF)).min(dim=1).values
        cols_max = torch.where(membership, col_idx.unsqueeze(0), torch.zeros_like(col_idx.unsqueeze(0))).max(dim=1).values

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

        The `scipy.ndimage.label` call itself runs on CPU (a real device
        round-trip) -- everything else, including the bbox extraction in
        `_labeled_to_region_tensors`, stays on `border_mask`'s own device.
        A fully-GPU alternative is possible (`cupyx.scipy.ndimage.label` --
        deliberately not `kornia.contrib.connected_components`, which has
        no 3D equivalent) but not implemented here; swapping it in only
        requires replacing this one method.
        """
        device = border_mask.device
        interior = ~border_mask.cpu().numpy()
        labeled_np, num_features = scipy_label(interior)
        return self._labeled_to_region_tensors(torch.from_numpy(labeled_np).to(device), num_features)

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
            `[B, N, interp_size**2]` (`C == 1`) or `[B, C, N, interp_size**2]`
            (`C > 1`).
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

        return patches.permute(0, 2, 1, 3) if C > 1 else patches[:, :, 0, :]

    def forward(self, img):
        """Adaptively patchifies `img` into a fixed-length sequence, entirely on `img`'s own device.

        Args:
            img: `[B, C, H, W]`.

        Returns:
            `(seq_img, seq_size, seq_pos)`: `seq_img` matches
            `Patchify.forward`'s existing shape contract
            (`UCF_VIT.dataloaders.transform`) -- `[B, fixed_length,
            interp_size**2]` (`C == 1`) or `[B, C, fixed_length,
            interp_size**2]` (`C > 1`). `seq_size` is `[B, fixed_length]`
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
