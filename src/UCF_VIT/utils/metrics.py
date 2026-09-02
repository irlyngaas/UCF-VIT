# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as torchF
        
def masked_mse(pred, y, mask):
    """Computes the mean squared error over only the masked (e.g. patch-level) positions.

    Args:
        pred: Predicted values.
        y: Target values, same shape as `pred`.
        mask: Mask indicating which positions along the batch/sequence dimension to
            include in the loss (1 = include, 0 = exclude); broadcastable against the
            per-position mean of `(pred - y) ** 2`.

    Returns:
        Scalar tensor with the mean squared error averaged over the masked positions.
    """

    loss = (pred - y) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss*mask).sum() / mask.sum()

    return loss

def _native_resolution_sample(y, size, pos, patch_size, twoD, mode):
    """Samples `y` at each adaptive patch's native-resolution location, fully vectorized.

    One batched `grid_sample` call, using an affine box per patch built from
    each patch's true center/size (`torch.nn.functional.grid_sample`) --
    entirely without a Python-level loop over (batch, seq).

    Axis convention: `pos`'s `x`/`y`[`/z`] components follow
    `UCF_VIT.dataloaders.quadtree.Rect`/`octree.Cube`'s own convention --
    `x`/`y` are the *last*/*second-to-last* spatial axes of the source image
    (`Rect.get_area`: `img[y1:y2, x1:x2, :]`), and for 3D, `z` is the
    *third-to-last* (`Cube.get_area`: `img[z1:z2, y1:y2, x1:x2, :]`). This
    matches `torch.nn.functional.grid_sample`'s own `x`/`y`[`/z`]-to-last/
    second-to-last[/third-to-last]-axis convention exactly, so `y`'s spatial
    axes are used in whatever native order they arrive in -- no permutation,
    and critically, no swap. For `basic_ct`'s 3D case specifically: nothing
    in this codebase's data pipeline reorders a volume's 3 spatial axes into
    a specific "height/width/depth" meaning -- `detect_img_size`,
    `img_size`/`tile_size`, and the raw loaded array all carry the same
    native, unreordered axis order end to end (confirmed by tracing
    `dataloaders/dataset.py`'s `np.moveaxis(np_image, 0, -1)`, which only
    relocates the channel axis, never the 3 spatial ones), so `y`'s axes
    here just need to be in that same native order -- which they already
    are, coming from the same `batch["data"]` tensor.

    Args:
        y: Source volume to sample, shape (N, 1, H, W) for 2D or (N, 1, <3
            native spatial axes, unreordered>) for 3D -- N is any flattened
            leading batch dim (e.g. Batch, or Batch*Channel).
        size: Side length of each adaptive patch, shape (N, Seq_Length).
        pos: Center coordinates of each adaptive patch, shape (N,
            Seq_Length, 2) (`(x_center, y_center)`) or `(..., 3)` for 3D
            (`(x_center, y_center, z_center)`).
        patch_size: Output side length to sample each patch at.
        twoD: Whether `y` is 2D (True) or 3D (False).
        mode: `torch.nn.functional.grid_sample` interpolation mode (e.g.
            `"bicubic"`/`"bilinear"` for 2D, `"bilinear"` for 3D --
            `grid_sample` doesn't support bicubic for 5D input; `"nearest"`
            for discrete class labels).

    Returns:
        Sampled patches, shape (N, Seq_Length, patch_size, patch_size[,
        patch_size]).
    """
    n, seq_len = size.shape
    device, dtype = y.device, y.dtype
    # Pixel-center fraction of each output position, mapped to [-1, 1] --
    # combined with center/size below into a per-patch affine sampling box.
    t = (torch.arange(patch_size, device=device, dtype=dtype) + 0.5) / patch_size
    u = 2 * t - 1  # (patch_size,)

    if twoD:
        w, h = y.shape[3], y.shape[2]
        tx = 2 * pos[..., 0] / w - 1
        ty = 2 * pos[..., 1] / h - 1
        sx = size / w
        sy = size / h
        grid_x = tx.unsqueeze(-1) + sx.unsqueeze(-1) * u  # (N, S, patch_size), varies along W_out
        grid_y = ty.unsqueeze(-1) + sy.unsqueeze(-1) * u  # (N, S, patch_size), varies along H_out
        grid = torch.stack([
            grid_x.unsqueeze(2).expand(n, seq_len, patch_size, patch_size),
            grid_y.unsqueeze(3).expand(n, seq_len, patch_size, patch_size),
        ], dim=-1)  # (N, S, patch_size, patch_size, 2)
        grid = grid.reshape(n, seq_len * patch_size, patch_size, 2)
        sampled = torchF.grid_sample(y, grid, mode=mode, align_corners=False, padding_mode="zeros")
        sampled = sampled.reshape(n, seq_len, patch_size, patch_size)
    else:
        axis2, axis3, axis4 = y.shape[2], y.shape[3], y.shape[4]  # native order, see docstring
        tx = 2 * pos[..., 0] / axis4 - 1
        ty = 2 * pos[..., 1] / axis3 - 1
        tz = 2 * pos[..., 2] / axis2 - 1
        sx = size / axis4
        sy = size / axis3
        sz = size / axis2
        grid_x = tx.unsqueeze(-1) + sx.unsqueeze(-1) * u
        grid_y = ty.unsqueeze(-1) + sy.unsqueeze(-1) * u
        grid_z = tz.unsqueeze(-1) + sz.unsqueeze(-1) * u
        grid = torch.stack([
            grid_x.view(n, seq_len, 1, 1, patch_size).expand(n, seq_len, patch_size, patch_size, patch_size),
            grid_y.view(n, seq_len, 1, patch_size, 1).expand(n, seq_len, patch_size, patch_size, patch_size),
            grid_z.view(n, seq_len, patch_size, 1, 1).expand(n, seq_len, patch_size, patch_size, patch_size),
        ], dim=-1)  # (N, S, patch_size, patch_size, patch_size, 3)
        grid = grid.reshape(n, seq_len * patch_size, patch_size, patch_size, 3)
        sampled = torchF.grid_sample(y, grid, mode=mode, align_corners=False, padding_mode="zeros")
        sampled = sampled.reshape(n, seq_len, patch_size, patch_size, patch_size)

    return sampled


def _native_resolution_patch_squared_error(output, y, size, pos, patch_size, twoD):
    """Computes per-patch MSE between adaptively-sized predicted patches and `y`, fully vectorized.

    Rather than resizing each prediction up to its patch's real size (the
    direction the original, un-vectorized version of this function used),
    samples the *ground truth* directly at the prediction's fixed
    `patch_size` resolution via `_native_resolution_sample` (`mode="bicubic"`
    for 2D / `"bilinear"` for 3D). Mathematically equivalent in spirit
    (still comparing the prediction against the correctly-scaled real-image
    region for that patch), just resampling in the other direction.

    Args:
        output: Predicted patch values, shape (Batch, Seq_Length, patch_dim),
            `patch_dim = patch_size**2 * Channel` (2D) or `patch_size**3 *
            Channel` (3D), `(pixel, channel)`-folded -- the same contract
            `MAE.forward`/`einops.rearrange(batch["seq"], 'b c s p -> b s (p
            c)')` already use (see `training.py`'s MAE `do_ap:True` branch).
        y: Target volume, shape (Batch, Channel, H, W) for 2D or (Batch,
            Channel, <3 native spatial axes, unreordered>) for 3D.
        size: Side length of each adaptive patch, shape (Batch,
            adaptive_patching_channels, Seq_Length) -- `adaptive_patching_channels`
            is 1 if one shared quadtree/octree was used across every real
            channel, or equal to `y`'s channel count if each channel got its
            own independent tree.
        pos: Center coordinates of each adaptive patch, shape (Batch,
            adaptive_patching_channels, Seq_Length, 2) (`(x_center,
            y_center)`) or `(..., 3)` for 3D (`(x_center, y_center,
            z_center)`).
        patch_size: Fixed spatial size predicted patches are stored at.
        twoD: Whether the data is 2D (True) or 3D (False).

    Returns:
        A tuple `(per_patch_mse, valid)`, both shape (Batch, Channel,
        Seq_Length): `per_patch_mse` is the MSE for every (batch, channel,
        seq) position (including padding slots -- `grid_sample`'s
        `padding_mode="zeros"` makes those well-defined, just meaningless);
        `valid` is `size != 0` (padding slots excluded), broadcast to `y`'s
        channel count.
    """
    ndims = 2 if twoD else 3
    batch_size, num_channels, seq_len = size.shape
    num_channels_y = y.shape[1]

    if num_channels == 1:
        size_full = size.expand(batch_size, num_channels_y, seq_len)
        pos_full = pos.expand(batch_size, num_channels_y, seq_len, ndims)
    else:
        size_full = size
        pos_full = pos

    n = batch_size * num_channels_y  # one real image (no replication) per (batch, channel)

    if twoD:
        out_r = output.reshape(batch_size, seq_len, patch_size, patch_size, num_channels_y)
        out_r = out_r.permute(0, 4, 1, 2, 3).reshape(n, seq_len, patch_size, patch_size)
    else:
        out_r = output.reshape(batch_size, seq_len, patch_size, patch_size, patch_size, num_channels_y)
        out_r = out_r.permute(0, 5, 1, 2, 3, 4).reshape(n, seq_len, patch_size, patch_size, patch_size)

    src = y.reshape(n, 1, *y.shape[2:])  # (N, 1, ...spatial)
    centers = pos_full.reshape(n, seq_len, ndims)
    sizes = size_full.reshape(n, seq_len)

    mode = "bicubic" if twoD else "bilinear"
    sampled = _native_resolution_sample(src, sizes, centers, patch_size, twoD, mode)

    squared_error = (sampled - out_r) ** 2
    reduce_dims = tuple(range(2, squared_error.dim()))
    per_patch_mse = squared_error.mean(dim=reduce_dims)  # (N, S)

    per_patch_mse = per_patch_mse.reshape(batch_size, num_channels_y, seq_len)
    valid = (sizes != 0).reshape(batch_size, num_channels_y, seq_len)
    return per_patch_mse, valid


def native_resolution_patch_mse(output, y, size, pos, patch_size, twoD):
    """Computes MSE loss between adaptively-sized predicted patches and the target image.

    Averaged over every non-padding patch (`size != 0`) -- see
    `_native_resolution_patch_squared_error` for the full contract and axis
    convention.

    Returns:
        Scalar tensor with the MSE loss averaged over all non-empty patches.
    """
    per_patch_mse, valid = _native_resolution_patch_squared_error(output, y, size, pos, patch_size, twoD)
    valid = valid.to(per_patch_mse.dtype)
    return (per_patch_mse * valid).sum() / valid.sum()


def native_resolution_patch_masked_mse(output, y, size, pos, patch_size, twoD, mask):
    """Like `native_resolution_patch_mse`, but additionally averaged only over masked (e.g. MAE-hidden) patches.

    The adaptive-patching analog of `masked_mse` -- excludes both padding
    (`size == 0`) and unmasked/visible (`mask == 0`) patches, the standard
    MAE-paper loss extended to variable-size patches.

    Args:
        mask: Binary mask (1 = include/masked, 0 = exclude/visible), shape
            (Batch, adaptive_patching_channels, Seq_Length) -- same shape
            convention as `size` (see `_native_resolution_patch_squared_error`).
        (All other args match `native_resolution_patch_mse`.)

    Returns:
        Scalar tensor with the MSE loss averaged over all masked, non-empty patches.
    """
    per_patch_mse, valid = _native_resolution_patch_squared_error(output, y, size, pos, patch_size, twoD)
    batch_size, num_channels_y, seq_len = valid.shape
    if mask.shape[1] == 1 and num_channels_y > 1:
        mask = mask.expand(batch_size, num_channels_y, seq_len)
    valid = (valid & (mask == 1)).to(per_patch_mse.dtype)
    return (per_patch_mse * valid).sum() / valid.sum()


def native_resolution_dice_loss(output, y, size, pos, patch_size, twoD, num_classes, weight=0.5, smooth=1.0):
    """Dice+BCE loss between per-token predicted patches and the native-resolution label.

    The adaptive-patching analog of `DiceBLoss`: instead of comparing
    against a label that's been resized (lossy) onto a dense grid, samples
    the real segmentation label directly at each token's own native
    position/size via `_native_resolution_sample` (`mode="nearest"`, the
    standard choice for discrete class labels), one-hot encodes it, and
    computes Dice+BCE over all non-padding tokens, excluding the background
    class (channel 0) as `DiceBLoss` does.

    Unlike `_native_resolution_patch_squared_error`, `size`/`pos` here have
    no `adaptive_patching_channels` dimension: `y` (the label) always has
    exactly 1 channel (class indices, not per-image-channel data), so there's
    no per-channel grid to select between -- `size`/`pos` apply directly per
    (batch, token).

    Args:
        output: Predicted per-token logits, shape (Batch, Seq_Length,
            num_classes, patch_size, patch_size[, patch_size]).
        y: Ground-truth class-index label volume, shape (Batch, 1, H, W[, D]).
        size: Side length of each adaptive patch, shape (Batch, Seq_Length).
        pos: Center coordinates of each adaptive patch, shape (Batch,
            Seq_Length, 2) (2D) or (..., 3) (3D) -- see
            `_native_resolution_sample` for the axis convention.
        patch_size: Fixed spatial size predicted patches are stored at.
        twoD: Whether the data is 2D (True) or 3D (False).
        num_classes: Number of segmentation classes.
        weight: Weight given to the BCE term; the Dice term is weighted by
            `1 - weight` (matches `DiceBLoss`).
        smooth: Smoothing constant added to the Dice coefficient's numerator/
            denominator.

    Returns:
        Scalar tensor with `weight * BCE + (1 - weight) * dice_loss`, averaged
        over all non-padding (`size != 0`) tokens.
    """
    sampled = _native_resolution_sample(y.to(output.dtype), size, pos, patch_size, twoD, mode="nearest")
    sampled = sampled.round().long().clamp(0, num_classes - 1)
    target = torchF.one_hot(sampled, num_classes)  # (B, S, patch..., num_classes)
    target = torch.movedim(target, -1, 2).to(output.dtype)  # (B, S, num_classes, patch...)

    valid_mask = (size != 0).to(output.dtype)  # (B, S)
    while valid_mask.dim() < output.dim():
        valid_mask = valid_mask.unsqueeze(-1)

    pred = torchF.sigmoid(output)[:, :, 1:] * valid_mask  # drop background class, zero out padding tokens
    true = target[:, :, 1:] * valid_mask

    pred = torch.flatten(pred)
    true = torch.flatten(true)

    intersection = (pred * true).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (pred.sum() + true.sum() + smooth)
    mask_full = valid_mask.expand_as(output[:, :, 1:]).flatten()
    BCE = torchF.binary_cross_entropy(pred, true, reduction='sum') / mask_full.sum().clamp(min=1)
    return weight * BCE + (1 - weight) * dice_loss


class DiceBLoss(nn.Module):
    """Combined Dice loss and binary cross-entropy loss for segmentation.

    Computes a weighted sum of binary cross-entropy and soft Dice loss, both
    excluding the first (background) class channel.
    """

    def __init__(self, weight=0.5, num_class=2, size_average=True):
        """Initializes the loss weighting between BCE and Dice loss.

        Args:
            weight: Weight given to the BCE term; the Dice term is weighted by
                `1 - weight`.
            num_class: Number of segmentation classes. Unused directly but stored
                for reference.
            size_average: Unused; kept for interface compatibility.
        """
        super(DiceBLoss, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def forward(self, inputs, targets, smooth=1, act=True):
        """Computes the combined Dice + BCE loss.

        Args:
            inputs: Predicted logits (or probabilities if `act=False`), shape
                (Batch, Class, ...).
            targets: Ground-truth one-hot/probability targets, same shape as
                `inputs`.
            smooth: Smoothing constant added to numerator/denominator of the Dice
                coefficient to avoid division by zero.
            act: If True, apply a sigmoid activation to `inputs` before computing
                the loss.

        Returns:
            Scalar tensor with `weight * BCE + (1 - weight) * dice_loss`.
        """
        #comment out if your model contains a sigmoid or equivalent activation layer
        if act:
            inputs = torchF.sigmoid(inputs)    
    
        # pred = torch.flatten(inputs)
        # true = torch.flatten(targets)
    
        # #flatten label and prediction tensors
        pred = torch.flatten(inputs[:,1:,:,:])
        true = torch.flatten(targets[:,1:,:,:])
    
        intersection = (pred * true).sum()
        coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)    
        dice_loss = 1 - (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)  
        BCE = torchF.binary_cross_entropy(pred, true, reduction='mean')
        dice_bce = self.weight*BCE + (1-self.weight)*dice_loss
        # dice_bce = dice_loss 
    
        return dice_bce
