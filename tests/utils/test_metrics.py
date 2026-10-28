import pytest
import torch

from UCF_VIT.utils.metrics import DiceBLoss, native_resolution_dice_loss, native_resolution_patch_masked_mse, native_resolution_patch_mse, masked_mse


def test_masked_mse_only_averages_masked_positions():
    pred = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    y = torch.zeros_like(pred)
    mask = torch.tensor([1.0, 0.0])  # only the first row counts

    loss = masked_mse(pred, y, mask)
    assert loss.item() == 1.0  # mean((1-0)^2) for row 0 == 1.0, row 1 excluded


def test_masked_mse_zero_when_prediction_matches_target():
    pred = torch.randn(4, 5)
    y = pred.clone()
    mask = torch.ones(4)
    loss = masked_mse(pred, y, mask)
    assert loss.item() == 0.0


def test_dice_bloss_near_zero_for_near_perfect_prediction():
    # channel 0 is background and is dropped by DiceBLoss; make channel 1 a
    # confident, correct foreground prediction everywhere.
    targets = torch.zeros(1, 2, 4, 4)
    targets[:, 1] = 1.0

    logits = torch.full((1, 2, 4, 4), -10.0)
    logits[:, 1] = 10.0  # sigmoid(10) ~= 1

    loss = DiceBLoss(weight=0.5)(logits, targets, act=True)
    assert loss.item() < 0.01


def test_dice_bloss_high_for_confidently_wrong_prediction():
    targets = torch.zeros(1, 2, 4, 4)
    targets[:, 1] = 1.0

    logits = torch.full((1, 2, 4, 4), 10.0)
    logits[:, 1] = -10.0  # confidently predicts the opposite of the target

    loss = DiceBLoss(weight=0.5)(logits, targets, act=True)
    assert loss.item() > 1.0


def test_dice_bloss_accepts_probabilities_without_activation():
    targets = torch.zeros(1, 2, 4, 4)
    targets[:, 1] = 1.0
    probs = targets.clone()  # already-perfect probabilities

    loss = DiceBLoss(weight=0.5)(probs, targets, act=False)
    assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# native_resolution_patch_mse / native_resolution_patch_masked_mse
# ---------------------------------------------------------------------------
#
# output's shape/layout contract mirrors MAE.forward's real one: (Batch,
# Seq_Length, patch_dim), patch_dim = patch_size**ndims * Channel, (pixel,
# channel)-folded -- the same contract training.py's do_ap:True MSE/maskMSE
# branches already use (einops.rearrange(batch["seq"], 'b c s p -> b s (p
# c)')). size/pos follow UCF_VIT.dataloaders.quadtree/octree's own real
# convention: size/pos's "channel" axis is 1 (one shared adaptive-patching
# grid across every real channel) or equal to y's real channel count (an
# independent grid per channel); pos is (x_center, y_center[, z_center]),
# matching Rect/Cube's own (x=last spatial axis, y=second-to-last[, z=
# third-to-last]) convention -- the same axis order torch.nn.functional.
# grid_sample itself uses, confirmed by tracing dataloaders/dataset.py's
# np.moveaxis(np_image, 0, -1) (only relocates the channel axis, never the
# spatial ones) all the way through to what Patchify/Patchify_3D receive.


def _patch_dim_output(values_per_patch, patch_size, twoD, num_channels_y=1):
    """Builds a real-shaped `output` tensor: one constant value per (patch,
    channel) -- each entry in `values_per_patch` is either a single scalar
    (broadcast across every channel of that patch) or a per-channel list --
    matching native_resolution_patch_mse's (Batch=1, Seq_Length, patch_dim)
    contract, (pixel, channel)-folded.
    """
    tile = patch_size ** (2 if twoD else 3)
    rows = []
    for v in values_per_patch:
        per_channel = torch.as_tensor(v, dtype=torch.float32).reshape(1, num_channels_y)
        rows.append(per_channel.expand(tile, num_channels_y).reshape(-1))
    return torch.stack(rows).unsqueeze(0)  # (1, S, patch_dim)


def test_native_resolution_patch_mse_2d_matches_true_crop_when_output_is_exact():
    patch_size = 8
    y = torch.arange(20 * 30, dtype=torch.float32).reshape(1, 1, 20, 30)
    true_crop = y[0, 0, 10 - 4:10 + 4, 15 - 4:15 + 4]  # (H, W)-ordered crop

    size = torch.tensor([[[8.0]]])  # (B=1, adaptive_channels=1, S=1)
    pos = torch.tensor([[[[15.0, 10.0]]]])  # (x_center, y_center)
    output = true_crop.reshape(1, 1, -1)

    loss = native_resolution_patch_mse(output, y, size, pos, patch_size, twoD=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_native_resolution_patch_mse_2d_axes_are_not_swapped():
    """Regression test for a real bug in the original, un-vectorized
    implementation: it indexed y[..., x_start:x_end, y_start:y_end], but
    y's real axis order is (..., H, W) while x/y-derived starts are
    (width, height)-derived (see Rect.get_area) -- swapped, invisible on
    square images. Deliberately non-square, with two distinct-valued
    regions placed so a swapped mapping would sample the wrong (out of
    bounds / zero-padded) region instead.
    """
    y = torch.zeros(1, 1, 20, 40)
    y[0, 0, 2:8, 20:26] = 100.0  # rows 2:8 (H), cols 20:26 (W)

    patch_size = 6
    size = torch.tensor([[[6.0]]])
    pos = torch.tensor([[[[23.0, 5.0]]]])  # x_center=23 (col), y_center=5 (row)
    output = torch.full((1, 1, patch_size * patch_size), 100.0)

    loss = native_resolution_patch_mse(output, y, size, pos, patch_size, twoD=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_native_resolution_patch_mse_skips_padding_patches():
    y = torch.zeros(1, 1, 20, 40)
    y[0, 0, 2:8, 20:26] = 100.0

    patch_size = 6
    # Second patch is padding (size == 0, pos == (-1, -1), matching
    # FixedQuadTree.serialize's own padding convention) with a wildly wrong
    # output value that must not affect the loss.
    size = torch.tensor([[[6.0, 0.0]]])
    pos = torch.tensor([[[[23.0, 5.0], [-1.0, -1.0]]]])
    output = _patch_dim_output([100.0, 9999.0], patch_size, twoD=True)

    loss = native_resolution_patch_mse(output, y, size, pos, patch_size, twoD=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_native_resolution_patch_mse_shared_grid_across_channels():
    # adaptive_channels == 1: the same (pos, size) applies to every real
    # channel of y.
    y = torch.zeros(1, 2, 16, 16)
    y[0, 0, 2:8, 2:8] = 10.0
    y[0, 1, 2:8, 2:8] = 20.0

    patch_size = 6
    size = torch.tensor([[[6.0]]])  # (B=1, adaptive_channels=1, S=1)
    pos = torch.tensor([[[[5.0, 5.0]]]])
    output = _patch_dim_output([[10.0, 20.0]], patch_size, twoD=True, num_channels_y=2)

    loss = native_resolution_patch_mse(output, y, size, pos, patch_size, twoD=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_native_resolution_patch_mse_independent_grid_per_channel():
    # adaptive_channels == num_channels_y: each channel has its own
    # independent (pos, size).
    y = torch.zeros(1, 2, 16, 16)
    y[0, 0, 2:8, 2:8] = 10.0
    y[0, 1, 8:14, 8:14] = 20.0

    patch_size = 6
    size = torch.tensor([[[6.0], [6.0]]])  # (B=1, adaptive_channels=2, S=1)
    pos = torch.tensor([[[[5.0, 5.0]], [[11.0, 11.0]]]])
    out_ch0 = torch.full((patch_size * patch_size,), 10.0)
    out_ch1 = torch.full((patch_size * patch_size,), 20.0)
    output = torch.stack([out_ch0, out_ch1], dim=-1).reshape(1, 1, -1)  # (p c)-folded

    loss = native_resolution_patch_mse(output, y, size, pos, patch_size, twoD=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_native_resolution_patch_mse_3d_matches_true_crop_and_axes_are_not_mixed_up():
    # Deliberately distinct sizes on all 3 spatial axes so a z/y/x mixup
    # would sample from the wrong axis entirely.
    axis2, axis3, axis4 = 12, 20, 30
    y = torch.zeros(1, 1, axis2, axis3, axis4)
    y[0, 0, 3:9, 5:11, 10:16] = 77.0  # z 3:9, y 5:11, x 10:16

    patch_size = 6
    size = torch.tensor([[[6.0]]])
    pos = torch.tensor([[[[13.0, 8.0, 6.0]]]])  # (x_center, y_center, z_center)
    output = torch.full((1, 1, patch_size ** 3), 77.0)

    loss = native_resolution_patch_mse(output, y, size, pos, patch_size, twoD=False)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_native_resolution_patch_masked_mse_only_averages_masked_nonpadding_patches():
    y = torch.zeros(1, 1, 20, 40)
    y[0, 0, 2:8, 20:26] = 100.0

    patch_size = 6
    size = torch.tensor([[[6.0, 0.0]]])
    pos = torch.tensor([[[[23.0, 5.0], [-1.0, -1.0]]]])
    output = _patch_dim_output([100.0, 9999.0], patch_size, twoD=True)

    # mask==0 on the only real (non-padding) patch -> nothing qualifies,
    # the same 0/0 -> nan behavior masked_mse already has for an all-zero
    # mask (this function doesn't special-case it, matching masked_mse's
    # own precedent).
    all_excluded_mask = torch.tensor([[[0.0, 1.0]]])
    loss_excluded = native_resolution_patch_masked_mse(output, y, size, pos, patch_size, twoD=True, mask=all_excluded_mask)
    assert torch.isnan(loss_excluded)

    including_real_patch_mask = torch.tensor([[[1.0, 0.0]]])
    loss = native_resolution_patch_masked_mse(output, y, size, pos, patch_size, twoD=True, mask=including_real_patch_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# native_resolution_dice_loss
# ---------------------------------------------------------------------------
#
# size/pos here have no adaptive_patching_channels dim (unlike
# native_resolution_patch_mse's): the label always has exactly 1 channel
# (class indices), so size/pos apply directly per (batch, token) -- see the
# function's own docstring.


def _confident_logits(num_classes, foreground_class, patch_size, twoD, confident=10.0):
    shape = (1, 1, num_classes, patch_size, patch_size) if twoD else (1, 1, num_classes, patch_size, patch_size, patch_size)
    logits = torch.full(shape, -confident)
    logits[:, :, foreground_class] = confident
    return logits


def test_native_resolution_dice_loss_near_zero_for_correct_confident_prediction():
    y = torch.zeros(1, 1, 20, 40)
    y[0, 0, 2:8, 20:26] = 1.0  # class 1 region

    patch_size = 4
    size = torch.tensor([[6.0]])  # (B=1, S=1)
    pos = torch.tensor([[[23.0, 5.0]]])  # (x_center, y_center), matches the region exactly
    output = _confident_logits(num_classes=2, foreground_class=1, patch_size=patch_size, twoD=True)

    loss = native_resolution_dice_loss(output, y, size, pos, patch_size, twoD=True, num_classes=2)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_native_resolution_dice_loss_high_for_confidently_wrong_prediction():
    y = torch.zeros(1, 1, 20, 40)
    y[0, 0, 2:8, 20:26] = 1.0

    patch_size = 4
    size = torch.tensor([[6.0]])
    pos = torch.tensor([[[23.0, 5.0]]])
    output = _confident_logits(num_classes=2, foreground_class=0, patch_size=patch_size, twoD=True)  # predicts background instead

    loss = native_resolution_dice_loss(output, y, size, pos, patch_size, twoD=True, num_classes=2)
    assert loss.item() > 1.0


def test_native_resolution_dice_loss_skips_padding_tokens():
    y = torch.zeros(1, 1, 20, 40)
    y[0, 0, 2:8, 20:26] = 1.0

    patch_size = 4
    # Second token is padding (size == 0, pos == (-1, -1), matching
    # FixedQuadTree.serialize's own padding convention) with wildly wrong
    # logits that must not affect the loss.
    size = torch.tensor([[6.0, 0.0]])
    pos = torch.tensor([[[23.0, 5.0], [-1.0, -1.0]]])
    correct = _confident_logits(num_classes=2, foreground_class=1, patch_size=patch_size, twoD=True)
    wrong = _confident_logits(num_classes=2, foreground_class=0, patch_size=patch_size, twoD=True)
    output = torch.cat([correct, wrong], dim=1)  # (1, 2, num_classes, patch_size, patch_size)

    loss = native_resolution_dice_loss(output, y, size, pos, patch_size, twoD=True, num_classes=2)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_native_resolution_dice_loss_3d_matches_true_region():
    axis2, axis3, axis4 = 12, 20, 30
    y = torch.zeros(1, 1, axis2, axis3, axis4)
    y[0, 0, 3:9, 5:11, 10:16] = 2.0  # class 2, z 3:9, y 5:11, x 10:16

    patch_size = 4
    size = torch.tensor([[6.0]])
    pos = torch.tensor([[[13.0, 8.0, 6.0]]])  # (x_center, y_center, z_center)
    output = _confident_logits(num_classes=3, foreground_class=2, patch_size=patch_size, twoD=False)

    loss = native_resolution_dice_loss(output, y, size, pos, patch_size, twoD=False, num_classes=3)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)
