"""Tests for UCF_VIT.training.train_step's MAE loss dispatch.

MAE supports two loss_fn values: "MSE" (plain nn.MSELoss over every patch,
masked and visible alike -- model.forward's own mask return is discarded)
and "maskMSE" (UCF_VIT.utils.metrics.masked_mse, averaged only over the
masked/encoder-hidden patches -- the standard MAE-paper loss). "maskMSE"
was previously dead code: the do_ap:True branch was a bare
"#TODO: elif ...", and the do_ap:False branch was fully written but
commented out. Both are now wired up; these tests exercise train_step's
actual dispatch for both, using a fake model (no real MAE/timm/monai/
xformers needed -- train_step only ever calls model.forward(...), so a
stub with the right (output, mask) return is enough) rather than a real
end-to-end training run.

Deliberately constructs output so masked and unmasked patches have a
different, known error (masked patches off by 2, unmasked patches exact)
-- this makes "maskMSE" (averages only the masked patches) and "MSE"
(averages every patch) produce different, independently-verifiable
numbers from the same fixture, rather than relying on both losses
happening to agree on a trivial all-matching or all-mismatching case.

train_step's real, previously-found bugs (see tests/README.md) were all
about picking the wrong branch/input tensor (do_ap not checked at all, or
checked but reading the wrong batch key) -- so each test also asserts the
fake model was actually called with the expected input tensor (batch["seq"]
vs batch["data"]), not just that the returned loss number happens to be
right.
"""

import einops
import pytest
import torch

from UCF_VIT.training import train_step
from UCF_VIT.utils.misc import patchify

PATCH_SIZE = 2
# 4x4 image, patch_size=2 -> 4 patches, patch_dim = patch_size**2 * channels = 4.
DATA = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
TARGET_NON_DO_AP = patchify(DATA, PATCH_SIZE, True)  # (1, 4, 4)

# Equivalent do_ap input: same 4 patches x patch_dim=4, channel-first
# (Batch, Channel, Seq_Length, Patch_Size*Patch_Size), matching train_step's
# own einops.rearrange(batch["seq"], 'b c s p -> b s (p c)') target
# construction.
SEQ_DO_AP = TARGET_NON_DO_AP.unsqueeze(1)  # (1, 1, 4, 4) : (B, C, S, P)
TARGET_DO_AP = einops.rearrange(SEQ_DO_AP, 'b c s p -> b s (p c)')  # (1, 4, 4)

# Masked patches (mask==1) are off by a constant 2 -> per-patch MSE 4.0;
# unmasked patches (mask==0) match exactly -> per-patch MSE 0.0.
MASK = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
ERROR_PER_ELEMENT = torch.tensor([[2.0], [0.0], [2.0], [0.0]]).expand(1, 4, 4)
EXPECTED_MASKED_MSE = 4.0  # mean((2)**2) over just the 2 masked patches
EXPECTED_FULL_MSE = 2.0  # mean((2)**2 or 0**2) over all 16 elements = 32/16


class _FakeMAEModel:
    """Stub standing in for a real MAE model -- train_step only ever calls
    `.forward(x, variables, seq_ps)`, so this records what it was called
    with and returns a fixed (output, mask), no real model needed.
    """

    def __init__(self, output, mask):
        self._output = output
        self._mask = mask
        self.calls = []

    def forward(self, x, variables, seq_ps):
        self.calls.append(x)
        return self._output, self._mask


def _conf(loss_fn, do_ap):
    return {
        "model": {"type": "MAE", "loss_fn": loss_fn},
        "ap": {"do_ap": do_ap},
        "data": {"patch_size": PATCH_SIZE, "twoD": True},
    }


def test_train_step_mae_maskmse_non_do_ap_averages_only_masked_patches():
    output = TARGET_NON_DO_AP + ERROR_PER_ELEMENT
    model = _FakeMAEModel(output, MASK)
    batch = {"data": DATA, "variables": ["v0"], "seq_ps": None}

    loss = train_step(_conf("maskMSE", do_ap=False), batch, model)

    assert loss.item() == pytest.approx(EXPECTED_MASKED_MSE)
    assert model.calls[0] is DATA


def test_train_step_mae_mse_non_do_ap_averages_over_every_patch():
    output = TARGET_NON_DO_AP + ERROR_PER_ELEMENT
    model = _FakeMAEModel(output, MASK)
    batch = {"data": DATA, "variables": ["v0"], "seq_ps": None}

    loss = train_step(_conf("MSE", do_ap=False), batch, model)

    assert loss.item() == pytest.approx(EXPECTED_FULL_MSE)
    assert model.calls[0] is DATA


def test_train_step_mae_maskmse_do_ap_averages_only_masked_patches():
    output = TARGET_DO_AP + ERROR_PER_ELEMENT
    model = _FakeMAEModel(output, MASK)
    batch = {"seq": SEQ_DO_AP, "variables": ["v0"], "seq_ps": None}

    loss = train_step(_conf("maskMSE", do_ap=True), batch, model)

    assert loss.item() == pytest.approx(EXPECTED_MASKED_MSE)
    assert model.calls[0] is SEQ_DO_AP


def test_train_step_mae_mse_do_ap_averages_over_every_patch():
    output = TARGET_DO_AP + ERROR_PER_ELEMENT
    model = _FakeMAEModel(output, MASK)
    batch = {"seq": SEQ_DO_AP, "variables": ["v0"], "seq_ps": None}

    loss = train_step(_conf("MSE", do_ap=True), batch, model)

    assert loss.item() == pytest.approx(EXPECTED_FULL_MSE)
    assert model.calls[0] is SEQ_DO_AP
