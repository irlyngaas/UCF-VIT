"""Tests for UCF_VIT.model.arch.VIT.effective_patch_size.

Covers the interp_size/patch_size dispatch introduced to stop patch_size
from being silently reused (with a different meaning) whenever
adaptive_patching is turned on: `interp_size` (the size adaptive
quadtree/octree leaf patches are interpolated to, and every dependent
model-layer calculation is based on) takes over patch_size's role entirely
in that mode, and is required in that mode.

Requires timm/monai/xformers (building_blocks.py's real, unconditional
top-level imports) -- see tests/distributed/test_tensor_parallel_correctness.py's
module docstring for why this skips cleanly via importorskip instead of
erroring at collection. Imports UCF_VIT.model.arch itself (not just the bare
top-level packages) so this also skips cleanly if an installed xformers
version has dropped a submodule building_blocks.py needs (as happened during
this test's own development).
"""

import pytest

VIT = pytest.importorskip(
    "UCF_VIT.model.arch",
    reason="needs the real UCF_VIT.model.building_blocks deps (timm/monai/xformers) -- run in the forge-vit env",
).VIT


def _make_vit(**overrides):
    kwargs = dict(
        img_size=16,
        patch_size=4,
        in_chans=1,
        num_classes=2,
        embed_dim=4,
        depth=1,
        num_heads=1,
        mlp_ratio=1.0,
        twoD=True,
        adaptive_patching=False,
        fixed_length=16,
        class_token=False,
        pos_embed="none",
    )
    kwargs.update(overrides)
    return VIT(**kwargs)


def test_effective_patch_size_non_adaptive_uses_patch_size():
    model = _make_vit(adaptive_patching=False, patch_size=4)
    assert model.effective_patch_size == 4


def test_effective_patch_size_adaptive_uses_interp_size():
    model = _make_vit(adaptive_patching=True, interp_size=8, fixed_length=16)
    assert model.effective_patch_size == 8
    # patch_size itself must stay untouched/truthful, not silently overwritten.
    assert model.patch_size == 4


def test_effective_patch_size_adaptive_missing_interp_size_raises():
    with pytest.raises(AssertionError, match="interp_size is required"):
        _make_vit(adaptive_patching=True, interp_size=None, fixed_length=16)
