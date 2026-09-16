"""Real-GPU correctness check for region_backend="cupyx" against a real cupy install.

Every other test of `region_backend="cupyx"` (`test_gpu_adaptive_patching.py`/
`test_gpu_adaptive_patching_3d.py`) fakes `cupy`/`cupyx` out entirely --
useful for the *dispatch* logic (is the right branch reached, is its
result wrapped back into `RegionTensors` correctly), but none of them can
say anything about whether `cupyx.scipy.ndimage.label`'s real numerics
actually match `scipy.ndimage.label`'s on real hardware. This file is that
missing check -- skips cleanly (module-level) unless a real CUDA/ROCm
device and a real `cupy` install are both present, so it's a no-op in
every environment except an actual GPU job with `cupy` installed (see
`launch/tests/run_cupyx_smoke.sh`).

Confirms `region_backend="cupyx"` produces the exact same detected regions
(and therefore the exact same `seq_img`/`seq_size`/`seq_pos`) as
`region_backend="scipy"` given the identical input, for both
`GPUPatchify2D` and `GPUPatchify3D` -- `region_backend` only changes how
the final connected-components step is computed, not the merge decisions
that precede it (those are identical either way, given identical input),
so an exact match here is the right bar, not an approximate one.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("region_backend='cupyx' needs a real CUDA/ROCm device", allow_module_level=True)

try:
    import cupy  # noqa: F401
except ImportError:
    pytest.skip(
        "region_backend='cupyx' needs a real cupy install (matching this device's CUDA/ROCm "
        "build) -- not installed here. See tests/README.md's cupyx region-detection entry.",
        allow_module_level=True,
    )

from UCF_VIT.model.gpu_adaptive_patching import GPUPatchify2D, GPUPatchify3D

DEVICE = torch.device("cuda:0")


def test_gpu_patchify_2d_cupyx_matches_scipy():
    img = torch.zeros(2, 1, 16, 16, device=DEVICE)
    img[0, 0, 0:8, 0:8] = 1.0
    img[0, 0, 0:8, 8:16] = 2.0
    img[0, 0, 8:16, 0:8] = 3.0
    img[0, 0, 8:16, 8:16] = 4.0
    img[1] = torch.rand(1, 16, 16, device=DEVICE)

    scipy_patchify = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2, region_backend="scipy")
    cupyx_patchify = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2, region_backend="cupyx")

    seq_img_scipy, seq_size_scipy, seq_pos_scipy = scipy_patchify(img)
    seq_img_cupyx, seq_size_cupyx, seq_pos_cupyx = cupyx_patchify(img)

    torch.testing.assert_close(seq_size_cupyx, seq_size_scipy)
    torch.testing.assert_close(seq_pos_cupyx, seq_pos_scipy)
    torch.testing.assert_close(seq_img_cupyx, seq_img_scipy)


def test_gpu_patchify_3d_cupyx_matches_scipy():
    vol = torch.zeros(2, 1, 16, 16, 16, device=DEVICE)
    val = 1.0
    for dd in (0, 1):
        for hh in (0, 1):
            for ww in (0, 1):
                vol[0, 0, dd * 8:dd * 8 + 8, hh * 8:hh * 8 + 8, ww * 8:ww * 8 + 8] = val
                val += 1.0
    vol[1] = torch.rand(1, 16, 16, 16, device=DEVICE)

    scipy_patchify = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, region_backend="scipy")
    cupyx_patchify = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, region_backend="cupyx")

    seq_img_scipy, seq_size_scipy, seq_pos_scipy = scipy_patchify(vol)
    seq_img_cupyx, seq_size_cupyx, seq_pos_cupyx = cupyx_patchify(vol)

    torch.testing.assert_close(seq_size_cupyx, seq_size_scipy)
    torch.testing.assert_close(seq_pos_cupyx, seq_pos_scipy)
    torch.testing.assert_close(seq_img_cupyx, seq_img_scipy)
