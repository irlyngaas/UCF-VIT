import pytest
import torch

from UCF_VIT.utils.misc import calculate_tile_overlap, is_power_of_two, patchify, unpatchify


@pytest.mark.parametrize(
    "n,expected",
    [(0, False), (1, True), (2, True), (3, False), (4, True), (5, False), (16, True), (1023, False), (1024, True)],
)
def test_is_power_of_two(n, expected):
    assert is_power_of_two(n) is expected


def test_calculate_tile_overlap_even():
    start, end = calculate_tile_overlap((4, 4))
    assert start == [2, 2]
    assert end == [2, 2]


def test_calculate_tile_overlap_odd():
    start, end = calculate_tile_overlap((5, 3))
    assert start == [2, 1]
    assert end == [3, 2]


def test_calculate_tile_overlap_zero():
    start, end = calculate_tile_overlap((0, 0, 0))
    assert start == [0, 0, 0]
    assert end == [0, 0, 0]


def test_patchify_unpatchify_roundtrip_2d():
    torch.manual_seed(0)
    patch_size = 4
    data = torch.randn(2, 3, 8, 8)  # 2x2 grid of patches

    patches = patchify(data, patch_size, twoD=True)
    assert patches.shape == (2, 4, patch_size * patch_size * 3)

    recon = unpatchify(patches, data, patch_size, twoD=True)
    assert recon.shape == data.shape
    torch.testing.assert_close(recon, data)


def test_patchify_unpatchify_roundtrip_3d():
    torch.manual_seed(0)
    patch_size = 4
    data = torch.randn(1, 2, 8, 8, 8)  # 2x2x2 grid of patches

    patches = patchify(data, patch_size, twoD=False)
    assert patches.shape == (1, 8, patch_size ** 3 * 2)

    recon = unpatchify(patches, data, patch_size, twoD=False)
    assert recon.shape == data.shape
    torch.testing.assert_close(recon, data)
