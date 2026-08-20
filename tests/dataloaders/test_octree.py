import numpy as np
import pytest

from UCF_VIT.dataloaders.octree import Cube, FixedOctTree


def test_cube_basic_geometry():
    c = Cube(x1=2, x2=6, y1=1, y2=5, z1=0, z2=4)
    assert c.get_coord() == (2, 6, 1, 5, 0, 4)
    assert c.get_size() == (4, 4, 4)
    assert c.get_center() == (4.0, 3.0, 2.0)


def test_cube_invalid_coords_raise():
    with pytest.raises(AssertionError):
        Cube(x1=5, x2=2, y1=0, y2=1, z1=0, z2=1)
    with pytest.raises(AssertionError):
        Cube(x1=0, x2=1, y1=5, y2=2, z1=0, z2=1)
    with pytest.raises(AssertionError):
        Cube(x1=0, x2=1, y1=0, y2=1, z1=5, z2=2)


def test_cube_contains():
    domain = np.zeros((16, 16, 16), dtype=np.float64)
    domain[0:8, 0:8, 0:8] = 255  # domain is indexed [z, y, x] by Cube.contains

    dense = Cube(x1=0, x2=8, y1=0, y2=8, z1=0, z2=8)
    assert dense.contains(domain, norm_factor=255) == 8 * 8 * 8

    empty = Cube(x1=8, x2=16, y1=8, y2=16, z1=8, z2=16)
    assert empty.contains(domain, norm_factor=255) == 0


def test_cube_get_area():
    img = np.arange(4 * 4 * 4 * 2).reshape(4, 4, 4, 2)
    c = Cube(x1=1, x2=3, y1=0, y2=2, z1=0, z2=4)
    area = c.get_area(img)
    assert area.shape == (4, 2, 2, 2)
    np.testing.assert_array_equal(area, img[0:4, 0:2, 1:3, :])


def _total_volume(tree):
    return sum(
        (x2 - x1) * (y2 - y1) * (z2 - z1)
        for x1, x2, y1, y2, z1, z2 in (c.get_coord() for c, _ in tree.nodes)
    )


def test_fixedocttree_splits_into_fixed_length_nodes():
    domain = np.zeros((16, 16, 16))
    tree = FixedOctTree(domain=domain, fixed_length=8, norm_factor=255)
    assert len(tree.nodes) == 8
    assert _total_volume(tree) == 16 ** 3


def test_fixedocttree_prioritizes_highest_density_region():
    domain = np.zeros((16, 16, 16))
    domain[0:8, 0:8, 0:8] = 255
    tree = FixedOctTree(domain=domain, fixed_length=8, norm_factor=255)
    sizes = sorted(c.get_size() for c, _ in tree.nodes)
    assert sizes == [(8, 8, 8)] * 8


def test_fixedocttree_further_subdivides_dense_region():
    domain = np.zeros((16, 16, 16))
    domain[0:8, 0:8, 0:8] = 255
    tree = FixedOctTree(domain=domain, fixed_length=15, norm_factor=255)
    assert len(tree.nodes) == 15
    assert _total_volume(tree) == 16 ** 3
    sizes = sorted(c.get_size() for c, _ in tree.nodes)
    # the single dense 8x8x8 octant gets split again into eight 4x4x4s; the
    # other seven original 8x8x8 octants stay untouched
    assert sizes == [(4, 4, 4)] * 8 + [(8, 8, 8)] * 7
