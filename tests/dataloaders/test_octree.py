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
    # contains() is deliberately unnormalized -- its score is only ever used
    # for relative (max-based) comparison, see its own docstring -- so this
    # is a raw sum, not divided by anything.
    domain = np.zeros((16, 16, 16), dtype=np.float64)
    domain[0:8, 0:8, 0:8] = 255  # domain is indexed [z, y, x] by Cube.contains

    dense = Cube(x1=0, x2=8, y1=0, y2=8, z1=0, z2=8)
    assert dense.contains(domain) == 8 * 8 * 8 * 255

    empty = Cube(x1=8, x2=16, y1=8, y2=16, z1=8, z2=16)
    assert empty.contains(domain) == 0


def test_cube_contains_variance_flat_region_scores_zero():
    domain = np.full((8, 8, 8), 3.0)
    c = Cube(x1=0, x2=8, y1=0, y2=8, z1=0, z2=8)
    assert c.contains(domain, score_fn="variance") == 0.0


def test_cube_contains_variance_matches_np_var_times_volume():
    domain = np.arange(8 ** 3, dtype=np.float64).reshape(8, 8, 8)
    c = Cube(x1=1, x2=5, y1=2, y2=6, z1=0, z2=4)
    patch = domain[0:4, 2:6, 1:5]
    expected = float(np.var(patch) * patch.size)
    assert c.contains(domain, score_fn="variance") == pytest.approx(expected)


def test_cube_contains_variance_sums_across_channels():
    domain = np.zeros((8, 8, 8, 2))
    domain[0:4, 0:4, 0:4, 0] = np.arange(64).reshape(4, 4, 4)
    domain[0:4, 0:4, 0:4, 1] = np.arange(64).reshape(4, 4, 4) * 10
    c = Cube(x1=0, x2=4, y1=0, y2=4, z1=0, z2=4)
    patch = domain[0:4, 0:4, 0:4]
    expected = sum(float(np.var(patch[..., ch]) * 64) for ch in range(2))
    assert c.contains(domain, score_fn="variance") == pytest.approx(expected)


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
    tree = FixedOctTree(domain=domain, fixed_length=8)
    assert len(tree.nodes) == 8
    assert _total_volume(tree) == 16 ** 3


def test_fixedocttree_prioritizes_highest_density_region():
    domain = np.zeros((16, 16, 16))
    domain[0:8, 0:8, 0:8] = 255
    tree = FixedOctTree(domain=domain, fixed_length=8)
    sizes = sorted(c.get_size() for c, _ in tree.nodes)
    assert sizes == [(8, 8, 8)] * 8


def test_fixedocttree_further_subdivides_dense_region():
    domain = np.zeros((16, 16, 16))
    domain[0:8, 0:8, 0:8] = 255
    tree = FixedOctTree(domain=domain, fixed_length=15)
    assert len(tree.nodes) == 15
    assert _total_volume(tree) == 16 ** 3
    sizes = sorted(c.get_size() for c, _ in tree.nodes)
    # the single dense 8x8x8 octant gets split again into eight 4x4x4s; the
    # other seven original 8x8x8 octants stay untouched
    assert sizes == [(4, 4, 4)] * 8 + [(8, 8, 8)] * 7


def test_fixedocttree_variance_further_subdivides_high_variance_region():
    """score_fn="variance" analog of test_fixedocttree_further_subdivides_
    dense_region -- domain here is the raw voxel content itself (not a
    precomputed edge volume), and the high-variance octant is the one that
    gets split further.
    """
    domain = np.full((16, 16, 16), 5.0)
    domain[0:8, 0:8, 0:8] = np.tile([[[1., 9.], [9., 1.]], [[9., 1.], [1., 9.]]], (4, 4, 4))
    tree = FixedOctTree(domain=domain, fixed_length=15, score_fn="variance")
    assert len(tree.nodes) == 15
    assert _total_volume(tree) == 16 ** 3
    sizes = sorted(c.get_size() for c, _ in tree.nodes)
    assert sizes == [(4, 4, 4)] * 8 + [(8, 8, 8)] * 7
