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


def test_fixedocttree_min_size_limits_smallest_leaf():
    domain = np.zeros((16, 16, 16))
    domain[0:4, 0:4, 0:4] = 255  # dense in a small region, plenty of room to keep splitting

    tree_default = FixedOctTree(domain=domain, fixed_length=50)
    assert min(c.get_size()[0] for c, _ in tree_default.nodes) == 2  # default min_size=2, unchanged

    tree_min4 = FixedOctTree(domain=domain, fixed_length=50, min_size=4)
    assert min(c.get_size()[0] for c, _ in tree_min4.nodes) == 4
    assert len(tree_min4.nodes) < 50  # stops early -- can't reach fixed_length without going below min_size


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


def test_fixedocttree_root_covers_whole_non_cuboid_domain_does_not_transpose_axes():
    """Regression test for a real axis-permutation bug: `_build_tree`'s root
    used to be built as `Cube(0, h, 0, w, 0, d)` (the raw `(h, w, d) =
    domain.shape[:3]` unpack order), but `Cube`'s own indexing (`contains`/
    `get_area`: `domain[z1:z2, y1:y2, x1:x2]`) requires the full reverse,
    `z <-> axis0, y <-> axis1, x <-> axis2` -- i.e. `Cube(0, d, 0, w, 0, h)`.
    Reusing the unpack order silently swapped axis0/axis2 (axis1 happens to
    land correctly either way, since it's the middle element of a 3-tuple
    reversal), which is undetectable on a cubic domain (h == w == d, where
    both calls are identical) -- every other test in this file uses one.
    `_total_volume`-style checks also can't catch this: bounding-box volume
    is a product of the three extents, unaffected by which axis each extent
    is assigned to. Only the actual extracted content, on a domain whose
    axis0 and axis2 sizes differ, reveals it.
    """
    Z, Y, X = 8, 4, 2  # axis0, axis1, axis2 -- all distinct, none square/cubic
    domain = np.arange(Z * Y * X).reshape(Z, Y, X).astype(np.float64)

    tree = FixedOctTree(domain=domain, fixed_length=1)
    assert len(tree.nodes) == 1
    root, _ = tree.nodes[0]

    img = domain[..., np.newaxis]  # (Z, Y, X, C), matching get_area's own expected shape
    area = root.get_area(img)
    assert area.shape == img.shape
    np.testing.assert_array_equal(area, img)


def test_fixedocttree_serialize_handles_non_cuboid_domain():
    """`serialize`/`Cube.set_area` used to hard-assert every leaf's native
    patch was cubic (`h1==w1==d1`) before resizing it via
    `RegularGridInterpolator` -- not actually required: every split bisects
    all three axes together, so a non-cuboid domain just makes every leaf
    inherit that same aspect ratio (verified directly via `get_size` below),
    and `RegularGridInterpolator` already handles an arbitrary source grid
    shape regardless. Regression test for that assert's removal (2D analog:
    test_quadtree.py's test_fixedquadtree_serialize_handles_non_square_domain).
    """
    Z, Y, X = 32, 16, 8
    domain = np.random.RandomState(0).rand(Z, Y, X).astype(np.float32)
    img = np.stack([domain, domain * 2], axis=-1)  # (Z, Y, X, 2)

    # fixed_length must be reachable: count starts at 1 and grows by +7 per
    # split (pop 1 parent, push 8 children) -- see FixedQuadTree's analogous
    # +3-per-split case. fixed_length=8 is exactly the first split.
    tree = FixedOctTree(domain=domain, fixed_length=8, score_fn="variance", min_size=4)
    assert len(tree.nodes) == 8
    for bbox, _ in tree.nodes:
        x, y, z = bbox.get_size()
        assert (x, y, z) == (4, 8, 16)  # every leaf inherits the root's 8:16:32 == 1:2:4 aspect ratio

    seq_patch, seq_size, seq_pos = tree.serialize(img, size=(8, 8, 8, 2))
    assert len(seq_patch) == 8
    for patch in seq_patch:
        assert patch.shape == (8, 8, 8, 2)
