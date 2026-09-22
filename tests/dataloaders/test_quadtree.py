import numpy as np
import pytest

from UCF_VIT.dataloaders.quadtree import FixedQuadTree, Rect


def test_rect_basic_geometry():
    r = Rect(x1=2, x2=6, y1=1, y2=5)
    assert r.get_coord() == (2, 6, 1, 5)
    assert r.get_size() == (4, 4)
    assert r.get_center() == (4.0, 3.0)


def test_rect_invalid_coords_raise():
    with pytest.raises(AssertionError):
        Rect(x1=5, x2=2, y1=0, y2=1)
    with pytest.raises(AssertionError):
        Rect(x1=0, x2=1, y1=5, y2=2)


def test_rect_contains():
    # contains() is deliberately unnormalized -- its score is only ever used
    # for relative (max-based) comparison, see its own docstring -- so this
    # is a raw sum, not divided by anything.
    domain = np.zeros((8, 8), dtype=np.float64)
    domain[0:4, 0:4] = 255  # domain is indexed [y, x]

    dense = Rect(x1=0, x2=4, y1=0, y2=4)
    assert dense.contains(domain) == 4 * 4 * 255

    empty = Rect(x1=4, x2=8, y1=4, y2=8)
    assert empty.contains(domain) == 0


def test_rect_contains_variance_flat_region_scores_zero():
    domain = np.full((8, 8), 3.0)
    r = Rect(x1=0, x2=8, y1=0, y2=8)
    assert r.contains(domain, score_fn="variance") == 0.0


def test_rect_contains_variance_matches_np_var_times_area():
    domain = np.arange(64, dtype=np.float64).reshape(8, 8)
    r = Rect(x1=1, x2=5, y1=2, y2=6)
    patch = domain[2:6, 1:5]
    expected = float(np.var(patch) * patch.size)
    assert r.contains(domain, score_fn="variance") == pytest.approx(expected)


def test_rect_contains_variance_sums_across_channels():
    domain = np.zeros((8, 8, 2))
    domain[0:4, 0:4, 0] = np.arange(16).reshape(4, 4)
    domain[0:4, 0:4, 1] = np.arange(16).reshape(4, 4) * 10
    r = Rect(x1=0, x2=4, y1=0, y2=4)
    patch = domain[0:4, 0:4]
    expected = sum(float(np.var(patch[..., c]) * 16) for c in range(2))
    assert r.contains(domain, score_fn="variance") == pytest.approx(expected)


def test_rect_get_area():
    img = np.arange(4 * 4 * 3).reshape(4, 4, 3)
    r = Rect(x1=1, x2=3, y1=0, y2=2)
    area = r.get_area(img)
    assert area.shape == (2, 2, 3)
    np.testing.assert_array_equal(area, img[0:2, 1:3, :])


def _total_area(qdt):
    return sum((x2 - x1) * (y2 - y1) for x1, x2, y1, y2 in (r.get_coord() for r, _ in qdt.nodes))


def test_fixedquadtree_splits_into_fixed_length_nodes():
    domain = np.zeros((16, 16))
    qdt = FixedQuadTree(domain=domain, fixed_length=4)
    assert qdt.count_patches() == 4
    assert _total_area(qdt) == 16 * 16


def test_fixedquadtree_min_size_limits_smallest_leaf():
    """min_size is the shared floor with GPUPatchify2D's own min_size --
    a node this size (or smaller) is never split further, even if a much
    larger fixed_length is requested and it's still the highest-scoring
    candidate."""
    domain = np.zeros((16, 16))
    domain[0:4, 0:4] = 255  # dense in a small region, plenty of room to keep splitting

    qdt_default = FixedQuadTree(domain=domain, fixed_length=50)
    assert min(r.get_size()[0] for r, _ in qdt_default.nodes) == 2  # default min_size=2, unchanged

    qdt_min4 = FixedQuadTree(domain=domain, fixed_length=50, min_size=4)
    assert min(r.get_size()[0] for r, _ in qdt_min4.nodes) == 4
    assert qdt_min4.count_patches() < 50  # stops early -- can't reach fixed_length without going below min_size


def test_fixedquadtree_prioritizes_highest_density_region():
    domain = np.zeros((16, 16))
    domain[0:8, 0:8] = 255  # dense in the low-y/low-x quadrant
    qdt = FixedQuadTree(domain=domain, fixed_length=4)
    sizes = sorted(r.get_size() for r, _ in qdt.nodes)
    assert sizes == [(8, 8)] * 4


def test_fixedquadtree_further_subdivides_dense_region():
    domain = np.zeros((16, 16))
    domain[0:8, 0:8] = 255
    qdt = FixedQuadTree(domain=domain, fixed_length=7)
    assert qdt.count_patches() == 7
    assert _total_area(qdt) == 16 * 16
    sizes = sorted(r.get_size() for r, _ in qdt.nodes)
    # the dense 8x8 quadrant gets split again into four 4x4s; the other
    # three original 8x8 quadrants stay untouched
    assert sizes == [(4, 4)] * 4 + [(8, 8)] * 3


def test_fixedquadtree_variance_further_subdivides_high_variance_region():
    """score_fn="variance" analog of test_fixedquadtree_further_subdivides_
    dense_region -- domain here is the raw pixel content itself (not a
    precomputed edge map), and the high-variance quadrant (a checkerboard,
    not flat) is the one that gets split further.
    """
    domain = np.full((16, 16), 5.0)
    domain[0:8, 0:8] = np.tile([[1., 9.], [9., 1.]], (4, 4))
    qdt = FixedQuadTree(domain=domain, fixed_length=7, score_fn="variance")
    assert qdt.count_patches() == 7
    assert _total_area(qdt) == 16 * 16
    sizes = sorted(r.get_size() for r, _ in qdt.nodes)
    assert sizes == [(4, 4)] * 4 + [(8, 8)] * 3


def test_fixedquadtree_nodes_value():
    domain = np.zeros((16, 16))
    qdt = FixedQuadTree(domain=domain, fixed_length=4)
    values = qdt.nodes_value()
    assert values == [[1.0], [1.0], [1.0], [1.0]]  # each 8x8 node -> size/8 == 1.0


def test_fixedquadtree_encode_decode_roundtrip():
    domain = np.random.RandomState(0).rand(16, 16) * 255
    qdt = FixedQuadTree(domain=domain, fixed_length=4)
    meta = qdt.encode_nodes()

    qdt2 = FixedQuadTree(domain=domain, fixed_length=4, build_from_info=True, meta_info=meta)
    assert [r.get_coord() for r, _ in qdt.nodes] == [r.get_coord() for r, _ in qdt2.nodes]


def test_fixedquadtree_serialize_handles_non_square_domain():
    """`serialize` used to hard-assert every leaf's native patch was square
    (`h1==w1`) before resizing it -- not actually required: every split
    bisects both axes together, so a non-square (H != W) domain just makes
    every leaf inherit that same non-square aspect ratio (verified directly
    via `get_size` below), and `cv.resize` already handles resizing an
    arbitrary source shape to a fixed target `size` regardless. Regression
    test for that assert's removal.
    """
    H, W = 32, 64
    domain = np.random.RandomState(0).rand(H, W).astype(np.float32)
    img = np.stack([domain, domain * 2, domain * 3], axis=-1)  # (H, W, 3)

    # fixed_length must be reachable: count starts at 1 and grows by +3 per
    # split (pop 1 parent, push 4 children), so only count == 1 (mod 3)
    # values are ever hit exactly -- see FixedOctTree's analogous %7 case.
    qdt = FixedQuadTree(domain=domain, fixed_length=10, score_fn="variance", min_size=4)
    for bbox, _ in qdt.nodes:
        w, h = bbox.get_size()
        assert w == 2 * h  # every leaf inherits the root's 64:32 == 2:1 aspect ratio

    seq_patch, seq_size, seq_pos = qdt.serialize(img, size=(16, 16, 3))
    assert len(seq_patch) == 10
    for patch in seq_patch:
        assert patch.shape == (16, 16, 3)
