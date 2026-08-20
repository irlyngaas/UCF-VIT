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
    domain = np.zeros((8, 8), dtype=np.float64)
    domain[0:4, 0:4] = 255  # domain is indexed [y, x]

    dense = Rect(x1=0, x2=4, y1=0, y2=4)
    assert dense.contains(domain) == 16  # 4*4 px * 255 / 255

    empty = Rect(x1=4, x2=8, y1=4, y2=8)
    assert empty.contains(domain) == 0


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
