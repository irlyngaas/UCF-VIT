import numpy as np
import pytest

from UCF_VIT.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)


def test_get_1d_sincos_pos_embed_from_grid_shape_and_range():
    embed_dim = 8
    pos = np.arange(5)
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    assert emb.shape == (5, embed_dim)
    assert np.all(emb >= -1.0) and np.all(emb <= 1.0)
    # position 0 -> sin(0)=0 for every "sin" half, cos(0)=1 for every "cos" half
    np.testing.assert_allclose(emb[0, : embed_dim // 2], 0.0, atol=1e-8)
    np.testing.assert_allclose(emb[0, embed_dim // 2 :], 1.0, atol=1e-8)


def test_get_1d_sincos_pos_embed_from_grid_requires_even_dim():
    with pytest.raises(AssertionError):
        get_1d_sincos_pos_embed_from_grid(7, np.arange(3))


def test_get_2d_sincos_pos_embed_shape():
    emb = get_2d_sincos_pos_embed(embed_dim=8, grid_size_h=4, grid_size_w=3)
    assert emb.shape == (4 * 3, 8)


def test_get_2d_sincos_pos_embed_with_cls_token():
    emb = get_2d_sincos_pos_embed(embed_dim=8, grid_size_h=4, grid_size_w=3, cls_token=True)
    assert emb.shape == (1 + 4 * 3, 8)
    np.testing.assert_allclose(emb[0], 0.0)


def test_get_3d_sincos_pos_embed_shape():
    # embed_dim must be divisible by 3 (per-axis split) *and* each third must be
    # even (get_1d_sincos_pos_embed_from_grid's requirement) -> divisible by 6.
    emb = get_3d_sincos_pos_embed(embed_dim=12, grid_size_h=2, grid_size_w=3, grid_size_d=4)
    assert emb.shape == (2 * 3 * 4, 12)
