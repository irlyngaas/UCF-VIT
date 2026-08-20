import numpy as np
import pytest
import torch

from UCF_VIT.utils.pos_embed import (
    SinusoidalEmbeddings,
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


def test_sinusoidal_embeddings_lookup():
    embed_dim = 6
    time_steps = 10
    module = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=embed_dim)

    x = torch.zeros(2, 1)
    t = torch.tensor([0, 5])
    out = module(x, t)

    assert out.shape == (2, embed_dim)
    torch.testing.assert_close(out[0], module.embeddings[0])
    torch.testing.assert_close(out[1], module.embeddings[5])
    # timestep 0 -> sin(0)=0, cos(0)=1
    torch.testing.assert_close(out[0, 0::2], torch.zeros(embed_dim // 2))
    torch.testing.assert_close(out[0, 1::2], torch.ones(embed_dim // 2))
