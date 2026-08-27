# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


import numpy as np
import torch
import math
import torch.nn as nn

class SinusoidalEmbeddings(nn.Module):
    """Fixed sinusoidal timestep embeddings, e.g. for conditioning a diffusion model on `t`.

    Precomputes a `(time_steps, embed_dim)` table of sine/cosine embeddings and looks
    up rows by timestep index.
    """

    def __init__(self, time_steps:int, embed_dim: int):
        """Precomputes the sinusoidal embedding table.

        Args:
            time_steps: Number of distinct timesteps to precompute embeddings for.
            embed_dim: Dimensionality of each embedding vector.
        """
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        """Looks up the embedding for each timestep in `t`.

        Args:
            x: Reference tensor used only to determine the target device.
            t: Tensor of timestep indices to look up.

        Returns:
            Tensor of embeddings for `t`, moved to `x`'s device.
        """
        embeds = self.embeddings[t].to(x.device)
        return embeds


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, grid_size_d, cls_token=False):
    """Builds a fixed 3D sine-cosine positional embedding for a grid of patches.

    Splits `embed_dim` evenly across the h/w/d axes and concatenates a 1D sine-cosine
    embedding computed independently along each axis.

    Args:
        embed_dim: Total embedding dimension; must be divisible by 3.
        grid_size_h: Number of grid positions along the height axis.
        grid_size_w: Number of grid positions along the width axis.
        grid_size_d: Number of grid positions along the depth axis.
        cls_token: Unused; kept for interface compatibility with `get_2d_sincos_pos_embed`.

    Returns:
        Numpy array of shape [grid_size_h*grid_size_w*grid_size_d, embed_dim].
    """
    assert embed_dim % 3 == 0
    d_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, np.arange(grid_size_d)) 
    w_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, np.arange(grid_size_w)) 
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, np.arange(grid_size_h)) 

    d_pos_embed = np.tile(d_pos_embed, (grid_size_h * grid_size_w, 1))
    w_pos_embed = np.tile(np.repeat(w_pos_embed, grid_size_d, axis=0), (grid_size_h, 1))
    h_pos_embed = np.repeat(h_pos_embed, grid_size_w * grid_size_d, axis=0)

    emb = np.concatenate((h_pos_embed, w_pos_embed, d_pos_embed), axis=1)
    
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """Builds a fixed 2D sine-cosine positional embedding for a grid of patches.

    Args:
        embed_dim: Total embedding dimension; must be divisible by 2.
        grid_size_h: Number of grid positions along the height axis.
        grid_size_w: Number of grid positions along the width axis.
        cls_token: If True, prepend a zero embedding row for a class token.

    Returns:
        Numpy array of shape [grid_size_h*grid_size_w, embed_dim], or
        [1+grid_size_h*grid_size_w, embed_dim] if `cls_token` is True.
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Builds a 2D sine-cosine embedding from explicit h/w grid coordinates.

    Args:
        embed_dim: Total embedding dimension; must be divisible by 2. Half is used
            to encode the h coordinate and half the w coordinate.
        grid: Array of shape [2, 1, H, W] with h coordinates in `grid[0]` and w
            coordinates in `grid[1]`.

    Returns:
        Numpy array of shape [H*W, embed_dim].
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Builds a 1D sine-cosine embedding from a list of positions.

    Args:
        embed_dim: Output embedding dimension for each position; must be divisible
            by 2.
        pos: Array of positions to encode, shape (M,).

    Returns:
        Numpy array of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(pos_embed, orig_grid_size, new_grid_size, num_prefix_tokens=0):
    """Interpolates a 2D grid positional embedding to a new grid resolution.

    Reshapes the spatial part of `pos_embed` back into an `orig_grid_size` grid and
    bicubic-resizes it to `new_grid_size`, independently per axis -- works for any
    height:width ratio, not just square/fixed-ratio grids. `orig_grid_size` is taken
    as an explicit argument rather than guessed from `pos_embed`'s flattened length,
    since a flat count alone can't be decomposed back into two independent axis
    sizes without assuming a ratio.

    Args:
        pos_embed: Positional embedding tensor, shape `(1, num_prefix_tokens +
            orig_grid_size[0]*orig_grid_size[1], embed_dim)`.
        orig_grid_size: `(h, w)` grid `pos_embed` was originally built for (e.g. the
            pretrained model's own `grid_size`).
        new_grid_size: `(h, w)` grid to resize to (e.g. the new model's own
            `grid_size`).
        num_prefix_tokens: Number of leading non-spatial tokens (e.g. 1 for a class
            token) to leave untouched, ahead of the spatial grid.

    Returns:
        Resized positional embedding tensor, shape `(1, num_prefix_tokens +
        new_grid_size[0]*new_grid_size[1], embed_dim)`, or `pos_embed` unchanged if
        `orig_grid_size == new_grid_size`.
    """
    orig_grid_size = tuple(orig_grid_size)
    new_grid_size = tuple(new_grid_size)
    if orig_grid_size == new_grid_size:
        return pos_embed

    embedding_size = pos_embed.shape[-1]
    prefix_tokens = pos_embed[:, :num_prefix_tokens]
    grid_tokens = pos_embed[:, num_prefix_tokens:]

    pos_tokens = grid_tokens.reshape(-1, orig_grid_size[0], orig_grid_size[1], embedding_size).permute(0, 3, 1, 2)
    new_pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=new_grid_size, mode="bicubic", align_corners=False
    )
    new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat([prefix_tokens, new_pos_tokens], dim=1)


def interpolate_pos_embed_3d(pos_embed, orig_grid_size, new_grid_size, num_prefix_tokens=0):
    """Interpolates a 3D grid positional embedding to a new grid resolution.

    Same as `interpolate_pos_embed`, but for a 3D `(h, w, d)` grid (trilinear
    instead of bicubic) -- works for any height:width:depth ratio, independently
    per axis.

    Args:
        pos_embed: Positional embedding tensor, shape `(1, num_prefix_tokens +
            orig_grid_size[0]*orig_grid_size[1]*orig_grid_size[2], embed_dim)`.
        orig_grid_size: `(h, w, d)` grid `pos_embed` was originally built for.
        new_grid_size: `(h, w, d)` grid to resize to.
        num_prefix_tokens: Number of leading non-spatial tokens (e.g. 1 for a class
            token) to leave untouched, ahead of the spatial grid.

    Returns:
        Resized positional embedding tensor, shape `(1, num_prefix_tokens +
        new_grid_size[0]*new_grid_size[1]*new_grid_size[2], embed_dim)`, or
        `pos_embed` unchanged if `orig_grid_size == new_grid_size`.
    """
    orig_grid_size = tuple(orig_grid_size)
    new_grid_size = tuple(new_grid_size)
    if orig_grid_size == new_grid_size:
        return pos_embed

    embedding_size = pos_embed.shape[-1]
    prefix_tokens = pos_embed[:, :num_prefix_tokens]
    grid_tokens = pos_embed[:, num_prefix_tokens:]

    pos_tokens = grid_tokens.reshape(
        -1, orig_grid_size[0], orig_grid_size[1], orig_grid_size[2], embedding_size
    ).permute(0, 4, 1, 2, 3)
    new_pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=new_grid_size, mode="trilinear", align_corners=False
    )
    new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 4, 1).flatten(1, 3)
    return torch.cat([prefix_tokens, new_pos_tokens], dim=1)
