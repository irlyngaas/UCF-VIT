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
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, grid_size_d, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
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
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
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
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
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
def interpolate_pos_embed(model, checkpoint_model, new_size=(64, 128)):
    if "net.pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["net.pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]
        patch_size = model.patch_size
        w_h_ratio = 2
        orig_h = int((orig_num_patches // w_h_ratio) ** 0.5)
        orig_w = w_h_ratio * orig_h
        orig_size = (orig_h, orig_w)
        new_size = (new_size[0] // patch_size, new_size[1] // patch_size)
        # print (orig_size)
        # print (new_size)
        if orig_size[0] != new_size[0]:
            print("Interpolate PEs from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            pos_tokens = pos_embed_checkpoint.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(
                0, 3, 1, 2
            )
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode="bicubic", align_corners=False
            )
            new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model["net.pos_embed"] = new_pos_tokens


def interpolate_channel_embed(checkpoint_model, new_len):
    if "net.channel_embed" in checkpoint_model:
        channel_embed_checkpoint = checkpoint_model["net.channel_embed"]
        old_len = channel_embed_checkpoint.shape[1]
        if new_len <= old_len:
            checkpoint_model["net.channel_embed"] = channel_embed_checkpoint[:, :new_len]
