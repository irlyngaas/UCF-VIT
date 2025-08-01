
import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist

from timm.layers.helpers import to_2tuple, to_3tuple
from timm.layers.trace_utils import _assert
from timm.layers import DropPath, AttentionPoolLatent, PatchDropout, \
    trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    get_act_layer, get_norm_layer, LayerType, use_fused_attn

#Keep these as references to where these functions came from in timm
#from timm.layers import PatchEmbed
#from timm.models.vision_transformer import Block
#from timm.models.vision_transformer import Attention
#from timm.layers import Mlp
from timm.models.vision_transformer import LayerScale
from monai.networks.blocks.dynunet_block import get_conv_layer

from UCF_VIT.utils.dist_functions import F_AllReduce_B_Identity, F_Identity_B_AllReduce, F_Identity_B_AllReduce_VariableMapping
from UCF_VIT.utils.fused_attn import FusedAttn

import xformers
from xformers.components.attention.core import scaled_dot_product_attention as xformers_sdpa


class PatchEmbed(nn.Module):
    """ 2D/3D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            twoD: Optional[bool] = True,
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
            tensor_par_size: int = 1,
            tensor_par_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.twoD = twoD
        if self.twoD:
            self.patch_size = to_2tuple(patch_size)
        else:
            self.patch_size = to_3tuple(patch_size)

        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        # flatten spatial dim and transpose to channels last, kept for bwd compat

        if self.twoD:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        else:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int], Tuple[int, int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        if self.twoD:
            img_size = to_2tuple(img_size)
        else:
            img_size = to_3tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        if self.twoD:
            num_patches = grid_size[0] * grid_size[1]
        else:
            num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        return img_size, grid_size, num_patches

    def forward(self, x):
        if self.twoD:
            B, C, H, W = x.shape
        else:
            B, C, H, W, D = x.shape
        if self.img_size is not None:
            _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
            _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            if not self.twoD:
                _assert(D == self.img_size[2], f"Input width ({D}) doesn't match model ({self.img_size[2]}).")
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC or NCHW -> NLC
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            tensor_par_size: int = 1,
            tensor_par_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group

        self.fc1 = linear_layer(in_features, hidden_features // self.tensor_par_size, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // self.tensor_par_size, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        if self.tensor_par_size > 1:
            x = F_Identity_B_AllReduce(x, group=self.tensor_par_group)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)

        if self.tensor_par_size > 1:
            x = F_AllReduce_B_Identity(x, op=dist.ReduceOp.SUM, group=self.tensor_par_group)

        return x

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            fused_attn: FusedAttn = FusedAttn.NONE,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            tensor_par_size: int = 1,
            tensor_par_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group

        self.qkv = nn.Linear(dim, dim * 3 // self.tensor_par_size, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // self.tensor_par_size, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        if self.tensor_par_size > 1:
            x = F_Identity_B_AllReduce(x, group=self.tensor_par_group)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads // self.tensor_par_size, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn == FusedAttn.FLASH:
            x = xformers.ops.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                p=self.attn_drop.p,
                op=xformers.ops.MemoryEfficientAttentionFlashAttentionOp
            )
        elif self.fused_attn == FusedAttn.CK:
            x = xformers.ops.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                p=self.attn_drop.p,
                op=xformers.ops.MemoryEfficientAttentionCkOp
            )
        elif self.fused_attn == FusedAttn.DEFAULT:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.transpose(1,2)
        else: # FusedAttn.NONE
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1,2)

        x = x.reshape(B, N, C // self.tensor_par_size)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.tensor_par_size > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tensor_par_group)
        
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            fused_attn: FusedAttn = FusedAttn.NONE,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            tensor_par_size: int = 1,
            tensor_par_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            fused_attn=fused_attn,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            tensor_par_size=tensor_par_size,
            tensor_par_group=tensor_par_group,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            tensor_par_size=tensor_par_size,
            tensor_par_group=tensor_par_group,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class MyUnetBlock(nn.Module):
    """     
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """     
        
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        #upsample_kernel_size: Sequence[int] | int,
        upsample_kernel_size: int,
        #norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

    def forward(self, inp):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        return out

class EmbeddingDenseLayer(nn.Module):
    def __init__(self, 
            c_in: int, 
            c_out: int,
            dropout_prob: float):
        super().__init__()
        self.linear1 = nn.Linear(c_in,c_out)
        self.linear2 = nn.Linear(c_out,c_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    #input here is gonna be just [B,C]
    def forward(self, x):
        return(self.linear2(self.dropout(self.relu(self.linear1(x)))))

class VariableMapping_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            fused_attn: FusedAttn = FusedAttn.NONE,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            tensor_par_size: int = 1,
            tensor_par_group = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group

        self.q = nn.Linear(dim, dim//tensor_par_size, bias=qkv_bias)

        self.kv = nn.Linear(dim, dim * 2 //tensor_par_size, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // tensor_par_size, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, var_query: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        if self.tensor_par_size >1:

            var_query= F_Identity_B_AllReduce_VariableMapping(var_query, group=self.tensor_par_group)
            x= F_Identity_B_AllReduce_VariableMapping(x, group=self.tensor_par_group)

        N_a = var_query.size(dim=1) #number of aggregated variables
        B, N_i, C = x.shape #B batch times sequence length, #N_i number of input variables, C embedding size

        q = self.q(var_query).reshape(B, N_a, self.num_heads // self.tensor_par_size, self.head_dim ).permute(0, 2, 1, 3)

        #print("var_query.shape",var_query.shape,"self.q",self.q,"q.shape",q.shape,flush=True)

        kv = self.kv(x).reshape(B, N_i, 2, self.num_heads // self.tensor_par_size, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn == FusedAttn.FLASH:
            x = xformers.ops.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                p=self.attn_drop.p,
                op=xformers.ops.MemoryEfficientAttentionFlashAttentionOp
            )
        elif self.fused_attn == FusedAttn.CK:
            x = xformers.ops.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                p=self.attn_drop.p,
                op=xformers.ops.MemoryEfficientAttentionCkOp
            )
        elif self.fused_attn == FusedAttn.DEFAULT:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.transpose(1, 2)
        else: # FusedAttn.NONE
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2)

        x = x.reshape(B, N_a, C//self.tensor_par_size)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.tensor_par_size >1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tensor_par_group)

        return x

