
from functools import lru_cache, partial
from typing import Callable, Optional, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
import torch.nn as nn

#from UCF_VIT.simple.building_blocks import Block, PatchEmbed, Mlp, DropPath, AttentionPoolLatent, PatchDropout, \
from .building_blocks import Block, PatchEmbed, Mlp, DropPath, AttentionPoolLatent, PatchDropout, \
    trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    get_act_layer, get_norm_layer, LayerType, \
    MyUnetBlock, EmbeddingDenseLayer, \
    VariableMapping_Attention

from timm.models._manipulate import named_apply, checkpoint_seq

from UCF_VIT.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
    SinusoidalEmbeddings,
)
import torch.distributed as dist

from UCF_VIT.utils.dist_functions import F_Identity_B_Broadcast,F_Broadcast_B_Identity, F_Identity_B_AllReduce
from UCF_VIT.utils.fused_attn import FusedAttn

from einops import rearrange

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock

#from timm.models._features import feature_take_indices
#Hacked in since feature_take_indices isn't included in Timm Release, maybe update timm package in conda environment
def feature_take_indices(
        num_features: int,
        indices: Optional[Union[int, List[int]]] = None,
        as_set: bool = False,
) -> Tuple[List[int], int]:
    """ Determine the absolute feature indices to 'take' from.

    Note: This function can be called in forwar() so must be torchscript compatible,
    which requires some incomplete typing and workaround hacks.

    Args:
        num_features: total number of features to select from
        indices: indices to select,
          None -> select all
          int -> select last n
          list/tuple of int -> return specified (-ve indices specify from end)
        as_set: return as a set

    Returns:
        List (or set) of absolute (from beginning) indices, Maximum index
    """
    if indices is None:
        indices = num_features  # all features if None

    if isinstance(indices, int):
        # convert int -> last n indices
        torch._assert(0 < indices <= num_features, f'last-n ({indices}) is out of range (1 to {num_features})')
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            torch._assert(0 <= idx < num_features, f'feature index {idx} is out of range (0 to {num_features - 1})')
            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)

def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def get_init_weights_vit(head_bias: float = 0.0) -> Callable:
    return init_weights_vit_timm

def global_pool_nlc(
        x: torch.Tensor,
        num_prefix_tokens: int = 1,
):
    if num_prefix_tokens == 1:
        x = x[:, 0]  # class token
    else:
        x = x[:, num_prefix_tokens:]

    return x

class VIT(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int], Tuple[int,int,int]] = 224,
            patch_size: Union[int, Tuple[int, int], Tuple[int,int,int]] = 16,
            in_chans: int = 3,
            num_classes: Optional[int] = None,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', ''] = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            twoD: Optional[bool] = True,
            adaptive_patching: Optional[bool] = False,
            fixed_length: Optional[int] = 4096,
            default_vars: List = None,
            single_channel: bool = False,
            use_varemb: bool = False,
            tensor_par_size: int = 1,
            tensor_par_group: Optional[dist.ProcessGroup] = None,
            FusedAttn_option = FusedAttn.NONE,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            twoD: Variable for indicating two or three dimensionsal input, if False, three dimensional input.
            adaptive_patching: Whether to use adaptive patching
            fixed_length: Length for adaptive patches, only used if adative_patching=True
            default_vars: List of different potential modalities to be used as input.
            single_channel: Variable for indicating that multiple modalities will be used, but the model will be fed with modalities separated into batches only containing a single modality
            use_varemb: Whether to use variable embedding tokens as an additional learnable parameter
        """
        super().__init__()
        assert pos_embed in ('', 'none', 'learn')
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        self.norm_layer = norm_layer
        act_layer = get_act_layer(act_layer) or nn.GELU
        self.act_layer = act_layer
        self.mlp_layer = mlp_layer

        self.num_classes = num_classes
        self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.twoD = twoD
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_path_rate = drop_path_rate
        self.proj_drop_rate = proj_drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.init_values = init_values
        self.block_fn = block_fn
        self.img_size = img_size
        self.num_heads = num_heads
        self.depth = depth
        self.adaptive_patching = adaptive_patching
        self.fixed_length = fixed_length
        self.default_vars = default_vars
        self.single_channel = single_channel
        self.use_varemb = use_varemb
        self.aggregated_variables = 1 #Change this to an argument when adding different variable aggregation strategies
        self.class_token = class_token
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group
        self.FusedAttn_option = FusedAttn_option


        #ASSUMES INPUT HAS ALREADY BEEN ADAPTIVELY PATCHED
        if self.adaptive_patching:
            num_patches = self.fixed_length
            #TODO: throw error if using linear decoder in unetr
        else:
            if self.use_varemb:
                self.patch_embed = embed_layer(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=1,
                    embed_dim=embed_dim,
                    twoD=twoD,
                )
            else:
                self.patch_embed = embed_layer(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    twoD=twoD,
                )
            num_patches = self.patch_embed.num_patches
            grid_size = self.patch_embed.grid_size
            self.grid_size = grid_size
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches + self.num_prefix_tokens
        self.embed_len = self.num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, self.embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                fused_attn=FusedAttn_option,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                tensor_par_size=tensor_par_size,
                tensor_par_group=tensor_par_group,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        if num_classes != None:
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = None

        #ASSUMES INPUT HAS ALREADY BEEN ADAPTIVELY PATCHED
        if self.twoD:
            self.patch_dim = self.in_chans*self.patch_size**2
            self.patch_dim_woc = self.patch_size**2
        else:
            self.patch_dim = self.in_chans*self.patch_size**3
            self.patch_dim_woc = self.patch_size**3

        if self.adaptive_patching:
            #TODO: Find a way to do convolutional patch embedding with adaptive token input, PatchEmbed doesn't work correctly
            if self.use_varemb:
                self.token_embeds = nn.ModuleList(
                    [nn.Sequential(nn.LayerNorm(self.patch_dim_woc),nn.Linear(self.patch_dim_woc, self.embed_dim),nn.LayerNorm(self.embed_dim)) for i in range(len(self.default_vars))]
                )
            else:
                self.token_embeds = nn.Sequential(nn.LayerNorm(self.patch_dim),nn.Linear(self.patch_dim, self.embed_dim),nn.LayerNorm(self.embed_dim))
        else:
            if self.use_varemb:
                self.token_embeds = nn.ModuleList(
                    #[self.patch_embed(img_size=self.img_size, patch_size=self.patch_size, in_chans=1, embed_dim=self.embed_dim, twoD=self.twoD) for i in range(len(self.default_vars))]
                    [self.patch_embed for i in range(len(self.default_vars))]
                )
            else:
                #self.token_embeds = self.patch_embed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim, twoD=self.twoD)
                self.token_embeds = self.patch_embed

        if self.use_varemb:
            self.var_embed, self.var_map = self.create_var_embedding(self.embed_dim)
            if self.single_channel or len(self.default_vars) == 1:
                self.var_query = None
                self.var_agg = None
            else:
                self.var_query = nn.Parameter(torch.zeros(1, self.aggregated_variables, self.embed_dim), requires_grad=True)
                #TODO: Different parameter for specifying num_heads in var_agg rather than encoder num_heads
                #self.var_agg = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
                self.var_agg = VariableMapping_Attention(self.embed_dim, fused_attn=self.FusedAttn_option, num_heads=self.num_heads, qkv_bias=False, tensor_par_size = self.tensor_par_size, tensor_par_group = self.tensor_par_group)

        if weight_init != 'skip':
            self.init_weights('')

    def init_weights(self, mode: str = '') -> None:
        head_bias = 0.
        if not self.adaptive_patching:
            if self.pos_embed is not None:
                #trunc_normal_(self.pos_embed, std=.02)
                if self.twoD:
                    pos_embed = get_2d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        cls_token=self.class_token,
                    )
                else: #3D
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        int(self.img_size[2] / self.patch_size),
                        cls_token=self.class_token,
                    )
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
    
        if not self.adaptive_patching:
            if self.use_varemb:
                for i in range(len(self.token_embeds)):
                    w = self.token_embeds[i].proj.weight.data
                    trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
            else:
                w = self.token_embeds.proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        if self.use_varemb:
            var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
            self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        named_apply(get_init_weights_vit(head_bias), self)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))

        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed

        return self.pos_drop(x)

    def create_var_embedding(self, dim):
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1

        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        #var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        #x , _ = self.var_agg(var_query, x, x)  # BxL, V~ , D, where V~ is the aggregated variables
        var_query = self.var_query.expand(x.shape[0], -1, -1).contiguous()
        x = self.var_agg(var_query, x)  # BxL, V~ , D, where V~ is the aggregated variables
        x = x.squeeze()

        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, V~, D

        if self.aggregated_variables >1:
            x = rearrange(x,'b l v d -> b v l d')

        return x

    def forward_features(self, x: torch.Tensor, variables) -> torch.Tensor:
        if self.use_varemb:
            embeds = []
            if isinstance(variables, list):
                variables = tuple(variables)
            var_ids = self.get_var_ids(variables, x.device)
            for i in range(len(var_ids)):
                id = var_ids[i]
                if self.single_channel:
                    if self.adaptive_patching:
                        x = self.token_embeds[id](torch.squeeze(x)) # B, L, D 
                    else:
                        x = self.token_embeds[id](x) # B, L, D 
                    break #Should only be one channel
                else:
                    if self.adaptive_patching:
                        embeds.append(self.token_embeds[id](torch.squeeze(x[:,i : i+1])))
                    else:
                        embeds.append(self.token_embeds[id](x[:,i : i+1]))
                    
            var_embed = self.get_var_emb(self.var_embed, variables) # 1, V, D
            if not self.single_channel: #V > 1
                x = torch.stack(embeds, dim=1)  # B, L, D -> B, V, L, D
                x = x + var_embed.unsqueeze(2)  # 1, V, D -> 1, V, 1, D
                x = self.aggregate_variables(x)  # B, V~ , L, D, where V~ is the aggregated variables
            else: # V=1
                #x -> B, L, D
                var_embed = var_embed.unsqueeze(2) # 1, V=1, D -> 1, V=1, L=1, D
                x = x + var_embed.squeeze(1)  # 1, V=1, L=1, D -> 1, L=1, D
        else:
            if self.adaptive_patching:
                x = rearrange(x, 'b c s p -> b s (p c)')
                x = self.token_embeds(x)
            else:
                x = self.token_embeds(x)

        x = self._pos_embed(x)
        x = self.patch_drop(x)
        
        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            dist.broadcast(x, src_rank, group=self.tensor_par_group)

        x = self.blocks(x)
        x = self.norm(x)

        if self.tensor_par_size > 1:
            x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)
        
        return x

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        x = global_pool_nlc(x, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.head_drop(x)
        return x

    def forward(self, x: torch.Tensor, variables) -> torch.Tensor:
        x = self.forward_features(x, variables)
        x = self.forward_head(x)
        return x

class SAP(VIT):
    def __init__(self, *args, **kwargs):
        self.sqrt_len = kwargs.pop('sqrt_len', '')
        super().__init__(*args, **kwargs)
        #Remove decoder from VIT
        self.head = None 

        if self.twoD:
            self.neck = nn.Sequential(
                    nn.ConvTranspose2d(
                        self.embed_dim,
                        256,
                        kernel_size=(self.patch_size, self.patch_size),
                        stride=(self.patch_size, self.patch_size),
                        bias=False,
                    )
            )
            self.mask_header = nn.Sequential(nn.Conv2d(256, self.num_classes,1))
        else:
            self.neck = nn.Sequential(
                    nn.ConvTranspose3d(
                        self.embed_dim,
                        256,
                        kernel_size=(self.patch_size, self.patch_size, self.patch_size),
                        stride=(self.patch_size, self.patch_size, self.patch_size),
                        bias=False,
                    )
            )
            self.mask_header = nn.Sequential(nn.Conv3d(256, self.num_classes,1))

        self.init_weights('')

    def mask_head(self, x: torch.Tensor):
        if self.twoD:
            x = rearrange(x, 'b (p1 p2) c -> b p1 p2 c', p1=self.sqrt_len, p2=self.sqrt_len)
            x = self.neck(x.permute(0,3,1,2))
        else:
            x = rearrange(x, 'b (p1 p2 p3) c -> b p1 p2 p3 c', p1=self.sqrt_len, p2=self.sqrt_len, p3=self.sqrt_len)
            x = self.neck(x.permute(0,4,1,2,3))
            
        x = self.mask_header(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.mask_head(x)

class MAE(VIT):

    def __init__(self, *args, **kwargs):
        self.mask_ratio = kwargs.pop('mask_ratio', '')
        self.linear_decoder = kwargs.pop('linear_decoder', '')
        self.decoder_depth = kwargs.pop('decoder_depth', '')
        self.decoder_embed_dim = kwargs.pop('decoder_embed_dim', '')
        self.decoder_num_heads = kwargs.pop('decoder_num_heads', '')
        self.mlp_ratio_decoder = kwargs.pop('mlp_ratio_decoder', '')
        super().__init__(*args, **kwargs)
        #Remove decoder from VIT
        self.head = None 

        if self.linear_decoder:
            self.decoder_pred = nn.Linear(self.embed_dim, self.patch_dim)
            self.mask_token = nn.Parameter(torch.zeros(1,1,self.embed_dim))
        else:
            self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_dim)
            self.mask_token = nn.Parameter(torch.zeros(1,1,self.decoder_embed_dim))

        if not self.linear_decoder:
            self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim)
            self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
            if self.adaptive_patching:
                self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.decoder_embed_dim) * .02)
            else:
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.decoder_embed_dim))
            dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]  # stochastic depth decay rule
            #ASSUME same settings as Transformer Encoder for now
            self.decoder_blocks = nn.Sequential(*[
                self.block_fn(
                    dim=self.decoder_embed_dim,
                    num_heads=self.decoder_num_heads,
                    fused_attn=self.FusedAttn_option,
                    mlp_ratio=self.mlp_ratio_decoder,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    init_values=self.init_values,
                    proj_drop=self.proj_drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    mlp_layer=self.mlp_layer,
                )
                for i in range(self.decoder_depth)])
        else:
            self.decoder_pos_embed = None

        self.init_weights('')

    def init_weights(self, mode: str = '') -> None:
        head_bias = 0.
        if not self.adaptive_patching:
            if self.pos_embed is not None:
                #trunc_normal_(self.pos_embed, std=.02)
                if self.twoD:
                    pos_embed = get_2d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        cls_token=False,
                    )
                else: #3D
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        int(self.img_size[2] / self.patch_size),
                        cls_token=False,
                    )
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            if self.decoder_pos_embed is not None:
                if self.twoD:
                    decoder_pos_embed = get_2d_sincos_pos_embed(
                        self.decoder_pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        cls_token=False,
                    )
                else: #3D
                    decoder_pos_embed = get_3d_sincos_pos_embed(
                        self.decoder_pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        int(self.img_size[2] / self.patch_size),
                        cls_token=False,
                    )
                self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
    
        if not self.adaptive_patching:
            if self.use_varemb:
                for i in range(len(self.token_embeds)):
                    w = self.token_embeds[i].proj.weight.data
                    trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
            else:
                w = self.token_embeds.proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        if self.use_varemb:
            var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
            self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        named_apply(get_init_weights_vit(head_bias), self)

    def random_masking(self, sequence, noise=None):
        if self.aggregated_variables > 1:
            batch_size, channels, seq_length, dim = sequence.shape
        else:
            batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1-self.mask_ratio))

        if noise is None:
            if self.tensor_par_size > 1: #Synchronize noise to have the same masks across all data in a tensor parallel group
                if dist.get_rank(self.tensor_par_group) == 0:
                    noise = torch.rand(batch_size, seq_length, device=sequence.device)
                else:
                    noise = torch.rand(batch_size, seq_length, device=sequence.device)
                dist.broadcast(noise, src=(dist.get_rank()//self.tensor_par_size*self.tensor_par_size), group=self.tensor_par_group)
            else:
                noise = torch.rand(batch_size, seq_length, device=sequence.device)
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)

        ids_restore = torch.argsort(ids_shuffle,dim=1).to(sequence.device)
        ids_keep = ids_shuffle[:,:len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index = ids_restore)

        return sequence_unmasked, mask, ids_restore

    def mask_head(self, x: torch.Tensor, ids_restore):
        if not self.linear_decoder:
            x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x,mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[2]))
        if self.linear_decoder:
            x = self.decoder_pred(x_)
        else:
            x = x_ + self.decoder_pos_embed

            if self.tensor_par_size > 1:
                src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
                dist.broadcast(x, src_rank, group=self.tensor_par_group)

            x = self.decoder_blocks(x)
            x = self.decoder_norm(x)

            if self.tensor_par_size > 1:
                x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

            x = self.decoder_pred(x)
    
        return x

    def forward_features(self, x: torch.Tensor, variables) -> torch.Tensor:
        if self.use_varemb:
            embeds = []
            if isinstance(variables, list):
                variables = tuple(variables)
            var_ids = self.get_var_ids(variables, x.device)
            for i in range(len(var_ids)):
                id = var_ids[i]
                if self.single_channel:
                    if self.adaptive_patching:
                        x = self.token_embeds[id](torch.squeeze(x)) # B, L, D 
                    else:
                        x = self.token_embeds[id](x) # B, L, D 
                    break #Should only be one channel
                else:
                    if self.adaptive_patching:
                        embeds.append(self.token_embeds[id](torch.squeeze(x[:,i : i+1])))
                    else:
                        embeds.append(self.token_embeds[id](x[:,i : i+1]))
                    
            var_embed = self.get_var_emb(self.var_embed, variables) # 1, V, D
            if not self.single_channel: #V > 1
                x = torch.stack(embeds, dim=1)  # B, L, D -> B, V, L, D
                x = x + var_embed.unsqueeze(2)  # 1, V, D -> 1, V, 1, D
                x = self.aggregate_variables(x)  # B, V~ , L, D, where V~ is the aggregated variables
            else: # V=1
                #x -> B, L, D
                var_embed = var_embed.unsqueeze(2) # 1, V=1, D -> 1, V=1, L=1, D
                x = x + var_embed.squeeze(1)  # 1, V=1, L=1, D -> 1, L=1, D
        else:
            if self.adaptive_patching:
                x = rearrange(x, 'b c s p -> b s (p c)')
                x = self.token_embeds(x)
            else:
                x = self.token_embeds(x)
               
        x = self._pos_embed(x)
        x, mask, ids_restore = self.random_masking(x)
        x = self.patch_drop(x)

        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            dist.broadcast(x, src_rank, group=self.tensor_par_group)

        x = self.blocks(x)
        x = self.norm(x)

        if self.tensor_par_size > 1:
            x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        return x, mask, ids_restore

    def forward_head(self, x: torch.Tensor, ids_restore):
        x = self.pool(x)
        return self.mask_head(x, ids_restore)


    def forward(self, x: torch.Tensor, variables) -> torch.Tensor:
        x, mask, ids_restore = self.forward_features(x, variables)
        x = self.forward_head(x, ids_restore)
        return x, mask

class UNETR(VIT):

    def __init__(self, *args, **kwargs):
        self.linear_decoder = kwargs.pop('linear_decoder', '')
        self.feature_size = kwargs.pop('feature_size', '')
        self.skip_connection = kwargs.pop('skip_connection', '')
        super().__init__(*args, **kwargs)
        #Remove decoder from VIT
        self.head = None 

        if self.twoD:
            self.feat_size = (
                int(self.img_size[0] / self.patch_size),
                int(self.img_size[1] / self.patch_size),
            )
        else:
            self.feat_size = (
                int(self.img_size[0] / self.patch_size),
                int(self.img_size[1] / self.patch_size),
                int(self.img_size[2] / self.patch_size),
            )

        if not self.linear_decoder:
            if self.twoD:
                spatial_dims = 2
            else:
                spatial_dims = 3

            if self.skip_connection:
                increment_size = self.depth//4
                self.skip_indices = []
                for i in range(3):
                    self.skip_indices.append((i+1)*increment_size)
                #self.skip_indices = [3,6,9]
                #self.skip_indices = [6,12,18]
                #self.skip_indices = [8,16,24]

                self.encoder1 = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=self.in_chans,
                    out_channels=self.feature_size,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                if self.patch_size == 8:
                    self.encoder2 = UnetrPrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels=self.embed_dim, #Hidden_size
                        out_channels=self.feature_size * 2,
                        num_layer=2,
                        kernel_size=3,
                        stride=1,
                        upsample_kernel_size=1,
                        norm_name="instance",
                        conv_block=True,
                        res_block=True,
                    )
                else:
                    self.encoder2 = UnetrPrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels=self.embed_dim, #Hidden_size
                        out_channels=self.feature_size * 2,
                        num_layer=2,
                        kernel_size=3,
                        stride=1,
                        upsample_kernel_size=2,
                        norm_name="instance",
                        conv_block=True,
                        res_block=True,
                    )
                self.encoder3 = UnetrPrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=self.embed_dim, #Hidden_size
                    out_channels=self.feature_size * 4,
                    num_layer=1,
                    kernel_size=3,
                    stride=1,
                    upsample_kernel_size=2,
                    norm_name="instance",
                    conv_block=True,
                    res_block=True,
                )
                self.encoder4 = UnetrPrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=self.embed_dim, #Hidden_size
                    out_channels=self.feature_size * 8,
                    num_layer=0,
                    kernel_size=3,
                    stride=1,
                    upsample_kernel_size=2,
                    norm_name="instance",
                    conv_block=True,
                    res_block=True,
                )
                self.decoder5 = UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=self.embed_dim, #Hidden_size
                    out_channels= self.feature_size * 8, #feature_size=4
                    kernel_size=3, #Conv Kernel Size
                    upsample_kernel_size=2, #Conv Kernel Stride
                    norm_name="instance",
                    res_block=True,
                )
                self.decoder4 = UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels= self.feature_size * 8, #Out_channels from decoder5
                    out_channels= self.feature_size * 4, #feature_size=4
                    kernel_size=3, #Conv Kernel Size
                    upsample_kernel_size=2, #Conv Kernel Stride
                    norm_name="instance",
                    res_block=True,
                )
                if self.patch_size == 4:
                    self.decoder3 = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels= self.feature_size * 4, #Out_channels from decoder4
                        out_channels= self.feature_size * 2, #feature_size=4
                        kernel_size=3, #Conv Kernel Size
                        upsample_kernel_size=1, #Conv Kernel Stride
                        norm_name="instance",
                        res_block=True,
                    )
                else:
                    self.decoder3 = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels= self.feature_size * 4, #Out_channels from decoder4
                        out_channels= self.feature_size * 2, #feature_size=4
                        kernel_size=3, #Conv Kernel Size
                        upsample_kernel_size=2, #Conv Kernel Stride
                        norm_name="instance",
                        res_block=True,
                    )
                if self.patch_size == 8 or self.patch_size == 4:
                    self.decoder2 = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels= self.feature_size * 2, #Out_channels from decoder3
                        out_channels= self.feature_size, #feature_size=4
                        kernel_size=3, #Conv Kernel Size
                        upsample_kernel_size=1, #Conv Kernel Stride
                        norm_name="instance",
                        res_block=True,
                    )
                else: #self.patch_size == 16
                    self.decoder2 = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels= self.feature_size * 2, #Out_channels from decoder3
                        out_channels= self.feature_size, #feature_size=4
                        kernel_size=3, #Conv Kernel Size
                        upsample_kernel_size=2, #Conv Kernel Stride
                        norm_name="instance",
                        res_block=True,
                    )
            else:
                self.decoder5 = MyUnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=self.embed_dim, #Hidden_size
                    out_channels= self.feature_size * 8, #feature_size=4
                    upsample_kernel_size=2, #Conv Kernel Stride
                    res_block=True,
                )
                self.decoder4 = MyUnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels= self.feature_size * 8, #Out_channels from decoder5
                    out_channels= self.feature_size * 4, #feature_size=4
                    upsample_kernel_size=2, #Conv Kernel Stride
                    res_block=True,
                )
                if self.patch_size == 4:
                    self.decoder3 = MyUnetBlock(
                        spatial_dims=spatial_dims,
                        in_channels= self.feature_size * 4, #Out_channels from decoder4
                        out_channels= self.feature_size * 2, #feature_size=4
                        upsample_kernel_size=1, #Conv Kernel Stride
                        res_block=True,
                    )
                else:
                    self.decoder3 = MyUnetBlock(
                        spatial_dims=spatial_dims,
                        in_channels= self.feature_size * 4, #Out_channels from decoder4
                        out_channels= self.feature_size * 2, #feature_size=4
                        upsample_kernel_size=2, #Conv Kernel Stride
                        res_block=True,
                    )
                if self.patch_size == 8 or self.patch_size == 4:
                    self.decoder2 = MyUnetBlock(
                        spatial_dims=spatial_dims,
                        in_channels= self.feature_size * 2, #Out_channels from decoder3
                        out_channels= self.feature_size, #feature_size=4
                        upsample_kernel_size=1, #Conv Kernel Stride
                        res_block=True,
                    )
                else: #self.patch_size == 16
                    self.decoder2 = MyUnetBlock(
                        spatial_dims=spatial_dims,
                        in_channels= self.feature_size * 2, #Out_channels from decoder3
                        out_channels= self.feature_size, #feature_size=4
                        upsample_kernel_size=2, #Conv Kernel Stride
                        #norm_name="instance",
                        res_block=True,
                    )
            self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feature_size, out_channels=self.num_classes)

        else: #Use Linear Decoder
            self.mlp_head = nn.Linear(self.embed_dim, self.num_classes) 
            self.upsample = nn.Upsample(scale_factor=self.patch_size,mode='trilinear',align_corners=True)

        self.init_weights('')

    def proj_feat(self, x, hidden_size, feat_size):
        if self.twoD:
            x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
            x = x.permute(0,3,1,2)
        else:
            x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
            x = x.permute(0,4,1,2,3)
        return x

    def unetr_head(self, x: torch.Tensor, intermediates, enc1):

        if not self.skip_connection:
            if self.linear_decoder:
                x = self.mlp_head(x)
                if self.twoD:
                    x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1=self.grid_size[0], p2=self.grid_size[1])
                else:
                    x = rearrange(x, 'b (p1 p2 p3) c -> b c p1 p2 p3', p1=self.grid_size[0], p2=self.grid_size[1], p3=self.grid_size[2])
                x = self.upsample(x)

            else:
                x = self.proj_feat(x, self.embed_dim, self.feat_size)
                dec3 = self.decoder5(x)
                dec2 = self.decoder4(dec3)
                dec1 = self.decoder3(dec2)
                out = self.decoder2(dec1)
                x = self.out(out)
        else:
            int_len = len(intermediates)
            dec4 = self.proj_feat(x, self.embed_dim, self.feat_size)
            enc4 = self.encoder4(self.proj_feat(intermediates[int_len-1], self.embed_dim, self.feat_size))
            dec3 = self.decoder5(dec4, enc4)
            enc3 = self.encoder3(self.proj_feat(intermediates[int_len-2], self.embed_dim, self.feat_size))
            dec2 = self.decoder4(dec3, enc3)
            enc2 = self.encoder2(self.proj_feat(intermediates[int_len-3], self.embed_dim, self.feat_size))
            dec1 = self.decoder3(dec2, enc2)
            out = self.decoder2(dec1, enc1)
            x = self.out(out)
        return x

    def forward_intermediates(
            self,
            x: torch.Tensor,
            variables,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            intermediates_only: Only return intermediate features
        Returns:

        """
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        if self.use_varemb:
            embeds = []
            if isinstance(variables, list):
                variables = tuple(variables)
            var_ids = self.get_var_ids(variables, x.device)
            for i in range(len(var_ids)):
                id = var_ids[i]
                if self.single_channel:
                    if self.adaptive_patching:
                        x = self.token_embeds[id](torch.squeeze(x)) # B, L, D 
                    else:
                        x = self.token_embeds[id](x) # B, L, D 
                    break #Should only be one channel
                else:
                    if self.adaptive_patching:
                        embeds.append(self.token_embeds[id](torch.squeeze(x[:,i : i+1])))
                    else:
                        embeds.append(self.token_embeds[id](x[:,i : i+1]))
                    
            var_embed = self.get_var_emb(self.var_embed, variables) # 1, V, D
            if not self.single_channel: #V > 1
                x = torch.stack(embeds, dim=1)  # B, L, D -> B, V, L, D
                x = x + var_embed.unsqueeze(2)  # 1, V, D -> 1, V, 1, D
                x = self.aggregate_variables(x)  # B, V~ , L, D, where V~ is the aggregated variables
            else: # V=1
                #x -> B, L, D
                var_embed = var_embed.unsqueeze(2) # 1, V=1, D -> 1, V=1, L=1, D
                x = x + var_embed.squeeze(1)  # 1, V=1, L=1, D -> 1, L=1, D
        else:
            if self.adaptive_patching:
                x = rearrange(x, 'b c s p -> b s (p c)')
                x = self.token_embeds(x)
            else:
                x = self.token_embeds(x)

        x = self._pos_embed(x)
        x = self.patch_drop(x)

        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            dist.broadcast(x, src_rank, group=self.tensor_par_group)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]

        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        if self.tensor_par_size > 1:
            x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

            indexer = 0
            for i, blk in enumerate(blocks):
                if i in take_indices:
                    intermediates[indexer] = F_Identity_B_Broadcast(intermediates[indexer], src_rank, group=self.tensor_par_group)
                    indexer = indexer + 1

        return x, intermediates

    def forward_head(self, x: torch.Tensor, intermediates, enc1):
        x = self.pool(x)
        return self.unetr_head(x, intermediates, enc1)

    def forward(self, x: torch.Tensor, variables) -> torch.Tensor:
        if self.skip_connection:
            enc1 = self.encoder1(x)
            x, intermediates = self.forward_intermediates(x, variables, indices=self.skip_indices)
            x = self.forward_head(x, intermediates, enc1)
        else:
            enc1 = None
            x = self.forward_features(x, variables)
            intermediates = None
            x = self.forward_head(x, intermediates, enc1)
        return x

class DiffusionVIT(VIT):

    def __init__(self, *args, **kwargs):
        self.linear_decoder = kwargs.pop('linear_decoder', '')
        self.decoder_depth = kwargs.pop('decoder_depth', '')
        self.decoder_embed_dim = kwargs.pop('decoder_embed_dim', '')
        self.decoder_num_heads = kwargs.pop('decoder_num_heads', '')
        self.mlp_ratio_decoder = kwargs.pop('mlp_ratio_decoder', '')
        self.time_steps = kwargs.pop('time_steps', '')
        super().__init__(*args, **kwargs)
        #Remove decoder from VIT
        self.head = None 

        self.temporalEmbeddings = SinusoidalEmbeddings(time_steps=self.time_steps, embed_dim=self.embed_dim)
        self.timeEmbeddingMap = EmbeddingDenseLayer(self.embed_dim, self.embed_dim, 0.5) # dropout_prob = 0.5

        if self.linear_decoder:
            self.decoder_pred = nn.Linear(self.embed_dim, self.patch_dim)
        else:
            self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_dim)

        if not self.linear_decoder:
            self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim)
            self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
            if self.adaptive_patching:
                self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.decoder_embed_dim) * .02)
            else:
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.decoder_embed_dim))
            dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]  # stochastic depth decay rule
            #ASSUME same settings as Transformer Encoder for now
            self.decoder_blocks = nn.Sequential(*[
                self.block_fn(
                    dim=self.decoder_embed_dim,
                    num_heads=self.decoder_num_heads,
                    fused_attn=self.FusedAttn_option,
                    mlp_ratio=self.mlp_ratio_decoder,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    init_values=self.init_values,
                    proj_drop=self.proj_drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    mlp_layer=self.mlp_layer,
                )
                for i in range(self.decoder_depth)])
        else:
            self.decoder_pos_embed = None

        self.init_weights('')

    def init_weights(self, mode: str = '') -> None:
        head_bias = 0.
        if not self.adaptive_patching:
            if self.pos_embed is not None:
                #trunc_normal_(self.pos_embed, std=.02)
                if self.twoD:
                    pos_embed = get_2d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        cls_token=False,
                    )
                else: #3D
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        int(self.img_size[2] / self.patch_size),
                        cls_token=False,
                    )
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            if self.decoder_pos_embed is not None:
                if self.twoD:
                    decoder_pos_embed = get_2d_sincos_pos_embed(
                        self.decoder_pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        cls_token=False,
                    )
                else: #3D
                    decoder_pos_embed = get_3d_sincos_pos_embed(
                        self.decoder_pos_embed.shape[-1],
                        int(self.img_size[0] / self.patch_size),
                        int(self.img_size[1] / self.patch_size),
                        int(self.img_size[2] / self.patch_size),
                        cls_token=False,
                    )
                self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
    
        if not self.adaptive_patching:
            if self.use_varemb:
                for i in range(len(self.token_embeds)):
                    w = self.token_embeds[i].proj.weight.data
                    trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
            else:
                w = self.token_embeds.proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        if self.use_varemb:
            var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
            self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        named_apply(get_init_weights_vit(head_bias), self)

    def forward_features(self, x: torch.Tensor, t, variables) -> torch.Tensor:
        if self.use_varemb:
            embeds = []
            if isinstance(variables, list):
                variables = tuple(variables)
            var_ids = self.get_var_ids(variables, x.device)
            for i in range(len(var_ids)):
                id = var_ids[i]
                if self.single_channel:
                    if self.adaptive_patching:
                        x = self.token_embeds[id](torch.squeeze(x)) # B, L, D 
                    else:
                        x = self.token_embeds[id](x) # B, L, D 
                    break #Should only be one channel
                else:
                    if self.adaptive_patching:
                        embeds.append(self.token_embeds[id](torch.squeeze(x[:,i : i+1])))
                    else:
                        embeds.append(self.token_embeds[id](x[:,i : i+1]))
                    
            var_embed = self.get_var_emb(self.var_embed, variables) # 1, V, D
            if not self.single_channel: #V > 1
                x = torch.stack(embeds, dim=1)  # B, L, D -> B, V, L, D
                x = x + var_embed.unsqueeze(2)  # 1, V, D -> 1, V, 1, D
                x = self.aggregate_variables(x)  # B, V~ , L, D, where V~ is the aggregated variables
            else: # V=1
                #x -> B, L, D
                var_embed = var_embed.unsqueeze(2) # 1, V=1, D -> 1, V=1, L=1, D
                x = x + var_embed.squeeze(1)  # 1, V=1, L=1, D -> 1, L=1, D
        else:
            if self.adaptive_patching:
                x = rearrange(x, 'b c s p -> b s (p c)')
                x = self.token_embeds(x)
            else:
                x = self.token_embeds(x)
               
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        time_emb = self.temporalEmbeddings(x,t)
        time_emb = self.timeEmbeddingMap(time_emb.to(x.dtype))[:,None,:]
        x = x + time_emb

        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            dist.broadcast(x.contiguous(), src_rank, group=self.tensor_par_group)

        x = self.blocks(x)
        x = self.norm(x)

        if self.tensor_par_size > 1:
            x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        return x

    def forward_head(self, x: torch.Tensor):
        x = self.pool(x)
        if not self.linear_decoder:
            if self.tensor_par_size > 1:
                src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
                dist.broadcast(x, src_rank, group=self.tensor_par_group)

            x = self.decoder_embed(x)
            x = x + self.decoder_pos_embed
            x = self.decoder_blocks(x)
            x = self.decoder_norm(x)

            if self.tensor_par_size > 1:
                x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        return self.decoder_pred(x)

    def forward(self, x: torch.Tensor, t, variables) -> torch.Tensor:
        t = t.to('cpu')
        x = self.forward_features(x, t, variables)
        x = self.forward_head(x)
        return x
