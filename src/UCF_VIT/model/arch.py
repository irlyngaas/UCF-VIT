
from functools import lru_cache, partial
from typing import Callable, Optional, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
import torch.nn as nn

from UCF_VIT.model.building_blocks import Block, PatchEmbed, Mlp, DropPath, AttentionPoolLatent, PatchDropout, \
    trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    get_act_layer, get_norm_layer, LayerType, \
    MyUnetBlock, VariableMapping_Attention

from timm.models._manipulate import named_apply, checkpoint_seq

from UCF_VIT.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
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
    """ViT weight initialization, original timm impl (for reproducibility).

    Truncated-normal initializes `nn.Linear` weights (zeroing biases), or
    delegates to a submodule's own `init_weights` method if it has one. Intended
    to be applied recursively via `named_apply`/`module.apply`.

    Args:
        module: Submodule to initialize.
        name: Fully-qualified name of `module` within the parent model; unused.
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def get_init_weights_vit(head_bias: float = 0.0) -> Callable:
    """Returns the weight-initialization function to apply across a ViT model.

    Args:
        head_bias: Unused; kept for interface compatibility.

    Returns:
        `init_weights_vit_timm`.
    """
    return init_weights_vit_timm

def global_pool_nlc(
        x: torch.Tensor,
        num_prefix_tokens: int = 1,
):
    """Pools a (Batch, N, Channel) sequence down to a single token per batch element.

    Args:
        x: Input sequence, shape (B, N, C).
        num_prefix_tokens: Number of leading prefix tokens (e.g. class tokens). If
            exactly 1, that single prefix token is returned; otherwise, all tokens
            after the prefix tokens are returned (no pooling reduction is applied).

    Returns:
        `x[:, 0]` if `num_prefix_tokens == 1`, otherwise `x[:, num_prefix_tokens:]`.
    """
    if num_prefix_tokens == 1:
        x = x[:, 0]  # class token
    else:
        x = x[:, num_prefix_tokens:]

    return x

class VIT(nn.Module):
    """Vision Transformer encoder (2D/3D), optionally with a classification head.

    Supports standard convolutional patch embedding or pre-adaptively-patched
    (quadtree/octree) input, per-variable/channel token embedding with attention-
    based channel aggregation, tensor-parallel sharding of attention/MLP layers,
    and either fixed sinusoidal or adaptive (position-dependent) positional
    embeddings. Used both standalone (for classification) and as the encoder base
    class for `SAP`, `MAE`, `UNETR`, and `DiffusionVIT`.
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int], Tuple[int,int,int]] = 224,
            patch_size: Union[int, Tuple[int, int], Tuple[int,int,int]] = 16,
            interp_size: Optional[int] = None,
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
            use_varemb: bool = False,
            tensor_par_size: int = 1,
            tensor_par_group: Optional[dist.ProcessGroup] = None,
            FusedAttn_option = FusedAttn.NONE,
            use_adaptive_pos_emb: bool = False,
            sqrt_len_method: bool = False,
            num_time_steps: int = None,
    ) -> None:
        """Builds the patch/token embedding, positional embedding, transformer blocks, and optional classification head.

        Args:
            img_size: Input image size.
            patch_size: Patch size. Only used when `adaptive_patching` is False;
                when `adaptive_patching` is True, `interp_size` is used instead
                (see `effective_patch_size`).
            interp_size: Side length each adaptive (quadtree/octree) leaf patch
                is interpolated to, and the size every dependent model-layer
                calculation is based on. Required when `adaptive_patching` is
                True; unused otherwise.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head; if None, no
                head is created (used when this class is a base for another head).
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            qk_norm: Whether to apply normalization to Q and K in attention.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            pos_embed: Positional embedding type; `''`/`'none'` disables it,
                `'learn'` creates a learned parameter (later re-initialized with a
                fixed sin-cos embedding in `init_weights`).
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            patch_drop_rate: Patch dropout rate (fraction of patch tokens randomly
                dropped); 0 disables patch dropout.
            proj_drop_rate: Dropout rate applied after attention/MLP projections.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme; `'skip'` to skip calling
                `init_weights` at the end of construction (e.g. when a subclass
                will call it itself after adding more layers).
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            mlp_layer: MLP layer used inside each transformer block.
            twoD: Variable for indicating two or three dimensionsal input, if False, three dimensional input.
            adaptive_patching: Whether to use adaptive patching
            fixed_length: Length for adaptive patches, only used if adative_patching=True
            default_vars: List of different potential modalities to be used as input.
            use_varemb: Whether to use variable embedding tokens as an additional learnable parameter
            tensor_par_size: Number of tensor-parallel ranks to shard attention/MLP
                layers across.
            tensor_par_group: Process group for tensor-parallel communication.
            FusedAttn_option: Which fused attention implementation the transformer
                blocks should use.
            use_adaptive_pos_emb: Whether to compute positional embeddings from
                each patch's adaptive size/position rather than using a fixed
                learned/sin-cos table.
            sqrt_len_method: Whether the (adaptively-patched) input is arranged as
                a dense square/cube grid, so a standard `embed_layer` can be used
                instead of the flattened linear token embedding path.
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
        self.interp_size = interp_size
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
        self.use_varemb = use_varemb
        self.aggregated_variables = 1 #Change this to an argument when adding different variable aggregation strategies
        self.class_token = class_token
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group
        self.FusedAttn_option = FusedAttn_option
        self.use_adaptive_pos_emb = use_adaptive_pos_emb
        self.sqrt_len_method = sqrt_len_method
        self.num_time_steps = num_time_steps

        if self.adaptive_patching:
            assert self.interp_size is not None, "interp_size is required when adaptive_patching is turned on"

        #ASSUMES INPUT HAS ALREADY BEEN ADAPTIVELY PATCHED
        if self.adaptive_patching and not self.sqrt_len_method:
            num_patches = self.fixed_length
            #TODO: throw error if using linear decoder in unetr
        else:
            if self.use_varemb:
                self.patch_embed = embed_layer(
                    img_size=img_size,
                    patch_size=self.effective_patch_size,
                    in_chans=1,
                    embed_dim=embed_dim,
                    twoD=twoD,
                    sqrt_len_method=sqrt_len_method,
                )
            else:
                self.patch_embed = embed_layer(
                    img_size=img_size,
                    patch_size=self.effective_patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    twoD=twoD,
                    sqrt_len_method=sqrt_len_method,
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
                num_time_steps=num_time_steps
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
            self.patch_dim = self.in_chans*self.effective_patch_size**2
            self.patch_dim_woc = self.effective_patch_size**2
        else:
            self.patch_dim = self.in_chans*self.effective_patch_size**3
            self.patch_dim_woc = self.effective_patch_size**3

        if self.adaptive_patching and not self.sqrt_len_method:
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
            self.var_query = nn.Parameter(torch.zeros(1, self.aggregated_variables, self.embed_dim), requires_grad=True)
            #TODO: Different parameter for specifying num_heads in var_agg rather than encoder num_heads
            #self.var_agg = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
            self.var_agg = VariableMapping_Attention(self.embed_dim, fused_attn=self.FusedAttn_option, num_heads=self.num_heads, qkv_bias=False, tensor_par_size = self.tensor_par_size, tensor_par_group = self.tensor_par_group)

        if self.use_adaptive_pos_emb:
            if self.twoD:
                self.adaptive_pos_dep_emb = nn.Sequential(
                    nn.Linear(in_features=3, out_features=self.embed_dim),
                    nn.GELU()
                )
            else:
                self.adaptive_pos_dep_emb = nn.Sequential(
                    nn.Linear(in_features=4, out_features=self.embed_dim),
                    nn.GELU()
                )


        if weight_init != 'skip':
            self.init_weights('')

    @property
    def effective_patch_size(self):
        """The size every patch/token-sizing calculation should actually use.

        `interp_size` when `adaptive_patching` is True (the size adaptive
        leaf patches are interpolated to, which every dependent model-layer
        size must match), `patch_size` otherwise. Keeps `self.patch_size`/
        `self.interp_size` themselves truthful to their raw config values.
        """
        return self.interp_size if self.adaptive_patching else self.patch_size

    def init_weights(self, mode: str = '') -> None:
        """Initializes positional embeddings, cls token, patch embedding weights, and all submodules.

        Overwrites the learned positional embedding with a fixed 2D/3D sin-cos
        embedding (when not adaptively patched or when `sqrt_len_method` is set),
        initializes the class token and variable embedding (if used), and applies
        `init_weights_vit_timm` recursively to every submodule.

        Args:
            mode: Unused; kept for interface compatibility with subclass overrides.
        """
        head_bias = 0.
        if not self.adaptive_patching or self.sqrt_len_method:
            if self.pos_embed is not None:
                #trunc_normal_(self.pos_embed, std=.02)
                if self.twoD:
                    pos_embed = get_2d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        int(self.img_size[0] / self.effective_patch_size),
                        int(self.img_size[1] / self.effective_patch_size),
                        cls_token=self.class_token,
                    )
                else: #3D
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        int(self.img_size[0] / self.effective_patch_size),
                        int(self.img_size[1] / self.effective_patch_size),
                        int(self.img_size[2] / self.effective_patch_size),
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

    def _pos_embed(self, x: torch.Tensor, seq_ps) -> torch.Tensor:
        """Prepends the class token (if any) and adds positional embeddings to the patch sequence.

        Args:
            x: Patch token sequence, shape (B, N, embed_dim).
            seq_ps: Per-patch size/position tensor used to compute adaptive
                positional embeddings, when `self.use_adaptive_pos_emb` is True.

        Returns:
            Token sequence with class token prepended and positional embeddings
            added, after position dropout. If `self.pos_embed` is None, `x` is
            returned reshaped to (B, -1, embed_dim) with no positional embedding
            added.
        """
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.use_adaptive_pos_emb:
            pos_embed = self.adaptive_pos_dep_emb(seq_ps)
            if self.cls_token is not None:
                # adaptive_pos_dep_emb computes one position embedding per real
                # patch from seq_ps, with no row for the class token (it has no
                # spatial position of its own) -- pos_embed is otherwise one
                # token short once the class token gets prepended to x below.
                # Prepending a zero row mirrors get_2d_sincos_pos_embed/
                # get_3d_sincos_pos_embed's own cls_token handling (utils/
                # pos_embed.py) for the non-adaptive case: "prepend a zero
                # embedding row for a class token".
                cls_pos_embed = torch.zeros(
                    pos_embed.shape[0], 1, pos_embed.shape[-1], device=pos_embed.device, dtype=pos_embed.dtype
                )
                pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)
        else:
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
        """Creates a learned embedding parameter and name-to-index map for each variable in `self.default_vars`.

        Args:
            dim: Embedding dimension for each variable.

        Returns:
            A tuple `(var_embed, var_map)`: `var_embed` is a
            `(1, len(default_vars), dim)` parameter, and `var_map` maps each
            variable name to its row index in `var_embed`.
        """
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1

        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        """Looks up the embedding row index for each variable name, cached per `(vars, device)`.

        Args:
            vars: Tuple of variable names to look up in `self.var_map`.
            device: Device to place the resulting index tensor on.

        Returns:
            LongTensor of indices into `self.var_embed`, one per entry in `vars`.
        """
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        """Selects the rows of `var_emb` corresponding to `vars`.

        Args:
            var_emb: Variable embedding table, shape (1, len(default_vars), D).
            vars: Variable names to select embeddings for.

        Returns:
            Tensor of shape (1, len(vars), D).
        """
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor):
        """Cross-attends over the per-variable dimension to aggregate a variable number of input channels into a fixed set.

        Args:
            x: Per-variable token sequence, shape (B, V, L, D).

        Returns:
            Aggregated token sequence, shape (B, L, D) if
            `self.aggregated_variables == 1`, otherwise (B, V~, L, D) where V~ is
            `self.aggregated_variables`.
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

    def forward_features(self, x: torch.Tensor, variables, seq_ps) -> torch.Tensor:
        """Embeds patches/tokens, adds positional embeddings, and runs them through the transformer encoder.

        When `self.use_varemb` is set, tokenizes each input channel separately,
        adds its variable embedding, and aggregates channels via
        `aggregate_variables` before the encoder. When `self.tensor_par_size > 1`,
        broadcasts the embedded sequence to the rest of the tensor-parallel group
        before the encoder blocks and re-broadcasts the encoder output afterward.

        Args:
            x: Input patch/pixel tensor (raw image or pre-tokenized adaptive-patch
                sequence, depending on `self.adaptive_patching`).
            variables: Variable/channel names corresponding to `x`'s channel
                dimension, used to look up variable embeddings when
                `self.use_varemb` is set.
            seq_ps: Per-patch size/position tensor, used for adaptive positional
                embeddings.

        Returns:
            Encoded token sequence, shape (B, N[+prefix], embed_dim).
        """
        if self.use_varemb:
            embeds = []
            if isinstance(variables, list):
                variables = tuple(variables)
            var_ids = self.get_var_ids(variables, x.device)
            for i in range(len(var_ids)):
                id = var_ids[i]
                if self.adaptive_patching:
                    embeds.append(self.token_embeds[id](torch.squeeze(x[:,i : i+1])))
                else:
                    embeds.append(self.token_embeds[id](x[:,i : i+1]))

            var_embed = self.get_var_emb(self.var_embed, variables) # 1, V, D
            x = torch.stack(embeds, dim=1)  # B, L, D -> B, V, L, D
            x = x + var_embed.unsqueeze(2)  # 1, V, D -> 1, V, 1, D
            x = self.aggregate_variables(x)  # B, V~ , L, D, where V~ is the aggregated variables
        else:
            if self.adaptive_patching and not self.sqrt_len_method:
                x = rearrange(x, 'b c s p -> b s (p c)')
                x = self.token_embeds(x)
            else:
                x = self.token_embeds(x)

        x = self._pos_embed(x, seq_ps)
        x = self.patch_drop(x)

        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            # dist.broadcast requires a contiguous tensor, and fills it
            # in place -- must reassign x = x.contiguous() first (not just
            # pass x.contiguous() inline), since .contiguous() returns a
            # NEW tensor whenever x isn't already contiguous; broadcasting
            # that unassigned copy would silently leave the original x
            # variable un-updated on every non-src rank. _pos_embed's
            # cls-token torch.cat (which would otherwise produce a fresh,
            # contiguous tensor) only runs when self.cls_token is not None
            # -- i.e. only for model_type "VIT" (get_model's
            # class_token=True if conf["model"]["type"] == "VIT" else
            # False). For every other model type, x here is whatever
            # self.token_embeds(x) produced, typically PatchEmbed's own
            # flatten+transpose, which is non-contiguous. Real Frontier
            # runs (basic_ct-unetr+twoD+tensor_par, basic_ct-sap+tensor_par)
            # hit "ValueError: Tensors must be contiguous" here once
            # get_model actually started wiring tensor_par_size into the
            # model (this branch was previously dead code).
            x = x.contiguous()
            dist.broadcast(x, src_rank, group=self.tensor_par_group)

        x = self.blocks(x)
        x = self.norm(x)

        if self.tensor_par_size > 1:
            x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        return x

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the token sequence to a single per-sample representation.

        Args:
            x: Token sequence, shape (B, N[+prefix], C).

        Returns:
            Pooled tensor; see `global_pool_nlc`.
        """
        x = global_pool_nlc(x, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pools the encoder output and applies the classification head.

        Args:
            x: Encoder output token sequence.

        Returns:
            Classification logits, shape (B, num_classes), or `x` pooled and
            dropout-applied if `self.head` is `nn.Identity()`.
        """
        x = self.pool(x)
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, x: torch.Tensor, variables, seq_ps=None) -> torch.Tensor:
        """Runs the full encoder + classification head forward pass.

        Args:
            x: Input patch/pixel tensor.
            variables: Variable/channel names for `x`.
            seq_ps: Per-patch size/position tensor for adaptive positional
                embeddings.

        Returns:
            Classification logits, shape (B, num_classes).
        """
        x = self.forward_features(x, variables, seq_ps)
        x = self.forward_head(x)
        return x

class SAP(VIT):
    """Segmentation-via-Adaptive-Patching model: a `VIT` encoder with a transposed-conv decoder head.

    Reshapes the encoder's flat patch sequence back into a dense
    `sqrt_len x sqrt_len[ x sqrt_len]` grid and upsamples it via a strided
    transposed convolution ("neck") followed by a 1x1 conv classifier
    ("mask_header") to produce a dense per-pixel segmentation mask.
    """

    def __init__(self, *args, **kwargs):
        """Builds the `VIT` encoder, then replaces its classification head with a segmentation decoder.

        Args:
            *args: Positional arguments forwarded to `VIT.__init__`.
            **kwargs: Keyword arguments forwarded to `VIT.__init__`; must include
                `sqrt_len`, the grid side length the flat adaptive-patch sequence
                is reshaped to.
        """
        self.sqrt_len = kwargs.pop('sqrt_len', '')
        super().__init__(*args, **kwargs)
        #Remove decoder from VIT
        self.head = None 

        if self.twoD:
            self.neck = nn.Sequential(
                    nn.ConvTranspose2d(
                        self.embed_dim,
                        256,
                        kernel_size=(self.effective_patch_size, self.effective_patch_size),
                        stride=(self.effective_patch_size, self.effective_patch_size),
                        bias=False,
                    )
            )
            self.mask_header = nn.Sequential(nn.Conv2d(256, self.num_classes,1))
        else:
            self.neck = nn.Sequential(
                    nn.ConvTranspose3d(
                        self.embed_dim,
                        256,
                        kernel_size=(self.effective_patch_size, self.effective_patch_size, self.effective_patch_size),
                        stride=(self.effective_patch_size, self.effective_patch_size, self.effective_patch_size),
                        bias=False,
                    )
            )
            self.mask_header = nn.Sequential(nn.Conv3d(256, self.num_classes,1))

        self.init_weights('')

    def mask_head(self, x: torch.Tensor):
        """Reshapes the flat patch sequence into a grid and decodes it into a dense segmentation mask.

        Args:
            x: Pooled encoder output, flat patch sequence of length
                `sqrt_len**2` (2D) or `sqrt_len**3` (3D).

        Returns:
            Per-class segmentation logits, shape (B, num_classes, H, W[, D]).
        """
        if self.twoD:
            x = rearrange(x, 'b (p1 p2) c -> b p1 p2 c', p1=self.sqrt_len, p2=self.sqrt_len)
            x = self.neck(x.permute(0,3,1,2))
        else:
            x = rearrange(x, 'b (p1 p2 p3) c -> b p1 p2 p3 c', p1=self.sqrt_len, p2=self.sqrt_len, p3=self.sqrt_len)
            x = self.neck(x.permute(0,4,1,2,3))
            
        x = self.mask_header(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pools the encoder output and decodes it into a segmentation mask.

        Args:
            x: Encoder output token sequence.

        Returns:
            Per-class segmentation logits, shape (B, num_classes, H, W[, D]).
        """
        x = self.pool(x)
        return self.mask_head(x)

class MAE(VIT):

    def __init__(self, *args, **kwargs):
        """Builds the `VIT` encoder, then adds a masked-token decoder for reconstruction pretraining.

        If `linear_decoder` is True, decodes directly with a single linear layer;
        otherwise builds a separate (smaller) transformer decoder with its own
        positional embedding and blocks.

        Args:
            *args: Positional arguments forwarded to `VIT.__init__`.
            **kwargs: Keyword arguments forwarded to `VIT.__init__`; must include
                `mask_ratio` (fraction of patches to mask), `linear_decoder`
                (whether to use a single linear decoder), and, when
                `linear_decoder` is False, `decoder_depth`, `decoder_embed_dim`,
                `decoder_num_heads`, and `decoder_mlp_ratio`.
        """
        self.mask_ratio = kwargs.pop('mask_ratio', '')
        self.linear_decoder = kwargs.pop('linear_decoder', '')
        self.decoder_depth = kwargs.pop('decoder_depth', '')
        self.decoder_embed_dim = kwargs.pop('decoder_embed_dim', '')
        self.decoder_num_heads = kwargs.pop('decoder_num_heads', '')
        self.decoder_mlp_ratio = kwargs.pop('decoder_mlp_ratio', '')
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
            if self.use_adaptive_pos_emb:
                self.decoder_pos_embed = None
            else:
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
                    mlp_ratio=self.decoder_mlp_ratio,
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

            if self.use_adaptive_pos_emb:
                if self.twoD:
                    self.decoder_adaptive_pos_dep_emb = nn.Sequential(
                        nn.Linear(in_features=3, out_features=self.decoder_embed_dim),
                        nn.GELU()
                    )
                else:
                    self.decoder_adaptive_pos_dep_emb = nn.Sequential(
                        nn.Linear(in_features=4, out_features=self.decoder_embed_dim),
                        nn.GELU()
                    )
        else:
            self.decoder_pos_embed = None

        self.init_weights('')

    def init_weights(self, mode: str = '') -> None:
        """Initializes encoder and decoder positional embeddings, cls token, patch embedding weights, and all submodules.

        Like `VIT.init_weights`, but also initializes `self.decoder_pos_embed`
        with a fixed sin-cos embedding when it exists.

        Args:
            mode: Unused; kept for interface compatibility.
        """
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
        """Randomly masks a fraction of patch tokens, keeping only `1 - mask_ratio` of them.

        When tensor parallelism is enabled, the random noise used to determine the
        mask is broadcast from rank 0 of the tensor-parallel group so every rank
        masks the same positions.

        Args:
            sequence: Patch token sequence, shape (B, L, D) or, with channel
                aggregation, (B, C, L, D).
            noise: Optional precomputed noise tensor, shape (B, L), used to
                determine the shuffle order; if None, random noise is generated.

        Returns:
            A tuple `(sequence_unmasked, mask, ids_restore)`: `sequence_unmasked`
            is the kept (unmasked) subset of tokens, shape (B, len_keep, D); `mask`
            is a (B, L) binary tensor (0 = kept, 1 = masked) in original token
            order; `ids_restore` are the indices to unshuffle tokens back to
            original order.
        """
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

    def mask_head(self, x: torch.Tensor, ids_restore, seq_ps):
        """Reinserts mask tokens and decodes the full patch sequence back into pixel-space predictions.

        Refills the masked positions (dropped in `random_masking`) with a learned
        mask token, restores original token order via `ids_restore`, then either
        applies a single linear projection (`self.linear_decoder`) or runs a full
        transformer decoder followed by the linear prediction head.

        Args:
            x: Encoded (unmasked-only) token sequence, shape (B, len_keep, D) (or
                already `decoder_embed`-projected if `not self.linear_decoder`).
            ids_restore: Indices to unshuffle tokens back to original order, as
                returned by `random_masking`.
            seq_ps: Per-patch size/position tensor, used for adaptive decoder
                positional embeddings.

        Returns:
            Reconstructed per-patch pixel values, shape (B, L, patch_dim).
        """
        if not self.linear_decoder:
            x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x,mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[2]))
        if self.linear_decoder:
            x = self.decoder_pred(x_)
        else:
            if self.use_adaptive_pos_emb:
                decoder_pos_embed = self.decoder_adaptive_pos_dep_emb(seq_ps)
            else:
                decoder_pos_embed = self.decoder_pos_embed
            x = x_ + decoder_pos_embed

            if self.tensor_par_size > 1:
                src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
                dist.broadcast(x, src_rank, group=self.tensor_par_group)

            x = self.decoder_blocks(x)
            x = self.decoder_norm(x)

            if self.tensor_par_size > 1:
                x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

            x = self.decoder_pred(x)
    
        return x

    def forward_features(self, x: torch.Tensor, variables, seq_ps) -> torch.Tensor:
        """Embeds patches/tokens, adds positional embeddings, randomly masks, and runs the encoder.

        Like `VIT.forward_features`, but applies `random_masking` after the
        positional embedding so only the unmasked tokens are processed by the
        encoder blocks.

        Args:
            x: Input patch/pixel tensor.
            variables: Variable/channel names for `x`.
            seq_ps: Per-patch size/position tensor for adaptive positional
                embeddings.

        Returns:
            A tuple `(x, mask, ids_restore)`: `x` is the encoded unmasked token
            sequence, `mask` the binary mask (0=kept, 1=masked) in original token
            order, and `ids_restore` the indices to restore original token order.
        """
        if self.use_varemb:
            embeds = []
            if isinstance(variables, list):
                variables = tuple(variables)
            var_ids = self.get_var_ids(variables, x.device)
            for i in range(len(var_ids)):
                id = var_ids[i]
                if self.adaptive_patching:
                    embeds.append(self.token_embeds[id](torch.squeeze(x[:,i : i+1])))
                else:
                    embeds.append(self.token_embeds[id](x[:,i : i+1]))

            var_embed = self.get_var_emb(self.var_embed, variables) # 1, V, D
            x = torch.stack(embeds, dim=1)  # B, L, D -> B, V, L, D
            x = x + var_embed.unsqueeze(2)  # 1, V, D -> 1, V, 1, D
            x = self.aggregate_variables(x)  # B, V~ , L, D, where V~ is the aggregated variables
        else:
            if self.adaptive_patching:
                x = rearrange(x, 'b c s p -> b s (p c)')
                x = self.token_embeds(x)
            else:
                x = self.token_embeds(x)

        x = self._pos_embed(x, seq_ps)
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

    def forward_head(self, x: torch.Tensor, ids_restore, seq_ps):
        """Pools the encoder output and reconstructs the full (unmasked+masked) patch sequence.

        Args:
            x: Encoded unmasked token sequence.
            ids_restore: Indices to restore original token order.
            seq_ps: Per-patch size/position tensor for adaptive decoder positional
                embeddings.

        Returns:
            Reconstructed per-patch pixel values, shape (B, L, patch_dim).
        """
        x = self.pool(x)
        return self.mask_head(x, ids_restore, seq_ps)


    def forward(self, x: torch.Tensor, variables, seq_ps=None) -> torch.Tensor:
        """Runs the full masked-autoencoding forward pass: mask, encode, decode/reconstruct.

        Args:
            x: Input patch/pixel tensor.
            variables: Variable/channel names for `x`.
            seq_ps: Per-patch size/position tensor for adaptive positional
                embeddings.

        Returns:
            A tuple `(x, mask)`: `x` is the reconstructed per-patch pixel values
            and `mask` is the binary mask (0=kept, 1=masked) in original token
            order.
        """
        x, mask, ids_restore = self.forward_features(x, variables, seq_ps)
        x = self.forward_head(x, ids_restore, seq_ps)
        return x, mask

class UNETR(VIT):
    """UNETR segmentation model: a `VIT` encoder with a convolutional U-Net-style decoder.

    Reshapes intermediate encoder feature maps back into spatial grids and
    progressively upsamples/decodes them (optionally with U-Net-style skip
    connections from convolutional encoder stages, à la
    "UNETR: Transformers for 3D Medical Image Segmentation") into a dense
    per-pixel segmentation mask. Also supports a simpler linear-decoder mode that
    skips the convolutional decoder entirely.
    """

    def __init__(self, *args, **kwargs):
        """Builds the `VIT` encoder, then adds the convolutional (or linear) segmentation decoder.

        Args:
            *args: Positional arguments forwarded to `VIT.__init__`.
            **kwargs: Keyword arguments forwarded to `VIT.__init__`; must include
                `linear_decoder` (whether to use a single linear decoder instead
                of the convolutional U-Net decoder), `feature_size` (base channel
                count for the convolutional decoder), `skip_connection` (whether
                to add U-Net-style convolutional skip connections from
                intermediate encoder features), and, when adaptive patching is
                used, `sqrt_len` (the grid side length the flat adaptive-patch
                sequence is reshaped to).
        """
        self.linear_decoder = kwargs.pop('linear_decoder', '')
        self.feature_size = kwargs.pop('feature_size', '')
        self.skip_connection = kwargs.pop('skip_connection', '')
        self.sqrt_len = kwargs.pop('sqrt_len', '')
        super().__init__(*args, **kwargs)
        #Remove decoder from VIT
        self.head = None 

        if self.adaptive_patching:
            if self.twoD:
                self.feat_size = (
                    self.sqrt_len,
                    self.sqrt_len,
                )
            else:
                self.feat_size = (
                    self.sqrt_len,
                    self.sqrt_len,
                    self.sqrt_len,
                )
        else:
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

                self.decoder3 = UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels= self.feature_size * 4, #Out_channels from decoder4
                    out_channels= self.feature_size * 2, #feature_size=4
                    kernel_size=3, #Conv Kernel Size
                    upsample_kernel_size=2, #Conv Kernel Stride
                    norm_name="instance",
                    res_block=True,
                )

                if self.feat_size[0]*16 == self.img_size[0]:
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
                    self.decoder2 = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels= self.feature_size * 2, #Out_channels from decoder3
                        out_channels= self.feature_size, #feature_size=4
                        kernel_size=3, #Conv Kernel Size
                        upsample_kernel_size=1, #Conv Kernel Stride
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

                self.decoder3 = MyUnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels= self.feature_size * 4, #Out_channels from decoder4
                    out_channels= self.feature_size * 2, #feature_size=4
                    upsample_kernel_size=2, #Conv Kernel Stride
                    res_block=True,
                )

                self.decoder2 = MyUnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels= self.feature_size * 2, #Out_channels from decoder3
                    out_channels= self.feature_size, #feature_size=4
                    upsample_kernel_size=2, #Conv Kernel Stride
                    res_block=True,
                )
            self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feature_size, out_channels=self.num_classes)

            if self.feat_size[0]*16 != self.img_size[0]:
                if self.twoD:
                    self.upsample = nn.Upsample(size=self.img_size,mode='bilinear',align_corners=True)
                else:
                    self.upsample = nn.Upsample(size=self.img_size,mode='trilinear',align_corners=True)

        else: #Use Linear Decoder
            self.mlp_head = nn.Linear(self.embed_dim, self.num_classes) 
            if self.twoD:
                self.upsample = nn.Upsample(scale_factor=self.effective_patch_size,mode='bilinear',align_corners=True)
            else:
                self.upsample = nn.Upsample(scale_factor=self.effective_patch_size,mode='trilinear',align_corners=True)

        self.init_weights('')

    def proj_feat(self, x, hidden_size, feat_size):
        """Reshapes a flat patch-token sequence back into a spatial feature map.

        Args:
            x: Flat token sequence, shape (B, prod(feat_size), hidden_size).
            hidden_size: Channel dimension of each token.
            feat_size: Target spatial grid size, `(H, W)` for 2D or `(H, W, D)`
                for 3D.

        Returns:
            Spatial feature map, shape (B, hidden_size, H, W[, D]).
        """
        if self.twoD:
            x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
            x = x.permute(0,3,1,2)
        else:
            x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
            x = x.permute(0,4,1,2,3)
        return x

    def unetr_head(self, x: torch.Tensor, intermediates, enc1):
        """Decodes the encoder output (and intermediate features) into a segmentation mask.

        With `self.linear_decoder`, applies a single linear layer and upsamples.
        Without skip connections, runs the pooled encoder output through a stack
        of upsampling decoder blocks. With skip connections, additionally fuses
        each decoder stage with a convolutional encoder feature computed from the
        corresponding `intermediates` entry (and the original-resolution `enc1`
        feature at the final stage).

        Args:
            x: Pooled encoder output token sequence.
            intermediates: List of intermediate encoder token sequences at the
                configured `skip_indices`, as returned by `forward_intermediates`;
                only used when `self.skip_connection` is True.
            enc1: Original-resolution convolutional encoder feature (from
                `self.encoder1`), used as the final skip connection; only used
                when `self.skip_connection` is True.

        Returns:
            Per-class segmentation logits, shape (B, num_classes, H, W[, D]).
        """

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
                if self.feat_size[0]*16 != self.img_size[0]:
                    out = self.upsample(out)
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
            if self.feat_size[0]*16 != self.img_size[0]:
                dec1 = self.upsample(dec1)
            out = self.decoder2(dec1, enc1)
            x = self.out(out)
        return x

    def forward_intermediates(
            self,
            x: torch.Tensor,
            variables,
            seq_ps,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Embeds patches/tokens and positional embeddings as in `forward_features`,
        then runs the transformer blocks one at a time, collecting the output of
        each block whose index is in `take_indices` (computed from `indices` via
        `feature_take_indices`).

        Args:
            x: Input image tensor
            variables: Variable/channel names for `x`.
            seq_ps: Per-patch size/position tensor for adaptive positional
                embeddings.
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            intermediates_only: Only return intermediate features
        Returns:
            If `intermediates_only` is True: just the list of intermediate
            tensors (or `(spatial, prefix)` tuples if `return_prefix_tokens` is
            True). Otherwise: a tuple `(x, intermediates)` where `x` is the final
            normalized encoder output.
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
                if self.adaptive_patching:
                    embeds.append(self.token_embeds[id](torch.squeeze(x[:,i : i+1])))
                else:
                    embeds.append(self.token_embeds[id](x[:,i : i+1]))
                    
            var_embed = self.get_var_emb(self.var_embed, variables) # 1, V, D
            x = torch.stack(embeds, dim=1)  # B, L, D -> B, V, L, D
            x = x + var_embed.unsqueeze(2)  # 1, V, D -> 1, V, 1, D
            x = self.aggregate_variables(x)  # B, V~ , L, D, where V~ is the aggregated variables
        else:
            if self.adaptive_patching and not self.sqrt_len_method:
                x = rearrange(x, 'b c s p -> b s (p c)')
                x = self.token_embeds(x)
            else:
                x = self.token_embeds(x)

        x = self._pos_embed(x, seq_ps)
        x = self.patch_drop(x)

        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            # Same non-contiguous-x fix as VIT.forward_features above --
            # this method has the identical _pos_embed -> patch_drop ->
            # broadcast shape with no gather/cat step in between to clean
            # up contiguity, and UNETR (the only caller) is class_token=False.
            x = x.contiguous()
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
        """Pools the encoder output and decodes it (and any intermediates) into a segmentation mask.

        Args:
            x: Encoder output token sequence.
            intermediates: Intermediate encoder features for skip connections, or
                None if `self.skip_connection` is False.
            enc1: Original-resolution convolutional encoder feature for the final
                skip connection, or None if `self.skip_connection` is False.

        Returns:
            Per-class segmentation logits, shape (B, num_classes, H, W[, D]).
        """
        x = self.pool(x)
        return self.unetr_head(x, intermediates, enc1)

    def forward(self, x: torch.Tensor, variables, seq_ps=None, x_seq=None) -> torch.Tensor:
        """Runs the full UNETR forward pass: convolutional stem (if skip connections), transformer encoder, decoder.

        Args:
            x: Original-resolution input tile, used for the convolutional
                `encoder1` skip connection when `self.skip_connection` is True, and
                as the transformer input directly when not adaptively patched.
            variables: Variable/channel names for the transformer input.
            seq_ps: Per-patch size/position tensor for adaptive positional
                embeddings.
            x_seq: Pre-tokenized adaptive-patch sequence, used as the transformer
                input instead of `x` when `self.adaptive_patching` is True.

        Returns:
            Per-class segmentation logits, shape (B, num_classes, H, W[, D]).
        """
        if self.adaptive_patching:
            if self.skip_connection:
                enc1 = self.encoder1(x)
                x, intermediates = self.forward_intermediates(x_seq, variables, seq_ps, indices=self.skip_indices)
                x = self.forward_head(x, intermediates, enc1)
            else:
                enc1 = None
                x = self.forward_features(x_seq, variables, seq_ps)
                intermediates = None
                x = self.forward_head(x, intermediates, enc1)
        else:
            if self.skip_connection:
                enc1 = self.encoder1(x)
                x, intermediates = self.forward_intermediates(x, variables, seq_ps, indices=self.skip_indices)
                x = self.forward_head(x, intermediates, enc1)
            else:
                enc1 = None
                x = self.forward_features(x, variables, seq_ps)
                intermediates = None
                x = self.forward_head(x, intermediates, enc1)
        return x

class DiffusionVIT(VIT):
    """Diffusion denoising model: a `VIT` encoder conditioned on a diffusion timestep, with a reconstruction decoder.

    Adds a sinusoidal timestep embedding (via `SinusoidalEmbeddings` +
    `EmbeddingDenseLayer`) to the patch token sequence before the transformer
    encoder, then decodes back to pixel-space per-patch predictions (the
    predicted noise, in the standard DDPM formulation) via a linear or full
    transformer decoder, mirroring `MAE`'s decoder but without masking.
    """

    def __init__(self, *args, **kwargs):
        """Builds the `VIT` encoder, then adds the timestep embedding and reconstruction decoder.

        Args:
            *args: Positional arguments forwarded to `VIT.__init__`.
            **kwargs: Keyword arguments forwarded to `VIT.__init__`; must include
                `linear_decoder` (whether to use a single linear decoder),
                `time_steps` (number of diffusion timesteps to embed), and, when
                `linear_decoder` is False, `decoder_depth`, `decoder_embed_dim`,
                `decoder_num_heads`, and `decoder_mlp_ratio`.
        """
        self.linear_decoder = kwargs.pop('linear_decoder', '')
        self.decoder_depth = kwargs.pop('decoder_depth', '')
        self.decoder_embed_dim = kwargs.pop('decoder_embed_dim', '')
        self.decoder_num_heads = kwargs.pop('decoder_num_heads', '')
        self.decoder_mlp_ratio = kwargs.pop('decoder_mlp_ratio', '')
        super().__init__(*args, **kwargs)
        #Remove decoder from VIT
        self.head = None 

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
                    mlp_ratio=self.decoder_mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    init_values=self.init_values,
                    proj_drop=self.proj_drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    mlp_layer=self.mlp_layer,
                    num_time_steps=self.num_time_steps,
                )
                for i in range(self.decoder_depth)])
        else:
            self.decoder_pos_embed = None

        self.init_weights('')

    def init_weights(self, mode: str = '') -> None:
        """Initializes encoder and decoder positional embeddings, cls token, patch embedding weights, and all submodules.

        Same as `MAE.init_weights`.

        Args:
            mode: Unused; kept for interface compatibility.
        """
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
        """Embeds patches/tokens, adds positional and timestep embeddings, and runs the transformer encoder.

        Like `VIT.forward_features`, but additionally computes a sinusoidal
        embedding of `t` and adds it (broadcast across the sequence dimension) to
        the token sequence after the positional embedding.

        Args:
            x: Input patch/pixel tensor.
            t: Diffusion timestep indices, one per batch element.
            variables: Variable/channel names for `x`.

        Returns:
            Encoded token sequence, shape (B, N[+prefix], embed_dim).
        """
        if self.use_varemb:
            embeds = []
            if isinstance(variables, list):
                variables = tuple(variables)
            var_ids = self.get_var_ids(variables, x.device)
            for i in range(len(var_ids)):
                id = var_ids[i]
                if self.adaptive_patching:
                    embeds.append(self.token_embeds[id](torch.squeeze(x[:,i : i+1])))
                else:
                    embeds.append(self.token_embeds[id](x[:,i : i+1]))
                    
            var_embed = self.get_var_emb(self.var_embed, variables) # 1, V, D
            x = torch.stack(embeds, dim=1)  # B, L, D -> B, V, L, D
            x = x + var_embed.unsqueeze(2)  # 1, V, D -> 1, V, 1, D
            x = self.aggregate_variables(x)  # B, V~ , L, D, where V~ is the aggregated variables
        else:
            if self.adaptive_patching:
                x = rearrange(x, 'b c s p -> b s (p c)')
                x = self.token_embeds(x)
            else:
                x = self.token_embeds(x)
               
        x = self._pos_embed(x, None)
        x = self.patch_drop(x)

        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            # Must reassign x = x.contiguous() (not pass x.contiguous()
            # inline) -- dist.broadcast fills its argument in place, and
            # .contiguous() returns a NEW tensor whenever x isn't already
            # contiguous, so broadcasting an unassigned copy would silently
            # leave this rank's own x un-updated. x here can be
            # non-contiguous even after x + time_emb (elementwise ops can
            # preserve a non-contiguous input's memory layout).
            x = x.contiguous()
            dist.broadcast(x, src_rank, group=self.tensor_par_group)

        for blk in self.blocks:
            x = blk(x, t)
        x = self.norm(x)

        if self.tensor_par_size > 1:
            x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        return x

    def forward_head(self, x: torch.Tensor, t):
        """Pools the encoder output and decodes it into a per-patch pixel-space prediction.

        Args:
            x: Encoder output token sequence.

        Returns:
            Predicted per-patch pixel values (e.g. predicted noise), shape (B, L,
            patch_dim).
        """
        x = self.pool(x)
        if not self.linear_decoder:
            if self.tensor_par_size > 1:
                src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
                dist.broadcast(x, src_rank, group=self.tensor_par_group)

            x = self.decoder_embed(x)
            x = x + self.decoder_pos_embed
            for blk in self.decoder_blocks:
                x = blk(x, t)
            x = self.decoder_norm(x)

            if self.tensor_par_size > 1:
                x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        return self.decoder_pred(x)

    def forward(self, x: torch.Tensor, t, variables) -> torch.Tensor:
        """Runs the full timestep-conditioned forward pass: encode then decode.

        Args:
            x: Noised input patch/pixel tensor.
            t: Diffusion timestep indices, one per batch element.
            variables: Variable/channel names for `x`.

        Returns:
            Predicted per-patch pixel values (e.g. predicted noise), shape (B, L,
            patch_dim).
        """
        t = t.to('cpu')
        x = self.forward_features(x, t, variables)
        x = self.forward_head(x,t)
        return x
