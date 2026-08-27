import os
import sys
import torch
import torch.distributed as dist
import functools
from UCF_VIT.utils.fused_attn import FusedAttn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
   size_based_auto_wrap_policy, wrap, transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing,
)

from torch.nn import Sequential
from UCF_VIT.model.building_blocks import Block
from UCF_VIT.utils.pos_embed import interpolate_pos_embed, interpolate_pos_embed_3d
from timm.layers import use_fused_attn

# Top-level state_dict key prefixes that UCF_VIT.model.arch.VIT.__init__ itself
# creates -- i.e. the shared transformer encoder, before any subclass
# (SAP/MAE/UNETR/DiffusionVIT) adds its own task-specific decoder/head on top.
# An allowlist rather than a "decoder"/"head"-name denylist: UNETR has its own
# self.encoder1/encoder2/encoder3/encoder4 (U-Net skip-connection convs feeding
# its *decoder*, not the transformer encoder) that a substring-based denylist
# would mishandle. Matched against each key's first dotted component only, so
# e.g. "encoder1.weight" (UNETR) can never collide with "encoder" (not even a
# real prefix here) or "norm" (never collides with "decoder_norm.weight",
# whose first component is "decoder_norm").
ENCODER_STATE_DICT_PREFIXES = {
    "patch_embed", "token_embeds", "cls_token", "pos_embed",
    "var_embed", "adaptive_pos_dep_emb", "blocks", "norm",
}


def extract_encoder_state_dict(state_dict):
    """Filters a model state dict down to just the shared VIT encoder.

    Keeps only entries whose top-level attribute name (the part before the first
    ".") is one VIT.__init__ itself creates -- see ENCODER_STATE_DICT_PREFIXES --
    dropping every subclass-specific decoder/head/task-specific addition
    (MAE/DiffusionVIT's decoder_*/mask_token, SAP's neck/mask_header, UNETR's
    encoderN/decoderN/out/upsample/mlp_head, VIT's own classification head).
    Works identically regardless of which model type the state dict came from.

    Args:
        state_dict: A model's state dict (e.g. a pretrained model's, after
            loading a checkpoint into it).

    Returns:
        A new dict containing only the encoder entries.
    """
    return {k: v for k, v in state_dict.items() if k.split(".")[0] in ENCODER_STATE_DICT_PREFIXES}


def _transplant_pos_embed(encoder_dict, pretrained_model, model):
    """Resizes encoder_dict's "pos_embed" entry (in place) to match model's own shape.

    A no-op if encoder_dict has no "pos_embed" entry (e.g. use_adaptive_pos_emb:True,
    where pos_embed is None and never appears in a state dict at all) or if the
    pretrained and new models already have the same pos_embed shape.

    Args:
        encoder_dict: Dict as returned by extract_encoder_state_dict, from the
            pretrained model; modified in place.
        pretrained_model: The constructed pretrained-model instance the checkpoint
            was loaded into (at the pretrained model's own original size).
        model: The constructed new model instance being fine-tuned (at its own,
            possibly different, size).

    Raises:
        NotImplementedError: If either model has sqrt_len_method=True (SAP, or
            UNETR with do_ap:True) and their pos_embed shapes differ. That regime's
            pos_embed is sized from patch_embed's raw img_size/patch_size grid,
            which does not actually match its real sqrt_len-based token count (a
            separate, pre-existing issue) -- interpolating via grid_size there
            would silently produce a wrong-shaped result, so this is rejected
            explicitly rather than attempted.
    """
    if "pos_embed" not in encoder_dict:
        return

    pos_embed = encoder_dict["pos_embed"]
    if tuple(pos_embed.shape) == tuple(model.pos_embed.shape):
        return

    if pretrained_model.sqrt_len_method or model.sqrt_len_method:
        raise NotImplementedError(
            "pos_embed interpolation for a pretrained/new size mismatch is not "
            "supported when sqrt_len_method is True (SAP, or UNETR with "
            "ap.do_ap:True) -- grid_size does not reflect the real sqrt_len-based "
            "token count for this regime. Use a pretrained checkpoint with the "
            "same size for these model types."
        )

    if hasattr(pretrained_model, "grid_size") and hasattr(model, "grid_size"):
        interp = interpolate_pos_embed if model.twoD else interpolate_pos_embed_3d
        # Sliced/re-prepended using the PRETRAINED model's own prefix count
        # (how pos_embed itself is actually laid out), not the new model's --
        # they can legitimately differ across a cross-architecture transplant
        # (e.g. MAE's class_token=False -> VIT's class_token=True). When they
        # do, there's no real pretrained data for the new model's own prefix
        # row(s), so its own existing (freshly-initialized/sincos) prefix is
        # kept instead of anything from the checkpoint.
        resized = interp(
            pos_embed, pretrained_model.grid_size, model.grid_size,
            num_prefix_tokens=pretrained_model.num_prefix_tokens,
        )
        if model.num_prefix_tokens != pretrained_model.num_prefix_tokens:
            resized = torch.cat(
                [model.pos_embed[:, :model.num_prefix_tokens], resized[:, pretrained_model.num_prefix_tokens:]],
                dim=1,
            )
        encoder_dict["pos_embed"] = resized
    else:
        # adaptive_patching and not sqrt_len_method: pos_embed is a flat,
        # learned, per-sequence-slot-index embedding (used only when
        # use_adaptive_pos_emb:False -- see VIT._pos_embed; when True,
        # pos_embed is allocated but never actually read in forward, the
        # geometry-derived adaptive_pos_dep_emb is used instead), not a
        # spatial grid -- unlike grid_size's case above, slot index N has no
        # reliable relationship to slot index N+1 at all (FixedQuadTree/
        # FixedOctTree's own node order reflects greedy-split order, not
        # spatial adjacency), so there's no principled way to resize it the
        # way a real spatial grid can be. (An earlier version of this
        # function tried a 1D linear interpolation along the slot-index axis
        # regardless -- archived at
        # ../UCF-VIT-claude-archive/src/UCF_VIT/utils/misc.py -- which both
        # rested on that unfounded adjacency assumption and, independently,
        # never sliced out num_prefix_tokens first, corrupting the
        # class-token row into the interpolation whenever class_token:True.)
        # Same reasoning as sqrt_len_method's rejection above, just handled
        # by dropping rather than raising, since a size mismatch here is a
        # real, unremarkable scenario (e.g. fine-tuning at a different
        # fixed_length): the new model just keeps its own fresh init.
        del encoder_dict["pos_embed"]


def _prune_incompatible_cls_token(encoder_dict, model):
    """Drops encoder_dict's "cls_token" entry (in place) if the new model has none.

    A no-op if encoder_dict has no "cls_token" entry at all (the pretrained
    source itself has class_token=False, so there's nothing to drop -- the new
    model just keeps its own) or if the new model does have its own cls_token
    (kept, transplanted normally -- a real, meaningful value in that case).

    Only VIT ever has class_token=True in practice (get_model's own
    class_token=True if conf["model"]["type"] == "VIT" else False) -- so this
    only matters for a pretrained VIT source paired with a non-VIT (or
    class_token=False VIT) downstream model, where cls_token is present in
    encoder_dict (extract_encoder_state_dict's allowlist includes it, since
    it's genuinely a shared VIT.__init__ attribute) but the new model has no
    such attribute at all, not just a differently-shaped one -- unlike
    pos_embed's prefix-count mismatch (_transplant_pos_embed above), which
    always has *some* pos_embed to reconcile shapes against. Found via a real
    VIT->MAE test: model.load_state_dict(..., strict=True) otherwise raises
    "Unexpected key(s) in state_dict: cls_token".

    Args:
        encoder_dict: Dict as returned by extract_encoder_state_dict, from the
            pretrained model; modified in place.
        model: The constructed new model instance being fine-tuned.
    """
    if "cls_token" in encoder_dict and model.cls_token is None:
        del encoder_dict["cls_token"]


def get_model(conf, p_conf, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group):
    """Build the model architecture, load its initial weights, and wrap it with FSDP.

    Instantiates the model type given in `conf`, initializes its weights either from
    scratch, from a pretrained checkpoint, or by resuming from an existing checkpoint,
    then wraps the model in FSDP with the sharding strategy implied by the configured
    parallelism sizes and applies activation checkpointing to each transformer block.

    Args:
        conf: Parsed training configuration dict (as returned by `parse_config`).
        p_conf: Parsed pretrained-model configuration dict (as returned by
            `parse_pretrained_config`), used only when `conf["trainer"]["use_pretrained_model"]`
            is True.
        device: Device to move the model to.
        local_rank: Local rank of this process, used as the FSDP `device_id`.
        fsdp_group: Process group over which parameters are fully/hybrid sharded.
        simple_ddp_group: Process group over which sharded replicas are data-parallel
            synchronized.
        tensor_par_group: Process group used for tensor-parallel weight synchronization.

    Returns:
        A tuple `(model, epoch_start, loss_list)` where `model` is the FSDP-wrapped
        model, `epoch_start` is the starting epoch (0 when training from scratch or
        from a pretrained model, or one past the checkpointed epoch when resuming),
        and `loss_list` is the loss history accumulated so far (empty when training
        from scratch or from a pretrained model, or restored from the checkpoint
        when resuming). Identical on every rank.
    """
    world_rank = dist.get_rank()

    if conf["trainer"]["data_type"] == "bfloat16":
        if conf["trainer"]["gpu_type"] == "amd":
            FusedAttn_option = FusedAttn.CK
        elif conf["trainer"]["gpu_type"] == "nvidia":
            FusedAttn_option = FusedAttn.FLASH #Can be switched to Default if Xformers is causing errors
    else:
        #Check whether flash attention is installed, if it's not use a basic python implementation
        if use_fused_attn():
            FusedAttn_option = FusedAttn.DEFAULT
        else:
            FusedAttn_option = FusedAttn.NONE

    if conf["model"]["type"] == "VIT":
        from UCF_VIT.model.arch import VIT as model_arch
    elif conf["model"]["type"] == "SAP":
        from UCF_VIT.model.arch import SAP as model_arch
    elif conf["model"]["type"] == "MAE":
        from UCF_VIT.model.arch import MAE as model_arch
    elif conf["model"]["type"] == "UNETR":
        from UCF_VIT.model.arch import UNETR as model_arch
    elif conf["model"]["type"] == "DiffusionVIT":
        from UCF_VIT.model.arch import DiffusionVIT as model_arch

    model = model_arch(
        img_size=tuple(conf["data"]["tile_size"]),
        patch_size=conf["data"]["patch_size"],
        interp_size=conf["data"].get("interp_size"),
        in_chans=conf["data"]["in_chans"],
        embed_dim=conf["model"]["embed_dim"],
        depth=conf["model"]["depth"],
        num_heads=conf["model"]["num_heads"],
        mlp_ratio=conf["model"]["mlp_ratio"],
        drop_path_rate=conf["model"]["drop_path"],
        drop_rate=conf["model"]["drop_rate"],
        twoD=conf["data"]["twoD"],
        default_vars=conf["data"]["default_vars"],
        use_varemb=conf["model"]["use_channel_aggregation"], #TODO: Change use_varemb to use_channel_aggregation in arch.py
        adaptive_patching=conf["ap"]["do_ap"],
        fixed_length=conf["ap"]["fixed_length"],
        FusedAttn_option=FusedAttn_option,
        use_adaptive_pos_emb=conf["ap"]["use_adaptive_pos_emb"],
        weight_init='' if conf["model"]["type"] == "VIT" else 'skip', #Choose ['' or 'skip'] If using VIT use '' otherwise use 'skip'. Option whether to use VITs weight initialization or use the one corresponding to the architecture you choose
        class_token=True if conf["model"]["type"] == "VIT" else False,
        # Without these, VIT/SAP/MAE/UNETR/DiffusionVIT all fall back to
        # their constructor defaults (tensor_par_size=1, tensor_par_group=
        # None) regardless of conf["parallelism"]["tensor_par_size"] --
        # conf['model']['kwargs'] (built by parse.py's get_kwargs) never
        # sets them either. That silently built a full, unsharded model on
        # every rank even when tensor_par_size > 1: every `if
        # self.tensor_par_size > 1:` guard in arch.py/building_blocks.py
        # (the real Attention/Mlp sharding, MAE's noise-mask broadcast,
        # etc.) never fired, so tensor_par_size > 1 ran to completion (data
        # was still correctly distributed by training.py's process_batch,
        # which gets tensor_par_group directly from train.py, not through
        # this function) but did zero actual model-parallel sharding --
        # pure redundant compute, not real tensor parallelism.
        tensor_par_size=conf["parallelism"]["tensor_par_size"],
        tensor_par_group=tensor_par_group,
        **conf['model']['kwargs'],
    ).to(device)

    if not conf["trainer"]["resume_from_checkpoint"]:
        epoch_start = 0
        loss_list = []
        if conf["trainer"]["use_pretrained_model"]: #Train from pre-trained model
            if world_rank==0:       
                print("Train starting from pretrained model.",flush=True)

            if p_conf["model_type"] == "VIT":
                from UCF_VIT.model.arch import VIT as pretrained_model_arch
            elif p_conf["model_type"] == "SAP":
                from UCF_VIT.model.arch import SAP as pretrained_model_arch
            elif p_conf["model_type"] == "MAE":
                from UCF_VIT.model.arch import MAE as pretrained_model_arch
            elif p_conf["model_type"] == "UNETR":
                from UCF_VIT.model.arch import UNETR as pretrained_model_arch
            elif p_conf["model_type"] == "DiffusionVIT":
                from UCF_VIT.model.arch import DiffusionVIT as pretrained_model_arch

            # Built at the pretrained model's own original img_size/twoD/
            # patch_size/interp_size/fixed_length/use_adaptive_pos_emb (from
            # p_conf, computed by parse_pretrained_config from the pretrained
            # model's own config file) -- not conf's (the new model's) --
            # so its parameter shapes match the checkpoint being loaded into
            # it below exactly, regardless of any difference from the new
            # model's own size/config. The resulting encoder gets resized
            # (pos_embed only, via _transplant_pos_embed) to the new model's
            # shape afterward, not before loading the checkpoint.
            pretrained_model = pretrained_model_arch(
                img_size=tuple(p_conf["tile_size"]),
                patch_size=p_conf["patch_size"],
                interp_size=p_conf["interp_size"],
                in_chans=conf["data"]["in_chans"],
                embed_dim=conf["model"]["embed_dim"],
                depth=conf["model"]["depth"],
                num_heads=conf["model"]["num_heads"],
                mlp_ratio=conf["model"]["mlp_ratio"],
                drop_path_rate=conf["model"]["drop_path"],
                drop_rate=conf["model"]["drop_rate"],
                twoD=p_conf["twoD"],
                default_vars=p_conf["default_vars"],
                use_varemb=conf["model"]["use_channel_aggregation"], #TODO: Change use_varemb to use_channel_aggregation in arch.py
                adaptive_patching=conf["ap"]["do_ap"],
                fixed_length=p_conf["fixed_length"],
                FusedAttn_option=FusedAttn_option,
                use_adaptive_pos_emb=p_conf["use_adaptive_pos_emb"],
                weight_init='' if p_conf["model_type"] == "VIT" else 'skip', #Choose ['' or 'skip'] If using VIT use '' otherwise use 'skip'. Option whether to use VITs weight initialization or use the one corresponding to the architecture you choose
                class_token=True if p_conf["model_type"] == "VIT" else False,
                # Same reasoning as the main model_arch(...) call above --
                # parse_pretrained_config already asserts the pretrained
                # model's own tensor_par_size matches conf's, and this
                # model's state_dict() gets merged into `model`'s below, so
                # it must be sharded the same way or the shapes won't
                # match.
                tensor_par_size=conf["parallelism"]["tensor_par_size"],
                tensor_par_group=tensor_par_group,
                **p_conf['kwargs'],
            ).to(device)

            if world_rank< conf["parallelism"]["tensor_par_size"]:
                # Loaded to CPU, not the target device -- see
                # load_optimizer_scheduler_from_checkpoint's map_location
                # comment in training.py for why.
                map_location = 'cpu'
                pretrained_checkpoint = torch.load(p_conf["checkpoint_path"]+"/"+conf["trainer"]["pretrained_checkpoint_filename"]+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'])

            # Works for any pretrained model type (not just MAE) -- see
            # extract_encoder_state_dict's own docstring for why an
            # allowlist generalizes safely across all of them.
            encoder_dict = extract_encoder_state_dict(pretrained_model.state_dict())
            _transplant_pos_embed(encoder_dict, pretrained_model, model)
            _prune_incompatible_cls_token(encoder_dict, model)

            #Load encoder states from pretrained model into the model we want to train
            model_dict = model.state_dict()
            model_dict.update(encoder_dict)
            model.load_state_dict(model_dict)

        else: #Train from scratch
            if world_rank==0:       
                print("Train from scratch.",flush=True)

                #Check whether the specified checkpointing path exists or not
                isExist = os.path.exists(conf["trainer"]["checkpoint_path"])
                if not isExist:
                    #Create a new directory because it does not exist
                    os.makedirs(conf["trainer"]["checkpoint_path"])
                    print("The new checkpoint directory is created!")

                #Save initial model weights and distribute to all GPUs in the tensor parallel group to synchronize model weights that do not belong to the training block
                init_model_dict = {k: v for k, v in model.state_dict().items() if ('attn' not in  k and 'mlp' not in k and 'var_agg' not in k)}

                torch.save(init_model_dict,
                        conf["trainer"]["checkpoint_path"]+'/initial_'+str(dist.get_rank())+'.pth')

                del init_model_dict

            dist.barrier()

            if world_rank!=0 and world_rank < conf["parallelism"]["tensor_par_size"]:

               #load initial model weights and synchronize model weights that are not in the training block among sequence parallel GPUs
               src_rank = dist.get_rank() - dist.get_rank(group=tensor_par_group)

               # Loaded to CPU, not the target device -- see
               # load_optimizer_scheduler_from_checkpoint's map_location
               # comment in training.py for why.
               map_location = 'cpu'
               model.load_state_dict(torch.load(conf["trainer"]["checkpoint_path"]+'/initial_'+str(0)+'.pth',map_location=map_location),strict=False)

    else: #Resume from checkpoint
        if world_rank < conf["parallelism"]["tensor_par_size"]:
            if os.path.exists(conf["trainer"]["checkpoint_path"]+"/"+conf["trainer"]["checkpoint_filename"]+"_rank_"+str(world_rank)+".ckpt"):
                print("resume from checkpoint was set to True. Checkpoint path found.",flush=True)

                # Loaded to CPU, not the target device -- see
                # load_optimizer_scheduler_from_checkpoint's map_location
                # comment in training.py for why.
                map_location = 'cpu'

                checkpoint = torch.load(conf["trainer"]["checkpoint_path"]+"/"+conf["trainer"]["checkpoint_filename"]+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                model.load_state_dict(checkpoint['model_state_dict'])
                loss_list = checkpoint['loss_list']
                epoch_start = checkpoint['epoch'] + 1
                del checkpoint

            else:
                print("resume from checkpoint was set to True. But the checkpoint path does not exist.",flush=True)
                sys.exit("checkpoint path does not exist")

        #Only ranks below tensor_par_size actually read a checkpoint file above; broadcast
        #the resumed epoch_start/loss_list from rank 0 so every rank starts the training
        #loop at the same epoch (required for collective ops to stay in sync) instead of
        #silently restarting other ranks' progress at epoch 0.
        epoch_start_tensor = torch.tensor(epoch_start if world_rank < conf["parallelism"]["tensor_par_size"] else 0, device=device)
        dist.broadcast(epoch_start_tensor, src=0)
        epoch_start = epoch_start_tensor.item()

        loss_list_holder = [loss_list] if world_rank < conf["parallelism"]["tensor_par_size"] else [None]
        dist.broadcast_object_list(loss_list_holder, src=0, device=device)
        loss_list = loss_list_holder[0]

    dist.barrier()
    
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block, Sequential   # < ---- Your Transformer layer class
        },
    )

    if conf["trainer"]["data_type"] == "float32":
        precision_dt = torch.float32
    elif conf["trainer"]["data_type"] == "bfloat16":
        precision_dt = torch.bfloat16
    else:
        raise RuntimeError("Data type not supported")

    bfloatPolicy = MixedPrecision(
        param_dtype=precision_dt,
        # Gradient communication precision.
        reduce_dtype=precision_dt,
        # Buffer precision.
        buffer_dtype=precision_dt,
    )

    #add hybrid sharded FSDP
    if conf["parallelism"]["fsdp_size"] > 1 and conf["parallelism"]["simple_ddp_size"] > 1:
        model = FSDP(model, device_id=local_rank, process_group= (fsdp_group,simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add fully sharded FSDP
    elif conf["parallelism"]["fsdp_size"] > 1 and conf["parallelism"]["simple_ddp_size"] == 1:
        model = FSDP(model, device_id=local_rank, process_group= fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add unsharded DDP
    else:
        model = FSDP(model, device_id=local_rank, process_group= simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )

    check_fn = lambda submodule: isinstance(submodule, Block)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )

    return model, epoch_start, loss_list
