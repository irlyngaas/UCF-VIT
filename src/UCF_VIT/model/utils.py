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
from UCF_VIT.fsdp.building_blocks import Block
from timm.layers import use_fused_attn

def get_model(conf, p_conf, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group):
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
        img_size=(conf["data"]["tile_size"][0],conf["data"]["tile_size"][1]) if conf["data"]["twoD"] else (conf["data"]["tile_size"][0],conf["data"]["tile_size"][1], conf["data"]["tile_size"][2]),
        patch_size=conf["data"]["patch_size"],
        in_chans=conf["data"]["in_chans"],
        embed_dim=conf["model"]["embed_dim"],
        depth=conf["model"]["depth"],
        num_heads=conf["model"]["num_heads"],
        mlp_ratio=conf["model"]["mlp_ratio"],
        drop_path_rate=conf["model"]["drop_path"],
        drop_rate=conf["model"]["drop_rate"],
        twoD=conf["data"]["twoD"],
        default_vars=conf["data"]["default_vars"],
        single_channel=False, #TODO: Take out single channel option altogether
        use_varemb=conf["model"]["use_channel_aggregation"], #TODO: Change use_varemb to use_channel_aggregation in arch.py
        adaptive_patching=conf["ap"]["do_ap"],
        fixed_length=conf["ap"]["fixed_length"],
        FusedAttn_option=FusedAttn_option,
        use_adaptive_pos_emb=conf["ap"]["use_adaptive_pos_emb"],
        weight_init='' if conf["model"]["type"] == "VIT" else 'skip', #Choose ['' or 'skip'] If using VIT use '' otherwise use 'skip'. Option whether to use VITs weight initialization or use the one corresponding to the architecture you choose
        class_token=True if conf["model"]["type"] == "VIT" else False,
        **conf['model']['kwargs'],
    ).to(device)

    if not conf["trainer"]["resume_from_checkpoint"]:
        epoch_start = 0
        loss_list = []
        if conf["trainer"]["use_pretrained_model"]: #Train from pre-trained model
            if world_rank==0:       
                print("Train starting from pretrained model.",flush=True)

            if p_conf["model_type"] == "VIT":
                from UCF_VIT.fsdp.arch import VIT as pretrained_model_arch
            elif p_conf["model_type"] == "SAP":
                from UCF_VIT.fsdp.arch import SAP as pretrained_model_arch
            elif p_conf["model_type"] == "MAE":
                from UCF_VIT.fsdp.arch import MAE as pretrained_model_arch
            elif p_conf["model_type"] == "UNETR":
                from UCF_VIT.fsdp.arch import UNETR as pretrained_model_arch
            elif p_conf["model_type"] == "DiffusionVIT":
                from UCF_VIT.fsdp.arch import DiffusionVIT as pretrained_model_arch

            pretrained_model = pretrained_model_arch(
                img_size=conf["data"]["tile_size"],
                patch_size=conf["data"]["patch_size"],
                in_chans=conf["data"]["in_chans"],
                embed_dim=conf["model"]["embed_dim"],
                depth=conf["model"]["depth"],
                num_heads=conf["model"]["num_heads"],
                mlp_ratio=conf["model"]["mlp_ratio"],
                drop_path_rate=conf["model"]["drop_path"],
                drop_rate=conf["model"]["drop_rate"],
                twoD=conf["data"]["twoD"],
                default_vars=p_conf["default_vars"],
                single_channel=False, #TODO: Take out single channel option altogether
                use_varemb=conf["model"]["use_channel_aggregation"], #TODO: Change use_varemb to use_channel_aggregation in arch.py
                adaptive_patching=conf["ap"]["do_ap"],
                fixed_length=conf["ap"]["fixed_length"],
                FusedAttn_option=FusedAttn_option,
                use_adaptive_pos_emb=conf["ap"]["use_adaptive_pos_emb"],
                weight_init='' if p_conf["model_type"] == "VIT" else 'skip', #Choose ['' or 'skip'] If using VIT use '' otherwise use 'skip'. Option whether to use VITs weight initialization or use the one corresponding to the architecture you choose
                class_token=True if p_conf["model_type"] == "VIT" else False,
                **p_conf['kwargs'],
            ).to(device)

            if world_rank< conf["parallelism"]["tensor_par_size"]:
                map_location = 'cpu' #TODO: Choose cpu or cuda+str
                #map_location = 'cuda:'+str(device)
                pretrained_checkpoint = torch.load(conf["pretrained_model"]["checkpoint_path"]+"/"+conf["trainer"]["pretrained_checkpoint_filename"]+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'])

            new_state_dict = OrderedDict()
            encoder_dict = new_state_dict
            model_dict = pretrained_model.state_dict()

            #Taking out encoder states from pretrained model. The decoder to take out from the model dict is different depending on the pretrained model
            #TODO: Add encoder_dict for different pretrained model types
            if p_conf["model_type"] == "MAE":
                #decoder_dict = {k: v for k, v in model_dict.items() if ('decoder_pred' in k or 'attn_layers_decoder' in k or 'mask_token' in k)}
                encoder_dict = {k: v for k,v in model_dict.items() if ('decoder' not in k and 'mask_token' not in k)}
                #state_dict.append(encoder_dict)

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

               map_location = 'cpu' #TODO: Choose cpu or cuda+str
               #map_location = 'cuda:'+str(device)
               model.load_state_dict(torch.load(conf["trainer"]["checkpoint_path"]+'/initial_'+str(0)+'.pth',map_location=map_location),strict=False)

    else: #Resume from checkpoint
        if world_rank < conf["parallelism"]["tensor_par_size"]:
            if os.path.exists(conf["trainer"]["checkpoint_path"]+"/"+conf["trainer"]["checkpoint_filename"]+"_rank_"+str(world_rank)+".ckpt"):
                print("resume from checkpoint was set to True. Checkpoint path found.",flush=True)

                map_location = 'cpu' #TODO: Choose cpu or cuda+str
                #map_location = 'cuda:'+str(device)

                checkpoint = torch.load(conf["trainer"]["checkpoint_path"]+"/"+conf["trainer"]["checkpoint_filename"]+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                model.load_state_dict(checkpoint['model_state_dict'])
                del checkpoint

            else:
                print("resume from checkpoint was set to True. But the checkpoint path does not exist.",flush=True)
                sys.exit("checkpoint path does not exist")

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
