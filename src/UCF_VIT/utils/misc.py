import torch
import torch.distributed as dist
from pathlib import Path
import re

from UCF_VIT.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

def patchify( data, patch_size, twoD):
    batch_size = data.shape[0]
    num_channels = data.shape[1]
    dim_x = data.shape[2]
    dim_y = data.shape[3]
    if not twoD:
        dim_z = data.shape[4]
    num_patches_x = dim_x // patch_size
    num_patches_y = dim_y // patch_size
    if not twoD:
        num_patches_z = dim_z // patch_size
    if twoD:
        patchified_pixel_values = data.reshape(batch_size, num_channels, num_patches_x, patch_size, num_patches_y, patch_size)
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape( batch_size, num_patches_x * num_patches_y, patch_size**2 * num_channels)
    else:
        patchified_pixel_values = data.reshape(batch_size, num_channels, num_patches_x, patch_size, num_patches_y, patch_size, num_patches_z, patch_size)
        patchified_pixel_values = torch.einsum("nchpwqdr->nhwdpqrc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape( batch_size, num_patches_x * num_patches_y * num_patches_z, patch_size**3 * num_channels)
    return patchified_pixel_values

def unpatchify(patchified_pixel_values,data, patch_size, twoD):
    if twoD:
        original_x, original_y = data.shape[2], data.shape[3]
    else:
        original_x, original_y, original_z = data.shape[2], data.shape[3], data.shape[4]

    num_patches_x = original_x // patch_size
    num_patches_y = original_y // patch_size
    if not twoD:
        num_patches_z = original_z // patch_size
    
    batch_size = patchified_pixel_values.shape[0]
    num_channels = data.shape[1]
    if twoD:
        patchified_pixel_values = patchified_pixel_values.reshape(batch_size, num_patches_x, num_patches_y, patch_size, patch_size, num_channels)
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(batch_size, num_channels, num_patches_x*patch_size, num_patches_y*patch_size)
    else:
        patchified_pixel_values = patchified_pixel_values.reshape(batch_size, num_patches_x, num_patches_y, num_patches_z, patch_size, patch_size, patch_size, num_channels)
        patchified_pixel_values = torch.einsum("nhwdpqrc->nchpwqdr", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(batch_size, num_channels, num_patches_x*patch_size, num_patches_y*patch_size, num_patches_z*patch_size)
    return pixel_values

def configure_optimizer(model,lr,beta_1,beta_2,weight_decay):
    decay = []
    no_decay = []
    for name, m in model.named_parameters():
        if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
            no_decay.append(m)
        else:
            decay.append(m)

    optimizer = torch.optim.AdamW(
        [
        {
            "params": decay,
            "lr": lr,
            "betas": (beta_1, beta_2),
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "lr": lr,
            "betas": (beta_1, beta_2),
            "weight_decay": 0,
        },
        ]
    )

    return optimizer

def configure_scheduler(optimizer,warmup_steps,max_steps,warmup_start_lr,eta_min):
    
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps,
        max_steps,
        warmup_start_lr,
        eta_min,
    )

    return lr_scheduler

def interpolate_pos_embed_adaptive(model, checkpoint_model, new_size=127):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]

        if orig_num_patches != new_size:
            pos_tokens = pos_embed_checkpoint.reshape(-1, orig_num_patches, embedding_size).permute(0, 2, 1)
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size, mode="linear", align_corners=False
            )
            new_pos_tokens = new_pos_tokens.permute(0,2,1)
            checkpoint_model["pos_embed"] = new_pos_tokens

            del new_pos_tokens

    if "decoder_pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["decoder_pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]

        if orig_num_patches != new_size:
            pos_tokens = pos_embed_checkpoint.reshape(-1, orig_num_patches, embedding_size).permute(0, 2, 1)
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size, mode="linear", align_corners=False
            )
            new_pos_tokens = new_pos_tokens.permute(0,2,1)
            checkpoint_model["decoder_pos_embed"] = new_pos_tokens

            del new_pos_tokens

def init_par_groups(world_rank, data_par_size, tensor_par_size, seq_par_size, fsdp_size, simple_ddp_size):

    tensor_par_group = None

    for i in range(data_par_size *seq_par_size):
        ranks = [j for j in range(i*tensor_par_size,(i+1)*tensor_par_size)]

        #if world_rank==0:
        #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," tensor_par_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:
            tensor_par_group = group




    seq_par_group = None

    for t in range(data_par_size):
        for i in range(tensor_par_size):
            ranks = [t*tensor_par_size*seq_par_size+i+j*tensor_par_size for j in range(seq_par_size)]

            #if world_rank==0:
            #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size, " TENSOR_PAR_SIZE ",tensor_par_size," seq_par_group ranks ",ranks,flush=True)

            group = dist.new_group(ranks)

            if world_rank in ranks:

                seq_par_group = group




    ddp_group = None

    fsdp_group = None

    simple_ddp_group = None

    for i in range(tensor_par_size *seq_par_size):
        ranks = [i+j*tensor_par_size *seq_par_size for j in range(data_par_size)]

        for k in range(simple_ddp_size):
            fsdp_begin_idx = k*fsdp_size
            fsdp_end_idx = (k+1)*fsdp_size
            fsdp_ranks = ranks[fsdp_begin_idx:fsdp_end_idx]


            #if world_rank==0:
            #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," fsdp_ranks",fsdp_ranks)


            group = dist.new_group(fsdp_ranks)

            if world_rank in fsdp_ranks:

                fsdp_group = group


        for k in range(fsdp_size):
            simple_ddp_begin_idx = k
            simple_ddp_end_idx = len(ranks)
            simple_ddp_ranks = ranks[simple_ddp_begin_idx:simple_ddp_end_idx:fsdp_size]


            #if world_rank==0:
            #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," simple_ddp_ranks",simple_ddp_ranks)


            group = dist.new_group(simple_ddp_ranks)

            if world_rank in simple_ddp_ranks:

                simple_ddp_group = group



        #if world_rank==0:
        #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," ddp_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:

            ddp_group = group






    data_seq_ort_group = None

    for i in range(tensor_par_size):
        ranks = [i+tensor_par_size*j for j in range(data_par_size * seq_par_size)]

        #if world_rank==0:
        #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," data_seq_ort_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:

            data_seq_ort_group = group


    return seq_par_group, ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group

def get_file_prefix(sample_path):
    '''
    Function to find base path prefix and timestamp group for a file of type:
    /lustre/orion/stf006/world-shared/muraligm/CFD135/data_iso/super_res/binary_data/P1F4R32_nx512ny512nz256_6vars/w_29.960000
    with variable length of var-name and timestamp
    '''
    # Regular expression to identify the variable name and timestamp
    pattern = r'/(.*)/([a-zA-Z0-9_]+)_([0-9]+\.[0-9]+)$'
    # Extract the base path and timestamp
    match = re.search(pattern, sample_path)
    base_path_prefix = match.group(1)
    timestamp = match.group(3)
    #   if match:
    #       base_path_prefix = match.group(1)
    #       timestamp = match.group(3)
    #       print(f"Base Path Prefix: {base_path_prefix}; timestamp: {timestamp}")
    #   else:
    #       print("No match found")
    return "/"+base_path_prefix, timestamp
