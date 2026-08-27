import torch
import torch.distributed as dist
import torch.nn as nn
import einops
import torch.distributed as dist

from UCF_VIT.utils.misc import patchify, unpatchify
from monai.losses import DiceCELoss
from monai.utils.enums import MetricReduction
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from UCF_VIT.utils.metrics import DiceBLoss, masked_mse, native_resolution_patch_masked_mse, native_resolution_patch_mse

def load_optimizer_scheduler_from_checkpoint(conf, optimizer, scheduler, data_seq_ort_group, device):
    """Restores optimizer and scheduler state, loss history, and epoch from a checkpoint.

    Args:
        conf: Parsed training configuration dict (as returned by `parse_config`).
        optimizer: Optimizer instance to load state into, in place.
        scheduler: LR scheduler instance to load state into, in place.
        data_seq_ort_group: Process group used to locate this rank's corresponding
            checkpoint file (the checkpoint from the equivalent tensor-parallel rank
            within this rank's data-parallel replica).
        device: Unused device argument; loading is done onto CPU regardless.

    Returns:
        A tuple `(optimizer, scheduler, loss_list, epoch_start)` where `loss_list`
        is the loss history restored from the checkpoint and `epoch_start` is the
        epoch to resume training from (one past the checkpointed epoch).
    """
    src_rank = dist.get_rank() - conf["parallelism"]["tensor_par_size"] * dist.get_rank(group=data_seq_ort_group)

    # Loaded to CPU (not the target device) to avoid transiently doubling GPU
    # memory (checkpoint + already-allocated model/optimizer state coexisting)
    # and to stay portable across runs where rank->GPU mapping may differ --
    # optimizer.load_state_dict below casts state tensors to each param's own
    # device automatically, so this doesn't leave anything stranded on CPU.
    map_location = 'cpu'

    checkpoint = torch.load(conf["trainer"]["checkpoint_path"]+"/"+conf["trainer"]["checkpoint_filename"]+"_rank_"+str(src_rank)+".ckpt",map_location=map_location)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    loss_list = checkpoint['loss_list']
    epoch_start = checkpoint['epoch'] + 1


    return optimizer, scheduler, loss_list, epoch_start

def train_step(conf, batch, model):
    """Runs a single forward pass and loss computation for one training batch.

    Dispatches to architecture-specific forward/loss logic based on
    `conf["model"]["type"]` (VIT classification cross-entropy, SAP/UNETR
    segmentation Dice(+CE) loss, MAE/DiffusionVIT reconstruction MSE loss).
    MAE additionally supports `conf["model"]["loss_fn"] == "maskMSE"`
    (`masked_mse`, `UCF_VIT.utils.metrics`): reconstruction MSE computed only
    over the masked (encoder-hidden) patches, the standard MAE-paper loss,
    instead of `"MSE"`'s plain `nn.MSELoss()` over every patch (masked and
    visible alike). MAE with `ap.do_ap:True` additionally supports
    `"nativeResMSE"`/`"nativeResMaskMSE"` (`native_resolution_patch_mse`/
    `native_resolution_patch_masked_mse`): instead of comparing the
    prediction against the already-resized, fixed-`interp_size` token
    (what `"MSE"`/`"maskMSE"` compare against under `do_ap:True`), compares
    it against the real, native-resolution image region each adaptive
    patch actually covers -- a more faithful reconstruction objective for
    small/detailed patches, whose fixed-size token is already a lossy,
    downsampled version of the real pixels.

    Args:
        conf: Parsed training configuration dict (as returned by `parse_config`).
        batch: Dict of batch tensors as returned by `process_batch`.
        model: Model to run the forward pass on.

    Returns:
        For "VIT" and "UNETR": a tuple `(loss, output)`. For "SAP", "MAE", and
        "DiffusionVIT": just `loss`.
    """

    if conf["model"]["type"] == "VIT":
        if conf["ap"]["do_ap"]:
            output = model.forward(batch["seq"], batch["variables"], batch["seq_ps"])
        else:
            output = model.forward(batch["data"], batch["variables"], batch["seq_ps"])
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, batch["label"])

        return loss, output
    
    elif conf["model"]["type"] == "SAP":
        if conf["data"]["twoD"]:
            #seq = torch.reshape(seq, shape=(-1,in_chans,patch_size*sqrt_len, patch_size*sqrt_len))
            seq = einops.rearrange(batch["seq"], 'b c (s1 s2) (ps1 ps2)-> b c (s1 ps1) (s2 ps2)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["interp_size"], ps2=conf["data"]["interp_size"])
            #seq_label = torch.reshape(seq_label, shape=(-1,num_classes,patch_size*sqrt_len, patch_size*sqrt_len))
            seq_label = einops.rearrange(batch["seq_label"], 'b c (ps1 ps2) (s1 s2)-> b c (s1 ps1) (s2 ps2)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["interp_size"], ps2=conf["data"]["interp_size"])

        else:
            #seq = torch.reshape(seq, shape=(-1,in_chans,patch_size*sqrt_len, patch_size*sqrt_len, patch_size*sqrt_len))
            seq = einops.rearrange(batch["seq"], 'b c (s1 s2 s3) (ps1 ps2 ps3)-> b c (s1 ps1) (s2 ps2) (s3 ps3)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], s3=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["interp_size"], ps2=conf["data"]["interp_size"], ps3=conf["data"]["interp_size"])
            #seq_label = torch.reshape(seq_label, shape=(-1,num_classes,patch_size*sqrt_len, patch_size*sqrt_len, patch_size*sqrt_len))
            seq_label = einops.rearrange(batch["seq_label"], 'b c (ps1 ps2 ps3) (s1 s2 s3)-> b c (s1 ps1) (s2 ps2) (s3 ps3)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], s3=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["interp_size"], ps2=conf["data"]["interp_size"], ps3=conf["data"]["interp_size"])
        
        output = model.forward(seq, batch["variables"], batch["seq_ps"])
        criterion = DiceBLoss(num_class=conf["model"]["kwargs"]["num_classes"])
        loss = criterion(output,seq_label)
        return loss

    elif conf["model"]["type"] == "MAE":
        if conf["ap"]["do_ap"]:
            if conf["model"]["loss_fn"] == "MSE":
                output, _ = model.forward(batch["seq"], batch["variables"], batch["seq_ps"])
                criterion = nn.MSELoss()
                target = einops.rearrange(batch["seq"], 'b c s p -> b s (p c)')
                loss = criterion(output, target)
            elif conf["model"]["loss_fn"] == "maskMSE":
                output, mask = model.forward(batch["seq"], batch["variables"], batch["seq_ps"])
                criterion = masked_mse
                target = einops.rearrange(batch["seq"], 'b c s p -> b s (p c)')
                loss = criterion(output, target, mask)
            elif conf["model"]["loss_fn"] in ("nativeResMSE", "nativeResMaskMSE"):
                # batch["seq_ps"] is process_batch's own combined
                # [size, pos] tensor (built for the adaptive positional
                # embedding), reused here as native_resolution_patch_mse's
                # size/pos arguments -- unsqueeze(1) restores the
                # adaptive_patching_channels==1 dim process_batch's own
                # torch.squeeze already dropped (see process_batch's own
                # comment; only the separate_channels:False case is
                # supported here, same limitation seq_ps itself already has).
                output, mask = model.forward(batch["seq"], batch["variables"], batch["seq_ps"])
                seq_size = batch["seq_ps"][..., 0].unsqueeze(1)
                seq_pos = batch["seq_ps"][..., 1:].unsqueeze(1)
                if conf["model"]["loss_fn"] == "nativeResMSE":
                    loss = native_resolution_patch_mse(output, batch["data"], seq_size, seq_pos, conf["data"]["interp_size"], conf["data"]["twoD"])
                else:
                    loss = native_resolution_patch_masked_mse(output, batch["data"], seq_size, seq_pos, conf["data"]["interp_size"], conf["data"]["twoD"], mask.unsqueeze(1))

        else:
            if conf["model"]["loss_fn"] == "MSE":
                output, _ = model.forward(batch["data"], batch["variables"], batch["seq_ps"])
                criterion = nn.MSELoss()
                target = patchify(batch["data"], conf["data"]["patch_size"], conf["data"]["twoD"])
                loss = criterion(output,target)
            elif conf["model"]["loss_fn"] == "maskMSE":
                output, mask = model.forward(batch["data"], batch["variables"], batch["seq_ps"])
                criterion = masked_mse
                target = patchify(batch["data"], conf["data"]["patch_size"], conf["data"]["twoD"])
                loss = criterion(output, target, mask)

        return loss

    elif conf["model"]["type"] == "UNETR":
        if conf["ap"]["do_ap"]:
            if conf["data"]["twoD"]:
                seq = einops.rearrange(batch["seq"], 'b c (s1 s2) (ps1 ps2)-> b c (s1 ps1) (s2 ps2)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["interp_size"], ps2=conf["data"]["interp_size"])
            else:
                seq = einops.rearrange(batch["seq"], 'b c (s1 s2 s3) (ps1 ps2 ps3)-> b c (s1 ps1) (s2 ps2) (s3 ps3)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], s3=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["interp_size"], ps2=conf["data"]["interp_size"], ps3=conf["data"]["interp_size"])

            output = model.forward(batch["data"], batch["variables"], batch["seq_ps"], seq)

        else:
            output = model.forward(batch["data"], batch["variables"])
            

        criterion = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6)
        loss = criterion(output, batch["label"])

        return loss, output

    elif conf["model"]["type"] == "DiffusionVIT":
        output = model.forward(batch["data"], batch["t"], batch["variables"])
        output = unpatchify(output, batch["data"], conf["data"]["patch_size"], conf["data"]["twoD"])
        criterion = nn.MSELoss()
        loss = criterion(output,batch["e"])

        return loss

def get_batch(conf, it_loader):
    """Pulls the next batch from a dataloader iterator and packages it into a dict.

    The exact tuple unpacked from `it_loader` depends on `conf["model"]["type"]` and
    whether adaptive patching is enabled.

    Args:
        conf: Parsed training configuration dict (as returned by `parse_config`).
        it_loader: Iterator over the training dataloader.

    Returns:
        Dict with keys "data", "variables", "dict_key", "seq", "seq_size",
        "seq_pos", "label", and "seq_label"; entries not applicable to the current
        model type/adaptive-patching setting are set to None.
    """
    if conf["model"]["type"] == "VIT":
        if conf["ap"]["do_ap"]:
            data, seq, seq_size, seq_pos, label, variables, dict_key = next(it_loader)
        else:
            data, label, variables, dict_key = next(it_loader)

    elif conf["model"]["type"] in ["UNETR", "SAP"]:
        if conf["ap"]["do_ap"]:
            data, seq, seq_size, seq_pos, label, seq_label, variables, dict_key = next(it_loader)
        else:
            data, label, variables, dict_key = next(it_loader)

    elif conf["model"]["type"] in ["MAE", "DiffusionVIT"]:
        if conf["ap"]["do_ap"]:
            data, seq, seq_size, seq_pos, variables, dict_key = next(it_loader)
        else:
            data, variables, dict_key = next(it_loader)
    #TODO: Add other Model types

    return { "data": data,
             "variables": variables,
             "dict_key": dict_key,
             "seq": seq if conf["ap"]["do_ap"] else None,
             "seq_size": seq_size if conf["ap"]["do_ap"] else None,
             "seq_pos": seq_pos if conf["ap"]["do_ap"] else None,
             "label": label if conf["dataloader"]["return_label"] else None,
             "seq_label": seq_label if conf["dataloader"]["return_label"] and conf["ap"]["do_ap"] and conf["model"]["type"] in ["UNETR", "SAP"] else None,
           }

def process_batch(conf, train_dataloader, device, tensor_par_group, ddpm_scheduler):
    """Fetches a training batch and distributes it across a tensor-parallel group.

    When tensor parallelism is enabled (`tensor_par_size > 1`), only rank 0 of each
    tensor-parallel group reads from `train_dataloader`; the batch's tensors,
    variable list, and dataset key are then broadcast to the rest of the group
    (other ranks pre-allocate correctly-shaped placeholder tensors to broadcast
    into). For "DiffusionVIT", also samples a random timestep `t` and noise `e` per
    batch and forms the noised input. Also reshapes adaptive-patching `seq_size`/
    `seq_pos` into the combined `seq_ps` tensor used for adaptive position
    embeddings.

    Args:
        conf: Parsed training configuration dict (as returned by `parse_config`).
        train_dataloader: Training dataloader to read the next batch from (read only
            by rank 0 of `tensor_par_group`).
        device: Device to move batch tensors to.
        tensor_par_group: Process group for tensor-parallel broadcast of the batch.
        ddpm_scheduler: `DDPM_Scheduler` used to look up alpha values for noising the
            input, when `conf["model"]["type"] == "DiffusionVIT"`.

    Returns:
        Dict with keys "data", "variables", "dict_key", "seq", "seq_ps", "label",
        "seq_label", "t", and "e"; entries not applicable to the current model
        type/adaptive-patching setting are set to None.
    """
    tensor_par_size = conf["parallelism"]["tensor_par_size"]

    if conf["trainer"]["data_type"] == "float32":
        precision_dt = torch.float32
    elif conf["trainer"]["data_type"] == "bfloat16":
        precision_dt = torch.bfloat16
    else:
        raise RuntimeError("Data type not supported")

    if tensor_par_size == 1:
        it_loader = iter(train_dataloader)

        if conf["ap"]["do_ap"]:
            batch = get_batch(conf, it_loader)
            data = batch["data"].to(precision_dt).to(device)
            seq = batch["seq"].to(precision_dt).to(device)
            # Assigned as locals (not just left inside `batch`) so the
            # seq_ps conversion below can read them the same way regardless
            # of tensor_par_size -- see that block's comment.
            seq_size = batch["seq_size"]
            seq_pos = batch["seq_pos"]
            dict_key = batch["dict_key"]
            variables = batch["variables"]
            if conf["dataloader"]["return_label"]:
                label = batch["label"].to(device)
                if conf["model"]["type"] in ["UNETR", "SAP"]: #Classification
                    seq_label = batch["seq_label"].to(device)
        else:
            batch = get_batch(conf, it_loader)
            data = batch["data"].to(precision_dt).to(device)
            variables = batch["variables"]
            dict_key = batch["dict_key"]
            if conf["dataloader"]["return_label"]:
                label = batch["label"].to(device)
            if conf["model"]["type"] == "DiffusionVIT":
                t = torch.randint(0,conf["model"]["kwargs"]["time_steps"],(conf["dataloader"]["batch_size"],))
                e = torch.randn_like(data, requires_grad=False)
                if conf["data"]["twoD"]:
                    a = ddpm_scheduler.alpha[t].view(conf["dataloader"]["batch_size"],1,1,1).to(precision_dt).to(device)
                else:
                    a = ddpm_scheduler.alpha[t].view(conf["dataloader"]["batch_size"],1,1,1,1).to(precision_dt).to(device)
                data = (torch.sqrt(a)*data) + (torch.sqrt(1-a)*e)

    else: #tensor_par_size > 1 
        dataset = conf["data"]["dataset"]
        batch_size = conf["dataloader"]["batch_size"]
        num_channels = conf["data"]["num_channels"]
        twoD = conf["data"]["twoD"]
        # tile_size is needed unconditionally below (it shapes the raw
        # `data` placeholder for non-rank-0 processes regardless of do_ap --
        # `data` is always the pre-patchification image), not just when
        # do_ap is False.
        tile_size = conf["data"]["tile_size"]
        if conf["ap"]["do_ap"]:
            fixed_length = conf["ap"]["fixed_length"]
            interp_size = conf["data"]["interp_size"]
            separate_channels = conf["ap"]["separate_channels"]

        if dist.get_rank(tensor_par_group) == 0:
            it_loader = iter(train_dataloader)

        if conf["ap"]["do_ap"]:
            if dist.get_rank(tensor_par_group) == 0:
                batch = get_batch(conf, it_loader)
                data = batch["data"].to(precision_dt).to(device)
                seq = batch["seq"].to(precision_dt).to(device)
                # Must match the receivers' placeholder (dtype=precision_dt,
                # .to(device)) exactly, or the broadcast below either fails
                # outright (NCCL has no CPU backend -- "No backend type
                # associated with device type cpu") or silently mismatches
                # dtype across ranks.
                seq_size = batch["seq_size"].to(precision_dt).to(device)
                seq_pos = batch["seq_pos"].to(precision_dt).to(device)
                variables = batch["variables"]
                dict_key = batch["dict_key"]
                if conf["dataloader"]["return_label"]:
                    label = batch["label"].to(device)
                    if conf["model"]["type"] in ["UNETR", "SAP"]:
                        # Must match the receiver's dtype=precision_dt
                        # placeholder below -- real seq_label is float32
                        # (datamodule.py's seq_mask.permute(2,0,1).float()),
                        # which only happens to already match precision_dt
                        # when data_type:float32 is configured; explicit here
                        # so bfloat16 configs don't hit the same
                        # sender/receiver dtype mismatch just fixed for
                        # label above.
                        seq_label = batch["seq_label"].to(precision_dt).to(device)

                if dataset == "imagenet":
                    dict_key = "imagenet"
            else:
                if dataset == "imagenet":
                    dict_key = "imagenet"

            if dataset != "imagenet":
                # broadcast_object_list already handles a variable-length
                # pickled object directly (that's the whole point of the
                # "object" collectives, unlike dist.broadcast which needs a
                # fixed pre-known tensor shape) -- broadcasting dict_key
                # itself as a single-element list is simpler and more robust
                # than the previous character-splitting scheme (a separate
                # length broadcast + list(dict_key) + ''.join(...)), which
                # was fragile and, even after fixing its str-vs-list and
                # missing-device bugs, still occasionally produced an empty
                # dict_key on the receiver for reasons never fully isolated.
                # device=device: broadcast_object_list's own docs warn that
                # for NCCL groups its internal object-size/pickled-bytes
                # tensors must live on this rank's GPU, and without an
                # explicit device it falls back to
                # torch.cuda.current_device() -- relying on that global
                # implicitly (rather than the device this function already
                # has in hand) risks the same kind of corruption.
                dict_key_holder = [dict_key] if dist.get_rank(tensor_par_group) == 0 else [None]
                dist.broadcast_object_list(dict_key_holder, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group, device=device)
                dict_key = dict_key_holder[0]

            if dist.get_rank(tensor_par_group) != 0:
                if twoD:
                    data = torch.zeros(batch_size, num_channels[dict_key], tile_size[0], tile_size[1], dtype=precision_dt).to(device)
                    seq = torch.zeros(batch_size, num_channels[dict_key], fixed_length, interp_size*interp_size, dtype=precision_dt).to(device)
                    if separate_channels:
                        seq_size = torch.zeros(batch_size, num_channels[dict_key], fixed_length, dtype=precision_dt).to(device)
                        seq_pos = torch.zeros(batch_size, num_channels[dict_key], fixed_length, 2, dtype=precision_dt).to(device)
                    else:
                        # Real seq_size is (batch_size, 1, fixed_length) --
                        # a scalar per-patch side length, no trailing
                        # coordinate dim (see datamodule.py's
                        # np.expand_dims(batch[i][2], axis=0)-based
                        # construction). Real seq_pos is (batch_size, 1,
                        # fixed_length, 2) for twoD -- a per-patch (x, y)
                        # center position, so the trailing dim must be 2,
                        # not 1.
                        seq_size = torch.zeros(batch_size, 1, fixed_length, dtype=precision_dt).to(device)
                        seq_pos = torch.zeros(batch_size, 1, fixed_length, 2, dtype=precision_dt).to(device)

                    if conf["dataloader"]["return_label"]:
                        if conf["model"]["type"] == "VIT": #Classification
                            # Real classification labels are flat per-sample
                            # class indices (shape (batch_size,), int64 --
                            # see e.g. datamodule.py's
                            # torch.tensor(batch[i][4])-based construction),
                            # not (batch_size, 1) floats -- this placeholder
                            # must match exactly, since dist.broadcast fills
                            # values into the existing tensor without
                            # reshaping/casting it.
                            label = torch.zeros(batch_size, dtype=torch.int64).to(device)
                        else: #Segmentation
                            # Real basic_ct label is uint8 (see dataset.py's
                            # np.asarray(np_label, dtype=np.uint8)), not
                            # precision_dt -- dist.broadcast fills values into
                            # the existing tensor without casting, so a
                            # dtype mismatch between sender and receiver here
                            # (previously precision_dt, a float type) silently
                            # corrupts the transfer -- this exact
                            # do_ap:True + Segmentation + tensor_par_size>1
                            # combination was never exercised before
                            # basic_ct-sap+tensor_par (SAP is the only model
                            # requiring do_ap:True).
                            label = torch.zeros(batch_size, 1, tile_size[0], tile_size[1], dtype=torch.uint8).to(device)
                            if conf["model"]["type"] in ["UNETR", "SAP"]:
                                # Real seq_label is (batch_size, num_classes,
                                # patch_size*patch_size, fixed_length) -- the
                                # patch-volume dim comes BEFORE fixed_length,
                                # not after (see dataset.py's
                                # np.reshape(seq_label, [patch_size**2, -1,
                                # 1]) and datamodule.py's
                                # seq_mask.permute(2, 0, 1) stacking, which
                                # together put patch_size**2 ahead of
                                # fixed_length). train_step's einops.rearrange
                                # ('b c (ps1 ps2) (s1 s2) -> ...') relies on
                                # this exact order -- the previous
                                # (fixed_length, patch_size*patch_size)
                                # ordering here matched the real tensor's
                                # *total* element count but not its per-axis
                                # shape, so it broadcast fine but then failed
                                # downstream with einops.EinopsError: Shape
                                # mismatch on non-rank-0 processes.
                                seq_label = torch.zeros(batch_size, conf["model"]["kwargs"]["num_classes"], interp_size*interp_size, fixed_length, dtype=precision_dt).to(device)
                else:
                    data = torch.zeros(batch_size, num_channels[dict_key], tile_size[0], tile_size[1], tile_size[2], dtype=precision_dt).to(device)
                    seq = torch.zeros(batch_size, num_channels[dict_key], fixed_length, interp_size*interp_size*interp_size, dtype=precision_dt).to(device)
                    if separate_channels:
                        seq_size = torch.zeros(batch_size, num_channels[dict_key], fixed_length, dtype=precision_dt).to(device)
                        seq_pos = torch.zeros(batch_size, num_channels[dict_key], fixed_length, 3, dtype=precision_dt).to(device)
                    else:
                        # Same reasoning as the twoD branch above -- real
                        # seq_pos for 3D has a trailing (x, y, z) coordinate
                        # dim of 3, not 1.
                        seq_size = torch.zeros(batch_size, 1, fixed_length, dtype=precision_dt).to(device)
                        seq_pos = torch.zeros(batch_size, 1, fixed_length, 3, dtype=precision_dt).to(device)

                    if conf["dataloader"]["return_label"]:
                        if conf["model"]["type"] == "VIT": #Classification
                            # Same reasoning as the twoD branch above.
                            label = torch.zeros(batch_size, dtype=torch.int64).to(device)
                        else: #Segmentation
                            # Same real-uint8-vs-precision_dt reasoning as
                            # the twoD branch above.
                            label = torch.zeros(batch_size, 1, tile_size[0], tile_size[1], tile_size[2], dtype=torch.uint8).to(device)
                            if conf["model"]["type"] in ["UNETR", "SAP"]:
                                # Same patch-volume-before-fixed_length
                                # reasoning as the twoD branch above.
                                seq_label = torch.zeros(batch_size, conf["model"]["kwargs"]["num_classes"], interp_size*interp_size*interp_size, fixed_length, dtype=precision_dt).to(device)
                variables = [None] * num_channels[dict_key]

            #Broadcast data batch to the rest of the tensor parallel group
            dist.broadcast(data, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast(seq, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast(seq_size, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast(seq_pos, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast_object_list(variables, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group, device=device)

            if conf["dataloader"]["return_label"]:
                dist.broadcast(label, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                if conf["model"]["type"] in ["UNETR", "SAP"]:
                    dist.broadcast(seq_label, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

        else:
            if dist.get_rank(tensor_par_group) == 0:
                batch = get_batch(conf, it_loader)
                data = batch["data"].to(precision_dt).to(device)
                dict_key = batch["dict_key"]
                variables = batch["variables"]
                if conf["dataloader"]["return_label"]:
                    label = batch["label"].to(device)
                if conf["model"]["type"] == "DiffusionVIT":
                    t = torch.randint(0,conf["model"]["kwargs"]["time_steps"],(batch_size,))
                    e = torch.randn_like(data, requires_grad=False)
                    if twoD:
                        a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1).to(precision_dt).to(device)
                    else:
                        a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1,1).to(precision_dt).to(device)
                    t = t.to(device)
                    data = (torch.sqrt(a)*data) + (torch.sqrt(1-a)*e)

                if dataset == "imagenet":
                    dict_key = "imagenet"
            else:
                if dataset == "imagenet":
                    dict_key = "imagenet"

            if dataset != "imagenet":
                # Same simplified single-object broadcast as the do_ap:True
                # branch above -- see its comment for why.
                dict_key_holder = [dict_key] if dist.get_rank(tensor_par_group) == 0 else [None]
                dist.broadcast_object_list(dict_key_holder, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group, device=device)
                dict_key = dict_key_holder[0]

            if dist.get_rank(tensor_par_group) != 0:
                if twoD:
                    data = torch.zeros(batch_size, num_channels[dict_key], tile_size[0], tile_size[1], dtype=precision_dt).to(device)
                    if conf["dataloader"]["return_label"]:
                        if conf["model"]["type"] == "VIT": #Classification
                            # Same reasoning as the do_ap:True branch above:
                            # real classification labels are (batch_size,)
                            # int64, not (batch_size, 1) float.
                            label = torch.zeros(batch_size, dtype=torch.int64).to(device)
                        else: #Segmentation
                            # Real basic_ct label (do_ap:False) is int64 (see
                            # dataset.py's
                            # np.array(label.dataobj).astype(np.int64)), not
                            # precision_dt -- same NCCL dtype/byte-size
                            # requirement as the t/e placeholders below and
                            # the do_ap:True Segmentation label placeholder
                            # above.
                            label = torch.zeros(batch_size, 1, tile_size[0], tile_size[1], dtype=torch.int64).to(device)
                else:
                    data = torch.zeros(batch_size, num_channels[dict_key], tile_size[0], tile_size[1], tile_size[2], dtype=precision_dt).to(device)
                    if conf["dataloader"]["return_label"]:
                        if conf["model"]["type"] == "VIT": #Classification
                            label = torch.zeros(batch_size, dtype=torch.int64).to(device)
                        else: #Segmentation
                            # Same real-int64-vs-precision_dt reasoning as
                            # the twoD branch above.
                            label = torch.zeros(batch_size, 1, tile_size[0], tile_size[1], tile_size[2], dtype=torch.int64).to(device)
                if conf["model"]["type"] == "DiffusionVIT":
                    # torch.randint's default dtype is int64, not int32
                    # ("torch.int") -- must match exactly for the broadcast
                    # below (NCCL requires identical dtype/byte-size across
                    # ranks).
                    t = torch.zeros(batch_size, dtype=torch.int64).to(device)
                    e = torch.zeros_like(data, requires_grad=False)
                variables = [None] * num_channels[dict_key]

            #Broadcast data batch to the rest of the tensor parallel group
            dist.broadcast(data, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast_object_list(variables, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group, device=device)

            if conf["dataloader"]["return_label"]:
                dist.broadcast(label, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

            if conf["model"]["type"] == "DiffusionVIT":
                dist.broadcast(t, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                t = t.to('cpu')
                dist.broadcast(e, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

    #Convert seq_size and seq_pos to form used for adaptive position embedding
    if conf["ap"]["do_ap"]:
        if conf["ap"]["separate_channels"]:
            #TODO: Move seq_size and seq_pos to a single channel
            seq_ps = None
        else:
            # Reads the local seq_size/seq_pos (not batch["seq_size"]/
            # batch["seq_pos"]) since `batch` is only ever assigned on
            # tensor_par_group-rank-0 when tensor_par_size > 1 -- every other
            # rank would hit UnboundLocalError here otherwise. The local
            # variables already hold the right values on every rank
            # regardless of tensor_par_size: for tensor_par_size == 1 they're
            # assigned straight from batch[...] above; for tensor_par_size > 1
            # they're each rank's own already-broadcast copy.
            seq_size = torch.squeeze(seq_size)
            seq_size = seq_size.to(torch.float32)
            seq_size = seq_size.to(device)
            seq_pos = torch.squeeze(seq_pos)
            seq_pos = seq_pos.to(torch.float32)
            seq_pos = seq_pos.to(device)
            seq_size = seq_size.unsqueeze(-1)
            seq_ps = torch.concat([seq_size, seq_pos],dim=-1)

    return { "data": data,
             "variables": variables,
             "dict_key": dict_key,
             "seq": seq if conf["ap"]["do_ap"] else None,
             "seq_ps": seq_ps if conf["ap"]["do_ap"] else None,
             "label": label if conf["dataloader"]["return_label"] else None,
             "seq_label": seq_label if conf["dataloader"]["return_label"] and conf["ap"]["do_ap"] and conf["model"]["type"] in ["UNETR", "SAP"] else None,
             "t": t if conf["model"]["type"] == "DiffusionVIT" else None,
             "e": e if conf["model"]["type"] == "DiffusionVIT" else None,
           }
      

def save_checkpoint(conf, model, optimizer, scheduler, epoch, loss_list):
    """Saves model/optimizer/scheduler state and loss history to a per-rank checkpoint file.

    Only ranks below `conf["parallelism"]["tensor_par_size"]` (i.e. rank 0 of each
    tensor-parallel group) write a checkpoint file. All ranks synchronize on a
    barrier afterward.

    Args:
        conf: Parsed training configuration dict (as returned by `parse_config`).
        model: Model whose state dict is saved.
        optimizer: Optimizer whose state dict is saved.
        scheduler: LR scheduler whose state dict is saved.
        epoch: Current epoch number, saved alongside the state and used in the
            checkpoint filename.
        loss_list: Accumulated loss history to save.
    """
    model_states = model.state_dict()
    optimizer_states = optimizer.state_dict()
    scheduler_states = scheduler.state_dict()

    if dist.get_rank() < conf["parallelism"]["tensor_par_size"]:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_states,
            'optimizer_state_dict': optimizer_states,
            'scheduler_state_dict': scheduler_states,
            'loss_list' : loss_list,
            }, conf["trainer"]["checkpoint_path"]+"/epoch_"+str(epoch)+"_rank_"+str(dist.get_rank())+".ckpt")

    dist.barrier()
    #del model_states
    #del optimizer_states
    #del scheduler_states

def train_epoch(conf, model, train_dataloader, epoch, iterations_per_epoch, optimizer, scheduler, grad_scaler, min_scale, loss_list, device, tensor_par_group, ddpm_scheduler):
    """Runs one full training epoch: batch loop, backward pass, optimizer/scheduler step, checkpointing.

    For each of `iterations_per_epoch` iterations, fetches and processes a batch,
    runs the forward pass and loss, computes accuracy/Dice metric where applicable,
    backpropagates (optionally through a gradient scaler), and steps the optimizer.
    After the loop, steps the scheduler, appends the epoch loss to `loss_list`, and
    saves a checkpoint if `epoch` falls on the configured save frequency.

    Args:
        conf: Parsed training configuration dict (as returned by `parse_config`).
        model: Model being trained.
        train_dataloader: Training dataloader.
        epoch: Current epoch number.
        iterations_per_epoch: Number of batches to process this epoch.
        optimizer: Optimizer to step.
        scheduler: LR scheduler to step once per epoch.
        grad_scaler: Gradient scaler used when `conf["grad_scaler"]["use_grad_scaler"]`
            is True.
        min_scale: Minimum allowed grad scaler scale, enforced after each update.
        loss_list: Loss history list; the epoch's total loss is appended in place.
        device: Device to run the epoch's loss/accuracy accumulators on.
        tensor_par_group: Process group for tensor-parallel batch distribution.
        ddpm_scheduler: `DDPM_Scheduler` used when training a "DiffusionVIT" model.
    """

    epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
    epoch_accuracy = torch.tensor(0.0 , dtype=torch.float32, device=device)
    counter = 0
    while counter < iterations_per_epoch:
        counter = counter + 1

        batch = process_batch(conf, train_dataloader, device, tensor_par_group, ddpm_scheduler)

        if conf["model"]["type"] in ["VIT", "UNETR"]:
            loss, output = train_step(conf, batch, model)
        elif conf["model"]["type"] in ["MAE", "SAP", "DiffusionVIT"]:
            loss = train_step(conf, batch, model)

        epoch_loss += loss.detach()

        if conf["model"]["type"] == "VIT":
            acc = (output.argmax(dim=1) == batch["label"]).float().mean()
            epoch_accuracy += acc.detach()

        elif conf["model"]["type"] == "UNETR":
            post_label = AsDiscrete(to_onehot=conf["model"]["kwargs"]["num_classes"])
            post_pred = AsDiscrete(argmax=True, to_onehot=conf["model"]["kwargs"]["num_classes"])
            dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)

            train_labels_list = decollate_batch(batch["label"])
            train_labels_convert = [post_label(train_label_tensor) for train_label_tensor in train_labels_list]
            train_outputs_list = decollate_batch(output)
            train_output_convert = [post_pred(train_pred_tensor) for train_pred_tensor in train_outputs_list]
            acc = dice_acc(y_pred=train_output_convert, y=train_labels_convert)


        if dist.get_rank() == 0:
            if conf["model"]["type"] in ["VIT", "UNETR"]:
                print("epoch: ",epoch, "batch_idx", counter, "it_loss ",loss, "it_acc", acc, flush=True)
            elif conf["model"]["type"] in ["MAE", "SAP", "DiffusionVIT"]:
                print("epoch: ", epoch, "batch_idx", counter, "it_loss ", loss, flush=True)

        if conf["grad_scaler"]["use_grad_scaler"]:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if grad_scaler._scale < min_scale:
                grad_scaler._scale = torch.tensor(min_scale).to(grad_scaler._scale)
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    loss_list.append(epoch_loss)
    if dist.get_rank() == 0:
        if conf["model"]["type"] == "VIT":
            print("epoch: ", epoch, "epoch_loss ", epoch_loss, "epoch_accuracy ", epoch_accuracy, flush=True)
        elif conf["model"]["type"] in ["MAE", "UNETR", "SAP", "DiffusionVIT"]:
            print("epoch: ", epoch, "epoch_loss ", epoch_loss, flush=True)

    if epoch % conf["trainer"]["save_frequency"] == 0:
        save_checkpoint(conf, model, optimizer, scheduler, epoch, loss_list)

