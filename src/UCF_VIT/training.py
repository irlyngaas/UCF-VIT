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
from UCF_VIT.utils.metrics import DiceBLoss

def load_optimizer_scheduler_from_checkpoint(conf, optimizer, scheduler, data_seq_ort_group, device):
    src_rank = dist.get_rank() - conf["parallelism"]["tensor_par_size"] * dist.get_rank(group=data_seq_ort_group)

    map_location = 'cpu' #TODO: Choose cpu or cuda+str
    #map_location = 'cuda:'+str(device)

    checkpoint = torch.load(conf["trainer"]["checkpoint_path"]+"/"+conf["trainer"]["checkpoint_filename"]+"_rank_"+str(src_rank)+".ckpt",map_location=map_location)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    #TODO: Load loss_list and epoch_start in create_model loading from checkpoint rather than here
    loss_list = checkpoint['loss_list']
    epoch_start = checkpoint['epoch'] + 1


    return optimizer, scheduler, loss_list, epoch_start

def train_step(conf, batch, model):

    if conf["model"]["type"] == "VIT":
        output = model.forward(batch["data"], batch["variables"], batch["seq_ps"])
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, batch["label"])

        return loss, output
    
    elif conf["model"]["type"] == "SAP":
        if conf["data"]["twoD"]:
            #seq = torch.reshape(seq, shape=(-1,in_chans,patch_size*sqrt_len, patch_size*sqrt_len))
            seq = einops.rearrange(batch["seq"], 'b c (s1 s2) (ps1 ps2)-> b c (s1 ps1) (s2 ps2)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["patch_size"], ps2=conf["data"]["patch_size"])
            #seq_label = torch.reshape(seq_label, shape=(-1,num_classes,patch_size*sqrt_len, patch_size*sqrt_len))
            seq_label = einops.rearrange(batch["seq_label"], 'b c (ps1 ps2) (s1 s2)-> b c (s1 ps1) (s2 ps2)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["patch_size"], ps2=conf["data"]["patch_size"])

        else:
            #seq = torch.reshape(seq, shape=(-1,in_chans,patch_size*sqrt_len, patch_size*sqrt_len, patch_size*sqrt_len))
            seq = einops.rearrange(batch["seq"], 'b c (s1 s2 s3) (ps1 ps2 ps3)-> b c (s1 ps1) (s2 ps2) (s3 ps3)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], s3=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["patch_size"], ps2=conf["data"]["patch_size"], ps3=conf["data"]["patch_size"])
            #seq_label = torch.reshape(seq_label, shape=(-1,num_classes,patch_size*sqrt_len, patch_size*sqrt_len, patch_size*sqrt_len))
            seq_label = einops.rearrange(batch["seq_label"], 'b c (ps1 ps2 ps3) (s1 s2 s3)-> b c (s1 ps1) (s2 ps2) (s3 ps3)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], s3=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["patch_size"], ps2=conf["data"]["patch_size"], ps3=conf["data"]["patch_size"])
        
        output = model.forward(seq, batch["variables"], batch["seq_ps"])
        criterion = DiceBLoss(num_class=conf["model"]["kwargs"]["num_classes"])
        loss = criterion(output,seq_label)
        return loss

    elif conf["model"]["type"] == "MAE":
        if conf["ap"]["do_ap"]:
            if conf["model"]["loss_fn"] == "MSE":
                output, _ = model.forward(batch["seq"], batch["variables"], batch["seq_ps"])
                criterion = nn.MSELoss()
                target = rearrange(seq, 'b c s p -> b s (p c)')
                loss = criterion(output, target)
            #TODO: elif conf["model"]["kwargs"]["loss_fn"] == "maskMSE":

        else:
            if conf["model"]["loss_fn"] == "MSE":
                output, _ = model.forward(batch["data"], batch["variables"], batch["seq_ps"])
                criterion = nn.MSELoss()
                target = patchify(batch["data"], conf["data"]["patch_size"], conf["data"]["twoD"])
                loss = criterion(output,target)
            #elif conf["model"]["kwargs"]["loss_fn"] == "maskMSE":
            #    output, mask = net.forward(data, variables, None)
            #    criterion = masked_mse
            #    target = patchify(data, patch_size, twoD)
            #    loss = criterion(output,target,mask)

        return loss

    elif conf["model"]["type"] == "UNETR":
        if conf["ap"]["do_ap"]:
            if conf["data"]["twoD"]:
                seq = einops.rearrange(batch["seq"], 'b c (s1 s2) (ps1 ps2)-> b c (s1 ps1) (s2 ps2)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["patch_size"], ps2=conf["data"]["patch_size"])
            else:
                seq = einops.rearrange(batch["seq"], 'b c (s1 s2 s3) (ps1 ps2 ps3)-> b c (s1 ps1) (s2 ps2) (s3 ps3)', s1=conf["model"]["kwargs"]["sqrt_len"], s2=conf["model"]["kwargs"]["sqrt_len"], s3=conf["model"]["kwargs"]["sqrt_len"], ps1=conf["data"]["patch_size"], ps2=conf["data"]["patch_size"], ps3=conf["data"]["patch_size"])

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
        if conf["ap"]["do_ap"]:
            fixed_length = conf["ap"]["fixed_length"]
            patch_size = conf["data"]["patch_size"]
            separate_channels = conf["ap"]["separate_channels"]
        else:
            tile_size = conf["data"]["tile_size"]

        if dist.get_rank(tensor_par_group) == 0:
            it_loader = iter(train_dataloader)

        if conf["ap"]["do_ap"]:
            if dist.get_rank(tensor_par_group) == 0:
                batch = get_batch(conf, it_loader)
                data = batch["data"].to(precision_dt).to(device)
                seq = batch["seq"].to(precision_dt).to(device)
                seq_size = batch["seq_size"]
                seq_pos = batch["seq_pos"]
                variables = batch["variables"]
                dict_key = batch["dict_key"]
                if conf["dataloader"]["return_label"]:
                    label = batch["label"].to(device)
                    if conf["model"]["type"] in ["UNETR", "SAP"]:
                        seq_label = batch["seq_label"].to(device)

                if dataset != "imagenet":
                    dict_key_len = torch.tensor(len(dict_key)).to(device)
                else:
                    dict_key = "imagenet"
            else:
                if dataset != "imagenet":
                    dict_key_len = torch.tensor(0).to(device)
                else: 
                    dict_key = "imagenet"

            if dataset != "imagenet":
                dist.broadcast(dict_key_len, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group = tensor_par_group)
                if dist.get_rank(tensor_par_group) != 0:
                    dict_key = [None] * dict_key_len.item()
                dist.broadcast_object_list(dict_key, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

                if dist.get_rank(tensor_par_group) != 0:
                    dict_key = ''.join(dict_key)

            if dist.get_rank(tensor_par_group) != 0:
                if twoD:
                    data = torch.zeros(batch_size, num_channels[dict_key], tile_size[0], tile_size[1], dtype=precision_dt).to(device)
                    seq = torch.zeros(batch_size, num_channels[dict_key], fixed_length, patch_size*patch_size, dtype=precision_dt).to(device)
                    if separate_channels:
                        seq_size = torch.zeros(batch_size, num_channels[dict_key], fixed_length, dtype=precision_dt).to(device)
                        seq_pos = torch.zeros(batch_size, num_channels[dict_key], fixed_length, 2, dtype=precision_dt).to(device)
                    else:
                        seq_size = torch.zeros(batch_size, 1, fixed_length, 1, dtype=precision_dt).to(device)
                        seq_pos = torch.zeros(batch_size, 1, fixed_length, 1, 1, dtype=precision_dt).to(device)

                    if conf["dataloader"]["return_label"]:
                        if conf["model"]["type"] == "VIT": #Classification
                            label = torch.zeros(batch_size, 1, dtype=precision_dt).to(device)
                        else: #Segmentation
                            label = torch.zeros(batch_size, 1, tile_size[0], tile_size[1], dtype=precision_dt).to(device)
                            if conf["model"]["type"] in ["UNETR", "SAP"]:
                                seq_label = torch.zeros(batch_size, conf["model"]["kwargs"]["num_classes"], fixed_length, patch_size*patch_size, dtype=precision_dt).to(device)
                else:
                    data = torch.zeros(batch_size, num_channels[dict_key], tile_size[0], tile_size[1], tile_size[2], dtype=precision_dt).to(device)
                    seq = torch.zeros(batch_size, num_channels[dict_key], fixed_length, patch_size*patch_size*patch_size, dtype=precision_dt).to(device)
                    if separate_channels:
                        seq_size = torch.zeros(batch_size, num_channels[dict_key], fixed_length, dtype=precision_dt).to(device)
                        seq_pos = torch.zeros(batch_size, num_channels[dict_key], fixed_length, 3, dtype=precision_dt).to(device)
                    else:
                        seq_size = torch.zeros(batch_size, 1, fixed_length, 1, dtype=precision_dt).to(device)
                        seq_pos = torch.zeros(batch_size, 1, fixed_length, 1, 1, 1, dtype=precision_dt).to(device)

                    if conf["dataloader"]["return_label"]:
                        if conf["model"]["type"] == "VIT": #Classification
                            label = torch.zeros(batch_size, 1, dtype=precision_dt).to(device)
                        else: #Segmentation
                            label = torch.zeros(batch_size, 1, tile_size[0], tile_size[1], tile_size[2], dtype=precision_dt).to(device)
                            if conf["model"]["type"] in ["UNETR", "SAP"]:
                                seq_label = torch.zeros(batch_size, conf["model"]["kwargs"]["num_classes"], fixed_length, patch_size*patch_size*patch_size, dtype=precision_dt).to(device)
                variables = [None] * num_channels[dict_key]

            #Broadcast data batch to the rest of the tensor parallel group
            dist.broadcast(data, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast(seq, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast(seq_size, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast(seq_pos, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast_object_list(variables, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

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

                if dataset != "imagenet":
                    dict_key_len = torch.tensor(len(dict_key)).to(device)
                else:
                    dict_key = "imagenet"
            else:
                if dataset != "imagenet":
                    dict_key_len = torch.tensor(0).to(device)
                else:
                    dict_key = "imagenet"

            if dataset != "imagenet":
                dist.broadcast(dict_key_len, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group = tensor_par_group)

                if dist.get_rank(tensor_par_group) != 0:
                    dict_key = [None] * dict_key_len.item()
                dist.broadcast_object_list(dict_key, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

                if dist.get_rank(tensor_par_group) != 0:
                    dict_key = ''.join(dict_key)

            if dist.get_rank(tensor_par_group) != 0:
                if twoD:
                    data = torch.zeros(batch_size, num_channels[dict_key], tile_size[0], tile_size[1], dtype=precision_dt).to(device)
                    if conf["dataloader"]["return_label"]:
                        if conf["model"]["type"] == "VIT": #Classification
                            label = torch.zeros(batch_size, 1, dtype=precision_dt).to(device)
                        else: #Segmentation
                            label = torch.zeros(batch_size, 1, tile_size[0], tile_size[1], dtype=precision_dt).to(device)
                else:
                    data = torch.zeros(batch_size, num_channels[dict_key], tile_size[0], tile_size[1], tile_size[2], dtype=precision_dt).to(device)
                    if conf["dataloader"]["return_label"]:
                        if conf["model"]["type"] == "VIT": #Classification
                            label = torch.zeros(batch_size, 1, dtype=precision_dt).to(device)
                        else: #Segmentation
                            label = torch.zeros(batch_size, 1, tile_size[0], tile_size[1], tile_size[2], dtype=precision_dt).to(device)
                if conf["model"]["type"] == "DiffusionVIT":
                    t = torch.zeros(batch_size, dtype=torch.int).to(device)
                    e = torch.zeros_like(data, requires_grad=False)
                variables = [None] * num_channels[dict_key]

            #Broadcast data batch to the rest of the tensor parallel group
            dist.broadcast(data, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
            dist.broadcast_object_list(variables, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

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
            seq_size = torch.squeeze(batch["seq_size"])
            seq_size = seq_size.to(torch.float32)
            seq_size = seq_size.to(device)
            seq_pos = torch.squeeze(batch["seq_pos"])
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
                grad_scaler._scale = torch.tensor(min_scale).to(scaler._scale)
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

