import yaml
import os
import sys
import math
import numpy as np
import torch.distributed as dist
from UCF_VIT.utils.misc import is_power_of_two

def get_kwargs(model_type, conf):
    """Build the architecture-specific keyword arguments for a given model type.

    Reads the model-type-specific settings out of the parsed config dict (e.g.
    `num_classes` for VIT, `mask_ratio`/decoder settings for MAE, `feature_size` for
    UNETR, `num_time_steps`/decoder settings for DiffusionVIT), applying defaults and
    exiting with an error message when a required setting is missing.

    Args:
        model_type: Model architecture type. One of "VIT", "SAP", "MAE", "UNETR",
            "DiffusionVIT".
        conf: Raw config dict loaded from the training YAML file.

    Returns:
        A dict of keyword arguments to be passed to the model architecture's
        constructor (via `**kwargs`), specific to `model_type`.
    """
    kwargs = {}
    #TODO: Add checking on each argument, e.g. > 0
    if model_type == "VIT":
        try:
            num_classes = conf['model']['num_classes']
        except KeyError:
            #TODO: If not specified, get num_classes based on dataset used
            sys.exit("num_classes is required")
        kwargs.update({"num_classes": num_classes})

    elif model_type == "SAP":
        try:
            num_classes = conf['model']['num_classes']
        except KeyError:
            #TODO: If not specified, get num_classes based on dataset used
            sys.exit("num_classes is required")
        kwargs.update({"num_classes": num_classes})

        assert conf["ap"]["do_ap"], "SAP requires adaptive patching to be turned on"

        kwargs.update({"sqrt_len_method": True})
        if conf["data"]["twoD"]:
            sqrt_len = int(math.sqrt(conf["ap"]["fixed_length"]))
        else:
            sqrt_len=int(np.rint(math.pow(conf["ap"]["fixed_length"],1/3)))
        kwargs.update({"sqrt_len": sqrt_len})

    elif model_type == "MAE":
        try:
            mask_ratio = conf["model"]["mask_ratio"]
        except KeyError:
            sys.exit("mask_ratio is required")
        kwargs.update({"mask_ratio": mask_ratio})


        try:
            linear_decoder = conf["model"]["linear_decoder"]
        except KeyError:
            if dist.get_rank() == 0:
                print("linear_decoder is not set, by default this is set to False")
            linear_decoder = False
        kwargs.update({"linear_decoder": linear_decoder})

        if not linear_decoder:
            try:
                decoder_depth = conf['model']['decoder_depth']
            except KeyError:
                sys.exit("decoder_depth is required")
            kwargs.update({"decoder_depth": decoder_depth})

            try:
                decoder_embed_dim = conf['model']['decoder_embed_dim']
            except KeyError:
                sys.exit("decoder_embed_dim is required")
            kwargs.update({"decoder_embed_dim": decoder_embed_dim})

            try:
                decoder_num_heads = conf['model']['decoder_num_heads']
            except KeyError:
                sys.exit("decoder_num_heads is required")
            kwargs.update({"decoder_num_heads": decoder_num_heads})

            try:
                decoder_mlp_ratio = conf['model']['decoder_mlp_ratio']
            except KeyError:
                if dist.get_rank() == 0:
                    print("decoder_mlp_ratio not set, by default setting to mlp_ratio")
                decoder_mlp_ratio = mlp_ratio
            kwargs.update({"decoder_mlp_ratio": decoder_mlp_ratio})

        else:
            kwargs.update({"decoder_embed_dim": None})
            kwargs.update({"decoder_depth": None})
            kwargs.update({"decoder_num_heads": None})
            kwargs.update({"decoder_mlp_ratio": None})

    elif model_type == "UNETR":
        try:
            num_classes = conf['model']['num_classes']
        except KeyError:
            #TODO: If not specified, get num_classes based on dataset used
            sys.exit("num_classes is required")
        kwargs.update({"num_classes": num_classes})

        try:
            linear_decoder = conf["model"]["linear_decoder"]
        except KeyError:
            if dist.get_rank() == 0:
                print("linear_decoder is not set, by default this is set to False")
            linear_decoder = False
        kwargs.update({"linear_decoder": linear_decoder})

        try:
            skip_connection = conf["model"]["skip_connection"]
        except KeyError:
            if dist.get_rank() == 0:
                print("skip_connection is not set, by default this is set to True")
            linear_decoder = True
        kwargs.update({"linear_decoder": linear_decoder})

        try:
            feature_size = conf['model']['feature_size']
        except KeyError:
            sys.exit("feature_size is required")
        kwargs.update({"feature_size": feature_size})

        if conf["ap"]["do_ap"]:
            kwargs.update({"sqrt_len_method": True})
            if conf["data"]["twoD"]:
                sqrt_len = int(math.sqrt(conf["ap"]["fixed_length"]))
            else:
                sqrt_len=int(np.rint(math.pow(conf["ap"]["fixed_length"],1/3)))
            kwargs.update({"sqrt_len": sqrt_len})
        else:
            kwargs.update({"sqrt_len_method": False})
            kwargs.update({"sqrt_len": None})

    elif model_type == "DiffusionVIT":
        assert not conf["ap"]["do_ap"], "Adaptive patching is not implemented for Diffusion yet"            

        try:
            num_time_steps = conf['model']['num_time_steps']
        except KeyError:
            #TODO: If not specified, get num_classes based on dataset used
            sys.exit("num_time_steps is required")
        kwargs.update({"time_steps": num_time_steps})

        try:
            linear_decoder = conf["model"]["linear_decoder"]
        except KeyError:
            if dist.get_rank() == 0:
                print("linear_decoder is not set, by default this is set to False")
            linear_decoder = False
        kwargs.update({"linear_decoder": linear_decoder})

        if not linear_decoder:
            try:
                decoder_depth = conf['model']['decoder_depth']
            except KeyError:
                sys.exit("decoder_depth is required")
            kwargs.update({"decoder_depth": decoder_depth})

            try:
                decoder_embed_dim = conf['model']['decoder_embed_dim']
            except KeyError:
                sys.exit("decoder_embed_dim is required")
            kwargs.update({"decoder_embed_dim": decoder_embed_dim})

            try:
                decoder_num_heads = conf['model']['decoder_num_heads']
            except KeyError:
                sys.exit("decoder_num_heads is required")
            kwargs.update({"decoder_num_heads": decoder_num_heads})

            try:
                decoder_mlp_ratio = conf['model']['decoder_mlp_ratio']
            except KeyError:
                if dist.get_rank() == 0:
                    print("decoder_mlp_ratio not set, by default setting to mlp_ratio")
                decoder_mlp_ratio = mlp_ratio
            kwargs.update({"decoder_mlp_ratio": decoder_mlp_ratio})

        else:
            kwargs.update({"decoder_embed_dim": None})
            kwargs.update({"decoder_depth": None})
            kwargs.update({"decoder_num_heads": None})
            kwargs.update({"decoder_mlp_ratio": None})
    

    return kwargs

def parse_config(args, load_balance_offline=False):
    """Load and validate a training configuration YAML file into a structured dict.

    Reads the YAML file at `args.config`, applies defaults for optional settings,
    validates required settings and cross-field consistency (e.g. parallelism sizes
    against world size, tile size divisibility by patch size, adaptive-patching
    fixed-length constraints), and exits with an error message via `sys.exit` when a
    required setting is missing or invalid.

    Args:
        args: Parsed command-line arguments; must have a `config` attribute giving
            the path to the training config YAML file.
        load_balance_offline: If True, skip checks that require a live distributed
            process group (e.g. matching parallelism sizes to `dist.get_world_size()`),
            for use when computing load balancing outside of a training run.

    Returns:
        A dict with keys "trainer", "parallelism", "optimizer", "scheduler",
        "grad_scaler", "model", "tiling", "ap", "data", "dataloader", and
        "dataset_options", each holding the validated configuration for that section.
    """

    with open(args.config,'r') as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)

# ---------------------------- TRAINER -------------------------------------------
    try:
        save_frequency = conf['trainer']['save_frequency']
    except KeyError:
        if dist.get_rank() == 0:
            print("Since no save_frequency was given in the config file, defaulting to saving every epoch")
        save_frequency = 1

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']

    try:
        optimizer_type = conf["optimizer"]["type"]
        assert optimizer_type.lower() in ['sgd', 'adam', 'adamw'], "Optimizer type not supported. Choose optimizer type from the following choices: sgd, adam, adamw"
    except KeyError:
        if dist.get_rank() == 0:
            print("Since no optimizer_type was given in the config file, defaulting to using SGD ")
        optimizer_type = "sgd"

    try:
        scheduler_type = conf["scheduler"]["type"]
        assert scheduler_type in ['constant', 'linear', 'exponential', 'linear-warmup-cosine-annealing', 'reduce-lr-on-plateau'], "Scheduler type not supported. Choose scheduler type from the following choices: constant, linear, exponential, linear-warmup-cosine-annealing, reduce-lr-on-plateau"
    except KeyError:
        if dist.get_rank() == 0:
            print("Since no scheduler_type was given in the config file, defaulting to using linear")
        scheduler_type = "linear"

    trainer_conf = {
        "max_epochs": conf['trainer']['max_epochs'],
        "data_type": conf['trainer']['data_type'],
        "gpu_type": conf['trainer']['gpu_type'],
        "checkpoint_path": conf['trainer']['checkpoint_path'],
        "checkpoint_filename": conf['trainer']['checkpoint_filename'],
        "resume_from_checkpoint": resume_from_checkpoint,
        "use_pretrained_model": conf['trainer']['use_pretrained_model'] if not resume_from_checkpoint else False,
        "save_frequency": save_frequency,
        "optimizer_type": optimizer_type,
        "scheduler_type": scheduler_type,
    }


# ---------------------------- PARALLELISM ---------------------------------------

    #TODO: Add checking on each argument, e.g. > 0
    fsdp_size = conf['parallelism']['fsdp_size']
    simple_ddp_size = conf['parallelism']['simple_ddp_size']
    data_par_size = fsdp_size * simple_ddp_size
    tensor_par_size = conf['parallelism']['tensor_par_size']
    if not load_balance_offline:
        assert (data_par_size * tensor_par_size) == dist.get_world_size(), "DATA_PAR_SIZE * TENSOR_PAR_SIZE must equal world_size"


    parallelism_conf = {
        "fsdp_size": fsdp_size,
        "simple_ddp_size": simple_ddp_size,
        "data_par_size": data_par_size,
        "tensor_par_size": tensor_par_size,
    }

    #Check that every tensor-parallel rank's checkpoint file exists, matching the
    #"<checkpoint_filename>_rank_<N>.ckpt" naming save_checkpoint (training.py)
    #actually writes and get_model (model/utils.py) actually reads -- not a bare
    #"<checkpoint_filename>" file, which is never created.
    if resume_from_checkpoint:
        for rank in range(tensor_par_size):
            checkpoint_file = trainer_conf["checkpoint_path"]+"/"+trainer_conf["checkpoint_filename"]+"_rank_"+str(rank)+".ckpt"
            if not os.path.isfile(checkpoint_file):
                sys.exit(f"Checkpoint file does not exist: {checkpoint_file}")

# ---------------------------- OPTIMIZER -----------------------------------------
    #TODO: Add checking on each argument, e.g. > 0
    optimizer_conf = {
        "lr": float(conf['optimizer']['lr']),
        "betas": (float(conf['optimizer']['beta_1']), float(conf['optimizer']['beta_2'])),
        "weight_decay": float(conf['optimizer']['weight_decay']),
    }

# ---------------------------- SCHEDULER -----------------------------------------
    #TODO: Add checking on each argument, e.g. > 0
    scheduler_conf = {
        "warmup_epochs": conf['scheduler']['warmup_epochs'],
        "warmup_start_lr": float(conf['scheduler']['warmup_start_lr']),
        "eta_min": float(conf['scheduler']['eta_min']),
        "max_epochs": trainer_conf['max_epochs'],
    }

# ---------------------------- GRAD SCALER ---------------------------------------
    try:
        use_grad_scaler = conf["grad_scaler"]["use_grad_scaler"]
        grad_scaler_conf = {
            "use_grad_scaler": use_grad_scaler,
            "init_scale": conf["grad_scaler"]["init_scale"] if use_grad_scaler else None,
            "min_scale": conf["grad_scaler"]["min_scale"] if use_grad_scaler else None,
            "growth_interval": conf["grad_scaler"]["growth_interval"] if use_grad_scaler else None,
            }
    except KeyError:
        if dist.get_rank() == 0:
            print("Since no grad_scaler_conf was given in the config file, defaulting to not using a grad_scaler ")
        grad_scaler_conf = {"use_grad_scaler": False, "init_scale": None, "min_scale": None, "growth_interval": None}

# ---------------------------- MODEL ---------------------------------------------
    model_type = conf['model']['type']

    assert model_type in ['VIT', 'SAP', 'MAE', 'UNETR', 'DiffusionVIT'], "Model architecture type not supported. Choose a model architecture from the following choices: VIT, SAP, MAE, UNETR, DiffusionVIT"


    #Check Arguments that are required across all different architectures, the same transformer encoder is acrossed all.
    #TODO: Add checking on each argument, e.g. > 0
    try:
        embed_dim = conf['model']['embed_dim']
    except KeyError:
        sys.exit("embed_dim is required")

    try:
        depth = conf['model']['depth']
    except KeyError:
        sys.exit("depth is required")

    try:
        num_heads = conf['model']['num_heads']
    except KeyError:
        sys.exit("num_heads is required")

    try:
        mlp_ratio = conf['model']['mlp_ratio']
    except KeyError:
        sys.exit("mlp_ratio is required")

    try:
        drop_path = conf['model']['drop_path']
    except KeyError:
        sys.exit("drop_path is required")

    try:
        drop_rate = conf['model']['drop_rate']
    except KeyError:
        sys.exit("drop_rate is required")

    try:
        use_channel_aggregation = conf['model']['use_channel_aggregation']
    except KeyError:
        if dist.get_rank() == 0:
            print("use_channel_aggregation is not set, by default this is set to False")
        use_channel_aggregation = False

    try:
        loss_fn = conf["model"]["loss_fn"]
    except KeyError:
        if not load_balance_offline:
            if dist.get_rank() == 0:
                print("loss_fn is not set, by default this is set to the default loss for your model")
        if model_type == "MAE":
            loss_fn = "MSE"
        else:
            loss_fn = None
    
    #Set model specific arguments via a model_kwarg config
    kwargs = get_kwargs(model_type, conf)

    model_conf = {
        "type": model_type,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "drop_path": drop_path,
        "drop_rate": drop_rate,
        "use_channel_aggregation": use_channel_aggregation,
        "loss_fn": loss_fn,
        "kwargs": kwargs,
    }


# ---------------------------- TILING ------------------------------------------
    #TODO: Add checking on each argument, e.g. > 0
    try:
        do_tiling = conf["tiling"]["do_tiling"]
        tiling_conf = {
            "do_tiling": do_tiling,
            "div": conf["tiling"]["div"] if do_tiling else 1,
            "tile_overlap": conf["tiling"]["tile_overlap"] if do_tiling else 0,
        }
    except KeyError:
        if dist.get_rank() == 0:
            print("Since no tiling_conf was given in the config file, this is defaulting to be ran without tiling")
        tiling_conf = {"do_tiling": False, "div": 1, "tile_overlap": 0}
        
# ---------------------------- AP ------------------------------------------------
    #TODO: Add checking on each argument, e.g. > 0
    try:
        do_ap = conf['ap']['do_ap']
        ap_conf = {
            "do_ap": do_ap,
            "fixed_length": conf['ap']['fixed_length'] if do_ap else None,
            "separate_channels": conf['ap']['separate_channels'] if do_ap else False,
            "use_adaptive_pos_emb": conf['ap']['use_adaptive_pos_emb'] if do_ap else False,
        }
    except KeyError:
        if dist.get_rank() == 0:
            print("Since no ap_conf was given in the config file, this is defaulting to be ran with standard patching")
        ap_conf = {"do_ap": False, "fixed_length": None, "separate_channels": False, "use_adaptive_pos_emb": False}

    if ap_conf["do_ap"]:
        if ap_conf["separate_channels"]:
            assert not ap_conf["use_adaptive_pos_emb"], "Capability to use separate channels and adaptive pos_emb not implemented yet"
            
# ---------------------------- DATA ----------------------------------------------
    #TODO: Add checking on each argument, e.g. > 0

    dataset = conf['data']['dataset']
    assert dataset in ["imagenet", "catsdogs", "basic_ct"], "This training script only supports the following datasets: imagenet, catsdogs, basic_ct"

    #To remove the need for specifying img_size in the config can add check_data_size function. The issue with automating this process is that raw data files come in various forms that are not consistent. This requires special functionality for each different dataset. Additionaly, it can be expensive to read individual datafiles that are very large on the fly.
    #TODO: Put assert with sys.exit around num_channels since its required
    img_size = conf['data']['img_size']

    assert len(img_size) == 2 or len(img_size) == 3, "Img_size needs to be 2D or 3D"
    if len(img_size) == 2:
        twoD = True
    elif len(img_size) == 3:
        twoD = conf['data']['twoD']

    if not isinstance(tiling_conf["tile_overlap"], tuple):
        if twoD:
            tiling_conf["tile_overlap"] = (tiling_conf["tile_overlap"], tiling_conf["tile_overlap"])
        else:
            tiling_conf["tile_overlap"] = (tiling_conf["tile_overlap"], tiling_conf["tile_overlap"], tiling_conf["tile_overlap"])
    else:
        if twoD:
            assert len(tiling_conf["tile_overlap"]) == 2, "Tile overlap dimension doesn't match the dimensions of the data"
        else:
            assert len(tiling_conf["tile_overlap"]) == 3, "Tile overlap dimension doesn't match the dimensions of the data"

    #Require overlap values to be ints (not e.g. a YAML float like 0.0), since a
    #float would otherwise silently turn tile_size into floats downstream
    assert all(isinstance(v, int) for v in tiling_conf["tile_overlap"]), "tiling.tile_overlap must be an int (or tuple of ints) in the config, not a float"

    if twoD:
        tile_size = (img_size[0]//tiling_conf["div"]+tiling_conf["tile_overlap"][0], img_size[1]//tiling_conf["div"]+tiling_conf["tile_overlap"][1])
    else:
        tile_size = (img_size[0]//tiling_conf["div"]+tiling_conf["tile_overlap"][0], img_size[1]//tiling_conf["div"]+tiling_conf["tile_overlap"][1], img_size[2]//tiling_conf["div"]+tiling_conf["tile_overlap"][2])

    if tiling_conf["do_tiling"]:
        for i in range(len(tile_size)):
            assert img_size[0] // tiling_conf["div"], "The image cannot be evenly divided into tiles. This assertion can be commented out and ignored if this was intended, however be aware not all of the image will be used in training"
        
    patch_size = conf['data']['patch_size']
    #If doing standard patching, check if img_size/tile_size is divisible by patch_size
    if not ap_conf["do_ap"]:
        checkDims = 2 if twoD else 3
        for i in range(checkDims):
            assert tile_size[i] % patch_size == 0, "img_size/tile_size not divisible by patch_size which is required when doing standard patching"


    #num_channels required because we aren't requiring dict_in_variables to be specified
    #TODO: Put assert with sys.exit around num_channels since its required
    num_channels = conf['data']['num_channels']
    for i,k in enumerate(num_channels):
        if i == 0:
            num_chan = num_channels[k]
        else:
            if not use_channel_aggregation: 
                assert num_chan == num_channels[k], "If not using channel aggregation, num_channels across different datasets must be the same"
    #in_chans is the num_channels to be used acrossed all datasets 
    in_chans = num_chan
        
    #Create default dict_in_variables if it doesn't exist that assumes the channels are the same across different datasets
    try:
        dict_in_variables = conf['data']['dict_in_variables']
        #Check if number of variables is valid 
        for i,k in enumerate(conf['data']['dict_in_variables']):
            assert len(conf['data']['dict_in_variables'][k]) == num_channels[k], "dict_in_variables must have the same amount as the num_channels"
    except KeyError:
        if dist.get_rank() == 0:
            print("Using a default in_variables, which assumes the datasets have channels that are all arranged in the same order. If you want to track input channels for uses such as training multi-modal data in a flexible manner it is recommended to create your own dict_in_variables giving each channel appropriate unique labels")
        dict_in_variables = {}
        for i,k in enumerate(num_channels):
            in_variables_list = []
            for i in range(num_channels[k]):
                in_variables_list.append(str(i))
            dict_in_variables.update(in_variables_list)

    #Create default_vars from dict_in_variables
    for i,k in enumerate(conf['data']['dict_in_variables']):
        if i == 0:
            default_vars = conf['data']['dict_in_variables'][k]
        else:
            default_vars = list(set(default_vars + conf['data']['dict_in_variables'][k]))

    #If using adaptive patching check if fixed length is compatible with tile_size
    if ap_conf['do_ap']:
        checkDims = 2 if twoD else 3
        for i in range(checkDims):
            p2 = is_power_of_two(tile_size[i])
            assert p2, f"Tile Size in the {i} dimension must be a power of 2"

        if twoD:
            assert ap_conf["fixed_length"] % 3 == 1 % 3, "Quadtree fixed length needs to be 3n+1, where n is some integer"
        else:
            assert ap_conf["fixed_length"] % 7 == 1 % 7, "Octtree fixed length needs to be 7n+1, where n is some integer"

        #If model is UNETR check whether fixed_length is a sqr or cube root
        if model_type in ["UNETR", "SAP"]:
            if twoD:
                sqrt_len = math.sqrt(ap_conf["fixed_length"])
                assert sqrt_len.is_integer(), "Square root of fixed length needs to be a whole number"
                sqrt_len = int(sqrt_len)
            else:
                sqrt_len=int(np.rint(math.pow(conf["ap"]["fixed_length"],1/3)))
                assert np.abs(np.rint(math.pow(conf["ap"]["fixed_length"],1/3)) - math.pow(conf["ap"]["fixed_length"], 1/3)) < 0.0001, "cube root of fixed length needs to be a whole number"
        else:
            sqrt_len = None
    else:
        sqrt_len = None
            
        
    data_conf = {    
        "dataset": dataset,
        "img_size": img_size,
        "tile_size": tile_size,
        "patch_size": patch_size,
        "default_vars": default_vars,
        "twoD": twoD,
        "dict_root_dirs": conf['data']['dict_root_dirs'],
        "num_channels": num_channels,
        "dict_in_variables": dict_in_variables,
        "in_chans": in_chans,
    }

# ---------------------------- DATALOADER ----------------------------------------
    if model_conf["type"] in ["VIT", "SAP", "UNETR"]:
        return_label = True
    else:
        return_label = False

    if return_label:
        if data_conf['dataset'] in ['imagenet', 'catsdogs']:
            assert model_conf['type'] in ["VIT"], "This dataset can only be used for classification"

        elif data_conf['dataset'] in ["basic_ct"]:
            assert model_conf['type'] in ["SAP", "UNETR"], "This dataset can only be used for segmentation"

    dataloader_type = conf['dataloader']['type']
    assert dataloader_type in ["dataloader", "iterative_dataloader"], "dataloader type not valid"

    if dataloader_type == "dataloader":
        assert dataset in ["catsdogs"], "Only the catsdogs datset is supported with the standard torch dataloader, add your dataset to UCF_VIT/datasets"
        if data_conf['dataset'] == 'catsdogs':
            from UCF_VIT.datasets.catsdogs import CatsDogsDataset as dataset_module
            from UCF_VIT.datasets.catsdogs import CatsDogsCollate as collate_fn

    dataloader_conf = {
        "type": dataloader_type,
        "dict_start_idx": conf['dataloader']['dict_start_idx'] if dataloader_type == "iterative_dataloader" else None,
        "dict_end_idx": conf['dataloader']['dict_end_idx'] if dataloader_type == "iterative_dataloader" else None,
        "dict_buffer_sizes": conf['dataloader']['dict_buffer_sizes'] if dataloader_type == "iterative_dataloader" else None,
        "batch_size": conf['dataloader']['batch_size'],
        "num_workers": conf['dataloader']['num_workers'],
        "pin_memory": conf['dataloader']['pin_memory'],
        "dataset_module": dataset_module if dataloader_type == "dataloader" else None,
        "collate_fn": collate_fn if dataloader_type == "dataloader" else None,
        "return_label": return_label,

    }

    
# ---------------------------- DATASET SPECIFIC OPTIONS --------------------------
    #TODO: Move this to its own function
    dataset_options_conf = {
        "resize": conf['dataset_options']['resize'] if dataset == "imagenet" else None, 
    }


    return { 
        "trainer": trainer_conf, 
        "parallelism": parallelism_conf, 
        "optimizer": optimizer_conf, 
        "scheduler": scheduler_conf, 
        "grad_scaler": grad_scaler_conf,
        "model": model_conf, 
        "tiling": tiling_conf, 
        "ap": ap_conf, 
        "data": data_conf, 
        "dataloader": dataloader_conf, 
        "dataset_options": dataset_options_conf,
    } 

def parse_pretrained_config(args, conf):
    """Load and validate the configuration of a pretrained model to fine-tune from.

    If `conf` indicates a pretrained model should be used (and training is not
    resuming from a checkpoint), loads the pretrained model's own config YAML file
    (`args.pretrained_config`) and checks that its architecture and data settings
    (embed_dim, depth, num_heads, mlp_ratio, tensor_par_size, image dimensionality,
    tile size, patch size, channel aggregation, input channels/variables) are
    compatible with the model described by `conf`.

    Args:
        args: Parsed command-line arguments; must have a `pretrained_config`
            attribute giving the path to the pretrained model's config YAML file.
        conf: Validated training configuration dict for the model to be trained, as
            returned by `parse_config`.

    Returns:
        A dict with keys "model_type", "default_vars", and "kwargs" describing the
        pretrained model, or an empty dict if no pretrained model is being used.
    """

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']

    #If not resuming from checkpoint, check if starting from a pretrained model
    if not resume_from_checkpoint:
        try:
            use_pretrained_model = conf["trainer"]["use_pretrained_model"]

        except KeyError:
            if dist.get_rank() == 0:
                print("Since use_pretrained_model wasn't given, defaulting to not using a pretrained model")
            use_pretrained_model = False

    else:
        use_pretrained_model = False

    if use_pretrained_model:
        assert args.pretrained_config != "", "If training using a pretrained model, you must pass in the config file for the pretrained model as an argument"
        
        try:
            pretrained_checkpoint_filename = conf["trainer"]["pretrained_checkpoint_filename"]

        except KeyError:
            sys.exit("pretrained_checkpoint_filename needs to be set in the trainer section")

        if args.pretrained_config != "":
            with open(args.pretrained_config,'r') as f:
                pretrained_conf = yaml.load(f,Loader=yaml.FullLoader)
        
# ---------------------------- MODEL ---------------------------------------------

        #Get and check model arguments
        embed_dim=pretrained_conf["model"]["embed_dim"]
        assert embed_dim == conf["model"]["embed_dim"], "Pretrained embed_dim does not match model to be trained"

        depth=pretrained_conf["model"]["depth"]
        assert depth == conf["model"]["depth"], "Pretrained depth does not match model to be trained"

        num_heads=pretrained_conf["model"]["num_heads"]
        assert num_heads == conf["model"]["num_heads"], "Pretrained num_heads does not match model to be trained"

        mlp_ratio=pretrained_conf["model"]["mlp_ratio"]
        assert mlp_ratio == conf["model"]["mlp_ratio"], "Pretrained num_heads does not match model to be trained"

        #TODO: Checking for drop_path_rate and drop_rate needed?
        drop_path_rate=pretrained_conf["model"]["drop_path"]
        drop_rate=pretrained_conf["model"]["drop_rate"]

        tensor_par_size = pretrained_conf["parallelism"]["tensor_par_size"]
        assert tensor_par_size == conf["parallelism"]["tensor_par_size"], "Tensor_par_size of the pre-trained model needs to match the tensor_par_size of the model to be trained"

        #TODO: If tensor_parallel check if all checkpoint files exist
        #Check if all checkpoint file exists, for given parallelism setup
        checkpointExists = os.path.isfile(os.path.join(pretrained_conf["trainer"]["checkpoint_path"],conf["trainer"]["pretrained_checkpoint_filename"]))
        if not checkpointExists:
            sys.exit("Checkpoint file does not exist")

        model_type = pretrained_conf["model"]["type"]
        kwargs = get_kwargs(model_type, conf)

# ---------------------------- TILING ------------------------------------------
        pretrained_do_tiling = pretrained_conf["tiling"]["do_tiling"]
        if pretrained_do_tiling:
            pretrained_div = pretrained_conf["tiling"]["div"]
            pretrained_tile_overlap = pretrained_conf["tiling"]["tile_overlap"]
        else:
            pretrained_div = 1
            pretrained_tile_overlap = 0

# ---------------------------- AP ------------------------------------------------
        pretrained_do_ap = pretrained_conf["ap"]["do_ap"]
        if pretrained_do_ap:
            assert conf["ap"]["do_ap"], "If pretrained model was trained with adaptive patching, the downstream model needs to use adaptive patching"
        else:
            assert not conf["ap"]["do_ap"], "If pretrained model was trained with standard patching, this model needs to use standard patching"
        ###fixed_length=conf["ap"]["fixed_length"],
        ###use_adaptive_pos_emb=conf["ap"]["use_adaptive_pos_emb"],

# ---------------------------- DATA ----------------------------------------------
        #Get and check data arguments
        pretrained_img_size = pretrained_conf["data"]["img_size"]
        if len(pretrained_img_size) == 2:
            twoD = True
        elif len(pretrained_img_size) == 3:
            twoD = pretrained_conf['data']['twoD']
        assert twoD == conf["data"]["twoD"], "Pretrained model and this model do not have the same dimension data"

        if not isinstance(pretrained_tile_overlap, tuple):
            if twoD:
                pretrained_tile_overlap = (pretrained_tile_overlap, pretrained_tile_overlap)
            else:
                pretrained_tile_overlap = (pretrained_tile_overlap, pretrained_tile_overlap, pretrained_tile_overlap)

        #Require overlap values to be ints (not e.g. a YAML float like 0.0), since a
        #float would otherwise silently turn pretrained_tile_size into floats downstream
        assert all(isinstance(v, int) for v in pretrained_tile_overlap), "tiling.tile_overlap must be an int (or tuple of ints) in the pretrained model's config, not a float"

        if twoD:
            pretrained_tile_size = (pretrained_img_size[0]//pretrained_div+pretrained_tile_overlap[0], pretrained_img_size[1]//pretrained_div+pretrained_tile_overlap[1])
        else:
            pretrained_tile_size = (pretrained_img_size[0]//pretrained_div+pretrained_tile_overlap[0], pretrained_img_size[1]//pretrained_div+pretrained_tile_overlap[1], pretrained_img_size[2]//pretrained_div+pretrained_tile_overlap[2])

        for i in range(len(pretrained_tile_size)):
            assert pretrained_tile_size[i] == conf["data"]["tile_size"][i], "Image/Tile size does not match between pretrained model and this model"

        pretrained_patch_size=pretrained_conf["data"]["patch_size"]
        assert pretrained_patch_size == conf["data"]["patch_size"], "Patch size does not match between pretrained model and this model"

        try:
            use_channel_aggregation = pretrained_conf['model']['use_channel_aggregation']
        except KeyError:
            if dist.get_rank() == 0:
                print("use_channel_aggregation is not set, by default this is set to False")
            use_channel_aggregation = False

        assert use_channel_aggregation == conf['model']['use_channel_aggregation'], "Use_channel_aggregation needs to match between pretrained model and this model"

        ##############################
        #num_channels required because we aren't requiring dict_in_variables to be specified
        num_channels = pretrained_conf['data']['num_channels']
        for i,k in enumerate(num_channels):
            if i == 0:
                num_chan = num_channels[k]
            else:
                if not use_channel_aggregation: 
                    assert num_chan == num_channels[k], "If not using channel aggregation, num_channels across different datasets must be the same"
        #in_chans is the num_channels to be used acrossed all datasets 
        in_chans = num_chan

        assert in_chans == conf['data']['in_chans'], "Number of input channels for the pre-trained model doesn't match the number of inputs for this model" 

        #Create default dict_in_variables if it doesn't exist that assumes the channels are the same across different datasets
        try:
            dict_in_variables = pretrained_conf['data']['dict_in_variables']
            #Check if number of variables is valid
            for i,k in enumerate(pretrained_conf['data']['dict_in_variables']):
                assert len(pretrained_conf['data']['dict_in_variables'][k]) == num_channels[k], "dict_in_variables must have the same amount as the num_channels"
        except KeyError:
            if dist.get_rank() == 0:
                print("Using a default in_variables, which assumes the datasets have channels that are all arranged in the same order. If you want to track input channels for uses such as training multi-modal data in a flexible manner it is recommended to create your own dict_in_variables giving each channel appropriate unique labels")
            dict_in_variables = {}
            for i,k in enumerate(num_channels):
                in_variables_list = []
                for i in range(num_channels[k]):
                    in_variables_list.append(str(i))
                dict_in_variables.update(in_variables_list)

        #Create default_vars from dict_in_variables
        for i,k in enumerate(pretrained_conf['data']['dict_in_variables']):
            if i == 0:
                default_vars = pretrained_conf['data']['dict_in_variables'][k]
            else:
                default_vars = list(set(default_vars + pretrained_conf['data']['dict_in_variables'][k]))

        #Check if dict_in_variables from this model are in the default_vars list for the pre-trained model
        if use_channel_aggregation: 
            for i,k in enumerate(conf['data']['dict_in_variables']):
                for j in range(len(conf['data']['dict_in_variables'][k])):
                    assert conf['data']['dict_in_variables'][k][j] in default_vars, f"dict_in_variable {conf['data']['dict_in_variables'][k][j]} for this model is not used in the pretrained model"

        p_conf = {
            "model_type": model_type,
            "default_vars": default_vars,
            "kwargs": kwargs
        }
    else:
        p_conf = {}

    return p_conf

