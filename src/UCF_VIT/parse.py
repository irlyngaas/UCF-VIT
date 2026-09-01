import yaml
import os
import sys
import math
import numpy as np
import torch.distributed as dist
from UCF_VIT.utils.misc import detect_img_size, detect_num_channels, is_power_of_two

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
            sys.exit("num_time_steps is required")
        kwargs.update({"num_time_steps": num_time_steps})

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


def _resolve_dataset_splits(dict_root_dirs, dict_start_idx, dict_end_idx,
                             dict_val_root_dirs, dict_val_start_idx, dict_val_end_idx,
                             dict_test_root_dirs, dict_test_start_idx, dict_test_end_idx,
                             val_split_ratio, test_split_ratio):
    """Resolves per-dataset-key train/val/test root dirs and start/end idx ratios.

    For each dataset key in `dict_root_dirs`: if a separate `dict_val_root_dirs`/
    `dict_test_root_dirs` entry exists for that key, val/test use that directory
    with their own start/end idx (defaulting to the full [0.0, 1.0) range) and
    train's own [dict_start_idx, dict_end_idx) window for that key is left
    untouched. Otherwise, val/test reuse `dict_root_dirs`'s *same* directory for
    that key, auto-splitting three contiguous, non-overlapping sub-ranges out of
    train's own already-configured [dict_start_idx, dict_end_idx) window (not
    assumed to be [0, 1) -- respects any existing narrowing): train keeps the
    first `1 - val_split_ratio - test_split_ratio` portion, val the next
    `val_split_ratio` portion, test the last `test_split_ratio` portion -- and
    train's own start/end idx for that key is narrowed in place to reflect its
    reduced share.

    A ratio of 0.0 (with no separate root given) means that split has no data
    for that key at all -- it's simply omitted from the returned val/test dicts,
    rather than given a degenerate zero-width range.

    Args:
        dict_root_dirs: Training dataset-key -> root directory dict.
        dict_start_idx: Training dataset-key -> start ratio dict (0.0-1.0).
        dict_end_idx: Training dataset-key -> end ratio dict (0.0-1.0).
        dict_val_root_dirs: Optional dataset-key -> val root directory dict --
            keys present here get a real, separate val split, not auto-split.
        dict_val_start_idx: Optional dataset-key -> val start ratio dict, only
            meaningful for keys in `dict_val_root_dirs` (defaults to 0.0).
        dict_val_end_idx: Optional dataset-key -> val end ratio dict, only
            meaningful for keys in `dict_val_root_dirs` (defaults to 1.0).
        dict_test_root_dirs: Same as `dict_val_root_dirs`, for test.
        dict_test_start_idx: Same as `dict_val_start_idx`, for test.
        dict_test_end_idx: Same as `dict_val_end_idx`, for test.
        val_split_ratio: Fraction of each auto-split key's train window to
            reserve for val (e.g. 0.1). Ignored for keys with a separate
            `dict_val_root_dirs` entry.
        test_split_ratio: Same as `val_split_ratio`, for test.

    Returns:
        An 8-tuple `(train_start_idx, train_end_idx, val_root_dirs,
        val_start_idx, val_end_idx, test_root_dirs, test_start_idx,
        test_end_idx)` of dataset-key -> value dicts.
    """
    train_start_idx = {}
    train_end_idx = {}
    val_root_dirs = {}
    val_start_idx = {}
    val_end_idx = {}
    test_root_dirs = {}
    test_start_idx = {}
    test_end_idx = {}

    for k in dict_root_dirs:
        this_train_start = dict_start_idx.get(k, 0.0)
        this_train_end = dict_end_idx.get(k, 1.0)
        window = this_train_end - this_train_start

        has_val_root = k in dict_val_root_dirs
        has_test_root = k in dict_test_root_dirs

        if has_val_root:
            val_root_dirs[k] = dict_val_root_dirs[k]
            val_start_idx[k] = dict_val_start_idx.get(k, 0.0)
            val_end_idx[k] = dict_val_end_idx.get(k, 1.0)
        if has_test_root:
            test_root_dirs[k] = dict_test_root_dirs[k]
            test_start_idx[k] = dict_test_start_idx.get(k, 0.0)
            test_end_idx[k] = dict_test_end_idx.get(k, 1.0)

        reserve_val = 0.0 if has_val_root else val_split_ratio
        reserve_test = 0.0 if has_test_root else test_split_ratio
        assert reserve_val >= 0.0 and reserve_test >= 0.0 and (reserve_val + reserve_test) < 1.0, (
            f"val_split_ratio + test_split_ratio must be < 1.0 for dataset '{k}' "
            f"(got {reserve_val} + {reserve_test})"
        )

        new_train_end = this_train_start + window * (1.0 - reserve_val - reserve_test)

        if not has_val_root and reserve_val > 0.0:
            val_root_dirs[k] = dict_root_dirs[k]
            val_start_idx[k] = new_train_end
            val_end_idx[k] = new_train_end + window * reserve_val
        if not has_test_root and reserve_test > 0.0:
            test_root_dirs[k] = dict_root_dirs[k]
            test_start_idx[k] = new_train_end + window * reserve_val
            test_end_idx[k] = this_train_end

        train_start_idx[k] = this_train_start
        train_end_idx[k] = new_train_end

    return (train_start_idx, train_end_idx, val_root_dirs, val_start_idx, val_end_idx,
            test_root_dirs, test_start_idx, test_end_idx)


def get_split_conf(conf, split):
    """Returns a shallow-copied `conf` pointed at the resolved val or test split.

    Swaps `data["dict_root_dirs"]` and `dataloader["dict_start_idx"]`/
    `["dict_end_idx"]` for the resolved `dict_val_root_dirs`/`dict_test_root_dirs`
    (etc., see `parse_config`'s own DATA section, which calls
    `_resolve_dataset_splits`) -- everything else in `conf` (num_channels,
    dict_in_variables, img_size, tile_size, model kwargs, trainer settings,
    etc.) passes through unchanged, since those describe the dataset/model, not
    which split of it this is. The result feeds into
    `calculate_load_balancing_on_the_fly`/`NativePytorchDataModule` exactly like
    `conf` itself, completely unchanged -- val/test scripts just call those with
    this instead of `conf`.

    Args:
        conf: Parsed training configuration dict, as returned by `parse_config`.
        split: "val" or "test".

    Returns:
        A new conf dict (shallow copy of `conf`, `conf["data"]`, and
        `conf["dataloader"]`) with `dict_root_dirs`/`dict_start_idx`/
        `dict_end_idx` pointed at the requested split.

    Raises:
        AssertionError: If `split` isn't "val"/"test", or if that split
            resolved to no data at all for every dataset key (no separate
            root given and its split ratio is 0.0 for every key).
    """
    assert split in ("val", "test"), "split must be 'val' or 'test'"
    root_dirs = conf["data"][f"dict_{split}_root_dirs"]
    start_idx = conf["dataloader"][f"dict_{split}_start_idx"]
    end_idx = conf["dataloader"][f"dict_{split}_end_idx"]
    assert root_dirs, (
        f"No {split} data available -- dict_{split}_root_dirs resolved empty. "
        f"Either set data.dict_{split}_root_dirs in the config, or make sure "
        f"dataloader.{split}_split_ratio is > 0 for at least one dataset key."
    )

    split_conf = dict(conf)
    split_conf["data"] = dict(conf["data"])
    split_conf["data"]["dict_root_dirs"] = root_dirs
    split_conf["dataloader"] = dict(conf["dataloader"])
    split_conf["dataloader"]["dict_start_idx"] = start_idx
    split_conf["dataloader"]["dict_end_idx"] = end_idx
    return split_conf


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

    # resume_from_checkpoint continues an existing run from its own checkpoint
    # (optimizer/scheduler state, loss history, epoch included); use_pretrained_model
    # starts a new run whose encoder is initialized from a *different* model's
    # weights. Previously, setting both True silently dropped use_pretrained_model
    # with no warning at all (see the "use_pretrained_model" entry in trainer_conf
    # below, and parse_pretrained_config's own identical silent override) -- now
    # rejected explicitly instead. .get (not a bare index) because
    # use_pretrained_model is allowed to be omitted entirely when
    # resume_from_checkpoint:True (trainer_conf's own ternary below never evaluates
    # it in that case either).
    if resume_from_checkpoint and conf['trainer'].get('use_pretrained_model', False):
        sys.exit("trainer.resume_from_checkpoint and trainer.use_pretrained_model cannot both be True -- resume_from_checkpoint continues an existing training run from its own checkpoint, use_pretrained_model starts a new run initialized from a different model's weights. Set only one of them to True.")

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
        # Never copied through before -- parse_pretrained_config's own read
        # of this (both its checkpoint-existence check and the actual
        # filename get_model's pretrained branch loads) always KeyError'd,
        # unconditionally, whenever use_pretrained_model:True.
        "pretrained_checkpoint_filename": conf['trainer'].get('pretrained_checkpoint_filename', ""),
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

    #img_size is always the real/native size of the data. If omitted,
    #auto-detect it by reading one real file (see detect_img_size's own
    #docstring). Resizing to a *different* size than native is a separate,
    #optional step -- see resize_conf below -- not what img_size means.
    try:
        img_size = conf['data']['img_size']
    except KeyError:
        if dist.get_rank() == 0:
            print("img_size is not set, auto-detecting from the real data files under dict_root_dirs...")
        img_size = detect_img_size(dataset, conf['data']['dict_root_dirs'])
        if dist.get_rank() == 0:
            print(f"Detected img_size: {img_size}")

    assert len(img_size) == 2 or len(img_size) == 3, "Img_size needs to be 2D or 3D"
    if len(img_size) == 2:
        twoD = True
    elif len(img_size) == 3:
        twoD = conf['data']['twoD']

    #resize is an optional, separate step -- for imagenet/catsdogs, resizes
    #the real data from its native img_size to a different target size
    #before training. Not supported for basic_ct (no resize step exists in
    #its read path -- see dataset.py). Computed here (rather than down with
    #the rest of dataset_options_conf) because tile_size below must be
    #computed from whatever size the data actually is once resize (if any)
    #has been applied, not from img_size directly.
    resize_conf = conf.get('dataset_options', {}).get('resize', {}) or {}
    assert "basic_ct" not in resize_conf, "resize is not supported for basic_ct -- it has no resize step (dataset.py reads NIfTI volumes at native resolution). Remove it from dataset_options.resize."
    effective_size = resize_conf.get(dataset, img_size)

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

    if len(img_size) == 2:
        # Genuinely 2D data (imagenet/catsdogs): a 2-tuple tile_size is both
        # correct and, via TileDataIter's `len(self.tile_size) == 3` check,
        # the signal that tells it there's no z-axis to slice at all.
        tile_size = (effective_size[0]//tiling_conf["div"]+tiling_conf["tile_overlap"][0], effective_size[1]//tiling_conf["div"]+tiling_conf["tile_overlap"][1])
    elif twoD:
        # 3D data (basic_ct) sliced into 2D z-planes: tile_size must still be
        # a 3-tuple -- collapsing it to 2D here (as if it were genuinely 2D
        # data) previously made TileDataIter's `len(self.tile_size) == 3`
        # dispatch take the wrong branch, silently keeping the full,
        # untouched z-axis on every tile and producing a 5D batch by the
        # time it reached PatchEmbed. The z entry itself isn't tiled (every
        # z-index is walked one at a time in TileDataIter's twoD branch), so
        # it's the raw, undivided depth.
        tile_size = (effective_size[0]//tiling_conf["div"]+tiling_conf["tile_overlap"][0], effective_size[1]//tiling_conf["div"]+tiling_conf["tile_overlap"][1], effective_size[2])
    else:
        tile_size = (effective_size[0]//tiling_conf["div"]+tiling_conf["tile_overlap"][0], effective_size[1]//tiling_conf["div"]+tiling_conf["tile_overlap"][1], effective_size[2]//tiling_conf["div"]+tiling_conf["tile_overlap"][2])

    if tiling_conf["do_tiling"]:
        for i in range(len(tile_size)):
            assert effective_size[0] // tiling_conf["div"], "The image cannot be evenly divided into tiles. This assertion can be commented out and ignored if this was intended, however be aware not all of the image will be used in training"
        
    #patch_size is unused when do_ap:True (interp_size takes over its role
    #entirely, see below), so it's only required in the config in that case.
    if not ap_conf["do_ap"]:
        patch_size = conf['data']['patch_size']
        #If doing standard patching, check if img_size/tile_size is divisible by patch_size
        checkDims = 2 if twoD else 3
        for i in range(checkDims):
            assert tile_size[i] % patch_size == 0, "img_size/tile_size not divisible by patch_size which is required when doing standard patching"
    else:
        patch_size = conf['data'].get('patch_size')

    #interp_size replaces patch_size as the size every adaptive (quadtree/octree)
    #leaf patch is interpolated to, and the size every dependent model-layer
    #calculation is based on -- required whenever do_ap:True so patch_size is
    #never silently reused for that purpose; unused/absent otherwise.
    if ap_conf["do_ap"]:
        try:
            interp_size = conf['ap']['interp_size']
        except KeyError:
            sys.exit("ap.interp_size is required when adaptive patching (ap.do_ap) is turned on")
    else:
        interp_size = None


    #num_channels required because we aren't requiring dict_in_variables to be specified.
    #If omitted, auto-detect it by reading one real file per dataset key
    #(see detect_num_channels's own docstring for per-dataset-type behavior
    #and why basic_ct raises rather than guessing for multi-channel files).
    try:
        num_channels = conf['data']['num_channels']
    except KeyError:
        if dist.get_rank() == 0:
            print("num_channels is not set, auto-detecting from the real data files under dict_root_dirs...")
        num_channels = detect_num_channels(dataset, conf['data']['dict_root_dirs'])
        if dist.get_rank() == 0:
            print(f"Detected num_channels: {num_channels}")

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
            
        
    # Resolves per-dataset-key train/val/test root dirs and start/end idx ratios --
    # used for both dataloader.type values (iterative_dataloader and the map-style
    # dataloader/catsdogs path both slice by dataset key and start/end ratio, just
    # via different mechanisms downstream -- FileReader for the former, a plain
    # list slice for the latter). See _resolve_dataset_splits's own docstring for
    # the full train/val/test semantics; get_split_conf is what val.py/test.py use
    # to consume the val_root_dirs/test_root_dirs results below.
    (resolved_train_start_idx, resolved_train_end_idx,
     resolved_val_root_dirs, resolved_val_start_idx, resolved_val_end_idx,
     resolved_test_root_dirs, resolved_test_start_idx, resolved_test_end_idx) = _resolve_dataset_splits(
        dict_root_dirs=conf['data']['dict_root_dirs'],
        dict_start_idx=conf['dataloader'].get('dict_start_idx', {}) or {},
        dict_end_idx=conf['dataloader'].get('dict_end_idx', {}) or {},
        dict_val_root_dirs=conf['data'].get('dict_val_root_dirs', {}) or {},
        dict_val_start_idx=conf['dataloader'].get('dict_val_start_idx', {}) or {},
        dict_val_end_idx=conf['dataloader'].get('dict_val_end_idx', {}) or {},
        dict_test_root_dirs=conf['data'].get('dict_test_root_dirs', {}) or {},
        dict_test_start_idx=conf['dataloader'].get('dict_test_start_idx', {}) or {},
        dict_test_end_idx=conf['dataloader'].get('dict_test_end_idx', {}) or {},
        val_split_ratio=conf['dataloader'].get('val_split_ratio', 0.1),
        test_split_ratio=conf['dataloader'].get('test_split_ratio', 0.1),
    )

    data_conf = {
        "dataset": dataset,
        "img_size": img_size,
        "tile_size": tile_size,
        "patch_size": patch_size,
        "interp_size": interp_size,
        "default_vars": default_vars,
        "twoD": twoD,
        "dict_root_dirs": conf['data']['dict_root_dirs'],
        "dict_val_root_dirs": resolved_val_root_dirs,
        "dict_test_root_dirs": resolved_test_root_dirs,
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
        # dict_start_idx/dict_end_idx are train's own (possibly narrowed by
        # _resolve_dataset_splits's auto-split above) start/end idx -- populated
        # for both dataloader.type values now (previously None for "dataloader",
        # since no slicing existed on that path at all; train.py's own catsdogs
        # branch now applies these via slice_file_list, same as
        # iterative_dataloader's FileReader does).
        "dict_start_idx": resolved_train_start_idx,
        "dict_end_idx": resolved_train_end_idx,
        "dict_val_start_idx": resolved_val_start_idx,
        "dict_val_end_idx": resolved_val_end_idx,
        "dict_test_start_idx": resolved_test_start_idx,
        "dict_test_end_idx": resolved_test_end_idx,
        "dict_buffer_sizes": conf['dataloader']['dict_buffer_sizes'] if dataloader_type == "iterative_dataloader" else None,
        "batch_size": conf['dataloader']['batch_size'],
        "num_workers": conf['dataloader']['num_workers'],
        "pin_memory": conf['dataloader']['pin_memory'],
        "dataset_module": dataset_module if dataloader_type == "dataloader" else None,
        "collate_fn": collate_fn if dataloader_type == "dataloader" else None,
        "return_label": return_label,
        # Optional -- None (the default, if omitted from the config) leaves DataLoader's
        # own default multiprocessing_context in place (fork on Linux). See
        # NativePytorchDataModule's multiprocessing_context docstring entry for why a
        # config would ever set this to "spawn".
        "multiprocessing_context": conf['dataloader'].get('multiprocessing_context'),
        # Optional, default False. When a dataset key has fewer files than the
        # number of DDP ranks/workers assigned to it, training normally fails
        # loudly (calculate_load_balancing_on_the_fly's and FileReader's own
        # asserts) rather than silently letting some ranks train on no data at
        # all. Setting this True instead lets every rank/worker get at least
        # one file, reusing (duplicating) files across ranks/workers as needed
        # -- with a printed warning quantifying how much reuse is happening.
        # Not just a small/debug-dataset concern: at the node counts this repo
        # targets, data_par_size can exceed a real (not toy) dataset's file
        # count too. See calculate_load_balancing_on_the_fly's and
        # FileReader.__iter__'s own comments for the mechanism.
        "allow_file_reuse": conf['dataloader'].get('allow_file_reuse', False),
        # Imagenet only -- seeds the shuffle bucket_file_list applies (to the
        # already train/val/test-sliced image list) before dividing it into
        # per-DDP-rank-group buckets. Without this, bucketing is a contiguous
        # split of a class-sorted list, so each bucket (and therefore each
        # rank) only ever sees a narrow range of classes every epoch -- a real
        # concern for data-parallel SGD (class-homogeneous local batches skew
        # BatchNorm statistics and correlate gradients within a rank's own
        # step sequence). Defaults to a fixed seed (not None/disabled) since
        # shuffling is a strict improvement with no reproducibility cost --
        # same seed always gives the same shuffle, independent of
        # data_par_size/process restarts, same as everything else about the
        # split. Set to `null` in the config to opt out and keep the original
        # contiguous ordering. See NativePytorchDataModule's/bucket_file_list's
        # own comments for the full rationale.
        "bucket_shuffle_seed": conf['dataloader'].get('bucket_shuffle_seed', 42),

    }

    
# ---------------------------- DATASET SPECIFIC OPTIONS --------------------------
    #TODO: Move this to its own function
    dataset_options_conf = {
        "resize": resize_conf,
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
        # Real filename save_checkpoint (training.py) actually writes is
        # "<pretrained_checkpoint_filename>_rank_<N>.ckpt", not a bare
        # "<pretrained_checkpoint_filename>" file -- the previous check here
        # looked for a file that's never written by any real training run,
        # so use_pretrained_model:True always failed with "Checkpoint file
        # does not exist" for real (confirmed by a real Frontier run, job
        # 5348717). Checking rank 0's file specifically (not per-tensor-
        # parallel-rank, per the TODO above) mirrors run_training_smoke.py's
        # own rank0_checkpoint_exists.
        checkpointExists = os.path.isfile(os.path.join(pretrained_conf["trainer"]["checkpoint_path"],conf["trainer"]["pretrained_checkpoint_filename"]+"_rank_0.ckpt"))
        if not checkpointExists:
            sys.exit("Checkpoint file does not exist")

        model_type = pretrained_conf["model"]["type"]
        # pretrained_conf, not conf: kwargs here are for constructing
        # pretrained_model at its own true architecture (model/utils.py's
        # get_model pretrained branch), so its checkpoint loads without a
        # shape mismatch -- reading conf's fields would be wrong whenever
        # the downstream model type differs from the pretrained one (e.g.
        # pretrained MAE -> downstream UNETR: conf["model"] wouldn't even
        # have MAE's mask_ratio/decoder_* keys at all), and subtly wrong
        # even when they match (the pretrained checkpoint's own decoder
        # shape must match for the initial strict load_state_dict to
        # succeed, even though only the encoder is kept afterward).
        kwargs = get_kwargs(model_type, pretrained_conf)

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
        # The pretrained model's own values (not conf's) -- needed so
        # get_model's pretrained branch can build pretrained_model at
        # exactly its own original architecture, or its checkpoint (saved
        # with these settings) won't load into it without a shape mismatch.
        pretrained_fixed_length = pretrained_conf['ap']['fixed_length'] if pretrained_do_ap else None
        pretrained_use_adaptive_pos_emb = pretrained_conf['ap']['use_adaptive_pos_emb'] if pretrained_do_ap else False

# ---------------------------- DATA ----------------------------------------------
        #Get and check data arguments. Same optional/auto-detect fallback
        #as parse_config's own img_size handling.
        try:
            pretrained_img_size = pretrained_conf["data"]["img_size"]
        except KeyError:
            if dist.get_rank() == 0:
                print("img_size is not set in the pretrained config, auto-detecting from the real data files under dict_root_dirs...")
            pretrained_img_size = detect_img_size(pretrained_conf["data"]["dataset"], pretrained_conf["data"]["dict_root_dirs"])
            if dist.get_rank() == 0:
                print(f"Detected img_size: {pretrained_img_size}")

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

        # Deliberately NOT asserted equal to conf's own img_size/tile_size/
        # patch_size/interp_size (unlike twoD/do_ap/in_chans/
        # use_channel_aggregation below, which are still required to match):
        # the whole point of pos_embed interpolation (see model/utils.py's
        # get_model pretrained branch and _transplant_pos_embed) is to let
        # the pretrained and new models differ in resolution. pretrained_*
        # below are the pretrained model's own true values, used to build it
        # at its own original size so its checkpoint loads without a shape
        # mismatch, then its encoder gets resized into the new model's shape.
        if twoD:
            pretrained_tile_size = (pretrained_img_size[0]//pretrained_div+pretrained_tile_overlap[0], pretrained_img_size[1]//pretrained_div+pretrained_tile_overlap[1])
        else:
            pretrained_tile_size = (pretrained_img_size[0]//pretrained_div+pretrained_tile_overlap[0], pretrained_img_size[1]//pretrained_div+pretrained_tile_overlap[1], pretrained_img_size[2]//pretrained_div+pretrained_tile_overlap[2])

        #patch_size is unused (and not required in the config) when do_ap:True.
        pretrained_patch_size = pretrained_conf["data"].get("patch_size") if pretrained_do_ap else pretrained_conf["data"]["patch_size"]

        if pretrained_do_ap:
            try:
                pretrained_interp_size = pretrained_conf["ap"]["interp_size"]
            except KeyError:
                sys.exit("ap.interp_size is required in the pretrained config when adaptive patching (ap.do_ap) is turned on")
        else:
            pretrained_interp_size = None

        try:
            use_channel_aggregation = pretrained_conf['model']['use_channel_aggregation']
        except KeyError:
            if dist.get_rank() == 0:
                print("use_channel_aggregation is not set, by default this is set to False")
            use_channel_aggregation = False

        assert use_channel_aggregation == conf['model']['use_channel_aggregation'], "Use_channel_aggregation needs to match between pretrained model and this model"

        ##############################
        #num_channels required because we aren't requiring dict_in_variables to be specified.
        #Same optional/auto-detect fallback as parse_config's own num_channels handling.
        try:
            num_channels = pretrained_conf['data']['num_channels']
        except KeyError:
            if dist.get_rank() == 0:
                print("num_channels is not set in the pretrained config, auto-detecting from the real data files under dict_root_dirs...")
            num_channels = detect_num_channels(pretrained_conf['data']['dataset'], pretrained_conf['data']['dict_root_dirs'])
            if dist.get_rank() == 0:
                print(f"Detected num_channels: {num_channels}")

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
            "kwargs": kwargs,
            # The pretrained model's own original architecture-determining
            # values -- see the comment above pretrained_tile_size for why
            # these are deliberately not required to match conf's own.
            "img_size": pretrained_img_size,
            "tile_size": pretrained_tile_size,
            "twoD": twoD,
            "patch_size": pretrained_patch_size,
            "interp_size": pretrained_interp_size,
            "fixed_length": pretrained_fixed_length,
            "use_adaptive_pos_emb": pretrained_use_adaptive_pos_emb,
            # The pretrained model's own checkpoint directory (already read
            # above for the existence check) -- get_model's pretrained
            # branch needs this to actually find the checkpoint file; there
            # is no "pretrained_model" section anywhere in a real config for
            # it to otherwise come from.
            "checkpoint_path": pretrained_conf["trainer"]["checkpoint_path"],
        }
    else:
        p_conf = {}

    return p_conf

