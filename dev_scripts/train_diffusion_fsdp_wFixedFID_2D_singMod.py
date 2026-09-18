import glob
import os
import sys
import copy
import random
from contextlib import contextmanager
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.utils import save_image
import time
import yaml
from einops import rearrange
from torch.nn import Sequential
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
import functools
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from timm.layers import use_fused_attn

from UCF_VIT.fsdp.arch import DiffusionVIT
from UCF_VIT.fsdp.building_blocks import Block
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, unpatchify, init_par_groups, calculate_load_balancing_on_the_fly
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler#, sample_images#,save_intermediate_data

# get all plottings
from plotting_generations import *
### training
def training_step(data, variables, t, e, net: DiffusionVIT, patch_size, twoD, loss_fn):


    output = net.forward(data, t, variables)
    output = unpatchify(output, data, patch_size, twoD)
    criterion = nn.MSELoss()
    loss = criterion(output,e)

    return loss


def require_finite_loss(loss, epoch, batch, device):
    """Return False on every rank if any rank produces a NaN/Inf loss."""
    all_finite = torch.isfinite(loss.detach()).to(device=device, dtype=torch.int32)
    dist.all_reduce(all_finite, op=dist.ReduceOp.MIN)
    if not all_finite.item():
        if dist.get_rank() == 0:
            print(
                f"Non-finite loss detected at epoch={epoch}, batch={batch}",
                flush=True,
            )
        return False
    return True


def clip_and_require_finite_gradients(model, max_grad_norm, epoch, batch):
    """Clip through FSDP and report gradient finiteness collectively."""
    clip_limit = max_grad_norm if max_grad_norm is not None else float("inf")
    grad_norm = model.clip_grad_norm_(clip_limit)
    all_finite = torch.isfinite(grad_norm.detach()).to(dtype=torch.int32)
    dist.all_reduce(all_finite, op=dist.ReduceOp.MIN)
    if not all_finite.item():
        if dist.get_rank() == 0:
            print(
                f"Non-finite gradient norm at epoch={epoch}, batch={batch}: "
                f"{grad_norm.item()}",
                flush=True,
            )
        return False, grad_norm
    return True, grad_norm


def rebase_optimizer_and_scheduler_lr(optimizer, scheduler, resume_lr):
    """Rebase a restored schedule while preserving parameter-group ratios."""
    resume_lr = float(resume_lr)
    if resume_lr <= 0:
        raise ValueError(f"RESUME_LR must be positive, got {resume_lr}")

    previous_base_lrs = list(scheduler.base_lrs)
    previous_peak_lr = max(previous_base_lrs)
    scheduler.base_lrs = [
        resume_lr * base_lr / previous_peak_lr
        for base_lr in previous_base_lrs
    ]
    if hasattr(scheduler, '_get_closed_form_lr'):
        current_lrs = scheduler._get_closed_form_lr()
    else:
        current_lrs = [resume_lr] * len(optimizer.param_groups)

    for param_group, current_lr, base_lr in zip(
        optimizer.param_groups, current_lrs, scheduler.base_lrs
    ):
        param_group['lr'] = current_lr
        param_group['initial_lr'] = base_lr
    scheduler._last_lr = current_lrs
    return current_lrs


def restart_optimizer_and_scheduler_lr(optimizer, scheduler, resume_lr,
                                       max_steps, eta_min):
    """Start a fresh cosine fine-tuning schedule at ``resume_lr``."""
    resume_lr = float(resume_lr)
    max_steps = int(max_steps)
    eta_min = float(eta_min)
    if resume_lr <= 0:
        raise ValueError(f"RESUME_LR must be positive, got {resume_lr}")
    if max_steps <= 1:
        raise ValueError(f"max_steps must be greater than 1, got {max_steps}")

    restarted_lrs = [
        resume_lr * float(param_group.get('lr_scale', 1.0))
        for param_group in optimizer.param_groups
    ]
    for param_group, restarted_lr in zip(
        optimizer.param_groups, restarted_lrs
    ):
        param_group['lr'] = restarted_lr
        param_group['initial_lr'] = restarted_lr
    scheduler.base_lrs = list(restarted_lrs)
    scheduler.warmup_epochs = 1
    scheduler.warmup_start_lr = resume_lr
    scheduler.max_epochs = max_steps
    scheduler.eta_min = eta_min
    scheduler.last_epoch = 1
    scheduler._step_count = 1
    scheduler._last_lr = list(restarted_lrs)
    return list(scheduler._last_lr)


def configure_optimizer_with_residual_lr(
    model, base_lr, residual_lr, beta_1, beta_2, weight_decay
):
    """Use a higher LR only for the newly added residual decoder."""
    if base_lr <= 0 or residual_lr <= 0:
        raise ValueError("Optimizer learning rates must be positive")

    buckets = {
        'backbone_decay': [],
        'backbone_no_decay': [],
        'residual_decay': [],
        'residual_no_decay': [],
    }
    counts = {key: 0 for key in buckets}
    for name, parameter in model.named_parameters():
        is_residual = 'decoder_residual' in name
        no_decay = any(token in name for token in (
            'var_embed', 'pos_embed', 'time_pos_embed'
        ))
        family = 'residual' if is_residual else 'backbone'
        decay_kind = 'no_decay' if no_decay else 'decay'
        key = f'{family}_{decay_kind}'
        buckets[key].append(parameter)
        counts[key] += parameter.numel()

    residual_count = counts['residual_decay'] + counts['residual_no_decay']
    if residual_count == 0:
        raise ValueError(
            "model.residual_decoder_lr was set but no decoder_residual "
            "parameters were found after model wrapping"
        )

    lr_scale = residual_lr / base_lr
    group_specs = (
        ('backbone_decay', base_lr, weight_decay, 1.0),
        ('backbone_no_decay', base_lr, 0.0, 1.0),
        ('residual_decay', residual_lr, weight_decay, lr_scale),
        ('residual_no_decay', residual_lr, 0.0, lr_scale),
    )
    groups = []
    for key, group_lr, group_weight_decay, group_scale in group_specs:
        if buckets[key]:
            groups.append({
                'params': buckets[key],
                'lr': group_lr,
                'initial_lr': group_lr,
                'lr_scale': group_scale,
                'betas': (beta_1, beta_2),
                'weight_decay': group_weight_decay,
                'group_name': key,
            })
    return torch.optim.AdamW(groups), counts


def load_model_checkpoint(model, state_dict, allow_residual_decoder_init=False):
    """Load a checkpoint, optionally allowing only a new residual head."""
    incompatible = model.load_state_dict(
        state_dict, strict=not allow_residual_decoder_init
    )
    if not allow_residual_decoder_init:
        return [], []

    allowed_missing_prefixes = (
        'decoder_residual.',
        'decoder_residual_conv3d.',
        'decoder_residual_feature_proj.',
        'decoder_residual_feature_conv3d.',
    )
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Partial resume may initialize only configured residual decoder "
            "parameters; "
            f"invalid missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return incompatible.missing_keys, incompatible.unexpected_keys


def initialize_ema_parameters(model, restored_parameters=None):
    """Create an EMA parameter copy, optionally restoring a saved copy."""
    parameters = list(model.parameters())
    if restored_parameters is not None:
        if len(parameters) != len(restored_parameters):
            raise ValueError(
                "Saved EMA parameter count does not match the model: "
                f"{len(restored_parameters)} != {len(parameters)}"
            )
        ema_parameters = []
        for index, (parameter, restored) in enumerate(zip(
            parameters, restored_parameters
        )):
            if parameter.shape != restored.shape:
                raise ValueError(
                    f"Saved EMA parameter {index} has shape "
                    f"{tuple(restored.shape)}, expected {tuple(parameter.shape)}"
                )
            ema_parameters.append(
                restored.detach().to(
                    device=parameter.device, dtype=parameter.dtype
                ).clone()
            )
        return ema_parameters
    return [parameter.detach().clone() for parameter in parameters]


@torch.no_grad()
def update_ema_parameters(model, ema_parameters, decay):
    """Update an in-memory EMA copy without invoking FSDP state-dict hooks."""
    one_minus_decay = 1.0 - decay
    for parameter, ema_parameter in zip(model.parameters(), ema_parameters):
        ema_parameter.mul_(decay).add_(
            parameter.detach(), alpha=one_minus_decay
        )


@contextmanager
def use_ema_parameters(model, ema_parameters):
    """Temporarily swap model parameters to EMA values for sampling."""
    if ema_parameters is None:
        yield
        return
    parameters = list(model.parameters())
    backups = [parameter.detach().clone() for parameter in parameters]
    try:
        with torch.no_grad():
            for parameter, ema_parameter in zip(parameters, ema_parameters):
                parameter.copy_(ema_parameter)
        yield
    finally:
        with torch.no_grad():
            for parameter, backup in zip(parameters, backups):
                parameter.copy_(backup)


def latest_complete_best_checkpoint(checkpoint_path, checkpoint_filename,
                                    tensor_par_size):
    """Find a stable or numbered BEST checkpoint present for every TP rank."""
    stable_name = f"{checkpoint_filename}_BEST"
    if all(
        os.path.exists(os.path.join(
            checkpoint_path,
            f"{stable_name}_rank_{rank}.ckpt",
        ))
        for rank in range(tensor_par_size)
    ):
        return stable_name

    pattern = os.path.join(
        checkpoint_path,
        f"{checkpoint_filename}_BEST_*_rank_0.ckpt",
    )
    candidates = []
    prefix = f"{checkpoint_filename}_BEST_"
    suffix = "_rank_0.ckpt"
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        epoch_text = basename[len(prefix):-len(suffix)]
        if epoch_text.isdigit():
            candidates.append((int(epoch_text), epoch_text))

    for _, epoch_text in sorted(candidates, reverse=True):
        checkpoint_name = f"{checkpoint_filename}_BEST_{epoch_text}"
        if all(
            os.path.exists(os.path.join(
                checkpoint_path,
                f"{checkpoint_name}_rank_{rank}.ckpt",
            ))
            for rank in range(tensor_par_size)
        ):
            return checkpoint_name
    return None


def continuation_epoch_fields(epoch, epoch_completed):
    """Return restart metadata for a routine wall-time checkpoint.

    A deadline can arrive in the middle of an epoch. In that case the current
    model/optimizer state is retained and the interrupted epoch is replayed.
    If the epoch completed, the next allocation starts at the following one.
    Crucially, neither case rewinds the logical position to ``best_epoch``.
    """
    epoch = int(epoch)
    return {
        'epoch': epoch,
        'next_epoch': epoch + 1 if epoch_completed else epoch,
    }


def completed_epoch_improves_best(epoch_completed, current_loss, best_loss):
    """Only allow a fully completed epoch to replace the best checkpoint."""
    return bool(epoch_completed) and float(current_loss) < float(best_loss)


def should_generate_preview(epoch, epoch_start, epoch_completed, period,
                            generate_on_allocation_start=True,
                            epoch_improved=False):
    """Generate after a new best or at the configured epoch cadence.

    ``epoch_start`` and ``generate_on_allocation_start`` remain in the
    signature for compatibility with callers/tests from the older behavior.
    Allocation-start previews now run before training and are handled by
    ``should_generate_resume_preview`` below.
    """
    if not epoch_completed or epoch <= 0:
        return False
    return bool(epoch_improved) or epoch % period == 0


def should_generate_resume_preview(resume_loaded_best_state,
                                   generate_on_allocation_start):
    """Preview a restored best state before taking another optimizer step."""
    return bool(resume_loaded_best_state and generate_on_allocation_start)


def should_save_periodic_checkpoint(epoch, epoch_completed, period):
    """Select completed current-state milestone epochs."""
    return bool(epoch_completed) and epoch > 0 and epoch % period == 0


def modality_metric_spec(dict_root_dirs, dict_in_variables, dataset):
    """Build metric labels for any number of configured datasets/modalities."""
    if dataset == "imagenet":
        return [('imagenet', 'imagenet')]

    result = []
    for dataset_key in dict_root_dirs:
        variables = dict_in_variables.get(dataset_key, [])
        if isinstance(variables, str):
            variables = [variables]
        variable_label = '+'.join(map(str, variables)) or 'unknown'
        result.append((str(dataset_key), variable_label))
    return result


def normalize_dataset_key(dataset_key):
    """Normalize collate/broadcast dataset identifiers to a metric key."""
    if isinstance(dataset_key, (list, tuple)):
        return ''.join(map(str, dataset_key))
    return str(dataset_key)


def update_loss_degradation_streak(current_loss, best_loss, factor, streak,
                                   patience):
    """Track consecutive finite epochs whose loss is far above the best."""
    current_loss = float(current_loss)
    best_loss = float(best_loss)
    if not np.isfinite(best_loss) or current_loss <= best_loss * factor:
        return 0, False
    streak += 1
    return streak, streak >= patience


def update_ema_degradation(current_loss, ema_loss, best_ema_loss, alpha,
                           factor, streak, patience):
    """Update an EMA and detect sustained relative regression from its best."""
    current_loss = float(current_loss)
    ema_loss = (
        current_loss
        if ema_loss is None or not np.isfinite(ema_loss)
        else alpha * current_loss + (1.0 - alpha) * float(ema_loss)
    )
    best_ema_loss = float(best_ema_loss)
    if not np.isfinite(best_ema_loss) or ema_loss < best_ema_loss:
        best_ema_loss = ema_loss
    if ema_loss <= best_ema_loss * factor:
        streak = 0
    else:
        streak += 1
    return ema_loss, best_ema_loss, streak, streak >= patience


def main(device):
#1. Load arguments from config file and setup parallelization
##############################################################################################################

    print("in main()","sys.argv[1] ",sys.argv[1],flush=True) 
    
    # Use torch.distributed + torchrun env, not SLURM
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    # LOCAL_RANK is set by torchrun; needed later for FSDP(device_id=...)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Emit flushed, per-rank breadcrumbs around each potentially blocking
    # training phase.  TRACE_RANK_SYNC makes an "after" marker trustworthy on
    # CUDA/ROCm, where kernels otherwise execute asynchronously.  The Frontier
    # launcher enables both switches for long-running jobs.
    trace_rank_phases = os.environ.get("TRACE_RANK_PHASES", "0").lower() in (
        "1", "true", "yes"
    )
    trace_rank_sync = os.environ.get("TRACE_RANK_SYNC", "0").lower() in (
        "1", "true", "yes"
    )

    def rank_phase(phase, epoch, batch, synchronize=False):
        if not trace_rank_phases:
            return
        if synchronize and trace_rank_sync and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        print(
            "[rank-phase]"
            f" time={time.time():.6f}"
            f" host={os.uname().nodename}"
            f" rank={world_rank}"
            f" local_rank={local_rank}"
            f" epoch={epoch}"
            f" batch={batch}"
            f" phase={phase}",
            flush=True,
        )

    
    # world_size = int(os.environ['SLURM_NTASKS'])
    # world_rank = dist.get_rank()

    config_path = sys.argv[1]

    if world_rank==0:
        print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    if world_rank==0: 
        print(conf,flush=True)

    max_epochs = conf['trainer']['max_epochs']

    data_type = conf['trainer']['data_type']

    gpu_type = conf['trainer']['gpu_type']

    checkpoint_path = conf['trainer'].get('checkpoint_path')
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    checkpoint_filename_for_loading = conf['trainer']['checkpoint_filename_for_loading']

    inference_path = conf['trainer'].get('inference_path')

    resume_checkpoint_path = conf['trainer'].get('resume_checkpoint_path')

    reset_scheduler_on_resume = conf['trainer'].get(
        'reset_scheduler_on_resume', False
    )

    reset_optimizer_on_resume = conf['trainer'].get(
        'reset_optimizer_on_resume', False
    )

    reset_best_metrics_on_resume = conf['trainer'].get(
        'reset_best_metrics_on_resume', False
    )

    allow_residual_decoder_init = bool(conf['trainer'].get(
        'allow_residual_decoder_init_on_resume', False
    ))

    preview_only = conf['trainer'].get('preview_only', False)
    sampling_audit = bool(conf['trainer'].get('sampling_audit', False))
    audit_num_samples = int(conf['trainer'].get('audit_num_samples', 1))
    if sampling_audit and not preview_only:
        raise ValueError("trainer.sampling_audit requires preview_only=True")
    if audit_num_samples <= 0:
        raise ValueError("trainer.audit_num_samples must be positive")

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']
    resume_override = os.environ.get('RESUME_FROM_CHECKPOINT')
    if resume_override is not None:
        resume_from_checkpoint = resume_override.lower() in ('1', 'true', 'yes')
        if resume_from_checkpoint:
            checkpoint_filename_for_loading = (
                os.environ.get('RESUME_CHECKPOINT_NAME')
                or f"{checkpoint_filename}_latest"
            )
    if 'RESUME_CHECKPOINT_PATH' in os.environ:
        resume_checkpoint_path = (
            os.environ.get('RESUME_CHECKPOINT_PATH') or None
        )
    # Dataset-change metric resets apply only to the explicit imported source
    # checkpoint. Automatic latest/recovery continuations clear this path.
    reset_best_metrics_on_resume = bool(
        reset_best_metrics_on_resume and resume_checkpoint_path
    )
    reset_scheduler_override = os.environ.get('RESET_SCHEDULER_ON_RESUME')
    if reset_scheduler_override is not None:
        reset_scheduler_on_resume = reset_scheduler_override.lower() in (
            '1', 'true', 'yes'
        )
    reset_optimizer_override = os.environ.get('RESET_OPTIMIZER_ON_RESUME')
    if reset_optimizer_override is not None:
        reset_optimizer_on_resume = reset_optimizer_override.lower() in (
            '1', 'true', 'yes'
        )

    recovery_conf = conf.get('numerical_recovery', {})
    numerical_recovery_enabled = recovery_conf.get('enabled', False)
    recovery_lr_decay_factor = float(
        recovery_conf.get('lr_decay_factor', 0.5)
    )
    recovery_min_lr = float(recovery_conf.get('min_lr', 1e-8))
    recovery_max_retries = int(recovery_conf.get('max_retries', 3))
    recovery_loss_degradation_factor = float(
        recovery_conf.get('loss_degradation_factor', 2.0)
    )
    recovery_loss_degradation_patience = int(
        recovery_conf.get('loss_degradation_patience', 2)
    )
    recovery_ema_alpha = float(recovery_conf.get('ema_alpha', 0.3))
    recovery_modality_degradation_factor = float(
        recovery_conf.get('modality_degradation_factor', 1.2)
    )
    recovery_modality_degradation_patience = int(
        recovery_conf.get('modality_degradation_patience', 3)
    )
    recovery_plateau_patience = int(
        recovery_conf.get('plateau_patience', 0)
    )
    recovery_stop_after_max_retries = bool(
        recovery_conf.get('stop_after_max_retries', False)
    )
    recovery_attempt = int(os.environ.get('RECOVERY_ATTEMPT', '0'))

    ema_decay = float(conf['model'].get('ema_decay', 0.0))
    if ema_decay < 0.0 or ema_decay >= 1.0:
        raise ValueError("model.ema_decay must be in [0, 1)")

    if not 0.0 < recovery_lr_decay_factor < 1.0:
        raise ValueError("numerical_recovery.lr_decay_factor must be in (0, 1)")
    if recovery_min_lr <= 0.0:
        raise ValueError("numerical_recovery.min_lr must be positive")
    if recovery_max_retries < 0:
        raise ValueError("numerical_recovery.max_retries cannot be negative")
    if recovery_loss_degradation_factor <= 1.0:
        raise ValueError(
            "numerical_recovery.loss_degradation_factor must be > 1"
        )
    if recovery_loss_degradation_patience < 1:
        raise ValueError(
            "numerical_recovery.loss_degradation_patience must be positive"
        )
    if not 0.0 < recovery_ema_alpha <= 1.0:
        raise ValueError("numerical_recovery.ema_alpha must be in (0, 1]")
    if recovery_modality_degradation_factor <= 1.0:
        raise ValueError(
            "numerical_recovery.modality_degradation_factor must be > 1"
        )
    if recovery_modality_degradation_patience < 1:
        raise ValueError(
            "numerical_recovery.modality_degradation_patience must be positive"
        )
    if recovery_plateau_patience < 0:
        raise ValueError(
            "numerical_recovery.plateau_patience cannot be negative"
        )

    fsdp_size = conf['parallelism']['fsdp_size']

    simple_ddp_size = conf['parallelism']['simple_ddp_size']

    tensor_par_size = conf['parallelism']['tensor_par_size']

    seq_par_size = conf['parallelism']['seq_par_size']

    cpu_offload_flag = conf['parallelism']['cpu_offloading']
 
    lr = float(conf['model']['lr'])

    residual_decoder_lr_value = conf['model'].get('residual_decoder_lr')
    residual_decoder_lr = (
        float(residual_decoder_lr_value)
        if residual_decoder_lr_value is not None else None
    )

    beta_1 = float(conf['model']['beta_1'])

    beta_2 = float(conf['model']['beta_2'])

    weight_decay = float(conf['model']['weight_decay'])

    warmup_steps = conf['model']['warmup_steps']

    max_steps = conf['model']['max_steps']

    warmup_start_lr = float(conf['model']['warmup_start_lr'])

    eta_min = float(conf['model']['eta_min'])

    loss_fn = conf['model']['loss_fn']

    default_vars =  conf['model']['net']['init_args']['default_vars']

    tile_size = conf['model']['net']['init_args']['tile_size']

    patch_size = conf['model']['net']['init_args']['patch_size']
 
    emb_dim = conf['model']['net']['init_args']['embed_dim']

    depth = conf['model']['net']['init_args']['depth']

    num_heads = conf['model']['net']['init_args']['num_heads']
    
    decoder_embed_dim = conf['model']['net']['init_args']['decoder_embed_dim']

    decoder_depth = conf['model']['net']['init_args']['decoder_depth']

    decoder_num_heads = conf['model']['net']['init_args']['decoder_num_heads']

    mlp_ratio = conf['model']['net']['init_args']['mlp_ratio']

    mlp_ratio_decoder = conf['model']['net']['init_args']['mlp_ratio_decoder']

    drop_path = conf['model']['net']['init_args']['drop_path']

    linear_decoder = conf['model']['net']['init_args']['linear_decoder'] 

    residual_mlp_decoder = bool(
        conf['model']['net']['init_args'].get('residual_mlp_decoder', False)
    )
    residual_mlp_ratio = float(
        conf['model']['net']['init_args'].get('residual_mlp_ratio', 2.0)
    )
    residual_conv3d_decoder = bool(
        conf['model']['net']['init_args'].get(
            'residual_conv3d_decoder', False
        )
    )
    residual_conv3d_channels = int(
        conf['model']['net']['init_args'].get(
            'residual_conv3d_channels', 16
        )
    )
    residual_feature_conv3d_decoder = bool(
        conf['model']['net']['init_args'].get(
            'residual_feature_conv3d_decoder', False
        )
    )
    residual_feature_conv3d_channels = int(
        conf['model']['net']['init_args'].get(
            'residual_feature_conv3d_channels', 8
        )
    )

    twoD = conf['model']['net']['init_args']['twoD']

    use_varemb = conf['model']['net']['init_args']['use_varemb']

    adaptive_patching = conf['model']['net']['init_args']['adaptive_patching']

    assert not adaptive_patching, "Adaptive Patching not implemented for DiffusionVIT yet"

    if adaptive_patching:
        fixed_length = conf['model']['net']['init_args']['fixed_length']
        separate_channels = conf['model']['net']['init_args']['separate_channels']

        if not twoD:
            assert not separate_channels, "Adaptive Patching in 3D with multiple channels (non-separated) is not currently implemented"
    else:
        fixed_length = None
        separate_channels = None

    num_time_steps = conf['model']['net']['init_args']['num_time_steps']

    use_grad_scaler = conf['model']['use_grad_scaler']

    max_grad_norm_value = conf['model'].get('max_grad_norm')
    max_grad_norm = (
        float(max_grad_norm_value) if max_grad_norm_value is not None else None
    )
    log_every_n_steps = max(1, int(conf['trainer'].get('log_every_n_steps', 1)))
    enable_performance_plots = bool(
        conf['trainer'].get('enable_performance_plots', True)
    )
    save_checkpoints = bool(conf['trainer'].get('save_checkpoints', True))
    loss_plot_period = max(
        1, int(conf['trainer'].get('loss_plot_period', 1))
    )
    generation_period = max(
        1, int(conf['trainer'].get('generation_period', 50))
    )
    checkpoint_period = max(
        1, int(conf['trainer'].get('checkpoint_period', 50))
    )
    generate_on_allocation_start = bool(
        conf['trainer'].get('generate_on_allocation_start', True)
    )
    preview_seed = int(conf['trainer'].get('preview_seed', 1234))
    preview_job_tag = f"job{os.environ.get('SLURM_JOB_ID', 'local')}"

    dataset = conf['data']['dataset']
    assert dataset in ["basic_ct", "imagenet", "xct"], "This training script only supports basic_ct, imagenet, or xct datasets"


    dict_root_dirs = conf['data']['dict_root_dirs']

    dict_start_idx = conf['data']['dict_start_idx']

    dict_end_idx = conf['data']['dict_end_idx']

    dict_buffer_sizes = conf['data']['dict_buffer_sizes']

    num_channels_used = conf['data']['num_channels_used']

    dict_in_variables = conf['data']['dict_in_variables']
    modality_metrics = modality_metric_spec(
        dict_root_dirs, dict_in_variables, dataset
    )
    modality_metric_keys = [key for key, _ in modality_metrics]
    modality_metric_labels = dict(modality_metrics)
    modality_metric_indices = {
        key: index for index, key in enumerate(modality_metric_keys)
    }

    batch_size = conf['data']['batch_size']

    num_workers = conf['data']['num_workers']

    pin_memory = conf['data']['pin_memory']

    single_channel = conf['data']['single_channel']

    tile_overlap = conf['data']['tile_overlap']

    use_all_data = conf['data']['use_all_data']

    parallel_factor = fsdp_size * tensor_par_size * seq_par_size
    if isinstance(simple_ddp_size, str) and simple_ddp_size.lower() == "auto":
        if world_size % parallel_factor != 0:
            raise ValueError(
                f"World size {world_size} is not divisible by the configured "
                f"FSDP/tensor/sequence factor {parallel_factor}"
            )
        simple_ddp_size = world_size // parallel_factor
    else:
        simple_ddp_size = int(simple_ddp_size)

    if conf['trainer'].get('auto_checkpoint_path', False):
        checkpoint_root = conf['trainer']['checkpoint_root']
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        run_name = (
            f"N{num_nodes}_G{world_size}_DDP{simple_ddp_size}_"
            f"FSDP{fsdp_size}_TP{tensor_par_size}_lr{lr:g}_"
            f"PS{patch_size}_BS{batch_size}_ED{emb_dim}_{data_type}"
        )
        checkpoint_path = os.path.join(checkpoint_root, run_name)
        inference_path = checkpoint_path

    if not checkpoint_path or not inference_path:
        raise ValueError(
            "Set trainer.checkpoint_path and trainer.inference_path, or enable "
            "trainer.auto_checkpoint_path and set trainer.checkpoint_root"
        )

    if world_rank == 0:
        print(f"Resolved simple_ddp_size: {simple_ddp_size}", flush=True)
        print(f"Resolved checkpoint_path: {checkpoint_path}", flush=True)
        print(f"Effective resume_from_checkpoint: {resume_from_checkpoint}", flush=True)

    #Datset specific options
    if dataset == "imagenet":
        imagenet_resize = conf['dataset_options']['imagenet_resize']
    else:
        imagenet_resize = None

    tile_size_x = tile_size[0]
    tile_size_y = tile_size[1]

    if dataset == "imagenet":
        tile_size_z = None
    else:
        tile_size_z = tile_size[2]
    
    assert (tile_size_x%patch_size)==0, "tile_size_x % patch_size must be 0"
    assert (tile_size_y%patch_size)==0, "tile_size_y % patch_size must be 0"
    if dataset != "imagenet":
        assert (tile_size_z%patch_size)==0, "tile_size_z % patch_size must be 0"

    data_par_size = fsdp_size * simple_ddp_size
    assert seq_par_size == 1, "Sequence parallelism not implemented"
    assert (data_par_size * seq_par_size * tensor_par_size)==world_size, "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal to world_size"
    assert (num_heads % tensor_par_size) == 0, "model heads % tensor parallel size must be 0"
    assert (decoder_num_heads % tensor_par_size) == 0, "decoder model heads % tensor parallel size must be 0"

    auto_load_balancing = conf['load_balancing']['auto_load_balancing']
    if auto_load_balancing:
        batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(config_path, data_par_size, batch_size)
    else:
        batches_per_rank_epoch = conf['load_balancing']['batches_per_rank_epoch']
        dataset_group_list = conf['load_balancing']['dataset_group_list']

#2. Initialize model, optimizer, and scheduler
##############################################################################################################
    configured_fused_attn = conf['model'].get('fused_attn')
    if configured_fused_attn is not None:
        try:
            FusedAttn_option = FusedAttn[str(configured_fused_attn).upper()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported model.fused_attn={configured_fused_attn!r}; "
                f"choose one of {[mode.name for mode in FusedAttn]}"
            ) from exc
    elif data_type == "bfloat16":
        if gpu_type == "amd":
            FusedAttn_option = FusedAttn.CK
        elif gpu_type == "nvidia":
            FusedAttn_option = FusedAttn.FLASH
        else:
            print("Invalid gpu_type used, reverting to using default FMHA")
            FusedAttn_option = FusedAttn.DEFAULT
    else:
        if use_fused_attn():
            FusedAttn_option = FusedAttn.DEFAULT
        else:
            FusedAttn_option = FusedAttn.NONE

    #Find correct in_chans to use
    if single_channel:
        max_channels = 1
    else:
        max_channels = 1
        for i,k in enumerate(num_channels_used):
            if num_channels_used[k] > 1:
                max_channels = num_channels_used[k]

    seq_par_group, ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(world_rank = world_rank, data_par_size = data_par_size, tensor_par_size = tensor_par_size, seq_par_size = seq_par_size, fsdp_size = fsdp_size, simple_ddp_size = simple_ddp_size)

    ddpm_scheduler = DDPM_Scheduler(num_time_steps=num_time_steps).to(device)
    model = DiffusionVIT(
        img_size=tile_size,
        patch_size=patch_size,
        in_chans=max_channels,
        embed_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_depth=decoder_depth,
        decoder_embed_dim=decoder_embed_dim, 
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path,
        linear_decoder=linear_decoder,
        residual_mlp_decoder=residual_mlp_decoder,
        residual_mlp_ratio=residual_mlp_ratio,
        residual_conv3d_decoder=residual_conv3d_decoder,
        residual_conv3d_channels=residual_conv3d_channels,
        residual_feature_conv3d_decoder=residual_feature_conv3d_decoder,
        residual_feature_conv3d_channels=residual_feature_conv3d_channels,
        twoD=twoD,
        mlp_ratio_decoder=mlp_ratio_decoder,
        default_vars=default_vars,
        single_channel=single_channel,
        use_varemb=use_varemb,
        adaptive_patching=adaptive_patching,
        fixed_length=fixed_length,
        tensor_par_size=tensor_par_size,
        tensor_par_group=tensor_par_group,
        FusedAttn_option=FusedAttn_option,
        time_steps=num_time_steps,
        class_token=False,
        weight_init='skip',
    ).to(device)

    if not resume_from_checkpoint: #train from scratch
        epoch_start = 0
        loss_list = []
        modality_loss_history = {
            key: [] for key in modality_metric_keys
        }
        if world_rank==0:       
            print("resume from checkpoint was set to False. Pretrain from scratch.",flush=True)

        if world_rank==0:

            # Check whether the specified checkpointing path exists or not

            isExist = os.path.exists(checkpoint_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(checkpoint_path)
                print("The new checkpoint directory is created!")

            #save initial model weights and distribute to all GPUs in the tensor parallel group to synchronize model weights that do not belong to the training block
            init_model_dict = {k: v for k, v in model.state_dict().items() if ('attn' not in  k and 'mlp' not in k and 'var_agg' not in k)}

            print("rank",dist.get_rank(),"init_model_dict.keys()",init_model_dict.keys(),flush=True)

            torch.save(init_model_dict,
                    checkpoint_path+'/initial_'+str(dist.get_rank())+'.pth')

            print("rank", dist.get_rank(),"after torch.save for initial",flush=True)

            del init_model_dict

        dist.barrier()

        if world_rank!=0 and world_rank <tensor_par_size:


           #load initial model weights and synchronize model weights that are not in the training block among sequence parallel GPUs
           src_rank = dist.get_rank() - dist.get_rank(group=tensor_par_group)

           print("rank",dist.get_rank(),"src_rank",src_rank,flush=True)

           map_location = 'cpu'
           #map_location = 'cuda:'+str(device)
           model.load_state_dict(torch.load(checkpoint_path+'/initial_'+str(0)+'.pth',map_location=map_location),strict=False)

    else:
        checkpoint_load_path = resume_checkpoint_path or checkpoint_path
        if world_rank< tensor_par_size:
            if os.path.exists(checkpoint_load_path+"/"+checkpoint_filename_for_loading+"_rank_"+str(world_rank)+".ckpt"):
                print("resume from checkpoint was set to True. Checkpoint path found.",flush=True)

                print("rank",dist.get_rank(),"src_rank",world_rank,flush=True)

                #map_location = 'cuda:'+str(device)
                map_location = 'cpu'

                checkpoint = torch.load(checkpoint_load_path+"/"+checkpoint_filename_for_loading+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                missing_keys, _ = load_model_checkpoint(
                    model,
                    checkpoint['model_state_dict'],
                    allow_residual_decoder_init,
                )
                if missing_keys:
                    print(
                        "Initialized new zero-residual decoder parameters: "
                        f"{missing_keys}",
                        flush=True,
                    )
                epoch_start = checkpoint.get('next_epoch', checkpoint['epoch'] + 1)
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

    if data_type == "float32":
        precision_dt = torch.float32
    elif data_type == "bfloat16":
        precision_dt = torch.bfloat16
    else:
        raise RuntimeError("Data type not supported")

    reduce_data_type = conf['trainer'].get('reduce_data_type', data_type)
    precision_types = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    if reduce_data_type not in precision_types:
        raise ValueError(
            f"Unsupported trainer.reduce_data_type={reduce_data_type!r}"
        )
    reduce_dt = precision_types[reduce_data_type]

    if world_rank == 0:
        print(
            f"Attention backend: {FusedAttn_option.name}; "
            f"parameter dtype: {precision_dt}; reduction dtype: {reduce_dt}; "
            f"max_grad_norm: {max_grad_norm}",
            flush=True,
        )

    bfloatPolicy = MixedPrecision(
        param_dtype=precision_dt,
        # Gradient communication precision.
        reduce_dtype=reduce_dt,
        # Buffer precision.
        buffer_dtype=precision_dt,
    )

    #add hybrid sharded FSDP
    if fsdp_size > 1 and simple_ddp_size > 1:
        model = FSDP(model, device_id=local_rank, process_group= (fsdp_group,simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add fully sharded FSDP
    elif fsdp_size > 1 and simple_ddp_size == 1:
        model = FSDP(model, device_id=local_rank, process_group= fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add unsharded DDP
    else:
        model = FSDP(model, device_id=local_rank, process_group= simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )

    check_fn = lambda submodule: isinstance(submodule, Block)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )

    if residual_decoder_lr is None:
        optimizer = configure_optimizer(
            model, lr, beta_1, beta_2, weight_decay
        )
    else:
        if not (
            residual_mlp_decoder
            or residual_conv3d_decoder
            or residual_feature_conv3d_decoder
        ):
            raise ValueError(
                "model.residual_decoder_lr requires a residual decoder"
            )
        optimizer, optimizer_group_counts = (
            configure_optimizer_with_residual_lr(
                model, lr, residual_decoder_lr,
                beta_1, beta_2, weight_decay,
            )
        )
        if world_rank == 0:
            print(
                "Configured residual decoder LR groups: "
                f"base_lr={lr}, residual_lr={residual_decoder_lr}, "
                f"parameter_counts={optimizer_group_counts}",
                flush=True,
            )
    scheduler = configure_scheduler(optimizer,warmup_steps,max_steps,warmup_start_lr,eta_min)

    resume_best_loss = None
    resume_best_epoch = -1
    resume_loaded_best_state = False
    resume_monitor_state = {}
    resume_epochs_without_improvement = 0
    resume_ema_parameters = None

    if resume_from_checkpoint:

        print("optimizer resume from checkpoint was set to True",flush=True)

        src_rank = world_rank - tensor_par_size * dist.get_rank(group=data_seq_ort_group)

        #map_location = 'cuda:'+str(device)
        map_location = 'cpu'

        checkpoint = torch.load(checkpoint_load_path+"/"+checkpoint_filename_for_loading+"_rank_"+str(src_rank)+".ckpt",map_location=map_location)
        if not reset_optimizer_on_resume:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif world_rank == 0:
            print(
                "Resetting optimizer state instead of restoring checkpoint "
                "moments",
                flush=True,
            )
        loss_list = checkpoint['loss_list']
        saved_modality_history = checkpoint.get('modality_loss_history', {})
        modality_loss_history = {
            key: list(saved_modality_history.get(key, []))
            for key in modality_metric_keys
        }
        epoch_start = checkpoint.get('next_epoch', checkpoint['epoch'] + 1)
        resume_best_epoch = checkpoint.get('best_epoch', checkpoint['epoch'])
        resume_best_loss = checkpoint.get('best_loss')
        resume_loaded_best_state = checkpoint.get(
            'is_best_state',
            '_BEST_' in checkpoint_filename_for_loading
            or checkpoint_filename_for_loading.endswith('_RECOVERY'),
        )
        resume_monitor_state = checkpoint.get('loss_monitor_state', {})
        resume_epochs_without_improvement = int(
            checkpoint.get('epochs_without_improvement', 0)
        )
        resume_ema_parameters = checkpoint.get('ema_parameters')
        if world_rank == 0:
            print(
                "Restored checkpoint: "
                f"type={checkpoint.get('checkpoint_type', 'legacy')}, "
                f"stored_epoch={checkpoint['epoch']}, "
                f"next_epoch={epoch_start}, "
                f"best_epoch={resume_best_epoch}",
                flush=True,
            )
        if resume_best_loss is None:
            finite_losses = [
                float(value.item() if torch.is_tensor(value) else value)
                for value in loss_list
                if np.isfinite(float(
                    value.item() if torch.is_tensor(value) else value
                ))
            ]
            if finite_losses:
                resume_best_loss = min(finite_losses)
        del checkpoint

        resume_lr_override = os.environ.get('RESUME_LR')
        if resume_lr_override:
            previous_lrs = [group['lr'] for group in optimizer.param_groups]
            if reset_scheduler_on_resume:
                rebased_lrs = restart_optimizer_and_scheduler_lr(
                    optimizer, scheduler, float(resume_lr_override),
                    max_steps, eta_min,
                )
                lr_action = "Restarted"
            else:
                rebased_lrs = rebase_optimizer_and_scheduler_lr(
                    optimizer, scheduler, float(resume_lr_override)
                )
                lr_action = "Rebased"
            if world_rank == 0:
                print(
                    f"{lr_action} restored optimizer/scheduler learning rates "
                    f"from {previous_lrs} to base={resume_lr_override}, "
                    f"current={rebased_lrs}",
                    flush=True,
                )

    if use_grad_scaler:
        scaler = ShardedGradScaler(init_scale=8192, growth_interval=100)
        min_scale= 128

#3. Initialize Dataloader
##############################################################################################################
    if dist.get_rank(tensor_par_group) == 0:
        data_module = NativePytorchDataModule(dict_root_dirs=dict_root_dirs,
            dict_start_idx=dict_start_idx,
            dict_end_idx=dict_end_idx,
            dict_buffer_sizes=dict_buffer_sizes,
            dict_in_variables=dict_in_variables,
            num_channels_used = num_channels_used,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            patch_size = patch_size,
            tile_size_x = tile_size_x,
            tile_size_y = tile_size_y,
            tile_size_z = tile_size_z,
            twoD = twoD,
            single_channel = single_channel,
            return_label = False,
            dataset_group_list = dataset_group_list,
            batches_per_rank_epoch = batches_per_rank_epoch,
            tile_overlap = tile_overlap,
            use_all_data = use_all_data,
            adaptive_patching = adaptive_patching,
            fixed_length = fixed_length,
            separate_channels = separate_channels,
            data_par_size = data_par_size,
            ddp_group = ddp_group,
            dataset = dataset,
            imagenet_resize = imagenet_resize,
        ).to(device)

        data_module.setup()

        train_dataloader = data_module.train_dataloader()

    audit_reference_volume = None
    if sampling_audit:
        if twoD:
            raise ValueError("The current sampling audit is specifically 3D")
        if world_rank < tensor_par_size:
            audit_batch = next(iter(train_dataloader))
            audit_reference_volume = audit_batch[0][
                :audit_num_samples
            ].detach().float().cpu()
            print(
                "Sampling audit reference: "
                f"shape={tuple(audit_reference_volume.shape)}, "
                f"mean={audit_reference_volume.mean().item():.6g}, "
                f"std={audit_reference_volume.std().item():.6g}, "
                f"min={audit_reference_volume.min().item():.6g}, "
                f"max={audit_reference_volume.max().item():.6g}",
                flush=True,
            )

#4. Training Loop
##############################################################################################################

    isExist = os.path.exists(inference_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(inference_path, exist_ok=True)
        print("The new inference directory is created!")

    #Find max batches
    iterations_per_epoch = 0
    for i,k in enumerate(batches_per_rank_epoch):
        if batches_per_rank_epoch[k] > iterations_per_epoch:
            iterations_per_epoch = batches_per_rank_epoch[k]

    ### to get the best loss saved
    fid_scores = []
    fid_epochs = []
    fid_eval_period = 10
    
    # A routine latest checkpoint is not itself the best state, but it still
    # carries the persistent best metric metadata. Preserve that comparison
    # baseline while continuing from the latest model/optimizer state.
    best_loss = (
        float(resume_best_loss)
        if resume_best_loss is not None and not reset_best_metrics_on_resume
        else float("inf")
    )
    best_epoch = resume_best_epoch if resume_loaded_best_state else -1
    epochs_without_improvement = (
        0 if resume_loaded_best_state
        else resume_epochs_without_improvement
    )
    max_patience = 50000
    patience = 2000
    lr_decay_count = 0
    decay_factor = 0.9
    patience_inc_rate = 1.25 

    ema_parameters = None
    if ema_decay > 0.0:
        ema_parameters = initialize_ema_parameters(
            model, resume_ema_parameters
        )
        if world_rank == 0:
            ema_source = 'checkpoint' if resume_ema_parameters else 'model'
            print(
                f"EMA enabled: decay={ema_decay}, initialized from "
                f"{ema_source}",
                flush=True,
            )
    del resume_ema_parameters
        
    best_model_state = (
        copy.deepcopy(model.state_dict())
        if resume_loaded_best_state and save_checkpoints else None
    )
    best_optimizer_state = (
        copy.deepcopy(optimizer.state_dict())
        if resume_loaded_best_state and save_checkpoints else None
    )
    best_scheduler_state = (
        copy.deepcopy(scheduler.state_dict())
        if resume_loaded_best_state and save_checkpoints else None
    )
    best_ema_parameters = (
        [parameter.detach().clone() for parameter in ema_parameters]
        if resume_loaded_best_state and ema_parameters is not None
        else None
    )
    # A deliberate best-state recovery starts a fresh monitoring phase. An
    # ordinary latest-state continuation carries its EMA and streaks forward.
    if resume_loaded_best_state:
        resume_monitor_state = {}
    overall_loss_ema = resume_monitor_state.get('overall_ema')
    overall_best_ema = float(
        resume_monitor_state.get('overall_best_ema', best_loss)
    )
    overall_degradation_streak = int(
        resume_monitor_state.get('overall_streak', 0)
    )
    modality_loss_emas = {
        key: resume_monitor_state.get('modality_ema', {}).get(key)
        for key in modality_metric_keys
    }
    modality_best_emas = {
        key: float(
            resume_monitor_state.get('modality_best_ema', {}).get(
                key, float('inf')
            )
        )
        for key in modality_metric_keys
    }
    modality_degradation_streaks = {
        key: int(
            resume_monitor_state.get('modality_streak', {}).get(key, 0)
        )
        for key in modality_metric_keys
    }

    def loss_monitor_state_dict():
        return {
            'overall_ema': overall_loss_ema,
            'overall_best_ema': overall_best_ema,
            'overall_streak': overall_degradation_streak,
            'modality_ema': dict(modality_loss_emas),
            'modality_best_ema': dict(modality_best_emas),
            'modality_streak': dict(modality_degradation_streaks),
        }

    def generate_preview_images(epoch, filename_suffix):
        """Generate one coordinated preview per tensor-parallel replica."""
        if world_rank < tensor_par_size:
            with use_ema_parameters(model, ema_parameters):
                for var in default_vars:
                    model.eval()
                    if sampling_audit:
                        timestep_rows, timestep_path = (
                            audit_noise_prediction_by_timestep(
                                model, audit_reference_volume, var, device,
                                precision_dt, patch_size, inference_path,
                                num_time_steps=num_time_steps,
                                seed=preview_seed,
                            )
                        )
                        summaries, raw_path, stats_path, figure_path = (
                            audit_sample_images(
                                model, audit_reference_volume, var, device,
                                tile_size, precision_dt, patch_size,
                                inference_path,
                                num_time_steps=num_time_steps,
                                seed=preview_seed,
                                num_samples=audit_num_samples,
                            )
                        )
                        print(
                            f"Sampling audit timestep metrics: {timestep_rows}",
                            flush=True,
                        )
                        print(
                            "Sampling audit outputs: "
                            f"timestep_csv={timestep_path}, raw={raw_path}, "
                            f"stats={stats_path}, figure={figure_path}, "
                            f"summaries={summaries}",
                            flush=True,
                        )
                    else:
                        sample_images(
                            model, var, device, tile_size, precision_dt,
                            patch_size, epoch=epoch, num_samples=5, twoD=twoD,
                            save_path=inference_path,
                            num_time_steps=num_time_steps,
                            seed=preview_seed,
                            filename_tag=(
                                f"{preview_job_tag}_{filename_suffix}"
                            ),
                        )
                    model.train()
        dist.barrier()

    # A BEST checkpoint imported from another run directory/prefix must also
    # exist in this run's namespace. Otherwise a later walltime continuation
    # retains only the latest state and numerical recovery cannot find the
    # original safe rollback point if fine-tuning has not improved yet.
    imported_best_needs_materialization = (
        resume_loaded_best_state
        and save_checkpoints
        and (
            checkpoint_load_path != checkpoint_path
            or checkpoint_filename_for_loading
            != f"{checkpoint_filename}_BEST"
        )
    )
    if imported_best_needs_materialization:
        if world_rank < tensor_par_size:
            imported_best_file = os.path.join(
                checkpoint_path,
                f"{checkpoint_filename}_BEST_rank_{world_rank}.ckpt",
            )
            imported_best_temporary_file = (
                f"{imported_best_file}.tmp-"
                f"{os.environ.get('SLURM_JOB_ID', 'local')}"
            )
            torch.save({
                'epoch': best_epoch,
                'next_epoch': best_epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': best_optimizer_state,
                'scheduler_state_dict': best_scheduler_state,
                'ema_parameters': best_ema_parameters,
                'loss_list': loss_list,
                'modality_loss_history': modality_loss_history,
                'loss_monitor_state': loss_monitor_state_dict(),
                'epochs_without_improvement': 0,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'is_best_state': True,
                'checkpoint_type': 'imported_best',
            }, imported_best_temporary_file)
            os.replace(imported_best_temporary_file, imported_best_file)
        dist.barrier()
        if world_rank == 0:
            print(
                "Materialized imported BEST checkpoint in the new run "
                "directory",
                flush=True,
            )

    if should_generate_resume_preview(
        resume_loaded_best_state, generate_on_allocation_start
    ):
        if world_rank == 0:
            print(
                f"Generating immediate preview from restored best epoch "
                f"{best_epoch}",
                flush=True,
            )
        generate_preview_images(best_epoch, "resume_best")

    if preview_only:
        if not resume_loaded_best_state:
            raise ValueError(
                "trainer.preview_only requires a restored BEST checkpoint"
            )
        if world_rank == 0:
            print("Preview-only run complete; skipping training.", flush=True)
        return False

    if world_rank == 0 and resume_loaded_best_state:
        print(
            f"Restored persistent best state: epoch={best_epoch}, "
            f"loss={best_loss:.6g}",
            flush=True,
        )
    deadline_epoch = int(os.environ.get('TRAINING_DEADLINE_EPOCH', '0'))

    def prepare_numerical_recovery(failure_kind, epoch, batch):
        """Save/select a safe best checkpoint and request a lower-LR retry."""
        if not numerical_recovery_enabled:
            raise FloatingPointError(
                f"Non-finite {failure_kind} at epoch={epoch}, batch={batch}"
            )
        if recovery_attempt >= recovery_max_retries:
            if recovery_stop_after_max_retries:
                if world_rank == 0:
                    print(
                        "Early stopping: numerical recovery retry limit "
                        f"{recovery_max_retries} reached after "
                        f"{failure_kind}",
                        flush=True,
                    )
                return {
                    'kind': 'early_stop',
                    'failure_kind': failure_kind,
                    'epoch': epoch,
                    'batch': batch,
                }
            raise FloatingPointError(
                f"Non-finite {failure_kind} at epoch={epoch}, batch={batch}; "
                f"recovery retry limit {recovery_max_retries} reached"
            )

        checkpoint_name = None
        local_best_available = torch.tensor(
            int(best_model_state is not None),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(local_best_available, op=dist.ReduceOp.MIN)

        if local_best_available.item():
            checkpoint_name = f"{checkpoint_filename}_RECOVERY"
            if world_rank < tensor_par_size:
                checkpoint_file = os.path.join(
                    checkpoint_path,
                    f"{checkpoint_name}_rank_{world_rank}.ckpt",
                )
                temporary_file = (
                    f"{checkpoint_file}.tmp-"
                    f"{os.environ.get('SLURM_JOB_ID', 'local')}"
                )
                torch.save({
                    'epoch': best_epoch,
                    'next_epoch': best_epoch + 1,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': best_optimizer_state,
                    'scheduler_state_dict': best_scheduler_state,
                    'ema_parameters': best_ema_parameters,
                    'loss_list': loss_list,
                    'modality_loss_history': modality_loss_history,
                    'loss_monitor_state': loss_monitor_state_dict(),
                    'epochs_without_improvement': 0,
                    'best_epoch': best_epoch,
                    'best_loss': best_loss,
                    'is_best_state': True,
                    'checkpoint_type': 'recovery_best',
                }, temporary_file)
                os.replace(temporary_file, checkpoint_file)
            dist.barrier()
        else:
            checkpoint_holder = [None]
            if world_rank == 0:
                checkpoint_holder[0] = latest_complete_best_checkpoint(
                    checkpoint_path,
                    checkpoint_filename,
                    tensor_par_size,
                )
            dist.broadcast_object_list(checkpoint_holder, src=0)
            checkpoint_name = checkpoint_holder[0]

        if not checkpoint_name:
            raise FloatingPointError(
                f"Non-finite {failure_kind} at epoch={epoch}, batch={batch}; "
                "no complete best checkpoint is available for recovery"
            )

        current_lr = max(group['lr'] for group in optimizer.param_groups)
        recovery_lr = max(
            current_lr * recovery_lr_decay_factor,
            recovery_min_lr,
        )
        if recovery_lr >= current_lr:
            raise FloatingPointError(
                f"Cannot reduce learning rate below {current_lr}; "
                f"configured minimum is {recovery_min_lr}"
            )

        request = {
            'kind': 'numerical_recovery',
            'failure_kind': failure_kind,
            'epoch': epoch,
            'batch': batch,
            'checkpoint_name': checkpoint_name,
            'resume_lr': recovery_lr,
            'recovery_attempt': recovery_attempt + 1,
        }
        if world_rank == 0:
            print(
                "Requesting numerical recovery: "
                f"checkpoint={checkpoint_name}, lr={current_lr:.8g}->"
                f"{recovery_lr:.8g}, attempt={recovery_attempt + 1}/"
                f"{recovery_max_retries}",
                flush=True,
            )
        return request

    for epoch in range(epoch_start,max_epochs):
        #Reset dataloader module every epoch to ensure all files get used
        if epoch != epoch_start:
            if dist.get_rank(tensor_par_group) == 0:
                data_module.reset()
                train_dataloader = data_module.train_dataloader()

        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        loss = 0.0
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        modality_loss_sums = torch.zeros(
            len(modality_metric_keys), dtype=torch.float64, device=device
        )
        modality_loss_counts = torch.zeros(
            len(modality_metric_keys), dtype=torch.int64, device=device
        )
        if world_rank==0:
            print("Starting epoch ",epoch,flush=True)

        if dist.get_rank(tensor_par_group) == 0:
            it_loader = iter(train_dataloader)

        counter = 0
        walltime_stop_requested = False
        with torch.autograd.set_detect_anomaly(False):
            while counter < iterations_per_epoch:
                counter = counter + 1
                rank_phase("batch_start", epoch, counter)
                if tensor_par_size > 1:
                    if dist.get_rank(tensor_par_group) == 0:
                        rank_phase("before_data_fetch", epoch, counter)
                        data, variables, dict_key = next(it_loader)
                        rank_phase("after_data_fetch", epoch, counter)
                        data = data.to(precision_dt)
                        data = data.to(device)
                        if dataset != "imagenet":
                            dict_key_holder = [normalize_dataset_key(dict_key)]
                        else:
                            dict_key = "imagenet"
                        t = torch.randint(0,num_time_steps,(batch_size,))
                        e = torch.randn_like(data, requires_grad=False)
                        if twoD:
                            a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1).to(precision_dt).to(device)
                        else:
                            a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1,1).to(precision_dt).to(device)
                        t = t.to(device)
                        data = (torch.sqrt(a)*data) + (torch.sqrt(1-a)*e)
                    else:
                        if dataset != "imagenet":
                            dict_key_holder = [None]
                        else: 
                            dict_key = "imagenet"

                    if dataset != "imagenet":
                        dist.broadcast_object_list(
                            dict_key_holder,
                            src=(dist.get_rank() // tensor_par_size
                                 * tensor_par_size),
                            group=tensor_par_group,
                        )
                        dict_key = dict_key_holder[0]

                    if dist.get_rank(tensor_par_group) != 0:
                        if twoD:
                            data = torch.zeros(batch_size, num_channels_used[dict_key], tile_size_x, tile_size_y, dtype=precision_dt).to(device)
                        else:
                            data = torch.zeros(batch_size, num_channels_used[dict_key], tile_size_x, tile_size_y, tile_size_z, dtype=precision_dt).to(device)
                        variables = [None] * num_channels_used[dict_key]
                        t = torch.zeros(batch_size, dtype=torch.int).to(device)
                        e = torch.zeros_like(data, requires_grad=False)
                    dist.broadcast(data, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    dist.broadcast_object_list(variables, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    dist.broadcast(t, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    t = t.to('cpu')
                    dist.broadcast(e, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

                else: #Avoid unnecesary broadcasts if not using tensor parallelism
                    rank_phase("before_data_fetch", epoch, counter)
                    data, variables, dict_key = next(it_loader)
                    rank_phase("after_data_fetch", epoch, counter)
                    data = data.to(precision_dt)
                    data = data.to(device)
                    t = torch.randint(0,num_time_steps,(batch_size,))
                    e = torch.randn_like(data, requires_grad=False)
                    if twoD:
                        a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1).to(precision_dt).to(device)
                    else:
                        a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1,1).to(precision_dt).to(device)
                    data = (torch.sqrt(a)*data) + (torch.sqrt(1-a)*e)
                rank_phase("after_data_prepare", epoch, counter, synchronize=True)
                rank_phase("before_forward", epoch, counter)
                loss = training_step(data, variables, t, e, model, patch_size, twoD, loss_fn)
                rank_phase("after_forward", epoch, counter, synchronize=True)

                if not require_finite_loss(loss, epoch, counter, device):
                    optimizer.zero_grad(set_to_none=True)
                    return prepare_numerical_recovery(
                        'loss', epoch, counter
                    )

                epoch_loss += loss.detach()
                metric_key = normalize_dataset_key(dict_key)
                metric_index = modality_metric_indices.get(metric_key)
                if metric_index is None:
                    raise KeyError(
                        f"Dataloader returned unknown dataset key {metric_key!r}; "
                        f"configured keys are {modality_metric_keys}"
                    )
                modality_loss_sums[metric_index] += loss.detach().double()
                modality_loss_counts[metric_index] += 1
    
                if world_rank == 0 and (
                    counter == 1 or counter % log_every_n_steps == 0
                ):
                    print("epoch: ",epoch,"batch_idx",counter,"world_rank",world_rank,"it_loss ",loss,flush=True)
    
                rank_phase("before_backward", epoch, counter)
                if use_grad_scaler:
                    scaler.scale(loss).backward()
                    rank_phase("after_backward", epoch, counter, synchronize=True)
                    scaler.unscale_(optimizer)
                    gradients_finite, _ = clip_and_require_finite_gradients(
                        model, max_grad_norm, epoch, counter
                    )
                    if not gradients_finite:
                        optimizer.zero_grad(set_to_none=True)
                        return prepare_numerical_recovery(
                            'gradient norm', epoch, counter
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler._scale < min_scale:
                        scaler._scale = torch.tensor(min_scale).to(scaler._scale)
                else:
                    loss.backward()
                    rank_phase("after_backward", epoch, counter, synchronize=True)
                    gradients_finite, _ = clip_and_require_finite_gradients(
                        model, max_grad_norm, epoch, counter
                    )
                    if not gradients_finite:
                        optimizer.zero_grad(set_to_none=True)
                        return prepare_numerical_recovery(
                            'gradient norm', epoch, counter
                        )
                    optimizer.step()
                rank_phase("after_optimizer", epoch, counter, synchronize=True)

                if ema_parameters is not None:
                    update_ema_parameters(model, ema_parameters, ema_decay)

                scheduler.step()
                optimizer.zero_grad()
                rank_phase("batch_complete", epoch, counter)

                # Check periodically on every rank so all processes leave the
                # training loop together with enough time to write a restart.
                if deadline_epoch and counter % 10 == 0:
                    stop_tensor = torch.tensor(
                        int(time.time() >= deadline_epoch), device=device
                    )
                    rank_phase("before_deadline_all_reduce", epoch, counter)
                    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                    rank_phase(
                        "after_deadline_all_reduce", epoch, counter,
                        synchronize=True,
                    )
                    if stop_tensor.item():
                        walltime_stop_requested = True
                        break
        epoch_completed = counter >= iterations_per_epoch
        epoch_loss /= max(counter, 1)
        # Every tensor-parallel rank represents the same data sample, while
        # data-parallel ranks see different samples. Averaging over the full
        # world weights every data replica equally (the TP duplication cancels)
        # and gives every rank the same metric for logging and best-model logic.
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss /= world_size
        dist.all_reduce(modality_loss_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(modality_loss_counts, op=dist.ReduceOp.SUM)

        epoch_modality_losses = {}
        for metric_index, metric_key in enumerate(modality_metric_keys):
            count = modality_loss_counts[metric_index].item()
            if count:
                epoch_modality_losses[metric_key] = (
                    modality_loss_sums[metric_index].item() / count
                )

        if epoch_completed:
            loss_list.append(epoch_loss)
            for metric_key, metric_loss in epoch_modality_losses.items():
                modality_loss_history[metric_key].append(metric_loss)

        if world_rank == 0:
            readable_modality_losses = {
                f"{key}[{modality_metric_labels[key]}]": value
                for key, value in epoch_modality_losses.items()
            }
            completion_label = "complete" if epoch_completed else "partial"
            print(
                f"epoch: {epoch} modality_losses ({completion_label}) "
                f"{readable_modality_losses}",
                flush=True,
            )

        loss_degradation_reasons = []
        if epoch_completed and numerical_recovery_enabled:
            (
                overall_loss_ema,
                overall_best_ema,
                overall_degradation_streak,
                overall_has_degraded,
            ) = update_ema_degradation(
                epoch_loss.item(),
                overall_loss_ema,
                overall_best_ema,
                recovery_ema_alpha,
                recovery_loss_degradation_factor,
                overall_degradation_streak,
                recovery_loss_degradation_patience,
            )
            if overall_has_degraded:
                loss_degradation_reasons.append('smoothed overall loss')

            for metric_key, metric_loss in epoch_modality_losses.items():
                (
                    modality_loss_emas[metric_key],
                    modality_best_emas[metric_key],
                    modality_degradation_streaks[metric_key],
                    modality_has_degraded,
                ) = update_ema_degradation(
                    metric_loss,
                    modality_loss_emas[metric_key],
                    modality_best_emas[metric_key],
                    recovery_ema_alpha,
                    recovery_modality_degradation_factor,
                    modality_degradation_streaks[metric_key],
                    recovery_modality_degradation_patience,
                )
                if modality_has_degraded:
                    loss_degradation_reasons.append(
                        f"smoothed modality loss {metric_key}"
                    )

            if world_rank == 0:
                modality_monitor = {
                    f"{key}[{modality_metric_labels[key]}]": {
                        'ema': modality_loss_emas[key],
                        'best_ema': modality_best_emas[key],
                        'ratio': (
                            modality_loss_emas[key]
                            / modality_best_emas[key]
                        ),
                        'streak': modality_degradation_streaks[key],
                    }
                    for key in epoch_modality_losses
                }
                print(
                    "loss_monitor: "
                    f"overall_ema={overall_loss_ema:.6g}, "
                    f"overall_best_ema={overall_best_ema:.6g}, "
                    f"overall_ratio={overall_loss_ema / overall_best_ema:.4f}, "
                    f"overall_streak={overall_degradation_streak}, "
                    f"modalities={modality_monitor}",
                    flush=True,
                )

        # Persist a newly improved completed epoch before handling a wall-time
        # exit. Otherwise a deadline that lands exactly at the epoch boundary
        # would advance the latest checkpoint while silently losing the new
        # best state. Partial epochs are never eligible.
        epoch_improved = completed_epoch_improves_best(
            epoch_completed, epoch_loss.item(), best_loss
        )
        if epoch_improved:
            best_loss = epoch_loss.item()
            best_epoch = epoch
            epochs_without_improvement = 0

            if save_checkpoints:
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_scheduler_state = copy.deepcopy(scheduler.state_dict())
                best_ema_parameters = (
                    [parameter.detach().clone()
                     for parameter in ema_parameters]
                    if ema_parameters is not None else None
                )
                if world_rank < tensor_par_size:
                    best_checkpoint_file = os.path.join(
                        checkpoint_path,
                        f"{checkpoint_filename}_BEST_rank_{world_rank}.ckpt",
                    )
                    best_temporary_file = (
                        f"{best_checkpoint_file}.tmp-"
                        f"{os.environ.get('SLURM_JOB_ID', 'local')}"
                    )
                    torch.save({
                        'epoch': best_epoch,
                        'next_epoch': best_epoch + 1,
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': best_optimizer_state,
                        'scheduler_state_dict': best_scheduler_state,
                        'ema_parameters': best_ema_parameters,
                        'loss_list': loss_list,
                        'modality_loss_history': modality_loss_history,
                        'loss_monitor_state': loss_monitor_state_dict(),
                        'epochs_without_improvement': 0,
                        'best_epoch': best_epoch,
                        'best_loss': best_loss,
                        'is_best_state': True,
                        'checkpoint_type': 'best',
                    }, best_temporary_file)
                    os.replace(best_temporary_file, best_checkpoint_file)
            dist.barrier()
        elif epoch_completed:
            # Update this before the wall-time checkpoint so the plateau count
            # survives an allocation boundary that lands after this epoch.
            epochs_without_improvement += 1

        # Also check at the epoch boundary in case it did not land on a
        # multiple of ten batches.
        if deadline_epoch and not walltime_stop_requested:
            stop_tensor = torch.tensor(
                int(time.time() >= deadline_epoch), device=device
            )
            rank_phase("before_epoch_deadline_all_reduce", epoch, counter)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            rank_phase(
                "after_epoch_deadline_all_reduce", epoch, counter,
                synchronize=True,
            )
            walltime_stop_requested = bool(stop_tensor.item())

        if walltime_stop_requested:
            # Routine wall-time continuation must retain the latest mutually
            # consistent model, optimizer, and scheduler states. The separate
            # BEST checkpoint remains available for explicit numerical
            # recovery, but must not rewind every ordinary continuation.
            rank_phase("before_restart_model_state", epoch, counter)
            restart_model_state = model.state_dict()
            rank_phase(
                "after_restart_model_state", epoch, counter,
                synchronize=True,
            )
            rank_phase("before_restart_optimizer_state", epoch, counter)
            restart_optimizer_state = optimizer.state_dict()
            rank_phase(
                "after_restart_optimizer_state", epoch, counter,
                synchronize=True,
            )
            restart_scheduler_state = scheduler.state_dict()
            restart_position = continuation_epoch_fields(
                epoch, epoch_completed
            )

            if world_rank < tensor_par_size:
                checkpoint_file = os.path.join(
                    checkpoint_path,
                    f"{checkpoint_filename}_latest_rank_{world_rank}.ckpt",
                )
                temporary_file = f"{checkpoint_file}.tmp-{os.environ.get('SLURM_JOB_ID', 'local')}"
                torch.save({
                    **restart_position,
                    'model_state_dict': restart_model_state,
                    'optimizer_state_dict': restart_optimizer_state,
                    'scheduler_state_dict': restart_scheduler_state,
                    'ema_parameters': ema_parameters,
                    'loss_list': loss_list,
                    'modality_loss_history': modality_loss_history,
                    'loss_monitor_state': loss_monitor_state_dict(),
                    'epochs_without_improvement': epochs_without_improvement,
                    'best_epoch': best_epoch,
                    'best_loss': best_loss,
                    'is_best_state': False,
                    'checkpoint_type': 'latest',
                }, temporary_file)
                os.replace(temporary_file, checkpoint_file)
                print(
                    f"Saved walltime restart checkpoint: {checkpoint_file}",
                    flush=True,
                )

            rank_phase("before_restart_barrier", epoch, counter)
            dist.barrier()
            rank_phase("after_restart_barrier", epoch, counter)
            if world_rank == 0:
                print("Walltime checkpoint complete; requesting continuation.", flush=True)
            return True

        if loss_degradation_reasons:
            optimizer.zero_grad(set_to_none=True)
            return prepare_numerical_recovery(
                ' and '.join(loss_degradation_reasons), epoch, counter
            )
        
        # if epoch % fid_eval_period == 0 and world_rank == 0:
        #     model.eval()
        #     for var in default_vars:
        #         fid = save_intermediate_data_with_fid(model, var, device, tile_size, precision_dt, patch_size,
        #                         epoch=epoch, num_samples=4, twoD=twoD, save_path=inference_path,
        #                         num_time_steps=num_time_steps,
        #                         test_volume_path="/lustre/fs0/scratch/lyngaasir/DiffusiveINR_Data/single_validation/XCT_Concrete/XCT_Concrete_256x256_Z00730_pixel30.8985um_Pair_97.npy") # ,downscale=1#add in config
        #     model.train()
        #     if fid is not None and np.isfinite(fid):
        #         log_and_plot_fid(fid, epoch, fid_scores, fid_epochs, inference_path)
        #     else:
        #         print(f"FID computation failed at epoch {epoch}")
        
        if world_rank==0:
            print("epoch: ",epoch," epoch_loss ",epoch_loss, flush=True)
            if epoch_completed and epoch % loss_plot_period == 0:
                plotLoss(
                    loss_list,
                    save_path=os.path.join(
                        checkpoint_path,
                        f'loss_N{simple_ddp_size//8}_BS{batch_size}_PS'
                        f'{patch_size}_ED{emb_dim}_rank0.png',
                    ),
                )
        if world_rank==1:
            if epoch_completed and epoch % loss_plot_period == 0:
                print("epoch: ",epoch," epoch_loss ",epoch_loss, flush=True)
                plotLoss(
                    loss_list,
                    save_path=os.path.join(
                        checkpoint_path,
                        f'loss_N{simple_ddp_size//8}_BS{batch_size}_PS'
                        f'{patch_size}_ED{emb_dim}_rank1.png',
                    ),
                )

        if enable_performance_plots and ((epoch==1) or (epoch % 50 == 0)) and (dist.get_rank(tensor_par_group) == 0):
            # grab a small batch from the current loader (only this rank has it)
            it_eval = iter(train_dataloader)
            x_eval, variables_eval, _ = next(it_eval)
            x_eval = x_eval.to(precision_dt).to(device)

            # plotPerformance
            plotPerformance(
                model=model,
                device=device,
                x=x_eval,
                variables=variables_eval,
                num_time_steps=num_time_steps,
                patch_size=patch_size,
                twoD=twoD,
                scheduler=ddpm_scheduler,   # your DDPM_Scheduler from above
                epoch=epoch,
                savefol=inference_path,     # where to save the figure
                precision_dt=precision_dt,
                Ntimes=9
            )

            # plotPerformance
            plotPerformanceImgs(
                model=model,
                device=device,
                x=x_eval,
                variables=variables_eval,
                num_time_steps=num_time_steps,
                patch_size=patch_size,
                twoD=twoD,
                scheduler=ddpm_scheduler,   # your DDPM_Scheduler from above
                epoch=epoch,
                savefol=inference_path,     # where to save the figure
                precision_dt=precision_dt,
                Ntimes=9
            )

        dist.barrier()
            
        if (
            numerical_recovery_enabled
            and recovery_plateau_patience > 0
            and epochs_without_improvement >= recovery_plateau_patience
        ):
            return prepare_numerical_recovery(
                f"loss plateau for {epochs_without_improvement} epochs",
                epoch,
                counter,
            )

        dist.barrier()

        periodic_checkpoint_due = should_save_periodic_checkpoint(
            epoch, epoch_completed, checkpoint_period
        )
        if periodic_checkpoint_due:
            # Unlike BEST checkpoints, this milestone is the mutually
            # consistent current training state at the named epoch.
            periodic_model_state = model.state_dict()
            periodic_optimizer_state = optimizer.state_dict()
            periodic_scheduler_state = scheduler.state_dict()
            if world_rank < tensor_par_size:
                periodic_checkpoint_file = os.path.join(
                    checkpoint_path,
                    f"{checkpoint_filename}_EPOCH_{epoch}_rank_"
                    f"{world_rank}.ckpt",
                )
                periodic_temporary_file = (
                    f"{periodic_checkpoint_file}.tmp-"
                    f"{os.environ.get('SLURM_JOB_ID', 'local')}"
                )
                torch.save({
                    'epoch': epoch,
                    'next_epoch': epoch + 1,
                    'model_state_dict': periodic_model_state,
                    'optimizer_state_dict': periodic_optimizer_state,
                    'scheduler_state_dict': periodic_scheduler_state,
                    'ema_parameters': ema_parameters,
                    'loss_list': loss_list,
                    'modality_loss_history': modality_loss_history,
                    'loss_monitor_state': loss_monitor_state_dict(),
                    'epochs_without_improvement': epochs_without_improvement,
                    'best_epoch': best_epoch,
                    'best_loss': best_loss,
                    'is_best_state': False,
                    'checkpoint_type': 'periodic',
                }, periodic_temporary_file)
                os.replace(
                    periodic_temporary_file, periodic_checkpoint_file
                )
                print(
                    "Saved periodic current-state checkpoint: "
                    f"{periodic_checkpoint_file}",
                    flush=True,
                )
            dist.barrier()

        generation_due = should_generate_preview(
            epoch,
            epoch_start,
            epoch_completed,
            generation_period,
            generate_on_allocation_start,
            epoch_improved,
        )
        if generation_due:
            # Always sample the current training state. Do not replace its
            # weights with an older best state or couple previews to best-state
            # availability.
            preview_reason = "new_best" if epoch_improved else "periodic"
            generate_preview_images(epoch, preview_reason)
        else:
            dist.barrier()

    # Final save of best model
    if world_rank == 0:
        print(f"Training completed. Best loss: {best_loss:.6f} at epoch {best_epoch}")

    if save_checkpoints and best_model_state is not None and world_rank < tensor_par_size:
        torch.save({
            'epoch': best_epoch,
            'next_epoch': best_epoch + 1,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'scheduler_state_dict': best_scheduler_state,
            'ema_parameters': best_ema_parameters,
            'loss_list': loss_list,
            'modality_loss_history': modality_loss_history,
            'loss_monitor_state': loss_monitor_state_dict(),
            'epochs_without_improvement': 0,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'is_best_state': True,
            'checkpoint_type': 'final_best',
        }, checkpoint_path+"/"+checkpoint_filename+"_FINALBEST_"+str(best_epoch)+"_rank_"+str(world_rank)+".ckpt".format(best_epoch)) 

        model.load_state_dict(best_model_state)

        with use_ema_parameters(model, best_ema_parameters):
            for var in default_vars:
                model.eval()
                sample_images(
                    model, var, device, tile_size, precision_dt, patch_size,
                    epoch=best_epoch, num_samples=10, twoD=twoD,
                    save_path=inference_path,
                    num_time_steps=num_time_steps, seed=preview_seed,
                    filename_tag=f"{preview_job_tag}_finalbest",
                )
                model.train()
            # save_intermediate_data(model, var, device, tile_size, precision_dt, patch_size,
            #                     epoch=best_epoch, num_samples=2, twoD=twoD, save_path=inference_path,
            #                     num_time_steps=num_time_steps)

    return False


if __name__ == "__main__":

    # This script should be launched with torchrun, e.g.:
    # torchrun --nnodes=1 --nproc_per_node=8 your_script.py

    # torchrun sets these:
    #   RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "RANK and WORLD_SIZE must be set. "
            "Did you launch with `torchrun`?"
        )

    world_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Optional: sanity print
    print(
        f"[rank {world_rank}] world_size={world_size}, "
        f"local_rank={local_rank}",
        flush=True,
    )

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")

    # Init process group; env:// will pick up MASTER_ADDR/PORT set by torchrun
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=7200000),
        # rank/world_size are optional when using env://, but you can keep them:
        rank=world_rank,
        world_size=world_size,
    )

    print("Using dist.init_process_group. world_size", world_size, flush=True)

    # Your main training function
    run_result = main(device)

    if run_result is True:
        continuation_marker = os.environ.get('CONTINUATION_MARKER')
        if not continuation_marker:
            raise RuntimeError(
                "CONTINUATION_MARKER must be set for walltime continuation"
            )
        if world_rank == 0:
            with open(continuation_marker, 'w') as marker_file:
                marker_file.write('resume\n')
            print(f"Created continuation marker: {continuation_marker}", flush=True)
        dist.barrier()

    elif isinstance(run_result, dict) and run_result.get('kind') == 'numerical_recovery':
        recovery_marker = os.environ.get('RECOVERY_MARKER')
        if not recovery_marker:
            raise RuntimeError(
                "RECOVERY_MARKER must be set for numerical recovery"
            )
        if world_rank == 0:
            with open(recovery_marker, 'w') as marker_file:
                marker_file.write(
                    f"{run_result['checkpoint_name']}\n"
                    f"{run_result['resume_lr']:.17g}\n"
                    f"{run_result['recovery_attempt']}\n"
                )
            print(f"Created numerical recovery marker: {recovery_marker}", flush=True)
        dist.barrier()

    dist.destroy_process_group()

# if __name__ == "__main__":

#     os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
#     os.environ['MASTER_PORT'] = "29500"
#     os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
#     os.environ['RANK'] = os.environ['SLURM_PROCID']

#     world_size = int(os.environ['SLURM_NTASKS'])
#     world_rank = int(os.environ['SLURM_PROCID'])
#     local_rank = int(os.environ['SLURM_LOCALID'])

#     torch.cuda.set_device(local_rank)
#     device = torch.cuda.current_device()



#     #torch.backends.cudnn.benchmark = True

#     dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

# #    initialize_process()

#     print("Using dist.init_process_group. world_size ",world_size,flush=True)
    
#     main(device)

#     dist.destroy_process_group()
