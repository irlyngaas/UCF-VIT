import os
import sys
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from einops import rearrange

from UCF_VIT.parse import parse_config
from UCF_VIT.model.utils import get_model
from UCF_VIT.utils.misc import init_par_groups, unpatchify
from UCF_VIT.utils.plotting import plot_2d_slices, plot_3d_center_slices
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler, diffusion_reverse_step

from train import init_dist, set_cuda_device


def main():
    """Parses CLI args and config, loads a trained DiffusionVIT checkpoint, and
    generates samples by running the full reverse-diffusion chain from pure noise.

    Entry point for actually exercising a trained diffusion checkpoint, which
    train.py/test.py/val.py never do themselves (they only ever run the forward
    noising process for a loss). No real data is read -- generation starts from
    `torch.randn` and repeatedly applies `diffusion_reverse_step` -- so this is
    dataset-agnostic and works for any DiffusionVIT config, single- or
    multi-modality alike.

    `--dict_key` selects which dataset key's `data.dict_in_variables` entry is
    passed as `variables` to the model; it is required whenever a config
    trains on more than one dataset key. When `model.use_modality_conditioning`
    is also set, the same `dict_key` is additionally passed to the model
    itself, to drive its per-block modality conditioning -- see
    DiffusionVIT.forward's own docstring.
    """
    parser = ArgumentParser(description="")
    parser.add_argument("config", type=str, help="Path to configuration YAML file")
    parser.add_argument(
        "--launcher",
        type=str,
        default="slurm",
        help="Type of launching to use ",
    )
    parser.add_argument("--dict_key", type=str, default=None, help="Which data.dict_root_dirs key's variables to generate with (required if the config has more than one)")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate per rank")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed (offset by rank so every rank generates different samples)")
    parser.add_argument("--variance_mode", type=str, default="current", choices=["current", "posterior", "ddim"], help="Reverse-step variance mode -- see UCF_VIT.ddpm.ddpm.diffusion_reverse_step")
    parser.add_argument("--num_saved_frames", type=int, default=10, help="Number of evenly-spaced intermediate frames to save for the plot")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated images/plots to (default: <trainer.checkpoint_path>/generated)")
    args = parser.parse_args()

    local_rank = init_dist(args)
    world_rank = dist.get_rank()

    conf = parse_config(args)
    assert conf["model"]["type"] == "DiffusionVIT", "inference_diffusion.py only supports model.type: DiffusionVIT"
    conf["trainer"]["resume_from_checkpoint"] = True

    dict_root_dirs = conf["data"]["dict_root_dirs"]
    if args.dict_key is not None:
        dict_key = args.dict_key
    elif len(dict_root_dirs) == 1:
        dict_key = list(dict_root_dirs)[0]
    else:
        sys.exit("This config trains on more than one dataset key -- pass --dict_key to pick which one's variables to generate with")
    variables = conf["data"]["dict_in_variables"][dict_key]
    use_modality_conditioning = conf["model"]["kwargs"].get("use_modality_conditioning", False)

    ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(world_rank=world_rank, data_par_size=conf["parallelism"]["data_par_size"], tensor_par_size=conf["parallelism"]["tensor_par_size"], fsdp_size=conf["parallelism"]["fsdp_size"], simple_ddp_size=conf["parallelism"]["simple_ddp_size"])

    device = set_cuda_device(local_rank)
    model, epoch_start, loss_list = get_model(conf, {}, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group)
    model.eval()
    if dist.get_rank() == 0:
        print("generating from checkpoint at epoch ", epoch_start - 1, flush=True)

    if conf["trainer"]["data_type"] == "float32":
        precision_dt = torch.float32
    elif conf["trainer"]["data_type"] == "bfloat16":
        precision_dt = torch.bfloat16
    else:
        raise RuntimeError("Data type not supported")

    twoD = conf["data"]["twoD"]
    # Mirrors UCF_VIT.model.utils._model_img_size's own (H, W)/(H, W, D)
    # truncation -- basic_ct's tile_size stays a 3-tuple even when twoD:True.
    img_size = tuple(conf["data"]["tile_size"][:2]) if twoD else tuple(conf["data"]["tile_size"])
    in_chans = conf["data"]["in_chans"]
    patch_size = conf["data"]["patch_size"]
    num_time_steps = conf["model"]["kwargs"]["num_time_steps"]

    ddpm_scheduler = DDPM_Scheduler(num_time_steps=num_time_steps).to(device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device='cpu').manual_seed(args.seed + world_rank)

    noise_shape = (args.num_samples, in_chans) + img_size
    z = torch.randn(noise_shape, generator=generator)

    num_saved_frames = min(args.num_saved_frames, num_time_steps)
    save_timesteps = {round(i * (num_time_steps - 1) / (num_saved_frames - 1)) for i in range(num_saved_frames)} if num_saved_frames > 1 else {0}

    frames = []
    with torch.no_grad():
        for t in reversed(range(num_time_steps)):
            t_batch = torch.full((args.num_samples,), t, dtype=torch.long, device=device)
            if use_modality_conditioning:
                predicted_noise = model(z.to(device=device, dtype=precision_dt), t_batch, variables, dict_key)
            else:
                predicted_noise = model(z.to(device=device, dtype=precision_dt), t_batch, variables)
            predicted_noise = unpatchify(predicted_noise, z.to(device=device, dtype=precision_dt), patch_size, twoD).detach().float().cpu()
            noise = torch.randn(z.shape, generator=generator) if t > 0 else None
            z = diffusion_reverse_step(z, predicted_noise, t, ddpm_scheduler, variance_mode=args.variance_mode, noise=noise)
            if t in save_timesteps:
                frames.append(z)

    output_dir = args.output_dir or os.path.join(conf["trainer"]["checkpoint_path"], "generated")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{'3D' if not twoD else '2D'}_gen_{dict_key}_epoch{epoch_start - 1}_rank{world_rank}.png")

    if not twoD:
        frames = [rearrange(frame, 'b c h w d -> b h w d c').numpy() for frame in frames]
        plot_3d_center_slices(frames, filename)
    else:
        frames = [rearrange(frame, 'b c h w -> b h w c').numpy() for frame in frames]
        plot_2d_slices(frames, filename)

    print(f"rank {world_rank}: saved generated samples to {filename}", flush=True)


if __name__ == "__main__":
    main()
    dist.destroy_process_group()
