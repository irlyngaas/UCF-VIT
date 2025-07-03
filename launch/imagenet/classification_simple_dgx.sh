#!/bin/bash
#SBATCH --partition defq
#SBATCH --nodes 1
#SBATCH --exclusive
#SBATCH --job-name=diffusion_fsdp
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

srun --mpi=pmix --container-mounts /lustre/fs0 --container-mount-home --container-image irlyngaas/ucf-vit:25.05 python $HOME/UCF-VIT/training_scripts/train_diffusion_fsdp.py $HOME/UCF-VIT/configs/xct/diffusion/base_config.yaml
