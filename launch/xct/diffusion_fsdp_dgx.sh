#!/bin/bash
#SBATCH --partition defq
#SBATCH --nodes 4
#SBATCH --exclusive
#SBATCH --job-name=diffusion_fsdp
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=/lustre/fs0/scratch/ziabariak/checkpoint/xct/diffusion/base/full_3D/GradScalar_off/XCTConcrete_1200Vol_synth/32x32x32/Pat2000_Dec0.9/N4_Adaptivelr0.001_P1_BS8_ED1152_nhead12/inference/%x_%j.out
#SBATCH --error=/lustre/fs0/scratch/ziabariak/checkpoint/xct/diffusion/base/full_3D/GradScalar_off/XCTConcrete_1200Vol_synth/32x32x32/Pat2000_Dec0.9/N4_Adaptivelr0.001_P1_BS8_ED1152_nhead12/inference/%x_%j.err

#[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
#[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

#ulimit -n 65536

srun --mpi=pmix --container-mounts /lustre/fs0 --container-mount-home --container-image /lustre/fs0/scratch/ziabariak/sqsh-files/0698614322576143+ucf-vit+25.05-upd4.sqsh python /home/ziabariak/git/UCF-VIT/dev_scripts/train_diffusion_fsdp_wFixedFID.py /home/ziabariak/git/UCF-VIT/configs/xct/diffusion/base_config_dgx.yaml


# srun --mpi=pmix --container-mounts /lustre/fs0 --container-mount-home --container-image /lustre/fs0/scratch/lyngaasir/sqsh-files/0698614322576143+ucf-vit+25.05-upd2.sqsh python $HOME/git/UCF-VIT/dev_scripts/train_diffusion_fsdp.py $HOME/git/UCF-VIT/configs/xct/diffusion/base_config_dgx.yaml

