#!/bin/bash
#SBATCH --partition defq
#SBATCH --nodes 2
#SBATCH --exclusive
#SBATCH --job-name=diffusion_fsdp
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

#[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
#[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

#ulimit -n 65536

srun --mpi=pmix --container-mounts /lustre/fs0 --container-mount-home --container-image /lustre/fs0/scratch/lyngaasir/sqsh-files/0698614322576143+ucf-vit+25.05-upd2.sqsh python $HOME/UCF-VIT/dev_scripts/train_diffusion_fsdp.py $HOME/UCF-VIT/configs/xct/diffusion/base_config_dgx.yaml

