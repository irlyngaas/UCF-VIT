#!/bin/bash
#SBATCH --partition defq
#SBATCH --nodes 16
#SBATCH --exclusive
#SBATCH --job-name=diffusion_fsdp
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=/lustre/fs0/scratch/ziabariak/checkpoint/xct/diffusion/base/full_2D/XCT_Conc_normalized/600Vols/256x256x256/Pat2000_Dec0.9/N16_Adaptivelr0.001_P8_BS256_ED4096_float32/inference/%x_%j.out
# /lustre/fs0/scratch/ziabariak/checkpoint/xct/diffusion/base/full_2D/XCT_Conc_Synth/1200Vols/32x32x32/Pat200000_Dec0.9_OneGPU/N1_GPU1_Adaptivelr0.00001_P1_BS256_ED1024_float32/inference/%x_%j.out
#SBATCH --error=/lustre/fs0/scratch/ziabariak/checkpoint/xct/diffusion/base/full_2D/XCT_Conc_normalized/600Vols/256x256x256/Pat2000_Dec0.9/N16_Adaptivelr0.001_P8_BS256_ED4096_float32/inference/%x_%j.err
# /lustre/fs0/scratch/ziabariak/checkpoint/xct/diffusion/base/full_2D/XCT_Conc_Synth/1200Vols/32x32x32/Pat200000_Dec0.9_OneGPU/N1_GPU1_Adaptivelr0.00001_P1_BS256_ED1024_float32/inference/%x_%j.err

#[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
#[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

#ulimit -n 65536

srun --mpi=pmix --container-mounts /lustre/fs0 --container-mount-home --container-image /lustre/fs0/scratch/ziabariak/sqsh-files/0698614322576143+ucf-vit+25.05-upd4.sqsh python /home/ziabariak/git/UCF-VIT/dev_scripts/train_diffusion_fsdp_wFixedFID_2D_singMod.py /home/ziabariak/git/UCF-VIT/configs/xct/diffusion/base_config_dgx_2D.yaml


# srun --mpi=pmix --container-mounts /lustre/fs0 --container-mount-home --container-image /lustre/fs0/scratch/lyngaasir/sqsh-files/0698614322576143+ucf-vit+25.05-upd2.sqsh python $HOME/git/UCF-VIT/dev_scripts/train_diffusion_fsdp.py $HOME/git/UCF-VIT/configs/xct/diffusion/base_config_dgx.yaml

# 
# 
