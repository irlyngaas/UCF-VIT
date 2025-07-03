#!/bin/bash
#SBATCH --partition defq
#SBATCH --nodes 2
#SBATCH --exclusive
#SBATCH --job-name=diffusion_fsdp_dh
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dh_env

export NNODES=$SLURM_JOB_NUM_NODES
export NTOTGPUS=$(( $NNODES * 8 ))
export NGPUS_PER_TRAINING=8
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRAINING ))
export OMP_NUM_THREADS=4

export TRAINING_SCRIPT="$HOME/UCF-VIT/dev_scripts/train_diffusion_fsdp_dh.py"
export CONFIG_FILE="$HOME/UCF-VIT/configs/xct/diffusion/base_config_dh_dgx.yaml"
export WALLTIME="1:00:00"
export MACHINE="DGX"

export DEEPHYPER_LOG_DIR="deephyper-experiment"-$SLURM_JOB_ID
mkdir -p $DEEPHYPER_LOG_DIR
export DEEPHYPER_DB_HOST=$HOST

sleep 5
echo "Doing something"
python ../../dev_scripts/main_dh_centralized.py 
