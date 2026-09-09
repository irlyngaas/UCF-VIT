#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J sst-mae
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -o sst-mae-%j.out
#SBATCH -e sst-mae-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536


eval "$(/lustre/orion/stf006/proj-shared/irl1/miniforge3/bin/conda shell.bash hook)"
conda activate forge-vit

module load PrgEnv-gnu
module load gcc/12.2.0

module load rocm/6.2.4

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

# Phase 1 of the sst pretrain -> finetune workflow -- MAE pretraining on
# configs/sst/mae/base_config.yaml's real data (unsupervised, r/u/v/w
# input only, no labels). Phase 2 is unetr.sh, finetuned from whichever
# epoch checkpoint this produces (see configs/sst/unetr/base_config.yaml's
# own comment). Not yet run against real Frontier data -- first attempt,
# same -t 00:15:00 padding as unetr.sh's own first attempt.
#
# "$@" forwards this script's own sbatch arguments straight to train.py.
time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../../training_scripts/train.py ../../configs/sst/mae/base_config.yaml "$@"
