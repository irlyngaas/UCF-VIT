#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J sst-unetr
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -o sst-unetr-%j.out
#SBATCH -e sst-unetr-%j.out

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

# Confirmed working against real Frontier data (job 5455196): clean 8-rank
# run, epoch_loss trending sharply down, no errors/NaN.
#
# "$@" forwards this script's own sbatch arguments straight to
# train.py -- to finetune from a configs/sst/mae/base_config.yaml
# pretraining run instead of training from scratch, first set
# trainer.use_pretrained_model:True/pretrained_checkpoint_filename in
# configs/sst/unetr/base_config.yaml (see its own comment), then:
#   sbatch unetr.sh --pretrained_config ../../configs/sst/mae/base_config.yaml
time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../../training_scripts/train.py ../../configs/sst/unetr/base_config.yaml "$@"
