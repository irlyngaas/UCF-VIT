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

# First real run of configs/sst/unetr/base_config.yaml -- new dataset, new
# UNETR regression path (model.loss_fn:"MSE"), never yet run against real
# data. -t 00:15:00 (vs. basic_ct/unetr.sh's 00:05:00) gives a bit more
# headroom for whatever real-data surprises show up on a first attempt --
# tighten it back down once this is known-working.
time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../../training_scripts/train.py ../../configs/sst/unetr/base_config.yaml
