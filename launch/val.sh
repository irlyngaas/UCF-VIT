#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J val
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -o val-%j.out
#SBATCH -e val-%j.out

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

# Generic across every dataset/model, unlike the per-[DATASET]/[MODEL]
# training launch scripts (launch/basic_ct/*.sh etc.) -- val.py takes the
# same config as the training run being evaluated, so one script covers all
# of them; just pass the config path in.
#
# Usage:
#   cd launch
#   sbatch val.sh ../configs/basic_ct/unetr/base_config.yaml

if [ -z "$1" ]; then
    echo "Usage: sbatch val.sh <path/to/base_config.yaml>"
    exit 1
fi

CONFIG=$1

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../training_scripts/val.py "$CONFIG"
