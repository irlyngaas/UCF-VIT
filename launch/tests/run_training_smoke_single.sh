#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J training-smoke-single
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 01:15:00
#SBATCH -p batch
#SBATCH -o training-smoke-single-%j.out
#SBATCH -e training-smoke-single-%j.out

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

# Same driver as run_training_smoke.sh, but for one config at a time instead
# of all 10 -- useful for iterating on a single failure without paying for
# (or waiting on) the other 9 every time. Also NOT run under a top-level
# srun -- see tests/integration/run_training_smoke.py's module docstring.
#
# Doesn't pass --timeout/--min-files itself, so it shares run_training_smoke.py's
# defaults (currently 300s / 256 files) with run_training_smoke.sh -- pass
# --timeout/--min-files explicitly below to override either for just this
# run.
#
# Usage:
#   cd launch/tests
#   sbatch run_training_smoke_single.sh ../../configs/basic_ct/unetr/base_config.yaml
#   sbatch run_training_smoke_single.sh ../../configs/basic_ct/unetr/base_config.yaml --timeout 600 --min-files 128

if [ -z "$1" ]; then
    echo "Usage: sbatch run_training_smoke_single.sh <path/to/base_config.yaml> [extra run_training_smoke.py args...]"
    exit 1
fi

CONFIG=$1
shift

time python ../../tests/integration/run_training_smoke.py "$CONFIG" "$@"
