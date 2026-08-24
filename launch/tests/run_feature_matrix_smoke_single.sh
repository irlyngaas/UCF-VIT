#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J feature-matrix-smoke-single
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:45:00
#SBATCH -p batch
#SBATCH -o feature-matrix-smoke-single-%j.out
#SBATCH -e feature-matrix-smoke-single-%j.out

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

# Same driver as run_feature_matrix_smoke.sh, but for one FEATURE_MATRIX
# cell (by label) at a time instead of all of them -- useful for iterating
# on a single failure without paying for (or waiting on) the rest every
# time. Also NOT run under a top-level srun -- see
# tests/integration/run_feature_matrix_smoke.py's module docstring.
#
# 00:45:00 is generous enough even for basic_ct-unetr+twoD's or
# basic_ct-mae+twoD+do_tiling's 1800s per-run timeout plus setup overhead.
#
# Usage:
#   cd launch/tests
#   sbatch run_feature_matrix_smoke_single.sh basic_ct-unetr+twoD
#   sbatch run_feature_matrix_smoke_single.sh basic_ct-unetr+twoD --timeout 2400
#
# Known labels: basic_ct-unetr+do_ap, basic_ct-mae+do_ap,
# imagenet-classification+do_ap, catsdogs-classification+do_ap,
# basic_ct-unetr+do_tiling, imagenet-classification+do_tiling,
# basic_ct-unetr+twoD, imagenet-classification+tensor_par,
# basic_ct-mae+tensor_par, catsdogs-classification+tensor_par,
# basic_ct-unetr+do_ap+do_tiling, basic_ct-mae+twoD+do_tiling,
# imagenet-classification+do_ap+tensor_par, basic_ct-unetr+do_ap+twoD,
# imagenet-classification+do_tiling+tensor_par,
# basic_ct-unetr+twoD+tensor_par, basic_ct-sap+tensor_par,
# catsdogs-diffusion+tensor_par

if [ -z "$1" ]; then
    echo "Usage: sbatch run_feature_matrix_smoke_single.sh <label> [extra run_feature_matrix_smoke.py args...]"
    echo "Known labels: basic_ct-unetr+do_ap, basic_ct-mae+do_ap, imagenet-classification+do_ap, catsdogs-classification+do_ap, basic_ct-unetr+do_tiling, imagenet-classification+do_tiling, basic_ct-unetr+twoD, imagenet-classification+tensor_par, basic_ct-mae+tensor_par, catsdogs-classification+tensor_par, basic_ct-unetr+do_ap+do_tiling, basic_ct-mae+twoD+do_tiling, imagenet-classification+do_ap+tensor_par, basic_ct-unetr+do_ap+twoD, imagenet-classification+do_tiling+tensor_par, basic_ct-unetr+twoD+tensor_par, basic_ct-sap+tensor_par, catsdogs-diffusion+tensor_par"
    exit 1
fi

LABEL=$1
shift

time python ../../tests/integration/run_feature_matrix_smoke.py "$LABEL" "$@"
