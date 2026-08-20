#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J training-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o training-smoke-%j.out
#SBATCH -e training-smoke-%j.out

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

# Deliberately NOT run under the top-level srun that every other launch/*/*.sh
# script uses: this driver spawns its own srun subprocess per training run
# (see tests/integration/run_training_smoke.py's module docstring for why),
# so it must itself run as a single plain process to avoid nesting.
time python ../../tests/integration/run_training_smoke.py
