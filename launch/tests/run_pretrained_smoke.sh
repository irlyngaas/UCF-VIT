#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J pretrained-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o pretrained-smoke-%j.out
#SBATCH -e pretrained-smoke-%j.out

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

# Tier 3: real end-to-end pretrained-checkpoint loading -- train once, then
# fine-tune from that checkpoint at a different resolution. Deliberately NOT
# run under the top-level srun every other launch/*/*.sh script uses: this
# driver spawns its own srun subprocess per training run (same reasoning as
# run_training_smoke.py/run_feature_matrix_smoke.py), so it must itself run
# as a single plain process to avoid nesting.
time python ../../tests/integration/run_pretrained_smoke.py
