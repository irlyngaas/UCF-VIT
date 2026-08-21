#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J pytest-dataloader-speed
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -o pytest-dataloader-speed-%j.out
#SBATCH -e pytest-dataloader-speed-%j.out

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


export PYTHONPATH=$PWD:$PYTHONPATH

# tests/dataloaders/test_dataset_speed.py is CPU-only (no GPU/torch.distributed
# needed) and just a single plain process -- no srun here, unlike
# run_distributed_tests.sh. Requesting a full node (matching every other
# launch/*/*.sh script) mainly so num_workers up to 7 has real cores behind
# it, not login-node contention with everyone else's shell sessions; the
# GPUs go unused.
#
# -m dataloader_speed is required -- these tests are excluded from the
# default `pytest` run (see addopts in pyproject.toml), since they're
# informational timing measurements with no pass/fail threshold, not
# correctness tests. -s shows the printed timing output (pytest captures
# stdout by default otherwise).
time python -m pytest -m dataloader_speed -s ../../tests/dataloaders/test_dataset_speed.py -v
