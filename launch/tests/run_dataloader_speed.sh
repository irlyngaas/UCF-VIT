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

# tests/dataloaders/*speed*.py are single plain processes -- no
# torch.distributed, no srun here, unlike run_distributed_tests.sh.
# Requesting a full node (matching every other launch/*/*.sh script) mainly
# so num_workers up to 7 has real cores behind it, not login-node contention
# with everyone else's shell sessions. test_dataset_speed.py and
# test_dataset_speed_real_data.py are CPU-only (the GPUs go unused for
# those); test_pin_memory_speed.py is the one file here that actually uses
# a GPU (host->device transfer is the whole thing being measured), which is
# why --gres=gpu:8 was already being requested even before that file
# existed.
#
# -m dataloader_speed is required -- these tests are excluded from the
# default `pytest` run (see addopts in pyproject.toml), since they're
# informational timing measurements with no pass/fail threshold, not
# correctness tests. -s shows the printed timing output (pytest captures
# stdout by default otherwise).
#
# Points at the whole tests/dataloaders/ directory (marker-filtered, so only
# dataloader_speed-marked tests actually run) rather than a single file, so
# every speed-test file (synthetic delay, real NIfTI/JPEG decode, pin_memory
# transfer timing -- needs real Frontier data/a real GPU, which is why this
# has to run here and not just anywhere) runs together; any future
# speed-test file gets picked up the same way.
#
# "$@" forwards this script's own sbatch arguments straight to pytest, so you
# can investigate one specific real config/problem (e.g. "is this config's
# buffer_size too large for its real per-rank shard size?") without editing
# this file or paying for the fixed sweep over every shipped config above --
# pass test_dataset_speed_real_data.py's -k/--speed-config/--speed-buffer-sizes/
# --speed-num-workers as extra sbatch arguments, e.g.:
#   sbatch run_dataloader_speed.sh -k test_real_decode_throughput_config --speed-config ../../configs/basic_ct/sap/base_config.yaml --speed-buffer-sizes 16,32,64,100
# See that test's own docstring for the full option reference. With no extra
# arguments, this still runs the full fixed sweep exactly as before.
time python -m pytest -m dataloader_speed -s ../../tests/dataloaders/ -v "$@"
