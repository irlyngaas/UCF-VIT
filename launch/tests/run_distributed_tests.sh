#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J pytest-distributed
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -o pytest-distributed-%j.out
#SBATCH -e pytest-distributed-%j.out

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

# tests/distributed/ needs an actual multi-process launch (unlike the rest of
# tests/, which run fine as a single local process) -- see tests/README.md.
#
# "$@" forwards this script's own sbatch arguments straight to pytest, same
# as launch/tests/run_dataloader_speed.sh -- with no extra arguments this
# still runs the normal Tier 2 distributed suite exactly as before (the
# dataloader_speed marker is excluded by default, see addopts in
# pyproject.toml). To run the real multi-rank decode-throughput measurement
# in test_dataloader_speed_real_pipeline.py against one specific config
# instead, pass:
#   sbatch run_distributed_tests.sh -m dataloader_speed \
#     -k test_real_decode_throughput_config_distributed \
#     --speed-config ../../configs/basic_ct/sap/base_config.yaml \
#     --speed-buffer-sizes 16,32,64,100
time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python -m pytest ../../tests/distributed/ -v "$@"
