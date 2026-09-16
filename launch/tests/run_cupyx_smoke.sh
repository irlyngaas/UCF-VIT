#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J cupyx-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -o cupyx-smoke-%j.out
#SBATCH -e cupyx-smoke-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536


eval "$(/lustre/orion/stf006/proj-shared/irl1/miniforge3/bin/conda shell.bash hook)"
conda activate forge-vit

module load PrgEnv-gnu
module load gcc/12.2.0

module load rocm/6.2.4

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export PYTHONPATH=$PWD:$PYTHONPATH

# tests/model/test_gpu_adaptive_patching_cupyx_real.py is a single plain
# process -- no torch.distributed, no srun here, unlike
# run_distributed_tests.sh. Only needs 1 GPU (GPUPatchify2D/GPUPatchify3D's
# region_backend="cupyx" path, exercised directly against region_backend=
# "scipy" on the same input as a real-numerics correctness cross-check --
# every other region_backend="cupyx" test fakes cupy/cupyx out entirely and
# can only check dispatch logic, not real numerics).
#
# Requires a real cupy install matching this environment's ROCm build (e.g.
# `pip install cupy-rocm-6-2` for rocm/6.2.4 above -- check
# https://docs.cupy.dev for the exact package matching whatever `module
# load rocm/X.Y.Z` is active) in the forge-vit conda env activated above --
# not a project dependency (see gpu_adaptive_patching.py's own
# _label_regions docstring for why it's opt-in). The test skips cleanly
# (not a failure) if cupy isn't actually installed here yet.
time python -m pytest ../../tests/model/test_gpu_adaptive_patching_cupyx_real.py -v
