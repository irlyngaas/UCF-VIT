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


eval "$(/lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/bin/conda shell.bash hook)"
conda activate UCF-rocm7.13

module load PrgEnv-gnu
module load gcc/12.2.0

module load rocm/7.13
export ROCM_HOME=${ROCM_PATH}

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
# Requires a real cupy install matching this environment's ROCm build, in
# the UCF-rocm7.13 conda env activated above -- not a project dependency
# (see gpu_adaptive_patching.py's own _label_regions docstring for why
# it's opt-in). CuPy no longer ships prebuilt ROCm wheels itself (dropped
# after v13.4.0) -- AMD hosts a ROCm-matched build instead, versioned per
# ROCm release, installed via (confirmed working against this exact
# rocm/7.13 build, README.md's own "Optional: cupy for
# region_backend=cupyx" section has the same steps):
#   module load rocm/7.13; export ROCM_HOME=${ROCM_PATH}
#   pip install amd-cupy --extra-index-url https://pypi.amd.com/rocm-7.13.0/simple
# ROCM_HOME must be set at runtime too (not just at install time), which is
# why it's exported above, before this script ever gets here. The test
# skips cleanly (not a failure) if cupy isn't actually installed here yet.
time python -m pytest ../../tests/model/test_gpu_adaptive_patching_cupyx_real.py -v
