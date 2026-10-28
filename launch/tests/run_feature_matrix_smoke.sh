#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J feature-matrix-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o feature-matrix-smoke-%j.out
#SBATCH -e feature-matrix-smoke-%j.out

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
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

# cupy (only actually imported if UCF_VIT_GPU_AP_REGION_BACKEND=cupyx is
# exported before this script's own `sbatch` call -- see UCF_VIT.model.
# gpu_adaptive_patching.GPUPatchify2D/GPUPatchify3D's own region_backend
# docstring) defaults its JIT kernel cache to $HOME/.cupy/kernel_cache,
# which blows past $HOME's disk quota on Frontier -- redirect it to the
# same node-local /tmp/$JOBID scratch space MIOPEN_USER_DB_PATH already
# uses above, same as launch/tests/run_cupyx_smoke.sh already does.
export CUPY_CACHE_DIR=/tmp/$JOBID/cupy_kernel_cache
mkdir -p $CUPY_CACHE_DIR


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

# Tier 3b: turns each "advanced feature" (adaptive patching, tiling, twoD,
# tensor parallelism) back on, one representative config at a time, against
# real Frontier data -- see tests/integration/run_feature_matrix_smoke.py's
# module docstring for the full FEATURE_MATRIX and reasoning behind each
# cell. Deliberately NOT run under the top-level srun that every other
# launch/*/*.sh script uses: this driver spawns its own srun subprocess per
# run (same reason as run_training_smoke.sh), so it must itself run as a
# single plain process to avoid nesting.
time python ../../tests/integration/run_feature_matrix_smoke.py
