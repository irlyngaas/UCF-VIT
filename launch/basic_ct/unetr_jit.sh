#!/bin/bash
#SBATCH -A stf006
#SBATCH -J unetr
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -o unetr-%j.out
#SBATCH -e unetr-%j.out
#SBATCH --network=disable_rdzv_get
#SBATCH -C nvme

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

if [[ ! -v CONFIG ]]; then
    echo "Fatal Error: CONFIG is not set!" >&2
    exit 1
fi

eval "$(/lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/bin/conda shell.bash hook)"
CONDA_ENV_DIR=/lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/envs/UCF-rocm7.13/
module reset
module load PrgEnv-gnu
module load rocm/7.13.0
module load craype-accel-amd-gfx90a
module unload libfabric

conda activate ${CONDA_ENV_DIR}

module load cray-mpich/8.1.31 libfabric rccl-net-plugin

SDK_CORE=${CONDA_ENV_DIR}/lib/python3.11/site-packages/_rocm_sdk_core
SDK_LIBS=${CONDA_ENV_DIR}/lib/python3.11/site-packages/_rocm_sdk_libraries_gfx90a

export LD_LIBRARY_PATH=${SDK_CORE}/lib:${SDK_LIBS}/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=/opt/rocm-7.13.0/lib/librccl.so.1

export HSA_NO_SCRATCH_RECLAIM=1
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1

export MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_CONV_WINOGRAD=0

export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

#MIOPEN_CACHE_ROOT (seed root) defaults to /lustre/orion/stf006/proj-shared/$USER/miopen_seed
#when unset -- see miopen_utils.py's default_seed_root(). Set it (or
#miopen_precompile.root in CONFIG) to override.
if ! CACHE_KEY=$(python ../kernel_jit/miopen_utils.py path "${CONFIG}" --print key); then
  echo "##SCALE_ERROR miopen_utils.py path --print key failed for CONFIG=${CONFIG}"
  exit 1
fi
if ! SEED_DIR=$(python ../kernel_jit/miopen_utils.py path "${CONFIG}" --print dir); then
  echo "##SCALE_ERROR miopen_utils.py path --print dir failed for CONFIG=${CONFIG}"
  exit 1
fi
SEED_TAR=${SEED_DIR}/seed.tar
if [ ! -f "${SEED_TAR}" ]; then
  echo "No seed at ${SEED_TAR}"
  echo "  build one first: sbatch ../kernel_jit/launch_jit_compile.sh"
  exit 1
fi

export SEED=${SEED_DIR}/tree
#export NODE_SEED=/tmp/miopen-seed-${USER}-${SLURM_JOB_ID}/tree
export NODE_SEED=/mnt/bb/${USER}/miopen-seed/tree

# Stage the seed tree once per node
srun -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 \
  bash -c "mkdir -p $NODE_SEED && cp -r $SEED/. $NODE_SEED/"

export MIOPEN_DISABLE_CACHE=0

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
../kernel_jit/miopen_wrap.sh python ../../training_scripts/train.py ${CONFIG}
