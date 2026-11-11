#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J unetr-apptainer
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -o unetr-apptainer-%j.out
#SBATCH -e unetr-apptainer-%j.out
#SBATCH --network=disable_rdzv_get
#SBATCH -C nvme

# Consumes the seed published by launch/kernel_jit/launch_jit_compile.sh
# (the bare-metal compile run) -- both paths install the exact same
# pip-wheel ROCm core (_rocm_sdk_core), so the compiled kernel cache is
# interchangeable between them.
#
# Stages the seed to the NVMe burst buffer, same as
# launch/basic_ct/unetr_jit.sh -- see project_deferred_nvme_vs_tmp_staging
# memory note; this is the NVMe half of that comparison (set
# LOCAL_SCRATCH_ROOT=/tmp and drop -C nvme/the /mnt/bb bind for the other
# half).

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

if [[ ! -v CONFIG ]]; then
    echo "Fatal Error: CONFIG is not set!" >&2
    exit 1
fi

SIF=../../apptainer/frontier-ubuntu-gnu-rocm7130-vit.sif

module reset
module load cray-mpich-abi/9.1.0

export NCCL_PROTO=Simple

export MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_CONV_WINOGRAD=0

export HSA_NO_SCRATCH_RECLAIM=1
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1

export MPICH_GPU_SUPPORT_ENABLED=1

export APPTAINER_BINDPATH=/usr/share/libdrm,/var/spool/slurmd,/opt/cray,/mnt/bb,${PWD}
export APPTAINERENV_LD_LIBRARY_PATH="$CRAY_MPICH_ROOTDIR/gtl/lib:/opt/cray/lib64:/pytorch/_rocm_sdk_core/lib:/pytorch/_rocm_sdk_libraries_gfx90a/lib:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/pe/lib64"
export APPTAINER_CONTAINLIBS="/usr/lib64/libjansson.so.4,/usr/lib64/libgfortran.so.5,/usr/lib64/libnl-3.so.200,/lib64/libtinfo.so.6,/usr/lib64/libjson-c.so.5,/usr/lib64/libcxi.so.1,/opt/xpmem/lib64/libxpmem.so.0"
export APPTAINERENV_LD_PRELOAD=$CRAY_MPICH_ROOTDIR/gtl/lib/libmpi_gtl_hsa.so.0:

export OMP_NUM_THREADS=7

export APPTAINERENV_UCF_VIT_REPO_ROOT="$(realpath ../..)"

# miopen_utils.py only needs stdlib + PyYAML -- run it via the container's
# own python rather than needing a host-side conda/module environment.
if ! CACHE_KEY=$(apptainer exec --rocm --pwd "$(pwd)" "$SIF" python ../kernel_jit/miopen_utils.py path "${CONFIG}" --print key); then
  echo "##SCALE_ERROR miopen_utils.py path --print key failed for CONFIG=${CONFIG}"
  exit 1
fi
if ! SEED_DIR=$(apptainer exec --rocm --pwd "$(pwd)" "$SIF" python ../kernel_jit/miopen_utils.py path "${CONFIG}" --print dir); then
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
export NODE_SEED=/mnt/bb/${USER}/miopen-seed/tree

# Stage the seed tree once per node
srun -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 \
  bash -c "mkdir -p $NODE_SEED && cp -r $SEED/. $NODE_SEED/"

export MIOPEN_DISABLE_CACHE=0

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
    ../kernel_jit/miopen_wrap.sh \
    apptainer exec --rocm --pwd "$(pwd)" "$SIF" \
    python ../../training_scripts/train.py ${CONFIG}
