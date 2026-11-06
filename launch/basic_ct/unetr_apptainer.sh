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

# UNTESTED -- drafted from OLCF's documented container MPI-binding recipe
# (https://docs.olcf.ornl.gov/software/containers_on_frontier.html), not yet
# run on real hardware. Validate before relying on it; see README.md's
# Apptainer section.

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

if [[ ! -v CONFIG ]]; then
    echo "Fatal Error: CONFIG is not set!" >&2
    exit 1
fi

SIF=../../apptainer/frontier-ubuntu-gnu-rocm7130-vit.sif

module reset
module load PrgEnv-gnu
module load gcc/12.2.0
# Loading the same rocm module the container's own ROCm install targets
module load rocm/7.13.0
# Provides the host's libmpi.so that the container's own MPICH build (built
# only for build-time symbol compatibility, not real MPI performance) will
# look for at runtime -- see apptainer/frontier-ubuntu-gnu-rocm7130.def
module load cray-mpich-abi/8.1.31

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

export MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_CONV_WINOGRAD=0

export HSA_NO_SCRATCH_RECLAIM=1
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1

export MPICH_GPU_SUPPORT_ENABLED=1

# Mount the host paths the Cray MPICH libraries need to be visible inside the
# container (see submit624.sbatch in OLCF's olcf_containers_examples repo)
export APPTAINER_BINDPATH=/usr/share/libdrm,/var/spool/slurmd,/opt/cray,${PWD}

# Replaces the container's own LD_LIBRARY_PATH (set at build time) with one
# that puts the host's Cray MPICH/GTL libraries first, falls back to the
# container's own ROCm libs, and keeps the host's own LD_LIBRARY_PATH/
# CRAY_LD_LIBRARY_PATH
export APPTAINERENV_LD_LIBRARY_PATH="$CRAY_MPICH_ROOTDIR/gtl/lib:/opt/cray/lib64:/opt/rocm/lib:/opt/rocm/lib64:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/pe/lib64"

# Host libraries the bound-in Cray MPICH libraries themselves depend on
export APPTAINER_CONTAINLIBS="/usr/lib64/libjansson.so.4,/usr/lib64/libgfortran.so.5,/usr/lib64/libnl-3.so.200,/lib64/libtinfo.so.6,/usr/lib64/libjson-c.so.5,/usr/lib64/libcxi.so.1,/opt/xpmem/lib64/libxpmem.so.0"

# The container's own MPICH build never saw the Cray MPICH libraries at
# build time, so it isn't linked against the GPU transport layer -- preload
# it so GPU-aware MPI is actually used instead of falling back to staged
# host-memory copies
export APPTAINERENV_LD_PRELOAD=$CRAY_MPICH_ROOTDIR/gtl/lib/libmpi_gtl_hsa.so.0:

export OMP_NUM_THREADS=7

# UCF_VIT isn't baked into the image (see the commented-out install step in
# apptainer/frontier-ubuntu-gnu-rocm7130.def) -- expose the host's live repo
# checkout instead, so code changes don't require a container rebuild
export APPTAINERENV_PYTHONPATH="$(realpath ../../src):${APPTAINERENV_PYTHONPATH}"

time srun -n $((SLURM_JOB_NUM_NODES*8)) --gpus-per-task=1 --gpu-bind=closest \
    apptainer exec --rocm --workdir "$(pwd)" "$SIF" \
    python ../../training_scripts/train.py ${CONFIG}
