#!/bin/bash
#SBATCH -A gen006
#SBATCH -J diffusion_fsdp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -o diffusion_fsdp-%j.out
#SBATCH -e diffusion_fsdp-%j.out

set -euo pipefail

JOBID="${JOBID:-${SLURM_JOB_ID}}"
JOBSIZE="${JOBSIZE:-${SLURM_JOB_NUM_NODES}}"

module load PrgEnv-gnu
module load rocm
module load rccl-net-plugin

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAINING_SCRIPT="${REPO_ROOT}/dev_scripts/train_diffusion_fsdp_wFixedFID_2D_singMod.py"
# Pass a Frontier config as the first sbatch argument or through CONFIG_FILE.
# The DGX config remains the default so this launcher tracks the exact model
# settings used by the current DGX command.
CONFIG_FILE="${CONFIG_FILE:-${1:-${REPO_ROOT}/configs/xct/diffusion/base_config_dgx_2D.yaml}}"

if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "Training script not found: $TRAINING_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="${TMPDIR:-/tmp}/miopen-${JOBID}"
mkdir -p "$MIOPEN_USER_DB_PATH"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800

export OMP_NUM_THREADS=7
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# The training entry point expects the environment normally populated by
# torchrun. On Frontier, one Slurm task per GPU gives better CPU/GPU affinity,
# so each task supplies the equivalent distributed environment directly.
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export TRAINING_SCRIPT CONFIG_FILE

time srun \
    --nodes="$JOBSIZE" \
    --ntasks="$SLURM_NTASKS" \
    --ntasks-per-node=4 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    bash -c '
        export RANK="$SLURM_PROCID"
        export WORLD_SIZE="$SLURM_NTASKS"
        export LOCAL_RANK=0
        exec python "$TRAINING_SCRIPT" "$CONFIG_FILE"
    '
