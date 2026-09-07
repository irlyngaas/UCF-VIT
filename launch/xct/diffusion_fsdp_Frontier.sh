#!/bin/bash

# This is the script that runs inside a Frontier allocation.
# Normally, start it through submit_diffusion_fsdp_Frontier_loop.sh.

# --- Slurm resources ----------------------------------------------------------
# Command-line options such as "sbatch --nodes=4" override these defaults.
#SBATCH -A gen006
#SBATCH -J diffusion_fsdp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -o diffusion_fsdp-%j.out
#SBATCH -e diffusion_fsdp-%j.out

# Stop on errors, unset variables, and failed pipeline commands.
set -euo pipefail

JOB_ID="$SLURM_JOB_ID"
NODE_COUNT="$SLURM_JOB_NUM_NODES"
TASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-8}"

# --- Load the Frontier software environment ----------------------------------
module load PrgEnv-gnu
module load rocm
module load rccl-net-plugin

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vit

# --- Find the repository, training code, and YAML config ---------------------
# Slurm runs a copy of this file from /var/spool. Therefore, we cannot locate
# UCF-VIT using this script's runtime location. SLURM_SUBMIT_DIR is the folder
# where sbatch was called; ~/UCF-VIT is the fallback.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -f "${REPO_ROOT}/dev_scripts/train_diffusion_fsdp_wFixedFID_2D_singMod.py" ]]; then
    REPO_ROOT="${HOME}/UCF-VIT"
fi
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"

# A caller may select a different compatible entry point. Keeping the legacy
# script as the default preserves existing submissions.
TRAINING_SCRIPT="${TRAINING_SCRIPT:-${REPO_ROOT}/dev_scripts/train_diffusion_fsdp_wFixedFID_2D_singMod.py}"
# A config passed as the first argument overrides this default config.
CONFIG_FILE="${CONFIG_FILE:-${1:-${REPO_ROOT}/configs/xct/diffusion/base_config_dgx_2D.yaml}}"

if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "Training script not found: $TRAINING_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# --- Create job-specific runtime files ---------------------------------------
# MIOpen uses a separate temporary directory for each job.
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="${TMPDIR:-/tmp}/miopen-${JOB_ID}"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Python creates this small file after saving its walltime checkpoint. The
# shell checks for it after srun exits to decide whether another job is needed.
export CONTINUATION_MARKER="${REPO_ROOT}/.diffusion_fsdp-continuation-${JOB_ID}"
# A separate marker requests rollback to a best checkpoint after a collective
# NaN/Inf detection. Its three lines are checkpoint name, reduced LR, and
# recovery-attempt number.
export RECOVERY_MARKER="${REPO_ROOT}/.diffusion_fsdp-recovery-${JOB_ID}"

# --- Configure distributed training -----------------------------------------
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export NCCL_TIMEOUT=1800

# Per-rank phase breadcrumbs make it possible to identify whether a missing
# rank stopped in data loading, GPU preparation, forward, backward, or the
# optimizer. GPU synchronization is intentionally enabled for diagnostic runs.
export TRACE_RANK_PHASES="${TRACE_RANK_PHASES:-0}"
export TRACE_RANK_SYNC="${TRACE_RANK_SYNC:-0}"

export OMP_NUM_THREADS=7
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/utils:${PYTHONPATH:-}"

# --- Reserve time to save a checkpoint ---------------------------------------
# The job lasts 3,600 seconds. Python stops training after 3,000 seconds,
# leaving ten minutes to write the checkpoint before Slurm ends the allocation.
# These values can be overridden as environment variables when submitting.
export CHECKPOINT_BUFFER_SECONDS="${CHECKPOINT_BUFFER_SECONDS:-600}"
export JOB_WALLTIME_SECONDS="${JOB_WALLTIME_SECONDS:-3600}"
if (( CHECKPOINT_BUFFER_SECONDS >= JOB_WALLTIME_SECONDS )); then
    echo "CHECKPOINT_BUFFER_SECONDS must be smaller than JOB_WALLTIME_SECONDS" >&2
    exit 1
fi
export TRAINING_DEADLINE_EPOCH=$(( $(date +%s) + JOB_WALLTIME_SECONDS - CHECKPOINT_BUFFER_SECONDS ))

# --- Tell all Python processes how to communicate ----------------------------
# The first node coordinates the processes. Each Slurm task receives one GPU,
# so that task sees its assigned GPU as local CUDA/ROCm device 0.
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export TRAINING_SCRIPT CONFIG_FILE

# --- Run one Python process per GPU ------------------------------------------
# Temporarily disable "exit on error" so we can inspect srun's return status.
set +e
time srun \
    --nodes="$NODE_COUNT" \
    --ntasks="$SLURM_NTASKS" \
    --ntasks-per-node="$TASKS_PER_NODE" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    bash -c '
        export RANK="$SLURM_PROCID"
        export WORLD_SIZE="$SLURM_NTASKS"
        export LOCAL_RANK=0
        exec python "$TRAINING_SCRIPT" "$CONFIG_FILE"
    '
TRAINING_STATUS=$?
set -e

# --- Decide whether to submit another one-hour job ---------------------------
# Numerical recovery is handled before ordinary walltime continuation. Python
# exits cleanly only after it has selected or atomically saved a safe best
# checkpoint and written all retry parameters to the recovery marker.
if (( TRAINING_STATUS == 0 )) && [[ -f "$RECOVERY_MARKER" ]]; then
    mapfile -t RECOVERY_VALUES < "$RECOVERY_MARKER"
    rm -f "$RECOVERY_MARKER"
    RECOVERY_CHECKPOINT="${RECOVERY_VALUES[0]:-}"
    RECOVERY_LR="${RECOVERY_VALUES[1]:-}"
    NEXT_RECOVERY_ATTEMPT="${RECOVERY_VALUES[2]:-}"
    if [[ -z "$RECOVERY_CHECKPOINT" || -z "$RECOVERY_LR" ||
          ! "$NEXT_RECOVERY_ATTEMPT" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid numerical recovery marker contents" >&2
        exit 1
    fi
    if [[ "${AUTO_RESUBMIT:-0}" == "1" ]]; then
        BATCH_SCRIPT="${REPO_ROOT}/launch/xct/diffusion_fsdp_Frontier.sh"
        echo "Submitting numerical rollback attempt ${NEXT_RECOVERY_ATTEMPT}"
        echo "Checkpoint: ${RECOVERY_CHECKPOINT}"
        echo "Rebased learning rate: ${RECOVERY_LR}"
        sbatch \
            --nodes="$NODE_COUNT" \
            --job-name="${JOB_NAME:-diffusion_fsdp}" \
            --output="${LOG_DIR}/%x-%j.out" \
            --error="${LOG_DIR}/%x-%j.out" \
            --export=ALL,AUTO_RESUBMIT=1,RESUME_FROM_CHECKPOINT=True,RESUME_CHECKPOINT_NAME="$RECOVERY_CHECKPOINT",RESUME_LR="$RECOVERY_LR",RECOVERY_ATTEMPT="$NEXT_RECOVERY_ATTEMPT" \
            "$BATCH_SCRIPT" "$CONFIG_FILE"
        exit 0
    fi

    echo "A numerical rollback was prepared; AUTO_RESUBMIT is disabled."
    exit 0
fi

# A continuation is submitted only when:
#   1. training exited cleanly, and
#   2. Python saved a checkpoint and created the marker file.
# A real training error therefore does not create an endless resubmission loop.
if (( TRAINING_STATUS == 0 )) && [[ -f "$CONTINUATION_MARKER" ]]; then
    rm -f "$CONTINUATION_MARKER"
    if [[ "${AUTO_RESUBMIT:-0}" == "1" ]]; then
        BATCH_SCRIPT="${REPO_ROOT}/launch/xct/diffusion_fsdp_Frontier.sh"
        echo "Submitting continuation with resume_from_checkpoint=True"
        sbatch \
            --nodes="$NODE_COUNT" \
            --job-name="${JOB_NAME:-diffusion_fsdp}" \
            --output="${LOG_DIR}/%x-%j.out" \
            --error="${LOG_DIR}/%x-%j.out" \
            --export=ALL,AUTO_RESUBMIT=1,RESUME_FROM_CHECKPOINT=True,RESUME_CHECKPOINT_NAME= \
            "$BATCH_SCRIPT" "$CONFIG_FILE"
        exit 0
    fi

    echo "A restart checkpoint was saved; AUTO_RESUBMIT is disabled."
    exit 0
fi

# No marker means training either finished normally or failed. Preserve srun's
# status so Slurm reports the correct result.
exit "$TRAINING_STATUS"
