#!/usr/bin/env bash

# Read-only Frontier/Slurm monitoring walkthrough for UCF-VIT training jobs.
#
# This collects the same categories of evidence used during the live review:
#   1. Slurm state and allocation metadata
#   2. training progress and checkpoint/resubmission messages
#   3. ROCm utilization, VRAM, and package-power counters
#   4. an immediate second GPU sample to distinguish transient dips from stalls
#   5. recent continuation logs to detect repeated epoch ranges
#
# The script does not cancel, modify, or resubmit a job. The two `srun` commands
# create small read-only, overlapping monitoring steps inside your allocation.
#
# Usage:
#   ./dev_scripts/monitor_frontier_run.sh JOB_ID
#   ./dev_scripts/monitor_frontier_run.sh JOB_ID /path/to/job.log
#
# Example from the monitored chain:
#   ./dev_scripts/monitor_frontier_run.sh 5445884

set -euo pipefail

if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 JOB_ID [LOG_FILE]" >&2
    exit 2
fi

JOB_ID=$1
LOG_FILE=${2:-}

if [[ ! $JOB_ID =~ ^[0-9]+$ ]]; then
    echo "JOB_ID must contain only digits: $JOB_ID" >&2
    exit 2
fi

section() {
    printf '\n===== %s =====\n' "$1"
}

section "1. Current jobs"
# Why: identify the active continuation, its state, elapsed time, node count,
# and pending reason. A pending job has no GPUs to measure yet.
squeue -u "$(id -un)" -h -o '%i|%j|%T|%M|%D|%R'

section "2. Accounting record for job ${JOB_ID}"
# Why: confirm terminal state and exit code even after a job leaves squeue.
sacct -j "$JOB_ID" -X \
    --format=JobID,JobName%28,State,Elapsed,Start,End,NNodes,ExitCode \
    -n -P

section "3. Allocation and launch metadata"
# Why: verify node/task/GPU counts, time limit, launcher, config arguments,
# output location, and the configured GPU power cap.
JOB_RECORD=$(scontrol show job "$JOB_ID" -o)
printf '%s\n' "$JOB_RECORD" | tr ' ' '\n' | rg \
    '^(JobId|JobName|JobState|RunTime|TimeLimit|StartTime|EndTime|NodeList|NumNodes|NumTasks|Command|SubmitLine|StdOut|AdminComment|TresPerNode)='
JOB_STATE=$(printf '%s\n' "$JOB_RECORD" | tr ' ' '\n' | sed -n 's/^JobState=//p')

if [[ -z $LOG_FILE ]]; then
    # Slurm reports StdOut=/absolute/path in the one-line job record.
    LOG_FILE=$(printf '%s\n' "$JOB_RECORD" | tr ' ' '\n' | sed -n 's/^StdOut=//p')
fi

section "4. Training progress from the live log"
# Why: GPU counters alone cannot tell us whether epochs advance, checkpoints
# succeed, or an auto-resubmission repeatedly restores an old epoch.
if [[ -n $LOG_FILE && -r $LOG_FILE ]]; then
    echo "Log: $LOG_FILE"
    rg 'Starting epoch|batch_idx|epoch_loss|modality_losses|Restored checkpoint|Saved walltime|checkpoint complete|Submitting continuation|numerical rollback|non-finite|degradation' \
        "$LOG_FILE" | tail -80 || true
else
    echo "No readable log was found. Pass it explicitly as the second argument."
fi

gpu_snapshot() {
    local label=$1

    section "$label"
    # One lightweight task per node sees all eight MI250X GCDs. --overlap lets
    # this diagnostic step coexist with the training step. rocm-smi CSV fields
    # for these exact flags are:
    #   $2 package power, $3 GPU use, $5 allocated VRAM percentage.
    # Four of eight GCD rows say "N/A (Secondary die)" for power, so the awk
    # average includes only numeric package-power rows.
    #
    # Output columns:
    #   node, mean_gpu_use, minimum_gcd_use, mean_vram, mean_package_power
    srun --jobid="$JOB_ID" --overlap \
        --nodes="$(squeue -j "$JOB_ID" -h -o '%D')" \
        --ntasks-per-node=1 --cpus-per-task=1 --mem=0 --gpu-bind=none \
        bash -lc '
            /opt/rocm-default/bin/rocm-smi \
                --showuse --showmemuse --showpower --csv 2>/dev/null |
            awk -F, -v host="$(hostname)" '\''
                NR > 1 && $3 ~ /^[0-9]+$/ {
                    use_sum += $3
                    vram_sum += $5
                    if ($2 + 0 > 0) {
                        power_sum += $2
                        power_count++
                    }
                    if (count == 0 || $3 < minimum_use) minimum_use = $3
                    count++
                }
                END {
                    if (count) {
                        printf "%s,%.2f,%d,%.2f,%.2f\n",
                            host,
                            use_sum / count,
                            minimum_use,
                            vram_sum / count,
                            power_count ? power_sum / power_count : 0
                    }
                }
            '\''
        '
}

if [[ $JOB_STATE == RUNNING ]]; then
    section "5. Locate the ROCm telemetry program on one compute node"
    # Why: rocm-smi was not on PATH in the non-interactive monitoring step, so
    # we first verified its installed absolute path. Frontier provides
    # rocm-default.
    srun --jobid="$JOB_ID" --overlap --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=0 --gpu-bind=none \
        bash -lc '
            hostname
            command -v rocm-smi || true
            ls /opt/rocm-default/bin/rocm-smi
        '

    gpu_snapshot "6. Allocation-wide GPU snapshot"

    # Why: an isolated low reading can land during data loading, an optimizer
    # boundary, or a collective. A second snapshot shows whether the same node
    # is persistently slow. In the observed run, initially low nodes recovered
    # to 100%, so they were not hardware/rank stalls.
    gpu_snapshot "7. Immediate confirmation snapshot"
else
    section "5-7. GPU telemetry skipped"
    echo "Job ${JOB_ID} is ${JOB_STATE}, not RUNNING; it has no live GPUs to sample."
fi

section "8. Continuation history near this log"
# Why: compare the first and last epoch logged by recent one-hour jobs. This is
# how the repeated epoch-18-to-26 replay was detected despite healthy GPUs.
if [[ -n $LOG_FILE && -d $(dirname "$LOG_FILE") ]]; then
    LOG_DIR=$(dirname "$LOG_FILE")
    JOB_NAME=$(basename "$LOG_FILE")
    JOB_NAME=${JOB_NAME%-*}

    find "$LOG_DIR" -maxdepth 1 -type f -name "${JOB_NAME}-*.out" \
        -printf '%T@ %p\n' | sort -n | tail -20 |
    while read -r _ file; do
        first_epoch=$(rg 'Starting epoch' "$file" | head -1 | awk '{print $3}' || true)
        last_epoch=$(rg 'Starting epoch' "$file" | tail -1 | awk '{print $3}' || true)
        disposition=$(rg -o \
            'Submitting continuation|Submitting numerical rollback|Training completed|CANCELLED|FAILED' \
            "$file" | tail -1 || true)
        printf '%s first_epoch=%s last_epoch=%s status=%s\n' \
            "$(basename "$file")" \
            "${first_epoch:-none}" \
            "${last_epoch:-none}" \
            "${disposition:-running-or-unknown}"
    done

    section "9. Persisted best-state messages"
    # Why: establishes whether the continuation follows the latest epoch or
    # always rolls back to a fixed best epoch.
    rg 'Restored persistent best state|Training completed. Best loss' \
        "$LOG_DIR/${JOB_NAME}-"*.out | tail -30 || true
else
    echo "Cannot inspect continuation history without a readable log directory."
fi

section "10. Interpretation guide"
cat <<'GUIDE'
GPU use near 100% + high power:
  The accelerator is compute-busy. More model parallelism is unlikely to help.

Low VRAM allocation by itself:
  This means a larger batch may fit, not that the GPU is underutilized.
  Benchmark samples/second before changing production training.

One low node in one snapshot, recovered in the next:
  Usually a transient phase/collective boundary, not a persistent stall.

The same node low in repeated snapshots:
  Investigate its ranks, data loading, collectives, clocks, and hardware.

Healthy GPU counters but repeated first_epoch/last_epoch ranges:
  The GPUs are efficiently repeating discarded work. Fix continuation logic
  before tuning batch size or parallelism.
GUIDE
