#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J compute-normalization-stats
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o compute-normalization-stats-%j.out
#SBATCH -e compute-normalization-stats-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

# CONFIG/OUTPUT/NUM_WORKERS/EXTRA_ARGS are all overridable at submission time,
# e.g.:
#   CONFIG=../../configs/sst/unetr/adaptive_config.yaml OUTPUT=../../stats/sst_stats.yaml sbatch run_compute_normalization_stats.sh
[ -z $CONFIG ] && CONFIG=../../configs/basic_ct/sap/base_config.yaml
[ -z $OUTPUT ] && OUTPUT=../../stats/basic_ct_stats.yaml
# Matches --cpus-per-task above -- utils/compute_normalization_stats.py's
# own docstring: this is normally the dominant lever for wall-clock time
# here (a real per-sample file decode, one process per worker), more than
# which node it runs on at all. num_workers=0 (the script's own default,
# used when run directly on a login node instead of through this launch
# script) keeps everything single-process -- safe anywhere, just serial.
[ -z $NUM_WORKERS ] && NUM_WORKERS=7

eval "$(/lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/bin/conda shell.bash hook)"
conda activate UCF-rocm7.13

module load PrgEnv-gnu
module load gcc/12.2.0

export PYTHONPATH=$PWD:$PYTHONPATH

# A single plain process -- no torch.distributed, no srun here, unlike
# launch/basic_ct/*.sh's real training jobs (compute_normalization_stats.py
# builds its own single-process dist group internally, see its own
# init_single_process_dist() call, the same trick utils/validate_config.py
# uses). No GPU work happens at all (pure CPU: file I/O/decode + numpy
# accumulation) -- --gres=gpu:1 above is requested only because Frontier's
# batch partition ties CPU allocation to GPU allocation per node, not
# because this script needs one.
time python ../../utils/compute_normalization_stats.py $CONFIG --output $OUTPUT --num-workers $NUM_WORKERS $EXTRA_ARGS
