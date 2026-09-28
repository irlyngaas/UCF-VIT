#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J compute-normalization-stats
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
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
# To scale across more/fewer nodes or tasks per node, override the #SBATCH
# defaults themselves on the sbatch command line (real SLURM resource
# requests, not shell variables, so they can't be read from env vars set
# below) -- SLURM's own CLI flags always win over a script's #SBATCH
# directives, e.g.:
#   sbatch --nodes=4 --ntasks-per-node=8 run_compute_normalization_stats.sh
[ -z $CONFIG ] && CONFIG=../../configs/basic_ct/sap/base_config.yaml
[ -z $OUTPUT ] && OUTPUT=../../stats/basic_ct_stats.yaml
# Matches --cpus-per-task above, leaving one core per task for the main
# process itself -- utils/compute_normalization_stats.py's own docstring:
# this (and real multi-rank parallelism below) are normally the dominant
# levers for wall-clock time here, more than which node it runs on at all.
# num_workers=0 (the script's own default, used when run directly on a
# login node instead of through this launch script) keeps everything
# single-process -- safe anywhere, just serial.
[ -z $NUM_WORKERS ] && NUM_WORKERS=6

eval "$(/lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/bin/conda shell.bash hook)"
conda activate UCF-rocm7.13

module load PrgEnv-gnu
module load gcc/12.2.0

export PYTHONPATH=$PWD:$PYTHONPATH

# Real multi-rank parallelism: compute_normalization_stats.py's own
# _init_distributed() detects a real srun -n N>1 launch (via SLURM_NTASKS/
# SLURM_PROCID, exactly like training_scripts/train.py's own dist_init) and
# builds a real multi-process "gloo" (CPU) torch.distributed group -- each
# rank gets its own file shard via NativePytorchDataModule's existing data_
# par_size/gx sharding (the same mechanism real distributed training uses),
# accumulates its own local count/sum/sum-of-squares, and the partial sums
# are combined exactly (not approximately) via one dist.all_gather_object at
# the end. "gloo", not train.py's "nccl" -- this is pure CPU/file-IO work,
# no GPU tensor communication happens anywhere, so no GPU is actually
# needed for this parallelism (--gres=gpu:8 above is requested only because
# Frontier's batch partition ties CPU allocation to GPU allocation per
# node, not because this script needs one). $SLURM_NTASKS (not a manually
# recomputed nodes*tasks-per-node) is what actually reflects the real
# negotiated allocation, whether from this script's own #SBATCH defaults or
# a submission-time --nodes/--ntasks-per-node override.
time srun -n $SLURM_NTASKS \
    python ../../utils/compute_normalization_stats.py $CONFIG --output $OUTPUT --num-workers $NUM_WORKERS $EXTRA_ARGS
