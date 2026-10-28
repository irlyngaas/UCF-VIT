#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J inference_diffusion
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -o inference_diffusion-%j.out
#SBATCH -e inference_diffusion-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536


eval "$(/lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/bin/conda shell.bash hook)"
conda activate UCF-rocm7.13

module load PrgEnv-gnu
module load gcc/12.2.0

module load rocm/7.13

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

# Generic across every DiffusionVIT config, same as test.sh -- just pass the
# config path in, plus any inference_diffusion.py flags (--dict_key,
# --num_samples, --seed, --variance_mode, --output_dir) after it.
#
# Usage:
#   cd launch/eval
#   sbatch inference_diffusion.sh ../../configs/xct/diffusion/base_config.yaml
#   sbatch inference_diffusion.sh ../../configs/xct/diffusion/base_config.yaml --num_samples 2 --seed 1234

if [ -z "$1" ]; then
    echo "Usage: sbatch inference_diffusion.sh <path/to/base_config.yaml> [inference_diffusion.py flags]"
    exit 1
fi

CONFIG=$1
shift

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../../training_scripts/inference_diffusion.py "$CONFIG" "$@"
