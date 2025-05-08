#!/bin/bash
#SBATCH -A stf006
#SBATCH -J unetr_simple
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -o unetr_simple-%j.out
#SBATCH -e unetr_simple-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536


source /lustre/orion/proj-shared/stf006/irl1/conda/bin/activate
conda activate /lustre/orion/stf006/proj-shared/irl1/vit

module load PrgEnv-gnu
module load gcc/12.2.0

module load rocm/6.2.4

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../../training_scripts/train_unetr_simple.py ../../configs/basic_ct/unetr/base_config.yaml
