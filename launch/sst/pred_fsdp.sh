#!/bin/bash
#SBATCH -A stf006
#SBATCH -J pred_fsdp
#SBATCH --nodes=5
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o pred_fsdp-%j.out
#SBATCH -e pred_fsdp-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

eval "$(/lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/bin/conda shell.bash hook)"
conda activate sst
#conda activate sst-rocm6.4.2

#module load PrgEnv-gnu
#module load gcc/12.2.0

#module load rocm/6.4.2
module load rocm/7.13.0
#module load rccl-net-plugin/1.0

export MIOPEN_DISABLE_CACHE=1
#export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../../dev_scripts/train_pred_fsdp.py ../../configs/sst/pred/base_config_2d_mimic_uniform.yaml
#python ../../dev_scripts/train_pred_fsdp.py ../../configs/sst/pred/base_config_2d_uniform.yaml
#python ../../dev_scripts/train_pred_fsdp.py ../../configs/sst/pred/base_config_2d.yaml
#python ../../dev_scripts/train_pred_fsdp.py ../../configs/sst/pred/base_config_pv.yaml
#python ../../dev_scripts/train_pred_fsdp.py ../../configs/sst/pred/base_config.yaml
