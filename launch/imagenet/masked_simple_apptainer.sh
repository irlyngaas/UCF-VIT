#!/bin/bash
#SBATCH -A stf006
#SBATCH -J masked_simple
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -o masked_simple-%j.out
#SBATCH -e masked_simple-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD/../src:$PYTHONPATH

module load rocm/6.2.4
module load cray-mpich-abi/8.1.31
module load olcf-container-tools
module load apptainer-enable-mpi
module load apptainer-enable-gpu

export ABS_PATH="$(realpath -s ../../apptainer/lib/8.1.31)"
export APPTAINER_BINDPATH=$ABS_PATH
export APPTAINER_WRAPPER_LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$ABS_PATH:/opt/cray/libfabric/1.22.0/lib64:$APPTAINER_WRAPPER_LD_LIBRARY_PATH

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
apptainer run ../../apptainer/frontier-ubuntu-gnu-rocm624-vit.sif -c \
"python ../../training_scripts/train_masked_simple.py ../../configs/imagenet/mae/base_config.yaml"
