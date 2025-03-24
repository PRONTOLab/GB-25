#!/bin/bash -l

#SBATCH --job-name="scaling_test_8"
#SBATCH --output=slurm.%j.o
#SBATCH --error=slurm.%j.e
#SBATCH --time=00:40:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --exclusive
        
#SBATCH --account=g191
export Ngpu=8
export resolution_fraction=32
export JULIA_DEBUG="Reactant,Reactant_jll"
export MPICH_GPU_SUPPORT_ENABLED=1

ulimit -s unlimited
alias julia='/capstor/scratch/cscs/gwagner/daint/juliaup/bin/julia'
srun --preserve-env --gpu-bind=per_task:1 --cpu_bind=sockets bash unset_then_launch.sh
                
