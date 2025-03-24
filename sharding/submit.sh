#!/bin/bash -l
#
#SBATCH --account=g191
#SBATCH --job-name=sharding-reactant
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --output sharding-reactant-out.txt
#SBATCH --error sharding-reactant-err.txt
#SBATCH --constraint=gpu
#SBATCH --exclusive

##SBATCH --partition=debug

export JULIA_DEBUG="Reactant,Reactant_jll"
alias julia='/capstor/scratch/cscs/gwagner/daint/juliaup/bin/julia'

# Important else XLA might hang indefinitely
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

ulimit -s unlimited
export Ngpu=4
srun --preserve-env --gpu-bind=per_task:1 --cpu_bind=sockets \
    julia --project --threads=auto -O0 ${HOME}/GB-25/sharding/sharded_baroclinic_instability.jl

