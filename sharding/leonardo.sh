#!/bin/bash -l
#
#SBATCH --job-name=sharding-reactant
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH -p boost_usr_prod
#SBATCH --output sharding-reactant-output-%j.txt
#SBATCH --error sharding-reactant-error-%j.txt

export JULIA_DEBUG="Reactant,Reactant_jll"
# export TF_CPP_MAX_VLOG_LEVEL=3
# export XLA_FLAGS="--xla_dump_to=${HOME}/repo/GB-25/sharding/xla_dump"

# Make ultra sure this env var isn't set before loading CUDA module
export LD_LIBRARY_PATH=""
unset LD_LIBRARY_PATH

module load cuda/12.3
srun --preserve-env ${HOME}/repo/GB-25/sharding/julia.sh --project --threads=auto -O0 ${HOME}/repo/GB-25/sharding/simple_sharding_problem.jl
