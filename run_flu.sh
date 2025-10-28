#!/bin/bash

#SBATCH --job-name=viral-mutation-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # INCREASED: Matching your memory request/previous attempts
#SBATCH --mem=256G                 # HIGH MEMORY REQUEST
#SBATCH --time=4:00:00
#SBATCH --partition=gpu            # CRITICAL FIX: Changed from 'cgpu' to 'gpu'
#SBATCH --gres=gpu:1              
#SBATCH --output=/home/aa3860/viral-mutation/slurm_logs/logs.%j
#SBATCH --error=/home/aa3860/viral-mutation/slurm_logs/err.%j
# Load necessary modules (CRITICAL: Fixes the GLIBCXX and CUDA issues)
module purge
module load gcc/10.2.0-bz186       # CRITICAL FIX: Provides GLIBCXX_3.4.26 runtime library
module load cuda/11.8.0            # CRITICAL FIX: Provides a compatible CUDA version (adjust if needed)


source /cache/home/aa3860/miniconda3/etc/profile.d/conda.sh

conda activate viral-mutation


export PYTHONNOUSERSITE=True


cd /cache/home/aa3860/viral-mutation

# Run the script
python bin/flu.py bilstm --checkpoint models/flu.hdf5 --embed
