#!/bin/bash
#SBATCH --job-name=viral-flu-semantics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # INCREASED: Matching your memory request/previous attempts
#SBATCH --mem=512G                 # HIGH MEMORY REQUEST
#SBATCH --time=4:00:00
#SBATCH --partition=gpu           # CRITICAL FIX: Changed from 'cgpu' to 'gpu'
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --output=/home/aa3860/viral-mutation/slurm_logs/logs.flu.semantics%j
#SBATCH --error=/home/aa3860/viral-mutation/slurm_logs/err.flu.semantics%j
# Load necessary modules (CRITICAL: Fixes the GLIBCXX and CUDA issues)
module purge
#module load gcc/10.2.0-bz186       # CRITICAL FIX: Provides GLIBCXX_3.4.26 runtime library
module load gcc/12.4
module load cuda/11.8.0            # CRITICAL FIX: Provides a compatible CUDA version (adjust if needed)
# Use user-installed Miniconda
source /cache/home/aa3860/miniconda3/etc/profile.d/conda.sh
# Activate environment (must have TensorFlow-GPU installed)
conda activate viral-mutation
# Prevent user site interference (Good Practice)
export PYTHONNOUSERSITE=True
# Change to project directory
cd /cache/home/aa3860/viral-mutation
# Optional: Verification check (will show up in the output file)
python -c "import tensorflow as tf; print('TF built with CUDA:', tf.test.is_built_with_cuda()); print('Devices detected:', tf.config.list_physical_devices('GPU'))"
# Run the script
python bin/flu.py bilstm --checkpoint models/flu.hdf5 --semantics
