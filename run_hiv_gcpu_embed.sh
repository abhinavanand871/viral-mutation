#!/bin/bash
#SBATCH --job-name=viral-hiv-embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --partition=cgpu
#SBATCH --gres=gpu:2
#SBATCH --output=/home/aa3860/viral-mutation/slurm_logs/logs.hiv.embed%j
#SBATCH --error=/home/aa3860/viral-mutation/slurm_logs/err.hiv.embed%j

# Load necessary modules
module purge
module load cuda/11.8.0  # Only load CUDA, no GCC module needed

# Use user-installed Miniconda
source /cache/home/aa3860/miniconda3/etc/profile.d/conda.sh
conda activate viral-mutation

# Ensure Conda's libstdc++.so.6 is used
export LD_LIBRARY_PATH=/cache/home/aa3860/miniconda3/envs/viral-mutation/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=True
export TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN warning

# Change to project directory
cd /cache/home/aa3860/viral-mutation

# Verification checks
gcc --version  # Should show Conda's GCC 11.2.0
python -c "import tensorflow as tf; print('TF built with CUDA:', tf.test.is_built_with_cuda()); print('Devices detected:', tf.config.list_physical_devices('GPU'))"
python -c "import ml_dtypes; print('ml_dtypes imported successfully')"

# Run the script
python bin/hiv.py bilstm --checkpoint models/hiv.hdf5 --embed
