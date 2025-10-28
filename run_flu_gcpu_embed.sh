#!/bin/bash
#SBATCH --partition=cgpu                    # Partition name 
#SBATCH --job-name=viral-flu-embed         # Your alphafold3 job name
#SBATCH --gres=gpu:1                       # Number of gpus needed, keep at 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         # INCREASED: Matching your memory request/previous attempts
#SBATCH --mem=512G                 # HIGH MEMORY REQUEST
#SBATCH --time=12:00:00                    # This may need to change according to the requirements of the job
#SBATCH --output=/home/aa3860/viral-mutation/slurm_logs/logs.flu.embed%j
#SBATCH --error=/home/aa3860/viral-mutation/slurm_logs/err.flu.embed%j
# Load necessary modules (CRITICAL: Fixes the GLIBCXX and CUDA issues)
source /cache/home/aa3860/miniconda3/etc/profile.d/conda.sh

conda activate viral-mutation
module purge
module load gcc/11.2.0
module load cuda/11.8.0            # CRITICAL FIX: Provides a compatible CUDA version (adjust if needed)
# Prevent user site interference (Good Practice)
export PYTHONNOUSERSITE=True
# Change to project directory
cd /cache/home/aa3860/viral-mutation
# Optional: Verification check (will show up in the output file)
python -c "import tensorflow as tf; print('TF built with CUDA:', tf.test.is_built_with_cuda()); print('Devices detected:', tf.config.list_physical_devices('GPU'))"
# Run the script
python bin/flu.py bilstm --checkpoint models/flu.hdf5 --embed
