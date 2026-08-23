#!/bin/bash
#SBATCH --job-name=HyperbolicRightDescent
#SBATCH --output=logs/%x_%j.out             # Output file
#SBATCH --error=logs/%x_%j.err              # Error file
#SBATCH --time=01:00:00                     # Time limit (hrs:min:sec)

#SBATCH --nodes=1                           # Number of nodes

# CPU specifications
#SBATCH --mem-per-cpu=4g                    # Memory request per CPU
#SBATCH --ntasks=1 --cpus-per-task=2

# GPU specifications
#SBATCH --partition=short                   # Specify partition
#SBATCH --gres=gpu:a100:1                   # GPU request

# Get notifications
#SBATCH --mail-type=BEGIN,END,FAIL

###########################
### End of SLURM params ###
###########################

# Load required modules
module purge
module use /m31/modulefiles/static
module load miniconda
module list

# Source Conda profile and activate the PyTorch environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate m31_pytorch

# 1. Resplit the raw inverse shortlex dataset into train.csv and test.csv
python resplit_train_test.py

# 2. Train the Transformer model
python -u Transformer.py
