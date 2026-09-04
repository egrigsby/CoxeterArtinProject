#!/bin/bash
#SBATCH --job-name=AffineA3InverseShortlexRight
#SBATCH --output=logs/%x_%j.out             # Output file
#SBATCH --error=logs/%x_%j.err              # Error file
#SBATCH --time=01:00:00                     # Time limit (hrs:min:sec)

#SBATCH --nodes=1                           # Number of nodes

# CPU specifications
#SBATCH --mem-per-cpu=4g                    # Memory request per CPU
#SBATCH --ntasks=1 --cpus-per-task=2

# GPU specifications
#SBATCH --partition=short                   # specify partition (interactive, short, medium, long)
#SBATCH --gres=gpu:a100:1                   # gpu:<gpu type>:<number of gpus> (model uses NUM_DEVICES=1)

# get notifications
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=<id>@bc.edu

###########################
### End of SLURM params ###
###########################

# Load the miniconda module and activate the CoxeterEnv python environment
module purge
module use /m31/modulefiles/static
module load miniconda
module list
conda activate /projects/expmmllab/CoxeterEnv

# The model lives in 2026/shared/ and is shared by every run. Putting THIS
# directory on PYTHONPATH is what makes `from config import *` inside the model
# resolve to the config.py sitting here, so the checkpoint and curves are
# written into this folder. shared/ deliberately contains no config.py.
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"

# Split this folder's raw labelled CSV into train.csv / test.csv, then train.
# Comment out the resplit line if train.csv and test.csv are already prepared.
python resplit_train_test.py
python ../../../shared/Transformer_presplit.py
