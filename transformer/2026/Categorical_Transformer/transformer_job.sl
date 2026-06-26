#!/bin/bash
#SBATCH --job-name=TransformerRun
#SBATCH --output=logs/%x_%j.out             # Output file
#SBATCH --error=logs/%x_%j.err              # Error file
#SBATCH --time=08:00:00                     # Time limit (hrs:min:sec)

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

# Build the per-prefix descent dataset for the configured group, then train.
# Comment out the build line if data.csv is already prepared.
# python build_descent_dataset.py
python Transformer.py
