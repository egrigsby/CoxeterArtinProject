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
#SBATCH --gres=gpu:a100:2                   # gpu:<gpu type>:<number of gpus to use in node>

# get notifications
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=linrya@bc.edu

###########################
### End of SLURM params ###
###########################

# Load the miniconda module and activate the CoxeterEnv python environment
module purge
module use /m31/modulefiles/static
module load miniconda
module list
conda activate /projects/expmmllab/CoxeterEnv

# Run Transformer.py from its own directory so relative paths resolve correctly
python Transformer.py
