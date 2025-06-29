#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=logs/%x_%j.out             # Output file
#SBATCH --error=logs/%x_%j.err              # Error file
#SBATCH --time=08:00:00                     # Time limit (hrs:min:sec)

#SBATCH --nodes=1                           # Number of nodes

# CPU specifications 
#SBATCH --mem-per-cpu=4g                    # Memory request per CPU
#SBATCH --ntasks=1 --cpus-per-task=2

# GPU specifications 
# use: `scontrol show job <job id>` of GUI initialized jupyter session to discern which flags work for this setup file  
#SBATCH --partition=short                   # specify partition (interacive, short, medium, long)   note: short 
#SBATCH --gres=gpu:a100:1                   # gpu:<gpu type>:<number of gpus to use in node>
    #note: testing shows that the typical g0XX gpu node has 4 a100 gpus 
# options: v100, a100, h200     (stick with a100)


# get notifications 
#SBATCH --mail-type=BEGIN,END,FAIL      # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=<bc email>	    # Email for notifications

###########################
### End of SLURM params ###
###########################


# Load the miniconda module and activate the "CoxeterEnv" python environment with Jupyter
module purge
module use /m31/modulefiles/static
module load miniconda
module list
conda activate /projects/expmmllab/CoxeterEnv       #note: already created this env with all required packages
python -m ipykernel install --user --name CoxeterEnv --display-name "Python (CoxeterEnv)"           # get recognized by vs code as a valid kernel

# Start the notebook
jupyter notebook --no-browser --ip=$(hostname -i)
