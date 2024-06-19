#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --nice=1
#SBATCH --qos=gpu_short
#SBATCH --output=jupyter_output.log
#SBATCH --error=jupyter_error.log

# Source conda to make 'conda activate' work
source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh

# Activate the new conda environment
conda activate my_env2

# Get the hostname
HOSTNAME=$(hostname)

# Get an available port (or specify a port if needed)
PORT=$(shuf -i 8000-9999 -n 1)

# Start Jupyter Notebook server (run in the background)
jupyter-notebook --no-browser --port=$PORT --ip=$HOSTNAME &

# Run the notebook using nbconvert to execute it
jupyter nbconvert --execute /home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Single_Cell_Classifier/single_cell_classification.ipynb --to notebook --inplace

# Print instructions to access the notebook
echo "Jupyter Notebook is running on $HOSTNAME:$PORT"

