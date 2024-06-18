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

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_env2
module load python/3.8
module load jupyter


HOSTNAME=$(hostname)



PORT=$(shuf -i 8000-9999 -n 1)

jupyter-notebook --no-browser --port=$PORT --ip=$HOSTNAME &

jupyter nbconvert --execute /home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Single_Cell_Classifier/single_cell_classification.ipynb --to notebook --inplace

echo "Jupyter Notebook is running on $HOSTNAME:$PORT"