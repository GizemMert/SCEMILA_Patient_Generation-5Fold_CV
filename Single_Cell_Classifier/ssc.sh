#!/bin/bash
#SBATCH --job-name=scc
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal

#SBATCH --output=scc_output.log
#SBATCH --error=scc_error.log


source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env

python single_cell_classification.py