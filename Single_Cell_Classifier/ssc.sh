#!/bin/bash
#SBATCH --job-name=scc
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=2:00:00
#SBATCH --nice=10000

#SBATCH --output=scc_output.log
#SBATCH --error=scc_error.log


source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env

python single_cell_classification.py