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

DATE=$(date +"%Y%m%d_%H%M%S")
OUTPUT_LOG="scc_output_$DATE.log"
ERROR_LOG="scc_error_$DATE.log"

#SBATCH --output=$OUTPUT_LOG
#SBATCH --error=$ERROR_LOG


source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env

python single_cell_classification.py