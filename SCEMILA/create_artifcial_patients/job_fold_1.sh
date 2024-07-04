#!/bin/bash
#SBATCH --job-name=fol_1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal

#SBATCH --output=FOLD_1_output_3.log
#SBATCH --error=FOLD_1_error_3.log

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env

python 3_shuffle_images_distribution_fold1.py