#!/bin/bash
#SBATCH --job-name=fol_0
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal

#SBATCH --output=FOLD_0_output_3.log
#SBATCH --error=FOLD_0_error_3.log

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env

python run_pipeline_mixed.py --result_folder=/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/result_fold_0_mixed/mixed_seed42_max50/result_folder --source_folder=/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/mixed_uncertain_fold_0_seed42/max_50_percent --target_folder=/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/result_fold_0_mixed/mixed_seed42_max50