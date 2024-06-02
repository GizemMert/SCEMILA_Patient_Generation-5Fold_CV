#!/bin/bash
#SBATCH --job-name=SC_5F_CV
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --nice=1
#SBATCH --qos=gpu_short
#SBATCH --output=5F_CV_output.log
#SBATCH --error=5F_CV_error.log

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_env2

python run_pipeline.py --result_folder=TEST_RUN