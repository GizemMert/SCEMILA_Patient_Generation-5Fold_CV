#!/bin/sh


#SBATCH --job-name=5KCV
#SBATCH -o slurm_jupyter_%j.txt
#SBATCH -e slurm_error_%j.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=4:00:00




source $HOME/.bashrc

chmod 600 $HOME/slurm_jupyter_$SLURM_JOB_ID.job

# Activate the correct conda environment
conda deactivate
conda activate my_new_env

# Select an available port between 8000-9000 for Jupyter Lab
PORT=$(/usr/bin/shuf -i 8000-9000 -n 1)


export JUPYTER_TOKEN=$(openssl rand -hex 32)


echo "Starting Jupyter Lab..." >> slurm_jupyter_${SLURM_JOB_ID}.txt


cat <<END >> slurm_jupyter_${SLURM_JOB_ID}.txt

################################################################################################################

                          --->>>>>>> Jupyter Lab Server Launch Instructions <<<<<<<<-----

################################################################################################################

Selected port: ${PORT}
Lauched Compute Node: ${HOSTNAME}

To access the Jupyter Lab server, follow these steps:

1. Establish an SSH tunnel from your local machine to the server:

   ssh ${USER}@hpc-build01.scidom.de -D 1234 -q

2. Open a web browser and navigate to:

   Point your web browser to http://${HOSTNAME}:${PORT}

3. Log in to Jupyter Lab using the following Token

   Your Jupyter Lab token is: ${JUPYTER_TOKEN}

   You can also use the password that you defined with "jupyter lab password"command to login.

Remember to terminate your Jupyter Lab session when done to free up resources:

1. Close the Jupyter Lab browser tab.

2. Terminate the SSH tunnel (if used).

3. Cancel the SLURM job if it's still running:

   scancel ${SLURM_JOB_ID}

END


jupyter-lab --port=${PORT} --no-browser --ip=0.0.0.0 --NotebookApp.token=${JUPYTER_TOKEN} --notebook-dir=$HOME/

