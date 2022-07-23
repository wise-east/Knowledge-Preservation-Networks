#!/bin/bash
#SBATCH --partition=hipri 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
#SBATCH --account=all

CONDA_ENV="kpn" 
source $HOME/miniconda/etc/profile.d/conda.sh
source $HOME/.bashrc 
cd $HOME/cldst/Knowledge-Preservation-Networks
conda activate $CONDA_ENV

python train_test.py
