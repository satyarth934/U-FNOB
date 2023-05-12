#!/bin/bash
#SBATCH --account=m1012_g
#SBATCH --qos=regular
###SBATCH --qos=debug
#SBATCH --mail-type=ALL
#SBATCH --mail-user=satyarth@lbl.gov
#SBATCH --time-min=1:00:00
#SBATCH --time=2:00:00
#SBATCH --nodes=1
###SBATCH --error=slurm-output/%x-%j.err
#SBATCH --output=slurm-output/%x-%j.out
#SBATCH -C gpu
#SBATCH -c 128
#SBATCH --gres=gpu:1 --mem=80G
#SBATCH --gpus 1
#SBATCH --gpus-per-task=1
###SBATCH --mem=80G

###### # user setting and executables go here
echo "--- Loading pytorch module ---"
module load pytorch

echo "--- Setting some environment variables ---"
# export PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:32"    # Controls GPU memory fragmentation
export WANDB_SILENT=1        # Does not print wandb updates to the terminal
export PL_LOGGER_SANITY=0    # Turns off PL sanity check logs

echo "--- Checking GPU status ---"
nvidia-smi

echo "--- Running the code ---"
cd /global/cfs/cdirs/m1012/satyarth/Projects/digitaltwin-pssm-pl/src/
python -u evaluations.py
