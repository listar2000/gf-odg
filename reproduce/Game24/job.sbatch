#!/bin/bash
#
#SBATCH --mail-user=jiaweizhang@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=general        # Use the 'general' partition
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:30:00            # Set a time limit of 3 minutes
#SBATCH --job-name=gflownet-game24  # Name the job
#SBATCH --output=slurm-%j.out       # Output file (stdout)
#SBATCH --error=slurm-%j.err        # Error file (stderr)

# Ensure Miniconda is initialized correctly
eval "$(~/miniconda3/bin/conda shell.bash hook)"  # Adjust path if needed
conda activate FoR

# Debugging: Print current Conda environment
echo "Current Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

#print GPU
nvidia-smi

# Run the script within the same directory
bash ./scripts/train.sh
