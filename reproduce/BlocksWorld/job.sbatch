#!/bin/bash
#
#SBATCH --mail-user=@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=general        # Use the 'general' partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=80G        # Request 2000MB (2GB) of memory
#SBATCH --time=02:00:00            # Set a time limit of 3 minutes
#SBATCH --job-name=gflownet-Blocksworld  # Name the job
#SBATCH --output=slurm-%j.out       # Output file (stdout)
#SBATCH --error=slurm-%j.err        # Error file (stderr)

# Ensure Miniconda is initialized correctly
eval "$(~/miniconda3/bin/conda shell.bash hook)"  # Adjust path if needed
conda activate FoR

# Debugging: Print current Conda environment
echo "Current Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

export OPENAI_API_KEY=
huggingface-cli login --tokenW

bash ./run.sh 2
