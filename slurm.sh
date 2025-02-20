#!/bin/bash
#SBATCH --job-name=gflownet    # Job name
#SBATCH --output=./log/gflownet_%j.out  # Output file (%j will be replaced with job ID)
#SBATCH --error=./log/gflownet_%j.err   # Error file
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=16                 # Number of CPUs per task (adjust as needed)
#SBATCH --time=2:00:00                  # Wall time limit (hh:mm:ss)
#SBATCH --mem=64000                         # Total memory requested
#SBATCH --gres=gpu:a100:1                 # Number of GPUs
#SBATCH --partition=general               # Partition

source .venv/bin/activate
/net/scratch2/listar2000/gfn-od/.venv/bin/python /net/scratch2/listar2000/gfn-od/src/gflownet/train.py