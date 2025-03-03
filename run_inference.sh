#!/bin/bash
#SBATCH --job-name=infer_numbers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --output=/home/jiaweizhang/gf-odg/inference_results/logs/inference_%j.out
#SBATCH --error=/home/jiaweizhang/gf-odg/inference_results/logs/inference_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --time=120:00
#SBATCH --partition=general

# Base model path - update this to your model path
MODEL_PATH="/net/scratch/llama3/Meta-Llama-3-8B-Instruct"

# Fine-tuned model adapter path
ADAPTER_PATH="/home/jiaweizhang/gf-odg/models/finetuned/train_number_6_lr3e-5/kl_0.0001"

# Prompt for inference
PROMPT="Generate 6 random numbers from 1 to 6, independently of each other, separated by commas. The generated numbers:"

# Output directory for inference results
BASE_OUTPUT_DIR="/home/jiaweizhang/gf-odg/inference_results"

# Create output directory if not exists
mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_OUTPUT_DIR}/logs

# Number of times to run inference
N=50 * 32
BATCH_SIZE=32  # Added batch size

# CSV output files
OUTPUT_CSV="${BASE_OUTPUT_DIR}/inference_number.csv"

# Load environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"  # Adjust path if needed
conda activate FoR

echo "Running inference on fine-tuned model..."
python /home/jiaweizhang/gf-odg/src/gflownet/inference_number.py \
    --adapter_path ${ADAPTER_PATH} \
    --base_model_path ${MODEL_PATH} \
    --prompt "${PROMPT}" \
    --N ${N} \
    --batch_size ${BATCH_SIZE} \
    --output_csv ${OUTPUT_CSV}

echo "Inference completed!"
