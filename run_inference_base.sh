#!/bin/bash
#SBATCH --job-name=infer_numbers
#SBATCH --output=/home/jiaweizhang/gf-odg/inference_results/logs/inference_%j.out
#SBATCH --error=/home/jiaweizhang/gf-odg/inference_results/logs/inference_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --time=30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=general

# Base model path
MODEL_PATH="/net/scratch/jiaweizhang/gemma2"

# Fine-tuned model adapter path

# Output directory for inference results
BASE_OUTPUT_DIR="/home/jiaweizhang/gf-odg/inference_results_gemma2"
OUTPUT_CSV_ADAPTER="${BASE_OUTPUT_DIR}/inference_fdsadwdasdasd.csv"

# Prompt for inference
PROMPT="Generate 3 random numbers from 1 to 6, independently, separated by commas:"

# Ensure output directory exists
mkdir -p "${BASE_OUTPUT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}/logs"

# Number of times to run inference
N=$((32 * 500))  # ✅ Fixed arithmetic
BATCH_SIZE=32  # Default batch size

# CSV output files
OUTPUT_CSV_BASE="${BASE_OUTPUT_DIR}/inference_base_3_1to6.csv"

echo "Running inference on both base and fine-tuned models..."
python /home/jiaweizhang/gf-odg/src/gflownet/inference_number.py \
    --adapter_path "${ADAPTER_PATH}" \
    --base_model_path "${MODEL_PATH}" \
    --prompt "${PROMPT}" \
    --N "${N}" \
    --batch_size "${BATCH_SIZE}" \
    --output_csv_base "${OUTPUT_CSV_BASE}" \
    --output_csv_adapter "${OUTPUT_CSV_ADAPTER}"\
    --max_new_tokens 20\
    --model_type base

echo "Inference completed!"
