#!/bin/bash

set -e  # Exit on any error

# Base model path - update this to your model path
MODEL_PATH="/net/scratch/llama3/Meta-Llama-3-8B-Instruct"

# Base prompt
PROMPT="Generate 5 random numbers from 1-5 independently of each other, ensuring no influence between selections. Separate the numbers with commas. Answer:"

# Base output directory
BASE_OUTPUT_DIR="/home/jiaweizhang/gf-odg/models/finetuned/train_number"

mkdir -p ${BASE_OUTPUT_DIR}/logs

# Define output directory and run name for this single run
output_dir="${BASE_OUTPUT_DIR}/number"
run_name="number"

# Export WANDB API key
export WANDB_API_KEY="94df40f69fe1711f227d8df8c9cf9ea389060b66"

# Create output directory if not exists
mkdir -p ${output_dir}

# Run the training script with the necessary hyperparameters
python /home/jiaweizhang/gf-odg/src/gflownet/train_number.py \
    --model_name_or_path ${MODEL_PATH} \
    --prompt "${PROMPT}" \
    --concept_name "animal" \
    --n_clusters 5 \
    --num_samples 320 \
    --buffer_size 500 \
    --update_clusters_every 100 \
    --min_samples_for_clustering 20 \
    --batch_size 32 \
    --max_new_tokens 30 \
    --num_epochs 10 \
    --num_steps_per_epoch 10 \
    --learning_rate 1e-4 \
    --final_learning_rate 3e-5 \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --output_dir ${output_dir} \
    --use_wandb \
    --wandb_project "gfn-diversity" \
    --wandb_name ${run_name}

echo "Training completed successfully!"
